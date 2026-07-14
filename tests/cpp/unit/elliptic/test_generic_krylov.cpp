// Generic MATRIX-FREE Krylov layer (generic_krylov.hpp): three solver loops -- Richardson, CG,
// BiCGStab -- that take the operator as a CALLBACK (ApplyFn), so any matrix-free apply can be
// plugged in. This is the reusable core that a later slice wires into the compiled time-program
// (ProgramContext / codegen); this test is PURE C++ and validates the loops in isolation.
//
// OPERATOR: an SPD Helmholtz operator A = I - alpha*Lap (alpha = 0.1), supplied as an ApplyFn that
//   fills the ghosts of `in` (periodic), applies the SHARED discrete 5-point Laplacian
//   (apply_laplacian, all optional coefficients null -> bit-identical bare Laplacian), then forms
//   out = in - alpha*Lap(in). The bare periodic Laplacian has a constant null space that breaks CG;
//   I - alpha*Lap is symmetric POSITIVE-DEFINITE (its spectrum is 1 + alpha*lambda, lambda >= 0 the
//   non-negative eigenvalues of -Lap), so CG is well-defined and the loop is well-conditioned.
//
// MANUFACTURED SOLUTION: phi_exact(x,y) = sin(2 pi x) sin(2 pi y) (periodic on the unit square). We
//   do NOT use the continuous eigenvalue: to test the SOLVER and not the discretization, we form
//   rhs = A(phi_exact) by APPLYING the same discrete operator to the sampled phi_exact. Then we
//   solve A x = rhs from x = 0 and require max|x - phi_exact| < 1e-8 (tight: same discrete A).
//
// We validate the four loops:
//   - cg_solve        (SPD operator),
//   - bicgstab_solve  (identity preconditioner -- empty ApplyFn),
//   - richardson_solve(omega = 1/(1 + alpha*8*pi^2) ~ 1/spectral-max, more iters allowed),
//   - gmres_solve     (restarted GMRES(m), identity preconditioner): on the SPD operator it matches
//                      CG, and on a NON-symmetric operator (Helmholtz + a one-sided advection term,
//                      where CG STAGNATES) it converges to phi_exact. The non-symmetric case is the
//                      gmres-specific guard -- cg_solve on the same operator must NOT recover phi_exact.
// Each must converge (converged == true, iters > 1, small residual) and recover phi_exact. We also
// assert that max_iters = 0 throws std::invalid_argument (spec error 13).
//
// SERIAL test: no MPI (single box, DistributionMapping(1, 1)); the dot products in the loops are
// nonetheless COLLECTIVE (pops::dot -> all_reduce_sum), the identity in serial.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic ghost exchange)

#include "test_harness.hpp"  // pops::test::kPi

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace pops;
using pops::test::kPi;

namespace {

constexpr int kN = 32;        // 32 x 32 periodic grid
constexpr Real kAlpha = 0.1;  // Helmholtz coefficient: A = I - alpha*Lap (SPD, well-conditioned)

// Named functor (device-clean): out(i,j) = in(i,j) - alpha*lap(i,j). Same recipe as the elliptic
// kernels (#93): a plain lambda is fine on the host Serial path here, but a named functor keeps the
// kernel emission robust on every backend.
struct HelmholtzCombineKernel {
  Array4 outv;
  ConstArray4 inv, lapv;
  Real alpha;
  POPS_HD void operator()(int i, int j) const { outv(i, j) = inv(i, j) - alpha * lapv(i, j); }
};

// Non-symmetric combine: out = in - alpha*Lap(in) + beta * (in(i) - in(i-1)) / h, a FIRST-order
// upwind x-derivative added to the SPD Helmholtz operator. The one-sided difference is NOT
// self-adjoint (its transpose is the opposite-sided difference), so the whole operator is
// non-symmetric -- CG stagnates on it while GMRES (and BiCGStab) converge. `in`'s ghosts are
// periodic (filled before the matvec), so in(i-1) wraps at the low edge.
struct AdvectionHelmholtzKernel {
  Array4 outv;
  ConstArray4 inv, lapv;
  Real alpha, beta, inv_h;
  POPS_HD void operator()(int i, int j) const {
    outv(i, j) = inv(i, j) - alpha * lapv(i, j) + beta * (inv(i, j) - inv(i - 1, j)) * inv_h;
  }
};

// phi_exact(x,y) = sum of several periodic sine modes. A SINGLE mode is an eigenvector of the
// discrete Laplacian, so CG/BiCGStab would converge in ONE step (masking the iteration loop); a SUM
// of modes with DISTINCT eigenvalues forces several Krylov steps and a genuine Richardson sweep,
// which is what we want to exercise. All modes are periodic on the unit square. POPS_HD so it is
// device-callable from the init kernel (else nvcc returns garbage on device, like phi_exact in
// test_krylov_solver).
POPS_HD Real phi_exact(Real x, Real y) {
  return std::sin(2 * kPi * x) * std::sin(2 * kPi * y) +
         Real(0.5) * std::sin(4 * kPi * x) * std::sin(2 * kPi * y) +
         Real(0.3) * std::cos(2 * kPi * x) * std::cos(6 * kPi * y) +
         Real(0.2) * std::sin(6 * kPi * x) * std::cos(4 * kPi * y);
}

struct SampleExactKernel {
  Array4 af;
  Geometry geom;
  POPS_HD void operator()(int i, int j) const {
    af(i, j) = phi_exact(geom.x_cell(i), geom.y_cell(j));
  }
};

// max|a - b| over the valid cells, reduced over all ranks (serial: identity). Host loop (a tiny
// grid; this is a correctness check, not a hot path).
Real max_abs_diff(const MultiFab& a, const MultiFab& b) {
  Real d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 pa = a.fab(li).const_array();
    const ConstArray4 pb = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        d = std::fmax(d, std::fabs(pa(i, j) - pb(i, j)));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(d)));
}

constexpr Real kRelTol = 1e-12;
constexpr Real kRecoverTol = 1e-8;

// Fixture : construit UNE fois la grille, les deux operateurs matrice-libre (SPD et non-symetrique)
// et leurs RHS manufactures (couteux : plusieurs MultiFab + solves de reference). Chaque TEST reste
// independant (aucun ne modifie geom_/A_/rhs_), seuls les vecteurs solution x sont locaux au TEST.
class GenericKrylov : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    dom_ = new Box2D(Box2D::from_extents(kN, kN));
    geom_ = new Geometry{*dom_, 0.0, 1.0, 0.0, 1.0};
    ba_ = new BoxArray(std::vector<Box2D>{*dom_});
    dm_ = new DistributionMapping(1, 1);
    bc_ = new BCRec{};  // all faces default to Periodic -> fill_ghosts wraps in x and y

    // SPD Helmholtz operator A(in) = in - alpha*Lap(in), matrix-free: fill periodic ghosts of `in`,
    // apply the shared discrete 5-point Laplacian into a scratch, then combine. `in` needs >= 1
    // ghost for the stencil; `tmp` here is captured once and reused across every matvec (no
    // per-call alloc).
    lap_tmp_ = new MultiFab(*ba_, *dm_, 1, 0);
    const Geometry& geom = *geom_;
    const BCRec& bc = *bc_;
    MultiFab& lap_tmp = *lap_tmp_;
    A_ = new ApplyFn([&geom, &bc, &lap_tmp](MultiFab& out, const MultiFab& in) {
      // in is const at the API level, but fill_ghosts / apply_laplacian take a mutable MultiFab&
      // (they only WRITE the ghosts of `in`, never the valid cells). Casting away const is the
      // same contract the solver loops rely on; the valid data of `in` is unchanged.
      MultiFab& in_mut = const_cast<MultiFab&>(in);
      fill_ghosts(in_mut, geom.domain, bc);
      apply_laplacian(in_mut, geom, lap_tmp);  // lap_tmp = Lap(in) (all coeffs null -> bare Laplacian)
      for (int li = 0; li < out.local_size(); ++li) {
        Array4 ov = out.fab(li).array();
        const ConstArray4 iv = in.fab(li).const_array();
        const ConstArray4 lv = lap_tmp.fab(li).const_array();
        for_each_cell(out.box(li), HelmholtzCombineKernel{ov, iv, lv, kAlpha});
      }
    });

    // Manufactured solution phi_exact and the discretization-exact rhs = A(phi_exact).
    phi_exact_mf_ = new MultiFab(*ba_, *dm_, 1, 1);  // >= 1 ghost (input of A)
    for (int li = 0; li < phi_exact_mf_->local_size(); ++li) {
      Array4 af = phi_exact_mf_->fab(li).array();
      for_each_cell(phi_exact_mf_->box(li), SampleExactKernel{af, geom});
    }
    rhs_ = new MultiFab(*ba_, *dm_, 1, 0);
    (*A_)(*rhs_, *phi_exact_mf_);  // rhs <- A(phi_exact): discrete RHS (tests the SOLVER, not the scheme)

    // NON-symmetric operator A_ns(in) = in - alpha*Lap(in) + beta * upwind dx(in): the Helmholtz
    // part is SPD, the one-sided advection term breaks symmetry. beta is large enough that the
    // operator is strongly non-self-adjoint (CG stagnates), but the spectrum stays in the right
    // half-plane so GMRES converges. Reuses lap_tmp; `in`'s periodic ghosts feed the upwind in(i-1).
    constexpr Real kBeta = 2.0;  // advection strength (CFL-irrelevant: this is a linear solve, not a step)
    const Real inv_h = Real(1) / geom.dx();
    A_ns_ = new ApplyFn([&geom, &bc, &lap_tmp, inv_h](MultiFab& out, const MultiFab& in) {
      MultiFab& in_mut = const_cast<MultiFab&>(in);
      fill_ghosts(in_mut, geom.domain, bc);
      apply_laplacian(in_mut, geom, lap_tmp);
      for (int li = 0; li < out.local_size(); ++li) {
        Array4 ov = out.fab(li).array();
        const ConstArray4 iv = in.fab(li).const_array();
        const ConstArray4 lv = lap_tmp.fab(li).const_array();
        for_each_cell(out.box(li), AdvectionHelmholtzKernel{ov, iv, lv, kAlpha, kBeta, inv_h});
      }
    });
    rhs_ns_ = new MultiFab(*ba_, *dm_, 1, 0);
    (*A_ns_)(*rhs_ns_, *phi_exact_mf_);  // rhs_ns <- A_ns(phi_exact): discrete RHS for the non-symmetric solve
  }

  static void TearDownTestSuite() {
    delete rhs_ns_;
    rhs_ns_ = nullptr;
    delete A_ns_;
    A_ns_ = nullptr;
    delete rhs_;
    rhs_ = nullptr;
    delete phi_exact_mf_;
    phi_exact_mf_ = nullptr;
    delete A_;
    A_ = nullptr;
    delete lap_tmp_;
    lap_tmp_ = nullptr;
    delete bc_;
    bc_ = nullptr;
    delete dm_;
    dm_ = nullptr;
    delete ba_;
    ba_ = nullptr;
    delete geom_;
    geom_ = nullptr;
    delete dom_;
    dom_ = nullptr;
  }

  static Box2D* dom_;
  static Geometry* geom_;
  static BoxArray* ba_;
  static DistributionMapping* dm_;
  static BCRec* bc_;
  static MultiFab* lap_tmp_;
  static ApplyFn* A_;
  static ApplyFn* A_ns_;
  static MultiFab* phi_exact_mf_;
  static MultiFab* rhs_;
  static MultiFab* rhs_ns_;
};

Box2D* GenericKrylov::dom_ = nullptr;
Geometry* GenericKrylov::geom_ = nullptr;
BoxArray* GenericKrylov::ba_ = nullptr;
DistributionMapping* GenericKrylov::dm_ = nullptr;
BCRec* GenericKrylov::bc_ = nullptr;
MultiFab* GenericKrylov::lap_tmp_ = nullptr;
ApplyFn* GenericKrylov::A_ = nullptr;
ApplyFn* GenericKrylov::A_ns_ = nullptr;
MultiFab* GenericKrylov::phi_exact_mf_ = nullptr;
MultiFab* GenericKrylov::rhs_ = nullptr;
MultiFab* GenericKrylov::rhs_ns_ = nullptr;

}  // namespace

// --- CG (SPD operator) ---
TEST_F(GenericKrylov, cg_converges_on_spd_operator) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r = cg_solve(*A_, x, *rhs_, kRelTol, 500);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("CG        : %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "cg_converged";
  EXPECT_TRUE(r.iters > 1) << "cg_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10) << "cg_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "cg_recovers_exact err=" << err;
}

// --- BiCGStab (identity preconditioner = empty ApplyFn) ---
TEST_F(GenericKrylov, bicgstab_converges_with_identity_preconditioner) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r = bicgstab_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 500);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("BiCGStab  : %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "bicgstab_converged";
  EXPECT_TRUE(r.iters > 1) << "bicgstab_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "bicgstab_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "bicgstab_recovers_exact err=" << err;
}

// --- Richardson (omega ~ 1/spectral-max; needs many more iters) ---
TEST_F(GenericKrylov, richardson_converges_with_spectral_omega) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  // Richardson is stable only for omega < 2/lambda_max(A). The DISCRETE -Lap has largest
  // eigenvalue 8/h^2 (NOT the continuous 8*pi^2: the continuous value would over-relax by the grid
  // factor and DIVERGE), so lambda_max(A) = 1 + alpha*8/h^2. omega = 1/lambda_max under-relaxes
  // safely for every mode; convergence is slow (the low modes have rate ~1 - omega) so we grant a
  // large iteration budget.
  const Real h = geom_->dx();  // = geom.dy() on the unit square (uniform)
  const Real lambda_max = Real(1) + kAlpha * Real(8) / (h * h);
  const Real omega = Real(1) / lambda_max;
  const SolveReport r = richardson_solve(*A_, x, *rhs_, omega, kRelTol, 200000);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("Richardson: %s in %d iters (rel=%.2e, omega=%.4f) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, omega, err);
  EXPECT_TRUE(r.solved()) << "richardson_converged";
  EXPECT_TRUE(r.iters > 1) << "richardson_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "richardson_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "richardson_recovers_exact err=" << err;
}

// --- GMRES on the SPD operator (identity preconditioner): must recover phi_exact like CG ---
TEST_F(GenericKrylov, gmres_converges_on_spd_operator) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r = gmres_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 500, 30);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("GMRES(SPD): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "gmres_spd_converged";
  EXPECT_TRUE(r.iters > 1) << "gmres_spd_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "gmres_spd_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "gmres_spd_recovers_exact err=" << err;
}

// --- GMRES on the NON-symmetric operator: the gmres-specific guard. CG STAGNATES on A_ns (it is
//     not self-adjoint), so we first confirm CG fails to recover phi_exact, then GMRES does. ---
TEST_F(GenericKrylov, gmres_converges_where_cg_stagnates_on_nonsymmetric_operator) {
  MultiFab x_cg(*ba_, *dm_, 1, 1);
  x_cg.set_val(0.0);
  const SolveReport rc = cg_solve(*A_ns_, x_cg, *rhs_ns_, kRelTol, 500);
  const Real err_cg = max_abs_diff(x_cg, *phi_exact_mf_);
  std::printf(
      "CG(nonsym): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e (expected to NOT "
      "recover)\n",
      rc.solved() ? "CONVERGED" : "FAILED", rc.iters, rc.rel_residual, err_cg);
  EXPECT_TRUE(!(rc.solved() && err_cg < kRecoverTol)) << "cg_stagnates_on_nonsymmetric";

  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r = gmres_solve(*A_ns_, ApplyFn{}, x, *rhs_ns_, kRelTol, 500, 30);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("GMRES(nsy): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "gmres_nonsym_converged";
  EXPECT_TRUE(r.iters > 1) << "gmres_nonsym_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "gmres_nonsym_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "gmres_nonsym_recovers_exact err=" << err;
}

// --- max_iters <= 0 must throw (spec error 13) ---
TEST_F(GenericKrylov, zero_max_iters_throws_invalid_argument) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  EXPECT_THROW(cg_solve(*A_, x, *rhs_, kRelTol, 0), std::invalid_argument)
      << "cg_max_iters_0_throws";
  EXPECT_THROW(richardson_solve(*A_, x, *rhs_, Real(0.1), kRelTol, 0), std::invalid_argument)
      << "richardson_max_iters_0_throws";
  EXPECT_THROW(bicgstab_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 0), std::invalid_argument)
      << "bicgstab_max_iters_0_throws";
  EXPECT_THROW(gmres_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 0), std::invalid_argument)
      << "gmres_max_iters_0_throws";
}

TEST_F(GenericKrylov, gmres_restart_is_exact_or_rejected) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  EXPECT_THROW(gmres_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 500, 0), std::invalid_argument)
      << "gmres_restart_0_rejected";
  EXPECT_THROW(gmres_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 500, -3), std::invalid_argument)
      << "gmres_restart_negative_rejected";
  EXPECT_THROW(gmres_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 500, 51), std::invalid_argument)
      << "gmres_restart_above_stack_cap_rejected";

  EXPECT_NO_THROW((void)gmres_solve(*A_, ApplyFn{}, x, *rhs_, kRelTol, 500, 1))
      << "gmres_restart_1_is_valid_and_not_clamped_or_rejected";
}

TEST_F(GenericKrylov, failed_solves_report_no_solved_value) {
  MultiFab x_limit(*ba_, *dm_, 1, 1);
  x_limit.set_val(0.0);
  const SolveReport limited = richardson_solve(*A_, x_limit, *rhs_, Real(1e-12), kRelTol, 1);
  EXPECT_FALSE(limited.solved_value_available());
  EXPECT_EQ(limited.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(limited.action, SolveAction::kFailRun);

  ApplyFn zero_op = [](MultiFab& out, const MultiFab&) { out.set_val(0.0); };
  MultiFab x_break(*ba_, *dm_, 1, 1);
  x_break.set_val(0.0);
  const SolveReport breakdown = cg_solve(zero_op, x_break, *rhs_, kRelTol, 10);
  EXPECT_FALSE(breakdown.solved_value_available());
  EXPECT_EQ(breakdown.status, SolveStatus::kBreakdown);
  EXPECT_EQ(breakdown.action, SolveAction::kFailRun);
}
