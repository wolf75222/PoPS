// Generic prepared MATRIX-FREE Krylov layer (generic_krylov.hpp): four solver loops -- Richardson,
// CG, BiCGStab and GMRES -- consume one snapshot-authenticated affine operator and persistent
// workspace. ProgramContext/codegen and direct native callers therefore share the same typed core;
// this test is PURE C++ and validates that core in isolation.
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
// We validate the four typed prepared methods:
//   - CG              (authenticated SPD operator),
//   - BiCGStab        (identity prepared preconditioner),
//   - Richardson      (omega = 1/(1 + alpha*8*pi^2) ~ 1/spectral-max, more iters allowed),
//   - GMRES           (restarted GMRES(m), identity prepared preconditioner): on the SPD operator it matches
//                      CG, and on a NON-symmetric operator (Helmholtz + a one-sided advection term,
//                      where CG STAGNATES) it converges to phi_exact. The non-symmetric case is the
//                      GMRES-specific guard -- CG refuses the same uncertified operator.
// Each must return a solved report (iters > 1, small residual) and recover phi_exact. We also
// assert that max_iters = 0 throws std::invalid_argument (spec error 13).
//
// SERIAL + MPI test: the serial registration retains one box; under a live multi-rank communicator
// the domain is tiled into four boxes and round-robin distributed. CTest registers a focused real
// np=2 variant, so CG, GMRES and BiCGStab execute with cross-rank halo exchange and collective
// reductions without also paying for the intentionally slow Richardson stress test.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/runtime/program/coeff_elliptic_ops.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic ghost exchange)
#include <pops/parallel/comm.hpp>

#include "test_harness.hpp"  // pops::test::kPi

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace pops;
using pops::test::kPi;

namespace {

class CommEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { comm_init(); }
  void TearDown() override { comm_finalize(); }
};

::testing::Environment* const kCommEnv =
    ::testing::AddGlobalTestEnvironment(new CommEnvironment);

#if defined(POPS_HAS_KOKKOS)
// Every case launches field-algebra and stencil kernels, including tests that only inspect the
// prepared snapshot after the suite-level manufactured fields have been assembled. Keep Kokkos
// alive for the complete GoogleTest process instead of constructing a per-test guard after those
// suite resources already exist.
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

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

// Radius-three periodic SPD operator.  This deliberately reads the outermost declared halo so the
// footprint test proves communication/storage depth rather than merely carrying an unused number.
struct RadiusThreeKernel {
  Array4 outv;
  ConstArray4 inv;
  POPS_HD void operator()(int i, int j) const {
    outv(i, j) = Real(3) * inv(i, j) - inv(i - 3, j) - inv(i + 3, j);
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
    if (n_ranks() > 1)
      ba_ = new BoxArray(BoxArray::from_domain(*dom_, kN / 2));
    else
      ba_ = new BoxArray(std::vector<Box2D>{*dom_});
    dm_ = new DistributionMapping(ba_->size(), n_ranks());
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
      apply_laplacian(in_mut, geom,
                      lap_tmp);  // lap_tmp = Lap(in) (all coeffs null -> bare Laplacian)
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
    (*A_)(*rhs_,
          *phi_exact_mf_);  // rhs <- A(phi_exact): discrete RHS (tests the SOLVER, not the scheme)

    // NON-symmetric operator A_ns(in) = in - alpha*Lap(in) + beta * upwind dx(in): the Helmholtz
    // part is SPD, the one-sided advection term breaks symmetry. beta is large enough that the
    // operator is strongly non-self-adjoint (CG stagnates), but the spectrum stays in the right
    // half-plane so GMRES converges. Reuses lap_tmp; `in`'s periodic ghosts feed the upwind in(i-1).
    constexpr Real kBeta =
        2.0;  // advection strength (CFL-irrelevant: this is a linear solve, not a step)
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
    (*A_ns_)(
        *rhs_ns_,
        *phi_exact_mf_);  // rhs_ns <- A_ns(phi_exact): discrete RHS for the non-symmetric solve
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

OperatorEvaluationSnapshot snapshot_for(const MultiFab& prototype, std::uint64_t revision = 1) {
  return {{UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_C(4)},
          revision,
          0,
          0,
          1,
          std::bit_cast<std::uint64_t>(1.0),
          0,
          1,
          detail::layout_fingerprint(prototype),
          {revision, UINT64_C(5), UINT64_C(6), UINT64_C(7)}};
}

SolveReport run_prepared_with_preconditioner(const ApplyFn& apply, MultiFab& iterate,
                                             const MultiFab& rhs, KrylovMethod method,
                                             LinearOperatorProperties properties,
                                             PreparedLinearPreconditioner preconditioner,
                                             Real rel_tol, Real abs_tol, int max_iterations,
                                             int restart, Real relaxation) {
  KrylovFootprint footprint{iterate.ncomp(), iterate.n_grow(), restart,
                            !preconditioner.is_identity()};
  OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
  PreparedAffineLinearProblem problem(iterate, apply, std::move(preconditioner), properties,
                                      footprint, [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  return solve_prepared_affine(
      problem, workspace, iterate, rhs,
      KrylovControls{method, rel_tol, abs_tol, max_iterations, restart, relaxation});
}

SolveReport run_prepared(const ApplyFn& apply, MultiFab& iterate, const MultiFab& rhs,
                         KrylovMethod method, LinearOperatorProperties properties,
                         Real rel_tol = kRelTol, Real abs_tol = Real(0), int max_iterations = 500,
                         int restart = 0, Real relaxation = Real(1)) {
  return run_prepared_with_preconditioner(apply, iterate, rhs, method, properties,
                                          PreparedLinearPreconditioner::identity(), rel_tol,
                                          abs_tol, max_iterations, restart, relaxation);
}

PreparedLinearPreconditioner affine_scaled_identity_preconditioner(const MultiFab& prototype,
                                                                   Real scale, Real offset) {
  auto constant = std::make_shared<MultiFab>(prototype.box_array(), prototype.dmap(),
                                             prototype.ncomp(), prototype.n_grow());
  constant->set_val(offset);
  return PreparedLinearPreconditioner(prototype,
                                      [scale, constant](MultiFab& out, const MultiFab& in) {
                                        PureFieldAlgebra::lincomb(out, scale, in, Real(0), in);
                                        PureFieldAlgebra::axpy(out, Real(1), *constant);
                                      });
}

struct PreparedMgTestContext {
  GridContext grid;
  mutable int kernel_count = 0;

  bool is_polar_geometry() const { return false; }
  GridContext grid_context() const { return grid; }
  void count_kernel() const { ++kernel_count; }
};

}  // namespace

TEST(test_solve_report, rejects_incoherent_status_action_pairs) {
  SolveReport report;
  report.status = SolveStatus::kSolved;
  report.action = SolveAction::kFailRun;
  EXPECT_FALSE(report.valid());
  EXPECT_FALSE(report.solved());
  EXPECT_FALSE(report.solved_value_available());
  EXPECT_THROW(report.mark_failed(SolveStatus::kSolved), std::invalid_argument);
  EXPECT_THROW(report.mark_failed(SolveStatus::kBreakdown, SolveAction::kNone),
               std::invalid_argument);
}

TEST_F(GenericKrylov, footprint_authenticates_layout_and_preconditioner_presence) {
  MultiFab no_ghosts(*ba_, *dm_, 1, 0);
  MultiFab one_ghost(*ba_, *dm_, 1, 1);
  EXPECT_NE(detail::layout_fingerprint(no_ghosts), detail::layout_fingerprint(one_ghost));

  ApplyFn identity_apply = [](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
  };
  OperatorEvaluationSnapshot snapshot = snapshot_for(one_ghost);
  OperatorSnapshotProbe probe = [&snapshot]() { return snapshot; };
  const KrylovFootprint claims_preconditioner{1, 1, 0, true};
  const KrylovFootprint hides_preconditioner{1, 1, 0, false};

  EXPECT_THROW((void)PreparedAffineLinearProblem(
                   one_ghost, identity_apply, PreparedLinearPreconditioner::identity(),
                   LinearOperatorProperties::general(), claims_preconditioner, probe),
               std::invalid_argument);
  EXPECT_THROW(
      (void)PreparedAffineLinearProblem(
          one_ghost, identity_apply, PreparedLinearPreconditioner(one_ghost, identity_apply),
          LinearOperatorProperties::general(), hides_preconditioner, probe),
      std::invalid_argument);
  EXPECT_THROW((void)KrylovWorkspace(one_ghost, KrylovMethod::kCg, claims_preconditioner),
               std::invalid_argument);
  EXPECT_THROW((void)KrylovWorkspace(one_ghost, KrylovMethod::kRichardson, claims_preconditioner),
               std::invalid_argument);

  MultiFab two_components(*ba_, *dm_, 2, 1);
  EXPECT_THROW(
      (void)PreparedAffineLinearProblem(
          one_ghost, identity_apply,
          PreparedLinearPreconditioner(two_components, identity_apply),
          LinearOperatorProperties::general(), claims_preconditioner, probe),
      std::invalid_argument);

  EXPECT_THROW(
      (void)PreparedAffineLinearProblem(
          one_ghost, identity_apply,
          PreparedLinearPreconditioner(no_ghosts, identity_apply),
          LinearOperatorProperties::general(), claims_preconditioner, probe),
      std::invalid_argument);

  const BoxArray other_boxes = n_ranks() > 1
                                   ? BoxArray(std::vector<Box2D>{*dom_})
                                   : BoxArray::from_domain(*dom_, kN / 2);
  const DistributionMapping other_mapping(other_boxes.size(), n_ranks());
  MultiFab other_layout(other_boxes, other_mapping, 1, 1);
  EXPECT_THROW(
      (void)PreparedAffineLinearProblem(
          one_ghost, identity_apply, PreparedLinearPreconditioner(other_layout, identity_apply),
          LinearOperatorProperties::general(), claims_preconditioner, probe),
      std::invalid_argument);

  if (n_ranks() > 1) {
    const DistributionMapping rank_zero_only(
        std::vector<int>(static_cast<std::size_t>(ba_->size()), 0));
    MultiFab empty_on_remote_rank(*ba_, rank_zero_only, 1, 1);
    EXPECT_THROW(
        (void)PreparedAffineLinearProblem(
            one_ghost, identity_apply,
            PreparedLinearPreconditioner(empty_on_remote_rank, identity_apply),
            LinearOperatorProperties::general(), claims_preconditioner, probe),
        std::invalid_argument);
  }
}

TEST_F(GenericKrylov, geometric_preconditioner_validates_controls_and_rebuilds_on_context_change) {
  using pops::runtime::program::GeometricMgPreconditioner;
  EXPECT_THROW((void)GeometricMgPreconditioner(-1, 2, 50, 2, 1), std::invalid_argument);
  EXPECT_THROW((void)GeometricMgPreconditioner(2, 2, 0, 2, 1), std::invalid_argument);
  EXPECT_THROW((void)GeometricMgPreconditioner(2, 2, 50, 0, 1), std::invalid_argument);
  EXPECT_THROW((void)GeometricMgPreconditioner(2, 2, 50, 2, 0), std::invalid_argument);

  MultiFab prototype(*ba_, *dm_, 1, 1);
  MultiFab output(*ba_, *dm_, 1, 1);
  PreparedMgTestContext context{GridContext{}};
  context.grid.dom = geom_->domain;
  context.grid.geom = *geom_;
  context.grid.bc = *bc_;
  GeometricMgPreconditioner preconditioner(0, 0, 1, 2, 1);
  EXPECT_EQ(preconditioner.preparation_generation(), 0u);
  EXPECT_THROW(preconditioner.apply(context, output, prototype), std::logic_error);

  MultiFab two_components(*ba_, *dm_, 2, 1);
  EXPECT_THROW(preconditioner.prepare(context, two_components), std::invalid_argument);

  EXPECT_NO_THROW(preconditioner.prepare(context, prototype));
  const std::uint64_t generation = preconditioner.preparation_generation();
  EXPECT_EQ(generation, 1u);
  EXPECT_NO_THROW(preconditioner.prepare(context, prototype));
  EXPECT_EQ(preconditioner.preparation_generation(), generation);

  const BoxArray incompatible_boxes = n_ranks() > 1
                                          ? BoxArray(std::vector<Box2D>{*dom_})
                                          : BoxArray::from_domain(*dom_, kN / 2);
  const DistributionMapping incompatible_mapping(incompatible_boxes.size(), n_ranks());
  MultiFab incompatible(incompatible_boxes, incompatible_mapping, 1, 1);
  EXPECT_THROW(preconditioner.apply(context, incompatible, incompatible), std::invalid_argument);

  context.grid.geom.xhi += 1.0;
  EXPECT_NO_THROW(preconditioner.prepare(context, prototype));
  EXPECT_EQ(preconditioner.preparation_generation(), generation + 1);

  context.grid.bc.xlo = BCType::Dirichlet;
  EXPECT_NO_THROW(preconditioner.prepare(context, prototype));
  EXPECT_EQ(preconditioner.preparation_generation(), generation + 2);
}

TEST_F(GenericKrylov, arbitrary_halo_depth_is_a_first_class_prepared_footprint) {
  MultiFab x(*ba_, *dm_, 1, 3);
  MultiFab exact(*ba_, *dm_, 1, 3);
  MultiFab rhs(*ba_, *dm_, 1, 0);
  PureFieldAlgebra::zero(x);
  for (int li = 0; li < exact.local_size(); ++li) {
    Array4 values = exact.fab(li).array();
    for_each_cell(exact.box(li), SampleExactKernel{values, *geom_});
  }
  ApplyFn radius_three = [this](MultiFab& out, const MultiFab& in) {
    MultiFab& mutable_in = const_cast<MultiFab&>(in);
    fill_ghosts(mutable_in, geom_->domain, *bc_);
    for (int li = 0; li < out.local_size(); ++li) {
      Array4 output = out.fab(li).array();
      const ConstArray4 input = in.fab(li).const_array();
      for_each_cell(out.box(li), RadiusThreeKernel{output, input});
    }
  };
  radius_three(rhs, exact);

  const SolveReport report = run_prepared(radius_three, x, rhs, KrylovMethod::kCg,
                                          LinearOperatorProperties::symmetric_positive_definite());
  EXPECT_TRUE(report.solved());
  EXPECT_EQ(x.n_grow(), 3);
  EXPECT_LT(max_abs_diff(x, exact), kRecoverTol);
}

TEST_F(GenericKrylov, mpi_variant_distributes_real_work_to_every_rank) {
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    ASSERT_EQ(n_ranks(), std::atoi(expected_ranks))
        << "the MPI CTest route must initialize the requested communicator";
  else if (n_ranks() == 1)
    GTEST_SKIP() << "the serial registration has no remote rank";
  MultiFab field(*ba_, *dm_, 1, 1);
  EXPECT_GT(field.local_size(), 0)
      << "the np=2 generic Krylov variant must not leave a rank as a collective-only spectator";
  EXPECT_EQ(static_cast<int>(all_reduce_sum(static_cast<double>(field.local_size()))), ba_->size());
}

TEST_F(GenericKrylov, geometric_preconditioner_preserves_a_custom_distribution) {
  std::vector<int> owners(static_cast<std::size_t>(ba_->size()), 0);
  for (std::size_t index = 0; index < owners.size(); ++index)
    owners[index] = n_ranks() > 1 ? (static_cast<int>(index) + 1) % n_ranks() : 0;
  const DistributionMapping custom(owners);
  GeometricMG preconditioner(*geom_, *ba_, custom, *bc_);
  EXPECT_EQ(preconditioner.phi().dmap().ranks(), custom.ranks());
  EXPECT_EQ(preconditioner.rhs().dmap().ranks(), custom.ranks());
  for (int li = 0; li < preconditioner.rhs().local_size(); ++li) {
    Array4 forcing = preconditioner.rhs().fab(li).array();
    for_each_cell(preconditioner.rhs().box(li), SampleExactKernel{forcing, *geom_});
  }
  scale(preconditioner.rhs(), Real(-8) * kPi * kPi);
  PureFieldAlgebra::zero_valid(preconditioner.phi());
  const Real initial_residual = preconditioner.current_residual();
  ASSERT_GT(initial_residual, Real(0));
  const int cycles = preconditioner.solve(Real(1e-8), 100);
  EXPECT_GT(cycles, 0);
  EXPECT_TRUE(preconditioner.last_solve_report().solved())
      << preconditioner.last_solve_report().status_name();
  EXPECT_LE(preconditioner.last_residual(), initial_residual * Real(1e-7));
}

TEST_F(GenericKrylov, cg_converges_on_spd_operator) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r = run_prepared(*A_, x, *rhs_, KrylovMethod::kCg,
                                     LinearOperatorProperties::symmetric_positive_definite());
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("CG        : %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "cg_converged";
  EXPECT_TRUE(r.iters > 1) << "cg_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "cg_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "cg_recovers_exact err=" << err;
}

TEST_F(GenericKrylov, bicgstab_converges_with_identity_preconditioner) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r =
      run_prepared(*A_, x, *rhs_, KrylovMethod::kBicgstab, LinearOperatorProperties::general());
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("BiCGStab  : %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "bicgstab_converged";
  EXPECT_TRUE(r.iters > 1) << "bicgstab_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "bicgstab_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "bicgstab_recovers_exact err=" << err;
}

TEST_F(GenericKrylov, bicgstab_avoids_a_true_residual_matvec_on_regular_iterations) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));
  int operator_calls = 0;
  ApplyFn counted = [this, &operator_calls](MultiFab& out, const MultiFab& in) {
    ++operator_calls;
    (*A_ns_)(out, in);
  };

  const SolveReport report =
      run_prepared(counted, x, *rhs_ns_, KrylovMethod::kBicgstab,
                   LinearOperatorProperties::general(), Real(0), Real(1e-30), 2);

  EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
  // prepare(A(0)) + initial true residual + 2*(Ap + As) + final true residual.
  EXPECT_EQ(operator_calls, 7);
}

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
  const SolveReport r =
      run_prepared(*A_, x, *rhs_, KrylovMethod::kRichardson, LinearOperatorProperties::general(),
                   kRelTol, Real(0), 200000, 0, omega);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("Richardson: %s in %d iters (rel=%.2e, omega=%.4f) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, omega, err);
  EXPECT_TRUE(r.solved()) << "richardson_converged";
  EXPECT_TRUE(r.iters > 1) << "richardson_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "richardson_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "richardson_recovers_exact err=" << err;
}

TEST_F(GenericKrylov, gmres_converges_on_spd_operator) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r =
      run_prepared(*A_, x, *rhs_, KrylovMethod::kGmres, LinearOperatorProperties::general(),
                   kRelTol, Real(0), 500, 30);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("GMRES(SPD): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "gmres_spd_converged";
  EXPECT_TRUE(r.iters > 1) << "gmres_spd_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "gmres_spd_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "gmres_spd_recovers_exact err=" << err;
}

TEST_F(GenericKrylov, gmres_restart_decision_is_invariant_to_preconditioner_scale) {
  MultiFab x_unit(*ba_, *dm_, 1, 1);
  MultiFab x_scaled(*ba_, *dm_, 1, 1);
  x_unit.set_val(0.0);
  x_scaled.set_val(0.0);

  const SolveReport unit = run_prepared_with_preconditioner(
      *A_, x_unit, *rhs_, KrylovMethod::kGmres, LinearOperatorProperties::general(),
      affine_scaled_identity_preconditioner(x_unit, Real(1), Real(3.25)), kRelTol, Real(0), 500, 30,
      Real(1));
  // Scale the affine response with the linear map as a physical rescaling of one preconditioner
  // would do. Keeping a fixed O(1) offset while shrinking only the direction response to 1e-8
  // makes M_raw(v)-M_raw(0) an intentionally ill-conditioned floating-point subtraction and no
  // longer tests restart invariance of equivalent preconditioners.
  const SolveReport scaled = run_prepared_with_preconditioner(
      *A_, x_scaled, *rhs_, KrylovMethod::kGmres, LinearOperatorProperties::general(),
      affine_scaled_identity_preconditioner(x_scaled, Real(1e-8), Real(3.25e-8)), kRelTol, Real(0),
      500, 30, Real(1));

  EXPECT_TRUE(unit.solved()) << "unit-scaled preconditioner failed after " << unit.iters;
  EXPECT_TRUE(scaled.solved()) << "small-scaled preconditioner failed after " << scaled.iters;
  EXPECT_LE(scaled.iters, unit.iters + 1);
  EXPECT_GE(scaled.iters + 1, unit.iters);
  EXPECT_TRUE(max_abs_diff(x_unit, *phi_exact_mf_) < kRecoverTol);
  EXPECT_TRUE(max_abs_diff(x_scaled, *phi_exact_mf_) < kRecoverTol);
}

TEST_F(GenericKrylov, bicgstab_linearizes_an_affine_preconditioner) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));
  const SolveReport report = run_prepared_with_preconditioner(
      *A_, x, *rhs_, KrylovMethod::kBicgstab, LinearOperatorProperties::general(),
      affine_scaled_identity_preconditioner(x, Real(0.75), Real(8.0)), kRelTol, Real(0), 500, 0,
      Real(1));
  EXPECT_TRUE(report.solved());
  EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol);
}

TEST_F(GenericKrylov, cg_refuses_unproven_operator_and_gmres_solves_nonsymmetric_operator) {
  MultiFab x_cg(*ba_, *dm_, 1, 1);
  x_cg.set_val(0.0);
  EXPECT_THROW((void)run_prepared(*A_ns_, x_cg, *rhs_ns_, KrylovMethod::kCg,
                                  LinearOperatorProperties::general()),
               std::invalid_argument);

  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r =
      run_prepared(*A_ns_, x, *rhs_ns_, KrylovMethod::kGmres, LinearOperatorProperties::general(),
                   kRelTol, Real(0), 500, 30);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("GMRES(nsy): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "gmres_nonsym_converged";
  EXPECT_TRUE(r.iters > 1) << "gmres_nonsym_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "gmres_nonsym_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "gmres_nonsym_recovers_exact err=" << err;
}

TEST_F(GenericKrylov, zero_max_iters_throws_invalid_argument) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  EXPECT_THROW((void)run_prepared(*A_, x, *rhs_, KrylovMethod::kCg,
                                  LinearOperatorProperties::symmetric_positive_definite(), kRelTol,
                                  Real(0), 0),
               std::invalid_argument);

  EXPECT_THROW((void)run_prepared(*A_, x, *rhs_, KrylovMethod::kCg,
                                  LinearOperatorProperties::symmetric_positive_definite(), kRelTol,
                                  Real(0), 10, 0, Real(0.5)),
               std::invalid_argument);
}

TEST_F(GenericKrylov, gmres_restart_is_exact_and_dynamically_sized) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  EXPECT_THROW((void)run_prepared(*A_, x, *rhs_, KrylovMethod::kGmres,
                                  LinearOperatorProperties::general(), kRelTol, Real(0), 500, 0),
               std::invalid_argument);
  const SolveReport dynamic =
      run_prepared(*A_, x, *rhs_, KrylovMethod::kGmres, LinearOperatorProperties::general(),
                   kRelTol, Real(0), 500, 51);
  EXPECT_TRUE(dynamic.solved());
}

TEST_F(GenericKrylov, failed_solves_report_no_solved_value) {
  MultiFab x_limit(*ba_, *dm_, 1, 1);
  x_limit.set_val(0.0);
  const SolveReport limited =
      run_prepared(*A_, x_limit, *rhs_, KrylovMethod::kRichardson,
                   LinearOperatorProperties::general(), kRelTol, Real(0), 1, 0, Real(1e-12));
  EXPECT_FALSE(limited.solved_value_available());
  EXPECT_EQ(limited.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(limited.action, SolveAction::kFailRun);

  ApplyFn zero_op = [](MultiFab& out, const MultiFab&) { out.set_val(0.0); };
  MultiFab x_break(*ba_, *dm_, 1, 1);
  x_break.set_val(0.0);
  const SolveReport breakdown =
      run_prepared(zero_op, x_break, *rhs_, KrylovMethod::kCg,
                   LinearOperatorProperties::symmetric_positive_definite(), kRelTol, Real(0), 10);
  EXPECT_FALSE(breakdown.solved_value_available());
  EXPECT_EQ(breakdown.status, SolveStatus::kBreakdown);
  EXPECT_EQ(breakdown.action, SolveAction::kFailRun);
}

TEST_F(GenericKrylov, affine_constant_is_removed_exactly) {
  constexpr Real offset = Real(3.25);
  MultiFab constant(*ba_, *dm_, 1, 0);
  constant.set_val(offset);
  ApplyFn affine = [&](MultiFab& out, const MultiFab& in) {
    (*A_)(out, in);
    PureFieldAlgebra::axpy(out, Real(1), constant);
  };
  MultiFab affine_rhs(*ba_, *dm_, 1, 0);
  affine(affine_rhs, *phi_exact_mf_);
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport report = run_prepared(affine, x, affine_rhs, KrylovMethod::kCg,
                                          LinearOperatorProperties::symmetric_positive_definite());
  EXPECT_TRUE(report.solved());
  EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol);
}

TEST_F(GenericKrylov, every_method_removes_the_affine_operator_constant) {
  MultiFab constant(*ba_, *dm_, 1, 1);
  constant.set_val(Real(2.75));
  ApplyFn affine_identity = [&](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
    PureFieldAlgebra::axpy(out, Real(1), constant);
  };
  MultiFab affine_rhs(*ba_, *dm_, 1, 0);
  affine_identity(affine_rhs, *phi_exact_mf_);

  struct MethodCase {
    KrylovMethod method;
    int restart;
  };
  const std::array<MethodCase, 4> methods{{
      {KrylovMethod::kCg, 0},
      {KrylovMethod::kBicgstab, 0},
      {KrylovMethod::kGmres, 4},
      {KrylovMethod::kRichardson, 0},
  }};
  for (const MethodCase& method : methods) {
    MultiFab x(*ba_, *dm_, 1, 1);
    x.set_val(Real(0));
    const LinearOperatorProperties properties =
        method.method == KrylovMethod::kCg ? LinearOperatorProperties::symmetric_positive_definite()
                                           : LinearOperatorProperties::general();
    const SolveReport report =
        run_prepared(affine_identity, x, affine_rhs, method.method, properties, kRelTol, Real(0), 8,
                     method.restart, Real(1));
    EXPECT_TRUE(report.solved()) << "method=" << static_cast<int>(method.method);
    EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol)
        << "method=" << static_cast<int>(method.method);
  }
}

TEST_F(GenericKrylov, warm_start_and_absolute_floor_use_true_residual) {
  MultiFab exact(*ba_, *dm_, 1, 1);
  PureFieldAlgebra::copy(exact, *phi_exact_mf_);
  const SolveReport warm =
      run_prepared(*A_, exact, *rhs_, KrylovMethod::kGmres, LinearOperatorProperties::general(),
                   kRelTol, Real(0), 50, 10);
  EXPECT_TRUE(warm.solved());
  EXPECT_EQ(warm.iters, 0);

  MultiFab zero(*ba_, *dm_, 1, 1);
  zero.set_val(0.0);
  const Real reference = PureFieldAlgebra::norm(*rhs_);
  const SolveReport absolute =
      run_prepared(*A_, zero, *rhs_, KrylovMethod::kBicgstab, LinearOperatorProperties::general(),
                   Real(0), reference, 50);
  EXPECT_TRUE(absolute.solved());
  EXPECT_EQ(absolute.iters, 0);

  ApplyFn identity = [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); };
  MultiFab zero_rhs(*ba_, *dm_, 1, 0);
  MultiFab zero_solution(*ba_, *dm_, 1, 1);
  zero_rhs.set_val(Real(0));
  zero_solution.set_val(Real(0));
  const SolveReport zero_forcing = run_prepared(
      identity, zero_solution, zero_rhs, KrylovMethod::kCg,
      LinearOperatorProperties::symmetric_positive_definite(), Real(0), Real(1e-14), 4);
  EXPECT_TRUE(zero_forcing.solved());
  EXPECT_EQ(zero_forcing.iters, 0);
  EXPECT_EQ(zero_forcing.reference_residual_norm, Real(0));

  MultiFab tiny_rhs(*ba_, *dm_, 1, 0);
  MultiFab tiny_solution(*ba_, *dm_, 1, 1);
  tiny_rhs.set_val(Real(1e-16));
  tiny_solution.set_val(Real(0));
  const SolveReport tiny_forcing = run_prepared(
      identity, tiny_solution, tiny_rhs, KrylovMethod::kCg,
      LinearOperatorProperties::symmetric_positive_definite(), Real(0), Real(1e-14), 4);
  EXPECT_TRUE(tiny_forcing.solved());
  EXPECT_EQ(tiny_forcing.iters, 0);
}

TEST_F(GenericKrylov, snapshot_rejects_nonfinite_or_incoherent_time_identity) {
  OperatorEvaluationSnapshot snapshot = snapshot_for(*phi_exact_mf_);
  EXPECT_TRUE(snapshot.valid());
  snapshot.dt_bits = std::bit_cast<std::uint64_t>(std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(snapshot.valid());
  snapshot = snapshot_for(*phi_exact_mf_);
  snapshot.stage_numerator = 2;
  snapshot.stage_denominator = 1;
  EXPECT_FALSE(snapshot.valid());
  snapshot = snapshot_for(*phi_exact_mf_);
  snapshot.macro_step = -1;
  EXPECT_FALSE(snapshot.valid());
}

TEST_F(GenericKrylov, snapshot_mutation_is_refused_and_workspace_is_reused) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  KrylovFootprint footprint{1, 1, 10, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(x);
  PreparedAffineLinearProblem problem(x, *A_, PreparedLinearPreconditioner::identity(),
                                      LinearOperatorProperties::general(), footprint,
                                      [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(x, KrylovMethod::kGmres, footprint);
  const std::size_t allocations = workspace.allocation_count();
  EXPECT_EQ(workspace.scalar_value_count(), 151u);
  EXPECT_EQ(workspace.collective_value_count(), 11u);
  const KrylovControls controls{KrylovMethod::kGmres, kRelTol, Real(0), 500, 10, Real(1)};

  problem.prepare(snapshot);
  workspace.bind(problem);
  const std::size_t halo_resources = x.halo_cache().exchange_pool_size();
  snapshot.revision += 1;
  EXPECT_THROW((void)solve_prepared_affine(problem, workspace, x, *rhs_, controls),
               std::logic_error);

  snapshot.resources[0] += 1;
  problem.prepare(snapshot);
  workspace.bind(problem);
  const SolveReport first = solve_prepared_affine(problem, workspace, x, *rhs_, controls);
  EXPECT_TRUE(first.solved());
  x.set_val(0.0);
  snapshot.revision += 1;
  problem.prepare(snapshot);
  workspace.bind(problem);
  const SolveReport second = solve_prepared_affine(problem, workspace, x, *rhs_, controls);
  EXPECT_TRUE(second.solved());
  EXPECT_EQ(workspace.allocation_count(), allocations);
  EXPECT_EQ(x.halo_cache().exchange_pool_size(), halo_resources);
}

TEST_F(GenericKrylov, iterate_and_rhs_must_not_alias) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  KrylovFootprint footprint{1, 1, 0, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(x);
  PreparedAffineLinearProblem problem(x, *A_, PreparedLinearPreconditioner::identity(),
                                      LinearOperatorProperties::symmetric_positive_definite(),
                                      footprint, [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(x, KrylovMethod::kCg, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  EXPECT_THROW((void)solve_prepared_affine(
                   problem, workspace, x, x,
                   KrylovControls{KrylovMethod::kCg, kRelTol, Real(0), 10, 0, Real(1)}),
               std::invalid_argument);
}

TEST_F(GenericKrylov, extension_apply_mutation_is_refused_before_result_consumption) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  KrylovFootprint footprint{1, 1, 0, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(x);
  bool mutate_during_apply = false;
  ApplyFn extension_apply = [&snapshot, &mutate_during_apply](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
    if (mutate_during_apply)
      ++snapshot.revision;
  };
  PreparedAffineLinearProblem problem(x, std::move(extension_apply),
                                      PreparedLinearPreconditioner::identity(),
                                      LinearOperatorProperties::symmetric_positive_definite(),
                                      footprint, [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(x, KrylovMethod::kCg, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  mutate_during_apply = true;

  EXPECT_THROW((void)solve_prepared_affine(
                   problem, workspace, x, *rhs_,
                   KrylovControls{KrylovMethod::kCg, kRelTol, Real(0), 10, 0, Real(1)}),
               std::logic_error);
}
