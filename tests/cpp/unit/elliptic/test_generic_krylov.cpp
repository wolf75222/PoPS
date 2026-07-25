// Generic prepared MATRIX-FREE Krylov layer (generic_krylov.hpp): four solver loops -- Richardson,
// CG, BiCGStab and GMRES -- consume one snapshot-authenticated affine operator and persistent
// workspace. ProgramContext/codegen and direct native callers therefore share the same typed core;
// this test is PURE C++ and validates that core in isolation.
//
// OPERATOR: an SPD Helmholtz operator A = I - alpha*Lap (alpha = 0.1), supplied as a prepared
//   session provider that fills the ghosts of `in` (periodic), applies the SHARED discrete 5-point Laplacian
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

#include <limits>

#include <pops/core/foundation/allocator.hpp>
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
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

::testing::Environment* const kCommEnv = ::testing::AddGlobalTestEnvironment(new CommEnvironment);

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

// The scalar fixture above deliberately retains the historical component-zero functor.  The
// prepared vector route needs an explicit component index: using Array4's default component here
// would make an ncomp=2 solve accidentally exercise only the first field.
struct HelmholtzComponentCombineKernel {
  Array4 outv;
  ConstArray4 inv, lapv;
  Real alpha;
  int component;
  POPS_HD void operator()(int i, int j) const {
    outv(i, j, component) = inv(i, j, component) - alpha * lapv(i, j, component);
  }
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
// device-callable from the initialization kernel on every Kokkos backend.
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

// Two deliberately different manufactured fields.  Different amplitudes and modal content make
// a component-indexing error visible even when the field algebra reduces all components together.
struct SampleDistinctComponentExactKernel {
  Array4 values;
  Geometry geometry;
  int component;
  POPS_HD void operator()(int i, int j) const {
    const Real x = geometry.x_cell(i);
    const Real y = geometry.y_cell(j);
    values(i, j, component) =
        component == 0 ? Real(1.25) * phi_exact(x, y)
                       : -Real(0.65) * phi_exact(x, y) +
                             Real(0.35) * std::sin(Real(8) * kPi * x) * std::sin(Real(2) * kPi * y);
  }
};

struct SampleSinglePeriodicModeKernel {
  Array4 values;
  Geometry geometry;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) =
        std::sin(Real(2) * kPi * geometry.x_cell(i)) * std::sin(Real(2) * kPi * geometry.y_cell(j));
  }
};

struct SparseAffineConstantKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = i == 0 && j == 0 ? Real(1e200) : Real(0);
  }
};

struct SparseTinyForcingKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = i == 1 && j == 0 ? Real(1e-200) : Real(0);
  }
};

struct SparseHugeReferenceKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = i == 0 && j == 0 ? Real(1e300) : i == 1 && j == 0 ? Real(1e-300) : Real(0);
  }
};

struct SparseHugeWarmStartKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = i == 0 && j == 0 ? Real(1e300) : Real(0);
  }
};

struct SparseUnitForcingKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = i == 0 && j == 0 ? Real(1) : Real(0);
  }
};

struct SubnormalArnoldiColumnKernel {
  Array4 output;
  ConstArray4 input;
  POPS_HD void operator()(int i, int j) const {
    output(i, j) = input(i, j);
    if (i == 1 && j == 0)
      output(i, j) += Real(1e-310) * input(i - 1, j);
  }
};

struct TwoScaleDiagonalKernel {
  Array4 output;
  ConstArray4 input;
  Real small_scale = Real(1e-320);
  POPS_HD void operator()(int i, int j) const {
    output(i, j) = i == 1 && j == 0 ? small_scale * input(i, j) : input(i, j);
  }
};

struct UnitAndSubnormalForcingKernel {
  Array4 values;
  Real small_scale = Real(1e-320);
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = i == 0 && j == 0 ? Real(1) : i == 1 && j == 0 ? small_scale : Real(0);
  }
};

struct UnitPairSolutionKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) = (i == 0 || i == 1) && j == 0 ? Real(1) : Real(0);
  }
};

// Two-cell matrix [[1, 1], [1, 0]].  For b=(1,0), the first BiCGStab alpha step leaves
// s=(0,-1) and (A s).s=0, exercising the exact omega-breakdown recovery path.
struct TwoCellOmegaBreakdownKernel {
  Array4 output;
  ConstArray4 input;
  POPS_HD void operator()(int i, int j) const {
    output(i, j) = i == 0 ? input(0, j) + input(1, j) : input(0, j);
  }
};

PreparedAffineOperatorProvider reentrant_test_operator(ApplyFn apply) {
  return PreparedAffineOperatorProvider::trusted_reentrant(std::move(apply),
                                                           [] { return std::size_t{0}; });
}

// Mutation-oracle tests deliberately model external physical state tracked by the snapshot probe.
// The callback itself owns no scratch; every session gets its own callback object while observing
// the same external revision source so the prepared boundary can detect illegal drift.
PreparedAffineOperatorProvider externally_observed_test_operator(std::string_view identity,
                                                                 ApplyFn apply) {
  return PreparedAffineOperatorProvider::trusted_extension(
      {identity, 1}, {}, [apply = std::move(apply)](const ExecutionLane&) {
        return PreparedAffineOperatorSessionCallbacks{{}, apply, [] { return std::size_t{0}; }};
      });
}

PreparedAffineOperatorProvider counted_test_operator(PreparedAffineOperatorProvider base,
                                                     std::shared_ptr<std::atomic<int>> calls) {
  const std::string base_contract(base.collective_contract());
  return PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.generic-krylov.counted-operator", 1}, base_contract,
      [base = std::move(base), calls = std::move(calls)](const ExecutionLane& lane) {
        auto session = std::make_shared<PreparedAffineOperatorSession>(base.make_session(lane));
        return PreparedAffineOperatorSessionCallbacks{
            [session] { session->prepare(); },
            [session, calls](MultiFab& out, const MultiFab& in) {
              calls->fetch_add(1, std::memory_order_relaxed);
              if (!prepared_apply_succeeded(session->apply(out, in)))
                throw std::runtime_error("counted test operator apply failed");
            },
            [session] { return session->allocation_count(); }};
      });
}

PreparedAffineOperatorProvider shifted_test_operator(PreparedAffineOperatorProvider base,
                                                     const MultiFab& constant) {
  const std::string base_contract(base.collective_contract());
  return PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.generic-krylov.shifted-operator", 1}, base_contract,
      [base = std::move(base), constant = &constant](const ExecutionLane& lane) {
        auto session = std::make_shared<PreparedAffineOperatorSession>(base.make_session(lane));
        return PreparedAffineOperatorSessionCallbacks{
            [session] { session->prepare(); },
            [session, constant](MultiFab& out, const MultiFab& in) {
              if (!prepared_apply_succeeded(session->apply(out, in)))
                throw std::runtime_error("shifted test operator apply failed");
              PureFieldAlgebra::axpy(out, Real(1), *constant);
            },
            [session] { return session->allocation_count(); }};
      });
}

struct PreparedTestOperator {
  PreparedAffineOperatorProvider provider;

  void apply_once(MultiFab& out, const MultiFab& in) const {
    ExecutionLane lane = ExecutionLane::world("pops.test.generic-krylov.reference-apply");
    PreparedAffineOperatorSession session = provider.make_session(lane);
    session.prepare();
    if (!prepared_apply_succeeded(session.apply(out, in)))
      throw std::runtime_error("prepared test operator apply failed");
  }
};

PreparedTestOperator prepared_helmholtz_operator(const MultiFab& prototype,
                                                 const Geometry& geometry, const BCRec& boundary,
                                                 bool nonsymmetric = false) {
  if (nonsymmetric && prototype.ncomp() != 1)
    throw std::invalid_argument("the nonsymmetric Helmholtz test operator is scalar");
  const BoxArray boxes = prototype.box_array();
  const DistributionMapping mapping = prototype.dmap();
  const int components = prototype.ncomp();
  constexpr Real beta = Real(2);
  const Real inverse_spacing = Real(1) / geometry.dx();
  return {PreparedAffineOperatorProvider::trusted_extension(
      {nonsymmetric ? "pops.test.generic-krylov.advection-helmholtz"
                    : "pops.test.generic-krylov.helmholtz",
       1},
      exact_provider_parameters(geometry.xlo, geometry.xhi, geometry.ylo, geometry.yhi,
                                boundary.xlo, boundary.xhi, boundary.ylo, boundary.yhi, components,
                                nonsymmetric, kAlpha, beta),
      [boxes, mapping, components, geometry, boundary, nonsymmetric,
       inverse_spacing](const ExecutionLane& lane) {
        auto laplacian = std::make_shared<MultiFab>(boxes, mapping, components, 0);
        return PreparedAffineOperatorSessionCallbacks{
            {},
            [laplacian, geometry, boundary, nonsymmetric, inverse_spacing, &lane](
                MultiFab& out, const MultiFab& in) {
              MultiFab& mutable_input = const_cast<MultiFab&>(in);
              fill_ghosts(mutable_input, geometry.domain, boundary, lane);
              apply_laplacian(mutable_input, geometry, *laplacian);
              for (int local = 0; local < out.local_size(); ++local) {
                Array4 output = out.fab(local).array();
                const ConstArray4 input = in.fab(local).const_array();
                const ConstArray4 lap = laplacian->fab(local).const_array();
                if (nonsymmetric) {
                  for_each_cell(out.box(local), AdvectionHelmholtzKernel{output, input, lap, kAlpha,
                                                                         beta, inverse_spacing});
                } else {
                  for (int component = 0; component < out.ncomp(); ++component)
                    for_each_cell(out.box(local), HelmholtzComponentCombineKernel{
                                                      output, input, lap, kAlpha, component});
                }
              }
            },
            [] { return std::size_t{1}; }};
      })};
}

PreparedTestOperator prepared_negative_periodic_laplacian(const MultiFab& prototype,
                                                          const Geometry& geometry,
                                                          const BCRec& boundary) {
  const BoxArray boxes = prototype.box_array();
  const DistributionMapping mapping = prototype.dmap();
  const int components = prototype.ncomp();
  return {PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.generic-krylov.negative-periodic-laplacian", 1},
      exact_provider_parameters(geometry.xlo, geometry.xhi, geometry.ylo, geometry.yhi, components),
      [boxes, mapping, components, geometry, boundary](const ExecutionLane& lane) {
        auto laplacian = std::make_shared<MultiFab>(boxes, mapping, components, 0);
        return PreparedAffineOperatorSessionCallbacks{
            {},
            [laplacian, geometry, boundary, &lane](MultiFab& out, const MultiFab& in) {
              MultiFab& mutable_input = const_cast<MultiFab&>(in);
              fill_ghosts(mutable_input, geometry.domain, boundary, lane);
              apply_laplacian(mutable_input, geometry, *laplacian);
              PureFieldAlgebra::lincomb(out, Real(-1), *laplacian, Real(0), *laplacian);
            },
            [] { return std::size_t{1}; }};
      })};
}

PreparedTestOperator prepared_radius_three_operator(const Geometry& geometry,
                                                    const BCRec& boundary) {
  return {PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.generic-krylov.radius-three", 1},
      exact_provider_parameters(geometry.xlo, geometry.xhi, geometry.ylo, geometry.yhi),
      [geometry, boundary](const ExecutionLane& lane) {
        return PreparedAffineOperatorSessionCallbacks{
            {},
            [geometry, boundary, &lane](MultiFab& out, const MultiFab& in) {
              MultiFab& mutable_input = const_cast<MultiFab&>(in);
              fill_ghosts(mutable_input, geometry.domain, boundary, lane);
              for (int local = 0; local < out.local_size(); ++local) {
                Array4 output = out.fab(local).array();
                const ConstArray4 input = in.fab(local).const_array();
                for_each_cell(out.box(local), RadiusThreeKernel{output, input});
              }
            },
            [] { return std::size_t{0}; }};
      })};
}

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

Real max_abs_diff_component(const MultiFab& a, const MultiFab& b, int component) {
  Real d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 pa = a.fab(li).const_array();
    const ConstArray4 pb = b.fab(li).const_array();
    const Box2D bx = a.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        d = std::fmax(d, std::fabs(pa(i, j, component) - pb(i, j, component)));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(d)));
}

constexpr Real kRelTol = 1e-12;
constexpr Real kRecoverTol = 1e-8;

// Fixture : construit UNE fois la grille, les deux fournisseurs matrice-libre (SPD et
// non-symetrique)
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
    // apply the shared discrete 5-point Laplacian into workspace-private scratch, then combine.
    // `in` needs >= 1 ghost for the stencil; each prepared session allocates once before iteration.
    const Geometry& geom = *geom_;
    PreparedTestOperator spd = prepared_helmholtz_operator(MultiFab(*ba_, *dm_, 1, 1), geom, *bc_);
    A_ = new PreparedAffineOperatorProvider(spd.provider);

    // Manufactured solution phi_exact and the discretization-exact rhs = A(phi_exact).
    phi_exact_mf_ = new MultiFab(*ba_, *dm_, 1, 1);  // >= 1 ghost (input of A)
    for (int li = 0; li < phi_exact_mf_->local_size(); ++li) {
      Array4 af = phi_exact_mf_->fab(li).array();
      for_each_cell(phi_exact_mf_->box(li), SampleExactKernel{af, geom});
    }
    rhs_ = new MultiFab(*ba_, *dm_, 1, 0);
    spd.apply_once(*rhs_, *phi_exact_mf_);

    // NON-symmetric operator A_ns(in) = in - alpha*Lap(in) + beta * upwind dx(in): the Helmholtz
    // part is SPD, the one-sided advection term breaks symmetry. beta is large enough that the
    // operator is strongly non-self-adjoint (CG stagnates), but the spectrum stays in the right
    // half-plane so GMRES converges. `in`'s periodic ghosts feed the upwind in(i-1).
    PreparedTestOperator nonsymmetric =
        prepared_helmholtz_operator(MultiFab(*ba_, *dm_, 1, 1), geom, *bc_, true);
    A_ns_ = new PreparedAffineOperatorProvider(nonsymmetric.provider);
    rhs_ns_ = new MultiFab(*ba_, *dm_, 1, 0);
    nonsymmetric.apply_once(*rhs_ns_, *phi_exact_mf_);
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
  static PreparedAffineOperatorProvider* A_;
  static PreparedAffineOperatorProvider* A_ns_;
  static MultiFab* phi_exact_mf_;
  static MultiFab* rhs_;
  static MultiFab* rhs_ns_;
};

Box2D* GenericKrylov::dom_ = nullptr;
Geometry* GenericKrylov::geom_ = nullptr;
BoxArray* GenericKrylov::ba_ = nullptr;
DistributionMapping* GenericKrylov::dm_ = nullptr;
BCRec* GenericKrylov::bc_ = nullptr;
PreparedAffineOperatorProvider* GenericKrylov::A_ = nullptr;
PreparedAffineOperatorProvider* GenericKrylov::A_ns_ = nullptr;
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

enum class TestKrylovFamily { kCg, kBicgstab, kGmres, kRichardson };

PreparedKrylovMethod test_krylov_method(TestKrylovFamily family, int restart = 30,
                                        Real relaxation = Real(1)) {
  switch (family) {
    case TestKrylovFamily::kCg:
      return cg_krylov_method();
    case TestKrylovFamily::kBicgstab:
      return bicgstab_krylov_method();
    case TestKrylovFamily::kGmres:
      return gmres_krylov_method(restart);
    case TestKrylovFamily::kRichardson:
      return richardson_krylov_method(relaxation);
  }
  throw std::logic_error("unknown test Krylov family");
}

SolveReport run_prepared_with_preconditioner(
    PreparedAffineOperatorProvider operator_provider, MultiFab& iterate, const MultiFab& rhs,
    PreparedKrylovMethod method, LinearOperatorProperties properties,
    PreparedLinearPreconditioner preconditioner, Real rel_tol, Real abs_tol, int max_iterations,
    PreparedNullspacePolicy nullspace_policy = PreparedNullspacePolicy::nonsingular()) {
  KrylovFootprint footprint{iterate.ncomp(), iterate.n_grow(), !preconditioner.is_identity()};
  OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
  PreparedAffineLinearProblem problem(
      iterate, std::move(operator_provider), std::move(preconditioner), properties, footprint,
      std::move(nullspace_policy), [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  return solve_prepared_affine(problem, workspace, iterate, rhs,
                               KrylovControls{method, rel_tol, abs_tol, max_iterations});
}

// Test-only convenience for callbacks whose complete capture set is immutable/thread-safe. Any
// operator with scratch or mutable state must pass an explicit session provider to the overload
// above instead of using this trust boundary.
SolveReport run_prepared_with_preconditioner(
    const ApplyFn& reentrant_apply, MultiFab& iterate, const MultiFab& rhs,
    PreparedKrylovMethod method, LinearOperatorProperties properties,
    PreparedLinearPreconditioner preconditioner, Real rel_tol, Real abs_tol, int max_iterations,
    PreparedNullspacePolicy nullspace_policy = PreparedNullspacePolicy::nonsingular()) {
  return run_prepared_with_preconditioner(
      reentrant_test_operator(reentrant_apply), iterate, rhs, std::move(method), properties,
      std::move(preconditioner), rel_tol, abs_tol, max_iterations, std::move(nullspace_policy));
}

SolveReport run_prepared(PreparedAffineOperatorProvider operator_provider, MultiFab& iterate,
                         const MultiFab& rhs, PreparedKrylovMethod method,
                         LinearOperatorProperties properties, Real rel_tol = kRelTol,
                         Real abs_tol = Real(0), int max_iterations = 500) {
  return run_prepared_with_preconditioner(
      std::move(operator_provider), iterate, rhs, std::move(method), properties,
      PreparedLinearPreconditioner::identity(), rel_tol, abs_tol, max_iterations);
}

SolveReport run_prepared(const ApplyFn& reentrant_apply, MultiFab& iterate, const MultiFab& rhs,
                         PreparedKrylovMethod method, LinearOperatorProperties properties,
                         Real rel_tol = kRelTol, Real abs_tol = Real(0), int max_iterations = 500) {
  return run_prepared(reentrant_test_operator(reentrant_apply), iterate, rhs, std::move(method),
                      properties, rel_tol, abs_tol, max_iterations);
}

PreparedLinearPreconditioner authenticated_test_preconditioner(
    const MultiFab& prototype, std::string_view implementation, ApplyFn apply,
    std::string exact_parameters = {}, PreparedResourceFn prepare = {},
    PreparedVectorDistribution distribution = PreparedVectorDistribution::Distributed) {
  return PreparedLinearPreconditioner(
      prototype,
      PreparedLinearPreconditionerProvider::trusted_extension(
          {implementation, 1}, std::move(exact_parameters),
          [prepare = std::move(prepare), apply = std::move(apply)](const ExecutionLane&) {
            return PreparedLinearPreconditionerSessionCallbacks{prepare, apply,
                                                                [] { return std::size_t{0}; }};
          }),
      distribution);
}

PreparedLinearPreconditioner affine_scaled_identity_preconditioner(const MultiFab& prototype,
                                                                   Real scale, Real offset) {
  auto constant = std::make_shared<MultiFab>(prototype.box_array(), prototype.dmap(),
                                             prototype.ncomp(), prototype.n_grow());
  constant->set_val(offset);
  return authenticated_test_preconditioner(
      prototype, "pops.test.generic-krylov.affine-scaled-identity",
      [scale, constant](MultiFab& out, const MultiFab& in) {
        PureFieldAlgebra::lincomb(out, scale, in, Real(0), in);
        PureFieldAlgebra::axpy(out, Real(1), *constant);
      },
      exact_provider_parameters(scale, offset));
}

PreparedLinearPreconditioner boundary_affine_preconditioner(const MultiFab& prototype,
                                                            const Geometry& geometry,
                                                            BCType boundary_type) {
  const BoxArray boxes = prototype.box_array();
  const DistributionMapping mapping = prototype.dmap();
  const int components = prototype.ncomp();
  BCRec boundary;
  boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = boundary_type;
  boundary.xlo_val = Real(1.5);
  boundary.xhi_val = Real(-0.75);
  boundary.ylo_val = Real(0.5);
  boundary.yhi_val = Real(-1.25);
  boundary.dx = geometry.dx();
  boundary.dy = geometry.dy();
  if (boundary_type == BCType::Robin) {
    boundary.xlo_alpha = boundary.xhi_alpha = boundary.ylo_alpha = boundary.yhi_alpha = Real(1);
    boundary.xlo_beta = boundary.xhi_beta = boundary.ylo_beta = boundary.yhi_beta = Real(0.25);
  }
  const Real strength = Real(0.05) * geometry.dx() * geometry.dx();
  PreparedLinearPreconditionerProvider provider =
      PreparedLinearPreconditionerProvider::trusted_extension(
          {"pops.test.generic-krylov.boundary-affine", 1},
          exact_provider_parameters(boundary_type, geometry.xlo, geometry.xhi, geometry.ylo,
                                    geometry.yhi, boundary.xlo_val, boundary.xhi_val,
                                    boundary.ylo_val, boundary.yhi_val, boundary.xlo_alpha,
                                    boundary.xlo_beta, boundary.dx, boundary.dy, strength),
          [boxes, mapping, components, geometry, boundary, strength](const ExecutionLane& lane) {
            auto laplacian = std::make_shared<MultiFab>(boxes, mapping, components, 0);
            return PreparedLinearPreconditionerSessionCallbacks{
                {},
                [laplacian, geometry, boundary, strength, &lane](MultiFab& out,
                                                                 const MultiFab& in) {
                  MultiFab& mutable_input = const_cast<MultiFab&>(in);
                  fill_ghosts(mutable_input, geometry.domain, boundary, lane);
                  apply_laplacian(mutable_input, geometry, *laplacian);
                  PureFieldAlgebra::lincomb(out, Real(1), in, -strength, *laplacian);
                },
                [] { return std::size_t{1}; }};
          });
  return PreparedLinearPreconditioner(prototype, std::move(provider));
}

LinearOperatorProperties periodic_nullspace_properties(TestKrylovFamily family) {
  return family == TestKrylovFamily::kCg
             ? LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement()
             : LinearOperatorProperties::general();
}

PreparedNullspacePolicy periodic_mean_zero_policy(const Geometry& geometry) {
  return PreparedNullspacePolicy::preserving(constant_mean_zero_nullspace(
      "test://generic-krylov/periodic-constant-nullspace@1",
      "test periodic -Laplacian constant mode", geometry.dx() * geometry.dy()));
}

struct PreparedMgTestContext {
  GridContext grid;
  mutable int kernel_count = 0;

  bool is_polar_geometry() const { return false; }
  GridContext grid_context() const { return grid; }
  void count_kernel() const { ++kernel_count; }
};

struct TensorMmsRhsKernel {
  Array4 values;
  Geometry geometry;
  Real cross_sum;
  POPS_HD void operator()(int i, int j) const {
    const Real x = geometry.x_cell(i);
    const Real y = geometry.y_cell(j);
    const Real sine = std::sin(kPi * x) * std::sin(kPi * y);
    const Real cosine = std::cos(kPi * x) * std::cos(kPi * y);
    values(i, j) = -Real(2) * kPi * kPi * sine + cross_sum * kPi * kPi * cosine;
  }
};

struct ShiftedDirichletExactKernel {
  Array4 values;
  Geometry geometry;
  Real boundary_value;
  POPS_HD void operator()(int i, int j) const {
    values(i, j) =
        boundary_value + std::sin(kPi * geometry.x_cell(i)) * std::sin(kPi * geometry.y_cell(j));
  }
};

void fill_tensor_mms_rhs(MultiFab& rhs, const Geometry& geometry, Real a_xy, Real a_yx) {
  for (int local = 0; local < rhs.local_size(); ++local) {
    Array4 values = rhs.fab(local).array();
    for_each_cell(rhs.box(local), TensorMmsRhsKernel{values, geometry, a_xy + a_yx});
  }
}

PreparedTestOperator prepared_tensor_operator(std::shared_ptr<GeometricMG> op) {
  const std::string exact =
      exact_provider_parameters(op->geom().xlo, op->geom().xhi, op->geom().ylo, op->geom().yhi,
                                op->bc().xlo, op->bc().xhi, op->bc().ylo, op->bc().yhi);
  return {PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.generic-krylov.tensor-operator", 1}, exact,
      [op = std::move(op)](const ExecutionLane& lane) {
        return PreparedAffineOperatorSessionCallbacks{
            {},
            [op, &lane](MultiFab& out, const MultiFab& in) {
              MultiFab& mutable_input = const_cast<MultiFab&>(in);
              device_fence();
              fill_ghosts(mutable_input, op->geom().domain, op->bc(), lane);
              apply_laplacian(mutable_input, op->geom(), out, op->op_coef(), op->op_eps(),
                              op->op_kappa(), op->op_eps_y(), op->op_a_xy(), op->op_a_yx());
            },
            [] { return std::size_t{0}; }};
      })};
}

PreparedLinearPreconditioner prepared_geometric_mg_preconditioner(const MultiFab& prototype,
                                                                  const Geometry& geometry,
                                                                  const BCRec& boundary) {
  using pops::runtime::program::GeometricMgPreconditioner;
  GridContext grid;
  grid.dom = geometry.domain;
  grid.geom = geometry;
  grid.bc = boundary;
  const MultiFab* prepared_layout = &prototype;
  PreparedLinearPreconditionerProvider provider =
      PreparedLinearPreconditionerProvider::trusted_extension(
          {"pops.test.generic-krylov.geometric-mg", 1},
          exact_provider_parameters(geometry.xlo, geometry.xhi, geometry.ylo, geometry.yhi,
                                    boundary.xlo, boundary.xhi, boundary.ylo, boundary.yhi,
                                    boundary.xlo_val, boundary.xhi_val, boundary.ylo_val,
                                    boundary.yhi_val, boundary.dx, boundary.dy),
          [grid, prepared_layout](const ExecutionLane& lane) {
            auto context = std::make_shared<PreparedMgTestContext>();
            context->grid = grid;
            auto implementation = std::make_shared<GeometricMgPreconditioner>();
            return PreparedLinearPreconditionerSessionCallbacks{
                [context, implementation, prepared_layout, &lane] {
                  implementation->prepare(*context, *prepared_layout, lane);
                },
                [context, implementation, &lane](MultiFab& out, const MultiFab& in) {
                  implementation->apply(*context, out, in, lane);
                },
                [implementation] { return implementation->persistent_field_count(); }};
          });
  return PreparedLinearPreconditioner(prototype, std::move(provider));
}

struct PreparedTensorCaseReport {
  SolveReport krylov;
  Real mg_initial = Real(0);
  Real mg_final = Real(0);
  int mg_cycles = 0;
};

void configure_tensor_cross_terms(GeometricMG& op, Real strength, bool nonsymmetric) {
  if (nonsymmetric) {
    // A pure-skew constant tensor is invisible to a scalar div(A grad) operator: the two mixed
    // derivatives cancel.  Vary the skew coefficient in x instead, which leaves the symmetric
    // elliptic part equal to the identity while producing the genuine first-order term
    // strength*d_y(phi).  The resulting discrete operator is non-self-adjoint without sacrificing
    // ellipticity.
    op.set_cross_terms(
        ScalarFieldProvider2D::trusted_extension({"pops.test.generic-krylov.variable-skew-xy", 1},
                                                 exact_provider_parameters(strength),
                                                 [strength](Real x, Real) { return strength * x; }),
        ScalarFieldProvider2D::trusted_extension(
            {"pops.test.generic-krylov.variable-skew-yx", 1}, exact_provider_parameters(strength),
            [strength](Real x, Real) { return -strength * x; }));
  } else {
    op.set_cross_terms(
        ScalarFieldProvider2D::trusted_extension({"pops.test.generic-krylov.symmetric-cross-xy", 1},
                                                 exact_provider_parameters(strength),
                                                 [strength](Real, Real) { return strength; }),
        ScalarFieldProvider2D::trusted_extension({"pops.test.generic-krylov.symmetric-cross-yx", 1},
                                                 exact_provider_parameters(strength),
                                                 [strength](Real, Real) { return strength; }));
  }
}

PreparedTensorCaseReport solve_prepared_tensor_case(int cells, Real cross, bool nonsymmetric) {
  const Box2D domain = Box2D::from_extents(cells, cells);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BoxArray boxes = BoxArray::from_domain(domain, cells / 2);
  const DistributionMapping distribution(boxes.size(), n_ranks());
  BCRec boundary;
  boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = BCType::Dirichlet;

  auto op = std::make_shared<GeometricMG>(geometry, boxes, distribution, boundary);
  op->set_epsilon_anisotropic(constant_scalar_field_provider(Real(1)),
                              constant_scalar_field_provider(Real(1)));
  configure_tensor_cross_terms(*op, cross, nonsymmetric);
  const PreparedTestOperator apply = prepared_tensor_operator(op);

  MultiFab rhs(boxes, distribution, 1, 0);
  MultiFab iterate(boxes, distribution, 1, 1);
  MultiFab exact(boxes, distribution, 1, 1);
  for (int local = 0; local < exact.local_size(); ++local) {
    Array4 values = exact.fab(local).array();
    for_each_cell(exact.box(local), ShiftedDirichletExactKernel{values, geometry, Real(0)});
  }
  // Manufacture the exact discrete forcing.  This keeps the regression about the solver and the
  // full-tensor stencil (including variable skew coefficients), not analytic/discrete truncation.
  apply.apply_once(rhs, exact);
  PureFieldAlgebra::zero_valid(iterate);
  const SolveReport krylov = run_prepared_with_preconditioner(
      apply.provider, iterate, rhs, bicgstab_krylov_method(), LinearOperatorProperties::general(),
      prepared_geometric_mg_preconditioner(iterate, geometry, boundary), Real(1e-10), Real(0), 300);

  GeometricMG mg(geometry, boxes, distribution, boundary);
  mg.set_epsilon_anisotropic(constant_scalar_field_provider(Real(1)),
                             constant_scalar_field_provider(Real(1)));
  configure_tensor_cross_terms(mg, cross, nonsymmetric);
  PureFieldAlgebra::copy(mg.rhs(), rhs);
  PureFieldAlgebra::zero_valid(mg.phi());
  const Real initial = mg.current_residual();
  Real residual = initial;
  int cycles = 0;
  for (; cycles < 60 && residual > Real(1e-10) * initial; ++cycles) {
    mg.vcycle();
    residual = mg.current_residual();
  }
  return {krylov, initial, residual, cycles};
}

struct AffineBoundarySolveReport {
  SolveReport probe;
  SolveReport converged;
  SolveReport warm;
  Real true_relative_residual = Real(0);
  Real effective_rhs_error = Real(0);
  Real affine_offset_max = Real(0);
};

struct AffineBoundaryCaseReport {
  AffineBoundarySolveReport homogeneous;
  AffineBoundarySolveReport offset;
  Real solution_difference = Real(0);
};

AffineBoundaryCaseReport solve_affine_boundary_case(BCType type) {
  constexpr int cells = 24;
  constexpr Real boundary_value = Real(1e3);
  constexpr Real solve_tolerance = Real(1e-9);
  const Box2D domain = Box2D::from_extents(cells, cells);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BoxArray boxes = BoxArray::from_domain(domain, cells / 2);
  const DistributionMapping distribution(boxes.size(), n_ranks());
  MultiFab homogeneous_iterate(boxes, distribution, 1, 1);
  MultiFab offset_iterate(boxes, distribution, 1, 1);

  const auto run = [&](Real value, MultiFab& iterate) {
    BCRec boundary;
    boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = type;
    boundary.xlo_val = boundary.xhi_val = boundary.ylo_val = boundary.yhi_val = value;
    boundary.dx = geometry.dx();
    boundary.dy = geometry.dy();
    if (type == BCType::Robin) {
      boundary.xlo_alpha = boundary.xhi_alpha = boundary.ylo_alpha = boundary.yhi_alpha = Real(1);
      boundary.xlo_beta = boundary.xhi_beta = boundary.ylo_beta = boundary.yhi_beta = Real(0.25);
    }

    auto op = std::make_shared<GeometricMG>(geometry, boxes, distribution, boundary);
    op->set_cross_terms(constant_scalar_field_provider(Real(0)),
                        constant_scalar_field_provider(Real(0)));
    const PreparedTestOperator apply = prepared_tensor_operator(op);
    MultiFab zero(boxes, distribution, 1, 1);
    MultiFab offset(boxes, distribution, 1, 0);
    MultiFab forcing(boxes, distribution, 1, 0);
    MultiFab rhs(boxes, distribution, 1, 0);
    MultiFab recovered_forcing(boxes, distribution, 1, 0);
    MultiFab applied(boxes, distribution, 1, 0);
    MultiFab true_residual(boxes, distribution, 1, 0);
    PureFieldAlgebra::zero_valid(zero);
    apply.apply_once(offset, zero);
    fill_tensor_mms_rhs(forcing, geometry, Real(0), Real(0));
    PureFieldAlgebra::lincomb(rhs, Real(1), offset, Real(1), forcing);
    PureFieldAlgebra::lincomb(recovered_forcing, Real(1), rhs, Real(-1), offset);

    const auto solve = [&](Real rel_tol, int max_iterations) {
      return run_prepared_with_preconditioner(
          apply.provider, iterate, rhs, bicgstab_krylov_method(),
          LinearOperatorProperties::general(),
          prepared_geometric_mg_preconditioner(iterate, geometry, boundary), rel_tol, Real(0),
          max_iterations);
    };
    PureFieldAlgebra::zero_valid(iterate);
    const SolveReport probe = solve(Real(0.5), 1);
    PureFieldAlgebra::zero_valid(iterate);
    const SolveReport converged = solve(solve_tolerance, 300);
    apply.apply_once(applied, iterate);
    PureFieldAlgebra::lincomb(true_residual, Real(1), rhs, Real(-1), applied);
    const Real true_relative_residual =
        PureFieldAlgebra::norm(true_residual) / PureFieldAlgebra::norm(forcing);
    const SolveReport warm = solve(solve_tolerance, 300);
    return AffineBoundarySolveReport{probe,
                                     converged,
                                     warm,
                                     true_relative_residual,
                                     max_abs_diff(recovered_forcing, forcing),
                                     PureFieldAlgebra::max_abs(offset)};
  };

  const AffineBoundarySolveReport homogeneous = run(Real(0), homogeneous_iterate);
  const AffineBoundarySolveReport offset = run(boundary_value, offset_iterate);
  return {homogeneous, offset, max_abs_diff(homogeneous_iterate, offset_iterate)};
}

class OneStepExternalKrylovProvider final : public PreparedKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override { return "pops.test.krylov.one-step"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.test.krylov.one-step@1";
  }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_common_krylov_controls(controls);
        !common.accepted())
      return common;
    const double* step =
        detail::exact_real_option(options, "pops.test.krylov.one-step.options@1", "physical_step");
    if (step == nullptr || !std::isfinite(*step) || *step <= 0.0)
      return KrylovMethodValidation::reject(1, "one-step physical_step is invalid");
    return KrylovMethodValidation::accept();
  }
  KrylovMethodValidation validate_problem(const KrylovMethodProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    if (facts.has_preconditioner)
      return KrylovMethodValidation::reject(2, "one-step provider has no preconditioner slot");
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request, const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("one-step workspace requires an unpreconditioned flat solve");
    return {.field_count = 2, .real_count = 1, .state_word_count = 1, .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions& options) const override {
    Real& physical_step = context.real_value(0);
    physical_step = static_cast<Real>(*detail::exact_real_option(
        options, "pops.test.krylov.one-step.options@1", "physical_step"));
    ++context.state_word(0);
    context.add_physical_direction(context.iterate(), physical_step, context.initial_residual());
    const Real residual = context.true_residual_norm(context.initial_residual());
    return context.report(residual, 1,
                          residual <= context.physical_threshold() ? SolveStatus::kSolved
                                                                   : SolveStatus::kIterationLimit);
  }
};

class ReportOnlyExternalKrylovProvider final : public PreparedKrylovMethodProvider {
 public:
  explicit ReportOnlyExternalKrylovProvider(SolveReport report) : report_(std::move(report)) {}

  std::string_view identity() const noexcept override { return "pops.test.krylov.report-only"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.test.krylov.report-only@1";
  }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (!detail::empty_options(options, "pops.test.krylov.report-only.options@1"))
      return KrylovMethodValidation::reject(1, "report-only options contract is invalid");
    return detail::validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(const KrylovMethodProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    if (facts.has_preconditioner)
      return KrylovMethodValidation::reject(2, "report-only provider has no preconditioner slot");
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request, const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("report-only workspace requires no preconditioner");
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext&, const PreparedProviderOptions&) const override {
    return report_;
  }

 private:
  SolveReport report_;
};

/// Third distribution strategy proving that Krylov routes provider callables rather than matching
/// either builtin preset. It realizes a disjoint storage topology, but constructs its layout
/// contract and reductions independently and deliberately requires non-zero persistent scratch.
struct CustomOwnedVectorDistributionProbe {
  std::size_t layout_matches_calls = 0;
  std::size_t layout_contract_calls = 0;
  std::size_t sum_calls = 0;
  std::size_t max_calls = 0;
  std::size_t exact_value_preflight_calls = 0;
};

struct CustomOwnedVectorDistributionSource {
  std::shared_ptr<CustomOwnedVectorDistributionProbe> probe;

  static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.vector-distribution.custom-owned", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::uint32_t{17});
  }
  bool layout_matches(const MultiFab& field) const {
    ++probe->layout_matches_calls;
    const auto& boxes = field.box_array().boxes();
    const auto& owners = field.dmap().ranks();
    if (field.ncomp() <= 0 || field.n_grow() < 0 || boxes.empty() || boxes.size() != owners.size())
      return false;
    return std::all_of(owners.begin(), owners.end(),
                       [](int owner) { return owner >= 0 && owner < n_ranks(); });
  }
  std::string layout_contract(const MultiFab& field) const {
    ++probe->layout_contract_calls;
    ExactContractBuilder contract;
    contract.text("pops.test.custom-owned-layout")
        .scalar(static_cast<std::int32_t>(field.ncomp()))
        .scalar(static_cast<std::int32_t>(field.n_grow()))
        .scalar(static_cast<std::uint64_t>(field.box_array().boxes().size()));
    const auto& owners = field.dmap().ranks();
    for (std::size_t index = 0; index < field.box_array().boxes().size(); ++index) {
      const Box2D& box = field.box_array().boxes()[index];
      contract.scalar(static_cast<std::int32_t>(box.lo[0]))
          .scalar(static_cast<std::int32_t>(box.lo[1]))
          .scalar(static_cast<std::int32_t>(box.hi[0]))
          .scalar(static_cast<std::int32_t>(box.hi[1]))
          .scalar(static_cast<std::int32_t>(owners[index]));
    }
    return std::move(contract).release();
  }
  std::size_t reduction_scratch_value_count(std::size_t value_count) const noexcept {
    return value_count + 3u;
  }
  std::size_t validation_scratch_byte_count() const noexcept { return 32u; }
  PreparedVectorDistributionStatus reduce_sum_values(std::span<double> values,
                                                     std::span<double> scratch, const char*,
                                                     const ExecutionLane& lane) const noexcept {
    ++probe->sum_calls;
    if (scratch.size() < reduction_scratch_value_count(values.size()))
      return PreparedVectorDistributionStatus::failure(
          1, "custom distribution sum scratch was not prepared");
    std::copy(values.begin(), values.end(), scratch.begin());
    try {
  all_reduce_sum_inplace(scratch.data(), values.size(), lane);
    } catch (...) {
      return PreparedVectorDistributionStatus::failure(2,
                                                       "custom distribution sum collective failed");
    }
    std::copy_n(scratch.begin(), values.size(), values.begin());
    return PreparedVectorDistributionStatus::success();
  }
  PreparedVectorDistributionStatus reduce_max_values(std::span<double> values,
                                                     std::span<double> scratch, const char*,
                                                     const ExecutionLane& lane) const noexcept {
    ++probe->max_calls;
    if (scratch.size() < reduction_scratch_value_count(values.size()))
      return PreparedVectorDistributionStatus::failure(
          3, "custom distribution max scratch was not prepared");
    std::copy(values.begin(), values.end(), scratch.begin());
    try {
  all_reduce_max_inplace(scratch.data(), values.size(), lane);
    } catch (...) {
      return PreparedVectorDistributionStatus::failure(4,
                                                       "custom distribution max collective failed");
    }
    std::copy_n(scratch.begin(), values.size(), values.begin());
    return PreparedVectorDistributionStatus::success();
  }
  PreparedVectorDistributionStatus require_exact_values(const MultiFab& field,
                                                        std::span<char> scratch, const char*,
                                                        const ExecutionLane&) const noexcept {
    ++probe->exact_value_preflight_calls;
    bool matches = false;
    try {
      matches = layout_matches(field);
    } catch (...) {
      return PreparedVectorDistributionStatus::failure(
          5, "custom distribution exact-value layout callback failed");
    }
    if (scratch.size() != validation_scratch_byte_count() || !matches)
      return PreparedVectorDistributionStatus::failure(
          6, "custom distribution exact-value scratch/layout was not prepared");
    scratch.front() = static_cast<char>(0x5a);
    return PreparedVectorDistributionStatus::success();
  }
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

TEST(test_krylov_controls, signed_int_limits_are_explicit_and_overflow_free) {
  const int int_max = std::numeric_limits<int>::max();
  const int gmres_restart_max = KrylovWorkspace::max_batched_basis_extent();
  EXPECT_NO_THROW(
      detail::validate_controls(KrylovControls{cg_krylov_method(), Real(1e-8), Real(0), int_max}));
  EXPECT_NO_THROW(detail::validate_controls(
      KrylovControls{gmres_krylov_method(gmres_restart_max), Real(1e-8), Real(0), int_max}));
  EXPECT_THROW(detail::validate_controls(KrylovControls{gmres_krylov_method(gmres_restart_max + 1),
                                                        Real(1e-8), Real(0), int_max}),
               std::invalid_argument);
  EXPECT_THROW(detail::validate_controls(KrylovControls{cg_krylov_method(), Real(1), Real(0), 1}),
               std::invalid_argument);
}

TEST(test_krylov_method_provider, external_method_uses_registry_and_generic_workspace_pools) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {3, 3}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab iterate(boxes, mapping, 1, 0);
  MultiFab rhs(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(3.25));
  const ApplyFn identity = [](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
  };

  PreparedKrylovMethodRegistry registry;
  registry.add(std::make_shared<OneStepExternalKrylovProvider>());
  const PreparedKrylovMethod method = registry.resolve(
      "pops.test.krylov.one-step",
      PreparedProviderOptions{"pops.test.krylov.one-step.options@1", {{"physical_step", 1.0}}});
  const KrylovFootprint footprint{1, 0, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
  PreparedAffineLinearProblem problem(
      iterate, reentrant_test_operator(identity), PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
      [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  const SolveReport report = solve_prepared_affine(problem, workspace, iterate, rhs,
                                                   KrylovControls{method, Real(1e-12), Real(0), 1});
  EXPECT_TRUE(report.solved());
  EXPECT_EQ(report.iters, 1);
  EXPECT_EQ(workspace.allocation_count(), 2u);
  EXPECT_EQ(workspace.scalar_value_count(), 1u);
  EXPECT_EQ(max_abs_diff(iterate, rhs), Real(0));
}

TEST(test_krylov_method_provider,
     malformed_external_report_enums_are_rejected_by_the_common_publication_boundary) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {3, 3}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab iterate(boxes, mapping, 1, 0);
  MultiFab rhs(boxes, mapping, 1, 0);
  rhs.set_val(Real(3.25));
  const ApplyFn identity = [](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
  };

  const auto publish = [&](SolveReport provider_report) {
    iterate.set_val(Real(0));
    PreparedKrylovMethodRegistry registry;
    registry.add(std::make_shared<ReportOnlyExternalKrylovProvider>(std::move(provider_report)));
    const PreparedKrylovMethod method =
        registry.resolve("pops.test.krylov.report-only",
                         PreparedProviderOptions{"pops.test.krylov.report-only.options@1", {}});
    return run_prepared(identity, iterate, rhs, method, LinearOperatorProperties::general(),
                        Real(1e-12), Real(0), 1);
  };

  SolveReport invalid_status;
  invalid_status.status = static_cast<SolveStatus>(999);
  invalid_status.action = SolveAction::kFailRun;
  invalid_status.reason = "unknown external provider status";
  const SolveReport rejected_status = publish(invalid_status);
  EXPECT_EQ(rejected_status.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(rejected_status.action, SolveAction::kFailRun);
  EXPECT_EQ(rejected_status.reason, "prepared Krylov provider published a malformed SolveReport");

  SolveReport invalid_action;
  invalid_action.status = SolveStatus::kBreakdown;
  invalid_action.action = static_cast<SolveAction>(999);
  invalid_action.reason = "unknown external provider action";
  const SolveReport rejected_action = publish(invalid_action);
  EXPECT_EQ(rejected_action.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(rejected_action.action, SolveAction::kFailRun);
  EXPECT_EQ(rejected_action.reason, "prepared Krylov provider published a malformed SolveReport");
}

TEST(test_vector_distribution_provider,
     third_strategy_routes_layout_reduction_preflight_and_workspace_without_core_branch) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {3, 3}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab iterate(boxes, mapping, 1, 0);
  MultiFab rhs(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(2.5));
  const auto probe = std::make_shared<CustomOwnedVectorDistributionProbe>();
  const PreparedVectorDistribution distribution(CustomOwnedVectorDistributionSource{probe});
  EXPECT_EQ(distribution.provider_identity().name,
            std::string_view("pops.test.vector-distribution.custom-owned"));
  EXPECT_NE(distribution, PreparedVectorDistribution::Distributed);

  const ApplyFn identity = [](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
  };
  const PreparedKrylovMethod method = bicgstab_krylov_method();
  const KrylovFootprint footprint{1, 0, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
  PreparedAffineLinearProblem problem(
      iterate, reentrant_test_operator(identity), PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
      [&snapshot]() { return snapshot; }, {}, distribution);
  KrylovWorkspace workspace(iterate, method, footprint, distribution);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const SolveReport report = solve_prepared_affine(problem, workspace, iterate, rhs,
                                                   KrylovControls{method, Real(1e-12), Real(0), 4});
  EXPECT_TRUE(report.solved());
  const Real roundoff = Real(8) * std::numeric_limits<Real>::epsilon() * Real(2.5);
  EXPECT_LE(max_abs_diff(iterate, rhs), roundoff);
  EXPECT_NEAR(PureFieldAlgebra::max_abs(iterate, distribution), Real(2.5), roundoff);
  EXPECT_GE(workspace.collective_value_count(), 4u);
  EXPECT_EQ(workspace.distribution_validation_byte_count(), 32u);
  EXPECT_GT(probe->layout_matches_calls, 0u);
  EXPECT_GT(probe->layout_contract_calls, 0u);
  EXPECT_GT(probe->sum_calls, 0u);
  EXPECT_GT(probe->max_calls, 0u);
  EXPECT_GT(probe->exact_value_preflight_calls, 0u);
}

TEST(test_prepared_layout_fingerprint, OwnershipIsDomainSeparatedEvenOnOneRank) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}});
  MultiFab field(boxes, DistributionMapping(std::vector<int>{my_rank()}), 1, 0);
  EXPECT_NE(detail::layout_fingerprint(field, PreparedVectorDistribution::Distributed),
            detail::layout_fingerprint(field, PreparedVectorDistribution::Replicated));
  EXPECT_THROW((void)PreparedVectorDistribution(static_cast<FieldDistribution>(255)),
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
  const KrylovFootprint claims_preconditioner{1, 1, true};
  const KrylovFootprint hides_preconditioner{1, 1, false};

  EXPECT_THROW((void)PreparedAffineLinearProblem(
                   one_ghost, reentrant_test_operator(identity_apply),
                   PreparedLinearPreconditioner::identity(), LinearOperatorProperties::general(),
                   claims_preconditioner, PreparedNullspacePolicy::nonsingular(), probe),
               std::invalid_argument);
  EXPECT_THROW((void)PreparedAffineLinearProblem(
                   one_ghost, reentrant_test_operator(identity_apply),
                   authenticated_test_preconditioner(one_ghost, "pops.test.generic-krylov.identity",
                                                     identity_apply),
                   LinearOperatorProperties::general(), hides_preconditioner,
                   PreparedNullspacePolicy::nonsingular(), probe),
               std::invalid_argument);
  EXPECT_THROW((void)KrylovWorkspace(one_ghost, cg_krylov_method(), claims_preconditioner),
               std::invalid_argument);
  EXPECT_THROW((void)KrylovWorkspace(one_ghost, richardson_krylov_method(), claims_preconditioner),
               std::invalid_argument);
  const KrylovFootprint gmres_footprint{1, 1, false};
  EXPECT_THROW(
      (void)KrylovWorkspace(one_ghost, gmres_krylov_method(std::numeric_limits<int>::max()),
                            gmres_footprint),
      std::invalid_argument);
  EXPECT_THROW((void)KrylovWorkspace::required_fields(
                   gmres_krylov_method(std::numeric_limits<int>::max()),
                   KrylovWorkspaceRequest{gmres_footprint, PreparedVectorDistribution::Distributed,
                                          detail::PreparedFieldAlgebra::kRobustDotPayloadWidth}),
               std::invalid_argument);

  MultiFab two_components(*ba_, *dm_, 2, 1);
  EXPECT_THROW((void)PreparedAffineLinearProblem(
                   one_ghost, reentrant_test_operator(identity_apply),
                   authenticated_test_preconditioner(
                       two_components, "pops.test.generic-krylov.identity", identity_apply),
                   LinearOperatorProperties::general(), claims_preconditioner,
                   PreparedNullspacePolicy::nonsingular(), probe),
               std::invalid_argument);

  EXPECT_THROW((void)PreparedAffineLinearProblem(
                   one_ghost, reentrant_test_operator(identity_apply),
                   authenticated_test_preconditioner(no_ghosts, "pops.test.generic-krylov.identity",
                                                     identity_apply),
                   LinearOperatorProperties::general(), claims_preconditioner,
                   PreparedNullspacePolicy::nonsingular(), probe),
               std::invalid_argument);

  const BoxArray other_boxes =
      n_ranks() > 1 ? BoxArray(std::vector<Box2D>{*dom_}) : BoxArray::from_domain(*dom_, kN / 2);
  const DistributionMapping other_mapping(other_boxes.size(), n_ranks());
  MultiFab other_layout(other_boxes, other_mapping, 1, 1);
  EXPECT_THROW((void)PreparedAffineLinearProblem(
                   one_ghost, reentrant_test_operator(identity_apply),
                   authenticated_test_preconditioner(
                       other_layout, "pops.test.generic-krylov.identity", identity_apply),
                   LinearOperatorProperties::general(), claims_preconditioner,
                   PreparedNullspacePolicy::nonsingular(), probe),
               std::invalid_argument);

  if (n_ranks() > 1) {
    const DistributionMapping rank_zero_only(
        std::vector<int>(static_cast<std::size_t>(ba_->size()), 0));
    MultiFab empty_on_remote_rank(*ba_, rank_zero_only, 1, 1);
    EXPECT_THROW((void)PreparedAffineLinearProblem(
                     one_ghost, reentrant_test_operator(identity_apply),
                     authenticated_test_preconditioner(
                         empty_on_remote_rank, "pops.test.generic-krylov.identity", identity_apply),
                     LinearOperatorProperties::general(), claims_preconditioner,
                     PreparedNullspacePolicy::nonsingular(), probe),
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

  const BoxArray incompatible_boxes =
      n_ranks() > 1 ? BoxArray(std::vector<Box2D>{*dom_}) : BoxArray::from_domain(*dom_, kN / 2);
  const DistributionMapping incompatible_mapping(incompatible_boxes.size(), n_ranks());
  MultiFab incompatible(incompatible_boxes, incompatible_mapping, 1, 1);
  EXPECT_THROW(preconditioner.apply(context, incompatible, incompatible), std::invalid_argument);

  context.grid.geom.xhi += 1.0;
  EXPECT_NO_THROW(preconditioner.prepare(context, prototype));
  EXPECT_EQ(preconditioner.preparation_generation(), generation + 1);

  // Periodicity is a paired-axis contract. Change both x faces so the mutated boundary remains a
  // physically valid construction request while still proving that boundary identity rebuilds.
  context.grid.bc.xlo = context.grid.bc.xhi = BCType::Dirichlet;
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
  const PreparedTestOperator radius_three = prepared_radius_three_operator(*geom_, *bc_);
  radius_three.apply_once(rhs, exact);

  const SolveReport report = run_prepared(radius_three.provider, x, rhs, cg_krylov_method(),
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

TEST_F(GenericKrylov, every_method_solves_two_components_and_preconditioned_routes) {
  MultiFab exact(*ba_, *dm_, 2, 1);
  for (int local = 0; local < exact.local_size(); ++local)
    for (int component = 0; component < exact.ncomp(); ++component)
      for_each_cell(exact.box(local), SampleDistinctComponentExactKernel{exact.fab(local).array(),
                                                                         *geom_, component});

  const PreparedTestOperator apply = prepared_helmholtz_operator(exact, *geom_, *bc_);
  MultiFab rhs(*ba_, *dm_, 2, 0);
  apply.apply_once(rhs, exact);

  const Real h = geom_->dx();
  const Real richardson_omega = Real(1) / (Real(1) + kAlpha * Real(8) / (h * h));
  struct MethodCase {
    TestKrylovFamily family;
    PreparedKrylovMethod method;
    bool preconditioned;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kCg, cg_krylov_method(), false},
      {TestKrylovFamily::kBicgstab, bicgstab_krylov_method(), true},
      {TestKrylovFamily::kGmres, gmres_krylov_method(8), true},
      {TestKrylovFamily::kRichardson, richardson_krylov_method(richardson_omega), false},
  }};

  for (const MethodCase& method : methods) {
    MultiFab iterate(*ba_, *dm_, 2, 1);
    PureFieldAlgebra::zero_valid(iterate);
    PreparedLinearPreconditioner preconditioner =
        method.preconditioned
            ? affine_scaled_identity_preconditioner(iterate, Real(0.75), Real(2.5))
            : PreparedLinearPreconditioner::identity();
    const LinearOperatorProperties properties =
        method.family == TestKrylovFamily::kCg
            ? LinearOperatorProperties::symmetric_positive_definite()
            : LinearOperatorProperties::general();
    const SolveReport report =
        run_prepared_with_preconditioner(apply.provider, iterate, rhs, method.method, properties,
                                         std::move(preconditioner), kRelTol, Real(0), 200000);

    EXPECT_TRUE(report.solved()) << "method=" << method.method.identity()
                                 << " reason=" << report.reason;
    EXPECT_GT(report.iters, 1) << "method=" << method.method.identity();
    EXPECT_LT(max_abs_diff_component(iterate, exact, 0), kRecoverTol)
        << "component=0 method=" << method.method.identity();
    EXPECT_LT(max_abs_diff_component(iterate, exact, 1), kRecoverTol)
        << "component=1 method=" << method.method.identity();
  }
}

TEST_F(GenericKrylov, mpi_empty_rank_runs_restarted_gmres_and_preconditioned_bicgstab) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "the serial registration has no empty MPI rank";

  const Box2D domain = Box2D::from_extents(kN, kN);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping owner_zero(std::vector<int>{0});
  const BCRec periodic{};
  MultiFab exact(boxes, owner_zero, 1, 1);
  for (int local = 0; local < exact.local_size(); ++local)
    for_each_cell(exact.box(local), SampleExactKernel{exact.fab(local).array(), geometry});
  const PreparedTestOperator apply = prepared_helmholtz_operator(exact, geometry, periodic);
  MultiFab rhs(boxes, owner_zero, 1, 0);
  apply.apply_once(rhs, exact);

  EXPECT_EQ(exact.local_size(), my_rank() == 0 ? 1 : 0);
  ASSERT_EQ(static_cast<int>(all_reduce_sum(static_cast<double>(exact.local_size()))), 1);

  struct MethodCase {
    TestKrylovFamily family;
    PreparedKrylovMethod method;
  };
  constexpr int gmres_restart = 2;
  const std::array<MethodCase, 2> methods{{
      {TestKrylovFamily::kGmres, gmres_krylov_method(gmres_restart)},
      {TestKrylovFamily::kBicgstab, bicgstab_krylov_method()},
  }};
  for (const MethodCase& method : methods) {
    MultiFab iterate(boxes, owner_zero, 1, 1);
    PureFieldAlgebra::zero_valid(iterate);
    const SolveReport report = run_prepared_with_preconditioner(
        apply.provider, iterate, rhs, method.method, LinearOperatorProperties::general(),
        affine_scaled_identity_preconditioner(iterate, Real(0.75), Real(2.5)), kRelTol, Real(0),
        500);

    EXPECT_TRUE(report.solved()) << "method=" << method.method.identity()
                                 << " reason=" << report.reason;
    if (method.family == TestKrylovFamily::kGmres)
      EXPECT_GT(report.iters, gmres_restart)
          << "restart=2 must cross a restart boundary and reuse the batched Arnoldi payload";
    EXPECT_LT(max_abs_diff(iterate, exact), kRecoverTol) << "method=" << method.method.identity();
  }
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
  const SolveReport r = run_prepared(*A_, x, *rhs_, cg_krylov_method(),
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
      run_prepared(*A_, x, *rhs_, bicgstab_krylov_method(), LinearOperatorProperties::general());
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
  auto operator_calls = std::make_shared<std::atomic<int>>(0);
  PreparedAffineOperatorProvider counted = counted_test_operator(*A_ns_, operator_calls);

  const SolveReport report =
      run_prepared(counted, x, *rhs_ns_, bicgstab_krylov_method(),
                   LinearOperatorProperties::general(), Real(0), Real(1e-30), 2);

  EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
  // Problem A(0) + private-session bind warm + initial residual + 2*(Ap + As) + final residual.
  EXPECT_EQ(operator_calls->load(std::memory_order_relaxed), 8);
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
  const SolveReport r = run_prepared(*A_, x, *rhs_, richardson_krylov_method(omega),
                                     LinearOperatorProperties::general(), kRelTol, Real(0), 200000);
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
  const SolveReport r = run_prepared(*A_, x, *rhs_, gmres_krylov_method(30),
                                     LinearOperatorProperties::general(), kRelTol, Real(0), 500);
  const Real err = max_abs_diff(x, *phi_exact_mf_);
  std::printf("GMRES(SPD): %s in %d iters (rel=%.2e) | max|x - exact| = %.3e\n",
              r.solved() ? "CONVERGED" : "FAILED", r.iters, r.rel_residual, err);
  EXPECT_TRUE(r.solved()) << "gmres_spd_converged";
  EXPECT_TRUE(r.iters > 1) << "gmres_spd_iters_gt_1 iters=" << r.iters;
  EXPECT_TRUE(r.rel_residual <= kRelTol * 10)
      << "gmres_spd_residual_small rel_residual=" << r.rel_residual;
  EXPECT_TRUE(err < kRecoverTol) << "gmres_spd_recovers_exact err=" << err;
}

TEST_F(GenericKrylov, every_method_is_invariant_to_extreme_finite_equation_scaling) {
  struct MethodCase {
    TestKrylovFamily family;
    int restart;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kCg, 0},
      {TestKrylovFamily::kBicgstab, 0},
      {TestKrylovFamily::kGmres, 4},
      {TestKrylovFamily::kRichardson, 0},
  }};
  const std::array<Real, 2> equation_scales{Real(1e-200), Real(1e200)};
  const Real unscaled_reference = PureFieldAlgebra::norm(*phi_exact_mf_);

  for (const MethodCase& method : methods) {
    std::optional<SolveStatus> expected_status;
    for (const Real equation_scale : equation_scales) {
      ApplyFn scaled_identity = [equation_scale](MultiFab& out, const MultiFab& in) {
        PureFieldAlgebra::lincomb(out, equation_scale, in, Real(0), in);
      };
      MultiFab scaled_rhs(*ba_, *dm_, 1, 0);
      PureFieldAlgebra::copy(scaled_rhs, *phi_exact_mf_);
      scale(scaled_rhs, equation_scale);
      MultiFab x(*ba_, *dm_, 1, 1);
      x.set_val(Real(0));
      const LinearOperatorProperties properties =
          method.family == TestKrylovFamily::kCg
              ? LinearOperatorProperties::symmetric_positive_definite()
              : LinearOperatorProperties::general();
      // Richardson's public omega belongs to the physical A, so it transforms as 1/lambda while
      // the prepared recurrence itself remains normalized and overflow-safe.
      const Real relaxation =
          method.family == TestKrylovFamily::kRichardson ? Real(1) / equation_scale : Real(1);
      const PreparedKrylovMethod configured_method =
          test_krylov_method(method.family, method.restart, relaxation);
      const SolveReport report = run_prepared(scaled_identity, x, scaled_rhs, configured_method,
                                              properties, kRelTol, Real(0), 10);
      if (!expected_status)
        expected_status = report.status;
      EXPECT_EQ(report.status, *expected_status)
          << "method=" << configured_method.identity() << " lambda=" << equation_scale;
      EXPECT_TRUE(report.solved())
          << "method=" << configured_method.identity() << " lambda=" << equation_scale;
      EXPECT_TRUE(std::isfinite(report.reference_residual_norm));
      EXPECT_TRUE(std::isfinite(report.residual_norm));
      EXPECT_NEAR(report.reference_residual_norm / equation_scale, unscaled_reference,
                  unscaled_reference * Real(1e-12));
      EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol)
          << "method=" << configured_method.identity() << " lambda=" << equation_scale;
    }
  }
}

TEST_F(GenericKrylov, scaled_ratio_composition_preserves_underflow_times_overflow) {
  const detail::ScaledScalar underflowing = detail::scaled_quotient(Real(1e-200), Real(1e200));
  const detail::ScaledScalar overflowing = detail::scaled_quotient(Real(1e200), Real(1e-200));
  const detail::ScaledScalar product = detail::scaled_product(underflowing, overflowing);
  Real materialized = Real(0);
  ASSERT_TRUE(product.try_materialize(materialized));
  EXPECT_NEAR(materialized, Real(1), Real(1e-14));
}

TEST_F(GenericKrylov, zero_relative_tolerance_cannot_converge_through_ratio_underflow) {
  KrylovControls controls;
  controls.rel_tol = Real(0);
  controls.abs_tol = std::numeric_limits<Real>::denorm_min();
  EXPECT_FALSE(detail::satisfies_stopping_controls(Real(1e-300), Real(1e100), controls));

  controls.rel_tol = Real(1e-300);
  EXPECT_TRUE(detail::satisfies_stopping_controls(Real(1e-300), Real(1e100), controls));
}

TEST_F(GenericKrylov, every_method_applies_a_binary_scaled_extreme_coefficient_in_one_iteration) {
  struct MethodCase {
    TestKrylovFamily family;
    int restart;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kRichardson, 0},
      {TestKrylovFamily::kCg, 0},
      {TestKrylovFamily::kBicgstab, 0},
      {TestKrylovFamily::kGmres, 1},
  }};
  constexpr Real kOperatorScale = Real(1e-109);
  constexpr Real kRhsValue = Real(1e198);
  constexpr Real kExactValue = Real(1e307);

  ApplyFn scaled_identity = [=](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::lincomb(out, kOperatorScale, in, Real(0), in);
  };

  for (const MethodCase& method : methods) {
    MultiFab rhs(*ba_, *dm_, 1, 0);
    MultiFab exact(*ba_, *dm_, 1, 0);
    MultiFab iterate(*ba_, *dm_, 1, 1);
    rhs.set_val(kRhsValue);
    exact.set_val(kExactValue);
    iterate.set_val(Real(0));

    const LinearOperatorProperties properties =
        method.family == TestKrylovFamily::kCg
            ? LinearOperatorProperties::symmetric_positive_definite()
            : LinearOperatorProperties::general();
    const PreparedKrylovMethod configured_method =
        test_krylov_method(method.family, method.restart, Real(1e109));
    const SolveReport report = run_prepared(scaled_identity, iterate, rhs, configured_method,
                                            properties, Real(0), Real(1e190), 1);

    EXPECT_TRUE(report.solved()) << "method=" << configured_method.identity()
                                 << " status=" << static_cast<int>(report.status)
                                 << " reason=" << report.reason;
    EXPECT_EQ(report.iters, 1) << "method=" << configured_method.identity();
    EXPECT_TRUE(std::isfinite(report.residual_norm));
    EXPECT_LT(max_abs_diff(iterate, exact) / kExactValue, Real(1e-12))
        << "method=" << configured_method.identity();
  }
}

TEST_F(GenericKrylov, every_method_preserves_a_tiny_warm_residual_beside_a_huge_reference) {
  struct MethodCase {
    TestKrylovFamily family;
    PreparedKrylovMethod method;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kCg, cg_krylov_method()},
      {TestKrylovFamily::kBicgstab, bicgstab_krylov_method()},
      {TestKrylovFamily::kGmres, gmres_krylov_method(4)},
      {TestKrylovFamily::kRichardson, richardson_krylov_method()},
  }};
  ApplyFn identity = [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); };

  for (const MethodCase& method : methods) {
    MultiFab rhs(*ba_, *dm_, 1, 0);
    MultiFab warm(*ba_, *dm_, 1, 1);
    for (int local = 0; local < rhs.local_size(); ++local) {
      for_each_cell(rhs.box(local), SparseHugeReferenceKernel{rhs.fab(local).array()});
      for_each_cell(warm.box(local), SparseHugeWarmStartKernel{warm.fab(local).array()});
    }
    const LinearOperatorProperties properties =
        method.family == TestKrylovFamily::kCg
            ? LinearOperatorProperties::symmetric_positive_definite()
            : LinearOperatorProperties::general();
    const SolveReport report =
        run_prepared(identity, warm, rhs, method.method, properties, Real(0), Real(1e-310), 4);

    EXPECT_TRUE(report.solved()) << "method=" << method.method.identity()
                                 << " reason=" << report.reason;
    EXPECT_EQ(report.iters, 1) << "method=" << method.method.identity();
    EXPECT_GT(report.reference_residual_norm, Real(1e299));
    EXPECT_EQ(report.residual_norm, Real(0));
  }
}

TEST_F(GenericKrylov, gmres_normalizes_a_representable_subnormal_arnoldi_column) {
  ApplyFn almost_identity = [](MultiFab& out, const MultiFab& in) {
    for (int local = 0; local < out.local_size(); ++local)
      for_each_cell(out.box(local), SubnormalArnoldiColumnKernel{out.fab(local).array(),
                                                                 in.fab(local).const_array()});
  };
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab solution(*ba_, *dm_, 1, 1);
  for (int local = 0; local < rhs.local_size(); ++local)
    for_each_cell(rhs.box(local), SparseUnitForcingKernel{rhs.fab(local).array()});
  solution.set_val(Real(0));

  const SolveReport report =
      run_prepared(almost_identity, solution, rhs, gmres_krylov_method(4),
                   LinearOperatorProperties::general(), Real(0), Real(1e-320), 4);

  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 2);
  EXPECT_EQ(report.residual_norm, Real(0));
}

TEST_F(GenericKrylov, gmres_restart_normalizes_a_representable_subnormal_residual) {
  ApplyFn two_scale_diagonal = [](MultiFab& out, const MultiFab& in) {
    for (int local = 0; local < out.local_size(); ++local)
      for_each_cell(out.box(local),
                    TwoScaleDiagonalKernel{out.fab(local).array(), in.fab(local).const_array()});
  };
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab solution(*ba_, *dm_, 1, 1);
  for (int local = 0; local < rhs.local_size(); ++local)
    for_each_cell(rhs.box(local), UnitAndSubnormalForcingKernel{rhs.fab(local).array()});
  solution.set_val(Real(0));

  const SolveReport report = run_prepared(two_scale_diagonal, solution, rhs, gmres_krylov_method(1),
                                          LinearOperatorProperties::general(), Real(0),
                                          std::numeric_limits<Real>::denorm_min(), 4);

  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 2);
  EXPECT_EQ(report.residual_norm, Real(0));
}

TEST_F(GenericKrylov, cg_and_bicgstab_rebase_an_extreme_residual_before_restarting) {
  constexpr Real small_scale = Real(1e-200);
  ApplyFn two_scale_diagonal = [small_scale](MultiFab& out, const MultiFab& in) {
    for (int local = 0; local < out.local_size(); ++local)
      for_each_cell(
          out.box(local),
          TwoScaleDiagonalKernel{out.fab(local).array(), in.fab(local).const_array(), small_scale});
  };
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab exact(*ba_, *dm_, 1, 1);
  for (int local = 0; local < rhs.local_size(); ++local) {
    for_each_cell(rhs.box(local),
                  UnitAndSubnormalForcingKernel{rhs.fab(local).array(), small_scale});
    for_each_cell(exact.box(local), UnitPairSolutionKernel{exact.fab(local).array()});
  }

  struct MethodCase {
    TestKrylovFamily family;
    PreparedKrylovMethod method;
  };
  for (const MethodCase& test_case :
       {MethodCase{TestKrylovFamily::kCg, cg_krylov_method()},
        MethodCase{TestKrylovFamily::kBicgstab, bicgstab_krylov_method()}}) {
    MultiFab solution(*ba_, *dm_, 1, 1);
    solution.set_val(Real(0));
    const LinearOperatorProperties properties =
        test_case.family == TestKrylovFamily::kCg
            ? LinearOperatorProperties::symmetric_positive_definite()
            : LinearOperatorProperties::general();
    const SolveReport report = run_prepared(two_scale_diagonal, solution, rhs, test_case.method,
                                            properties, Real(0), Real(1e-210), 4);

    EXPECT_TRUE(report.solved()) << "method=" << test_case.method.identity()
                                 << " reason=" << report.reason;
    EXPECT_EQ(report.iters, 2) << "method=" << test_case.method.identity();
    EXPECT_EQ(report.reference_residual_norm, Real(1));
    EXPECT_EQ(report.residual_norm, Real(0));
    EXPECT_LT(max_abs_diff(solution, exact), Real(1e-14))
        << "method=" << test_case.method.identity();
  }
}

TEST_F(GenericKrylov, gmres_does_not_count_an_unusable_arnoldi_column_as_an_iteration) {
  ApplyFn zero_operator = [](MultiFab& out, const MultiFab&) { PureFieldAlgebra::zero_valid(out); };
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab solution(*ba_, *dm_, 1, 1);
  rhs.set_val(Real(1));
  solution.set_val(Real(0));

  const SolveReport report = run_prepared(zero_operator, solution, rhs, gmres_krylov_method(1),
                                          LinearOperatorProperties::general(), kRelTol, Real(0), 4);

  EXPECT_EQ(report.status, SolveStatus::kBreakdown);
  EXPECT_EQ(report.iters, 0);
  EXPECT_EQ(report.residual_norm, report.reference_residual_norm);
}

TEST_F(GenericKrylov, reference_preserves_tiny_forcing_beside_a_cancelling_huge_affine_term) {
  MultiFab affine_constant(*ba_, *dm_, 1, 0);
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab exact(*ba_, *dm_, 1, 1);
  for (int li = 0; li < affine_constant.local_size(); ++li) {
    Array4 constant_values = affine_constant.fab(li).array();
    Array4 rhs_values = rhs.fab(li).array();
    Array4 exact_values = exact.fab(li).array();
    for_each_cell(affine_constant.box(li), SparseAffineConstantKernel{constant_values});
    for_each_cell(rhs.box(li), SparseTinyForcingKernel{rhs_values});
    for_each_cell(exact.box(li), SparseTinyForcingKernel{exact_values});
  }
  PureFieldAlgebra::axpy(rhs, Real(1), affine_constant);
  ApplyFn affine_identity = [&affine_constant](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::lincomb(out, Real(1), in, Real(1), affine_constant);
  };
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));

  const SolveReport report =
      run_prepared(affine_identity, x, rhs, cg_krylov_method(),
                   LinearOperatorProperties::symmetric_positive_definite(), kRelTol, Real(0), 4);
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_NEAR(report.reference_residual_norm / Real(1e-200), Real(1), Real(1e-14));
  EXPECT_NEAR(PureFieldAlgebra::norm(x) / Real(1e-200), Real(1), Real(1e-12));
  EXPECT_EQ(report.residual_norm, Real(0));
}

TEST_F(GenericKrylov, every_method_solves_a_compatible_periodic_nullspace_problem) {
  struct MethodCase {
    TestKrylovFamily family;
    int restart;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kCg, 0},
      {TestKrylovFamily::kBicgstab, 0},
      {TestKrylovFamily::kGmres, 4},
      {TestKrylovFamily::kRichardson, 0},
  }};

  MultiFab exact(*ba_, *dm_, 1, 1);
  for (int li = 0; li < exact.local_size(); ++li) {
    Array4 values = exact.fab(li).array();
    for_each_cell(exact.box(li), SampleSinglePeriodicModeKernel{values, *geom_});
  }
  const PreparedTestOperator negative_laplacian =
      prepared_negative_periodic_laplacian(exact, *geom_, *bc_);
  MultiFab compatible_rhs(*ba_, *dm_, 1, 0);
  negative_laplacian.apply_once(compatible_rhs, exact);

  const Real h = geom_->dx();
  const Real single_mode_eigenvalue =
      Real(8) * std::pow(std::sin(kPi / static_cast<Real>(kN)), Real(2)) / (h * h);
  for (const MethodCase& method : methods) {
    MultiFab x(*ba_, *dm_, 1, 1);
    x.set_val(Real(3));  // The declared gauge must be applied before the initial residual.
    const Real relaxation =
        method.family == TestKrylovFamily::kRichardson ? Real(1) / single_mode_eigenvalue : Real(1);
    const PreparedKrylovMethod configured_method =
        test_krylov_method(method.family, method.restart, relaxation);
    const SolveReport report = run_prepared_with_preconditioner(
        negative_laplacian.provider, x, compatible_rhs, configured_method,
        periodic_nullspace_properties(method.family), PreparedLinearPreconditioner::identity(),
        kRelTol, Real(0), 20, periodic_mean_zero_policy(*geom_));

    EXPECT_TRUE(report.solved()) << "method=" << configured_method.identity()
                                 << " reason=" << report.reason;
    EXPECT_LT(max_abs_diff(x, exact), kRecoverTol) << "method=" << configured_method.identity();
    EXPECT_NEAR(reduce_sum(x) / static_cast<Real>(kN * kN), Real(0), Real(1e-13))
        << "the final published value must satisfy the declared mean-zero gauge";
  }
}

TEST_F(GenericKrylov, final_residual_confirmation_has_no_redundant_nonsingular_apply) {
  auto warm_apply_count = std::make_shared<std::atomic<int>>(0);
  PreparedAffineOperatorProvider counted_identity =
      counted_test_operator(reentrant_test_operator([](MultiFab& out, const MultiFab& in) {
                              PureFieldAlgebra::copy(out, in);
                            }),
                            warm_apply_count);
  MultiFab warm(*ba_, *dm_, 1, 1);
  MultiFab identity_rhs(*ba_, *dm_, 1, 0);
  PureFieldAlgebra::copy(warm, *phi_exact_mf_);
  PureFieldAlgebra::copy(identity_rhs, *phi_exact_mf_);
  const SolveReport warm_report =
      run_prepared(counted_identity, warm, identity_rhs, cg_krylov_method(),
                   LinearOperatorProperties::symmetric_positive_definite(), kRelTol, Real(0), 4);
  EXPECT_TRUE(warm_report.solved());
  EXPECT_EQ(warm_report.iters, 0);
  EXPECT_EQ(warm_apply_count->load(std::memory_order_relaxed), 3)
      << "problem A(0), private-session bind warm, then one initial true residual";

  auto limited_apply_count = std::make_shared<std::atomic<int>>(0);
  PreparedAffineOperatorProvider limited_identity =
      counted_test_operator(reentrant_test_operator([](MultiFab& out, const MultiFab& in) {
                              PureFieldAlgebra::copy(out, in);
                            }),
                            limited_apply_count);
  MultiFab limited(*ba_, *dm_, 1, 1);
  limited.set_val(Real(0));
  const SolveReport limited_report =
      run_prepared(limited_identity, limited, identity_rhs, richardson_krylov_method(Real(0.25)),
                   LinearOperatorProperties::general(), Real(1e-14), Real(0), 1);
  EXPECT_EQ(limited_report.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(limited_apply_count->load(std::memory_order_relaxed), 4)
      << "problem A(0), private-session bind warm, initial residual, and one Richardson step";

  struct OneStepRoute {
    PreparedKrylovMethod method;
    LinearOperatorProperties properties;
  };
  const std::array<OneStepRoute, 3> routes{{
      {cg_krylov_method(), LinearOperatorProperties::symmetric_positive_definite()},
      {bicgstab_krylov_method(), LinearOperatorProperties::general()},
      {gmres_krylov_method(1), LinearOperatorProperties::general()},
  }};
  for (const OneStepRoute& route : routes) {
    auto apply_count = std::make_shared<std::atomic<int>>(0);
    PreparedAffineOperatorProvider counted =
        counted_test_operator(reentrant_test_operator([](MultiFab& out, const MultiFab& in) {
                                PureFieldAlgebra::copy(out, in);
                              }),
                              apply_count);
    MultiFab iterate(*ba_, *dm_, 1, 1);
    PureFieldAlgebra::zero_valid(iterate);
    const SolveReport report = run_prepared(counted, iterate, identity_rhs, route.method,
                                            route.properties, Real(1e-14), Real(0), 1);

    EXPECT_TRUE(report.solved()) << "method=" << route.method.identity();
    EXPECT_EQ(report.iters, 1) << "method=" << route.method.identity();
    EXPECT_EQ(apply_count->load(std::memory_order_relaxed), 5)
        << "problem A(0), private-session bind warm, initial residual, one method matvec, and one "
           "common final confirmation; method="
        << route.method.identity();
  }
}

TEST_F(GenericKrylov, singular_solved_value_pays_exactly_one_post_gauge_confirmation) {
  MultiFab exact(*ba_, *dm_, 1, 1);
  for (int li = 0; li < exact.local_size(); ++li) {
    Array4 values = exact.fab(li).array();
    for_each_cell(exact.box(li), SampleSinglePeriodicModeKernel{values, *geom_});
  }
  const PreparedTestOperator negative_laplacian =
      prepared_negative_periodic_laplacian(exact, *geom_, *bc_);
  MultiFab compatible_rhs(*ba_, *dm_, 1, 0);
  negative_laplacian.apply_once(compatible_rhs, exact);

  auto apply_count = std::make_shared<std::atomic<int>>(0);
  PreparedAffineOperatorProvider counted =
      counted_test_operator(negative_laplacian.provider, apply_count);
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(3));
  const SolveReport report =
      run_prepared_with_preconditioner(counted, x, compatible_rhs, cg_krylov_method(),
                                       periodic_nullspace_properties(TestKrylovFamily::kCg),
                                       PreparedLinearPreconditioner::identity(), kRelTol, Real(0),
                                       4, periodic_mean_zero_policy(*geom_));

  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 1);
  EXPECT_EQ(apply_count->load(std::memory_order_relaxed), 6)
      << "problem A(0), private-session bind warm, initial residual, one CG matvec, convergence "
         "confirmation, post-gauge check";
}

TEST_F(GenericKrylov, every_method_reports_an_incompatible_rhs_before_gauge_or_iteration) {
  struct MethodCase {
    TestKrylovFamily family;
    int restart;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kCg, 0},
      {TestKrylovFamily::kBicgstab, 0},
      {TestKrylovFamily::kGmres, 4},
      {TestKrylovFamily::kRichardson, 0},
  }};

  MultiFab exact(*ba_, *dm_, 1, 1);
  for (int li = 0; li < exact.local_size(); ++li) {
    Array4 values = exact.fab(li).array();
    for_each_cell(exact.box(li), SampleSinglePeriodicModeKernel{values, *geom_});
  }
  const PreparedTestOperator negative_laplacian =
      prepared_negative_periodic_laplacian(exact, *geom_, *bc_);
  MultiFab incompatible_rhs(*ba_, *dm_, 1, 0);
  negative_laplacian.apply_once(incompatible_rhs, exact);
  MultiFab constant(*ba_, *dm_, 1, 0);
  constant.set_val(Real(1));
  PureFieldAlgebra::axpy(incompatible_rhs, Real(1), constant);
  const Real expected_warm_residual = PureFieldAlgebra::norm(constant);

  for (const MethodCase& method : methods) {
    auto operator_calls = std::make_shared<std::atomic<int>>(0);
    PreparedAffineOperatorProvider counted =
        counted_test_operator(negative_laplacian.provider, operator_calls);
    MultiFab x(*ba_, *dm_, 1, 1);
    PureFieldAlgebra::copy(x, exact);
    const PreparedKrylovMethod configured_method =
        test_krylov_method(method.family, method.restart);
    const SolveReport report = run_prepared_with_preconditioner(
        counted, x, incompatible_rhs, configured_method,
        periodic_nullspace_properties(method.family), PreparedLinearPreconditioner::identity(),
        kRelTol, Real(0), 20, periodic_mean_zero_policy(*geom_));

    EXPECT_EQ(report.status, SolveStatus::kIncompatibleRhs)
        << "method=" << configured_method.identity();
    EXPECT_EQ(report.action, SolveAction::kFailRun);
    EXPECT_EQ(report.iters, 0);
    EXPECT_NE(report.reason.find("incompatible"), std::string::npos);
    EXPECT_EQ(operator_calls->load(std::memory_order_relaxed), 3)
        << "compatibility rejection permits problem A(0), private-session bind warm, and one "
           "report residual";
    EXPECT_NEAR(report.residual_norm, expected_warm_residual, expected_warm_residual * Real(1e-14));
    EXPECT_LT(max_abs_diff(x, exact), Real(1e-14))
        << "compatibility must be checked before the initial gauge or iterate mutation";
  }
}

TEST_F(GenericKrylov, nonfinite_nullspace_rhs_is_invalid_not_incompatible) {
  const PreparedTestOperator negative_laplacian =
      prepared_negative_periodic_laplacian(*phi_exact_mf_, *geom_, *bc_);
  MultiFab nonfinite_rhs(*ba_, *dm_, 1, 0);
  nonfinite_rhs.set_val(std::numeric_limits<Real>::quiet_NaN());
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));

  const SolveReport report = run_prepared_with_preconditioner(
      negative_laplacian.provider, x, nonfinite_rhs, cg_krylov_method(),
      LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(),
      PreparedLinearPreconditioner::identity(), kRelTol, Real(0), 4,
      periodic_mean_zero_policy(*geom_));
  EXPECT_EQ(report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_NE(report.status, SolveStatus::kIncompatibleRhs);
}

TEST_F(GenericKrylov, nullspace_and_positive_definiteness_certificates_must_be_coherent) {
  ApplyFn identity = [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); };
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));

  EXPECT_THROW((void)run_prepared_with_preconditioner(
                   identity, x, *rhs_, cg_krylov_method(),
                   LinearOperatorProperties::symmetric_positive_definite(),
                   PreparedLinearPreconditioner::identity(), kRelTol, Real(0), 4,
                   periodic_mean_zero_policy(*geom_)),
               std::invalid_argument);
  EXPECT_THROW((void)run_prepared(
                   identity, x, *rhs_, cg_krylov_method(),
                   LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(),
                   kRelTol, Real(0), 4),
               std::invalid_argument);
  EXPECT_THROW((void)run_prepared_with_preconditioner(
                   identity, x, *rhs_, cg_krylov_method(), LinearOperatorProperties::symmetric(),
                   PreparedLinearPreconditioner::identity(), kRelTol, Real(0), 4,
                   periodic_mean_zero_policy(*geom_)),
               std::invalid_argument);
}

TEST_F(GenericKrylov, periodic_nullspace_solve_is_collective_safe_with_an_empty_rank) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "the serial registration has no empty MPI rank";

  const Box2D domain = Box2D::from_extents(kN, kN);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping owner_zero(std::vector<int>{0});
  const BCRec periodic{};
  MultiFab exact(boxes, owner_zero, 1, 1);
  for (int li = 0; li < exact.local_size(); ++li) {
    Array4 values = exact.fab(li).array();
    for_each_cell(exact.box(li), SampleSinglePeriodicModeKernel{values, geometry});
  }
  const PreparedTestOperator negative_laplacian =
      prepared_negative_periodic_laplacian(exact, geometry, periodic);
  MultiFab rhs(boxes, owner_zero, 1, 0);
  negative_laplacian.apply_once(rhs, exact);
  MultiFab x(boxes, owner_zero, 1, 1);
  x.set_val(Real(2));

  EXPECT_EQ(x.local_size(), my_rank() == 0 ? 1 : 0);
  ASSERT_EQ(static_cast<int>(all_reduce_sum(static_cast<double>(x.local_size()))), 1);
  const SolveReport report = run_prepared_with_preconditioner(
      negative_laplacian.provider, x, rhs, cg_krylov_method(),
      LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(),
      PreparedLinearPreconditioner::identity(), kRelTol, Real(0), 20,
      periodic_mean_zero_policy(geometry));

  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(max_abs_diff(x, exact), kRecoverTol);
  EXPECT_NEAR(reduce_sum(x) / static_cast<Real>(kN * kN), Real(0), Real(1e-13));
}

TEST_F(GenericKrylov, gmres_restart_decision_is_invariant_to_preconditioner_scale) {
  MultiFab x_unit(*ba_, *dm_, 1, 1);
  MultiFab x_scaled(*ba_, *dm_, 1, 1);
  x_unit.set_val(0.0);
  x_scaled.set_val(0.0);

  const SolveReport unit = run_prepared_with_preconditioner(
      *A_, x_unit, *rhs_, gmres_krylov_method(30), LinearOperatorProperties::general(),
      affine_scaled_identity_preconditioner(x_unit, Real(1), Real(3.25)), kRelTol, Real(0), 500);
  // Scale the affine response with the linear map as a physical rescaling of one preconditioner
  // would do. Keeping a fixed O(1) offset while shrinking only the direction response to 1e-8
  // makes M_raw(v)-M_raw(0) an intentionally ill-conditioned floating-point subtraction and no
  // longer tests restart invariance of equivalent preconditioners.
  const SolveReport scaled = run_prepared_with_preconditioner(
      *A_, x_scaled, *rhs_, gmres_krylov_method(30), LinearOperatorProperties::general(),
      affine_scaled_identity_preconditioner(x_scaled, Real(1e-8), Real(3.25e-8)), kRelTol, Real(0),
      500);

  EXPECT_TRUE(unit.solved()) << "unit-scaled preconditioner failed after " << unit.iters;
  EXPECT_TRUE(scaled.solved()) << "small-scaled preconditioner failed after " << scaled.iters;
  EXPECT_LE(scaled.iters, unit.iters + 1);
  EXPECT_GE(scaled.iters + 1, unit.iters);
  EXPECT_TRUE(max_abs_diff(x_unit, *phi_exact_mf_) < kRecoverTol);
  EXPECT_TRUE(max_abs_diff(x_scaled, *phi_exact_mf_) < kRecoverTol);
}

TEST_F(GenericKrylov, prepared_methods_remove_extreme_finite_preconditioner_scaling) {
  struct MethodCase {
    PreparedKrylovMethod method;
  };
  const std::array<MethodCase, 2> methods{{
      {bicgstab_krylov_method()},
      {gmres_krylov_method(30)},
  }};
  const std::array<Real, 2> scales{{Real(1e-200), Real(1e200)}};

  for (const MethodCase& method : methods) {
    for (const Real scale : scales) {
      MultiFab x(*ba_, *dm_, 1, 1);
      x.set_val(Real(0));
      const SolveReport report = run_prepared_with_preconditioner(
          *A_, x, *rhs_, method.method, LinearOperatorProperties::general(),
          affine_scaled_identity_preconditioner(x, scale, Real(3.25) * scale), kRelTol, Real(0),
          500);

      EXPECT_TRUE(report.solved()) << "method=" << method.method.identity() << " scale=" << scale
                                   << " reason=" << report.reason;
      EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol)
          << "method=" << method.method.identity() << " scale=" << scale;
    }
  }
}

TEST_F(GenericKrylov, bicgstab_linearizes_an_affine_preconditioner) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));
  const SolveReport report = run_prepared_with_preconditioner(
      *A_, x, *rhs_, bicgstab_krylov_method(), LinearOperatorProperties::general(),
      affine_scaled_identity_preconditioner(x, Real(0.75), Real(8.0)), kRelTol, Real(0), 500);
  EXPECT_TRUE(report.solved());
  EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol);
}

TEST_F(GenericKrylov, generic_methods_linearize_real_dirichlet_and_robin_preconditioners) {
  struct Case {
    PreparedKrylovMethod method;
    BCType boundary;
  };
  const std::array<Case, 4> cases{{
      {gmres_krylov_method(30), BCType::Dirichlet},
      {gmres_krylov_method(30), BCType::Robin},
      {bicgstab_krylov_method(), BCType::Dirichlet},
      {bicgstab_krylov_method(), BCType::Robin},
  }};
  for (const Case& test_case : cases) {
    MultiFab x(*ba_, *dm_, 1, 1);
    x.set_val(Real(0));
    const SolveReport report = run_prepared_with_preconditioner(
        *A_, x, *rhs_, test_case.method, LinearOperatorProperties::general(),
        boundary_affine_preconditioner(x, *geom_, test_case.boundary), kRelTol, Real(0), 500);
    EXPECT_TRUE(report.solved()) << "method=" << test_case.method.identity()
                                 << " boundary=" << static_cast<int>(test_case.boundary)
                                 << " iters=" << report.iters;
    EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol)
        << "method=" << test_case.method.identity()
        << " boundary=" << static_cast<int>(test_case.boundary);
  }
}

TEST_F(GenericKrylov, prepared_full_tensor_bicgstab_converges_where_the_diagonal_vcycle_stalls) {
  constexpr int cells = 64;
  struct TensorCase {
    Real strength;
    bool nonsymmetric;
  };
  const std::array<TensorCase, 6> cases{{
      {Real(0.1), false},
      {Real(0.4), false},
      {Real(0.7), false},
      {Real(0.5), true},
      {Real(2), true},
      {Real(8), true},
  }};
  int strong_nonsymmetric_iterations = -1;
  for (const TensorCase& test_case : cases) {
    const PreparedTensorCaseReport report =
        solve_prepared_tensor_case(cells, test_case.strength, test_case.nonsymmetric);
    EXPECT_TRUE(report.krylov.solved())
        << "strength=" << test_case.strength << " nonsymmetric=" << test_case.nonsymmetric
        << " status=" << report.krylov.status_name();
    EXPECT_LT(report.krylov.rel_residual, Real(1e-10))
        << "strength=" << test_case.strength << " nonsymmetric=" << test_case.nonsymmetric;
    EXPECT_GT(report.mg_initial, Real(0));
    if (test_case.strength == Real(8) && test_case.nonsymmetric) {
      strong_nonsymmetric_iterations = report.krylov.iters;
      EXPECT_TRUE(std::isfinite(report.mg_final));
      EXPECT_GE(report.mg_final, Real(1e-6) * report.mg_initial)
          << "the strong variable-skew tensor is the regression where a diagonal-block "
             "V-cycle alone must not be mistaken for a converged solve";
      EXPECT_GT(report.krylov.iters, 1)
          << "the strong variable-skew case must exercise a real Krylov iteration sequence";
    }
  }

  ASSERT_GE(strong_nonsymmetric_iterations, 0);
  const long local_iterations = strong_nonsymmetric_iterations;
  const long minimum_iterations =
      -static_cast<long>(all_reduce_max(static_cast<double>(-local_iterations)));
  const long maximum_iterations =
      static_cast<long>(all_reduce_max(static_cast<double>(local_iterations)));
  EXPECT_EQ(minimum_iterations, maximum_iterations)
      << "collective stopping must choose one iteration count on every MPI rank";
}

TEST_F(GenericKrylov, prepared_tensor_identity_matches_the_geometric_mg_reference) {
  constexpr int cells = 64;
  const Box2D domain = Box2D::from_extents(cells, cells);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BoxArray boxes = BoxArray::from_domain(domain, cells / 2);
  const DistributionMapping distribution(boxes.size(), n_ranks());
  BCRec boundary;
  boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = BCType::Dirichlet;

  MultiFab rhs(boxes, distribution, 1, 0);
  fill_tensor_mms_rhs(rhs, geometry, Real(0), Real(0));
  GeometricMG reference(geometry, boxes, distribution, boundary);
  PureFieldAlgebra::copy(reference.rhs(), rhs);
  PureFieldAlgebra::zero_valid(reference.phi());
  reference.solve(Real(1e-12), 100);
  ASSERT_TRUE(reference.last_solve_report().solved());

  auto op = std::make_shared<GeometricMG>(geometry, boxes, distribution, boundary);
  op->set_cross_terms(constant_scalar_field_provider(Real(0)),
                      constant_scalar_field_provider(Real(0)));
  MultiFab iterate(boxes, distribution, 1, 1);
  PureFieldAlgebra::zero_valid(iterate);
  const SolveReport report = run_prepared_with_preconditioner(
      prepared_tensor_operator(op).provider, iterate, rhs, bicgstab_krylov_method(),
      LinearOperatorProperties::general(),
      prepared_geometric_mg_preconditioner(iterate, geometry, boundary), Real(1e-12), Real(0), 300);

  EXPECT_TRUE(report.solved()) << report.status_name();
  EXPECT_LT(max_abs_diff(iterate, reference.phi()), Real(1e-8));
}

TEST_F(GenericKrylov,
       prepared_physical_boundary_offsets_use_the_effective_rhs_and_a_linear_preconditioner) {
  constexpr Real solve_tolerance = Real(1e-9);
  for (const BCType type : {BCType::Dirichlet, BCType::Robin}) {
    SCOPED_TRACE(type == BCType::Dirichlet ? "Dirichlet" : "Robin");
    const AffineBoundaryCaseReport report = solve_affine_boundary_case(type);
    EXPECT_EQ(report.offset.probe.iters, 1)
        << "a huge affine offset must not make a small effective forcing look converged";
    for (const AffineBoundarySolveReport* solve : {&report.homogeneous, &report.offset}) {
      EXPECT_TRUE(solve->converged.solved())
          << solve->converged.reason << " iters=" << solve->converged.iters
          << " rel=" << solve->converged.rel_residual
          << " residual=" << solve->converged.residual_norm
          << " reference=" << solve->converged.reference_residual_norm
          << " independent_rel=" << solve->true_relative_residual;
      EXPECT_LE(solve->converged.rel_residual, solve_tolerance);
      EXPECT_LE(solve->true_relative_residual, solve_tolerance)
          << "an independently materialized b-A(x) residual must satisfy the authored tolerance";
      EXPECT_TRUE(solve->warm.solved()) << solve->warm.status_name();
      EXPECT_EQ(solve->warm.iters, 0);
      EXPECT_LE(solve->warm.rel_residual, solve_tolerance);
    }

    const Real rhs_roundoff_bound =
        Real(64) * std::numeric_limits<Real>::epsilon() * report.offset.affine_offset_max;
    EXPECT_LE(report.offset.effective_rhs_error, rhs_roundoff_bound)
        << "forming b-A(0) may lose only the unavoidable rounding from the authored affine rhs";
    EXPECT_NEAR(report.offset.converged.reference_residual_norm,
                report.homogeneous.converged.reference_residual_norm,
                report.homogeneous.converged.reference_residual_norm * Real(1e-9))
        << "the Krylov reference must be the effective forcing, independent of A(0)";
    EXPECT_LT(report.solution_difference, Real(1e-7))
        << "homogeneous and 1e3-offset boundary problems have the same linearized equation";
  }
}

TEST_F(GenericKrylov, prepared_nonzero_dirichlet_problem_matches_mg_and_the_mms) {
  constexpr int cells = 64;
  constexpr Real boundary_value = Real(1);
  const Box2D domain = Box2D::from_extents(cells, cells);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BoxArray boxes = BoxArray::from_domain(domain, cells / 2);
  const DistributionMapping distribution(boxes.size(), n_ranks());
  BCRec boundary;
  boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = BCType::Dirichlet;
  boundary.xlo_val = boundary.xhi_val = boundary.ylo_val = boundary.yhi_val = boundary_value;

  MultiFab rhs(boxes, distribution, 1, 0);
  fill_tensor_mms_rhs(rhs, geometry, Real(0), Real(0));
  GeometricMG reference(geometry, boxes, distribution, boundary);
  PureFieldAlgebra::copy(reference.rhs(), rhs);
  PureFieldAlgebra::zero_valid(reference.phi());
  reference.solve(Real(1e-12), 200);
  ASSERT_TRUE(reference.last_solve_report().solved());

  auto op = std::make_shared<GeometricMG>(geometry, boxes, distribution, boundary);
  op->set_cross_terms(constant_scalar_field_provider(Real(0)),
                      constant_scalar_field_provider(Real(0)));
  MultiFab iterate(boxes, distribution, 1, 1);
  MultiFab exact(boxes, distribution, 1, 1);
  for (int local = 0; local < exact.local_size(); ++local) {
    Array4 values = exact.fab(local).array();
    for_each_cell(exact.box(local), ShiftedDirichletExactKernel{values, geometry, boundary_value});
  }
  PureFieldAlgebra::zero_valid(iterate);
  const SolveReport report = run_prepared_with_preconditioner(
      prepared_tensor_operator(op).provider, iterate, rhs, bicgstab_krylov_method(),
      LinearOperatorProperties::general(),
      prepared_geometric_mg_preconditioner(iterate, geometry, boundary), Real(1e-10), Real(0), 300);

  EXPECT_TRUE(report.solved()) << report.status_name();
  EXPECT_LT(report.rel_residual, Real(1e-10));
  EXPECT_LT(max_abs_diff(iterate, reference.phi()), Real(1e-7));
  EXPECT_LT(max_abs_diff(iterate, exact), Real(2e-2));
}

TEST_F(GenericKrylov, cg_refuses_unproven_operator_and_gmres_solves_nonsymmetric_operator) {
  MultiFab x_cg(*ba_, *dm_, 1, 1);
  x_cg.set_val(0.0);
  EXPECT_THROW((void)run_prepared(*A_ns_, x_cg, *rhs_ns_, cg_krylov_method(),
                                  LinearOperatorProperties::general()),
               std::invalid_argument);

  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport r = run_prepared(*A_ns_, x, *rhs_ns_, gmres_krylov_method(30),
                                     LinearOperatorProperties::general(), kRelTol, Real(0), 500);
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
  EXPECT_THROW((void)run_prepared(*A_, x, *rhs_, cg_krylov_method(),
                                  LinearOperatorProperties::symmetric_positive_definite(), kRelTol,
                                  Real(0), 0),
               std::invalid_argument);
}

TEST_F(GenericKrylov, richardson_relaxation_is_validated_by_the_prepared_method) {
  EXPECT_THROW((void)richardson_krylov_method(Real(0)), std::invalid_argument);
  EXPECT_THROW((void)richardson_krylov_method(std::numeric_limits<Real>::quiet_NaN()),
               std::invalid_argument);
}

TEST_F(GenericKrylov, gmres_restart_is_exact_and_dynamically_sized) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  EXPECT_THROW((void)run_prepared(*A_, x, *rhs_, gmres_krylov_method(0),
                                  LinearOperatorProperties::general(), kRelTol, Real(0), 500),
               std::invalid_argument);
  const SolveReport dynamic =
      run_prepared(*A_, x, *rhs_, gmres_krylov_method(51), LinearOperatorProperties::general(),
                   kRelTol, Real(0), 500);
  EXPECT_TRUE(dynamic.solved());
}

TEST_F(GenericKrylov, failed_solves_report_no_solved_value) {
  MultiFab x_limit(*ba_, *dm_, 1, 1);
  x_limit.set_val(0.0);
  const SolveReport limited =
      run_prepared(*A_, x_limit, *rhs_, richardson_krylov_method(Real(1e-12)),
                   LinearOperatorProperties::general(), kRelTol, Real(0), 1);
  EXPECT_FALSE(limited.solved_value_available());
  EXPECT_EQ(limited.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(limited.action, SolveAction::kFailRun);

  ApplyFn zero_op = [](MultiFab& out, const MultiFab&) { out.set_val(0.0); };
  MultiFab x_break(*ba_, *dm_, 1, 1);
  x_break.set_val(0.0);
  const SolveReport breakdown =
      run_prepared(zero_op, x_break, *rhs_, cg_krylov_method(),
                   LinearOperatorProperties::symmetric_positive_definite(), kRelTol, Real(0), 10);
  EXPECT_FALSE(breakdown.solved_value_available());
  EXPECT_EQ(breakdown.status, SolveStatus::kBreakdown);
  EXPECT_EQ(breakdown.action, SolveAction::kFailRun);

  for (const int maximum_iterations : {1, 10}) {
    SCOPED_TRACE(maximum_iterations);
    MultiFab x_bicgstab_break(*ba_, *dm_, 1, 1);
    x_bicgstab_break.set_val(0.0);
    const SolveReport bicgstab_breakdown =
        run_prepared(zero_op, x_bicgstab_break, *rhs_, bicgstab_krylov_method(),
                     LinearOperatorProperties::general(), kRelTol, Real(0), maximum_iterations);
    EXPECT_FALSE(bicgstab_breakdown.solved_value_available());
    EXPECT_EQ(bicgstab_breakdown.status, SolveStatus::kBreakdown);
    EXPECT_EQ(bicgstab_breakdown.action, SolveAction::kFailRun);
    EXPECT_NE(bicgstab_breakdown.reason.find("alpha denominator"), std::string::npos);
  }
}

TEST_F(GenericKrylov, bicgstab_omega_breakdown_commits_alpha_then_fails_honestly) {
  const Box2D domain = Box2D::from_extents(2, 1);
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping distribution(boxes.size(), n_ranks());
  MultiFab rhs(boxes, distribution, 1, 0);
  MultiFab iterate(boxes, distribution, 1, 0);
  MultiFab expected(boxes, distribution, 1, 0);
  for (int local = 0; local < rhs.local_size(); ++local) {
    for_each_cell(rhs.box(local), SparseUnitForcingKernel{rhs.fab(local).array()});
    for_each_cell(expected.box(local), SparseUnitForcingKernel{expected.fab(local).array()});
  }
  iterate.set_val(Real(0));

  const ApplyFn omega_breakdown = [](MultiFab& out, const MultiFab& in) {
    for (int local = 0; local < out.local_size(); ++local) {
      for_each_cell(out.box(local), TwoCellOmegaBreakdownKernel{out.fab(local).array(),
                                                                in.fab(local).const_array()});
    }
  };
  const SolveReport report = run_prepared(omega_breakdown, iterate, rhs, bicgstab_krylov_method(),
                                          LinearOperatorProperties::general(), kRelTol, Real(0), 4);

  EXPECT_EQ(report.status, SolveStatus::kBreakdown);
  EXPECT_FALSE(report.solved_value_available());
  EXPECT_NE(report.reason.find("alpha denominator"), std::string::npos);
  EXPECT_EQ(max_abs_diff(iterate, expected), Real(0))
      << "the valid alpha correction must be retained before omega-breakdown recovery";
}

TEST_F(GenericKrylov, affine_constant_is_removed_exactly) {
  constexpr Real offset = Real(3.25);
  MultiFab constant(*ba_, *dm_, 1, 0);
  constant.set_val(offset);
  PreparedTestOperator affine{shifted_test_operator(*A_, constant)};
  MultiFab affine_rhs(*ba_, *dm_, 1, 0);
  affine.apply_once(affine_rhs, *phi_exact_mf_);
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  const SolveReport report = run_prepared(affine.provider, x, affine_rhs, cg_krylov_method(),
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
    TestKrylovFamily family;
    PreparedKrylovMethod method;
  };
  const std::array<MethodCase, 4> methods{{
      {TestKrylovFamily::kCg, cg_krylov_method()},
      {TestKrylovFamily::kBicgstab, bicgstab_krylov_method()},
      {TestKrylovFamily::kGmres, gmres_krylov_method(4)},
      {TestKrylovFamily::kRichardson, richardson_krylov_method()},
  }};
  for (const MethodCase& method : methods) {
    MultiFab x(*ba_, *dm_, 1, 1);
    x.set_val(Real(0));
    const LinearOperatorProperties properties =
        method.family == TestKrylovFamily::kCg
            ? LinearOperatorProperties::symmetric_positive_definite()
            : LinearOperatorProperties::general();
    const SolveReport report = run_prepared(affine_identity, x, affine_rhs, method.method,
                                            properties, kRelTol, Real(0), 8);
    EXPECT_TRUE(report.solved()) << "method=" << method.method.identity();
    EXPECT_LT(max_abs_diff(x, *phi_exact_mf_), kRecoverTol)
        << "method=" << method.method.identity();
  }
}

TEST_F(GenericKrylov, warm_start_and_absolute_floor_use_true_residual) {
  const Real forcing_reference = PureFieldAlgebra::norm(*rhs_);
  MultiFab exact(*ba_, *dm_, 1, 1);
  PureFieldAlgebra::copy(exact, *phi_exact_mf_);
  const SolveReport warm = run_prepared(*A_, exact, *rhs_, gmres_krylov_method(10),
                                        LinearOperatorProperties::general(), kRelTol, Real(0), 50);
  EXPECT_TRUE(warm.solved());
  EXPECT_EQ(warm.iters, 0);
  EXPECT_NEAR(warm.reference_residual_norm, forcing_reference, forcing_reference * Real(1e-14))
      << "the reference is ||b-A(0)||, not the warm-start residual";

  // Exercise the inclusive absolute floor on an exactly representable boundary.  Reusing the
  // Helmholtz norm above would compare two deliberately different reduction algorithms whose
  // OpenMP summation orders may differ by one ULP, even though the mathematical fields agree.
  ApplyFn identity = [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); };
  MultiFab unit_rhs(*ba_, *dm_, 1, 0);
  MultiFab zero(*ba_, *dm_, 1, 1);
  unit_rhs.set_val(Real(1));
  zero.set_val(Real(0));
  const Real exact_unit_norm = static_cast<Real>(kN);  // sqrt(kN*kN), exact for kN=32.
  ASSERT_EQ(PureFieldAlgebra::norm(unit_rhs), exact_unit_norm);
  const SolveReport absolute = run_prepared(identity, zero, unit_rhs, bicgstab_krylov_method(),
                                            LinearOperatorProperties::general(), Real(0),
                                            exact_unit_norm, 50);
  EXPECT_TRUE(absolute.solved());
  EXPECT_EQ(absolute.iters, 0);
  EXPECT_EQ(absolute.reference_residual_norm, exact_unit_norm);
  EXPECT_EQ(absolute.residual_norm, exact_unit_norm);

  MultiFab below_floor(*ba_, *dm_, 1, 1);
  below_floor.set_val(Real(0));
  const SolveReport below = run_prepared(identity, below_floor, unit_rhs,
                                         bicgstab_krylov_method(),
                                         LinearOperatorProperties::general(), Real(0),
                                         exact_unit_norm / Real(2), 50);
  EXPECT_TRUE(below.solved());
  EXPECT_EQ(below.iters, 1);
  EXPECT_EQ(below.residual_norm, Real(0));

  MultiFab zero_rhs(*ba_, *dm_, 1, 0);
  MultiFab zero_solution(*ba_, *dm_, 1, 1);
  zero_rhs.set_val(Real(0));
  zero_solution.set_val(Real(0));
  const SolveReport zero_forcing = run_prepared(
      identity, zero_solution, zero_rhs, cg_krylov_method(),
      LinearOperatorProperties::symmetric_positive_definite(), Real(0), Real(1e-14), 4);
  EXPECT_TRUE(zero_forcing.solved());
  EXPECT_EQ(zero_forcing.iters, 0);
  EXPECT_EQ(zero_forcing.reference_residual_norm, Real(0));

  MultiFab nonzero_guess(*ba_, *dm_, 1, 1);
  nonzero_guess.set_val(Real(1));
  const SolveReport zero_reference_does_not_turn_relative_tolerance_absolute = run_prepared(
      identity, nonzero_guess, zero_rhs, cg_krylov_method(),
      LinearOperatorProperties::symmetric_positive_definite(), Real(0.5), Real(1e-14), 4);
  EXPECT_TRUE(zero_reference_does_not_turn_relative_tolerance_absolute.solved());
  EXPECT_GT(zero_reference_does_not_turn_relative_tolerance_absolute.iters, 0);
  EXPECT_EQ(zero_reference_does_not_turn_relative_tolerance_absolute.reference_residual_norm,
            Real(0));
  EXPECT_LE(zero_reference_does_not_turn_relative_tolerance_absolute.residual_norm, Real(1e-14));

  MultiFab tiny_rhs(*ba_, *dm_, 1, 0);
  MultiFab tiny_solution(*ba_, *dm_, 1, 1);
  tiny_rhs.set_val(Real(1e-16));
  tiny_solution.set_val(Real(0));
  const SolveReport tiny_forcing = run_prepared(
      identity, tiny_solution, tiny_rhs, cg_krylov_method(),
      LinearOperatorProperties::symmetric_positive_definite(), Real(0), Real(1e-14), 4);
  EXPECT_TRUE(tiny_forcing.solved());
  EXPECT_EQ(tiny_forcing.iters, 0);
}

TEST_F(GenericKrylov, true_residual_report_is_scale_safe_for_an_extreme_warm_start) {
  ApplyFn identity = [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); };
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab warm(*ba_, *dm_, 1, 1);
  rhs.set_val(Real(1e-200));
  warm.set_val(Real(1e200));

  const SolveReport report = run_prepared(identity, warm, rhs, cg_krylov_method(),
                                          LinearOperatorProperties::symmetric_positive_definite(),
                                          Real(0), Real(4e201), 4);

  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
  EXPECT_TRUE(std::isfinite(report.reference_residual_norm));
  EXPECT_TRUE(std::isfinite(report.residual_norm));
  EXPECT_GT(report.residual_norm, Real(1e201));
  EXPECT_LE(report.residual_norm, Real(4e201));
}

TEST_F(GenericKrylov, final_true_residual_promotes_an_exhausted_iteration_budget) {
  ApplyFn identity = [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); };
  MultiFab rhs(*ba_, *dm_, 1, 0);
  MultiFab iterate(*ba_, *dm_, 1, 1);
  rhs.set_val(Real(1));
  PureFieldAlgebra::zero_valid(iterate);

  // Richardson reaches the exact identity solution on its last permitted step. The builtin
  // deliberately publishes only a terminal candidate there; the common provider-independent
  // wrapper owns the single true-residual matvec and must promote that candidate to Solved.
  const SolveReport report =
      run_prepared(identity, iterate, rhs, richardson_krylov_method(Real(1)),
                   LinearOperatorProperties::general(), Real(1e-14), Real(0), 1);

  EXPECT_TRUE(report.solved());
  EXPECT_EQ(report.iters, 1);
  EXPECT_EQ(report.residual_norm, Real(0));
  EXPECT_EQ(max_abs_diff(iterate, rhs), Real(0));
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
  KrylovFootprint footprint{1, 1, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(x);
  PreparedAffineLinearProblem problem(
      x, *A_, PreparedLinearPreconditioner::identity(), LinearOperatorProperties::general(),
      footprint, PreparedNullspacePolicy::nonsingular(), [&snapshot]() { return snapshot; });
  const PreparedKrylovMethod method = gmres_krylov_method(10);
  KrylovWorkspace workspace(x, method, footprint);
  const std::size_t unbound_allocations = workspace.allocation_count();
  EXPECT_EQ(workspace.scalar_value_count(), 282u);
  EXPECT_EQ(workspace.collective_value_count(), 748u);
  const KrylovControls controls{method, kRelTol, Real(0), 500};

  problem.prepare(snapshot);
  workspace.bind(problem);
  const std::size_t allocations = workspace.allocation_count();
  EXPECT_EQ(allocations, unbound_allocations + 1u);
  const std::size_t halo_resources = x.halo_cache().exchange_pool_size();
  if (n_ranks() > 1)
    EXPECT_GT(halo_resources, 0u);
  snapshot.revision += 1;
  EXPECT_THROW((void)solve_prepared_affine(problem, workspace, x, *rhs_, controls),
               std::logic_error);

  snapshot.resources[0] += 1;
  problem.prepare(snapshot);
  workspace.bind(problem);
  const AllocationEventStats before_hot_solve = allocation_event_stats();
  const SolveReport first = solve_prepared_affine(problem, workspace, x, *rhs_, controls);
  const AllocationEventStats after_hot_solve = allocation_event_stats();
  EXPECT_TRUE(first.solved());
  EXPECT_GT(first.iters, 0);
  EXPECT_EQ(after_hot_solve, before_hot_solve)
      << "prepared GMRES must not allocate Fab or communication storage in its hot solve";
  x.set_val(0.0);
  snapshot.revision += 1;
  problem.prepare(snapshot);
  workspace.bind(problem);
  const SolveReport second = solve_prepared_affine(problem, workspace, x, *rhs_, controls);
  EXPECT_TRUE(second.solved());
  EXPECT_EQ(workspace.allocation_count(), allocations);
  EXPECT_EQ(x.halo_cache().exchange_pool_size(), halo_resources);
}

TEST_F(GenericKrylov, prepared_hot_solves_allocate_no_fab_or_communication_storage) {
  const Real h = geom_->dx();
  const Real richardson_omega = Real(1) / (Real(1) + kAlpha * Real(8) / (h * h));
  struct Route {
    TestKrylovFamily family;
    PreparedKrylovMethod method;
    bool preconditioned;
  };
  // The first four routes cover every solver loop.  The final two additionally cover the only
  // preconditionable loops; CG and Richardson intentionally reject a preconditioner footprint.
  const std::array<Route, 6> routes{{
      {TestKrylovFamily::kCg, cg_krylov_method(), false},
      {TestKrylovFamily::kBicgstab, bicgstab_krylov_method(), false},
      {TestKrylovFamily::kGmres, gmres_krylov_method(8), false},
      {TestKrylovFamily::kRichardson, richardson_krylov_method(richardson_omega), false},
      {TestKrylovFamily::kBicgstab, bicgstab_krylov_method(), true},
      {TestKrylovFamily::kGmres, gmres_krylov_method(8), true},
  }};

  for (const Route& route : routes) {
    MultiFab iterate(*ba_, *dm_, 1, 1);
    PureFieldAlgebra::zero_valid(iterate);
    PreparedLinearPreconditioner preconditioner =
        route.preconditioned ? affine_scaled_identity_preconditioner(iterate, Real(0.75), Real(2.5))
                             : PreparedLinearPreconditioner::identity();
    const KrylovFootprint footprint{iterate.ncomp(), iterate.n_grow(), route.preconditioned};
    OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
    const LinearOperatorProperties properties =
        route.family == TestKrylovFamily::kCg
            ? LinearOperatorProperties::symmetric_positive_definite()
            : LinearOperatorProperties::general();
    PreparedAffineLinearProblem problem(iterate, *A_, std::move(preconditioner), properties,
                                        footprint, PreparedNullspacePolicy::nonsingular(),
                                        [&snapshot]() { return snapshot; });
    KrylovWorkspace workspace(iterate, route.method, footprint);
    const KrylovControls controls{route.method, Real(1e-6), Real(0), 5000};

    problem.prepare(snapshot);
    workspace.bind(problem);
    for (int repetition = 0; repetition < 2; ++repetition) {
      const AllocationEventStats before_hot_solve = allocation_event_stats();
      const std::uint64_t consensus_calls_before = exact_consensus_dynamic_storage_calls();
      const SolveReport report =
          solve_prepared_affine(problem, workspace, iterate, *rhs_, controls);
      const std::uint64_t consensus_calls_after = exact_consensus_dynamic_storage_calls();
      const AllocationEventStats after_hot_solve = allocation_event_stats();

      EXPECT_TRUE(report.solved())
          << "method=" << route.method.identity() << " preconditioned=" << route.preconditioned
          << " repetition=" << repetition << " reason=" << report.reason;
      EXPECT_EQ(after_hot_solve, before_hot_solve)
          << "after prepare+bind, the hot solve must not allocate PoPS Fab or communication storage"
          << " method=" << route.method.identity() << " preconditioned=" << route.preconditioned
          << " repetition=" << repetition;
      EXPECT_EQ(consensus_calls_after, consensus_calls_before)
          << "after prepare+bind, solve must not call the exact-consensus helper that owns "
             "dynamic host vectors"
          << " method=" << route.method.identity() << " preconditioned=" << route.preconditioned
          << " repetition=" << repetition;

      if (repetition == 0) {
        PureFieldAlgebra::zero_valid(iterate);
        ++snapshot.revision;
        problem.prepare(snapshot);
        workspace.bind(problem);
      }
    }
  }
}

TEST_F(GenericKrylov, iterate_and_rhs_must_not_alias) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  KrylovFootprint footprint{1, 1, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(x);
  PreparedAffineLinearProblem problem(x, *A_, PreparedLinearPreconditioner::identity(),
                                      LinearOperatorProperties::symmetric_positive_definite(),
                                      footprint, PreparedNullspacePolicy::nonsingular(),
                                      [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(x, cg_krylov_method(), footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  EXPECT_THROW(
      (void)solve_prepared_affine(problem, workspace, x, x,
                                  KrylovControls{cg_krylov_method(), kRelTol, Real(0), 10}),
      std::invalid_argument);
}

TEST_F(GenericKrylov, extension_apply_mutation_is_refused_before_result_consumption) {
  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(0.0);
  KrylovFootprint footprint{1, 1, false};
  OperatorEvaluationSnapshot snapshot = snapshot_for(x);
  bool mutate_during_apply = false;
  ApplyFn extension_apply = [&snapshot, &mutate_during_apply](MultiFab& out, const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
    if (mutate_during_apply)
      ++snapshot.revision;
  };
  PreparedAffineLinearProblem problem(
      x,
      externally_observed_test_operator("pops.test.generic-krylov.apply-mutation",
                                        std::move(extension_apply)),
      PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(x, cg_krylov_method(), footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  mutate_during_apply = true;

  EXPECT_THROW(
      (void)solve_prepared_affine(problem, workspace, x, *rhs_,
                                  KrylovControls{cg_krylov_method(), kRelTol, Real(0), 10}),
      std::logic_error);
}

TEST_F(GenericKrylov, rank_local_snapshot_drift_is_refused_collectively_at_all_safe_boundaries) {
  if (n_ranks() < 2)
    GTEST_SKIP() << "this regression requires a real multi-rank CTest route";

  MultiFab x(*ba_, *dm_, 1, 1);
  x.set_val(Real(0));
  const KrylovFootprint footprint{1, 1, false};
  const OperatorEvaluationSnapshot expected = snapshot_for(x);
  OperatorEvaluationSnapshot observed = expected;
  bool mutate_during_resource_freeze = false;
  bool mutate_during_apply = false;
  const ApplyFn extension_apply = [&observed, &mutate_during_apply](MultiFab& out,
                                                                    const MultiFab& in) {
    PureFieldAlgebra::copy(out, in);
    if (mutate_during_apply && my_rank() == 0)
      ++observed.revision;
  };
  PreparedAffineLinearProblem problem(
      x,
      externally_observed_test_operator("pops.test.generic-krylov.rank-local-snapshot-drift",
                                        extension_apply),
      PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&observed]() { return observed; },
      [&observed, &mutate_during_resource_freeze]() {
        if (mutate_during_resource_freeze && my_rank() == 0)
          ++observed.revision;
      });
  KrylovWorkspace workspace(x, cg_krylov_method(), footprint);
  const KrylovControls controls{cg_krylov_method(), kRelTol, Real(0), 4};

  const auto expect_collective_logic_error = [](const auto& operation,
                                                const char* expected_message) {
    bool threw = false;
    std::string message;
    try {
      operation();
    } catch (const std::logic_error& error) {
      threw = true;
      message = error.what();
    }
    EXPECT_TRUE(threw);
    EXPECT_EQ(message, expected_message);
    EXPECT_EQ(all_reduce_sum(static_cast<long>(threw)), static_cast<long>(n_ranks()))
        << "every rank must leave the collective snapshot gate through the same exception";
  };

  if (my_rank() == 0)
    ++observed.revision;
  expect_collective_logic_error(
      [&]() { problem.prepare(expected); },
      "operator snapshot changed before preparation on at least one communicator rank");

  OperatorEvaluationSnapshot rank_local_expected = expected;
  if (my_rank() == 0)
    ++rank_local_expected.revision;
  observed = rank_local_expected;
  expect_collective_logic_error(
      [&]() { problem.prepare(rank_local_expected); },
      "operator snapshot differs across communicator ranks before preparation");

  observed = expected;
  mutate_during_resource_freeze = true;
  expect_collective_logic_error(
      [&]() { problem.prepare(expected); },
      "operator snapshot changed during resource preparation on at least one communicator rank");

  observed = expected;
  mutate_during_resource_freeze = false;
  problem.prepare(expected);
  workspace.bind(problem);
  if (my_rank() == 0)
    ++observed.revision;
  expect_collective_logic_error(
      [&]() { (void)solve_prepared_affine(problem, workspace, x, *rhs_, controls); },
      "operator snapshot mutated after preparation on at least one communicator rank");

  observed = expected;
  problem.prepare(expected);
  workspace.bind(problem);
  mutate_during_apply = true;
  expect_collective_logic_error(
      [&]() { (void)solve_prepared_affine(problem, workspace, x, *rhs_, controls); },
      "operator snapshot mutated after preparation on at least one communicator rank");
}
