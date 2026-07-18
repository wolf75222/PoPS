// Multigrille geometrique : convergence rapide et quasi independante du maillage
// sur des solutions manufacturees (Dirichlet et periodique), precision O(dx^2).

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace pops;

static constexpr double kPi = 3.14159265358979323846;

namespace {

class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { comm_init(); }
  void TearDown() override { comm_finalize(); }
};

[[maybe_unused]] const ::testing::Environment* const kCommEnvironment =
    ::testing::AddGlobalTestEnvironment(new CommEnvironment);

int first_boundary_residual_iteration = -1;

void noop_boundary_prepare_residual(int, const MultiFab&, MultiFab&, const Geometry&,
                                    const FieldBoundaryExecutionContext& context) {
  if (first_boundary_residual_iteration < 0)
    first_boundary_residual_iteration = context.point.iteration;
}
void noop_boundary_prepare_jvp(int, const MultiFab&, const MultiFab&, MultiFab&, const Geometry&,
                               const FieldBoundaryExecutionContext&) {}
void noop_boundary_residual(int, const MultiFab&, MultiFab&, const Geometry&,
                            const FieldBoundaryExecutionContext&) {}
void noop_boundary_jvp(int, const MultiFab&, const MultiFab&, MultiFab&, const Geometry&,
                       const FieldBoundaryExecutionContext&) {}

GeometricMG make_replicated_geometric_mg(int n = 8) {
  const Box2D domain = Box2D::from_extents(n, n);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  BCRec boundary;
  boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = BCType::Dirichlet;
  return GeometricMG(geometry, BoxArray::from_domain(domain, n), boundary, {},
                     FieldDistribution::Replicated);
}

void set_isometric_rank_permutation(MultiFab& field) {
  field.set_val(Real(0));
  field.sync_host();
  const int local = field.local_index_of(0);
  ASSERT_GE(local, 0);
  Array4 values = field.fab(local).array();
  const Box2D box = field.box(local);
  const int i = my_rank() == 0 ? box.lo[0] : box.lo[0] + 1;
  values(i, box.lo[1], 0) = Real(1);
}

template <class Operation>
void expect_uniform_collective_rejection(Operation&& operation) {
  bool rejected = false;
  std::string message;
  try {
    std::forward<Operation>(operation)();
  } catch (const std::exception& error) {
    rejected = true;
    message = error.what();
  }

  const long rejected_ranks = all_reduce_sum(rejected ? 1L : 0L);
  const bool same_message = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("geometric-mg-collective-rejection"), std::string_view(message)}});
  EXPECT_EQ(rejected_ranks, n_ranks());
  EXPECT_TRUE(same_message);
  EXPECT_FALSE(message.empty());
}

std::string canonical_valid_bytes(const MultiFab& field) {
  field.sync_host();
  std::string bytes;
  for (int global = 0; global < field.box_array().size(); ++global) {
    const int local = field.local_index_of(global);
    if (local < 0)
      return {};
    const ConstArray4 values = field.fab(local).const_array();
    const Box2D box = field.box(local);
    for (int component = 0; component < field.ncomp(); ++component) {
      for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const Real value = values(i, j, component);
          bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
        }
      }
    }
  }
  return bytes;
}

bool published_replicas_agree_exactly(const MultiFab& field) {
  const std::string bytes = canonical_valid_bytes(field);
  return !bytes.empty() &&
         all_ranks_agree_exact_ordered_byte_pairs(
             {{std::string_view("geometric-mg-published-phi"), std::string_view(bytes)}});
}

}  // namespace

TEST(GeometricMgCollectiveContract, MpiRouteInitializesRequestedCommunicator) {
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    ASSERT_EQ(n_ranks(), std::atoi(expected_ranks))
        << "the MPI CTest route must initialize the requested communicator";
  else if (n_ranks() == 1)
    GTEST_SKIP() << "the serial registration has no remote rank";
}

TEST(GeometricMgCollectiveContract, RejectsRankDivergentValidControls) {
  if (n_ranks() != 2)
    GTEST_SKIP() << "the collective contract regression is registered at exactly two MPI ranks";

  GeometricMG mg = make_replicated_geometric_mg();
  mg.rhs().set_val(Real(0));
  mg.phi().set_val(Real(0));
  const Real rel_tol = my_rank() == 0 ? Real(1e-8) : Real(1e-6);
  expect_uniform_collective_rejection(
      [&] { (void)mg.solve(rel_tol, /*max_cycles=*/4, /*abs_tol=*/Real(0)); });
}

TEST(GeometricMgCollectiveContract, ConvertsOneRankInvalidControlsToOneUniformFailure) {
  if (n_ranks() != 2)
    GTEST_SKIP() << "the collective contract regression is registered at exactly two MPI ranks";

  GeometricMG mg = make_replicated_geometric_mg();
  mg.rhs().set_val(Real(0));
  mg.phi().set_val(Real(0));
  const Real rel_tol = my_rank() == 0 ? Real(1e-8) : Real(0);
  expect_uniform_collective_rejection(
      [&] { (void)mg.solve(rel_tol, /*max_cycles=*/4, /*abs_tol=*/Real(0)); });
}

TEST(GeometricMgCollectiveContract,
     BoundaryNewtonConfigurationRollbackAndClearPreserveCollectiveLifecycle) {
  if (n_ranks() != 2)
    GTEST_SKIP() << "the prepared-cache lifecycle regression is registered at two MPI ranks";

  GeometricMG mg = make_replicated_geometric_mg();
  mg.rhs().set_val(Real(0));
  mg.phi().set_val(Real(0));
  CompiledFieldBoundaryKernel kernel{"mpi-noop-iteration-boundary",
                                     "mpi-noop-iteration-boundary-residual",
                                     "mpi-noop-iteration-boundary-jvp",
                                     noop_boundary_prepare_residual,
                                     noop_boundary_prepare_jvp,
                                     noop_boundary_residual,
                                     noop_boundary_jvp,
                                     true};
  mg.set_boundary_kernel(kernel, FieldBoundaryExecutionContext{});
  FieldNewtonOptions options;
  options.restart = 5;
  options.max_iterations = 2;
  options.linear_max_iterations = 20;
  mg.set_field_newton_options(options);
  const auto generation = mg.boundary_newton_cache_generation();
  const auto allocation_count = mg.boundary_newton_cache_allocation_count();
  ASSERT_GT(generation, 0u);
  ASSERT_GT(allocation_count, 0u);

  FieldBoundaryExecutionContext divergent_context;
  if (my_rank() == 1)
    divergent_context.state_count = 1;  // invalid without the required state pointer tables
  expect_uniform_collective_rejection([&] { mg.set_boundary_kernel(kernel, divergent_context); });
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), allocation_count);
  const SolveReport after_rollback = mg.solve_boundary_newton(options);
  EXPECT_TRUE(after_rollback.solved());

  EXPECT_NO_THROW(mg.clear_boundary_kernel());
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), 0u);
  const SolveReport cleared = mg.solve_boundary_newton(options);
  EXPECT_EQ(cleared.status, SolveStatus::kCapabilityFailure);
}

TEST(GeometricMgCollectiveContract, RejectsDivergentReplicatedRhsBeforeIteration) {
  if (n_ranks() != 2)
    GTEST_SKIP() << "the replica contract regression is registered at exactly two MPI ranks";

  {
    GeometricMG mg = make_replicated_geometric_mg();
    mg.rhs().set_val(my_rank() == 0 ? Real(1) : Real(2));
    mg.phi().set_val(Real(0));
    expect_uniform_collective_rejection(
        [&] { (void)mg.solve(Real(1e-8), /*max_cycles=*/4, /*abs_tol=*/Real(0)); });
  }

  // Equal max norms and equal sums are not replica equality certificates.
  {
    GeometricMG mg = make_replicated_geometric_mg();
    set_isometric_rank_permutation(mg.rhs());
    mg.phi().set_val(Real(0));
    expect_uniform_collective_rejection(
        [&] { (void)mg.solve(Real(1e-8), /*max_cycles=*/4, /*abs_tol=*/Real(0)); });
  }
}

TEST(GeometricMgCollectiveContract, RejectsDivergentReplicatedWarmStartBeforeIteration) {
  if (n_ranks() != 2)
    GTEST_SKIP() << "the replica contract regression is registered at exactly two MPI ranks";

  GeometricMG mg = make_replicated_geometric_mg();
  mg.rhs().set_val(Real(0));
  set_isometric_rank_permutation(mg.phi());
  expect_uniform_collective_rejection(
      [&] { (void)mg.solve(Real(1e-8), /*max_cycles=*/4, /*abs_tol=*/Real(0)); });
}

TEST(GeometricMgCollectiveContract, PublishesOneExactReplicatedPhi) {
  if (n_ranks() != 2)
    GTEST_SKIP() << "the replica publication regression is registered at exactly two MPI ranks";

  constexpr int n = 8;
  GeometricMG mg = make_replicated_geometric_mg(n);
  const Geometry& geometry = mg.geom();
  for (int li = 0; li < mg.rhs().local_size(); ++li) {
    Array4 rhs = mg.rhs().fab(li).array();
    const Box2D box = mg.rhs().box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const Real exact = std::sin(kPi * geometry.x_cell(i)) * std::sin(kPi * geometry.y_cell(j));
        rhs(i, j, 0) = Real(-2) * Real(kPi * kPi) * exact;
      }
    }
  }
  mg.phi().set_val(Real(0));

  const int cycles = mg.solve(Real(1e-8), /*max_cycles=*/100, /*abs_tol=*/Real(0));
  ASSERT_TRUE(mg.last_solve_report().solved()) << mg.last_solve_report().status_name();
  EXPECT_GT(cycles, 0);
  EXPECT_EQ(all_reduce_min(static_cast<long>(cycles)), all_reduce_max(static_cast<long>(cycles)));
  EXPECT_GT(norm_inf(mg.phi()), Real(0));
  EXPECT_TRUE(published_replicas_agree_exactly(mg.phi()));
}

static void expect_zero_probe_forcing_scale(const BCRec& bc) {
  constexpr int n = 16;
  const Box2D domain = Box2D::from_extents(n, n);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, n);

  GeometricMG zero_start(geometry, boxes, bc);
  zero_start.rhs().set_val(Real(0));
  zero_start.phi().set_val(Real(0));
  const Real forcing_norm = zero_start.current_residual();
  ASSERT_TRUE(std::isfinite(static_cast<double>(forcing_norm)));
  ASSERT_GT(forcing_norm, Real(2));  // distinguishes R(0) from the zero-RHS fallback scale 1
  EXPECT_EQ(zero_start.solve(Real(1), /*max_cycles=*/4), 0);
  const SolveReport& zero_report = zero_start.last_solve_report();
  EXPECT_TRUE(zero_report.solved());
  EXPECT_NEAR(zero_report.rel_residual, Real(1), Real(1e-14));

  GeometricMG warm_start(geometry, boxes, bc);
  warm_start.rhs().set_val(Real(0));
  warm_start.phi().set_val(Real(0.375));
  const Real warm_residual = warm_start.current_residual();
  ASSERT_TRUE(std::isfinite(static_cast<double>(warm_residual)));
  ASSERT_GT(warm_residual, Real(0));
  const Real expected_relative = warm_residual / forcing_norm;
  const Real warm_tolerance =
      expected_relative * (Real(1) + Real(128) * std::numeric_limits<Real>::epsilon());
  EXPECT_EQ(warm_start.solve(warm_tolerance, /*max_cycles=*/4), 0);
  const SolveReport& warm_report = warm_start.last_solve_report();
  EXPECT_TRUE(warm_report.solved());
  EXPECT_NEAR(warm_report.rel_residual, expected_relative, Real(1e-14));

  EXPECT_EQ(warm_start.solve_robust(warm_tolerance, /*max_cycles=*/4), 0);
  const SolveReport& robust_report = warm_start.last_solve_report();
  EXPECT_TRUE(robust_report.solved());
  EXPECT_NEAR(robust_report.rel_residual, expected_relative, Real(1e-14));
}

// Resout lap(phi)=f pour phi_ex donne, renvoie (cycles, erreur_inf).
template <class PhiEx, class RhsF>
static void solve_case(int n, const BCRec& bc, bool periodic, PhiEx phi_ex, RhsF rhs_f, int& cycles,
                       double& err) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);

  GeometricMG mg(geom, ba, bc);
  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom, rhs_f](int i, int j) {
    af(i, j, 0) = rhs_f(geom.x_cell(i), geom.y_cell(j));
  });
  mg.phi().set_val(0.0);

  const Real r0 = mg.current_residual();
  Real rn = r0;
  cycles = 0;
  while (rn > 1e-9 * r0 && cycles < 50) {
    mg.vcycle();
    rn = mg.current_residual();
    ++cycles;
  }

  // pour le cas periodique, la solution est definie a une constante pres
  Fab2D& p = mg.phi().fab(0);
  if (periodic) {
    Real mean = sum(mg.phi()) / static_cast<Real>(dom.num_cells());
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        p(i, j, 0) -= mean;
  }
  err = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      err = std::max(err, std::fabs(p(i, j, 0) - phi_ex(geom.x_cell(i), geom.y_cell(j))));
}

TEST(GeometricMgTest, rejects_invalid_field_distribution) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, 8);

  EXPECT_THROW(GeometricMG(geometry, boxes, BCRec{}, {}, static_cast<FieldDistribution>(0xff)),
               std::invalid_argument);
}

// --- Dirichlet : phi = sin(pi x) sin(pi y), lap phi = -2 pi^2 phi ---
TEST(GeometricMgTest, dirichlet_converges_mesh_independent_second_order) {
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  auto pe = [](double x, double y) { return std::sin(kPi * x) * std::sin(kPi * y); };
  auto fr = [&](double x, double y) { return -2 * kPi * kPi * pe(x, y); };

  int c32 = 0, c64 = 0;
  double e32 = 0, e64 = 0;
  solve_case(32, bc, false, pe, fr, c32, e32);
  solve_case(64, bc, false, pe, fr, c64, e64);
  std::printf("Dirichlet : c32=%d e32=%.2e | c64=%d e64=%.2e\n", c32, e32, c64, e64);
  EXPECT_TRUE(c64 <= 25) << "dir_converged_fast: c64=" << c64;
  EXPECT_TRUE(std::abs(c64 - c32) <= 5) << "dir_mesh_independent: c32=" << c32 << " c64=" << c64;
  EXPECT_TRUE(e64 < 5e-3) << "dir_accurate: e64=" << e64;
  EXPECT_TRUE(e64 < e32) << "dir_second_order: e32=" << e32
                         << " e64=" << e64;  // erreur baisse en raffinant
}

// --- periodique : phi = sin(2 pi x) sin(2 pi y), lap phi = -8 pi^2 phi ---
TEST(GeometricMgTest, periodic_converges_accurate) {
  BCRec bc;  // periodique par defaut sur les 4 faces
  auto pe = [](double x, double y) { return std::sin(2 * kPi * x) * std::sin(2 * kPi * y); };
  auto fr = [&](double x, double y) { return -8 * kPi * kPi * pe(x, y); };

  int c64 = 0;
  double e64 = 0;
  solve_case(64, bc, true, pe, fr, c64, e64);
  std::printf("Periodique : c64=%d e64=%.2e\n", c64, e64);
  EXPECT_TRUE(c64 <= 30) << "per_converged: c64=" << c64;
  EXPECT_TRUE(e64 < 5e-3) << "per_accurate: e64=" << e64;
}

TEST(GeometricMgTest, periodic_mean_zero_warm_start_exits_without_mutation) {
  constexpr int n = 32;
  constexpr Real rel_tol = Real(1e-8);

  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba = BoxArray::from_domain(dom, n);
  GeometricMG mg(geom, ba, BCRec{});  // periodic on all four faces

  // A periodic Laplacian is solvable only on the mean-zero subspace.  Subtract the
  // discrete mean explicitly so this test remains valid for every even resolution,
  // independently of floating-point summation symmetry.
  Fab2D& rhs = mg.rhs().fab(0);
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double x = geom.x_cell(i);
      const double y = geom.y_cell(j);
      rhs(i, j, 0) = std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
    }
  const Real rhs_mean = sum(mg.rhs()) / static_cast<Real>(dom.num_cells());
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      rhs(i, j, 0) -= rhs_mean;
  mg.phi().set_val(Real(0));

  const int first_cycles = mg.solve(rel_tol, /*max_cycles=*/100);
  const SolveReport first = mg.last_solve_report();
  ASSERT_TRUE(first.solved()) << "status=" << first.status_name();
  ASSERT_GT(first_cycles, 0);

  device_fence();
  const MultiFab phi_before = mg.phi();

  // A second solve with the same RHS must publish a solved, zero-cycle report and preserve
  // the warm-start iterate bit-for-bit.  No explicit absolute floor is supplied: this catches
  // the old relative-criterion path that needlessly re-cycled an already converged state.
  const int second_cycles = mg.solve(rel_tol, /*max_cycles=*/100);
  const SolveReport second = mg.last_solve_report();
  ASSERT_TRUE(second.solved()) << "status=" << second.status_name();
  EXPECT_EQ(second_cycles, 0);
  EXPECT_EQ(second.iters, 0);
  EXPECT_LE(second.rel_residual, rel_tol);

  device_fence();
  Real max_delta = Real(0);
  for (int li = 0; li < mg.phi().local_size(); ++li) {
    const ConstArray4 before = phi_before.fab(li).const_array();
    const ConstArray4 after = mg.phi().fab(li).const_array();
    const Box2D valid = mg.phi().box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        max_delta = std::max(max_delta, std::fabs(after(i, j, 0) - before(i, j, 0)));
  }
  EXPECT_EQ(max_delta, Real(0));
}

TEST(GeometricMgTest, zero_forcing_requires_exact_zero_without_absolute_tolerance) {
  constexpr int n = 8;
  const Box2D domain = Box2D::from_extents(n, n);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geometry, boxes, bc);
  mg.rhs().set_val(Real(0));  // exact affine forcing R(0) = 0
  mg.phi().set_val(Real(1e-6));
  const Real initial_residual = mg.current_residual();
  ASSERT_GT(initial_residual, Real(0));
  ASSERT_LT(initial_residual, Real(1));

  // A unit fallback in the stop would accept this nonzero residual at rel_tol=1. The correct
  // zero-forcing threshold is exactly zero, so at least one V-cycle must be attempted.
  EXPECT_EQ(mg.solve(Real(1), /*max_cycles=*/1, /*abs_tol=*/Real(0)), 1);

  mg.phi().set_val(Real(1e-6));
  const Real reset_residual = mg.current_residual();
  EXPECT_EQ(mg.solve(Real(1e-8), /*max_cycles=*/1, /*abs_tol=*/Real(2) * reset_residual), 0);
  EXPECT_TRUE(mg.last_solve_report().solved());
  EXPECT_NEAR(mg.last_solve_report().rel_residual, reset_residual, Real(1e-14));

  mg.phi().set_val(Real(1e-6));
  EXPECT_NE(mg.solve_robust(Real(1), /*max_cycles=*/1, /*abs_tol=*/Real(0)), 0);
  mg.phi().set_val(Real(1e-6));
  const Real robust_residual = mg.current_residual();
  EXPECT_EQ(mg.solve_robust(Real(1e-8), /*max_cycles=*/1,
                            /*abs_tol=*/Real(2) * robust_residual),
            0);
  EXPECT_TRUE(mg.last_solve_report().solved());
  EXPECT_NEAR(mg.last_solve_report().rel_residual, robust_residual, Real(1e-14));
}

TEST(GeometricMgTest, inhomogeneous_dirichlet_uses_zero_probe_scale) {
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  bc.xlo_val = Real(1.0);
  bc.xhi_val = Real(-0.5);
  bc.ylo_val = Real(0.75);
  bc.yhi_val = Real(-1.25);
  expect_zero_probe_forcing_scale(bc);
}

TEST(GeometricMgTest, inhomogeneous_robin_uses_zero_probe_scale) {
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Robin;
  bc.xlo_alpha = bc.xhi_alpha = bc.ylo_alpha = bc.yhi_alpha = Real(1);
  bc.xlo_beta = bc.xhi_beta = bc.ylo_beta = bc.yhi_beta = Real(0.25);
  bc.xlo_val = Real(1.5);
  bc.xhi_val = Real(-0.75);
  bc.ylo_val = Real(0.5);
  bc.yhi_val = Real(-1.0);
  expect_zero_probe_forcing_scale(bc);
}

TEST(GeometricMgTest, nonfinite_rhs_and_residual_are_invalid_evaluations) {
  constexpr int n = 8;
  const Box2D domain = Box2D::from_extents(n, n);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, n);

  for (const Real invalid :
       {std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::infinity()}) {
    GeometricMG mg(geometry, boxes, BCRec{});
    mg.rhs().set_val(invalid);
    EXPECT_TRUE(std::isinf(static_cast<double>(norm_inf(mg.rhs()))));
    EXPECT_EQ(mg.solve(Real(1e-8), /*max_cycles=*/4), 0);
    const SolveReport& report = mg.last_solve_report();
    EXPECT_EQ(report.status, SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(report.action, SolveAction::kRejectAttempt);
    EXPECT_FALSE(report.solved());
    EXPECT_TRUE(std::isinf(static_cast<double>(report.rel_residual)));
  }

  GeometricMG invalid_iterate(geometry, boxes, BCRec{});
  invalid_iterate.rhs().set_val(Real(0));
  invalid_iterate.phi().set_val(std::numeric_limits<Real>::infinity());
  EXPECT_TRUE(std::isinf(static_cast<double>(invalid_iterate.current_residual())));
  EXPECT_EQ(invalid_iterate.solve(Real(1e-8), /*max_cycles=*/4), 0);
  EXPECT_EQ(invalid_iterate.last_solve_report().status, SolveStatus::kInvalidEvaluation);

  GeometricMG invalid_robust(geometry, boxes, BCRec{});
  invalid_robust.rhs().set_val(std::numeric_limits<Real>::quiet_NaN());
  EXPECT_EQ(invalid_robust.solve_robust(Real(1e-8), /*max_cycles=*/4), 0);
  EXPECT_EQ(invalid_robust.last_solve_report().status, SolveStatus::kInvalidEvaluation);
}

TEST(GeometricMgTest, rejects_nonfinite_or_out_of_domain_controls) {
  const Box2D domain = Box2D::from_extents(4, 4);
  GeometricMG mg(Geometry{domain, 0.0, 1.0, 0.0, 1.0}, BoxArray(std::vector<Box2D>{domain}),
                 BCRec{});
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW((void)mg.solve(nan, 4), std::invalid_argument);
  EXPECT_THROW((void)mg.solve(Real(1e-8), 0), std::invalid_argument);
  EXPECT_THROW((void)mg.solve(Real(1e-8), 4, nan), std::invalid_argument);
  EXPECT_THROW((void)mg.solve_robust(nan, 4), std::invalid_argument);
  EXPECT_THROW(mg.set_abs_tol(nan), std::invalid_argument);

  FieldNewtonOptions newton;
  newton.tolerance = std::numeric_limits<Real>::infinity();
  EXPECT_THROW(mg.set_field_newton_options(newton), std::invalid_argument);
  newton = FieldNewtonOptions{};
  newton.linear_tolerance = std::numeric_limits<Real>::infinity();
  EXPECT_THROW(mg.set_field_newton_options(newton), std::invalid_argument);
  newton = FieldNewtonOptions{};
  newton.restart = 51;
  EXPECT_NO_THROW(mg.set_field_newton_options(newton));
}

TEST(GeometricMgTest, nonlinear_boundary_snapshot_reuses_cache_with_opaque_stage_slot) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  GeometricMG mg(geometry, BoxArray(std::vector<Box2D>{domain}), bc);
  mg.phi().set_val(Real(0));
  mg.rhs().set_val(Real(1));

  CompiledFieldBoundaryKernel kernel{"noop-iteration-boundary",
                                     "noop-iteration-boundary-residual",
                                     "noop-iteration-boundary-jvp",
                                     noop_boundary_prepare_residual,
                                     noop_boundary_prepare_jvp,
                                     noop_boundary_residual,
                                     noop_boundary_jvp,
                                     true};
  FieldBoundaryExecutionContext context;
  context.point.clock_slot = 3;
  context.point.partition_slot = 5;
  context.point.stage_slot = 17;  // generated wire ids are not Runge--Kutta fractions
  context.point.step = 2;
  context.point.substep = 7;
  context.point.time = Real(0.25);
  context.point.dt = Real(0.01);
  mg.set_boundary_kernel(kernel, context);

  FieldNewtonOptions options;
  options.max_iterations = 3;
  options.linear_max_iterations = 80;
  options.restart = 12;
  EXPECT_EQ(mg.boundary_newton_cache_generation(), 0u);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), 0u);
  EXPECT_NO_THROW(mg.prepare_boundary_newton(options));
  const auto generation = mg.boundary_newton_cache_generation();
  const auto allocation_count = mg.boundary_newton_cache_allocation_count();
  EXPECT_GT(generation, 0u);
  EXPECT_GT(allocation_count, 0u);

  SolveReport first_report;
  first_boundary_residual_iteration = -1;
  EXPECT_NO_THROW(first_report = mg.solve_boundary_newton(options));
  EXPECT_EQ(first_boundary_residual_iteration, 0);
  EXPECT_NE(first_report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), allocation_count);

  // A new logical evaluation point and different per-call stopping controls do not change storage.
  // Only the field layout or GMRES restart may rebuild this cache.
  mg.phi().set_val(Real(0));
  mg.rhs().set_val(Real(1));
  context.point.stage_slot = 29;
  context.point.time = Real(0.5);
  mg.set_boundary_context(context);
  options.max_iterations = 2;
  options.linear_tolerance = Real(5e-4);
  options.linear_max_iterations = 64;

  SolveReport second_report;
  first_boundary_residual_iteration = -1;
  EXPECT_NO_THROW(second_report = mg.solve_boundary_newton(options));
  EXPECT_EQ(first_boundary_residual_iteration, 0);
  EXPECT_NE(second_report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), allocation_count);

  // A repeated direct call without a fresh context must not inherit the previous Newton iteration.
  mg.phi().set_val(Real(0));
  mg.rhs().set_val(Real(1));
  first_boundary_residual_iteration = -1;
  SolveReport repeated_report;
  EXPECT_NO_THROW(repeated_report = mg.solve_boundary_newton(options));
  EXPECT_EQ(first_boundary_residual_iteration, 0);
  EXPECT_NE(repeated_report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), allocation_count);

  // A staged replacement that fails its collective configuration contract must leave the last
  // valid kernel, context, cache, and generation intact. In particular, solve may not silently
  // continue with the invalid context that was temporarily inspected by the control gate.
  FieldBoundaryExecutionContext invalid_context = context;
  invalid_context.state_count = 1;
  invalid_context.states = nullptr;
  invalid_context.state_distributions = nullptr;
  EXPECT_THROW(mg.set_boundary_kernel(kernel, invalid_context), std::invalid_argument);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), allocation_count);
  mg.phi().set_val(Real(0));
  mg.rhs().set_val(Real(1));
  SolveReport after_failed_reconfiguration;
  EXPECT_NO_THROW(after_failed_reconfiguration = mg.solve_boundary_newton(options));
  EXPECT_NE(after_failed_reconfiguration.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);

  // Restart changes the GMRES basis shape and is therefore the one per-call control that rebuilds
  // storage. The replacement remains a single persistent cache, not an accumulating cache family.
  mg.phi().set_val(Real(0));
  mg.rhs().set_val(Real(1));
  options.restart = 8;
  EXPECT_THROW((void)mg.solve_boundary_newton(options), std::logic_error);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_NO_THROW(mg.prepare_boundary_newton(options));
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation + 1);
  SolveReport resized_report;
  EXPECT_NO_THROW(resized_report = mg.solve_boundary_newton(options));
  EXPECT_NE(resized_report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation + 1);
  EXPECT_NE(mg.boundary_newton_cache_allocation_count(), allocation_count);
}

TEST(GeometricMgTest, nonlinear_boundary_configuration_materializes_before_noarg_solve) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  GeometricMG mg(geometry, BoxArray(std::vector<Box2D>{domain}), bc);
  mg.phi().set_val(Real(0));
  mg.rhs().set_val(Real(0));

  CompiledFieldBoundaryKernel kernel{"noop-iteration-boundary-control",
                                     "noop-iteration-boundary-control-residual",
                                     "noop-iteration-boundary-control-jvp",
                                     noop_boundary_prepare_residual,
                                     noop_boundary_prepare_jvp,
                                     noop_boundary_residual,
                                     noop_boundary_jvp,
                                     true};
  mg.set_boundary_kernel(kernel, FieldBoundaryExecutionContext{});
  EXPECT_EQ(mg.boundary_newton_cache_generation(), 0u);

  FieldNewtonOptions options;
  options.max_iterations = 2;
  options.linear_max_iterations = 40;
  options.restart = 7;
  EXPECT_NO_THROW(mg.set_field_newton_options(options));
  const auto generation = mg.boundary_newton_cache_generation();
  const auto allocation_count = mg.boundary_newton_cache_allocation_count();
  EXPECT_GT(generation, 0u);
  EXPECT_GT(allocation_count, 0u);

  EXPECT_NO_THROW(mg.solve());
  EXPECT_EQ(mg.boundary_newton_cache_generation(), generation);
  EXPECT_EQ(mg.boundary_newton_cache_allocation_count(), allocation_count);

  // Configuration order is not semantic: installing the controls first leaves no hidden solve
  // work. The later kernel installation observes that both halves are present and completes the
  // same collective materialization immediately.
  GeometricMG controls_first(geometry, BoxArray(std::vector<Box2D>{domain}), bc);
  controls_first.phi().set_val(Real(0));
  controls_first.rhs().set_val(Real(0));
  EXPECT_NO_THROW(controls_first.set_field_newton_options(options));
  EXPECT_EQ(controls_first.boundary_newton_cache_generation(), 0u);
  EXPECT_NO_THROW(controls_first.set_boundary_kernel(kernel, FieldBoundaryExecutionContext{}));
  const auto controls_first_generation = controls_first.boundary_newton_cache_generation();
  EXPECT_GT(controls_first_generation, 0u);
  EXPECT_GT(controls_first.boundary_newton_cache_allocation_count(), 0u);
  EXPECT_NO_THROW(controls_first.solve());
  EXPECT_EQ(controls_first.boundary_newton_cache_generation(), controls_first_generation);
  EXPECT_NO_THROW(controls_first.clear_boundary_kernel());
  EXPECT_EQ(controls_first.boundary_newton_cache_allocation_count(), 0u);
  const SolveReport cleared = controls_first.solve_boundary_newton(options);
  EXPECT_EQ(cleared.status, SolveStatus::kCapabilityFailure);
}

TEST(GeometricMgTest, nullspace_compatibility_rejects_nonfinite_moment) {
  const Box2D domain = Box2D::from_extents(4, 4);
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping mapping(1, 1);
  MultiFab rhs(boxes, mapping, 1, 0);
  rhs.set_val(std::numeric_limits<Real>::quiet_NaN());
  const FieldNullspacePlan plan = constant_mean_zero_nullspace("nonfinite-test", "unit-test");

  try {
    (void)require_field_nullspace_compatible(rhs, plan);
    FAIL() << "a non-finite nullspace compatibility moment must be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("non-finite compatibility moment"), std::string::npos)
        << error.what();
  }
}
