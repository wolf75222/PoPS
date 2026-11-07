#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/amr/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

using pops::BoundaryTopology;
using pops::Box;
using pops::CompiledFieldBoundaryKernel;
using pops::Extent;
using pops::ExecutionLane;
using pops::FieldBoundaryExecutionContext;
using pops::FieldBoundaryFailure;
using pops::Geometry;
using pops::FieldNullspacePlan;
using pops::FieldNewtonOptions;
using pops::Index;
using pops::PhysicalBoundaryConditions;
using pops::PhysicalBoundaryFace;
using pops::PhysicalBoundaryKind;
using pops::PreparedFieldBoundaryContextSet;
using pops::PreparedVectorDistribution;
using pops::Real;
using pops::RealVector;
using pops::amr::RefinementRatio;
using pops::elliptic::mg::CompositeFacBuildRequest;
using pops::elliptic::mg::CompositeFacPoisson;
using pops::elliptic::mg::GeometricMG;
using pops::elliptic::mg::GeometricMultigridOptions;
using pops::elliptic::mg::GeometricSmoother;
using pops::elliptic::mg::GeometricProlongation;
using pops::elliptic::mg::CellCentreBallDomain;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

template <int Dim>
struct DynamicBoundaryContextLifetime {
  explicit DynamicBoundaryContextLifetime(std::size_t levels) : failures(levels) {}

  std::vector<Real> parameters{Real(0.25)};
  std::vector<FieldBoundaryFailure<Dim>> failures;
};

template <int Dim, class Value>
auto filled(Value value) {
  std::array<Value, Dim> result{};
  result.fill(value);
  return result;
}

template <int Dim>
Extent<Dim> extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Index<Dim> index(int value) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::EllipticBuildRequest<Dim> request(const Geometry<Dim>& geometry, BoxArray<Dim> boxes,
                                        bool periodic = false) {
  Extent<Dim> rank_extent = extent<Dim>(1);
  rank_extent[0] = pops::n_ranks();
  Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  const RankSpace<Dim> ranks{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution = Distribution<Dim>::replicated(boxes, ranks);
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  if (!periodic)
    faces.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0)});
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const std::size_t pairs = boxes.size() * (boxes.size() - 1) / 2;
  std::array<bool, Dim> periodic_axes{};
  periodic_axes.fill(periodic);
  return {
      geometry,
      std::move(boxes),
      distribution,
      local_rank,
      PhysicalBoundaryConditions<Dim>{periodic ? BoundaryTopology<Dim>::axis_periodic(periodic_axes)
                                               : BoundaryTopology<Dim>::physical(),
                                      faces, spacing},
      Extent<Dim>{},
      extent<Dim>(1),
      {distribution.box_count(), pairs}};
}

template <int Dim>
pops::EllipticBuildRequest<Dim> request(int cells, BoxArray<Dim> boxes, bool periodic = false) {
  const Box<Dim> domain{index<Dim>(0), index<Dim>(cells - 1)};
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, RealVector<Dim>{}, [&] {
    RealVector<Dim> upper{};
    for (int axis = 0; axis < Dim; ++axis)
      upper[axis] = Real(1);
    return upper;
  }());
  return request<Dim>(geometry, std::move(boxes), periodic);
}

template <int Dim>
pops::EllipticBuildRequest<Dim> complete_request(int cells, bool periodic = false) {
  const Box<Dim> domain{index<Dim>(0), index<Dim>(cells - 1)};
  return request<Dim>(cells, BoxArray<Dim>{std::vector<Box<Dim>>{domain}}, periodic);
}

template <int Dim>
void expect_geometric_cycle() {
  const ExecutionLane lane = ExecutionLane::world("tests.geometric-mg.cycle");
  GeometricMultigridOptions options;
  options.relative_tolerance = Real(1e-7);
  options.absolute_tolerance = Real(1e-10);
  options.maximum_cycles = 80;
  options.bottom_sweeps = 40;
  GeometricMG<Dim> solver(complete_request<Dim>(8), lane, options);
  solver.install_nullspace(FieldNullspacePlan<Dim>{},
                           PreparedVectorDistribution<Dim>::replicated());
  EXPECT_GE(solver.num_levels(), 2);
  solver.phi().set_val(Real(0));
  solver.rhs().set_val(Real(1));
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_GT(report.iters, 0);
  EXPECT_LT(report.rel_residual, Real(1e-7));
}

template <int Dim>
void expect_partial_composite_preparation() {
  const ExecutionLane lane = ExecutionLane::world("tests.composite-fac.partial");
  auto coarse = complete_request<Dim>(8);
  const Box<Dim> fine_patch{index<Dim>(4), index<Dim>(11)};
  auto fine = request<Dim>(16, BoxArray<Dim>{std::vector<Box<Dim>>{fine_patch}});
  CompositeFacBuildRequest<Dim> hierarchy{{std::move(coarse), std::move(fine)},
                                          {RefinementRatio<Dim>{filled<Dim>(2)}}};
  CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  solver.install_nullspace(FieldNullspacePlan<Dim>{},
                           {PreparedVectorDistribution<Dim>::replicated(),
                            PreparedVectorDistribution<Dim>::replicated()});
  EXPECT_EQ(solver.n_levels(), 2);
  solver.phi_level(0).set_val(Real(0));
  solver.phi_level(1).set_val(Real(0));
  solver.rhs_level(0).set_val(Real(0));
  solver.rhs_level(1).set_val(Real(0));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
}

template <int Dim>
void expect_exact_rank_composite_ratio(const std::array<int, Dim>& components) {
  const ExecutionLane lane = ExecutionLane::world("tests.composite-fac.ratio");
  auto coarse = complete_request<Dim>(4);
  Extent<Dim> ratio_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    ratio_extent[axis] = components[static_cast<std::size_t>(axis)];
  const Geometry<Dim> fine_geometry = coarse.geometry.refine(ratio_extent);
  auto fine =
      request<Dim>(fine_geometry, BoxArray<Dim>{std::vector<Box<Dim>>{fine_geometry.domain()}});
  CompositeFacBuildRequest<Dim> hierarchy{{std::move(coarse), std::move(fine)},
                                          {RefinementRatio<Dim>{components}}};
  CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  EXPECT_EQ(solver.n_levels(), 2);
}

template <int Dim>
void expect_singular_authority_rejects_incompatible_rhs_and_applies_gauge() {
  const ExecutionLane lane = ExecutionLane::world("tests.geometric-mg.nullspace");
  auto build = complete_request<Dim>(8, true);
  Real measure = Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    measure *= build.geometry.spacing(axis);
  GeometricMG<Dim> solver(std::move(build), lane);
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<Dim>("periodic-nullspace", "unit-test", measure),
      PreparedVectorDistribution<Dim>::replicated());

  solver.phi().set_val(Real(3));
  solver.rhs().set_val(Real(1));
  const pops::SolveReport incompatible = solver.solve();
  EXPECT_EQ(incompatible.status, pops::SolveStatus::kIncompatibleRhs);
  EXPECT_EQ(pops::norm_inf(solver.phi()), Real(3));

  solver.rhs().set_val(Real(0));
  const pops::SolveReport solved = solver.solve();
  ASSERT_TRUE(solved.solved()) << solved.reason;
  EXPECT_EQ(pops::norm_inf(solver.phi()), Real(0));
}

template <int Dim>
void expect_composite_singular_authority_covers_the_complete_hierarchy() {
  const ExecutionLane lane = ExecutionLane::world("tests.composite-fac.nullspace");
  auto coarse = complete_request<Dim>(8, true);
  const Box<Dim> fine_patch{index<Dim>(4), index<Dim>(11)};
  auto fine = request<Dim>(16, BoxArray<Dim>{std::vector<Box<Dim>>{fine_patch}}, true);
  Real coarse_measure = Real(1);
  Real fine_measure = Real(1);
  for (int axis = 0; axis < Dim; ++axis) {
    coarse_measure *= coarse.geometry.spacing(axis);
    fine_measure *= fine.geometry.spacing(axis);
  }
  CompositeFacBuildRequest<Dim> hierarchy{{std::move(coarse), std::move(fine)},
                                          {RefinementRatio<Dim>{filled<Dim>(2)}}};
  CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  FieldNullspacePlan<Dim> plan = pops::constant_mean_zero_nullspace<Dim>(
      "periodic-composite-nullspace", "unit-test", coarse_measure);
  plan.bases.front().cell_measure = {coarse_measure, fine_measure};
  solver.install_nullspace(std::move(plan), {PreparedVectorDistribution<Dim>::replicated(),
                                             PreparedVectorDistribution<Dim>::replicated()});

  solver.phi_level(0).set_val(Real(2));
  solver.phi_level(1).set_val(Real(2));
  solver.rhs_level(0).set_val(Real(1));
  solver.rhs_level(1).set_val(Real(0));
  const pops::SolveReport incompatible = solver.solve();
  EXPECT_EQ(incompatible.status, pops::SolveStatus::kIncompatibleRhs);
  EXPECT_EQ(pops::norm_inf(solver.phi_level(0)), Real(2));
  EXPECT_EQ(pops::norm_inf(solver.phi_level(1)), Real(2));

  solver.rhs_level(0).set_val(Real(0));
  const pops::SolveReport solved = solver.solve();
  ASSERT_TRUE(solved.solved()) << solved.reason;
  // Removing a constant nullspace component is already an exact solution, including
  // the covered parent values used by fine interpolation. No FAC cycle is needed.
  EXPECT_EQ(solved.iters, 0);
  EXPECT_EQ(solved.residual_norm, Real(0));
  EXPECT_LE(pops::norm_inf(solver.phi_level(0)), Real(1e-8));
  EXPECT_LE(pops::norm_inf(solver.phi_level(1)), Real(1e-8));
}

template <int Dim>
struct DynamicBoundaryProbe {
  inline static int prepare_residual_calls = 0;
  inline static int prepare_jvp_calls = 0;
  inline static int residual_calls = 0;
  inline static int jvp_calls = 0;
  inline static int maximum_iteration = -1;
  inline static std::array<bool, 3> levels{};
  inline static bool immutable_operator_inputs = true;

  static void reset() {
    prepare_residual_calls = 0;
    prepare_jvp_calls = 0;
    residual_calls = 0;
    jvp_calls = 0;
    maximum_iteration = -1;
    levels.fill(false);
    immutable_operator_inputs = true;
  }

  static bool observe(int face, const FieldBoundaryExecutionContext<Dim>& context) {
    maximum_iteration = std::max(maximum_iteration, context.point.iteration);
    if (context.point.level >= 0 && context.point.level < static_cast<int>(levels.size()))
      levels[static_cast<std::size_t>(context.point.level)] = true;
    if (face < 0 || face >= 2 * Dim || context.failure == nullptr ||
        context.parameters == nullptr || context.parameter_count != 1 ||
        context.parameters->size() != 1) {
      if (context.failure != nullptr) {
        context.failure->code = 901;
        context.failure->face = face;
      }
      return false;
    }
    return true;
  }

  static void prepare_residual(int face, const pops::MultiFab<Dim>& iterate,
                               pops::MultiFab<Dim>& operator_view, const Geometry<Dim>& geometry,
                               const FieldBoundaryExecutionContext<Dim>& context) {
    (void)iterate;
    (void)geometry;
    ++prepare_residual_calls;
    immutable_operator_inputs &= !iterate.shares_storage_with(operator_view);
    observe(face, context);
  }

  static void prepare_jvp(int face, const pops::MultiFab<Dim>& iterate,
                          const pops::MultiFab<Dim>& direction, pops::MultiFab<Dim>& direction_view,
                          const Geometry<Dim>& geometry,
                          const FieldBoundaryExecutionContext<Dim>& context) {
    (void)geometry;
    ++prepare_jvp_calls;
    immutable_operator_inputs &= !iterate.shares_storage_with(direction_view) &&
                                 !direction.shares_storage_with(direction_view);
    observe(face, context);
  }

  static void add_residual(int face, const pops::MultiFab<Dim>& iterate,
                           pops::MultiFab<Dim>& output, const Geometry<Dim>& geometry,
                           const FieldBoundaryExecutionContext<Dim>& context) {
    ++residual_calls;
    if (!observe(face, context))
      return;
    const int axis = face / 2;
    const Box<Dim> domain = geometry.domain();
    const int coordinate = face % 2 == 0 ? domain.lo[axis] : domain.hi[axis];
    const Real alpha = context.parameters->front();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      Box<Dim> region = output.box(local).intersect(domain);
      if (coordinate < region.lo[axis] || coordinate > region.hi[axis])
        continue;
      region.lo[axis] = coordinate;
      region.hi[axis] = coordinate;
      const auto phi = iterate.fab(local).view();
      const auto residual = output.fab(local).view();
      pops::for_each_cell(region, [=] POPS_HD(const Index<Dim>& cell) {
        residual(cell, 0) += alpha * phi(cell, 0) * phi(cell, 0);
      });
    }
    Kokkos::fence();
  }

  static void apply_jvp(int face, const pops::MultiFab<Dim>& iterate,
                        const pops::MultiFab<Dim>& direction, pops::MultiFab<Dim>& output,
                        const Geometry<Dim>& geometry,
                        const FieldBoundaryExecutionContext<Dim>& context) {
    ++jvp_calls;
    if (!observe(face, context))
      return;
    const int axis = face / 2;
    const Box<Dim> domain = geometry.domain();
    const int coordinate = face % 2 == 0 ? domain.lo[axis] : domain.hi[axis];
    const Real alpha = context.parameters->front();
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      Box<Dim> region = output.box(local).intersect(domain);
      if (coordinate < region.lo[axis] || coordinate > region.hi[axis])
        continue;
      region.lo[axis] = coordinate;
      region.hi[axis] = coordinate;
      const auto phi = iterate.fab(local).view();
      const auto delta = direction.fab(local).view();
      const auto image = output.fab(local).view();
      pops::for_each_cell(region, [=] POPS_HD(const Index<Dim>& cell) {
        image(cell, 0) -= Real(2) * alpha * phi(cell, 0) * delta(cell, 0);
      });
    }
    Kokkos::fence();
  }

  static CompiledFieldBoundaryKernel<Dim> kernel() {
    return {"test.dynamic-boundary",
            "test.dynamic-boundary.residual",
            "test.dynamic-boundary.jvp",
            &prepare_residual,
            &prepare_jvp,
            &add_residual,
            &apply_jvp,
            true};
  }
};

inline FieldNewtonOptions dynamic_newton_options() {
  FieldNewtonOptions options;
  options.tolerance = Real(1e-9);
  options.max_iterations = 8;
  options.linear_tolerance = Real(1e-10);
  options.linear_max_iterations = 120;
  options.restart = 30;
  return options;
}

template <int Dim>
void expect_dynamic_geometric_newton() {
  const ExecutionLane lane = ExecutionLane::world("tests.geometric-mg.dynamic");
  DynamicBoundaryProbe<Dim>::reset();
  GeometricMultigridOptions controls;
  controls.reaction = Real(50);
  GeometricMG<Dim> solver(complete_request<Dim>(4), lane, controls);
  solver.install_nullspace(FieldNullspacePlan<Dim>{},
                           PreparedVectorDistribution<Dim>::replicated());
  solver.install_newton(dynamic_newton_options());
  solver.install_boundary_kernel(DynamicBoundaryProbe<Dim>::kernel());
  auto lifetime = std::make_shared<DynamicBoundaryContextLifetime<Dim>>(1);
  FieldBoundaryExecutionContext<Dim> context;
  context.point.level = 0;
  context.parameters = &lifetime->parameters;
  context.parameter_count = 1;
  context.failure = &lifetime->failures[0];
  auto mutable_contexts = std::make_shared<PreparedFieldBoundaryContextSet<Dim>>(lifetime, 1);
  mutable_contexts->bind(0, context);
  std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts =
      std::move(mutable_contexts);
  solver.set_boundary_contexts(std::move(contexts));
  solver.phi().set_val(Real(0));
  solver.rhs().set_val(Real(1));

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_GE(report.iters, 2);
  EXPECT_GT(pops::norm_inf(solver.phi()), Real(0));
  EXPECT_GT(DynamicBoundaryProbe<Dim>::prepare_residual_calls, 0);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::prepare_jvp_calls, 0);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::residual_calls, 0);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::jvp_calls, 0);
  EXPECT_GE(DynamicBoundaryProbe<Dim>::maximum_iteration, 1);
  EXPECT_TRUE(DynamicBoundaryProbe<Dim>::levels[0]);
  EXPECT_TRUE(DynamicBoundaryProbe<Dim>::immutable_operator_inputs);
}

template <int Dim>
void expect_dynamic_composite_newton() {
  const ExecutionLane lane = ExecutionLane::world("tests.composite-fac.dynamic");
  DynamicBoundaryProbe<Dim>::reset();
  auto coarse = complete_request<Dim>(4);
  const Box<Dim> fine_patch{index<Dim>(0), index<Dim>(3)};
  auto fine = request<Dim>(8, BoxArray<Dim>{std::vector<Box<Dim>>{fine_patch}});
  CompositeFacBuildRequest<Dim> hierarchy{{std::move(coarse), std::move(fine)},
                                          {RefinementRatio<Dim>{filled<Dim>(2)}}};
  CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane, {}, Real(50));
  solver.install_nullspace(FieldNullspacePlan<Dim>{},
                           {PreparedVectorDistribution<Dim>::replicated(),
                            PreparedVectorDistribution<Dim>::replicated()});
  solver.install_newton(dynamic_newton_options());
  solver.install_boundary_kernel(DynamicBoundaryProbe<Dim>::kernel());
  auto lifetime = std::make_shared<DynamicBoundaryContextLifetime<Dim>>(2);
  auto mutable_contexts = std::make_shared<PreparedFieldBoundaryContextSet<Dim>>(lifetime, 2);
  for (std::size_t level = 0; level < mutable_contexts->size(); ++level) {
    FieldBoundaryExecutionContext<Dim> context;
    context.point.level = static_cast<int>(level);
    context.parameters = &lifetime->parameters;
    context.parameter_count = 1;
    context.failure = &lifetime->failures[level];
    mutable_contexts->bind(level, context);
  }
  std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts =
      std::move(mutable_contexts);
  solver.set_boundary_contexts(std::move(contexts));
  for (int level = 0; level < 2; ++level) {
    solver.phi_level(level).set_val(Real(0));
    solver.rhs_level(level).set_val(Real(1));
  }

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_GE(report.iters, 2);
  EXPECT_GT(pops::norm_inf(solver.phi_level(0)), Real(0));
  EXPECT_GT(pops::norm_inf(solver.phi_level(1)), Real(0));
  EXPECT_TRUE(DynamicBoundaryProbe<Dim>::levels[0]);
  EXPECT_TRUE(DynamicBoundaryProbe<Dim>::levels[1]);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::prepare_residual_calls, 0);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::prepare_jvp_calls, 0);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::residual_calls, 0);
  EXPECT_GT(DynamicBoundaryProbe<Dim>::jvp_calls, 0);
  EXPECT_TRUE(DynamicBoundaryProbe<Dim>::immutable_operator_inputs);
}

}  // namespace

TEST(test_geometric_mg_nd, one_algorithm_executes_true_cycles_in_one_two_and_three_dimensions) {
  expect_geometric_cycle<1>();
  expect_geometric_cycle<2>();
  expect_geometric_cycle<3>();
}

TEST(test_geometric_mg_nd, partial_composite_hierarchies_prepare_in_exact_rank) {
  expect_partial_composite_preparation<1>();
  expect_partial_composite_preparation<2>();
  expect_partial_composite_preparation<3>();
}

TEST(test_geometric_mg_nd, composite_fac_accepts_ranked_and_partially_refined_axes) {
  expect_exact_rank_composite_ratio<1>({3});
  expect_exact_rank_composite_ratio<2>({3, 1});
  expect_exact_rank_composite_ratio<3>({1, 2, 3});
}

TEST(test_geometric_mg_nd,
     explicit_nullspace_authority_rejects_incompatible_rhs_without_projection) {
  expect_singular_authority_rejects_incompatible_rhs_and_applies_gauge<1>();
  expect_singular_authority_rejects_incompatible_rhs_and_applies_gauge<2>();
  expect_singular_authority_rejects_incompatible_rhs_and_applies_gauge<3>();
}

TEST(test_geometric_mg_nd, composite_nullspace_authority_spans_every_exact_ranked_level) {
  expect_composite_singular_authority_covers_the_complete_hierarchy<1>();
  expect_composite_singular_authority_covers_the_complete_hierarchy<2>();
  expect_composite_singular_authority_covers_the_complete_hierarchy<3>();
}

TEST(test_geometric_mg_nd,
     dynamic_boundary_residual_and_jvp_execute_persistent_newton_in_every_exact_rank) {
  expect_dynamic_geometric_newton<1>();
  expect_dynamic_geometric_newton<2>();
  expect_dynamic_geometric_newton<3>();
}

TEST(test_geometric_mg_nd,
     composite_dynamic_boundary_newton_uses_level_qualified_exact_ranked_contexts) {
  expect_dynamic_composite_newton<1>();
  expect_dynamic_composite_newton<2>();
  expect_dynamic_composite_newton<3>();
}

TEST(test_geometric_mg_nd, prepared_operator_contract_authenticates_the_exact_layout_budget) {
  const ExecutionLane lane = ExecutionLane::world("tests.geometric-mg.contract");
  auto build = complete_request<2>(8);
  GeometricMultigridOptions options;
  const auto expected = GeometricMG<2>::expected_operator_contract(build, options);
  GeometricMG<2> solver(std::move(build), lane, options);
  EXPECT_EQ(solver.prepared_operator_contract().exact_fingerprint(), expected.exact_fingerprint());
}

TEST(test_geometric_mg_nd, unsupported_operator_families_fail_closed_at_capability_selection) {
  constexpr auto mg = GeometricMG<3>::capabilities();
  EXPECT_TRUE(mg.scalar_constant_coefficient);
  EXPECT_TRUE(mg.scalar_reaction);
  EXPECT_TRUE(mg.variable_diagonal);
  EXPECT_FALSE(mg.cross_tensor);
  EXPECT_TRUE(mg.embedded_boundary);

  constexpr auto fac = CompositeFacPoisson<3>::capabilities();
  EXPECT_TRUE(fac.partial_refinement);
  EXPECT_TRUE(fac.arbitrary_level_count);
  EXPECT_TRUE(fac.distributed_mpi);
  EXPECT_TRUE(fac.variable_diagonal);
  EXPECT_FALSE(fac.cross_tensor);
  EXPECT_TRUE(fac.embedded_boundary);
}

TEST(test_geometric_mg_nd, small_forcing_obeys_the_authored_relative_initial_residual) {
  const auto lane = ExecutionLane::world("tests.geometric-mg.small-forcing");
  GeometricMultigridOptions options;
  options.reaction = Real(1);
  options.relative_tolerance = Real(1e-9);
  options.absolute_tolerance = Real(0);
  options.maximum_cycles = 100;
  for (const Real initial : {Real(0), Real(1e-10)}) {
    GeometricMG<1> solver(complete_request<1>(4, true), lane, options);
    solver.install_nullspace(FieldNullspacePlan<1>{}, PreparedVectorDistribution<1>::replicated());
    solver.rhs().set_val(Real(1e-13));
    solver.phi().set_val(initial);
    const auto report = solver.solve();
    // A constant periodic field has no Laplacian: ||R(initial)||=|rhs-initial|.
    // The second guess is 1000 times the solution; both references remain below one.
    const Real expected_reference = std::abs(Real(1e-13) - initial);
    EXPECT_NEAR(report.reference_residual_norm, expected_reference,
                Real(1e-14) * expected_reference);
    ASSERT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
    EXPECT_GT(report.iters, 0);
    EXPECT_LE(report.residual_norm, options.relative_tolerance * expected_reference);
  }
}

template <int Dim>
void expect_tiled_anisotropic_correction_linearity(PhysicalBoundaryKind kind) {
  const auto lane = ExecutionLane::world("tests.geometric-mg.anisotropic-correction");
  Index<Dim> high = index<Dim>(7);
  high[0] = 31;
  const Box<Dim> domain{Index<Dim>{}, high};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real(1);
  const auto geometry = Geometry<Dim>::from_bounds(domain, RealVector<Dim>{}, upper);
  auto build = request<Dim>(geometry, BoxArray<Dim>::from_domain(domain, extent<Dim>(4)),
                            kind == PhysicalBoundaryKind::external);
  if (kind == PhysicalBoundaryKind::neumann) {
    std::array<PhysicalBoundaryFace, 2 * Dim> faces{};
    faces.fill(PhysicalBoundaryFace{kind, Real(0)});
    build.boundary = PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::physical(), faces,
                                                     build.boundary.spacing()};
  }
  GeometricMultigridOptions options;
  options.reaction = Real(1);
  options.relative_tolerance = Real(1e-10);
  options.absolute_tolerance = Real(0);
  options.maximum_cycles = 192;
  GeometricMG<Dim> solver(std::move(build), lane, options);
  solver.install_nullspace(FieldNullspacePlan<Dim>{},
                           PreparedVectorDistribution<Dim>::replicated());
  using Field = typename GeometricMG<Dim>::field_type;
  std::array<std::unique_ptr<Field>, 4> solutions;
  for (int mode = 0; mode < 4; ++mode) {
    solver.phi().set_val(Real(0));
    const Real scale = mode == 3 ? Real(1e-12) : Real(1);
    for (std::size_t local = 0; local < solver.rhs().local_size(); ++local) {
      auto values = solver.rhs().fab(local).view();
      pops::for_each_cell(solver.rhs().box(local), [=] POPS_HD(const Index<Dim>& cell) {
        const Real a = Real(1);
        const Real b = (cell[Dim - 1] % 3 == 0) ? Real(0.5) : Real(-0.25);
        values(cell, 0) = scale * (mode == 0 ? a : mode == 1 ? b : a + b);
      });
    }
    const auto report = solver.solve();
    ASSERT_TRUE(report.solved()) << "Dim=" << Dim << " mode=" << mode
                                 << " boundary=" << static_cast<int>(kind) << " " << report.reason
                                 << " relative=" << report.rel_residual;
    EXPECT_LE(report.residual_norm, options.relative_tolerance * report.reference_residual_norm);
    solutions[mode] = std::make_unique<Field>(solver.phi().layout(), solver.phi().distribution(),
                                              solver.phi().local_rank(), 1, Extent<Dim>{});
    pops::elliptic::mg::copy_scalar_valid(solver.phi(), *solutions[mode]);
  }
  // A linear correction hierarchy must respect superposition and forcing amplitude;
  // the residual bound and reaction=1 also bound the absolute solution error.
  pops::saxpy(*solutions[2], Real(-1), *solutions[0]);
  pops::saxpy(*solutions[2], Real(-1), *solutions[1]);
  EXPECT_LE(pops::norm_inf(*solutions[2]), Real(4) * options.relative_tolerance);
  pops::saxpy(*solutions[0], Real(1), *solutions[1]);
  pops::saxpy(*solutions[0], Real(-1e12), *solutions[3]);
  EXPECT_LE(pops::norm_inf(*solutions[0]), Real(4) * options.relative_tolerance);
}

TEST(test_geometric_mg_nd, tiled_anisotropic_corrections_preserve_linearity_and_small_forcing) {
  for (const auto kind : {PhysicalBoundaryKind::dirichlet, PhysicalBoundaryKind::neumann,
                          PhysicalBoundaryKind::external}) {
    expect_tiled_anisotropic_correction_linearity<1>(kind);
    expect_tiled_anisotropic_correction_linearity<2>(kind);
    expect_tiled_anisotropic_correction_linearity<3>(kind);
  }
}

TEST(test_geometric_mg_nd, strict_field_conductor_preserves_initial_warm_start_then_projects) {
  const ExecutionLane lane = ExecutionLane::world("tests.geometric-mg.strict-field-domain");
  const Box<2> domain{Index<2>{0, 0}, Index<2>{2, 2}};
  const auto geometry = Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{3, 3});
  auto build = request<2>(geometry, BoxArray<2>{std::vector<Box<2>>{domain}});
  build.distribution = Distribution<2>::partitioned(build.boxes, build.distribution.rank_space(),
                                                    std::vector<Index<2>>{Index<2>{}});
  GeometricMultigridOptions options;
  options.smoother = GeometricSmoother::red_black_gauss_seidel;
  options.prolongation = GeometricProlongation::constant_injection;
  options.pre_smooth_bottom = true;
  options.pre_sweeps = options.post_sweeps = 2;
  options.bottom_sweeps = 50;
  options.minimum_coarse_extent = 2;
  options.maximum_cycles = 1;
  options.relative_tolerance = Real(1e-8);
  options.field_domain = CellCentreBallDomain{2, {Real(1.5), Real(1.5), Real(0)}, Real(1)};
  GeometricMG<2> solver(std::move(build), lane, options);
  solver.install_nullspace(FieldNullspacePlan<2>{}, PreparedVectorDistribution<2>::distributed());
  solver.phi().set_val(Real(2));
  solver.rhs().set_val(Real(1));
  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    auto phi = solver.phi().fab(local).view();
    pops::for_each_cell(solver.phi().box(local), [=] POPS_HD(const Index<2>& cell) {
      if (cell[0] == 1 && cell[1] == 1)
        phi(cell, 0) = Real(0);
    });
  }
  const auto first = solver.solve();
  EXPECT_FALSE(first.solved());
  EXPECT_EQ(first.reference_residual_norm, Real(9));
  EXPECT_EQ(first.residual_norm, Real(8));
  EXPECT_EQ(first.iters, 1);
  // The four centres at distance exactly radius=1 are conductors (strict '<').
  const auto second = solver.solve();
  ASSERT_TRUE(second.solved()) << second.reason;
  EXPECT_EQ(second.reference_residual_norm, Real(8));
  EXPECT_EQ(second.residual_norm, Real(0));
  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    const auto phi = std::as_const(solver.phi()).fab(local).view();
    const Real error =
        pops::for_each_cell_reduce_max(solver.phi().box(local), [=] POPS_HD(const Index<2>& cell) {
          const Real expected = cell[0] == 1 && cell[1] == 1 ? Real(0.25) : Real(0);
          return Kokkos::abs(phi(cell, 0) - expected);
        });
    EXPECT_EQ(error, Real(0));
  }
}

TEST(test_geometric_mg_nd, field_conductor_fac_solves_a_partial_partitioned_hierarchy) {
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("tests.geometric-mg.field-domain-fac");
  const Box<2> coarse_domain{Index<2>{0, 0}, Index<2>{15, 15}};
  const auto coarse_geometry =
      Geometry<2>::from_bounds(coarse_domain, RealVector<2>{0, 0}, RealVector<2>{1, 1});
  const auto fine_geometry = coarse_geometry.refine(Extent<2>{2, 2});
  BoxArray<2> coarse_boxes{
      std::vector<Box<2>>{{Index<2>{0, 0}, Index<2>{7, 15}}, {Index<2>{8, 0}, Index<2>{15, 15}}}};
  BoxArray<2> fine_boxes{std::vector<Box<2>>{{Index<2>{12, 12}, Index<2>{15, 19}},
                                             {Index<2>{16, 12}, Index<2>{19, 19}}}};
  auto coarse = request<2>(coarse_geometry, coarse_boxes);
  auto fine = request<2>(fine_geometry, fine_boxes);
  const std::vector<Index<2>> owners{Index<2>{}, Index<2>{std::min(1, pops::n_ranks() - 1), 0}};
  coarse.distribution =
      Distribution<2>::partitioned(coarse_boxes, coarse.distribution.rank_space(), owners);
  fine.distribution =
      Distribution<2>::partitioned(fine_boxes, fine.distribution.rank_space(), owners);
  pops::elliptic::amr::CompositeFacBuildRequest<2> hierarchy;
  hierarchy.levels = {coarse, fine};
  hierarchy.ratios = {RefinementRatio<2>{2, 2}};
  hierarchy.budget = {2,
                      1,
                      16,
                      512,
                      65536,
                      {{4, 16}, 4096, 8192, 9, 8, 65536, 65536, 65536},
                      {512, 8, 65536, 65536, 65536},
                      {512, 8, 65536, 65536, 65536}};
  GeometricMultigridOptions controls;
  controls.smoother = GeometricSmoother::red_black_gauss_seidel;
  controls.prolongation = GeometricProlongation::constant_injection;
  controls.pre_smooth_bottom = true;
  controls.minimum_coarse_extent = 2;
  controls.field_domain = CellCentreBallDomain{2, {Real(0.5), Real(0.5), Real(0)}, Real(0.3)};
  pops::CompositeFacOptions fac;
  fac.fine_sweeps = 4;
  fac.max_iters = 80;
  fac.rel_tol = Real(1e-8);
  fac.abs_tol = Real(0);
  fac.coarse_rel_tol = Real(1e-8);
  fac.coarse_abs_tol = Real(0);
  fac.coarse_cycles = 80;
  pops::elliptic::amr::CompositeFacPoisson<2> solver(hierarchy, fac, Real(0), &lane, false, true,
                                                     controls);
  solver.install_nullspace(FieldNullspacePlan<2>{}, {PreparedVectorDistribution<2>::distributed(),
                                                     PreparedVectorDistribution<2>::distributed()});
  for (int level = 0; level < 2; ++level) {
    auto& rhs = solver.rhs_level(level);
    const auto geometry = level == 0 ? coarse_geometry : fine_geometry;
    const auto ball = *controls.field_domain;
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      const auto values = rhs.fab(local).view();
      pops::for_each_cell(rhs.box(local), [=] POPS_HD(const Index<2>& cell) {
        Real load = Real(0);
        if (ball.contains(geometry, cell))
          for (int axis = 0; axis < 2; ++axis) {
            Index<2> lower = cell, upper = cell;
            --lower[axis];
            ++upper[axis];
            const Real h = geometry.spacing(axis);
            load += (Real(2) - (ball.contains(geometry, lower) ? Real(1) : Real(0)) -
                     (ball.contains(geometry, upper) ? Real(1) : Real(0))) /
                    (h * h);
          }
        values(cell, 0) = load;
      });
    }
    solver.phi_level(level).set_val(Real(0));
  }
  const auto report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason << " relative=" << report.rel_residual;
  EXPECT_LE(report.residual_norm, fac.rel_tol * report.reference_residual_norm);
  for (int level = 0; level < 2; ++level) {
    const auto& phi = std::as_const(solver).phi_level(level);
    const auto geometry = level == 0 ? coarse_geometry : fine_geometry;
    const auto ball = *controls.field_domain;
    for (std::size_t local = 0; local < phi.local_size(); ++local) {
      const auto values = phi.fab(local).view();
      EXPECT_LE(pops::for_each_cell_reduce_max(phi.box(local),
                                               [=] POPS_HD(const Index<2>& cell) {
                                                 const Real expected = ball.contains(geometry, cell)
                                                                           ? Real(1)
                                                                           : Real(0);
                                                 return Kokkos::abs(values(cell, 0) - expected);
                                               }),
                Real(1e-6));
    }
  }
}

TEST(test_geometric_mg_nd, roundoff_residual_is_opt_in_and_reports_physical_accuracy) {
  const ExecutionLane lane = ExecutionLane::world("tests.geometric-mg.roundoff-aware");
  const Box<2> domain{Index<2>{0, 0}, Index<2>{2, 2}};
  const auto geometry = Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{3, 3});
  GeometricMultigridOptions options;
  options.smoother = GeometricSmoother::red_black_gauss_seidel;
  options.prolongation = GeometricProlongation::constant_injection;
  options.pre_smooth_bottom = true;
  options.minimum_coarse_extent = 2;
  options.field_domain = CellCentreBallDomain{2, {Real(1.5), Real(1.5), Real(0)}, Real(1)};
  for (bool enabled : {false, true}) {
    options.roundoff_aware_residual = enabled;
    GeometricMG<2> solver(request<2>(geometry, BoxArray<2>{std::vector<Box<2>>{domain}}), lane,
                          options);
    solver.install_nullspace(FieldNullspacePlan<2>{}, PreparedVectorDistribution<2>::replicated());
    solver.phi().set_val(Real(0));
    solver.rhs().set_val(std::nextafter(Real(4e13), std::numeric_limits<Real>::infinity()));
    for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
      const auto phi = solver.phi().fab(local).view();
      pops::for_each_cell(solver.phi().box(local), [=] POPS_HD(const Index<2>& cell) {
        if (cell[0] == 1 && cell[1] == 1)
          phi(cell, 0) = Real(1e13);
      });
    }
    const auto report = solver.solve();
    ASSERT_TRUE(report.solved()) << report.reason;
    if (!enabled) {
      EXPECT_FALSE(report.residual_evaluation.has_value());
      EXPECT_EQ(report.iters, 1);
    } else {
      ASSERT_TRUE(report.residual_evaluation.has_value());
      const auto& evidence = *report.residual_evaluation;
      EXPECT_EQ(report.iters, 0);
      EXPECT_TRUE(evidence.roundoff_limited);
      EXPECT_GT(report.residual_norm, evidence.requested_threshold);
      EXPECT_LE(report.residual_norm, evidence.applied_threshold);
      ASSERT_TRUE(evidence.relative_to_rhs_upper.has_value());
      EXPECT_LT(*evidence.relative_to_rhs_upper, Real(1e-13));
      // This diagonal one-cell problem has exactly known physical error r/4.
      EXPECT_LT(report.residual_norm / Real(4) / Real(1e13), Real(3e-16));
    }
  }
  options.roundoff_aware_residual = true;
  for (const Real forcing : {Real(1e-24), Real(1e12)}) {
    GeometricMG<2> solver(request<2>(geometry, BoxArray<2>{std::vector<Box<2>>{domain}}), lane,
                          options);
    solver.install_nullspace(FieldNullspacePlan<2>{}, PreparedVectorDistribution<2>::replicated());
    solver.phi().set_val(Real(0));
    solver.rhs().set_val(forcing);
    const auto report = solver.solve();
    ASSERT_TRUE(report.solved()) << report.reason;
    ASSERT_TRUE(report.residual_evaluation.has_value());
    EXPECT_EQ(report.residual_evaluation->rhs_norm, forcing);
    EXPECT_LE(report.residual_norm, options.relative_tolerance * report.reference_residual_norm);
    EXPECT_FALSE(report.residual_evaluation->roundoff_limited);
  }
  GeometricMG<2> unsupported(request<2>(geometry, BoxArray<2>{std::vector<Box<2>>{domain}}), lane,
                             options);
  unsupported.install_nullspace(FieldNullspacePlan<2>{},
                                PreparedVectorDistribution<2>::replicated());
  unsupported.phi().set_val(Real(0));
  unsupported.rhs().set_val(std::numeric_limits<Real>::denorm_min() * Real(16));
  EXPECT_FALSE(unsupported.solve().solved());
}

TEST(test_geometric_mg_nd, roundoff_absolute_term_sum_does_not_overflow_before_gamma) {
  const Box<2> box{Index<2>{0, 0}, Index<2>{2, 2}};
  const auto layout = BoxArray<2>{std::vector<Box<2>>{box}};
  const RankSpace<2> ranks{Index<2>{}, Extent<2>{pops::n_ranks(), 1}};
  const auto distribution = Distribution<2>::replicated(layout, ranks);
  const Index<2> rank{pops::my_rank(), 0};
  pops::MultiFab<2> phi(layout, distribution, rank, 1, Extent<2>{1, 1});
  pops::MultiFab<2> rhs(layout, distribution, rank, 1, Extent<2>{});
  phi.set_val(Real(1e200));
  rhs.set_val(Real(1));
  for (std::size_t local = 0; local < phi.local_size(); ++local) {
    pops::elliptic::mg::roundoff::FivePointBoundKernel<2> kernel{
        std::as_const(phi).fab(local).view(),
        std::as_const(rhs).fab(local).view(),
        {},
        {Real(1e109), Real(1e109)},
        pops::elliptic::mg::roundoff::stencil_gamma<2>()};
    const Real bound = pops::for_each_cell_reduce_max(box, kernel);
    EXPECT_TRUE(std::isfinite(bound));
    EXPECT_GT(bound, Real(1e294));
    EXPECT_LT(bound, Real(1e296));
  }
}
