#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using pops::BoundaryTopology;
using pops::Box;
using pops::CompiledFieldBoundaryKernel;
using pops::Extent;
using pops::FieldBoundaryExecutionContext;
using pops::FieldBoundaryFailure;
using pops::Geometry;
using pops::FieldNullspacePlan;
using pops::FieldNewtonOptions;
using pops::Index;
using pops::PhysicalBoundaryConditions;
using pops::PhysicalBoundaryFace;
using pops::PhysicalBoundaryKind;
using pops::PreparedVectorDistribution;
using pops::Real;
using pops::RealVector;
using pops::amr::RefinementRatio;
using pops::elliptic::mg::CompositeFacBuildRequest;
using pops::elliptic::mg::CompositeFacPoisson;
using pops::elliptic::mg::GeometricMG;
using pops::elliptic::mg::GeometricMultigridOptions;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

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
  EXPECT_LE(pops::norm_inf(solver.phi_level(0)), Real(1e-14));
  EXPECT_LE(pops::norm_inf(solver.phi_level(1)), Real(1e-14));
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
  std::vector<Real> parameters{Real(0.25)};
  FieldBoundaryFailure<Dim> failure;
  FieldBoundaryExecutionContext<Dim> context;
  context.point.level = 0;
  context.parameters = &parameters;
  context.parameter_count = 1;
  context.failure = &failure;
  solver.set_boundary_context(context);
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
  std::vector<Real> parameters{Real(0.25)};
  std::array<FieldBoundaryFailure<Dim>, 2> failures{};
  std::vector<FieldBoundaryExecutionContext<Dim>> contexts(2);
  for (std::size_t level = 0; level < contexts.size(); ++level) {
    contexts[level].point.level = static_cast<int>(level);
    contexts[level].parameters = &parameters;
    contexts[level].parameter_count = 1;
    contexts[level].failure = &failures[level];
  }
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
  EXPECT_FALSE(mg.variable_diagonal);
  EXPECT_FALSE(mg.cross_tensor);
  EXPECT_FALSE(mg.embedded_boundary);

  constexpr auto fac = CompositeFacPoisson<3>::capabilities();
  EXPECT_TRUE(fac.partial_refinement);
  EXPECT_TRUE(fac.arbitrary_level_count);
  EXPECT_FALSE(fac.distributed_mpi);
  EXPECT_FALSE(fac.variable_diagonal);
  EXPECT_FALSE(fac.cross_tensor);
  EXPECT_FALSE(fac.embedded_boundary);
}
