#include <gtest/gtest.h>
#include <bit>
#include <iostream>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/nd/general_field_operator.hpp>

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_bc_rec_adapter.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
Extent<Dim> filled_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Box<Dim> island_domain() {
  Index<Dim> lo{};
  Index<Dim> hi{};
  hi[0] = 3;
  for (int axis = 1; axis < Dim; ++axis)
    hi[axis] = 1;
  return Box<Dim>{lo, hi};
}

template <int Dim>
mesh::RankSpace<Dim> one_rank_space() {
  return mesh::RankSpace<Dim>{Index<Dim>{}, filled_extent<Dim>(1)};
}

enum class TestFieldPattern : unsigned char {
  labels,
  compatible_rhs,
  island_constants,
  ones,
  ramp,
  alternating,
  overlapping_combination,
};

template <int Dim>
struct FillTestField {
  FieldView<Real, Dim> values;
  TestFieldPattern pattern;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = Real(0);
    switch (pattern) {
      case TestFieldPattern::labels:
        value = index[0] < 2 ? Real(1) : Real(2);
        break;
      case TestFieldPattern::compatible_rhs: {
        const Real magnitude = index[0] < 2 ? Real(1) : Real(2);
        value = index[0] % 2 == 0 ? magnitude : -magnitude;
        break;
      }
      case TestFieldPattern::island_constants:
        value = index[0] < 2 ? Real(3) : Real(-5);
        break;
      case TestFieldPattern::ones:
        value = Real(1);
        break;
      case TestFieldPattern::ramp:
        value = static_cast<Real>(index[0] + 1);
        break;
      case TestFieldPattern::alternating:
        value = index[0] == 0 ? Real(1) : Real(-1);
        break;
      case TestFieldPattern::overlapping_combination:
        value = Real(3) - Real(2) * static_cast<Real>(index[0] + 1);
        break;
    }
    values(index, 0) = value;
  }
};

template <int Dim>
struct ReplaceFirstSlab {
  FieldView<Real, Dim> values;
  Real replacement;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (index[0] == 0)
      values(index, 0) = replacement;
  }
};

template <int Dim>
struct AbsoluteFieldValue {
  FieldView<const Real, Dim> values;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real value = values(index, 0);
    return value < Real(0) ? -value : value;
  }
};

template <int Dim>
void fill(MultiFab<Dim>& field, TestFieldPattern pattern) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), FillTestField<Dim>{field.fab(local).view(), pattern});
}

template <int Dim>
void replace_first_slab(MultiFab<Dim>& field, Real replacement) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), ReplaceFirstSlab<Dim>{field.fab(local).view(), replacement});
}

template <int Dim>
Real maximum_absolute_value(const MultiFab<Dim>& field) {
  Real result = Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result =
        std::max(result, for_each_cell_reduce_max(
                             field.box(local), AbsoluteFieldValue<Dim>{field.fab(local).view()}));
  return all_reduce_max(result);
}

template <int Dim>
struct TwoIslandFixture {
  ExecutionLane lane = ExecutionLane::world("tests.field-nullspace.fixture:" + std::to_string(Dim));
  Box<Dim> domain = island_domain<Dim>();
  mesh::BoxArray<Dim> layout = mesh::BoxArray<Dim>(std::vector<Box<Dim>>{domain});
  mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
  std::shared_ptr<MultiFab<Dim>> labels =
      std::make_shared<MultiFab<Dim>>(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});

  TwoIslandFixture() { fill(*labels, TestFieldPattern::labels); }

  std::array<PreparedVectorDistribution<Dim>, 1> distributions() const {
    return {PreparedVectorDistribution<Dim>::Replicated};
  }

  FieldNullspacePlan<Dim> plan(int field_component = 0, int first_level = 0) const {
    const auto prepared_distributions = distributions();
    return labelled_mean_zero_nullspace<Dim>(
        "two-island-nullspace", "two-island-layout",
        std::vector<std::shared_ptr<const MultiFab<Dim>>>{labels},
        {{1, "island-a", "fixture:cell-label:1"}, {2, "island-b", "fixture:cell-label:2"}}, {},
        {Real(0.5)}, field_component,
        std::span<const PreparedVectorDistribution<Dim>>(prepared_distributions), lane,
        first_level);
  }

  MultiFab<Dim> field(int ncomp = 1, Extent<Dim> ghosts = Extent<Dim>{}) const {
    return MultiFab<Dim>(layout, distribution, Index<Dim>{}, ncomp, ghosts);
  }
};

template <int Dim>
void verify_exact_ranked_boundary_adapter() {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> physical_faces{};
  for (int axis = 0; axis < Dim; ++axis) {
    physical_faces[static_cast<std::size_t>(2 * axis)] =
        PhysicalBoundaryFace{PhysicalBoundaryKind::neumann, Real(0), Real(0), Real(1)};
    physical_faces[static_cast<std::size_t>(2 * axis + 1)] =
        PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0), Real(0), Real(1)};
  }
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = Real(1);
  const PhysicalBoundaryConditions<Dim> physical(BoundaryTopology<Dim>::physical(), physical_faces,
                                                 spacing);
  const FieldNullspaceOperatorFacts facts =
      field_nullspace_operator_facts_from_physical_boundary(physical, false);
  ASSERT_EQ(facts.boundaries.size(), static_cast<std::size_t>(2 * Dim));
  EXPECT_EQ(facts.boundary_set_identity,
            "pops.mesh.boundary.physical-conditions.cartesian-" + std::to_string(Dim) + "d@1");
  for (int axis = 0; axis < Dim; ++axis) {
    EXPECT_EQ(facts.boundaries[static_cast<std::size_t>(2 * axis)].behavior,
              FieldBoundaryNullspaceBehavior::PreservesConstantMode);
    EXPECT_EQ(facts.boundaries[static_cast<std::size_t>(2 * axis + 1)].behavior,
              FieldBoundaryNullspaceBehavior::ConstrainsConstantMode);
  }

  std::array<bool, Dim> periodic_axes{};
  periodic_axes.fill(true);
  const PhysicalBoundaryConditions<Dim> periodic(
      BoundaryTopology<Dim>::axis_periodic(periodic_axes), {}, spacing);
  const FieldNullspaceOperatorFacts periodic_facts =
      field_nullspace_operator_facts_from_physical_boundary(periodic, false);
  ASSERT_EQ(periodic_facts.boundaries.size(), static_cast<std::size_t>(2 * Dim));
  for (const FieldBoundaryNullspaceFact& fact : periodic_facts.boundaries)
    EXPECT_EQ(fact.behavior, FieldBoundaryNullspaceBehavior::PreservesConstantMode);

  const auto provider = make_default_field_nullspace_provider_registry<Dim>()->resolve(
      "pops.field-nullspace.operator-topology-derived");
  EXPECT_EQ(provider->interface_version(), 2U);
  EXPECT_EQ(provider->collective_contract(), "pops.field-nullspace.operator-topology-derived@2");
}

template <int Dim>
void verify_nullspace_execution() {
  TwoIslandFixture<Dim> fixture;
  const FieldNullspacePlan<Dim> plan = fixture.plan();
  ASSERT_EQ(plan.bases.size(), 2U);
  ASSERT_EQ(plan.gauges.size(), 2U);
  EXPECT_EQ(plan.bases[0].identity, "island-a");
  EXPECT_EQ(plan.bases[1].identity, "island-b");

  MultiFab<Dim> rhs = fixture.field();
  MultiFab<Dim> phi = fixture.field();
  fill(rhs, TestFieldPattern::compatible_rhs);
  fill(phi, TestFieldPattern::island_constants);
  const auto prepared_distributions = fixture.distributions();

  const std::vector<const MultiFab<Dim>*> rhs_levels{&rhs};
  const std::vector<double> witness = require_field_nullspace_compatible<Dim>(
      rhs_levels, plan, std::span<const PreparedVectorDistribution<Dim>>(prepared_distributions),
      fixture.lane, 0);
  ASSERT_EQ(witness.size(), 4U);
  EXPECT_NEAR(witness[0], 0.0, 1e-13);
  EXPECT_NEAR(witness[2], 0.0, 1e-13);

  const std::vector<MultiFab<Dim>*> phi_levels{&phi};
  apply_field_gauge<Dim>(phi_levels, plan,
                         std::span<const PreparedVectorDistribution<Dim>>(prepared_distributions),
                         fixture.lane);
  EXPECT_NEAR(maximum_absolute_value(phi), Real(0), Real(1e-13));

  fill(phi, TestFieldPattern::island_constants);
  FieldNullspaceWorkspace<Dim> workspace(
      plan, {fixture.labels.get()},
      std::vector<PreparedVectorDistribution<Dim>>(prepared_distributions.begin(),
                                                   prepared_distributions.end()),
      fixture.lane);
  const AllocationEventStats before = allocation_event_stats();
  workspace.apply_gauge(phi);
  const std::span<const double> persistent_witness = workspace.require_compatible(rhs);
  const AllocationEventStats after = allocation_event_stats();
  EXPECT_NEAR(maximum_absolute_value(phi), Real(0), Real(1e-13));
  ASSERT_EQ(persistent_witness.size(), 4U);
  EXPECT_EQ(after, before);

  replace_first_slab(rhs, Real(7));
  EXPECT_THROW(workspace.require_compatible(rhs), FieldNullspaceIncompatibleRhs);
}

template <int Dim>
void verify_overlapping_independent_bases() {
  TwoIslandFixture<Dim> fixture;
  auto first = std::make_shared<MultiFab<Dim>>(fixture.field());
  auto second = std::make_shared<MultiFab<Dim>>(fixture.field());
  fill(*first, TestFieldPattern::ones);
  fill(*second, TestFieldPattern::ramp);

  FieldNullspacePlan<Dim> plan;
  plan.identity = "overlapping-independent:" + std::to_string(Dim);
  plan.layout_identity = "overlapping-independent-layout:" + std::to_string(Dim);
  plan.bases = {{"first", "unit-test:first", "unit-test:first@1", 0, {first}, {}, {Real(1)}},
                {"second", "unit-test:second", "unit-test:second@1", 0, {second}, {}, {Real(1)}}};
  plan.gauges = {{"first", Real(0)}, {"second", Real(0)}};

  MultiFab<Dim> prepared_layout = fixture.field();
  MultiFab<Dim> value = fixture.field(1, filled_extent<Dim>(1));
  fill(value, TestFieldPattern::overlapping_combination);
  const auto distributions = fixture.distributions();
  FieldNullspaceWorkspace<Dim> workspace(
      plan, {&prepared_layout},
      std::vector<PreparedVectorDistribution<Dim>>(distributions.begin(), distributions.end()),
      fixture.lane);

  const AllocationEventStats before = allocation_event_stats();
  workspace.apply_gauge(value);
  const std::span<const double> witness = workspace.require_compatible(value);
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_NEAR(maximum_absolute_value(value), Real(0), Real(1e-13));
  ASSERT_EQ(witness.size(), 4U);
  EXPECT_NEAR(witness[0], 0.0, 1e-13);
  EXPECT_NEAR(witness[2], 0.0, 1e-13);
  EXPECT_EQ(after, before);
}

template <int Dim>
void verify_level_local_contract() {
  TwoIslandFixture<Dim> fixture;
  MultiFab<Dim> field = fixture.field();
  field.set_val(Real(0));
  FieldNullspacePlan<Dim> plan =
      constant_mean_zero_nullspace<Dim>("level-one-nullspace", "unit-test", Real(0.25));
  plan.bases[0].cell_measure = {Real(0), Real(0.25)};
  auto active_mask = std::make_shared<MultiFab<Dim>>(field);
  fill(*active_mask, TestFieldPattern::ones);
  plan.bases[0].masks = {nullptr, active_mask};
  const auto distributions = fixture.distributions();

  const std::vector<const MultiFab<Dim>*> const_levels{&field};
  const std::vector<MultiFab<Dim>*> mutable_levels{&field};
  const std::span<const PreparedVectorDistribution<Dim>> distribution_span(distributions);
  EXPECT_NO_THROW(
      validate_field_nullspace_basis<Dim>(const_levels, plan, distribution_span, fixture.lane, 1));
  EXPECT_NO_THROW((void)require_field_nullspace_compatible<Dim>(
      const_levels, plan, distribution_span, fixture.lane, 1));
  EXPECT_NO_THROW(apply_field_gauge<Dim>(mutable_levels, plan, distribution_span, fixture.lane, 1));

  FieldNullspacePlan<Dim> zero_measure = plan;
  zero_measure.bases[0].cell_measure[1] = Real(0);
  EXPECT_THROW(validate_field_nullspace_basis<Dim>(const_levels, zero_measure, distribution_span,
                                                   fixture.lane, 1),
               std::runtime_error);

  FieldNullspacePlan<Dim> missing_mask = plan;
  missing_mask.bases[0].masks[1].reset();
  EXPECT_THROW(validate_field_nullspace_basis<Dim>(const_levels, missing_mask, distribution_span,
                                                   fixture.lane, 1),
               std::runtime_error);

  const std::array<PreparedVectorDistribution<Dim>, 0> missing_distribution{};
  EXPECT_THROW(
      validate_field_nullspace_basis<Dim>(
          const_levels, plan,
          std::span<const PreparedVectorDistribution<Dim>>(missing_distribution), fixture.lane, 1),
      std::runtime_error);
}

}  // namespace

TEST(test_field_nullspace, canonicalizes_generic_boundary_facts_and_rejects_forged_sequences) {
  const FieldNullspaceOperatorFacts facts = make_field_nullspace_operator_facts(
      "test.boundary-set@1",
      {{"wall:z", FieldBoundaryNullspaceBehavior::Opaque},
       {"wall:a", FieldBoundaryNullspaceBehavior::PreservesConstantMode}},
      true);
  ASSERT_EQ(facts.boundaries.size(), 2U);
  EXPECT_EQ(facts.boundaries[0].boundary_id, "wall:a");
  EXPECT_EQ(facts.boundaries[1].boundary_id, "wall:z");
  EXPECT_NO_THROW((void)facts.exact_contract());

  FieldNullspaceOperatorFacts forged = facts;
  std::swap(forged.boundaries[0], forged.boundaries[1]);
  EXPECT_FALSE(forged.is_canonical());
  EXPECT_THROW((void)forged.exact_contract(), std::invalid_argument);
  EXPECT_THROW((void)make_field_nullspace_operator_facts(
                   "test.boundary-set@1",
                   {{"wall:a", FieldBoundaryNullspaceBehavior::PreservesConstantMode},
                    {"wall:a", FieldBoundaryNullspaceBehavior::ConstrainsConstantMode}},
                   false),
               std::invalid_argument);
  EXPECT_FALSE(FieldNullspaceOperatorFacts{}.is_canonical());
}

TEST(test_field_nullspace, exact_ranked_boundary_adapter_covers_1d_2d_and_3d) {
  verify_exact_ranked_boundary_adapter<1>();
  verify_exact_ranked_boundary_adapter<2>();
  verify_exact_ranked_boundary_adapter<3>();
  EXPECT_EQ(detail::physical_boundary_nullspace_behavior(
                PhysicalBoundaryFace{PhysicalBoundaryKind::robin, Real(0), Real(0), Real(1)}),
            FieldBoundaryNullspaceBehavior::PreservesConstantMode);
  EXPECT_EQ(detail::physical_boundary_nullspace_behavior(
                PhysicalBoundaryFace{PhysicalBoundaryKind::robin, Real(0), Real(2), Real(1)}),
            FieldBoundaryNullspaceBehavior::ConstrainsConstantMode);
}

TEST(test_field_nullspace, exact_ranked_nullspace_execution_covers_1d_2d_and_3d) {
  verify_nullspace_execution<1>();
  verify_nullspace_execution<2>();
  verify_nullspace_execution<3>();
}

TEST(test_field_nullspace, overlapping_independent_bases_cover_1d_2d_and_3d) {
  verify_overlapping_independent_bases<1>();
  verify_overlapping_independent_bases<2>();
  verify_overlapping_independent_bases<3>();
}

TEST(test_field_nullspace, exact_ranked_level_local_contract_covers_1d_2d_and_3d) {
  verify_level_local_contract<1>();
  verify_level_local_contract<2>();
  verify_level_local_contract<3>();
}

TEST(test_field_nullspace, active_mask_projects_only_marked_cells) {
  TwoIslandFixture<1> fixture;
  MultiFab<1> phi = fixture.field();
  MultiFab<1> rhs = fixture.field();
  phi.set_val(Real(4));
  rhs.set_val(Real(0));
  auto mask = std::make_shared<MultiFab<1>>(fixture.field());
  mask->set_val(Real(0));
  for (std::size_t local = 0; local < mask->local_size(); ++local) {
    auto& fab = mask->fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<1>& box = fab.box();
    for (int i = box.lo[0]; i <= box.hi[0]; ++i)
      host(static_cast<std::size_t>(i - box.lo[0])) = i < 2 ? Real(1) : Real(0);
    fab.copy_from_host(host);
  }
  FieldNullspacePlan<1> plan =
      constant_mean_zero_nullspace<1>("fac-active-mask", "unit-test", Real(1));
  plan.bases[0].masks = {mask};
  const auto distributions = fixture.distributions();
  FieldNullspaceWorkspace<1> workspace(
      plan, {&rhs},
      std::vector<PreparedVectorDistribution<1>>(distributions.begin(), distributions.end()),
      fixture.lane);
  workspace.apply_gauge(phi);
  for (std::size_t local = 0; local < phi.local_size(); ++local) {
    auto host = phi.fab(local).create_host_mirror();
    phi.fab(local).copy_to_host(host);
    const Box<1>& box = phi.box(local);
    for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
      const Real value = host(static_cast<std::size_t>(i - box.lo[0]));
      if (i < 2)
        EXPECT_NEAR(static_cast<double>(value), 0.0, 1e-12);
      else
        EXPECT_NEAR(static_cast<double>(value), 4.0, 1e-12);
    }
  }
  rhs.set_val(Real(1));
  EXPECT_THROW(workspace.require_compatible(rhs), FieldNullspaceIncompatibleRhs);
}

TEST(test_field_nullspace, labelled_topology_preserves_target_component_and_rejects_bad_labels) {
  TwoIslandFixture<2> fixture;
  const FieldNullspacePlan<2> component_plan = fixture.plan(3);
  ASSERT_EQ(component_plan.bases.size(), 2U);
  EXPECT_EQ(component_plan.bases[0].field_component, 3);
  EXPECT_EQ(component_plan.bases[1].field_component, 3);

  replace_first_slab(*fixture.labels, Real(3));
  EXPECT_THROW(fixture.plan(), std::runtime_error);
  replace_first_slab(*fixture.labels, Real(1.5));
  EXPECT_THROW(fixture.plan(), std::runtime_error);
}

TEST(test_field_nullspace, rejects_a_gauge_that_references_an_unknown_basis) {
  TwoIslandFixture<2> fixture;
  FieldNullspacePlan<2> plan = fixture.plan();
  plan.gauges[0].basis_identity = "missing-island";
  const auto distributions = fixture.distributions();
  const std::vector<const MultiFab<2>*> layouts{fixture.labels.get()};
  EXPECT_THROW(validate_field_nullspace_basis<2>(
                   layouts, plan, std::span<const PreparedVectorDistribution<2>>(distributions),
                   fixture.lane),
               std::runtime_error);
}

TEST(test_field_nullspace, validates_native_collective_capacities_before_size_arithmetic) {
  const std::size_t native_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t gram_edge = 1;
  while (gram_edge <= native_max / gram_edge)
    ++gram_edge;
  --gram_edge;
  EXPECT_EQ(detail::checked_field_nullspace_collective_product(gram_edge, gram_edge,
                                                               "synthetic Gram matrix"),
            gram_edge * gram_edge);
  EXPECT_THROW(detail::checked_field_nullspace_collective_product(gram_edge + 1, gram_edge + 1,
                                                                  "synthetic Gram matrix"),
               std::overflow_error);
  EXPECT_EQ(detail::checked_field_nullspace_collective_product(native_max / 2, std::size_t{2},
                                                               "synthetic moments"),
            (native_max / 2) * 2);
  EXPECT_THROW(detail::checked_field_nullspace_collective_product(
                   native_max / 2 + 1, std::size_t{2}, "synthetic moments"),
               std::overflow_error);
  EXPECT_EQ(detail::checked_field_nullspace_collective_sum(native_max - 1, std::size_t{1},
                                                           "synthetic label counts"),
            native_max);
  EXPECT_THROW(detail::checked_field_nullspace_collective_sum(native_max, std::size_t{1},
                                                              "synthetic label counts"),
               std::overflow_error);
  EXPECT_EQ(detail::checked_field_nullspace_collective_count(native_max, "synthetic collective"),
            std::numeric_limits<int>::max());
  EXPECT_THROW(
      detail::checked_field_nullspace_collective_count(native_max + 1, "synthetic collective"),
      std::overflow_error);
}

TEST(test_field_nullspace, validates_hierarchy_level_capacity_without_materializing_levels) {
  const int native_max = std::numeric_limits<int>::max();
  EXPECT_NO_THROW(
      detail::validate_field_nullspace_level_capacity(1, native_max, "synthetic hierarchy"));
  EXPECT_THROW(
      detail::validate_field_nullspace_level_capacity(2, native_max, "synthetic hierarchy"),
      std::overflow_error);
  EXPECT_THROW(detail::validate_field_nullspace_level_capacity(1, -1, "synthetic hierarchy"),
               std::invalid_argument);
}

TEST(test_field_nullspace, shared_constant_mode_preserves_individual_means) {
  constexpr int Dim = 2;
  TwoIslandFixture<Dim> fixture;
  auto rhs = fixture.field(2);
  auto phi = fixture.field(2);
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    const auto f = rhs.fab(local).view();
    const auto u = phi.fab(local).view();
    for_each_cell(rhs.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      f(cell, 0) = Real(1.5);
      f(cell, 1) = Real(-1.5);
      u(cell, 0) = Real(7.375);
      u(cell, 1) = Real(6.625);
    });
  }
  auto plan = constant_mean_zero_nullspace<Dim>("joint-lambda-kernel", "physical kernel (1,1)");
  plan.bases.front().component_count = 2;
  const auto distributions = fixture.distributions();
  FieldNullspaceWorkspace<Dim> workspace(
      plan, {&rhs},
      std::vector<PreparedVectorDistribution<Dim>>(distributions.begin(), distributions.end()),
      fixture.lane);
  const auto witness = workspace.require_compatible(rhs);
  ASSERT_EQ(witness.size(), 2U);
  EXPECT_NEAR(witness[0], 0.0, 1e-13);
  workspace.apply_gauge(phi);
  Real error = Real(0);
  for (std::size_t local = 0; local < phi.local_size(); ++local) {
    const auto u = std::as_const(phi).fab(local).view();
    error = std::max(
        error, for_each_cell_reduce_max(phi.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          return Kokkos::abs(u(cell, 0) - Real(0.375)) + Kokkos::abs(u(cell, 1) + Real(0.375));
        }));
  }
  EXPECT_NEAR(error, 0.0, 1e-12);
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    const auto f = rhs.fab(local).view();
    for_each_cell(rhs.box(local), [=] POPS_HD(const Index<Dim>& cell) { f(cell, 1) = Real(0); });
  }
  EXPECT_THROW(workspace.require_compatible(rhs), FieldNullspaceIncompatibleRhs);
  auto malformed = plan;
  malformed.bases.front().component_count = 3;
  EXPECT_THROW((FieldNullspaceWorkspace<Dim>(malformed, {&rhs},
                                             std::vector<PreparedVectorDistribution<Dim>>(
                                                 distributions.begin(), distributions.end()),
                                             fixture.lane)),
               std::runtime_error);
}

namespace {
struct GeneralFieldNorms {
  double l2;
  double linf;
  double relative_residual;
};

template <int Components>
GeneralFieldNorms solve_manufactured_general_field(int cells, bool constant_load = false,
                                                   bool incompatible = false) {
  constexpr int Dim = 2;
  const Box<Dim> domain{Index<Dim>{0, 0}, Index<Dim>{cells - 1, cells - 1}};
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
  const auto geometry =
      Geometry<Dim>::from_bounds(domain, RealVector<Dim>{0, 0}, RealVector<Dim>{1, 1});
  const auto topology =
      BoundaryTopology<Dim>::axis_periodic(std::array<bool, Dim>{Components == 1, Components == 1});
  const Extent<Dim> ghosts{1, 1};
  MultiFab<Dim> prototype(layout, distribution, Index<Dim>{}, Components, ghosts);
  MultiFab<Dim> first_load(layout, distribution, Index<Dim>{}, Components, ghosts);
  MultiFab<Dim> second_load(layout, distribution, Index<Dim>{}, Components, ghosts);
  MultiFab<Dim> rhs(layout, distribution, Index<Dim>{}, Components, ghosts);
  MultiFab<Dim> solution(layout, distribution, Index<Dim>{}, Components, ghosts);
  MultiFab<Dim> exact(layout, distribution, Index<Dim>{}, Components, ghosts);
  auto coefficients =
      std::make_shared<MultiFab<Dim>>(layout, distribution, Index<Dim>{}, Components, ghosts);
  prototype.set_val(Real(0));
  solution.set_val(Real(0));
  constexpr Real pi = Real(3.1415926535897932384626433832795);
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    const auto u = exact.fab(local).view();
    const auto a = coefficients->fab(local).view();
    const auto f1 = first_load.fab(local).view();
    const auto f2 = second_load.fab(local).view();
    for_each_cell(rhs.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      const Real x = geometry.cell_coordinate(0, cell[0]);
      const Real y = geometry.cell_coordinate(1, cell[1]);
      if constexpr (Components == 1) {
        const Real c = Kokkos::cos(Real(2) * pi * x) * Kokkos::cos(Real(2) * pi * y);
        const Real coefficient = Real(2) + Real(0.5) * Kokkos::sin(Real(2) * pi * x);
        const Real load = Real(8) * pi * pi * coefficient * c +
                          Real(2) * pi * pi * Kokkos::cos(Real(2) * pi * x) *
                              Kokkos::sin(Real(2) * pi * x) * Kokkos::cos(Real(2) * pi * y);
        u(cell, 0) = c;
        a(cell, 0) = coefficient;
        f1(cell, 0) = load / Real(2);
        f2(cell, 0) = load / Real(2);
      } else {
        const Real c1 = constant_load ? Real(0) : Kokkos::cos(pi * x) * Kokkos::cos(pi * y);
        const Real c2 =
            constant_load ? Real(0) : Kokkos::cos(Real(2) * pi * x) * Kokkos::cos(pi * y);
        u(cell, 0) = Real(0.375) + c1;
        u(cell, 1) = Real(-0.375) + Real(0.5) * c2;
        a(cell, 0) = a(cell, 1) = Real(1);
        f1(cell, 0) = Real(1.5) + (Real(2) * pi * pi + Real(2)) * c1 - c2;
        f1(cell, 1) = Real(0);
        f2(cell, 0) = Real(0);
        f2(cell, 1) = incompatible
                          ? Real(0)
                          : Real(-1.5) + (Real(2.5) * pi * pi + Real(1)) * c2 - Real(2) * c1;
      }
    });
    const auto load = rhs.fab(local).view();
    for_each_cell(rhs.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int component = 0; component < Components; ++component)
        load(cell, component) = f1(cell, component) + f2(cell, component);
    });
  }
  using Boundary = runtime::program::PreparedScalarBoundarySession<Dim>;
  const auto preparation_lane = ExecutionLane::world("test.general-field.coefficients");
  auto coefficient_boundary =
      Boundary::prepare(geometry, topology, *coefficients, preparation_lane, 1);
  elliptic::nd::prepare_general_field_coefficients(*coefficients, *coefficient_boundary);
  std::array<Real, Components * Components> reaction{};
  if constexpr (Components == 2)
    reaction = {Real(2), Real(-2), Real(-2), Real(2)};
  std::array<elliptic::nd::PhysicalFieldBoundary, 2 * Dim> physical{};
  physical.fill(Components == 1 ? elliptic::nd::PhysicalFieldBoundary::periodic
                                : elliptic::nd::PhysicalFieldBoundary::homogeneous_neumann);
  auto provider = PreparedAffineOperatorProvider<Dim>::trusted_extension(
      {"pops.test.general-field.manufactured", 1}, {}, [=](const ExecutionLane& lane) {
        auto scratch =
            std::make_shared<MultiFab<Dim>>(layout, distribution, Index<Dim>{}, Components, ghosts);
        auto boundary = Boundary::prepare(geometry, topology, *scratch, lane, 1);
        return PreparedAffineOperatorSessionCallbacks<Dim>{
            {},
            [=](MultiFab<Dim>& out, const MultiFab<Dim>& in) {
              PureFieldAlgebra::copy(*scratch, in);
              elliptic::nd::apply_general_field<Dim, Components>(out, *scratch, *coefficients,
                                                                 *boundary, reaction, physical);
            },
            [] { return std::size_t{0}; }};
      });
  auto nullspace = constant_mean_zero_nullspace<Dim>(
      "manufactured-shared-kernel", "one physical constant mode", Real(1) / Real(cells * cells));
  nullspace.bases.front().component_count = Components;
  OperatorEvaluationSnapshot snapshot{{11, 12, 13, 14},
                                      1,
                                      0,
                                      0,
                                      1,
                                      std::bit_cast<std::uint64_t>(1.0),
                                      0,
                                      1,
                                      detail::layout_fingerprint(prototype),
                                      {21, 22, 23, 24}};
  const KrylovFootprint<Dim> footprint{Components, ghosts, false};
  PreparedAffineLinearProblem<Dim> problem(
      prototype, std::move(provider), PreparedLinearPreconditioner<Dim>::identity(),
      LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
      PreparedNullspacePolicy<Dim>::preserving(std::move(nullspace)),
      [&snapshot] { return snapshot; }, {}, PreparedVectorDistribution<Dim>::Replicated);
  const auto method = cg_krylov_method<Dim>();
  KrylovWorkspace<Dim> workspace(prototype, method, footprint,
                                 PreparedVectorDistribution<Dim>::Replicated);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const auto report = detail::solve_prepared_affine_in_place(
      problem, workspace, solution, rhs,
      KrylovControls<Dim>{method, Real(1e-11), Real(1e-12), 4000});
  if (incompatible) {
    EXPECT_EQ(report.status, SolveStatus::kIncompatibleRhs);
    EXPECT_FALSE(report.solved());
    return {0, 0, 0};
  }
  EXPECT_TRUE(report.solved()) << report.reason;
  Real square_error = Real(0), max_error = Real(0);
  for (std::size_t local = 0; local < solution.local_size(); ++local) {
    const auto result = std::as_const(solution).fab(local).view();
    const auto reference = std::as_const(exact).fab(local).view();
    square_error +=
        for_each_cell_reduce_sum(solution.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          Real square = Real(0);
          for (int component = 0; component < Components; ++component) {
            const Real difference = result(cell, component) - reference(cell, component);
            square += difference * difference;
          }
          return square;
        });
    max_error =
        std::max(max_error,
                 for_each_cell_reduce_max(solution.box(local), [=] POPS_HD(const Index<Dim>& cell) {
                   Real error = Real(0);
                   for (int component = 0; component < Components; ++component)
                     error = Kokkos::max(
                         error, Kokkos::abs(result(cell, component) - reference(cell, component)));
                   return error;
                 }));
  }
  const GeneralFieldNorms norms{std::sqrt(static_cast<double>(square_error) / (cells * cells)),
                                static_cast<double>(max_error),
                                static_cast<double>(report.rel_residual)};
  std::cout << "FIELD_MMS components=" << Components << " cells=" << cells
            << " constant=" << constant_load << " l2=" << norms.l2 << " linf=" << norms.linf
            << " residual=" << norms.relative_residual << '\n';
  return norms;
}
}  // namespace

TEST(test_field_nullspace, general_fields_complete_predeclared_refinement_matrix) {
  comm_init();
  if (n_ranks() != 1)
    GTEST_SKIP() << "Declared native manufactured matrix is one MPI rank";
  double prior_scalar_l2 = 0, prior_scalar_linf = 0, prior_joint_l2 = 0, prior_joint_linf = 0;
  for (const int cells : {16, 32, 64}) {
    const auto scalar = solve_manufactured_general_field<1>(cells);
    const auto joint = solve_manufactured_general_field<2>(cells);
    const auto constants = solve_manufactured_general_field<2>(cells, true);
    EXPECT_LE(scalar.relative_residual, 1e-10);
    EXPECT_LE(joint.relative_residual, 1e-10);
    EXPECT_LE(constants.relative_residual, 1e-10);
    EXPECT_LE(constants.linf, 1e-10);
    if (cells > 16) {
      EXPECT_GT(std::log2(prior_scalar_l2 / scalar.l2), 1.8);
      EXPECT_GT(std::log2(prior_scalar_linf / scalar.linf), 1.8);
      EXPECT_GT(std::log2(prior_joint_l2 / joint.l2), 1.8);
      EXPECT_GT(std::log2(prior_joint_linf / joint.linf), 1.8);
    }
    if (cells == 64) {
      EXPECT_LT(scalar.l2, 0.002);
      EXPECT_LT(joint.l2, 0.002);
    }
    prior_scalar_l2 = scalar.l2;
    prior_scalar_linf = scalar.linf;
    prior_joint_l2 = joint.l2;
    prior_joint_linf = joint.linf;
    (void)solve_manufactured_general_field<2>(cells, true, true);
  }
}

TEST(test_field_nullspace, general_field_preflight_refuses_rank_local_storage_before_exchange) {
  comm_init();
  constexpr int Dim = 2;
  const Box<Dim> domain{Index<Dim>{0, 0}, Index<Dim>{3, 3}};
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> ranks{Index<Dim>{}, Extent<Dim>{n_ranks(), 1}};
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  const Index<Dim> local_rank{my_rank(), 0};
  const Extent<Dim> ghosts{1, 1};
  MultiFab<Dim> input(layout, distribution, local_rank, 1, ghosts);
  MultiFab<Dim> output(layout, distribution, local_rank, 1, ghosts);
  MultiFab<Dim> coefficient(layout, distribution, local_rank, my_rank() == 0 ? 2 : 1, ghosts);
  input.set_val(Real(0));
  coefficient.set_val(Real(1));
  const auto geometry =
      Geometry<Dim>::from_bounds(domain, RealVector<Dim>{0, 0}, RealVector<Dim>{1, 1});
  const auto topology = BoundaryTopology<Dim>::axis_periodic(std::array<bool, Dim>{true, true});
  const auto lane = ExecutionLane::world("field-preflight");
  auto boundary = runtime::program::PreparedScalarBoundarySession<Dim>::prepare(geometry, topology,
                                                                                input, lane, 1);
  std::array<elliptic::nd::PhysicalFieldBoundary, 2 * Dim> laws{};
  laws.fill(elliptic::nd::PhysicalFieldBoundary::periodic);
  EXPECT_THROW((elliptic::nd::apply_general_field<Dim, 1>(output, input, coefficient, *boundary,
                                                          std::array<Real, 1>{0}, laws)),
               std::invalid_argument);
}
