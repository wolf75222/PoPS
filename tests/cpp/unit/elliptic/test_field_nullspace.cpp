#include <gtest/gtest.h>

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
        std::span<const PreparedVectorDistribution<Dim>>(prepared_distributions), first_level);
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
      0);
  ASSERT_EQ(witness.size(), 4U);
  EXPECT_NEAR(witness[0], 0.0, 1e-13);
  EXPECT_NEAR(witness[2], 0.0, 1e-13);

  const std::vector<MultiFab<Dim>*> phi_levels{&phi};
  apply_field_gauge<Dim>(phi_levels, plan,
                         std::span<const PreparedVectorDistribution<Dim>>(prepared_distributions));
  EXPECT_NEAR(maximum_absolute_value(phi), Real(0), Real(1e-13));

  fill(phi, TestFieldPattern::island_constants);
  FieldNullspaceWorkspace<Dim> workspace(
      plan, {fixture.labels.get()},
      std::vector<PreparedVectorDistribution<Dim>>(prepared_distributions.begin(),
                                                   prepared_distributions.end()));
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
      std::vector<PreparedVectorDistribution<Dim>>(distributions.begin(), distributions.end()));

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
  EXPECT_NO_THROW(validate_field_nullspace_basis<Dim>(const_levels, plan, distribution_span, 1));
  EXPECT_NO_THROW(
      (void)require_field_nullspace_compatible<Dim>(const_levels, plan, distribution_span, 1));
  EXPECT_NO_THROW(apply_field_gauge<Dim>(mutable_levels, plan, distribution_span, 1));

  FieldNullspacePlan<Dim> zero_measure = plan;
  zero_measure.bases[0].cell_measure[1] = Real(0);
  EXPECT_THROW(
      validate_field_nullspace_basis<Dim>(const_levels, zero_measure, distribution_span, 1),
      std::runtime_error);

  FieldNullspacePlan<Dim> missing_mask = plan;
  missing_mask.bases[0].masks[1].reset();
  EXPECT_THROW(
      validate_field_nullspace_basis<Dim>(const_levels, missing_mask, distribution_span, 1),
      std::runtime_error);

  const std::array<PreparedVectorDistribution<Dim>, 0> missing_distribution{};
  EXPECT_THROW(validate_field_nullspace_basis<Dim>(
                   const_levels, plan,
                   std::span<const PreparedVectorDistribution<Dim>>(missing_distribution), 1),
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
                   layouts, plan, std::span<const PreparedVectorDistribution<2>>(distributions)),
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
