#include <gtest/gtest.h>

#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_bc_rec_adapter.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace pops;

namespace {

const std::array<PreparedVectorDistribution, 1> kDistributedLevel{
    PreparedVectorDistribution::Distributed};

struct TwoIslandFixture {
  Box2D domain = Box2D::from_extents(4, 2);
  BoxArray boxes = BoxArray(std::vector<Box2D>{domain});
  DistributionMapping mapping = DistributionMapping(1, 1);
  std::shared_ptr<MultiFab> labels = std::make_shared<MultiFab>(boxes, mapping, 1, 0);

  TwoIslandFixture() {
    Array4 values = labels->fab(0).array();
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
      for (int i = domain.lo[0]; i <= domain.hi[0]; ++i)
        values(i, j, 0) = i < 2 ? Real(1) : Real(2);
  }

  FieldNullspacePlan plan() const {
    return labelled_mean_zero_nullspace(
        "two-island-nullspace", "two-island-layout", {labels},
        {{1, "island-a", "fixture:cell-label:1"}, {2, "island-b", "fixture:cell-label:2"}}, {},
        {Real(0.5)}, 0, kDistributedLevel);
  }
};

}  // namespace

TEST(test_field_nullspace, canonicalizes_generic_boundary_facts_and_rejects_forged_sequences) {
  const FieldNullspaceOperatorFacts facts = make_field_nullspace_operator_facts(
      "test.boundary-set@1",
      {{"wall:z", FieldBoundaryNullspaceBehavior::Opaque},
       {"wall:a", FieldBoundaryNullspaceBehavior::PreservesConstantMode}},
      /*has_reaction=*/true);

  ASSERT_EQ(facts.boundaries.size(), 2U);
  EXPECT_EQ(facts.boundaries[0].boundary_id, "wall:a");
  EXPECT_EQ(facts.boundaries[1].boundary_id, "wall:z");
  EXPECT_TRUE(facts.has_reaction);
  EXPECT_NO_THROW((void)facts.exact_contract());

  FieldNullspaceOperatorFacts forged = facts;
  std::swap(forged.boundaries[0], forged.boundaries[1]);
  EXPECT_FALSE(forged.is_canonical());
  EXPECT_THROW((void)forged.exact_contract(), std::invalid_argument);

  EXPECT_THROW(
      (void)make_field_nullspace_operator_facts(
          "test.boundary-set@1",
          {{"wall:a", FieldBoundaryNullspaceBehavior::PreservesConstantMode},
           {"wall:a", FieldBoundaryNullspaceBehavior::ConstrainsConstantMode}},
          false),
      std::invalid_argument);
  EXPECT_THROW(
      (void)make_field_nullspace_operator_facts(
          "", {{"wall:a", FieldBoundaryNullspaceBehavior::PreservesConstantMode}}, false),
      std::invalid_argument);
  EXPECT_THROW(
      (void)make_field_nullspace_operator_facts(
          "test.boundary-set@1",
          {{"wall:a", static_cast<FieldBoundaryNullspaceBehavior>(255)}}, false),
      std::invalid_argument);

  const FieldNullspaceOperatorFacts boundaryless =
      make_field_nullspace_operator_facts("test.boundaryless-topology@1", {}, false);
  EXPECT_TRUE(boundaryless.is_canonical());
  EXPECT_TRUE(boundaryless.boundaries.empty());
  EXPECT_NO_THROW((void)boundaryless.exact_contract());
  EXPECT_FALSE(FieldNullspaceOperatorFacts{}.is_canonical());
}

TEST(test_field_nullspace, bc_rec_adapter_is_the_only_cartesian_boundary_mapping) {
  BCRec boundary;
  boundary.xhi = BCType::Dirichlet;
  boundary.ylo = BCType::Robin;
  boundary.ylo_alpha = Real(0);
  boundary.yhi = BCType::External;

  const FieldNullspaceOperatorFacts facts =
      field_nullspace_operator_facts_from_bc_rec(boundary, false);
  ASSERT_EQ(facts.boundaries.size(), 4U);
  EXPECT_EQ(facts.boundary_set_identity, "pops.mesh.boundary.bc-rec.cartesian-2d@1");
  EXPECT_EQ(facts.boundaries[0].boundary_id, "axis:0:lower");
  EXPECT_EQ(facts.boundaries[0].behavior,
            FieldBoundaryNullspaceBehavior::PreservesConstantMode);
  EXPECT_EQ(facts.boundaries[1].boundary_id, "axis:0:upper");
  EXPECT_EQ(facts.boundaries[1].behavior,
            FieldBoundaryNullspaceBehavior::ConstrainsConstantMode);
  EXPECT_EQ(facts.boundaries[2].boundary_id, "axis:1:lower");
  EXPECT_EQ(facts.boundaries[2].behavior,
            FieldBoundaryNullspaceBehavior::PreservesConstantMode);
  EXPECT_EQ(facts.boundaries[3].boundary_id, "axis:1:upper");
  EXPECT_EQ(facts.boundaries[3].behavior, FieldBoundaryNullspaceBehavior::Opaque);

  const auto provider = make_default_field_nullspace_provider_registry()->resolve(
      "pops.field-nullspace.operator-topology-derived");
  EXPECT_EQ(provider->interface_version(), 2U);
  EXPECT_EQ(provider->collective_contract(),
            "pops.field-nullspace.operator-topology-derived@2");
  EXPECT_EQ(provider->default_options().schema_identity,
            "pops.field-nullspace.operator-topology-derived.options@1");
}

TEST(test_field_nullspace, materializes_one_basis_and_gauge_per_connected_component) {
  TwoIslandFixture fixture;
  const FieldNullspacePlan plan = fixture.plan();

  ASSERT_EQ(plan.bases.size(), 2U);
  ASSERT_EQ(plan.gauges.size(), 2U);
  EXPECT_EQ(plan.bases[0].identity, "island-a");
  EXPECT_EQ(plan.bases[1].identity, "island-b");
  EXPECT_EQ(plan.gauges[0].basis_identity, "island-a");
  EXPECT_EQ(plan.gauges[1].basis_identity, "island-b");

  const ConstArray4 left = plan.bases[0].masks[0]->fab(0).const_array();
  const ConstArray4 right = plan.bases[1].masks[0]->fab(0).const_array();
  for (int j = fixture.domain.lo[1]; j <= fixture.domain.hi[1]; ++j) {
    for (int i = fixture.domain.lo[0]; i <= fixture.domain.hi[0]; ++i) {
      EXPECT_EQ(left(i, j, 0), i < 2 ? Real(1) : Real(0));
      EXPECT_EQ(right(i, j, 0), i < 2 ? Real(0) : Real(1));
    }
  }
}

TEST(test_field_nullspace, labelled_topology_preserves_an_arbitrary_target_field_component) {
  TwoIslandFixture fixture;
  const FieldNullspacePlan plan = labelled_mean_zero_nullspace(
      "component-three-nullspace", "component-three-layout", {fixture.labels},
      {{1, "island-a", "fixture:cell-label:1"}, {2, "island-b", "fixture:cell-label:2"}}, {},
      {Real(0.5)}, 3, kDistributedLevel);

  ASSERT_EQ(plan.bases.size(), 2U);
  EXPECT_EQ(plan.bases[0].field_component, 3);
  EXPECT_EQ(plan.bases[1].field_component, 3);
}

TEST(test_field_nullspace, labelled_level_local_plan_uses_absolute_level_metadata) {
  TwoIslandFixture fixture;
  const FieldNullspacePlan plan = labelled_mean_zero_nullspace(
      "level-one-components", "level-one-layout", {fixture.labels},
      {{1, "island-a", "fixture:cell-label:1"}, {2, "island-b", "fixture:cell-label:2"}}, {},
      {Real(0.5)}, 0, kDistributedLevel, 1);

  ASSERT_EQ(plan.bases.size(), 2U);
  for (const FieldNullspaceBasis& basis : plan.bases) {
    ASSERT_EQ(basis.masks.size(), 2U);
    EXPECT_EQ(basis.masks[0], nullptr);
    EXPECT_NE(basis.masks[1], nullptr);
    ASSERT_EQ(basis.cell_measure.size(), 2U);
    EXPECT_EQ(basis.cell_measure[0], Real(0));
    EXPECT_EQ(basis.cell_measure[1], Real(0.5));
  }
  EXPECT_NO_THROW(
      validate_field_nullspace_basis({fixture.labels.get()}, plan, kDistributedLevel, 1));
}

TEST(test_field_nullspace, checks_rhs_and_applies_gauges_component_by_component) {
  TwoIslandFixture fixture;
  const FieldNullspacePlan plan = fixture.plan();
  MultiFab rhs(fixture.boxes, fixture.mapping, 1, 0);
  MultiFab phi(fixture.boxes, fixture.mapping, 1, 0);
  Array4 r = rhs.fab(0).array();
  Array4 p = phi.fab(0).array();
  for (int j = fixture.domain.lo[1]; j <= fixture.domain.hi[1]; ++j) {
    r(0, j, 0) = Real(1);
    r(1, j, 0) = Real(-1);
    r(2, j, 0) = Real(2);
    r(3, j, 0) = Real(-2);
    p(0, j, 0) = p(1, j, 0) = Real(3);
    p(2, j, 0) = p(3, j, 0) = Real(-5);
  }

  const std::vector<double> witness = require_field_nullspace_compatible(rhs, plan);
  ASSERT_EQ(witness.size(), 4U);
  EXPECT_EQ(witness[0], 0.0);
  EXPECT_EQ(witness[2], 0.0);

  apply_field_gauge(phi, plan);
  for (int j = fixture.domain.lo[1]; j <= fixture.domain.hi[1]; ++j)
    for (int i = fixture.domain.lo[0]; i <= fixture.domain.hi[0]; ++i)
      EXPECT_NEAR(p(i, j, 0), Real(0), Real(1e-13));

  r(0, 0, 0) += Real(1);
  EXPECT_THROW(require_field_nullspace_compatible(rhs, plan), std::runtime_error);
}

TEST(test_field_nullspace, prepared_workspace_supports_overlapping_independent_bases_without_hot_allocations) {
  const Box2D domain = Box2D::from_extents(2, 1);
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping mapping(1, 1);
  auto first = std::make_shared<MultiFab>(boxes, mapping, 1, 0);
  auto second = std::make_shared<MultiFab>(boxes, mapping, 1, 0);
  first->fab(0).array()(0, 0, 0) = Real(1);
  first->fab(0).array()(1, 0, 0) = Real(1);
  second->fab(0).array()(0, 0, 0) = Real(1);
  second->fab(0).array()(1, 0, 0) = Real(2);

  FieldNullspacePlan plan;
  plan.identity = "overlapping-independent";
  plan.layout_identity = "overlapping-independent-layout";
  plan.bases = {{"first", "unit-test:first", "unit-test:first@1", 0, {first}, {}, {Real(1)}},
                {"second", "unit-test:second", "unit-test:second@1", 0, {second}, {},
                 {Real(1)}}};
  plan.gauges = {{"first", Real(0)}, {"second", Real(0)}};

  MultiFab prepared_layout(boxes, mapping, 1, 0);
  // Compatibility is a property of the valid vector space.  A caller may use
  // a different ghost width for its RHS and iterate without changing that
  // space or requiring a second nullspace preparation.
  MultiFab value(boxes, mapping, 1, 1);
  value.fab(0).array()(0, 0, 0) = Real(1);   // 3*first - 2*second
  value.fab(0).array()(1, 0, 0) = Real(-1);
  FieldNullspaceWorkspace workspace(
      plan, {&prepared_layout}, {PreparedVectorDistribution::Distributed});

  const AllocationEventStats before = allocation_event_stats();
  workspace.apply_gauge(value);
  const std::span<const double> witness = workspace.require_compatible(value);
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_NEAR(value.fab(0).const_array()(0, 0, 0), Real(0), Real(1e-13));
  EXPECT_NEAR(value.fab(0).const_array()(1, 0, 0), Real(0), Real(1e-13));
  ASSERT_EQ(witness.size(), 4U);
  EXPECT_NEAR(witness[0], 0.0, 1e-13);
  EXPECT_NEAR(witness[2], 0.0, 1e-13);
  EXPECT_EQ(after, before);
}

TEST(test_field_nullspace, rejects_invalid_or_undeclared_labels_collectively) {
  TwoIslandFixture fixture;
  fixture.labels->fab(0).array()(0, 0, 0) = Real(3);
  EXPECT_THROW(fixture.plan(), std::runtime_error);

  fixture.labels->fab(0).array()(0, 0, 0) = Real(1.5);
  EXPECT_THROW(fixture.plan(), std::runtime_error);
}

TEST(test_field_nullspace, rejects_a_gauge_that_references_an_unknown_basis) {
  TwoIslandFixture fixture;
  FieldNullspacePlan plan = fixture.plan();
  plan.gauges[0].basis_identity = "missing-island";

  EXPECT_THROW(validate_field_nullspace_basis({fixture.labels.get()}, plan, kDistributedLevel),
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

TEST(test_field_nullspace, level_local_plan_validates_only_its_resolved_absolute_level) {
  const Box2D domain = Box2D::from_extents(2, 2);
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping mapping(1, 1);
  MultiFab field(boxes, mapping, 1, 0);
  field.set_val(Real(0));

  FieldNullspacePlan plan =
      constant_mean_zero_nullspace("level-one-nullspace", "unit-test", Real(0.25));
  plan.bases[0].cell_measure = {Real(0), Real(0.25)};
  auto active_mask = std::make_shared<MultiFab>(field);
  active_mask->set_val(Real(1));
  plan.bases[0].masks = {nullptr, active_mask};

  EXPECT_NO_THROW(validate_field_nullspace_basis({&field}, plan, kDistributedLevel, 1));
  EXPECT_NO_THROW(
      (void)require_field_nullspace_compatible({&field}, plan, kDistributedLevel, 1));
  EXPECT_NO_THROW(apply_field_gauge({&field}, plan, kDistributedLevel, 1));

  FieldNullspacePlan zero_active_measure = plan;
  zero_active_measure.bases[0].cell_measure[1] = Real(0);
  EXPECT_THROW(validate_field_nullspace_basis({&field}, zero_active_measure, kDistributedLevel, 1),
               std::runtime_error);

  FieldNullspacePlan missing_active_mask = plan;
  missing_active_mask.bases[0].masks[1].reset();
  EXPECT_THROW(validate_field_nullspace_basis({&field}, missing_active_mask, kDistributedLevel, 1),
               std::runtime_error);

  const std::array<PreparedVectorDistribution, 0> missing_distribution{};
  EXPECT_THROW(validate_field_nullspace_basis({&field}, plan, missing_distribution, 1),
               std::runtime_error);

  FieldNullspacePlan uniform_with_zero =
      constant_mean_zero_nullspace("uniform-nullspace", "unit-test", Real(1));
  uniform_with_zero.bases[0].cell_measure.push_back(Real(0));
  EXPECT_NO_THROW(
      validate_field_nullspace_basis({&field}, uniform_with_zero, kDistributedLevel));
}
