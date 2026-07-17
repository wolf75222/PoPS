#include <gtest/gtest.h>

#include <pops/numerics/elliptic/interface/field_nullspace.hpp>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace pops;

namespace {

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
        "two-island-nullspace", "two-island-layout", FieldNullspaceScope::Uniform, {labels},
        {{1, "island-a", "fixture:cell-label:1"}, {2, "island-b", "fixture:cell-label:2"}}, {},
        {Real(0.5)}, 0);
  }
};

}  // namespace

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
      "component-three-nullspace", "component-three-layout", FieldNullspaceScope::Uniform,
      {fixture.labels},
      {{1, "island-a", "fixture:cell-label:1"}, {2, "island-b", "fixture:cell-label:2"}}, {},
      {Real(0.5)}, 3);

  ASSERT_EQ(plan.bases.size(), 2U);
  EXPECT_EQ(plan.bases[0].field_component, 3);
  EXPECT_EQ(plan.bases[1].field_component, 3);
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
      EXPECT_EQ(p(i, j, 0), Real(0));

  r(0, 0, 0) += Real(1);
  EXPECT_THROW(require_field_nullspace_compatible(rhs, plan), std::runtime_error);
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

  EXPECT_THROW(validate_field_nullspace_basis({fixture.labels.get()}, plan), std::runtime_error);
}

TEST(test_field_nullspace, validates_native_collective_capacities_before_size_arithmetic) {
  const std::size_t native_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  std::size_t gram_edge = 1;
  while (gram_edge <= native_max / gram_edge)
    ++gram_edge;
  --gram_edge;
  EXPECT_EQ(detail::checked_field_nullspace_collective_product(
                gram_edge, gram_edge, "synthetic Gram matrix"),
            gram_edge * gram_edge);
  EXPECT_THROW(detail::checked_field_nullspace_collective_product(
                   gram_edge + 1, gram_edge + 1, "synthetic Gram matrix"),
               std::overflow_error);

  EXPECT_EQ(detail::checked_field_nullspace_collective_product(
                native_max / 2, std::size_t{2}, "synthetic moments"),
            (native_max / 2) * 2);
  EXPECT_THROW(detail::checked_field_nullspace_collective_product(
                   native_max / 2 + 1, std::size_t{2}, "synthetic moments"),
               std::overflow_error);

  EXPECT_EQ(detail::checked_field_nullspace_collective_sum(
                native_max - 1, std::size_t{1}, "synthetic label counts"),
            native_max);
  EXPECT_THROW(detail::checked_field_nullspace_collective_sum(
                   native_max, std::size_t{1}, "synthetic label counts"),
               std::overflow_error);
  EXPECT_EQ(detail::checked_field_nullspace_collective_count(native_max,
                                                              "synthetic collective"),
            std::numeric_limits<int>::max());
  EXPECT_THROW(detail::checked_field_nullspace_collective_count(native_max + 1,
                                                                 "synthetic collective"),
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
