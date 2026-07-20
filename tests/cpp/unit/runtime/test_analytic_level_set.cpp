#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/runtime/analytic/expression.hpp>
#include <pops/runtime/analytic/level_set.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

using pops::Box2D;
using pops::Geometry;
using pops::Real;
using pops::analytic::AnalyticLevelSetMaterialization;
using pops::analytic::AnalyticNode;
using pops::analytic::AnalyticOp;
using pops::analytic::compile_analytic_expression;
using pops::analytic::make_analytic_level_set;
using pops::analytic::materialize_analytic_level_set;
using pops::analytic::replace_analytic_level_set_materialization;

AnalyticNode binary(AnalyticOp op, AnalyticNode left, AnalyticNode right) {
  return AnalyticNode::apply(op, {std::move(left), std::move(right)});
}

AnalyticNode circle_level_set(Real cx, Real cy, Real radius) {
  const AnalyticNode dx =
      binary(AnalyticOp::Sub, AnalyticNode::x(), AnalyticNode::constant(cx));
  const AnalyticNode dy =
      binary(AnalyticOp::Sub, AnalyticNode::y(), AnalyticNode::constant(cy));
  return binary(AnalyticOp::Sub,
                binary(AnalyticOp::Hypot, dx, dy), AnalyticNode::constant(radius));
}

TEST(AnalyticLevelSet, CallableUsesTheStrictNegativeActiveConvention) {
  const auto program = compile_analytic_expression(circle_level_set(Real(0), Real(0), Real(1)));
  const auto level_set = make_analytic_level_set(program);

  EXPECT_LT(level_set(Real(0), Real(0)), Real(0));
  EXPECT_TRUE(level_set.cell_active(Real(0), Real(0)));
  EXPECT_DOUBLE_EQ(level_set(Real(1), Real(0)), Real(0));
  EXPECT_FALSE(level_set.cell_active(Real(1), Real(0)));
  EXPECT_GT(level_set.level_set(Real(2), Real(0)), Real(0));
}

TEST(AnalyticLevelSet, MaterializationIncludesEveryGhostAndRunsThroughKokkos) {
  const Box2D domain = Box2D::from_extents(65, 65);
  const Geometry geometry{domain, Real(-1), Real(1), Real(-1), Real(1)};
  const auto program =
      compile_analytic_expression(circle_level_set(Real(0.1), Real(-0.2), Real(0.55)));

  const auto materialized = materialize_analytic_level_set(program, geometry, domain, 2);
  EXPECT_EQ(materialized.box(), domain);
  EXPECT_EQ(materialized.grown_box(), domain.grow(2));
  EXPECT_EQ(materialized.n_ghost(), 2);

  pops::sync_host();
  const auto phi = materialized.values.const_array();
  const auto mask = materialized.active_mask.const_array();
  const Box2D sampled = domain.grow(2);
  for (int j = sampled.lo[1]; j <= sampled.hi[1]; ++j)
    for (int i = sampled.lo[0]; i <= sampled.hi[0]; ++i) {
      const Real x = geometry.x_cell(i);
      const Real y = geometry.y_cell(j);
      const Real expected = std::hypot(x - Real(0.1), y + Real(0.2)) - Real(0.55);
      ASSERT_NEAR(phi(i, j), expected, 1e-14) << "cell (" << i << "," << j << ")";
      EXPECT_DOUBLE_EQ(mask(i, j), expected < Real(0) ? Real(1) : Real(0))
          << "cell (" << i << "," << j << ")";
    }
}

TEST(AnalyticLevelSet, NonFiniteReplacementLeavesPublishedFieldsUntouched) {
  const Box2D domain = Box2D::from_extents(16, 12);
  const Geometry geometry{domain, Real(-1), Real(1), Real(-1), Real(1)};
  const auto finite = compile_analytic_expression(AnalyticNode::constant(Real(-2)));
  AnalyticLevelSetMaterialization published =
      materialize_analytic_level_set(finite, geometry, domain, 1);
  pops::sync_host();
  const Real* const old_values = published.values.data();
  const Real* const old_mask = published.active_mask.data();
  const Real old_first_value = old_values[0];
  const Real old_first_mask = old_mask[0];

  const AnalyticNode zero =
      binary(AnalyticOp::Sub, AnalyticNode::x(), AnalyticNode::x());
  const auto non_finite = compile_analytic_expression(
      binary(AnalyticOp::Div, AnalyticNode::constant(Real(1)), zero));
  EXPECT_THROW(replace_analytic_level_set_materialization(
                   published, non_finite, geometry, domain, 1),
               std::domain_error);

  pops::sync_host();
  EXPECT_EQ(published.values.data(), old_values);
  EXPECT_EQ(published.active_mask.data(), old_mask);
  EXPECT_DOUBLE_EQ(published.values.data()[0], old_first_value);
  EXPECT_DOUBLE_EQ(published.active_mask.data()[0], old_first_mask);
}

TEST(AnalyticLevelSet, PredicateAndInvalidSamplingRequestsFailBeforePublication) {
  const auto predicate = compile_analytic_expression(binary(
      AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::constant(Real(0))));
  EXPECT_THROW(make_analytic_level_set(predicate), std::invalid_argument);

  const Box2D domain = Box2D::from_extents(8, 8);
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const auto scalar = compile_analytic_expression(AnalyticNode::x());
  EXPECT_THROW(materialize_analytic_level_set(scalar, geometry, Box2D{}, 1),
               std::invalid_argument);
  EXPECT_THROW(materialize_analytic_level_set(scalar, geometry, domain, -1),
               std::invalid_argument);
  EXPECT_THROW(materialize_analytic_level_set(
                   scalar, geometry, Box2D{{-1, 0}, {domain.hi[0], domain.hi[1]}}, 0),
               std::invalid_argument);
}

}  // namespace
