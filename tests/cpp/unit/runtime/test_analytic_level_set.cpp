#include <gtest/gtest.h>

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/runtime/analytic/expression.hpp>
#include <pops/runtime/analytic/level_set.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

using pops::Real;
using pops::analytic::AnalyticNode;
using pops::analytic::AnalyticOp;
using pops::analytic::compile_analytic_expression;
using pops::analytic::make_analytic_level_set;
using pops::analytic::materialize_analytic_level_set;
using pops::analytic::replace_analytic_level_set_materialization;

AnalyticNode unary(AnalyticOp op, AnalyticNode value) {
  return AnalyticNode::apply(op, {std::move(value)});
}

AnalyticNode binary(AnalyticOp op, AnalyticNode left, AnalyticNode right) {
  return AnalyticNode::apply(op, {std::move(left), std::move(right)});
}

AnalyticNode square(AnalyticNode value) {
  return binary(AnalyticOp::Mul, value, value);
}

template <int Dim>
AnalyticNode ranked_radius_level_set(Real radius) {
  AnalyticNode squared_radius = square(AnalyticNode::x());
  if constexpr (Dim >= 2)
    squared_radius = binary(AnalyticOp::Add, std::move(squared_radius), square(AnalyticNode::y()));
  if constexpr (Dim >= 3)
    squared_radius = binary(AnalyticOp::Add, std::move(squared_radius), square(AnalyticNode::z()));
  return binary(AnalyticOp::Sub, unary(AnalyticOp::Sqrt, std::move(squared_radius)),
                AnalyticNode::constant(radius));
}

template <int Dim>
pops::Extent<Dim> uniform_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Box<Dim> execution_box() {
  constexpr std::int64_t width = Dim == 1 ? 4097 : (Dim == 2 ? 65 : 17);
  return pops::Box<Dim>::from_extents(uniform_extent<Dim>(width));
}

template <int Dim>
pops::Geometry<Dim> centered_geometry(const pops::Box<Dim>& domain) {
  pops::RealVector<Dim> lower{};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = Real(-1);
    upper[axis] = Real(1);
  }
  return pops::Geometry<Dim>::from_bounds(domain, lower, upper);
}

template <int Dim, class Function>
void for_each_host_index(const pops::Box<Dim>& box, Function&& function) {
  for (std::int64_t linear = 0; linear < box.numPts(); ++linear) {
    std::int64_t remaining = linear;
    pops::Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    function(index);
  }
}

template <int Dim>
std::size_t host_offset(const pops::Box<Dim>& storage, const pops::Index<Dim>& index,
                        int component = 0) {
  std::int64_t linear = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + linear);
}

template <int Dim>
void check_ranked_materialization() {
  const pops::Box<Dim> domain = execution_box<Dim>();
  const pops::Geometry<Dim> geometry = centered_geometry(domain);
  const auto program = compile_analytic_expression(ranked_radius_level_set<Dim>(Real(0.55)));
  ASSERT_EQ(program.required_dimension(), Dim);

  const auto materialized = materialize_analytic_level_set(program, geometry, domain, 2);
  EXPECT_EQ(materialized.box(), domain);
  EXPECT_EQ(materialized.grown_box(), domain.grow(2));
  EXPECT_EQ(materialized.n_ghost(), 2);

  auto phi = materialized.values.create_host_mirror();
  auto mask = materialized.active_mask.create_host_mirror();
  materialized.values.copy_to_host(phi);
  materialized.active_mask.copy_to_host(mask);
  const pops::Box<Dim> sampled = domain.grow(2);
  for_each_host_index(sampled, [&](const pops::Index<Dim>& index) {
    const pops::RealVector<Dim> point = geometry.cell_center(index);
    Real squared_radius = Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      squared_radius += point[axis] * point[axis];
    const Real expected = std::sqrt(squared_radius) - Real(0.55);
    ASSERT_NEAR(phi(host_offset(sampled, index)), expected, Real(2e-14));
    EXPECT_DOUBLE_EQ(mask(host_offset(sampled, index)), expected < Real(0) ? Real(1) : Real(0));
  });
}

TEST(AnalyticLevelSet, CallableUsesTheStrictNegativeActiveConvention) {
  const auto program = compile_analytic_expression(ranked_radius_level_set<2>(Real(1)));
  const auto level_set = make_analytic_level_set<2>(program);

  EXPECT_LT(level_set(pops::RealVector<2>{Real(0), Real(0)}), Real(0));
  EXPECT_TRUE(level_set.cell_active(pops::RealVector<2>{Real(0), Real(0)}));
  EXPECT_DOUBLE_EQ(level_set(pops::RealVector<2>{Real(1), Real(0)}), Real(0));
  EXPECT_FALSE(level_set.cell_active(pops::RealVector<2>{Real(1), Real(0)}));
  EXPECT_GT(level_set.level_set(pops::RealVector<2>{Real(2), Real(0)}), Real(0));
}

TEST(AnalyticLevelSet, MaterializationIncludesEveryGhostInOneTwoAndThreeDimensions) {
  check_ranked_materialization<1>();
  check_ranked_materialization<2>();
  check_ranked_materialization<3>();
}

TEST(AnalyticLevelSet, NonFiniteReplacementLeavesRankedPublishedFieldsUntouched) {
  const pops::Box<3> domain = pops::Box<3>::from_extents(pops::Extent<3>{4, 3, 2});
  const pops::Geometry<3> geometry = centered_geometry(domain);
  const auto finite = compile_analytic_expression(AnalyticNode::constant(Real(-2)));
  pops::analytic::AnalyticLevelSetMaterialization<3> published =
      materialize_analytic_level_set(finite, geometry, domain, 1);
  const Real* const old_values = published.values.view().data;
  const Real* const old_mask = published.active_mask.view().data;

  const AnalyticNode zero = binary(AnalyticOp::Sub, AnalyticNode::x(), AnalyticNode::x());
  const auto non_finite =
      compile_analytic_expression(binary(AnalyticOp::Div, AnalyticNode::constant(Real(1)), zero));
  EXPECT_THROW(
      replace_analytic_level_set_materialization(published, non_finite, geometry, domain, 1),
      std::domain_error);

  EXPECT_EQ(published.values.view().data, old_values);
  EXPECT_EQ(published.active_mask.view().data, old_mask);
  auto values = published.values.create_host_mirror();
  auto mask = published.active_mask.create_host_mirror();
  published.values.copy_to_host(values);
  published.active_mask.copy_to_host(mask);
  for_each_host_index(published.grown_box(), [&](const pops::Index<3>& index) {
    EXPECT_DOUBLE_EQ(values(host_offset(published.grown_box(), index)), Real(-2));
    EXPECT_DOUBLE_EQ(mask(host_offset(published.grown_box(), index)), Real(1));
  });
}

TEST(AnalyticLevelSet, PredicateRankAndInvalidSamplingRequestsFailBeforePublication) {
  const auto predicate = compile_analytic_expression(
      binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::constant(Real(0))));
  EXPECT_THROW(make_analytic_level_set<2>(predicate), std::invalid_argument);

  const pops::Box<2> domain = pops::Box<2>::from_extents(pops::Extent<2>{8, 8});
  const pops::Geometry<2> geometry = centered_geometry(domain);
  const auto z_program = compile_analytic_expression(AnalyticNode::z());
  EXPECT_EQ(z_program.required_dimension(), 3);
  EXPECT_THROW(make_analytic_level_set<2>(z_program), std::invalid_argument);
  EXPECT_THROW(materialize_analytic_level_set(z_program, geometry, domain, 0),
               std::invalid_argument);

  const auto scalar = compile_analytic_expression(AnalyticNode::x());
  EXPECT_THROW(materialize_analytic_level_set(scalar, geometry, pops::Box<2>{}, 1),
               std::invalid_argument);
  EXPECT_THROW(materialize_analytic_level_set(scalar, geometry, domain, -1), std::invalid_argument);
  pops::Box<2> outside = domain;
  --outside.lo[0];
  EXPECT_THROW(materialize_analytic_level_set(scalar, geometry, outside, 0), std::invalid_argument);
}

}  // namespace
