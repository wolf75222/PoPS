#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/analytic/expression.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using pops::Real;
using pops::analytic::AnalyticLimits;
using pops::analytic::AnalyticNode;
using pops::analytic::AnalyticOp;
using pops::analytic::AnalyticProgramView;
using pops::analytic::AnalyticToken;
using pops::analytic::AnalyticValueType;
using pops::analytic::analytic_op_from_name;
using pops::analytic::compile_analytic_expression;
using pops::analytic::compile_analytic_postfix;

AnalyticNode unary(AnalyticOp op, AnalyticNode value) {
  return AnalyticNode::apply(op, {std::move(value)});
}

AnalyticNode binary(AnalyticOp op, AnalyticNode left, AnalyticNode right) {
  return AnalyticNode::apply(op, {std::move(left), std::move(right)});
}

AnalyticNode select(AnalyticNode condition, AnalyticNode selected, AnalyticNode otherwise) {
  return AnalyticNode::apply(AnalyticOp::Select,
                             {std::move(condition), std::move(selected), std::move(otherwise)});
}

AnalyticNode diocotron_density() {
  auto radius = [] { return binary(AnalyticOp::Hypot, AnalyticNode::x(), AnalyticNode::y()); };
  const AnalyticNode annulus = AnalyticNode::apply(
      AnalyticOp::Between,
      {radius(), AnalyticNode::constant(Real(0.35)), AnalyticNode::constant(Real(0.40))});
  const AnalyticNode theta = binary(AnalyticOp::Atan2, AnalyticNode::y(), AnalyticNode::x());
  const AnalyticNode modulation = binary(
      AnalyticOp::Add, AnalyticNode::constant(Real(0.9)),
      binary(
          AnalyticOp::Mul, AnalyticNode::constant(Real(0.1)),
          unary(AnalyticOp::Sin, binary(AnalyticOp::Mul, AnalyticNode::constant(Real(4)), theta))));
  return select(annulus, modulation, AnalyticNode::constant(Real(1e-4)));
}

template <int Dim>
AnalyticNode coordinate_sum() {
  AnalyticNode result = AnalyticNode::x();
  if constexpr (Dim >= 2)
    result = binary(AnalyticOp::Add, std::move(result), AnalyticNode::y());
  if constexpr (Dim >= 3)
    result = binary(AnalyticOp::Add, std::move(result), AnalyticNode::z());
  return result;
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
    lower[axis] = Real(-axis - 1);
    upper[axis] = Real(axis + 1);
  }
  return pops::Geometry<Dim>::from_bounds(domain, lower, upper);
}

template <int Dim>
pops::Geometry<Dim> unit_geometry(const pops::Box<Dim>& domain) {
  pops::RealVector<Dim> lower{};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real(1);
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
pops::MultiFab<Dim> one_patch_field(const pops::Box<Dim>& box, int ncomp,
                                    pops::Extent<Dim> ghosts = {}) {
  const pops::mesh::BoxArray<Dim> layout(std::vector<pops::Box<Dim>>{box});
  const pops::mesh::RankSpace<Dim> ranks(pops::Index<Dim>{}, uniform_extent<Dim>(1));
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  return pops::MultiFab<Dim>(layout, distribution, pops::Index<Dim>{}, ncomp, ghosts);
}

template <int Dim>
struct EvaluateAnalyticKernel {
  AnalyticProgramView expression;
  pops::Geometry<Dim> geometry;
  pops::FieldView<Real, Dim> output;

  POPS_HD void operator()(const pops::Index<Dim>& index) const {
    output(index) = expression.eval(index, geometry);
  }
};

template <int Dim>
void check_program_view_kernel() {
  const auto program = compile_analytic_expression(coordinate_sum<Dim>());
  EXPECT_EQ(program.required_dimension(), Dim);
  const pops::Box<Dim> box = execution_box<Dim>();
  const pops::Geometry<Dim> geometry = centered_geometry(box);
  pops::Fab<Dim> output(box, 1);

  pops::for_each_cell(box, EvaluateAnalyticKernel<Dim>{program.view(), geometry, output.view()});
  auto host = output.create_host_mirror();
  output.copy_to_host(host);

  std::array<pops::Index<Dim>, 3> samples{};
  samples[0] = box.lo;
  samples[2] = box.hi;
  for (int axis = 0; axis < Dim; ++axis)
    samples[1][axis] = box.lo[axis] + static_cast<int>(box.length(axis) / 2);
  for (const auto& index : samples) {
    const auto point = geometry.cell_center(index);
    Real expected = Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      expected += point[axis];
    EXPECT_NEAR(host(host_offset(output.grown_box(), index)), expected, Real(2e-14));
  }
}

template <int Dim>
void check_initial_materializers() {
  pops::Extent<Dim> extents{};
  for (int axis = 0; axis < Dim; ++axis)
    extents[axis] = 5 + axis;
  const pops::Box<Dim> box = pops::Box<Dim>::from_extents(extents);
  const pops::Geometry<Dim> geometry = unit_geometry(box);
  auto values = one_patch_field(box, 1);

  {
    std::vector<pops::analytic::AnalyticProgram> programs;
    programs.push_back(compile_analytic_expression(AnalyticNode::constant(Real(3.25))));
    EXPECT_EQ(pops::analytic::materialize_cell_average(values, geometry, programs), box.numPts());
  }

  const auto& field = values.fab(0);
  auto constant_host = field.create_host_mirror();
  field.copy_to_host(constant_host);
  for (const pops::Index<Dim>& index : std::array{box.lo, box.hi})
    EXPECT_DOUBLE_EQ(constant_host(host_offset(field.grown_box(), index)), Real(3.25));

  pops::RealVector<Dim> center{};
  for (int axis = 0; axis < Dim; ++axis)
    center[axis] = Real(0.5);
  EXPECT_EQ(pops::analytic::materialize_gaussian_cell_average(values, geometry, center, Real(1.25),
                                                              Real(0), Real(20)),
            box.numPts());
  auto gaussian_host = field.create_host_mirror();
  field.copy_to_host(gaussian_host);
  for (const pops::Index<Dim>& index : std::array{box.lo, box.hi})
    EXPECT_DOUBLE_EQ(gaussian_host(host_offset(field.grown_box(), index)), Real(1.25));
}

TEST(AnalyticExpression, DiocotronRingIsARegularTypedExpression) {
  const auto program = compile_analytic_expression(diocotron_density());
  EXPECT_EQ(program.result_type(), AnalyticValueType::Scalar);
  EXPECT_EQ(program.required_dimension(), 2);
  EXPECT_GT(program.instruction_count(), 0u);
  EXPECT_LE(program.required_stack(), pops::analytic::kAnalyticMaxStack);

  EXPECT_NEAR(program.evaluate(pops::RealVector<2>{Real(0.375), Real(0)}), Real(0.9), 1e-14);
  const Real angle = Real(3.14159265358979323846) / Real(8);
  EXPECT_NEAR(program.evaluate(pops::RealVector<2>{Real(0.375) * std::cos(angle),
                                                   Real(0.375) * std::sin(angle)}),
              Real(1), 1e-14);
  EXPECT_DOUBLE_EQ(program.evaluate(pops::RealVector<2>{Real(0.2), Real(0)}), Real(1e-4));
  EXPECT_DOUBLE_EQ(program.evaluate(pops::RealVector<2>{Real(0.45), Real(0)}), Real(1e-4));
}

TEST(AnalyticExpression, ScalarBooleanAndSelectionOpcodesHaveStrictSemantics) {
  const AnalyticNode scalar = binary(
      AnalyticOp::Add,
      binary(AnalyticOp::Min,
             binary(AnalyticOp::Pow,
                    unary(AnalyticOp::Sqrt,
                          unary(AnalyticOp::Abs, unary(AnalyticOp::Neg, AnalyticNode::x()))),
                    AnalyticNode::constant(Real(2))),
             binary(AnalyticOp::Max, AnalyticNode::constant(Real(2)),
                    AnalyticNode::constant(Real(3)))),
      binary(AnalyticOp::Div,
             unary(AnalyticOp::Log,
                   unary(AnalyticOp::Exp, unary(AnalyticOp::Cos, AnalyticNode::constant(Real(0))))),
             AnalyticNode::constant(Real(2))));
  const auto scalar_program = compile_analytic_expression(scalar);
  EXPECT_NEAR(scalar_program.evaluate(pops::RealVector<1>{Real(-4)}), Real(3.5), 1e-13);

  const AnalyticNode predicate =
      binary(AnalyticOp::Or,
             unary(AnalyticOp::Not, binary(AnalyticOp::Eq, AnalyticNode::x(), AnalyticNode::y())),
             binary(AnalyticOp::And,
                    binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::constant(Real(0))),
                    binary(AnalyticOp::Ne, AnalyticNode::y(), AnalyticNode::constant(Real(0)))));
  const auto predicate_program = compile_analytic_expression(predicate);
  EXPECT_EQ(predicate_program.result_type(), AnalyticValueType::Predicate);
  const auto predicate_true =
      predicate_program.view().eval_checked(pops::RealVector<2>{Real(1), Real(2)});
  const auto predicate_false =
      predicate_program.view().eval_checked(pops::RealVector<2>{Real(1), Real(1)});
  ASSERT_TRUE(predicate_true.valid);
  ASSERT_TRUE(predicate_false.valid);
  EXPECT_NE(predicate_true.value, Real(0));
  EXPECT_EQ(predicate_false.value, Real(0));

  const auto greater =
      compile_analytic_expression(binary(AnalyticOp::Gt, AnalyticNode::x(), AnalyticNode::y()));
  const auto less_equal =
      compile_analytic_expression(binary(AnalyticOp::Le, AnalyticNode::x(), AnalyticNode::y()));
  const auto greater_result = greater.view().eval_checked(pops::RealVector<2>{Real(2), Real(1)});
  const auto less_equal_result =
      less_equal.view().eval_checked(pops::RealVector<2>{Real(1), Real(1)});
  ASSERT_TRUE(greater_result.valid);
  ASSERT_TRUE(less_equal_result.valid);
  EXPECT_NE(greater_result.value, Real(0));
  EXPECT_NE(less_equal_result.value, Real(0));

  const auto subtraction =
      compile_analytic_expression(binary(AnalyticOp::Sub, AnalyticNode::x(), AnalyticNode::y()));
  EXPECT_DOUBLE_EQ(subtraction.evaluate(pops::RealVector<2>{Real(5), Real(2)}), Real(3));
}

TEST(AnalyticExpression, InvalidIntermediatesCannotMasqueradeAsFiniteResults) {
  const AnalyticNode invalid_log = unary(AnalyticOp::Log, AnalyticNode::constant(Real(-1)));
  const AnalyticNode invalid_predicate =
      binary(AnalyticOp::Lt, invalid_log, AnalyticNode::constant(Real(0)));
  const auto masked_predicate = compile_analytic_expression(
      select(invalid_predicate, AnalyticNode::constant(Real(1)), AnalyticNode::constant(Real(2))));
  const auto masked_predicate_result =
      masked_predicate.view().eval_checked(pops::RealVector<1>{Real(0)});
  EXPECT_FALSE(masked_predicate_result.valid);
  EXPECT_TRUE(std::isnan(masked_predicate.evaluate(pops::RealVector<1>{Real(0)})));

  const auto masked_minimum = compile_analytic_expression(
      binary(AnalyticOp::Min, invalid_log, AnalyticNode::constant(Real(3))));
  EXPECT_FALSE(masked_minimum.view().eval_checked(pops::RealVector<1>{Real(0)}).valid);
  EXPECT_TRUE(std::isnan(masked_minimum.evaluate(pops::RealVector<1>{Real(0)})));

  const AnalyticNode guarded_log =
      select(binary(AnalyticOp::Gt, AnalyticNode::x(), AnalyticNode::constant(Real(0))),
             unary(AnalyticOp::Log, AnalyticNode::x()), AnalyticNode::constant(Real(0)));
  const auto guarded = compile_analytic_expression(guarded_log);
  const auto outside = guarded.view().eval_checked(pops::RealVector<1>{Real(-1)});
  ASSERT_TRUE(outside.valid);
  EXPECT_DOUBLE_EQ(outside.value, Real(0));
  const auto inside =
      guarded.view().eval_checked(pops::RealVector<1>{Real(2.71828182845904523536)});
  ASSERT_TRUE(inside.valid);
  EXPECT_NEAR(inside.value, Real(1), 1e-14);
}

TEST(AnalyticExpression, FlatPostfixSeamIsValidatedAndEquivalent) {
  EXPECT_EQ(analytic_op_from_name("minimum"), AnalyticOp::Min);
  EXPECT_EQ(analytic_op_from_name("where"), AnalyticOp::Select);
  EXPECT_THROW(analytic_op_from_name("min"), std::invalid_argument);
  EXPECT_THROW(analytic_op_from_name("max"), std::invalid_argument);
  EXPECT_THROW(analytic_op_from_name("select"), std::invalid_argument);

  const std::vector<AnalyticToken> tokens = {
      {AnalyticOp::X, Real(0)},        {AnalyticOp::Y, Real(0)},   {AnalyticOp::Hypot, Real(0)},
      {AnalyticOp::Constant, Real(2)}, {AnalyticOp::Mul, Real(0)},
  };
  const auto program = compile_analytic_postfix(tokens);
  EXPECT_EQ(program.required_dimension(), 2);
  EXPECT_DOUBLE_EQ(program.evaluate(pops::RealVector<2>{Real(3), Real(4)}), Real(10));

  EXPECT_THROW(compile_analytic_postfix({}), std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix({{AnalyticOp::Add, Real(0)}}), std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_postfix(
          {{AnalyticOp::X, Real(0)}, {AnalyticOp::Y, Real(0)}, {AnalyticOp::Pow, Real(0)}}),
      std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix({{AnalyticOp::X, Real(0)},
                                         {AnalyticOp::Constant, Real(2)},
                                         {AnalyticOp::Constant, Real(1)},
                                         {AnalyticOp::Between, Real(0)}}),
               std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_postfix({{AnalyticOp::Constant, Real(1)}, {AnalyticOp::Constant, Real(2)}}),
      std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix({{AnalyticOp::X, Real(1)}}), std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_postfix({{AnalyticOp::Constant, std::numeric_limits<Real>::infinity()}}),
      std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix({{static_cast<AnalyticOp>(255), Real(0)}}),
               std::invalid_argument);
}

TEST(AnalyticExpression, TreeArityTypesAndResourceLimitsFailOnHost) {
  EXPECT_THROW(
      compile_analytic_expression(AnalyticNode{AnalyticOp::Add, Real(0), {AnalyticNode::x()}}),
      std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_expression(binary(AnalyticOp::And, AnalyticNode::x(), AnalyticNode::y())),
      std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(select(
                   binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::y()), AnalyticNode::x(),
                   binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::constant(Real(0))))),
               std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_expression(AnalyticNode::constant(std::numeric_limits<Real>::quiet_NaN())),
      std::invalid_argument);

  const AnalyticNode three_nodes = binary(AnalyticOp::Add, AnalyticNode::x(), AnalyticNode::y());
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{2, 64, 64}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{4096, 1, 64}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{4096, 64, 1}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{4097, 64, 64}),
               std::invalid_argument);
}

TEST(AnalyticExpression, ProgramViewRunsInsideRankedKokkosKernelsInOneTwoAndThreeDimensions) {
  check_program_view_kernel<1>();
  check_program_view_kernel<2>();
  check_program_view_kernel<3>();
}

TEST(AnalyticExpression, InitialMaterializersAreRankGenericAndCompleteBeforeProgramsExpire) {
  check_initial_materializers<1>();
  check_initial_materializers<2>();
  check_initial_materializers<3>();
}

TEST(AnalyticExpression, ZOnATwoDimensionalTargetFailsClosedBeforePublication) {
  const auto program = compile_analytic_expression(AnalyticNode::z());
  EXPECT_EQ(program.required_dimension(), 3);
  EXPECT_THROW(program.evaluate(pops::RealVector<2>{Real(1), Real(2)}), std::invalid_argument);
  const auto rejected = program.view().eval_checked(pops::RealVector<2>{Real(1), Real(2)});
  EXPECT_FALSE(rejected.valid);
  EXPECT_TRUE(std::isnan(rejected.value));

  const pops::Box<2> box = pops::Box<2>::from_extents(pops::Extent<2>{4, 3});
  const pops::Geometry<2> geometry = unit_geometry(box);
  auto values = one_patch_field(box, 1);
  values.set_val(Real(7));
  std::vector<pops::analytic::AnalyticProgram> programs;
  programs.push_back(compile_analytic_expression(AnalyticNode::z()));
  EXPECT_THROW(pops::analytic::materialize_cell_average(values, geometry, programs),
               std::invalid_argument);

  const auto& field = values.fab(0);
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  for_each_host_index(box, [&](const pops::Index<2>& index) {
    EXPECT_EQ(host(host_offset(field.grown_box(), index)), Real(7));
  });
}

}  // namespace
