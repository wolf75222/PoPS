#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/analytic/expression.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace {

using pops::Array4;
using pops::Box2D;
using pops::Fab2D;
using pops::Geometry;
using pops::Real;
using pops::analytic::AnalyticLimits;
using pops::analytic::AnalyticNode;
using pops::analytic::AnalyticOp;
using pops::analytic::AnalyticProgramView;
using pops::analytic::AnalyticToken;
using pops::analytic::AnalyticValueType;
using pops::analytic::compile_analytic_expression;
using pops::analytic::compile_analytic_postfix;
using pops::analytic::analytic_op_from_name;

AnalyticNode unary(AnalyticOp op, AnalyticNode value) {
  return AnalyticNode::apply(op, {std::move(value)});
}

AnalyticNode binary(AnalyticOp op, AnalyticNode left, AnalyticNode right) {
  return AnalyticNode::apply(op, {std::move(left), std::move(right)});
}

AnalyticNode select(AnalyticNode condition, AnalyticNode selected, AnalyticNode otherwise) {
  return AnalyticNode::apply(
      AnalyticOp::Select,
      {std::move(condition), std::move(selected), std::move(otherwise)});
}

AnalyticNode diocotron_density() {
  auto radius = [] {
    return binary(AnalyticOp::Hypot, AnalyticNode::x(), AnalyticNode::y());
  };
  const AnalyticNode annulus = AnalyticNode::apply(
      AnalyticOp::Between,
      {radius(), AnalyticNode::constant(Real(0.35)), AnalyticNode::constant(Real(0.40))});
  const AnalyticNode theta =
      binary(AnalyticOp::Atan2, AnalyticNode::y(), AnalyticNode::x());
  const AnalyticNode modulation = binary(
      AnalyticOp::Add, AnalyticNode::constant(Real(0.9)),
      binary(AnalyticOp::Mul, AnalyticNode::constant(Real(0.1)),
             unary(AnalyticOp::Sin,
                   binary(AnalyticOp::Mul, AnalyticNode::constant(Real(4)), theta))));
  return select(annulus, modulation, AnalyticNode::constant(Real(1e-4)));
}

struct EvaluateAnalyticKernel {
  AnalyticProgramView expression;
  Geometry geometry;
  Array4 output;

  POPS_HD void operator()(int i, int j) const {
    output(i, j) = expression.eval(geometry.x_cell(i), geometry.y_cell(j));
  }
};

TEST(AnalyticExpression, DiocotronRingIsARegularTypedExpression) {
  const auto program = compile_analytic_expression(diocotron_density());
  EXPECT_EQ(program.result_type(), AnalyticValueType::Scalar);
  EXPECT_GT(program.instruction_count(), 0u);
  EXPECT_LE(program.required_stack(), pops::analytic::kAnalyticMaxStack);

  EXPECT_NEAR(program.evaluate(Real(0.375), Real(0)), Real(0.9), 1e-14);
  const Real angle = Real(3.14159265358979323846) / Real(8);
  EXPECT_NEAR(program.evaluate(Real(0.375) * std::cos(angle),
                               Real(0.375) * std::sin(angle)),
              Real(1), 1e-14);
  EXPECT_DOUBLE_EQ(program.evaluate(Real(0.2), Real(0)), Real(1e-4));
  EXPECT_DOUBLE_EQ(program.evaluate(Real(0.45), Real(0)), Real(1e-4));
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
                   unary(AnalyticOp::Exp,
                         unary(AnalyticOp::Cos, AnalyticNode::constant(Real(0))))),
             AnalyticNode::constant(Real(2))));
  const auto scalar_program = compile_analytic_expression(scalar);
  // x=-4: min((sqrt(abs(-x)))^2, max(2,3)) + log(exp(cos(0)))/2 = 3.5.
  EXPECT_NEAR(scalar_program.evaluate(Real(-4), Real(7)), Real(3.5), 1e-13);

  const AnalyticNode predicate = binary(
      AnalyticOp::Or,
      unary(AnalyticOp::Not,
            binary(AnalyticOp::Eq, AnalyticNode::x(), AnalyticNode::y())),
      binary(AnalyticOp::And,
             binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::constant(Real(0))),
             binary(AnalyticOp::Ne, AnalyticNode::y(), AnalyticNode::constant(Real(0)))));
  const auto predicate_program = compile_analytic_expression(predicate);
  EXPECT_EQ(predicate_program.result_type(), AnalyticValueType::Predicate);
  const auto predicate_true = predicate_program.view().eval_checked(Real(1), Real(2));
  const auto predicate_false = predicate_program.view().eval_checked(Real(1), Real(1));
  ASSERT_TRUE(predicate_true.valid);
  ASSERT_TRUE(predicate_false.valid);
  EXPECT_NE(predicate_true.value, Real(0));
  EXPECT_EQ(predicate_false.value, Real(0));

  const auto greater = compile_analytic_expression(
      binary(AnalyticOp::Gt, AnalyticNode::x(), AnalyticNode::y()));
  const auto less_equal = compile_analytic_expression(
      binary(AnalyticOp::Le, AnalyticNode::x(), AnalyticNode::y()));
  const auto greater_result = greater.view().eval_checked(Real(2), Real(1));
  const auto less_equal_result = less_equal.view().eval_checked(Real(1), Real(1));
  ASSERT_TRUE(greater_result.valid);
  ASSERT_TRUE(less_equal_result.valid);
  EXPECT_NE(greater_result.value, Real(0));
  EXPECT_NE(less_equal_result.value, Real(0));

  const auto subtraction = compile_analytic_expression(
      binary(AnalyticOp::Sub, AnalyticNode::x(), AnalyticNode::y()));
  EXPECT_DOUBLE_EQ(subtraction.evaluate(Real(5), Real(2)), Real(3));
}

TEST(AnalyticExpression, InvalidIntermediatesCannotMasqueradeAsFiniteResults) {
  const AnalyticNode invalid_log =
      unary(AnalyticOp::Log, AnalyticNode::constant(Real(-1)));
  const AnalyticNode invalid_predicate = binary(
      AnalyticOp::Lt, invalid_log, AnalyticNode::constant(Real(0)));
  const auto masked_predicate = compile_analytic_expression(select(
      invalid_predicate, AnalyticNode::constant(Real(1)),
      AnalyticNode::constant(Real(2))));
  EXPECT_FALSE(masked_predicate.view().eval_checked(Real(0), Real(0)).valid);
  EXPECT_TRUE(std::isnan(masked_predicate.evaluate(Real(0), Real(0))));

  const auto masked_minimum = compile_analytic_expression(binary(
      AnalyticOp::Min, invalid_log, AnalyticNode::constant(Real(3))));
  EXPECT_FALSE(masked_minimum.view().eval_checked(Real(0), Real(0)).valid);
  EXPECT_TRUE(std::isnan(masked_minimum.evaluate(Real(0), Real(0))));

  const AnalyticNode guarded_log = select(
      binary(AnalyticOp::Gt, AnalyticNode::x(), AnalyticNode::constant(Real(0))),
      unary(AnalyticOp::Log, AnalyticNode::x()), AnalyticNode::constant(Real(0)));
  const auto guarded = compile_analytic_expression(guarded_log);
  const auto outside = guarded.view().eval_checked(Real(-1), Real(0));
  ASSERT_TRUE(outside.valid);
  EXPECT_DOUBLE_EQ(outside.value, Real(0));
  const auto inside = guarded.view().eval_checked(
      Real(2.71828182845904523536), Real(0));
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
      {AnalyticOp::X, Real(0)},        {AnalyticOp::Y, Real(0)},
      {AnalyticOp::Hypot, Real(0)},    {AnalyticOp::Constant, Real(2)},
      {AnalyticOp::Mul, Real(0)},
  };
  const auto program = compile_analytic_postfix(tokens);
  EXPECT_DOUBLE_EQ(program.evaluate(Real(3), Real(4)), Real(10));

  EXPECT_THROW(compile_analytic_postfix({}), std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix({{AnalyticOp::Add, Real(0)}}), std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_postfix({
          {AnalyticOp::X, Real(0)}, {AnalyticOp::Y, Real(0)}, {AnalyticOp::Pow, Real(0)}}),
      std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_postfix({
          {AnalyticOp::X, Real(0)}, {AnalyticOp::Constant, Real(2)},
          {AnalyticOp::Constant, Real(1)}, {AnalyticOp::Between, Real(0)}}),
      std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix(
                   {{AnalyticOp::Constant, Real(1)}, {AnalyticOp::Constant, Real(2)}}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix({{AnalyticOp::X, Real(1)}}), std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix(
                   {{AnalyticOp::Constant, std::numeric_limits<Real>::infinity()}}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_postfix(
                   {{static_cast<AnalyticOp>(255), Real(0)}}),
               std::invalid_argument);
}

TEST(AnalyticExpression, TreeArityTypesAndResourceLimitsFailOnHost) {
  EXPECT_THROW(compile_analytic_expression(
                   AnalyticNode{AnalyticOp::Add, Real(0), {AnalyticNode::x()}}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(binary(AnalyticOp::And, AnalyticNode::x(),
                                                  AnalyticNode::y())),
               std::invalid_argument);
  EXPECT_THROW(
      compile_analytic_expression(select(
          binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::y()), AnalyticNode::x(),
          binary(AnalyticOp::Lt, AnalyticNode::x(), AnalyticNode::constant(Real(0))))),
      std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(
                   AnalyticNode::constant(std::numeric_limits<Real>::quiet_NaN())),
               std::invalid_argument);

  const AnalyticNode three_nodes =
      binary(AnalyticOp::Add, AnalyticNode::x(), AnalyticNode::y());
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{2, 64, 64}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{4096, 1, 64}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{4096, 64, 1}),
               std::invalid_argument);
  EXPECT_THROW(compile_analytic_expression(three_nodes, AnalyticLimits{4097, 64, 64}),
               std::invalid_argument);
}

TEST(AnalyticExpression, ProgramViewRunsInsideTheKokkosCellKernel) {
  const auto program = compile_analytic_expression(diocotron_density());
  const Box2D box = Box2D::from_extents(65, 65);  // exceeds the host tiny-box fallback threshold
  const Geometry geometry{box, Real(-0.5), Real(0.5), Real(-0.5), Real(0.5)};
  Fab2D output(box, 1, 0);

  pops::for_each_cell(box, EvaluateAnalyticKernel{program.view(), geometry, output.array()});
  pops::sync_host();

  for (const auto [i, j] : {std::pair{0, 0}, std::pair{20, 31}, std::pair{32, 32},
                            std::pair{56, 42}, std::pair{64, 64}}) {
    const Real x = geometry.x_cell(i);
    const Real y = geometry.y_cell(j);
    const Real radius = std::hypot(x, y);
    const Real expected =
        radius >= Real(0.35) && radius <= Real(0.40)
            ? Real(0.9) + Real(0.1) * std::sin(Real(4) * std::atan2(y, x))
            : Real(1e-4);
    EXPECT_NEAR(output(i, j), expected, 1e-14) << "cell (" << i << "," << j << ")";
  }
}

TEST(AnalyticExpression, InitialMaterializersCompleteBeforeBorrowedInputsExpire) {
  const Box2D box = Box2D::from_extents(65, 65);
  const pops::BoxArray boxes(std::vector<Box2D>{box});
  const pops::DistributionMapping distribution(1, 1);
  pops::MultiFab values(boxes, distribution, 1, 0);

  {
    std::vector<pops::analytic::AnalyticProgram> programs;
    programs.push_back(
        compile_analytic_expression(AnalyticNode::constant(Real(3.25))));
    EXPECT_EQ(pops::analytic::materialize_cell_average(
                  values, Real(0), Real(0), Real(1) / Real(65), Real(1) / Real(65), programs),
              box.num_cells());
  }  // Every AnalyticProgramView captured by a device kernel is invalid beyond this point.

  // Deliberately no sync_host()/fence in the test: completion is the materializer's API contract.
  EXPECT_DOUBLE_EQ(values.fab(0)(0, 0, 0), Real(3.25));
  EXPECT_DOUBLE_EQ(values.fab(0)(32, 41, 0), Real(3.25));
  EXPECT_DOUBLE_EQ(values.fab(0)(64, 64, 0), Real(3.25));

  EXPECT_EQ(pops::analytic::materialize_gaussian_cell_average(
                values, Real(0), Real(0), Real(1) / Real(65), Real(1) / Real(65), Real(0.5),
                Real(0.5), Real(1.25), Real(0), Real(20)),
            box.num_cells());
  EXPECT_DOUBLE_EQ(values.fab(0)(0, 0, 0), Real(1.25));
  EXPECT_DOUBLE_EQ(values.fab(0)(64, 64, 0), Real(1.25));
}

}  // namespace
