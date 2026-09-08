#include <gtest/gtest.h>
#include <pops/runtime/program/prepared_spatial_residual.hpp>
#include <pops/mesh/execution/for_each.hpp>

using namespace pops;

namespace {

MultiFab<2> scalar_field() {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{15, 15}};
  const auto layout = mesh::BoxArray<2>::from_domain(domain, Extent<2>{16, 16});
  const auto distribution = mesh::Distribution<2>::replicated(
      layout, mesh::RankSpace<2>{Index<2>{}, Extent<2>{1, 1}});
  return MultiFab<2>(layout, distribution, Index<2>{}, 1, Extent<2>{});
}

struct AccumulationResidual {
  FieldView<const Real, 2> q;
  FieldView<Real, 2> residual;
  Real power;
  bool quadratic;
  POPS_HD void operator()(const Index<2>& cell) const {
    const Real temperature = q(cell, 0);
    residual(cell, 0) = temperature + (quadratic ? temperature * temperature : Real(0)) - power;
  }
};

auto residual(Real power, bool quadratic) {
  return [=](const MultiFab<2>& q, MultiFab<2>& result, int) {
    for (std::size_t local = 0; local < q.local_size(); ++local)
      for_each_cell(q.box(local), AccumulationResidual{
          q.fab(local).view(), result.fab(local).view(), power, quadratic});
  };
}

FieldNewtonOptions options() {
  FieldNewtonOptions result;
  result.tolerance = Real(1e-12);
  result.linear_tolerance = Real(1e-10);
  result.max_iterations = 15;
  result.linear_max_iterations = 30;
  result.restart = 10;
  return result;
}

struct ShiftedPeriodicResidual {
  FieldView<const Real, 2> q;
  FieldView<Real, 2> result;
  POPS_HD void operator()(const Index<2>& cell) const {
    constexpr Real pi = Real(3.14159265358979323846);
    const Real forcing = Real(1) + std::sin(Real(2) * pi * (cell[0] + Real(0.5)) / Real(16)) *
                                      std::sin(Real(2) * pi * (cell[1] + Real(0.5)) / Real(16));
    Real laplacian = -Real(4) * q(cell, 0);
    for (int axis = 0; axis < 2; ++axis) {
      auto lower = cell, upper = cell;
      lower[axis] = (cell[axis] + 15) % 16;
      upper[axis] = (cell[axis] + 1) % 16;
      laplacian += q(lower, 0) + q(upper, 0);
    }
    result(cell, 0) = q(cell, 0) - Real(0.01 * 0.1 * 16 * 16) * laplacian - forcing;
  }
};

}  // namespace

TEST(PreparedSpatialResidual, NonlinearAccumulationFreezesEnergySeparatelyFromSeed) {
  auto seed = scalar_field();
  seed.set_val(Real(0.2));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto lane = ExecutionLane::world("test.spatial-accumulation");
  const auto report = solve.solve(&seed, residual(Real(1), true), lane);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  const Real expected = (std::sqrt(Real(5)) - Real(1)) / Real(2);
  EXPECT_NEAR(reduce_min(solve.candidate()), expected, 1e-11);
  EXPECT_NEAR(reduce_max(solve.candidate()), expected, 1e-11);
  EXPECT_NEAR(reduce_min(seed), 0.2, 1e-15);
  EXPECT_NEAR(expected + expected * expected, 1.0, 1e-15);
  EXPECT_NE(expected, Real(0.5));
  EXPECT_GT(solve.derivative_evaluations(), 0);
  EXPECT_GT(solve.residual_evaluations(), 2 * solve.derivative_evaluations());
  EXPECT_EQ(report.evaluations, solve.residual_evaluations());
}

TEST(PreparedSpatialResidual, IdentityDeterminesNonzeroConstantWithoutGauge) {
  auto seed = scalar_field();
  seed.set_val(Real(0));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto report = solve.solve(&seed, residual(Real(2.5), false),
                                   ExecutionLane::world("test.spatial-constant"));
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_NEAR(reduce_min(solve.candidate()), 2.5, 1e-11);
  EXPECT_NEAR(reduce_max(solve.candidate()), 2.5, 1e-11);
}

TEST(PreparedSpatialResidual, LastResidualEvaluationBelongsToSolvedCandidate) {
  const Real root = (std::sqrt(Real(5)) - Real(1)) / Real(2);
  for (const Real initial : {root, Real(0.2)}) {
    auto seed = scalar_field();
    auto last_evaluation = scalar_field();
    seed.set_val(initial);
    runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
    auto equation = residual(Real(1), true);
    const auto report = solve.solve(&seed,
        [&](const MultiFab<2>& q, MultiFab<2>& output, int evaluation) {
          lincomb(last_evaluation, Real(1), q, Real(0), q);
          equation(q, output, evaluation);
        }, ExecutionLane::world("test.spatial-final-evaluation"));
    ASSERT_TRUE(report.solved_value_available()) << report.reason;
    saxpy(last_evaluation, Real(-1), solve.candidate());
    EXPECT_EQ(reduce_min(last_evaluation), Real(0));
    EXPECT_EQ(reduce_max(last_evaluation), Real(0));
    if (initial != root) EXPECT_GT(solve.derivative_evaluations(), 0);
  }
}

TEST(PreparedSpatialResidual, InvalidResidualCannotBecomeReadableOrMutateSeed) {
  auto seed = scalar_field();
  seed.set_val(Real(0.2));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto report = solve.solve(&seed,
      [](const MultiFab<2>&, MultiFab<2>& output, int) {
        output.set_val(std::numeric_limits<Real>::quiet_NaN());
      }, ExecutionLane::world("test.spatial-invalid"));
  EXPECT_FALSE(report.solved_value_available());
  EXPECT_EQ(report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_NEAR(reduce_min(seed), 0.2, 1e-15);
  EXPECT_EQ(solve.derivative_evaluations(), 0);
}

TEST(PreparedSpatialResidual, NewtonCorrectionCouplesPeriodicNeighborDegreesOfFreedom) {
  auto seed = scalar_field();
  seed.set_val(Real(0));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto report = solve.solve(&seed,
      [](const MultiFab<2>& q, MultiFab<2>& result, int) {
        for (std::size_t local = 0; local < q.local_size(); ++local)
          for_each_cell(q.box(local), ShiftedPeriodicResidual{
              q.fab(local).view(), result.fab(local).view()});
      }, ExecutionLane::world("test.spatial-periodic"));
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_LT(report.residual_norm, Real(1e-10));
  const Real sine = std::sin(std::acos(Real(-1)) / Real(16));
  const Real amplitude = Real(1) / (Real(1) + Real(0.01 * 8 * 0.1 * 16 * 16) * sine * sine);
  const Real mode_max = std::pow(std::cos(std::acos(Real(-1)) / Real(16)), 2);
  EXPECT_NEAR(reduce_max(solve.candidate()), Real(1) + amplitude * mode_max, 1e-10);
  EXPECT_NEAR(reduce_min(solve.candidate()), Real(1) - amplitude * mode_max, 1e-10);
}
