#include <gtest/gtest.h>
#include <array>
#include <pops/runtime/program/prepared_spatial_residual.hpp>
#include <pops/runtime/program/prepared_amr_spatial_residual.hpp>
#include <pops/mesh/execution/for_each.hpp>

using namespace pops;

namespace {

MultiFab<2> scalar_field(int components = 1, int width = 16) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{width - 1, width - 1}};
  const auto layout = mesh::BoxArray<2>::from_domain(domain, Extent<2>{width, width});
  const auto distribution =
      mesh::Distribution<2>::replicated(layout, mesh::RankSpace<2>{Index<2>{}, Extent<2>{1, 1}});
  return MultiFab<2>(layout, distribution, Index<2>{}, components, Extent<2>{});
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
      for_each_cell(q.box(local), AccumulationResidual{q.fab(local).view(),
                                                       result.fab(local).view(), power, quadratic});
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
  const auto report =
      solve.solve(&seed, residual(Real(2.5), false), ExecutionLane::world("test.spatial-constant"));
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
    const auto report = solve.solve(
        &seed,
        [&](const MultiFab<2>& q, MultiFab<2>& output, int evaluation) {
          lincomb(last_evaluation, Real(1), q, Real(0), q);
          equation(q, output, evaluation);
        },
        ExecutionLane::world("test.spatial-final-evaluation"));
    ASSERT_TRUE(report.solved_value_available()) << report.reason;
    saxpy(last_evaluation, Real(-1), solve.candidate());
    EXPECT_EQ(reduce_min(last_evaluation), Real(0));
    EXPECT_EQ(reduce_max(last_evaluation), Real(0));
    if (initial != root)
      EXPECT_GT(solve.derivative_evaluations(), 0);
  }
}

TEST(PreparedSpatialResidual, InvalidResidualCannotBecomeReadableOrMutateSeed) {
  auto seed = scalar_field();
  seed.set_val(Real(0.2));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto report = solve.solve(
      &seed,
      [](const MultiFab<2>&, MultiFab<2>& output, int) {
        output.set_val(std::numeric_limits<Real>::quiet_NaN());
      },
      ExecutionLane::world("test.spatial-invalid"));
  EXPECT_FALSE(report.solved_value_available());
  EXPECT_EQ(report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_NEAR(reduce_min(seed), 0.2, 1e-15);
  EXPECT_EQ(solve.derivative_evaluations(), 0);
}

TEST(PreparedSpatialResidual, NewtonCorrectionCouplesPeriodicNeighborDegreesOfFreedom) {
  auto seed = scalar_field();
  seed.set_val(Real(0));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto report = solve.solve(
      &seed,
      [](const MultiFab<2>& q, MultiFab<2>& result, int) {
        for (std::size_t local = 0; local < q.local_size(); ++local)
          for_each_cell(q.box(local),
                        ShiftedPeriodicResidual{q.fab(local).view(), result.fab(local).view()});
      },
      ExecutionLane::world("test.spatial-periodic"));
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_LT(report.residual_norm, Real(1e-10));
  const Real sine = std::sin(std::acos(Real(-1)) / Real(16));
  const Real amplitude = Real(1) / (Real(1) + Real(0.01 * 8 * 0.1 * 16 * 16) * sine * sine);
  const Real mode_max = std::pow(std::cos(std::acos(Real(-1)) / Real(16)), 2);
  EXPECT_NEAR(reduce_max(solve.candidate()), Real(1) + amplitude * mode_max, 1e-10);
  EXPECT_NEAR(reduce_min(solve.candidate()), Real(1) - amplitude * mode_max, 1e-10);
}

namespace {
void set_pair(MultiFab<2>& field, Real first, Real second) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto values = field.fab(local).view();
    for_each_cell(field.box(local), [=] POPS_HD(const Index<2>& cell) {
      values(cell, 0) = first;
      values(cell, 1) = second;
    });
  }
}

// Pure algebraic coupled vector residual: the first component can already be solved while
// the second is not, so a scalar-only norm would falsely report success.
void evaluate_pair(const MultiFab<2>& coordinate, const MultiFab<2>& previous,
                   MultiFab<2>& result) {
  for (std::size_t local = 0; local < coordinate.local_size(); ++local) {
    const auto q = coordinate.fab(local).view();
    const auto old = previous.fab(local).view();
    const auto out = result.fab(local).view();
    for_each_cell(coordinate.box(local), [=] POPS_HD(const Index<2>& cell) {
      out(cell, 0) = q(cell, 0) + q(cell, 0) * q(cell, 0) - old(cell, 0);
      out(cell, 1) = q(cell, 1) + q(cell, 0) * q(cell, 1) - old(cell, 1);
    });
  }
}
}  // namespace

TEST(PreparedSpatialResidual, CoupledVectorAccumulationMeasuresEveryComponent) {
  auto seed = scalar_field(2);
  auto previous = scalar_field(2);
  set_pair(seed, Real(1), Real(0));
  set_pair(previous, Real(2), Real(4));
  runtime::program::PreparedSpatialResidual<2> solve(seed, options(), Real(6e-6));
  const auto report = solve.solve(
      &seed, [&](const auto& q, auto& result, int) { evaluate_pair(q, previous, result); },
      ExecutionLane::world("test.vector-spatial-accumulation"));
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_GT(solve.derivative_evaluations(), 0);
  ASSERT_EQ(solve.candidate().ncomp(), 2);
  EXPECT_NEAR(reduce_min(solve.candidate(), 0), Real(1), 1e-10);
  EXPECT_NEAR(reduce_max(solve.candidate(), 0), Real(1), 1e-10);
  EXPECT_NEAR(reduce_min(solve.candidate(), 1), Real(2), 1e-10);
  EXPECT_NEAR(reduce_max(solve.candidate(), 1), Real(2), 1e-10);
  EXPECT_EQ(reduce_max(seed, 1), Real(0));
  auto wrong = scalar_field();
  EXPECT_THROW(
      solve.solve(
          &wrong, [&](const auto& q, auto& result, int) { evaluate_pair(q, previous, result); },
          ExecutionLane::world("test.vector-wrong-width")),
      std::invalid_argument);
}

TEST(PreparedSpatialResidual, CompositeVectorNormAndCoveredProjectionRetainEveryComponent) {
  auto coarse = scalar_field(2, 4), fine = scalar_field(2, 8);
  auto coarse_mask = scalar_field(1, 4), fine_mask = scalar_field(1, 8);
  coarse_mask.set_val(Real(0));  // Entire parent is covered by the refined representation.
  fine_mask.set_val(Real(1));
  set_pair(coarse, Real(7), Real(9));
  set_pair(fine, Real(1), Real(0));
  const std::array<const MultiFab<2>*, 2> layouts{&coarse, &fine};
  const std::array<const MultiFab<2>*, 2> masks{&coarse_mask, &fine_mask};
  const std::array<Real, 2> measures{Real(1) / 16, Real(1) / 64};
  runtime::program::PreparedAmrSpatialResidual<2> solve(layouts, masks, measures, options(),
                                                        Real(6e-6));
  auto coarse_previous = scalar_field(2, 4), fine_previous = scalar_field(2, 8);
  set_pair(coarse_previous, Real(2), Real(4));
  set_pair(fine_previous, Real(2), Real(4));
  solve.stage(0, coarse_previous, &coarse);
  solve.stage(1, fine_previous, &fine);
  const auto report = solve.solve(
      [&](const auto& q, const auto& previous, auto& result, int) {
        for (std::size_t level = 0; level < q.size(); ++level)
          evaluate_pair(q[level], previous[level], result[level]);
      },
      ExecutionLane::world("test.vector-amr-accumulation"));
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_GT(solve.derivative_evaluations(), 0);
  EXPECT_NEAR(reduce_min(solve.candidate(1), 0), Real(1), 1e-10);
  EXPECT_NEAR(reduce_max(solve.candidate(1), 0), Real(1), 1e-10);
  EXPECT_NEAR(reduce_min(solve.candidate(1), 1), Real(2), 1e-10);
  EXPECT_NEAR(reduce_max(solve.candidate(1), 1), Real(2), 1e-10);
  EXPECT_EQ(reduce_min(solve.candidate(0), 0), Real(7));
  EXPECT_EQ(reduce_max(solve.candidate(0), 0), Real(7));
  EXPECT_EQ(reduce_min(solve.candidate(0), 1), Real(9));
  EXPECT_EQ(reduce_max(solve.candidate(0), 1), Real(9));
  auto wrong = scalar_field(1, 8);
  set_pair(fine_previous, Real(22), Real(44));
  EXPECT_THROW(solve.stage(1, fine_previous, &wrong), std::invalid_argument);
  EXPECT_EQ(reduce_max(solve.previous(1), 0), Real(2));
  EXPECT_EQ(reduce_max(solve.previous(1), 1), Real(4));
  const std::array<const MultiFab<2>*, 2> mixed_width{&coarse, &wrong};
  EXPECT_THROW((runtime::program::PreparedAmrSpatialResidual<2>(mixed_width, masks, measures,
                                                                options(), Real(6e-6))),
               std::invalid_argument);
}
