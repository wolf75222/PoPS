#include <gtest/gtest.h>

#include <pops/runtime/program/residual_operator.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

using namespace pops::runtime::program;

namespace {

ResidualDomain coupled_domain() { return {{{2}}, {{"fluid", 0, 1}, {"field", 1, 1}}}; }

void coupled_residual(const std::vector<double>& x, std::vector<double>& r) {
  r[0] = x[0] * x[0] + 3.0 * x[1];
  r[1] = std::sin(x[0]) - 2.0 * x[1];
}

void coupled_exact_jvp(const std::vector<double>& x, const std::vector<double>& v,
                       std::vector<double>& out) {
  out[0] = 2.0 * x[0] * v[0] + 3.0 * v[1];
  out[1] = std::cos(x[0]) * v[0] - 2.0 * v[1];
}

}  // namespace

TEST(ResidualOperator, ExactAndFiniteDifferenceJvpAgreeOnCoupledFixture) {
  const MassDescriptor identity{};
  ResidualOperator exact(coupled_domain(), identity, DaeIndex::kNotDae,
                         LinearizationFidelity::kExact, coupled_residual, coupled_exact_jvp);
  ResidualOperator approx(coupled_domain(), identity, DaeIndex::kNotDae,
                          LinearizationFidelity::kApproximate, coupled_residual);
  const std::vector<double> x{1.25, -0.4}, v{0.3, -0.7};
  const auto je = exact.apply_jvp(x, v);
  const auto jf = approx.apply_jvp(x, v);
  ASSERT_EQ(je.size(), jf.size());
  EXPECT_NEAR(je[0], jf[0], 2e-7);
  EXPECT_NEAR(je[1], jf[1], 2e-7);
}

TEST(ResidualOperator, FidelityAndDomainContractsFailClosed) {
  ResidualOperator missing(coupled_domain(), {}, DaeIndex::kNotDae,
                           LinearizationFidelity::kExact, coupled_residual);
  EXPECT_EQ(missing.support().refusal, SupportRefusal::kUnsupportedLinearization);
  EXPECT_THROW(missing.apply_jvp({1.0, 2.0}, {1.0, 0.0}), std::logic_error);

  auto bad = coupled_domain();
  bad.blocks[1].offset = 0;
  ResidualOperator overlap(bad, {}, DaeIndex::kNotDae, LinearizationFidelity::kApproximate,
                           coupled_residual);
  EXPECT_EQ(overlap.support().refusal, SupportRefusal::kInvalidDomain);
}

TEST(ResidualOperator, EvaluatorOutputsAreValidated) {
  auto nonfinite_residual = [](const std::vector<double>&, std::vector<double>& r) {
    r[0] = std::numeric_limits<double>::quiet_NaN();
  };
  ResidualOperator bad_residual(coupled_domain(), {}, DaeIndex::kNotDae,
                                LinearizationFidelity::kApproximate, nonfinite_residual);
  EXPECT_THROW(bad_residual.evaluate({1.0, 2.0}), std::runtime_error);

  auto wrong_size_jvp = [](const std::vector<double>&, const std::vector<double>&,
                           std::vector<double>& out) {
    out.pop_back();
  };
  ResidualOperator bad_jvp(coupled_domain(), {}, DaeIndex::kNotDae,
                           LinearizationFidelity::kJvp, coupled_residual, {}, wrong_size_jvp);
  EXPECT_THROW(bad_jvp.apply_jvp({1.0, 2.0}, {1.0, 0.0}), std::runtime_error);
}

TEST(ResidualOperator, RealIndex1DaeChecksConsistentInitialization) {
  // y' = -y + z, 0 = y + z - 1: a minimal semi-explicit index-1 DAE.
  const MassDescriptor mass{MassKind::kAlgebraic, {1.0, 0.0}};
  auto dae_residual = [](const std::vector<double>& x, std::vector<double>& r) {
    r[0] = -x[0] + x[1];
    r[1] = x[0] + x[1] - 1.0;
  };
  auto initialize = [](std::vector<double>& x, double) {
    x[1] = 1.0 - x[0];
    return SupportDecision{};
  };
  ResidualOperator dae(coupled_domain(), mass, DaeIndex::kIndex1,
                       LinearizationFidelity::kApproximate, dae_residual, {}, {},
                       ConsistentInitializationPolicy::kRequireInitializer, initialize);
  EXPECT_TRUE(dae.support());
  EXPECT_TRUE(dae.validate_consistent_initial_state({0.25, 0.75}, 1e-14));
  EXPECT_EQ(dae.validate_consistent_initial_state({0.25, 0.5}, 1e-14).refusal,
            SupportRefusal::kInconsistentInitialState);
  std::vector<double> state{0.25, 0.5};
  EXPECT_TRUE(dae.consistent_initialize(state, 1e-14));
  EXPECT_DOUBLE_EQ(state[0], 0.25);
  EXPECT_DOUBLE_EQ(state[1], 0.75);
  EXPECT_LE(std::abs(dae.evaluate(state)[1]), 1e-14);
}

TEST(ResidualOperator, ConsistentInitializationFailsClosed) {
  const MassDescriptor mass{MassKind::kAlgebraic, {1.0, 0.0}};
  ResidualOperator missing(coupled_domain(), mass, DaeIndex::kIndex1,
                           LinearizationFidelity::kApproximate, coupled_residual, {}, {},
                           ConsistentInitializationPolicy::kRequireInitializer);
  EXPECT_EQ(missing.support().refusal, SupportRefusal::kInconsistentInitialState);

  auto failed = [](std::vector<double>&, double) {
    return SupportDecision{SupportRefusal::kInconsistentInitialState, "fixture failed"};
  };
  ResidualOperator failure(coupled_domain(), mass, DaeIndex::kIndex1,
                           LinearizationFidelity::kApproximate, coupled_residual, {}, {},
                           ConsistentInitializationPolicy::kRequireInitializer, failed);
  std::vector<double> state{0.25, 0.5};
  EXPECT_EQ(failure.consistent_initialize(state, 1e-14).refusal,
            SupportRefusal::kInconsistentInitialState);

  auto nonfinite = [](std::vector<double>& x, double) {
    x[1] = std::numeric_limits<double>::quiet_NaN();
    return SupportDecision{};
  };
  ResidualOperator invalid(coupled_domain(), mass, DaeIndex::kIndex1,
                           LinearizationFidelity::kApproximate, coupled_residual, {}, {},
                           ConsistentInitializationPolicy::kRequireInitializer, nonfinite);
  state = {0.25, 0.5};
  EXPECT_EQ(invalid.consistent_initialize(state, 1e-14).refusal,
            SupportRefusal::kInconsistentInitialState);
}

TEST(ResidualOperator, RefusesHigherIndexAndUnsupportedMass) {
  const MassDescriptor algebraic{MassKind::kAlgebraic, {1.0, 0.0}};
  ResidualOperator higher(coupled_domain(), algebraic, DaeIndex::kHigherIndex,
                          LinearizationFidelity::kApproximate, coupled_residual);
  EXPECT_EQ(higher.support().refusal, SupportRefusal::kHigherIndex);

  const MassDescriptor singular_constant{MassKind::kConstant, {1.0, 0.0}};
  ResidualOperator unsupported(coupled_domain(), singular_constant, DaeIndex::kNotDae,
                               LinearizationFidelity::kApproximate, coupled_residual);
  EXPECT_EQ(unsupported.support().refusal, SupportRefusal::kUnsupportedMass);
}

TEST(ResidualOperator, SolveResultReasonIsDistinctFromKrylovStatus) {
  static_assert(std::is_enum_v<ResidualSolveReason>);
  const ResidualSolveResult result{3, 1e-12, ResidualSolveReason::kConverged};
  EXPECT_TRUE(result.converged());
}
