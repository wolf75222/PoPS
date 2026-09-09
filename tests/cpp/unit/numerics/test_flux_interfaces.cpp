#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

struct ConstantAdvection {
  using State = pops::StateVec<1>;
  static constexpr int dimension = 1;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 1;

  POPS_HD State flux(const State& state, const auto& providers, int) const {
    return State{state[0] * providers.template provider<0>()};
  }
  POPS_HD pops::Real max_wave_speed(const State&, const auto& providers, int) const {
    const pops::Real velocity = providers.template provider<0>();
    return velocity < pops::Real(0) ? -velocity : velocity;
  }
};

struct NoProviderAdvection {
  using State = pops::StateVec<1>;
  static constexpr int dimension = 1;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  POPS_HD State flux(const State& state, const auto&, int) const { return state; }
  POPS_HD pops::Real max_wave_speed(const State&, const auto&, int) const { return pops::Real(1); }
};

struct NonFiniteStabilityAdvection {
  using State = pops::StateVec<1>;
  static constexpr int dimension = 1;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  POPS_HD State flux(const State& state, const auto&, int) const { return state; }
  POPS_HD pops::Real max_wave_speed(const State& state, const auto&, int) const {
    return state[0] < pops::Real(0) ? std::numeric_limits<pops::Real>::quiet_NaN()
                                    : pops::Real(1);
  }
};

struct SameSpellingOwnerA {
  static constexpr int dimension = 1;
  static constexpr int n_providers = 1;
};
struct SameSpellingOwnerB {
  static constexpr int dimension = 1;
  static constexpr int n_providers = 1;
};

template <int Dim>
struct FalliblePhysicalLaw {
  using State = pops::StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;
  POPS_HD State flux(const State& state, const auto&, int) const { return state; }
  POPS_HD pops::FluxDensity<State> flux_evaluation(const State& state, const auto&, int) const {
    if (state[0] < -2)
      return {State{123}, pops::EvaluationStatus::kFailed, 91};
    if (state[0] < 0)
      return {State{123}, pops::EvaluationStatus::kRetry, 27};
    if (state[0] > 2)
      return {State{std::numeric_limits<pops::Real>::infinity()}};
    return {state};
  }
  POPS_HD pops::Real max_wave_speed(const State&, const auto&, int) const { return 1; }
  POPS_HD void wave_speeds(const State&, const auto&, int, pops::Real& lower,
                           pops::Real& upper) const { lower = -1; upper = 1; }
  template <int Axis>
  POPS_HD State roe_dissipation(const State& left, const auto&, const State& right,
                                 const auto&) const { return State{right[0] - left[0]}; }
};

template <int Dim, class Policy>
void assert_physical_failures(const Policy& policy) {
  using Law = FalliblePhysicalLaw<Dim>;
  const auto providers = pops::bind_flux_providers<Law>(pops::FluxProviderValues<Law>{});
  for (auto orientation : {pops::FaceOrientation::kPositive, pops::FaceOrientation::kNegative}) {
    const auto face = pops::FaceContext::axis_aligned(Dim - 1, pops::Real(1), orientation);
    // Two different failures: strongest status/reason wins independently of face orientation.
    const auto mixed = pops::evaluate_numerical_flux(
        policy, Law{}, typename Law::State{-1}, providers,
        typename Law::State{-3}, providers, face);
    EXPECT_EQ(mixed.status, pops::EvaluationStatus::kFailed);
    EXPECT_EQ(mixed.reason_code, 91u);
    EXPECT_TRUE(std::isnan(mixed.checked_density().value[0]));
    const auto retry = pops::evaluate_numerical_flux(
        policy, Law{}, typename Law::State{-1}, providers,
        typename Law::State{1}, providers, face);
    EXPECT_EQ(retry.status, pops::EvaluationStatus::kRetry);
    EXPECT_EQ(retry.reason_code, 27u);
    const auto nonfinite = pops::evaluate_numerical_flux(
        policy, Law{}, typename Law::State{3}, providers,
        typename Law::State{1}, providers, face);
    constexpr bool roe = std::is_same_v<Policy, pops::RoeFlux>;
    EXPECT_EQ(nonfinite.status, roe ? pops::EvaluationStatus::kReject : pops::EvaluationStatus::kFailed);
    EXPECT_EQ(nonfinite.reason_code, pops::riemann_reason_code(
        roe ? pops::RiemannFailureCause::kRoeNonFiniteFlux :
              pops::RiemannFailureCause::kNonFinitePhysicalFlux));
  }
}

}  // namespace

TEST(FluxProviders, PhysicalFailuresSurviveThreeNumericalPoliciesAndRankedFaces) {
  assert_physical_failures<1>(pops::RusanovFlux{});
  assert_physical_failures<3>(pops::HLLFlux{});
  assert_physical_failures<3>(pops::RoeFlux{});
}

TEST(FluxProviders, SourceFreeAdapterRetainsPhysicalRetryReason) {
  using Adapted = pops::SourceFreeModel<FalliblePhysicalLaw<2>>;
  const auto providers = pops::bind_flux_providers<Adapted>(pops::FluxProviderValues<Adapted>{});
  const auto result = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, Adapted{}, Adapted::State{-1}, providers,
      Adapted::State{1}, providers, pops::FaceContext::axis_aligned(1));
  EXPECT_EQ(result.status, pops::EvaluationStatus::kRetry);
  EXPECT_EQ(result.reason_code, 27u);
  EXPECT_TRUE(std::isnan(result.checked_density().value[0]));
}

static_assert(pops::FluxProviderValues<ConstantAdvection>::size == 1);
static_assert(pops::FluxProviderValues<NoProviderAdvection>::size == 0);
static_assert(!std::is_same_v<pops::BoundFluxProviders<SameSpellingOwnerA>,
                              pops::BoundFluxProviders<SameSpellingOwnerB>>);

TEST(FluxProviders, ExactDensePackBindsOnlyDeclaredSlots) {
  pops::FluxProviderValues<ConstantAdvection> values{};
  values[0] = pops::Real(-3);
  const auto providers = pops::bind_flux_providers<ConstantAdvection>(values);
  EXPECT_EQ(providers.template provider<0>(), pops::Real(-3));
  EXPECT_EQ(ConstantAdvection{}.max_wave_speed({pops::Real(2)}, providers, 0), pops::Real(3));
}

TEST(FluxProviders, EmptyPackDoesNotNeedStorageOrAReservedPrefix) {
  const pops::FluxProviderValues<NoProviderAdvection> values{};
  const auto providers = pops::bind_flux_providers<NoProviderAdvection>(values);
  EXPECT_EQ(NoProviderAdvection{}.flux({pops::Real(4)}, providers, 0)[0], pops::Real(4));
}

TEST(FluxProviders, RusanovUsesExplicitProviderValueOnBothTraces) {
  pops::FluxProviderValues<ConstantAdvection> left_values{};
  pops::FluxProviderValues<ConstantAdvection> right_values{};
  left_values[0] = pops::Real(2);
  right_values[0] = pops::Real(2);
  const auto left = pops::bind_flux_providers<ConstantAdvection>(left_values);
  const auto right = pops::bind_flux_providers<ConstantAdvection>(right_values);
  const auto result = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, ConstantAdvection{}, ConstantAdvection::State{pops::Real(1)}, left,
      ConstantAdvection::State{pops::Real(3)}, right, pops::FaceContext::axis_aligned(0));
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.checked_density().value[0], pops::Real(2));
}

TEST(FluxProviders, IndependentConsumersWithHomonymousSlotsRemainIndependent) {
  pops::FluxProviderValues<SameSpellingOwnerA> a_values{};
  pops::FluxProviderValues<SameSpellingOwnerB> b_values{};
  a_values[0] = pops::Real(2);
  b_values[0] = pops::Real(9);
  const auto a = pops::bind_flux_providers<SameSpellingOwnerA>(a_values);
  const auto b = pops::bind_flux_providers<SameSpellingOwnerB>(b_values);
  EXPECT_EQ(a.template provider<0>(), pops::Real(2));
  EXPECT_EQ(b.template provider<0>(), pops::Real(9));
}

TEST(test_flux_interfaces, equal_state_consistency_and_declared_stability) {
  pops::FluxProviderValues<ConstantAdvection> values{};
  values[0] = pops::Real(2);
  const auto providers = pops::bind_flux_providers<ConstantAdvection>(values);
  const ConstantAdvection::State state{pops::Real(3)};
  const auto result = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, ConstantAdvection{}, state, providers, state, providers,
      pops::FaceContext::axis_aligned(0));

  ASSERT_EQ(result.status, pops::EvaluationStatus::kOk);
  EXPECT_EQ(result.checked_density().value[0], pops::Real(6));
  EXPECT_EQ(result.stability.value, pops::Real(2));
  EXPECT_EQ(result.stability.unit, pops::StabilityUnit::kLengthPerTime);
  EXPECT_EQ(result.stability.convention, pops::StabilityConvention::kNormalSpectralRadius);
}

TEST(test_flux_interfaces, invalid_trace_stability_is_rejected_on_both_orientations) {
  const pops::FluxProviderValues<NonFiniteStabilityAdvection> values{};
  const auto providers = pops::bind_flux_providers<NonFiniteStabilityAdvection>(values);
  const NonFiniteStabilityAdvection::State invalid{pops::Real(-1)};
  const NonFiniteStabilityAdvection::State valid{pops::Real(1)};

  for (const auto orientation : {pops::FaceOrientation::kPositive,
                                 pops::FaceOrientation::kNegative}) {
    const auto face = pops::FaceContext::axis_aligned(0, pops::Real(1), orientation);
    const auto left_invalid = pops::evaluate_numerical_flux(
        pops::RusanovFlux{}, NonFiniteStabilityAdvection{}, invalid, providers, valid, providers,
        face);
    const auto right_invalid = pops::evaluate_numerical_flux(
        pops::RusanovFlux{}, NonFiniteStabilityAdvection{}, valid, providers, invalid, providers,
        face);
    EXPECT_EQ(left_invalid.status, pops::EvaluationStatus::kReject);
    EXPECT_EQ(right_invalid.status, pops::EvaluationStatus::kReject);
    EXPECT_TRUE(std::isnan(left_invalid.checked_density().value[0]));
    EXPECT_TRUE(std::isnan(right_invalid.checked_density().value[0]));
  }
}
