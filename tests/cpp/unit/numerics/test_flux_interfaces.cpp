#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>

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

}  // namespace

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
