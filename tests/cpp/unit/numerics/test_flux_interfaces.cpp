#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>

#include <cmath>
#include <initializer_list>
#include <type_traits>

namespace {

struct Advect {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  pops::Real speed = pops::Real(2);

  POPS_HD State flux(const State& state, const Aux&, int) const {
    return State{state[0] * speed};
  }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const {
    return speed < pops::Real(0) ? -speed : speed;
  }
};

struct OtherAdvect : Advect {};

struct ProviderAdvect {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 3;

  POPS_HD State flux(const State& state, const Aux& providers, int) const {
    return State{state[0] * providers.grad_x};
  }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux& providers, int) const {
    return providers.grad_x < pops::Real(0) ? -providers.grad_x : providers.grad_x;
  }
};

struct ProviderStorage {
  pops::Real gradient = pops::Real(0);

  POPS_HD pops::Real operator()(int, int, int component) const {
    return component == 1 ? gradient : pops::Real(0);
  }
};

template <class Model>
auto providers(std::initializer_list<pops::Real> values = {}) {
  pops::FluxProviderValues<Model> resolved{};
  int component = 0;
  for (const auto value : values)
    resolved[component++] = value;
  return pops::bind_flux_providers<Model>(resolved);
}

struct RejectFlux {
  template <pops::PhysicalFlux Physical>
  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(
      const Physical&, const typename Physical::Trace&, const typename Physical::Trace&,
      const pops::FaceContext&) const {
    return pops::FluxEvaluation<typename Physical::State>::reject(0x682u);
  }
};

}  // namespace

TEST(test_flux_interfaces, equal_state_consistency_and_declared_stability) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const auto bound = providers<Advect>();
  const auto face = pops::FaceContext::axis_aligned(0);
  const auto evaluation = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, state, bound, state, bound, face);

  ASSERT_EQ(evaluation.status, pops::EvaluationStatus::kOk);
  EXPECT_DOUBLE_EQ(evaluation.checked_density().value[0], physical.speed * state[0]);
  EXPECT_DOUBLE_EQ(evaluation.stability.value, physical.speed);
  EXPECT_EQ(evaluation.stability.unit, pops::StabilityUnit::kLengthPerTime);
  EXPECT_EQ(evaluation.stability.convention,
            pops::StabilityConvention::kNormalSpectralRadius);
}

TEST(test_flux_interfaces, orientation_reversal_swaps_traces_and_negates_flux) {
  const Advect physical{};
  const Advect::State left{pops::Real(1)}, right{pops::Real(4)};
  const auto bound = providers<Advect>();
  const auto positive = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, left, bound, right, bound,
      pops::FaceContext::axis_aligned(0, pops::Real(1), pops::FaceOrientation::kPositive));
  const auto reversed = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, right, bound, left, bound,
      pops::FaceContext::axis_aligned(0, pops::Real(1), pops::FaceOrientation::kNegative));

  ASSERT_EQ(positive.status, pops::EvaluationStatus::kOk);
  ASSERT_EQ(reversed.status, pops::EvaluationStatus::kOk);
  EXPECT_DOUBLE_EQ(reversed.checked_density().value[0],
                   -positive.checked_density().value[0]);
}

TEST(test_flux_interfaces, spatial_operator_applies_face_measure_once) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const auto bound = providers<Advect>();
  const auto face = pops::FaceContext::axis_aligned(0, pops::Real(2.5));
  const auto evaluation = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, state, bound, state, bound, face);
  const auto density = evaluation.checked_density();
  const auto integrated = pops::apply_face_measure(density, face);

  EXPECT_DOUBLE_EQ(integrated.value[0], pops::Real(2.5) * physical.speed * state[0]);
  static_assert(!std::is_same_v<decltype(density), decltype(integrated)>);
}

TEST(test_flux_interfaces, provider_pack_is_model_qualified_and_failure_action_is_explicit) {
  static_assert(!std::is_same_v<pops::BoundFluxProviders<Advect>,
                                pops::BoundFluxProviders<OtherAdvect>>);
  static_assert(std::is_trivially_copyable_v<pops::BoundFluxProviders<Advect>>);
  static_assert(!std::is_constructible_v<pops::BoundFluxProviders<Advect>, pops::Aux>);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kOk),
            pops::TransactionFailureAction::kNone);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kRetry),
            pops::TransactionFailureAction::kRetryStep);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kReject),
            pops::TransactionFailureAction::kRejectStep);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kFailed),
            pops::TransactionFailureAction::kAbortRun);
}

TEST(test_flux_interfaces, failed_evaluation_never_publishes_a_density) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const auto bound = providers<Advect>();
  const auto evaluation = pops::evaluate_numerical_flux(
      RejectFlux{}, physical, state, bound, state, bound,
      pops::FaceContext::axis_aligned(1, pops::Real(4)));

  EXPECT_EQ(evaluation.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(evaluation.failure_action(), pops::TransactionFailureAction::kRejectStep);
  EXPECT_EQ(evaluation.reason_code, 0x682u);
  EXPECT_TRUE(std::isnan(evaluation.checked_density().value[0]));
}

TEST(test_flux_interfaces, native_storage_binds_only_the_exact_model_pack) {
  const ProviderAdvect physical{};
  const ProviderAdvect::State state{pops::Real(3)};
  const ProviderStorage storage{pops::Real(4)};
  const auto evaluation = pops::evaluate_numerical_flux_at(
      pops::RusanovFlux{}, physical, state, storage, 2, 3, state, storage, 2, 3,
      pops::FaceContext::axis_aligned(0));

  ASSERT_TRUE(evaluation.succeeded());
  EXPECT_DOUBLE_EQ(evaluation.checked_density().value[0], pops::Real(12));
  EXPECT_DOUBLE_EQ(evaluation.stability.value, pops::Real(4));
  static_assert(pops::FluxProviderValues<ProviderAdvect>::size == 3);
}
