#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>

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

}  // namespace

TEST(test_flux_interfaces, equal_state_consistency_and_declared_stability) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const pops::Aux providers{};
  const auto face = pops::FaceContext::axis_aligned(0);
  const auto evaluation = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, state, providers, state, providers, face);

  ASSERT_EQ(evaluation.status, pops::EvaluationStatus::kOk);
  EXPECT_DOUBLE_EQ(evaluation.density.value[0], physical.speed * state[0]);
  EXPECT_DOUBLE_EQ(evaluation.stability.value, physical.speed);
  EXPECT_EQ(evaluation.stability.unit, pops::StabilityUnit::kLengthPerTime);
  EXPECT_EQ(evaluation.stability.convention,
            pops::StabilityConvention::kNormalSpectralRadius);
}

TEST(test_flux_interfaces, orientation_reversal_swaps_traces_and_negates_flux) {
  const Advect physical{};
  const Advect::State left{pops::Real(1)}, right{pops::Real(4)};
  const pops::Aux providers{};
  const auto positive = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, left, providers, right, providers,
      pops::FaceContext::axis_aligned(0, pops::Real(1), pops::FaceOrientation::kPositive));
  const auto reversed = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, right, providers, left, providers,
      pops::FaceContext::axis_aligned(0, pops::Real(1), pops::FaceOrientation::kNegative));

  ASSERT_EQ(positive.status, pops::EvaluationStatus::kOk);
  ASSERT_EQ(reversed.status, pops::EvaluationStatus::kOk);
  EXPECT_DOUBLE_EQ(reversed.density.value[0], -positive.density.value[0]);
}

TEST(test_flux_interfaces, spatial_operator_applies_face_measure_once) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const pops::Aux providers{};
  const auto face = pops::FaceContext::axis_aligned(0, pops::Real(2.5));
  const auto evaluation = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, physical, state, providers, state, providers, face);
  const auto integrated = pops::apply_face_measure(evaluation.density, face);

  EXPECT_DOUBLE_EQ(integrated.value[0], pops::Real(2.5) * physical.speed * state[0]);
  static_assert(!std::is_same_v<decltype(evaluation.density), decltype(integrated)>);
}

TEST(test_flux_interfaces, provider_pack_is_model_qualified_and_failure_action_is_explicit) {
  static_assert(!std::is_same_v<pops::BoundFluxProviders<Advect>,
                                pops::BoundFluxProviders<OtherAdvect>>);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kOk),
            pops::TransactionFailureAction::kNone);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kRetry),
            pops::TransactionFailureAction::kRetryStep);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kReject),
            pops::TransactionFailureAction::kRejectStep);
  EXPECT_EQ(pops::transaction_action(pops::EvaluationStatus::kFailed),
            pops::TransactionFailureAction::kAbortRun);
}
