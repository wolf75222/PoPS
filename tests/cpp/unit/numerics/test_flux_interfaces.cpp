#include <gtest/gtest.h>

#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct Advect {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  pops::Real speed = pops::Real(2);

  POPS_HD State flux(const State& state, const auto&, int) const { return State{state[0] * speed}; }
  POPS_HD pops::Real max_wave_speed(const State&, const auto&, int) const {
    return speed < pops::Real(0) ? -speed : speed;
  }
};

struct OtherAdvect : Advect {};

template <int Dim>
struct ExactAxisAdvect {
  using State = pops::StateVec<1>;
  using Aux = pops::AuxState<Dim>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;

  template <int Axis>
  POPS_HD State flux(const State& state, const auto&) const {
    static_assert(Axis >= 0 && Axis < Dim);
    return State{pops::Real(Axis + 1) * state[0]};
  }

  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State&, const auto&) const {
    static_assert(Axis >= 0 && Axis < Dim);
    return pops::Real(Axis + 1);
  }
};

struct NonFiniteRoeAdvect : Advect {
  POPS_HD State roe_dissipation(const State&, const auto&, const State&, const auto&, int) const {
    return State{std::numeric_limits<pops::Real>::quiet_NaN()};
  }
};

struct NonFiniteRoeFluxAdvect : Advect {
  POPS_HD State flux(const State&, const auto&, int) const {
    return State{std::numeric_limits<pops::Real>::quiet_NaN()};
  }
  POPS_HD State roe_dissipation(const State&, const auto&, const State&, const auto&, int) const {
    return State{};
  }
};

enum class RiemannPolicyCase : std::uint8_t { kRequestedSucceeds, kFallbackSucceeds, kRejects };

struct RiemannPolicyAdvect : Advect {
  RiemannPolicyCase policy_case = RiemannPolicyCase::kRequestedSucceeds;

  RiemannPolicyAdvect() = default;
  POPS_HD explicit RiemannPolicyAdvect(RiemannPolicyCase selected) : policy_case(selected) {}

  POPS_HD pops::Real max_wave_speed(const State&, const auto&, int) const {
    return policy_case == RiemannPolicyCase::kRejects ? std::numeric_limits<pops::Real>::quiet_NaN()
                                                      : pops::Real(2);
  }
  POPS_HD void wave_speeds(const State&, const auto&, int, pops::Real& lower,
                           pops::Real& upper) const {
    if (policy_case == RiemannPolicyCase::kRejects) {
      lower = upper = std::numeric_limits<pops::Real>::quiet_NaN();
      return;
    }
    lower = pops::Real(-1);
    upper = pops::Real(3);
  }
  POPS_HD State roe_dissipation(const State& left, const auto&, const State& right, const auto&,
                                int) const {
    if (policy_case == RiemannPolicyCase::kFallbackSucceeds)
      return State{std::numeric_limits<pops::Real>::quiet_NaN()};
    return State{pops::Real(2) * (right[0] - left[0])};
  }
};

using PreparedRoeRecovery =
    pops::PreparedRiemannRecoveryPolicy<pops::RoeFlux, pops::HLLFlux, pops::RusanovFlux,
                                        pops::RejectRiemannRecovery>;

struct DeviceRiemannRecoveryProbe {
  POPS_HD void operator()(int, int, std::uint64_t& encoded) const {
    pops::FluxProviderValues<RiemannPolicyAdvect> values{};
    const auto bound = pops::bind_flux_providers<RiemannPolicyAdvect>(values);
    const auto evaluation = pops::evaluate_numerical_flux(
        PreparedRoeRecovery{}, RiemannPolicyAdvect{RiemannPolicyCase::kFallbackSucceeds},
        RiemannPolicyAdvect::State{pops::Real(1)}, bound, RiemannPolicyAdvect::State{pops::Real(2)},
        bound, pops::FaceContext::axis_aligned(0));
    encoded = (static_cast<std::uint64_t>(evaluation.used_solver) << 8) |
              static_cast<std::uint64_t>(evaluation.attempt_count);
  }
};

enum class HllcFailureSite { kPhysicalFlux, kPressure, kContact, kStarState, kFinalFlux };

struct SelectiveInvalidHllc {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  HllcFailureSite failure_site;

  POPS_HD State flux(const State& state, const auto&, int) const {
    return failure_site == HllcFailureSite::kPhysicalFlux
               ? State{std::numeric_limits<pops::Real>::quiet_NaN()}
               : state;
  }
  POPS_HD pops::Real max_wave_speed(const State&, const auto&, int) const { return pops::Real(1); }
  POPS_HD void wave_speeds(const State&, const auto&, int, pops::Real& lower,
                           pops::Real& upper) const {
    const pops::Real magnitude = failure_site == HllcFailureSite::kFinalFlux
                                     ? std::numeric_limits<pops::Real>::max()
                                     : pops::Real(1);
    lower = -magnitude;
    upper = magnitude;
  }
  POPS_HD pops::Real pressure(const State&) const {
    return failure_site == HllcFailureSite::kPressure ? std::numeric_limits<pops::Real>::quiet_NaN()
                                                      : pops::Real(1);
  }
  POPS_HD pops::Real contact_speed(const State&, const State&, pops::Real, pops::Real, pops::Real,
                                   pops::Real, int) const {
    return failure_site == HllcFailureSite::kContact ? std::numeric_limits<pops::Real>::quiet_NaN()
                                                     : pops::Real(0);
  }
  POPS_HD State hllc_star_state(const State& state, pops::Real, pops::Real, pops::Real, int) const {
    if (failure_site == HllcFailureSite::kStarState)
      return State{std::numeric_limits<pops::Real>::quiet_NaN()};
    if (failure_site == HllcFailureSite::kFinalFlux)
      return State{std::numeric_limits<pops::Real>::max()};
    return state;
  }
};

struct SelectiveInvalidAdvect {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  POPS_HD State flux(const State& state, const auto&, int) const { return State{state[0]}; }
  POPS_HD pops::Real max_wave_speed(const State& state, const auto&, int) const {
    return state[0] == pops::Real(-1) ? std::numeric_limits<pops::Real>::quiet_NaN()
                                      : pops::Real(2);
  }
  POPS_HD void wave_speeds(const State& state, const auto&, int, pops::Real& lower,
                           pops::Real& upper) const {
    if (state[0] == pops::Real(-2)) {
      lower = upper = std::numeric_limits<pops::Real>::quiet_NaN();
      return;
    }
    lower = pops::Real(-1);
    upper = pops::Real(1);
  }
};

struct ProviderAdvect {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int dimension = pops::kNativeDimension;
  static constexpr int gradient_component =
      pops::AuxComponentLayout<dimension>::template gradient_component<0>();

  POPS_HD State flux(const State& state, const auto& providers, int) const {
    return State{state[0] * providers.template flux_provider<gradient_component>()};
  }
  POPS_HD pops::Real max_wave_speed(const State&, const auto& providers, int) const {
    const pops::Real gradient = providers.template flux_provider<gradient_component>();
    return gradient < pops::Real(0) ? -gradient : gradient;
  }
};

struct ProviderStorage {
  pops::Real gradient = pops::Real(0);

  template <int Dim>
  POPS_HD pops::Real operator()(const pops::Index<Dim>&, int component) const {
    return component == pops::AuxComponentLayout<Dim>::template gradient_component<0>()
               ? gradient
               : pops::Real(0);
  }
};

struct QualifiedProviderAdvect : ProviderAdvect {
  static constexpr int n_flux_providers = 1;
  inline static constexpr std::array<pops::QualifiedProviderRequirement, 1>
      flux_provider_requirements{{
          {"model::qualified", "field", "electric", "grad_x", "scalar", "cell", "",
           "layout::primary", "", "field::electric", true, 1},
      }};
};

struct UnavailableQualifiedProviderAdvect : ProviderAdvect {
  static constexpr int n_flux_providers = 1;
  inline static constexpr std::array<pops::QualifiedProviderRequirement, 1>
      flux_provider_requirements{{
          {"model::unavailable", "field", "electric", "grad_x", "scalar", "cell", "",
           "layout::primary", "", "field::electric", false, 1},
      }};
};

struct IncompleteQualifiedProviderAdvect : ProviderAdvect {
  static constexpr int n_flux_providers = 1;
};

struct DuplicateQualifiedProviderAdvect : ProviderAdvect {
  static constexpr int n_flux_providers = 2;
  inline static constexpr std::array<pops::QualifiedProviderRequirement, 2>
      flux_provider_requirements{{
          {"model::duplicate", "field", "electric", "grad_x", "scalar", "cell", "",
           "layout::primary", "", "field::electric", true, 1},
          {"model::duplicate", "field", "magnetic", "grad_x", "scalar", "cell", "",
           "layout::primary", "", "field::magnetic", true, 1},
      }};
};

struct CountingProviderStorage {
  pops::Real values[pops::kAuxBaseComps]{};
  mutable int reads[pops::kAuxBaseComps]{};

  template <int Dim>
  POPS_HD pops::Real operator()(const pops::Index<Dim>&, int component) const {
    ++reads[component];
    return values[component];
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

template <int Axis, int Dim>
void expect_exact_axis_dispatch() {
  using Model = ExactAxisAdvect<Dim>;
  const auto bound = providers<Model>();
  const typename Model::State state{pops::Real(3)};
  const auto evaluation =
      pops::evaluate_numerical_flux(pops::RusanovFlux{}, Model{}, state, bound, state, bound,
                                    pops::FaceContext::axis_aligned(Axis));
  ASSERT_TRUE(evaluation.succeeded());
  EXPECT_DOUBLE_EQ(evaluation.checked_density().value[0], pops::Real(3 * (Axis + 1)));
  EXPECT_DOUBLE_EQ(evaluation.stability.value, pops::Real(Axis + 1));
  if constexpr (Axis + 1 < Dim)
    expect_exact_axis_dispatch<Axis + 1, Dim>();
}

struct RejectFlux {
  template <pops::PhysicalFlux Physical>
  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(
      const Physical&, const typename Physical::Trace&, const typename Physical::Trace&,
      const pops::FaceContext&) const {
    return pops::FluxEvaluation<typename Physical::State>::reject(0x682u);
  }
};

struct RecordRecoverableFluxFailures {
  pops::FluxEvaluationRecorder recorder;

  POPS_HD void operator()(int i, int, std::uint64_t& failure) const {
    using Evaluation = pops::FluxEvaluation<pops::StateVec<1>>;
    if (i == 0)
      recorder.record(Evaluation::retry(0xffffu), failure);
    else if (i == 1)
      recorder.record(Evaluation::reject(0x10u), failure);
    else
      recorder.record(Evaluation::reject(0x20u), failure);
  }
};

struct RecordFatalFluxFailures {
  pops::FluxEvaluationRecorder recorder;

  POPS_HD void operator()(int i, int, std::uint64_t& failure) const {
    using Evaluation = pops::FluxEvaluation<pops::StateVec<1>>;
    recorder.record(i == 0 ? Evaluation::reject(0xffffffffu) : Evaluation::failed(0x42u), failure);
  }
};

}  // namespace

TEST(test_flux_interfaces, equal_state_consistency_and_declared_stability) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const auto bound = providers<Advect>();
  const auto face = pops::FaceContext::axis_aligned(0);
  const auto evaluation = pops::evaluate_numerical_flux(pops::RusanovFlux{}, physical, state, bound,
                                                        state, bound, face);

  ASSERT_EQ(evaluation.status, pops::EvaluationStatus::kOk);
  EXPECT_DOUBLE_EQ(evaluation.checked_density().value[0], physical.speed * state[0]);
  EXPECT_DOUBLE_EQ(evaluation.stability.value, physical.speed);
  EXPECT_EQ(evaluation.stability.unit, pops::StabilityUnit::kLengthPerTime);
  EXPECT_EQ(evaluation.stability.convention, pops::StabilityConvention::kNormalSpectralRadius);
  EXPECT_EQ(evaluation.requested_solver, pops::RiemannSolverId::kRusanov);
  EXPECT_EQ(evaluation.used_solver, pops::RiemannSolverId::kRusanov);
  EXPECT_EQ(evaluation.last_attempted_solver, pops::RiemannSolverId::kRusanov);
  EXPECT_EQ(evaluation.attempt_count, 1);
  EXPECT_FALSE(evaluation.used_fallback());
}

TEST(test_flux_interfaces, exact_ranked_template_models_dispatch_every_compiled_axis) {
  expect_exact_axis_dispatch<0, 1>();
  expect_exact_axis_dispatch<0, 2>();
  expect_exact_axis_dispatch<0, 3>();
}

TEST(test_flux_interfaces, prepared_riemann_recovery_is_ordered_typed_and_device_copyable) {
  static_assert(std::is_trivially_copyable_v<PreparedRoeRecovery>);
  static_assert(std::is_empty_v<PreparedRoeRecovery>);
  static_assert(PreparedRoeRecovery::candidate_count == 3);
  static_assert(PreparedRoeRecovery::ordered_solver_ids[0] == pops::RiemannSolverId::kRoe);
  static_assert(PreparedRoeRecovery::ordered_solver_ids[1] == pops::RiemannSolverId::kHll);
  static_assert(PreparedRoeRecovery::ordered_solver_ids[2] == pops::RiemannSolverId::kRusanov);
  static_assert(PreparedRoeRecovery::ordered_solver_ids[3] == pops::RiemannSolverId::kReject);

  const auto evaluate = [](RiemannPolicyCase policy_case) {
    const RiemannPolicyAdvect physical{policy_case};
    const auto bound = providers<RiemannPolicyAdvect>();
    return pops::evaluate_numerical_flux(
        pops::prepare_riemann_recovery_policy<pops::RoeFlux, pops::HLLFlux, pops::RusanovFlux,
                                              pops::RejectRiemannRecovery>(),
        physical, RiemannPolicyAdvect::State{pops::Real(1)}, bound,
        RiemannPolicyAdvect::State{pops::Real(2)}, bound, pops::FaceContext::axis_aligned(0));
  };

  const auto requested = evaluate(RiemannPolicyCase::kRequestedSucceeds);
  ASSERT_TRUE(requested.succeeded());
  EXPECT_EQ(requested.requested_solver, pops::RiemannSolverId::kRoe);
  EXPECT_EQ(requested.used_solver, pops::RiemannSolverId::kRoe);
  EXPECT_EQ(requested.last_attempted_solver, pops::RiemannSolverId::kRoe);
  EXPECT_EQ(requested.attempt_count, 1);
  EXPECT_EQ(requested.recovery_reason_code, 0u);
  EXPECT_FALSE(requested.used_fallback());

  const auto recovered = evaluate(RiemannPolicyCase::kFallbackSucceeds);
  ASSERT_TRUE(recovered.succeeded());
  EXPECT_EQ(recovered.requested_solver, pops::RiemannSolverId::kRoe);
  EXPECT_EQ(recovered.used_solver, pops::RiemannSolverId::kHll);
  EXPECT_EQ(recovered.last_attempted_solver, pops::RiemannSolverId::kHll);
  EXPECT_EQ(recovered.attempt_count, 2);
  EXPECT_EQ(recovered.recovery_reason_code,
            pops::riemann_reason_code(pops::RiemannFailureCause::kRoeNonFiniteDissipation));
  EXPECT_TRUE(recovered.used_fallback());

  std::uint64_t device_encoded = 0;
  DeviceRiemannRecoveryProbe{}(0, 0, device_encoded);
  EXPECT_EQ(device_encoded >> 8, static_cast<std::uint64_t>(pops::RiemannSolverId::kHll));
  EXPECT_EQ(device_encoded & UINT64_C(0xff), UINT64_C(2));
}

TEST(test_flux_interfaces, prepared_riemann_recovery_exhaustion_is_typed_and_cannot_publish) {
  const RiemannPolicyAdvect physical{RiemannPolicyCase::kRejects};
  const auto bound = providers<RiemannPolicyAdvect>();
  const auto rejected = pops::evaluate_numerical_flux(
      pops::prepare_riemann_recovery_policy<pops::RoeFlux, pops::HLLFlux, pops::RusanovFlux,
                                            pops::RejectRiemannRecovery>(),
      physical, RiemannPolicyAdvect::State{pops::Real(1)}, bound,
      RiemannPolicyAdvect::State{pops::Real(2)}, bound, pops::FaceContext::axis_aligned(0));

  EXPECT_EQ(rejected.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(rejected.requested_solver, pops::RiemannSolverId::kRoe);
  EXPECT_EQ(rejected.used_solver, pops::RiemannSolverId::kReject);
  EXPECT_EQ(rejected.last_attempted_solver, pops::RiemannSolverId::kRusanov);
  EXPECT_EQ(rejected.attempt_count, 3);
  EXPECT_EQ(rejected.recovery_reason_code,
            pops::riemann_reason_code(pops::RiemannFailureCause::kRoeInvalidStability));
  EXPECT_EQ(rejected.reason_code,
            pops::riemann_reason_code(pops::RiemannFailureCause::kRusanovInvalidStability));
  EXPECT_TRUE(std::isnan(rejected.checked_density().value[0]));
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
  EXPECT_DOUBLE_EQ(reversed.checked_density().value[0], -positive.checked_density().value[0]);
}

TEST(test_flux_interfaces, invalid_trace_stability_is_rejected_on_both_orientations) {
  const SelectiveInvalidAdvect physical{};
  const SelectiveInvalidAdvect::State invalid{pops::Real(-1)}, valid{pops::Real(1)};
  const auto bound = providers<SelectiveInvalidAdvect>();
  const auto positive =
      pops::FaceContext::axis_aligned(0, pops::Real(1), pops::FaceOrientation::kPositive);
  const auto negative =
      pops::FaceContext::axis_aligned(0, pops::Real(1), pops::FaceOrientation::kNegative);

  for (const auto policy : {0, 1}) {
    const auto left_invalid =
        policy == 0 ? pops::evaluate_numerical_flux(pops::RusanovFlux{}, physical, invalid, bound,
                                                    valid, bound, positive)
                    : pops::evaluate_numerical_flux(pops::HLLFlux{}, physical, invalid, bound,
                                                    valid, bound, positive);
    const auto right_invalid =
        policy == 0 ? pops::evaluate_numerical_flux(pops::RusanovFlux{}, physical, valid, bound,
                                                    invalid, bound, positive)
                    : pops::evaluate_numerical_flux(pops::HLLFlux{}, physical, valid, bound,
                                                    invalid, bound, positive);
    const auto reversed = policy == 0
                              ? pops::evaluate_numerical_flux(pops::RusanovFlux{}, physical, valid,
                                                              bound, invalid, bound, negative)
                              : pops::evaluate_numerical_flux(pops::HLLFlux{}, physical, valid,
                                                              bound, invalid, bound, negative);
    EXPECT_EQ(left_invalid.status, pops::EvaluationStatus::kReject);
    EXPECT_EQ(right_invalid.status, pops::EvaluationStatus::kReject);
    EXPECT_EQ(reversed.status, pops::EvaluationStatus::kReject);
    EXPECT_TRUE(std::isnan(left_invalid.checked_density().value[0]));
  }
}

TEST(test_flux_interfaces, hll_intervals_and_stability_metadata_are_validated_per_trace) {
  const pops::Real nan = std::numeric_limits<pops::Real>::quiet_NaN();
  pops::Real lower = pops::Real(0), upper = pops::Real(0);
  pops::detail::union_hll_speed_intervals(nan, pops::Real(1), pops::Real(-2), pops::Real(3), lower,
                                          upper);
  EXPECT_TRUE(std::isnan(lower) && std::isnan(upper));
  pops::detail::union_hll_speed_intervals(pops::Real(-2), pops::Real(3), nan, pops::Real(1), lower,
                                          upper);
  EXPECT_TRUE(std::isnan(lower) && std::isnan(upper));
  pops::detail::union_hll_speed_intervals(pops::Real(2), pops::Real(-2), pops::Real(-3),
                                          pops::Real(4), lower, upper);
  EXPECT_TRUE(std::isnan(lower) && std::isnan(upper));
  pops::detail::union_hll_speed_intervals(pops::Real(-2), pops::Real(1), pops::Real(-3),
                                          pops::Real(4), lower, upper);
  EXPECT_EQ(lower, pops::Real(-3));
  EXPECT_EQ(upper, pops::Real(4));

  const SelectiveInvalidAdvect physical{};
  const SelectiveInvalidAdvect::State invalid_waves{pops::Real(-2)}, valid_state{pops::Real(1)};
  const auto providers_pack = providers<SelectiveInvalidAdvect>();
  const auto face = pops::FaceContext::axis_aligned(0);
  EXPECT_EQ(pops::evaluate_numerical_flux(pops::HLLFlux{}, physical, invalid_waves, providers_pack,
                                          valid_state, providers_pack, face)
                .status,
            pops::EvaluationStatus::kReject);
  EXPECT_EQ(pops::evaluate_numerical_flux(pops::HLLFlux{}, physical, valid_state, providers_pack,
                                          invalid_waves, providers_pack, face)
                .status,
            pops::EvaluationStatus::kReject);

  pops::StabilityBound result{};
  const pops::StabilityBound valid{pops::Real(2), pops::StabilityUnit::kLengthPerTime,
                                   pops::StabilityConvention::kNormalSpectralRadius};
  EXPECT_FALSE(pops::detail::max_normal_stability_bound(
      {nan, pops::StabilityUnit::kLengthPerTime, pops::StabilityConvention::kNormalSpectralRadius},
      valid, result));
  EXPECT_FALSE(
      pops::detail::max_normal_stability_bound({pops::Real(-1), pops::StabilityUnit::kLengthPerTime,
                                                pops::StabilityConvention::kNormalSpectralRadius},
                                               valid, result));
  EXPECT_FALSE(
      pops::detail::max_normal_stability_bound({pops::Real(1), pops::StabilityUnit::kInverseTime,
                                                pops::StabilityConvention::kNormalSpectralRadius},
                                               valid, result));
  EXPECT_FALSE(
      pops::detail::max_normal_stability_bound({pops::Real(1), pops::StabilityUnit::kLengthPerTime,
                                                pops::StabilityConvention::kSourceFrequency},
                                               valid, result));
}

TEST(test_flux_interfaces, spatial_operator_applies_face_measure_once) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const auto bound = providers<Advect>();
  const auto face = pops::FaceContext::axis_aligned(0, pops::Real(2.5));
  const auto evaluation = pops::evaluate_numerical_flux(pops::RusanovFlux{}, physical, state, bound,
                                                        state, bound, face);
  const auto density = evaluation.checked_density();
  const auto integrated = pops::apply_face_measure(density, face);

  EXPECT_DOUBLE_EQ(integrated.value[0], pops::Real(2.5) * physical.speed * state[0]);
  static_assert(!std::is_same_v<decltype(density), decltype(integrated)>);
}

TEST(test_flux_interfaces, provider_pack_is_model_qualified_and_failure_action_is_explicit) {
  static_assert(
      !std::is_same_v<pops::BoundFluxProviders<Advect>, pops::BoundFluxProviders<OtherAdvect>>);
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

TEST(test_flux_interfaces, generated_provider_requirements_own_native_slot_reads) {
  static_assert(pops::has_qualified_flux_provider_requirements<QualifiedProviderAdvect>);
  static_assert(pops::qualified_flux_provider_requirements_valid<QualifiedProviderAdvect>());
  static_assert(
      !pops::qualified_flux_provider_requirements_valid<UnavailableQualifiedProviderAdvect>());
  static_assert(
      !pops::qualified_flux_provider_requirements_valid<IncompleteQualifiedProviderAdvect>());
  static_assert(
      !pops::qualified_flux_provider_requirements_valid<DuplicateQualifiedProviderAdvect>());

  CountingProviderStorage storage{};
  constexpr int gradient_component = ProviderAdvect::gradient_component;
  storage.values[gradient_component] = pops::Real(4);
  const auto bound = pops::bind_flux_providers_at<QualifiedProviderAdvect>(
      storage, pops::Index<pops::kNativeDimension>{});
  for (int component = 0; component < pops::kAuxBaseComps; ++component)
    EXPECT_EQ(storage.reads[component], component == gradient_component ? 1 : 0);

  const QualifiedProviderAdvect::State state{pops::Real(3)};
  const auto trace = pops::make_face_trace(state, bound);
  const auto density =
      pops::PhysicalFluxView<QualifiedProviderAdvect>{QualifiedProviderAdvect{}}.evaluate(
          trace, pops::FaceContext::axis_aligned(0));
  EXPECT_DOUBLE_EQ(density.value[0], pops::Real(12));
}

TEST(test_flux_interfaces, failed_evaluation_never_publishes_a_density) {
  const Advect physical{};
  const Advect::State state{pops::Real(3)};
  const auto bound = providers<Advect>();
  const auto evaluation =
      pops::evaluate_numerical_flux(RejectFlux{}, physical, state, bound, state, bound,
                                    pops::FaceContext::axis_aligned(1, pops::Real(4)));

  EXPECT_EQ(evaluation.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(evaluation.failure_action(), pops::TransactionFailureAction::kRejectStep);
  EXPECT_EQ(evaluation.reason_code, 0x682u);
  EXPECT_EQ(evaluation.requested_solver, pops::RiemannSolverId::kExternal);
  EXPECT_EQ(evaluation.used_solver, pops::RiemannSolverId::kReject);
  EXPECT_EQ(evaluation.last_attempted_solver, pops::RiemannSolverId::kExternal);
  EXPECT_EQ(evaluation.attempt_count, 1);
  EXPECT_TRUE(std::isnan(evaluation.checked_density().value[0]));
}

TEST(test_flux_interfaces, roe_rejects_nonfinite_dissipation_with_a_typed_cause) {
  const NonFiniteRoeAdvect physical{};
  const NonFiniteRoeAdvect::State left{pops::Real(1)}, right{pops::Real(2)};
  const auto bound = providers<NonFiniteRoeAdvect>();
  const auto evaluation = pops::evaluate_numerical_flux(
      pops::RoeFlux{}, physical, left, bound, right, bound, pops::FaceContext::axis_aligned(0));

  EXPECT_EQ(evaluation.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(evaluation.failure_action(), pops::TransactionFailureAction::kRejectStep);
  EXPECT_EQ(evaluation.reason_code,
            pops::riemann_reason_code(pops::RiemannFailureCause::kRoeNonFiniteDissipation));
  EXPECT_TRUE(std::isnan(evaluation.checked_density().value[0]));

  const NonFiniteRoeFluxAdvect invalid_flux{};
  const auto invalid_flux_bound = providers<NonFiniteRoeFluxAdvect>();
  const auto flux_evaluation = pops::evaluate_numerical_flux(
      pops::RoeFlux{}, invalid_flux, NonFiniteRoeFluxAdvect::State{pops::Real(1)},
      invalid_flux_bound, NonFiniteRoeFluxAdvect::State{pops::Real(2)}, invalid_flux_bound,
      pops::FaceContext::axis_aligned(0));
  EXPECT_EQ(flux_evaluation.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(flux_evaluation.reason_code,
            pops::riemann_reason_code(pops::RiemannFailureCause::kRoeNonFiniteFlux));
  EXPECT_TRUE(std::isnan(flux_evaluation.checked_density().value[0]));
}

TEST(test_flux_interfaces, hllc_rejects_each_nonfinite_provider_stage_with_a_typed_cause) {
  struct ExpectedFailure {
    HllcFailureSite site;
    pops::RiemannFailureCause cause;
  };
  const ExpectedFailure expected[] = {
      {HllcFailureSite::kPhysicalFlux, pops::RiemannFailureCause::kHllcNonFinitePhysicalFlux},
      {HllcFailureSite::kPressure, pops::RiemannFailureCause::kHllcNonFinitePressure},
      {HllcFailureSite::kContact, pops::RiemannFailureCause::kHllcNonFiniteContact},
      {HllcFailureSite::kStarState, pops::RiemannFailureCause::kHllcNonFiniteStarState},
      {HllcFailureSite::kFinalFlux, pops::RiemannFailureCause::kHllcNonFiniteFlux},
  };

  for (const auto& failure : expected) {
    const SelectiveInvalidHllc physical{failure.site};
    const auto bound = providers<SelectiveInvalidHllc>();
    const auto evaluation = pops::evaluate_numerical_flux(
        pops::HLLCFlux{}, physical, SelectiveInvalidHllc::State{pops::Real(1)}, bound,
        SelectiveInvalidHllc::State{pops::Real(2)}, bound, pops::FaceContext::axis_aligned(0));

    EXPECT_EQ(evaluation.status, pops::EvaluationStatus::kReject);
    EXPECT_EQ(evaluation.failure_action(), pops::TransactionFailureAction::kRejectStep);
    EXPECT_EQ(evaluation.reason_code, pops::riemann_reason_code(failure.cause));
    EXPECT_TRUE(std::isnan(evaluation.checked_density().value[0]));
  }
}

TEST(test_flux_interfaces, device_failure_reduction_orders_status_then_reason_deterministically) {
  static_assert(std::is_trivially_copyable_v<pops::FluxEvaluationTracker>);
  static_assert(sizeof(pops::FluxEvaluationTracker) == sizeof(std::uint64_t));
  pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
  std::uint64_t packed = 0;
  const RecordRecoverableFluxFailures record{tracker.recorder()};
  for (int cell = 0; cell < 3; ++cell)
    record(cell, 0, packed);
  tracker.merge(packed);

  const pops::FluxFailureReport report = tracker.collective_report();
  EXPECT_EQ(report.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(report.reason_code, 0x20u);
  EXPECT_EQ(report.action(), pops::TransactionFailureAction::kRejectStep);
}

TEST(test_flux_interfaces, fatal_flux_failure_remains_typed_and_preserves_reason) {
  pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
  std::uint64_t packed = 0;
  const RecordFatalFluxFailures record{tracker.recorder()};
  record(0, 0, packed);
  record(1, 0, packed);
  tracker.merge(packed);

  try {
    tracker.throw_if_failed("unit_flux_phase");
  } catch (const pops::FluxEvaluationFailure& failure) {
    EXPECT_EQ(failure.status(), pops::EvaluationStatus::kFailed);
    EXPECT_EQ(failure.action(), pops::TransactionFailureAction::kAbortRun);
    EXPECT_EQ(failure.reason_code(), 0x42u);
    EXPECT_EQ(failure.phase(), "unit_flux_phase");
    return;
  }
  FAIL() << "fatal device flux failure was not propagated as FluxEvaluationFailure";
}

TEST(test_flux_interfaces, recovery_report_uses_the_flux_failure_reduction_without_type_erasure) {
  pops::RecoveryReport recovery;
  recovery.status = pops::RecoveryStatus::kRejected;
  recovery.cause = pops::RecoveryCause::kExplicitRejection;
  recovery.reason_code = 0x755u;

  std::uint64_t packed = 0;
  pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
  tracker.recorder().record_recovery(recovery, packed);
  tracker.merge(packed);

  const pops::FluxFailureReport report = tracker.collective_report();
  EXPECT_EQ(report.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(report.reason_code, 0x755u);
  EXPECT_EQ(report.action(), pops::TransactionFailureAction::kRejectStep);
}

TEST(test_flux_interfaces, native_storage_binds_only_the_exact_model_pack) {
  const ProviderAdvect physical{};
  const ProviderAdvect::State state{pops::Real(3)};
  const ProviderStorage storage{pops::Real(4)};
  const pops::Index<pops::kNativeDimension> index{};
  const auto evaluation =
      pops::evaluate_numerical_flux_at(pops::RusanovFlux{}, physical, state, storage, index, state,
                                       storage, index, pops::FaceContext::axis_aligned(0));

  ASSERT_TRUE(evaluation.succeeded());
  EXPECT_DOUBLE_EQ(evaluation.checked_density().value[0], pops::Real(12));
  EXPECT_DOUBLE_EQ(evaluation.stability.value, pops::Real(4));
  static_assert(pops::FluxProviderValues<ProviderAdvect>::size == pops::kAuxBaseComps);
}
