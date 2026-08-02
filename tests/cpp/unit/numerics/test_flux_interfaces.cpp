#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial_operator.hpp>

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

  POPS_HD State flux(const State& state, const Aux&, int) const { return State{state[0] * speed}; }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const {
    return speed < pops::Real(0) ? -speed : speed;
  }
};

struct OtherAdvect : Advect {};

struct NonFiniteRoeAdvect : Advect {
  POPS_HD State roe_dissipation(const State&, const Aux&, const State&, const Aux&, int) const {
    return State{std::numeric_limits<pops::Real>::quiet_NaN()};
  }
};

struct NonFiniteRoeFluxAdvect : Advect {
  POPS_HD State flux(const State&, const Aux&, int) const {
    return State{std::numeric_limits<pops::Real>::quiet_NaN()};
  }
  POPS_HD State roe_dissipation(const State&, const Aux&, const State&, const Aux&, int) const {
    return State{};
  }
};

enum class HllcFailureSite { kPhysicalFlux, kPressure, kContact, kStarState, kFinalFlux };

struct SelectiveInvalidHllc {
  using State = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  HllcFailureSite failure_site;

  POPS_HD State flux(const State& state, const Aux&, int) const {
    return failure_site == HllcFailureSite::kPhysicalFlux
               ? State{std::numeric_limits<pops::Real>::quiet_NaN()}
               : state;
  }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const { return pops::Real(1); }
  POPS_HD void wave_speeds(const State&, const Aux&, int, pops::Real& lower,
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

  POPS_HD State flux(const State& state, const Aux&, int) const { return State{state[0]}; }
  POPS_HD pops::Real max_wave_speed(const State& state, const Aux&, int) const {
    return state[0] == pops::Real(-1) ? std::numeric_limits<pops::Real>::quiet_NaN()
                                      : pops::Real(2);
  }
  POPS_HD void wave_speeds(const State& state, const Aux&, int, pops::Real& lower,
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
  pops::Real values[3]{pops::Real(11), pops::Real(4), pops::Real(13)};
  mutable int reads[3]{};

  POPS_HD pops::Real operator()(int, int, int component) const {
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

struct NonFinitePrimitiveModel {
  using State = pops::StateVec<1>;
  using Prim = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  POPS_HD State flux(const State& state, const Aux&, int) const { return state; }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const { return pops::Real(1); }
  POPS_HD State source(const State&, const Aux&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
  POPS_HD Prim to_primitive(const State&) const {
    return Prim{std::numeric_limits<pops::Real>::quiet_NaN()};
  }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }
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

  const CountingProviderStorage storage{};
  const auto bound = pops::bind_flux_providers_at<QualifiedProviderAdvect>(storage, 0, 0);
  EXPECT_EQ(storage.reads[0], 0);
  EXPECT_EQ(storage.reads[1], 1);
  EXPECT_EQ(storage.reads[2], 0);

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
  tracker.merge(pops::reduce_max_uint64_cell(pops::Box2D{{0, 0}, {2, 0}},
                                             RecordRecoverableFluxFailures{tracker.recorder()}));

  const pops::FluxFailureReport report = tracker.collective_report();
  EXPECT_EQ(report.status, pops::EvaluationStatus::kReject);
  EXPECT_EQ(report.reason_code, 0x20u);
  EXPECT_EQ(report.action(), pops::TransactionFailureAction::kRejectStep);
}

TEST(test_flux_interfaces, fatal_flux_failure_remains_typed_and_preserves_reason) {
  pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
  tracker.merge(pops::reduce_max_uint64_cell(pops::Box2D{{0, 0}, {1, 0}},
                                             RecordFatalFluxFailures{tracker.recorder()}));

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

TEST(test_flux_interfaces, face_recovery_refusal_never_reaches_the_numerical_flux) {
  static_assert(
      std::is_trivially_copyable_v<pops::ReconstructedFaceState<NonFinitePrimitiveModel>>);
  static_assert(
      std::is_trivially_copyable_v<pops::RecoveredFacePrimitive<NonFinitePrimitiveModel>>);
  const pops::Box2D domain = pops::Box2D::from_extents(4, 4);
  const pops::BoxArray cells(std::vector<pops::Box2D>{domain});
  const pops::DistributionMapping distribution(1, pops::n_ranks());
  pops::MultiFab state(cells, distribution, 1, 2);
  pops::MultiFab providers_field(cells, distribution, pops::kAuxBaseComps, 2);
  state.set_val(pops::Real(1));
  providers_field.set_val(pops::Real(0));

  const auto local_state = state.fab(0).const_array();
  const auto reconstructed = pops::reconstruct_pp_recovered<NonFinitePrimitiveModel>(
      NonFinitePrimitiveModel{}, local_state, domain.lo[0] + 1, domain.lo[1] + 1, 0, pops::Real(1),
      pops::Minmod{}, true, pops::Real(0), 0);
  ASSERT_FALSE(reconstructed.publication_permitted());
  EXPECT_EQ(reconstructed.recovery.status, pops::RecoveryStatus::kInvalidContract);
  EXPECT_EQ(reconstructed.recovery.cause, pops::RecoveryCause::kNonFiniteCandidate);
  EXPECT_EQ(reconstructed.value[0], pops::Real(1));
  const auto value_only = pops::reconstruct_pp<NonFinitePrimitiveModel>(
      NonFinitePrimitiveModel{}, local_state, domain.lo[0] + 1, domain.lo[1] + 1, 0, pops::Real(1),
      pops::Minmod{}, true, pops::Real(0), 0);
  EXPECT_TRUE(std::isnan(value_only[0]));

  std::vector<pops::Box2D> x_faces{pops::xface_box(domain)};
  std::vector<pops::Box2D> y_faces{pops::yface_box(domain)};
  pops::MultiFab flux_x(pops::BoxArray(std::move(x_faces)), distribution, 1, 0);
  pops::MultiFab flux_y(pops::BoxArray(std::move(y_faces)), distribution, 1, 0);
  try {
    pops::compute_face_fluxes<pops::Minmod, pops::RusanovFlux>(NonFinitePrimitiveModel{}, state,
                                                               providers_field, flux_x, flux_y,
                                                               pops::Real(1), pops::Real(1), true);
  } catch (const pops::FluxEvaluationFailure& failure) {
    EXPECT_EQ(failure.status(), pops::EvaluationStatus::kFailed);
    EXPECT_EQ(failure.reason_code(),
              pops::detail::kVariableRecoveryReasonBase |
                  static_cast<std::uint32_t>(pops::RecoveryCause::kNonFiniteCandidate));
    EXPECT_EQ(failure.phase(), "compute_face_fluxes");
    return;
  }
  FAIL() << "a refused primitive recovery reached or escaped the face-flux path";
}

TEST(test_flux_interfaces, native_storage_binds_only_the_exact_model_pack) {
  const ProviderAdvect physical{};
  const ProviderAdvect::State state{pops::Real(3)};
  const ProviderStorage storage{pops::Real(4)};
  const auto evaluation =
      pops::evaluate_numerical_flux_at(pops::RusanovFlux{}, physical, state, storage, 2, 3, state,
                                       storage, 2, 3, pops::FaceContext::axis_aligned(0));

  ASSERT_TRUE(evaluation.succeeded());
  EXPECT_DOUBLE_EQ(evaluation.checked_density().value[0], pops::Real(12));
  EXPECT_DOUBLE_EQ(evaluation.stability.value, pops::Real(4));
  static_assert(pops::FluxProviderValues<ProviderAdvect>::size == 3);
}
