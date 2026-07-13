#pragma once

// Final pointwise finite-volume interfaces (ADC-682).
//
// A numerical flux receives only a narrow physical-flux view, two typed traces and a face
// context.  It never receives a System, a runtime Model, a mesh, or an unqualified auxiliary
// slot.  The spatial operator remains the sole owner of geometric face measures.

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/state.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace pops {

enum class FaceOrientation : std::int8_t { kNegative = -1, kPositive = 1 };

/// Geometry and orientation of one face evaluation.  `face_measure` is deliberately not consumed
/// by PhysicalFlux or NumericalFlux; only `apply_face_measure` may turn a density into an
/// IntegratedFaceFlux.
struct FaceContext {
  int axis = 0;
  FaceOrientation orientation = FaceOrientation::kPositive;
  Real normal[3]{Real(1), Real(0), Real(0)};
  Real face_measure = Real(1);
  Real cell_measure = Real(1);

  POPS_HD static FaceContext axis_aligned(int axis_, Real measure = Real(1),
                                          FaceOrientation orientation_ =
                                              FaceOrientation::kPositive,
                                          Real cell_measure_ = Real(1)) {
    FaceContext result{};
    result.axis = axis_;
    result.orientation = orientation_;
    result.face_measure = measure;
    result.cell_measure = cell_measure_;
    result.normal[0] = result.normal[1] = result.normal[2] = Real(0);
    const Real sign = orientation_ == FaceOrientation::kPositive ? Real(1) : Real(-1);
    if (axis_ >= 0 && axis_ < 3)
      result.normal[axis_] = sign;
    return result;
  }

  POPS_HD Real orientation_sign() const {
    return orientation == FaceOrientation::kPositive ? Real(1) : Real(-1);
  }

  POPS_HD FaceContext canonical_orientation() const {
    return axis_aligned(axis, face_measure, FaceOrientation::kPositive, cell_measure);
  }
};

enum class StabilityUnit : std::uint8_t { kLengthPerTime, kInverseTime, kTime };
enum class StabilityConvention : std::uint8_t {
  kNormalSpectralRadius,
  kSourceFrequency,
  kAdmissibleStep
};

struct StabilityBound {
  Real value = Real(0);
  StabilityUnit unit = StabilityUnit::kLengthPerTime;
  StabilityConvention convention = StabilityConvention::kNormalSpectralRadius;
};

/// Host-side schema emitted beside every generated physical-flux brick.  It is the exact logical
/// ABI pack resolved before a device pack can be bound; storage_slot is evidence supplied by the
/// producer, never an argument position or a user-visible lookup key.
struct QualifiedProviderRequirement {
  const char* owner_qid;
  const char* space_kind;
  const char* space_name;
  const char* component;
  const char* representation;
  const char* centering;
  const char* unit;
  const char* layout;
  const char* value_kind;
  const char* producer;
  int storage_slot;
};

enum class EvaluationStatus : std::uint8_t { kOk, kRetry, kReject, kFailed };
enum class TransactionFailureAction : std::uint8_t { kNone, kRetryStep, kRejectStep, kAbortRun };

POPS_HD constexpr TransactionFailureAction transaction_action(EvaluationStatus status) {
  switch (status) {
    case EvaluationStatus::kOk:
      return TransactionFailureAction::kNone;
    case EvaluationStatus::kRetry:
      return TransactionFailureAction::kRetryStep;
    case EvaluationStatus::kReject:
      return TransactionFailureAction::kRejectStep;
    case EvaluationStatus::kFailed:
      return TransactionFailureAction::kAbortRun;
  }
  return TransactionFailureAction::kAbortRun;
}

template <class Model>
struct PhysicalFluxView;

/// Opaque, model-qualified provider values used by the native pointwise bridge.
///
/// The public/provider ABI is the exact qualified ProviderPack generated from the Module.  This
/// small native value is its device representation after resolve/bind.  The underlying Aux value
/// is private: numerical fluxes cannot inspect fixed slots, named extras, or missing-value
/// sentinels.  Only the narrow PhysicalFluxView for the same Model type can consume it.
template <class Model>
class BoundFluxProviders {
 public:
  BoundFluxProviders() = delete;

 private:
  Aux values_;

  POPS_HD explicit BoundFluxProviders(const Aux& values) : values_(values) {}
  friend struct PhysicalFluxView<Model>;
  template <class M>
  friend POPS_HD BoundFluxProviders<M> bind_flux_providers(const Aux&);
};

template <class Model>
POPS_HD BoundFluxProviders<Model> bind_flux_providers(const Aux& values) {
  return BoundFluxProviders<Model>(values);
}

template <class State, class ProviderPack>
struct FaceTrace {
  State state;
  ProviderPack providers;
};

template <class Model>
POPS_HD FaceTrace<typename Model::State, BoundFluxProviders<Model>> make_face_trace(
    const typename Model::State& state, const Aux& providers) {
  return {state, bind_flux_providers<Model>(providers)};
}

template <class State>
struct FluxDensity {
  State value{};
};

template <class State>
struct IntegratedFaceFlux {
  State value{};
};

template <class State>
struct FluxEvaluation {
  EvaluationStatus status = EvaluationStatus::kFailed;
  FluxDensity<State> density{};
  StabilityBound stability{};
  std::uint32_t reason_code = 0;

  POPS_HD static FluxEvaluation ok(const State& value, StabilityBound bound) {
    return {EvaluationStatus::kOk, FluxDensity<State>{value}, bound, 0};
  }
  POPS_HD static FluxEvaluation retry(std::uint32_t reason) {
    return {EvaluationStatus::kRetry, invalid_density(), {}, reason};
  }
  POPS_HD static FluxEvaluation reject(std::uint32_t reason) {
    return {EvaluationStatus::kReject, invalid_density(), {}, reason};
  }
  POPS_HD static FluxEvaluation failed(std::uint32_t reason) {
    return {EvaluationStatus::kFailed, invalid_density(), {}, reason};
  }

 private:
  POPS_HD static FluxDensity<State> invalid_density() {
    State value{};
    for (int component = 0; component < State::size(); ++component)
      value[component] = std::numeric_limits<Real>::quiet_NaN();
    return {value};
  }
};

/// The only operation which accepts a FluxDensity and a geometric measure.  Its distinct return
/// type has no overload here, so an IntegratedFaceFlux cannot accidentally be integrated twice.
template <class State>
POPS_HD IntegratedFaceFlux<State> apply_face_measure(const FluxDensity<State>& density,
                                                     const FaceContext& face) {
  State value = density.value;
  for (int component = 0; component < State::size(); ++component)
    value[component] *= face.face_measure;
  return {value};
}

/// Narrow physical constitutive interface over a bound provider pack.  Numerical-flux policies see
/// this value, never the complete runtime Model.  The current native formulas still use Aux
/// internally; that storage representation is sealed behind BoundFluxProviders and cannot leak
/// into a numerical-flux signature.
template <class Model>
struct PhysicalFluxView {
  using State = typename Model::State;
  using ProviderPack = BoundFluxProviders<Model>;
  using Trace = FaceTrace<State, ProviderPack>;
  static constexpr int n_vars = Model::n_vars;

  Model physical;

  POPS_HD FluxDensity<State> evaluate(const Trace& trace, const FaceContext& face) const {
    State result = physical.flux(trace.state, trace.providers.values_, face.axis);
    const Real sign = face.orientation_sign();
    if (sign < Real(0)) {
      for (int component = 0; component < n_vars; ++component)
        result[component] = -result[component];
    }
    return {result};
  }

  POPS_HD StabilityBound stability(const Trace& trace, const FaceContext& face) const {
    return {physical.max_wave_speed(trace.state, trace.providers.values_, face.axis),
            StabilityUnit::kLengthPerTime, StabilityConvention::kNormalSpectralRadius};
  }

  POPS_HD void signed_wave_speeds(const Trace& trace, const FaceContext& face, Real& lower,
                                  Real& upper) const
    requires requires(const Model& model, const State& state, const Aux& providers, int axis,
                      Real& lo, Real& hi) {
      model.wave_speeds(state, providers, axis, lo, hi);
    }
  {
    physical.wave_speeds(trace.state, trace.providers.values_, face.axis, lower, upper);
    if (face.orientation == FaceOrientation::kNegative) {
      const Real old_lower = lower;
      lower = -upper;
      upper = -old_lower;
    }
  }

  POPS_HD Real pressure(const State& state) const
    requires requires(const Model& model, const State& value) { model.pressure(value); }
  {
    return physical.pressure(state);
  }

  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real speed_left, Real speed_right,
                             const FaceContext& face) const
    requires requires(const Model& model, const State& l, const State& r, Real pl, Real pr,
                      Real sl, Real sr, int axis) {
      model.contact_speed(l, r, pl, pr, sl, sr, axis);
    }
  {
    return physical.contact_speed(left, right, pressure_left, pressure_right, speed_left,
                                  speed_right, face.axis);
  }

  POPS_HD State star_state(const State& state, Real pressure, Real speed, Real contact,
                           const FaceContext& face) const
    requires requires(const Model& model, const State& value, Real p, Real s, Real c, int axis) {
      model.hllc_star_state(value, p, s, c, axis);
    }
  {
    return physical.hllc_star_state(state, pressure, speed, contact, face.axis);
  }

  POPS_HD State roe_dissipation(const Trace& left, const Trace& right,
                                const FaceContext& face) const
    requires requires(const Model& model, const State& l, const Aux& lp, const State& r,
                      const Aux& rp, int axis) {
      model.roe_dissipation(l, lp, r, rp, axis);
    }
  {
    return physical.roe_dissipation(left.state, left.providers.values_, right.state,
                                    right.providers.values_, face.axis);
  }
};

template <class T>
concept PhysicalFlux = requires(const T& flux, const typename T::Trace& trace,
                                const FaceContext& face) {
  typename T::State;
  typename T::ProviderPack;
  { T::n_vars } -> std::convertible_to<int>;
  { flux.evaluate(trace, face) } -> std::same_as<FluxDensity<typename T::State>>;
  { flux.stability(trace, face) } -> std::same_as<StabilityBound>;
};

template <class T, class Physical>
concept NumericalFlux = PhysicalFlux<Physical> &&
    requires(const T& numerical, const Physical& physical, const typename Physical::Trace& left,
             const typename Physical::Trace& right, const FaceContext& face) {
      { numerical(physical, left, right, face) } ->
          std::same_as<FluxEvaluation<typename Physical::State>>;
    };

/// Constitutive capability gates used only during route resolution.  NumericalFlux policies do not
/// receive these Models; installation wraps a conforming value in the narrow PhysicalFluxView.
template <class Model>
concept HasHLLCStructure =
    requires(const Model& model, const typename Model::State& state,
             const typename Model::State& other, const Aux& providers, Real scalar, int axis,
             Real& lower, Real& upper) {
      { model.pressure(state) } -> std::convertible_to<Real>;
      model.wave_speeds(state, providers, axis, lower, upper);
      { model.contact_speed(state, other, scalar, scalar, scalar, scalar, axis) } ->
          std::convertible_to<Real>;
      { model.hllc_star_state(state, scalar, scalar, scalar, axis) } ->
          std::same_as<typename Model::State>;
    };

template <class Model>
concept HasRoeDissipation =
    requires(const Model& model, const typename Model::State& left, const Aux& left_providers,
             const typename Model::State& right, const Aux& right_providers, int axis) {
      { model.roe_dissipation(left, left_providers, right, right_providers, axis) } ->
          std::same_as<typename Model::State>;
    };

template <class Numerical, class Model>
POPS_HD FluxEvaluation<typename Model::State> evaluate_numerical_flux(
    const Numerical& numerical, const Model& model, const typename Model::State& left_state,
    const Aux& left_providers, const typename Model::State& right_state,
    const Aux& right_providers, const FaceContext& face) {
  const PhysicalFluxView<Model> physical{model};
  const auto left = make_face_trace<Model>(left_state, left_providers);
  const auto right = make_face_trace<Model>(right_state, right_providers);
  static_assert(NumericalFlux<Numerical, PhysicalFluxView<Model>>,
                "numerical flux does not satisfy the typed two-trace contract");
  return numerical(physical, left, right, face);
}

}  // namespace pops
