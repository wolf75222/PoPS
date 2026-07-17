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
                                          FaceOrientation orientation_ = FaceOrientation::kPositive,
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

template <class Model>
inline constexpr int flux_provider_count = [] {
  if constexpr (requires { Model::n_aux; })
    return static_cast<int>(Model::n_aux);
  return kAuxBaseComps;
}();

/// Exact, model-qualified values before they are sealed into a bound device pack.
///
/// Unlike the historical global Aux object this type has exactly the width requested by Model.
/// The model type is part of the ABI, so values for two unrelated physical providers cannot be
/// exchanged accidentally.  The values are populated only by resolve/bind or by a typed test
/// fixture; a missing component cannot be requested through this interface.
template <class Model>
struct FluxProviderValues {
  static constexpr int size = flux_provider_count<Model>;
  static_assert(size >= kAuxBaseComps,
                "physical flux provider packs must declare the required base providers");
  static_assert(size <= kAuxMaxComps,
                "physical flux provider pack exceeds the native model capability");

  Real values[size]{};

  POPS_HD Real& operator[](int component) { return values[component]; }
  POPS_HD Real operator[](int component) const { return values[component]; }
};

/// Opaque, model-qualified provider values used by the native pointwise bridge.
///
/// The public/provider ABI is the exact qualified ProviderPack generated from the Module.  This
/// small native value is its device representation after resolve/bind.  It stores exactly the
/// model-qualified values, never the process-global Aux representation: numerical fluxes cannot
/// inspect fixed slots, named extras, or missing-value sentinels.  Only the narrow
/// PhysicalFluxView for the same Model type can consume it.
template <class Model>
class BoundFluxProviders {
 public:
  static constexpr int value_count = FluxProviderValues<Model>::size;
  BoundFluxProviders() = delete;
  POPS_HD BoundFluxProviders(const BoundFluxProviders&) = default;
  BoundFluxProviders& operator=(const BoundFluxProviders&) = delete;

 private:
  FluxProviderValues<Model> values_;

  POPS_HD explicit BoundFluxProviders(const FluxProviderValues<Model>& values) : values_(values) {}
  friend struct PhysicalFluxView<Model>;
  template <class M>
  friend POPS_HD BoundFluxProviders<M> bind_flux_providers(const FluxProviderValues<M>&);
};

template <class Model>
POPS_HD BoundFluxProviders<Model> bind_flux_providers(const FluxProviderValues<Model>& values) {
  return BoundFluxProviders<Model>(values);
}

/// Bind one exact provider pack directly from native field storage.  The caller supplies a
/// model-qualified component count at compile time; there is no global Aux object, truncation, or
/// zero-on-missing branch on this path.
template <class Model, class Storage>
POPS_HD BoundFluxProviders<Model> bind_flux_providers_at(const Storage& storage, int i, int j) {
  FluxProviderValues<Model> values{};
  for (int component = 0; component < FluxProviderValues<Model>::size; ++component)
    values[component] = storage(i, j, component);
  return bind_flux_providers<Model>(values);
}

template <class State, class ProviderPack>
struct FaceTrace {
  State state;
  ProviderPack providers;
};

template <class Model>
POPS_HD FaceTrace<typename Model::State, BoundFluxProviders<Model>> make_face_trace(
    const typename Model::State& state, const BoundFluxProviders<Model>& providers) {
  return {state, providers};
}

template <class Model, class Storage>
POPS_HD FaceTrace<typename Model::State, BoundFluxProviders<Model>> make_face_trace_at(
    const typename Model::State& state, const Storage& providers, int i, int j) {
  return {state, bind_flux_providers_at<Model>(providers, i, j)};
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
  StabilityBound stability{};
  std::uint32_t reason_code = 0;

  POPS_HD static FluxEvaluation ok(const State& value, StabilityBound bound) {
    return FluxEvaluation(EvaluationStatus::kOk, bound, 0, FluxDensity<State>{value});
  }
  POPS_HD static FluxEvaluation retry(std::uint32_t reason) {
    return FluxEvaluation(EvaluationStatus::kRetry, {}, reason, invalid_density());
  }
  POPS_HD static FluxEvaluation reject(std::uint32_t reason) {
    return FluxEvaluation(EvaluationStatus::kReject, {}, reason, invalid_density());
  }
  POPS_HD static FluxEvaluation failed(std::uint32_t reason) {
    return FluxEvaluation(EvaluationStatus::kFailed, {}, reason, invalid_density());
  }

  POPS_HD bool succeeded() const { return status == EvaluationStatus::kOk; }
  POPS_HD TransactionFailureAction failure_action() const { return transaction_action(status); }

  /// Sole access to a flux density.  A failed evaluator can never smuggle a plausible value into
  /// a spatial kernel: every non-success status produces an invalid density independently of the
  /// payload supplied by an external implementation.
  POPS_HD FluxDensity<State> checked_density() const {
    return succeeded() ? density_ : invalid_density();
  }

  /// Orientation reversal is meaningful only for a successful evaluation.  Failure status,
  /// action and qualified reason remain byte-for-byte unchanged.
  POPS_HD void reverse_orientation() {
    if (!succeeded())
      return;
    for (int component = 0; component < State::size(); ++component)
      density_.value[component] = -density_.value[component];
  }

 private:
  FluxDensity<State> density_{};

  POPS_HD FluxEvaluation(EvaluationStatus status_, StabilityBound stability_,
                         std::uint32_t reason_code_, FluxDensity<State> density)
      : status(status_), stability(stability_), reason_code(reason_code_), density_(density) {}

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

 private:
  POPS_HD static Aux physical_providers(const ProviderPack& providers) {
    Aux result{};
    if constexpr (ProviderPack::value_count > 0)
      result.phi = providers.values_[0];
    if constexpr (ProviderPack::value_count > 1)
      result.grad_x = providers.values_[1];
    if constexpr (ProviderPack::value_count > 2)
      result.grad_y = providers.values_[2];
#define POPS_FLUX_PROVIDER_ASSIGN(name, index)     \
  if constexpr (ProviderPack::value_count > index) \
    result.name = providers.values_[index];
    POPS_AUX_FIELDS(POPS_FLUX_PROVIDER_ASSIGN)
#undef POPS_FLUX_PROVIDER_ASSIGN
    if constexpr (ProviderPack::value_count > kAuxNamedBase) {
      for (int component = kAuxNamedBase; component < ProviderPack::value_count; ++component)
        result.extra[component - kAuxNamedBase] = providers.values_[component];
    }
    return result;
  }

 public:
  POPS_HD FluxDensity<State> evaluate(const Trace& trace, const FaceContext& face) const {
    const Aux providers = physical_providers(trace.providers);
    State result = physical.flux(trace.state, providers, face.axis);
    const Real sign = face.orientation_sign();
    if (sign < Real(0)) {
      for (int component = 0; component < n_vars; ++component)
        result[component] = -result[component];
    }
    return {result};
  }

  POPS_HD StabilityBound stability(const Trace& trace, const FaceContext& face) const {
    const Aux providers = physical_providers(trace.providers);
    return {physical.max_wave_speed(trace.state, providers, face.axis),
            StabilityUnit::kLengthPerTime, StabilityConvention::kNormalSpectralRadius};
  }

  POPS_HD void signed_wave_speeds(const Trace& trace, const FaceContext& face, Real& lower,
                                  Real& upper) const
    requires requires(const Model& model, const State& state, const Aux& providers, int axis,
                      Real& lo, Real& hi) { model.wave_speeds(state, providers, axis, lo, hi); }
  {
    const Aux providers = physical_providers(trace.providers);
    physical.wave_speeds(trace.state, providers, face.axis, lower, upper);
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
    requires requires(const Model& model, const State& l, const State& r, Real pl, Real pr, Real sl,
                      Real sr, int axis) { model.contact_speed(l, r, pl, pr, sl, sr, axis); }
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
                      const Aux& rp, int axis) { model.roe_dissipation(l, lp, r, rp, axis); }
  {
    const Aux left_values = physical_providers(left.providers);
    const Aux right_values = physical_providers(right.providers);
    return physical.roe_dissipation(left.state, left_values, right.state, right_values, face.axis);
  }
};

template <class T>
concept PhysicalFlux =
    requires(const T& flux, const typename T::Trace& trace, const FaceContext& face) {
      typename T::State;
      typename T::ProviderPack;
      { T::n_vars } -> std::convertible_to<int>;
      { flux.evaluate(trace, face) } -> std::same_as<FluxDensity<typename T::State>>;
      { flux.stability(trace, face) } -> std::same_as<StabilityBound>;
    };

template <class T, class Physical>
concept NumericalFlux =
    PhysicalFlux<Physical> &&
    requires(const T& numerical, const Physical& physical, const typename Physical::Trace& left,
             const typename Physical::Trace& right, const FaceContext& face) {
      {
        numerical(physical, left, right, face)
      } -> std::same_as<FluxEvaluation<typename Physical::State>>;
    };

/// Constitutive capability gates used only during route resolution.  NumericalFlux policies do not
/// receive these Models; installation wraps a conforming value in the narrow PhysicalFluxView.
template <class Model>
concept HasHLLCStructure = requires(const Model& model, const typename Model::State& state,
                                    const typename Model::State& other, const Aux& providers,
                                    Real scalar, int axis, Real& lower, Real& upper) {
  { model.pressure(state) } -> std::convertible_to<Real>;
  model.wave_speeds(state, providers, axis, lower, upper);
  {
    model.contact_speed(state, other, scalar, scalar, scalar, scalar, axis)
  } -> std::convertible_to<Real>;
  {
    model.hllc_star_state(state, scalar, scalar, scalar, axis)
  } -> std::same_as<typename Model::State>;
};

template <class Model>
concept HasRoeDissipation =
    requires(const Model& model, const typename Model::State& left, const Aux& left_providers,
             const typename Model::State& right, const Aux& right_providers, int axis) {
      {
        model.roe_dissipation(left, left_providers, right, right_providers, axis)
      } -> std::same_as<typename Model::State>;
    };

template <class Numerical, class Model>
POPS_HD FluxEvaluation<typename Model::State> evaluate_numerical_flux(
    const Numerical& numerical, const Model& model, const typename Model::State& left_state,
    const BoundFluxProviders<Model>& left_providers, const typename Model::State& right_state,
    const BoundFluxProviders<Model>& right_providers, const FaceContext& face) {
  const PhysicalFluxView<Model> physical{model};
  const auto left = make_face_trace<Model>(left_state, left_providers);
  const auto right = make_face_trace<Model>(right_state, right_providers);
  static_assert(NumericalFlux<Numerical, PhysicalFluxView<Model>>,
                "numerical flux does not satisfy the typed two-trace contract");
  return numerical(physical, left, right, face);
}

template <class Numerical, class Model, class Storage>
POPS_HD FluxEvaluation<typename Model::State> evaluate_numerical_flux_at(
    const Numerical& numerical, const Model& model, const typename Model::State& left_state,
    const Storage& left_providers, int left_i, int left_j, const typename Model::State& right_state,
    const Storage& right_providers, int right_i, int right_j, const FaceContext& face) {
  return evaluate_numerical_flux(
      numerical, model, left_state, bind_flux_providers_at<Model>(left_providers, left_i, left_j),
      right_state, bind_flux_providers_at<Model>(right_providers, right_i, right_j), face);
}

}  // namespace pops
