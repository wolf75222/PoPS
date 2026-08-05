#pragma once

// Final pointwise finite-volume interfaces (ADC-682).
//
// A numerical flux receives only a narrow physical-flux view, two typed traces and a face
// context.  It never receives a System, a runtime Model, a mesh, or an unqualified auxiliary
// slot.  The spatial operator remains the sole owner of geometric face measures.

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/state.hpp>
#include <pops/mesh/index/index.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

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

/// Typed Harten entropy correction owned by a Roe physical provider.
///
/// The numerical Roe flux only consumes the provider's final ``|A| dU`` action.  Choosing and
/// parameterizing an entropy correction is therefore constitutive evidence, not a hidden
/// solver-wide constant.
struct HartenEntropyFix {
  Real relative_width = Real(0.1);

  POPS_HD Real operator()(Real eigenvalue, Real spectral_scale) const {
    const Real absolute = eigenvalue < Real(0) ? -eigenvalue : eigenvalue;
    const Real width = relative_width * spectral_scale;
    if (width <= Real(0) || absolute >= width)
      return absolute;
    return Real(0.5) * (eigenvalue * eigenvalue / width + width);
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
  bool available;
  int storage_slot;
};

enum class EvaluationStatus : std::uint8_t { kOk, kRetry, kReject, kFailed };
enum class TransactionFailureAction : std::uint8_t { kNone, kRetryStep, kRejectStep, kAbortRun };

/// Stable identity of a numerical Riemann candidate.
///
/// The value is carried by every production face result, so a successful declared fallback can
/// never be reported as if the requested solver had produced the flux.  `kReject` is a terminal
/// policy action rather than an evaluated numerical solver; `kExternal` identifies a statically
/// installed user flux whose component identity remains owned by the external-brick manifest.
enum class RiemannSolverId : std::uint8_t {
  kUnspecified = 0,
  kRusanov = 1,
  kHll = 2,
  kHllc = 3,
  kRoe = 4,
  kExternal = 254,
  kReject = 255,
};

/// Stable, device-copyable causes emitted by the built-in Riemann candidates.
///
/// External numerical-flux providers may retain their own qualified reason codes.  Built-ins use
/// this enum instead of scattering untyped literals through face kernels, so one rejected candidate
/// remains attributable after device/MPI reduction and step-transaction rollback.
enum class RiemannFailureCause : std::uint32_t {
  kRusanovInvalidStability = UINT32_C(0x53544201),
  kHllInvalidWaveInterval = UINT32_C(0x484c4c01),
  kHllInvalidStability = UINT32_C(0x53544202),
  kHllcInvalidWaveInterval = UINT32_C(0x484c4c02),
  kHllcInvalidStability = UINT32_C(0x53544203),
  kHllcNonFinitePhysicalFlux = UINT32_C(0x484c4301),
  kHllcNonFinitePressure = UINT32_C(0x484c4302),
  kHllcNonFiniteContact = UINT32_C(0x484c4303),
  kHllcNonFiniteStarState = UINT32_C(0x484c4304),
  kHllcNonFiniteFlux = UINT32_C(0x484c4305),
  kRoeInvalidStability = UINT32_C(0x53544204),
  kRoeNonFiniteDissipation = UINT32_C(0x524f4501),
  kRoeNonFiniteFlux = UINT32_C(0x524f4502),
};

POPS_HD constexpr std::uint32_t riemann_reason_code(RiemannFailureCause cause) {
  return static_cast<std::uint32_t>(cause);
}

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
inline constexpr int physical_model_dimension = [] {
  if constexpr (requires { Model::dimension; })
    return static_cast<int>(Model::dimension);
  return kNativeDimension;
}();

template <class Model>
inline constexpr int flux_provider_count = [] {
  if constexpr (requires { Model::n_aux; })
    return static_cast<int>(Model::n_aux);
  return kAuxBaseCompsFor<physical_model_dimension<Model>>;
}();

template <class Model>
inline constexpr bool has_qualified_flux_provider_requirements = requires {
  Model::n_flux_providers;
  Model::flux_provider_requirements;
};

/// Authenticate the generated logical provider ABI before a device pack can be instantiated.
///
/// Hand-written C++ test models may omit both members. Generated models must provide both, and
/// every selected provider must be available, fully qualified, and backed by one in-range native
/// storage slot. The binder consumes exactly these rows; they are not inspection-only metadata.
template <class Model>
consteval bool qualified_flux_provider_requirements_valid() {
  constexpr bool has_count = requires { Model::n_flux_providers; };
  constexpr bool has_rows = requires { Model::flux_provider_requirements; };
  if constexpr (has_count != has_rows) {
    return false;
  } else if constexpr (!has_count) {
    return true;
  } else {
    if (Model::n_flux_providers < 0 || static_cast<std::size_t>(Model::n_flux_providers) !=
                                           Model::flux_provider_requirements.size())
      return false;
    const auto nonempty = [](const char* value) { return value != nullptr && value[0] != '\0'; };
    for (std::size_t index = 0; index < Model::flux_provider_requirements.size(); ++index) {
      const auto& row = Model::flux_provider_requirements[index];
      if (!row.available || row.storage_slot < 0 ||
          row.storage_slot >= flux_provider_count<Model> || !nonempty(row.owner_qid) ||
          !nonempty(row.space_kind) || !nonempty(row.space_name) || !nonempty(row.component) ||
          !nonempty(row.representation) || !nonempty(row.centering) || !nonempty(row.layout) ||
          !nonempty(row.producer))
        return false;
      for (std::size_t previous = 0; previous < index; ++previous)
        if (Model::flux_provider_requirements[previous].storage_slot == row.storage_slot)
          return false;
    }
    return true;
  }
}

/// Exact, model-qualified values before they are sealed into a bound device pack.
///
/// Unlike the historical global Aux object this type has exactly the width requested by Model.
/// The model type is part of the ABI, so values for two unrelated physical providers cannot be
/// exchanged accidentally.  The values are populated only by resolve/bind or by a typed test
/// fixture; a missing component cannot be requested through this interface.
template <class Model>
struct FluxProviderValues {
  static constexpr int dimension = physical_model_dimension<Model>;
  static constexpr int size = flux_provider_count<Model>;
  static_assert(dimension >= 1 && dimension <= 3,
                "physical flux provider model dimension must be 1, 2, or 3");
  static_assert(qualified_flux_provider_requirements_valid<Model>(),
                "generated physical flux provider requirements are invalid");
  static_assert(size >= kAuxBaseCompsFor<dimension>,
                "physical flux provider packs must declare the required base providers");
  static_assert(size <= kAuxMaxCompsFor<dimension>,
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
  static constexpr int dimension = physical_model_dimension<Model>;
  static constexpr int value_count = FluxProviderValues<Model>::size;
  BoundFluxProviders() = delete;
  POPS_HD BoundFluxProviders(const BoundFluxProviders&) = default;
  BoundFluxProviders& operator=(const BoundFluxProviders&) = delete;

  template <int Component>
  POPS_HD Real flux_provider() const {
    static_assert(Component >= 0 && Component < value_count,
                  "physical law requested a provider outside its exact qualified pack");
    return values_[Component];
  }

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

namespace detail {

template <class Model, std::size_t Index>
inline constexpr int qualified_flux_provider_storage_slot =
    Model::flux_provider_requirements[Index].storage_slot;

template <class Model, int Dim, class Storage, std::size_t... Indices>
POPS_HD BoundFluxProviders<Model> bind_qualified_flux_providers_at(
    const Storage& storage, const Index<Dim>& index, std::index_sequence<Indices...>) {
  FluxProviderValues<Model> values{};
  ((values[qualified_flux_provider_storage_slot<Model, Indices>] =
        storage(index, qualified_flux_provider_storage_slot<Model, Indices>)),
   ...);
  return bind_flux_providers<Model>(values);
}

}  // namespace detail

/// Bind one exact provider pack directly from native field storage.  The caller supplies a
/// model-qualified component count at compile time; there is no global Aux object, truncation, or
/// zero-on-missing branch on this path.
template <class Model, int Dim, class Storage>
POPS_HD BoundFluxProviders<Model> bind_flux_providers_at(const Storage& storage,
                                                         const Index<Dim>& index) {
  static_assert(Dim == physical_model_dimension<Model>,
                "provider storage index rank differs from the physical model rank");
  if constexpr (has_qualified_flux_provider_requirements<Model>) {
    static_assert(qualified_flux_provider_requirements_valid<Model>(),
                  "generated physical flux provider requirements are invalid");
    constexpr std::size_t count = qualified_flux_provider_requirements_valid<Model>()
                                      ? static_cast<std::size_t>(Model::n_flux_providers)
                                      : 0;
    return detail::bind_qualified_flux_providers_at<Model>(storage, index,
                                                           std::make_index_sequence<count>{});
  } else {
    FluxProviderValues<Model> values{};
    for (int component = 0; component < FluxProviderValues<Model>::size; ++component)
      values[component] = storage(index, component);
    return bind_flux_providers<Model>(values);
  }
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

template <class Model, int Dim, class Storage>
POPS_HD FaceTrace<typename Model::State, BoundFluxProviders<Model>> make_face_trace_at(
    const typename Model::State& state, const Storage& providers, const Index<Dim>& index) {
  return {state, bind_flux_providers_at<Model>(providers, index)};
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
  RiemannSolverId requested_solver = RiemannSolverId::kUnspecified;
  RiemannSolverId used_solver = RiemannSolverId::kUnspecified;
  RiemannSolverId last_attempted_solver = RiemannSolverId::kUnspecified;
  std::uint32_t recovery_reason_code = 0;
  std::uint8_t attempt_count = 0;

  POPS_HD static FluxEvaluation ok(const State& value, StabilityBound bound) {
    return FluxEvaluation(EvaluationStatus::kOk, bound, 0, FluxDensity<State>{value});
  }
  POPS_HD static FluxEvaluation retry(std::uint32_t reason) {
    return FluxEvaluation(EvaluationStatus::kRetry, {}, reason, invalid_density());
  }
  POPS_HD static FluxEvaluation reject(std::uint32_t reason) {
    return FluxEvaluation(EvaluationStatus::kReject, {}, reason, invalid_density());
  }
  POPS_HD static FluxEvaluation reject(RiemannFailureCause cause) {
    return reject(riemann_reason_code(cause));
  }
  POPS_HD static FluxEvaluation failed(std::uint32_t reason) {
    return FluxEvaluation(EvaluationStatus::kFailed, {}, reason, invalid_density());
  }

  POPS_HD bool succeeded() const { return status == EvaluationStatus::kOk; }
  POPS_HD TransactionFailureAction failure_action() const { return transaction_action(status); }
  POPS_HD bool used_fallback() const {
    return succeeded() && requested_solver != RiemannSolverId::kUnspecified &&
           used_solver != requested_solver;
  }

  /// Complete provenance for one explicitly selected solver.  External policies retain their
  /// own qualified reason codes; the common evaluator supplies `kExternal` when they do not expose
  /// a native built-in identity.  A refusal has no flux-producing solver and therefore records
  /// `kReject` as the used policy action.
  POPS_HD FluxEvaluation with_single_solver(RiemannSolverId solver) const {
    FluxEvaluation result = *this;
    result.requested_solver = solver;
    result.used_solver = succeeded() ? solver : RiemannSolverId::kReject;
    result.last_attempted_solver = solver;
    result.recovery_reason_code = 0;
    result.attempt_count = 1;
    return result;
  }

  /// Complete provenance after an explicit prepared recovery chain has run.
  POPS_HD FluxEvaluation with_recovery_provenance(RiemannSolverId requested, RiemannSolverId used,
                                                  RiemannSolverId last_attempted,
                                                  std::uint32_t first_recovery_reason,
                                                  std::uint8_t attempts) const {
    FluxEvaluation result = *this;
    result.requested_solver = requested;
    result.used_solver = used;
    result.last_attempted_solver = last_attempted;
    result.recovery_reason_code = first_recovery_reason;
    result.attempt_count = attempts;
    return result;
  }

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

/// Final numerical vocabulary: retain the established FluxEvaluation spelling while exposing the
/// Riemann-specific name used by prepared recovery policies.
template <class State>
using RiemannResult = FluxEvaluation<State>;

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

namespace detail {

template <int Axis, class Model>
concept ModelFluxAt =
    requires(const Model model, const typename Model::State state,
             const BoundFluxProviders<Model> providers) {
      { model.template flux<Axis>(state, providers) } -> std::same_as<typename Model::State>;
    } ||
    requires(const Model model, const typename Model::State state,
             const BoundFluxProviders<Model> providers) {
      { model.flux(state, providers, Axis) } -> std::same_as<typename Model::State>;
    } ||
    requires(const Model model, const typename Model::State state) {
      { model.template flux<Axis>(state) } -> std::same_as<typename Model::State>;
    };

template <int Axis, class Model>
concept ModelMaximumWaveSpeedAt =
    requires(const Model model, const typename Model::State state,
             const BoundFluxProviders<Model> providers) {
      { model.template max_wave_speed<Axis>(state, providers) } -> std::convertible_to<Real>;
    } ||
    requires(const Model model, const typename Model::State state,
             const BoundFluxProviders<Model> providers) {
      { model.max_wave_speed(state, providers, Axis) } -> std::convertible_to<Real>;
    } ||
    requires(const Model model, const typename Model::State state) {
      { model.template max_wave_speed<Axis>(state) } -> std::convertible_to<Real>;
    };

template <int Axis, class Model>
concept ModelWaveSpeedsAt =
    requires(const Model model, const typename Model::State state,
             const BoundFluxProviders<Model> providers, Real& lower,
             Real& upper) { model.template wave_speeds<Axis>(state, providers, lower, upper); } ||
    requires(const Model model, const typename Model::State state,
             const BoundFluxProviders<Model> providers, Real& lower,
             Real& upper) { model.wave_speeds(state, providers, Axis, lower, upper); } ||
    requires(const Model model, const typename Model::State state, Real& lower, Real& upper) {
      model.template wave_speeds<Axis>(state, lower, upper);
    };

template <int Axis, class Model>
concept ModelContactSpeedAt =
    requires(const Model model, const typename Model::State left, const typename Model::State right,
             Real scalar) {
      {
        model.template contact_speed<Axis>(left, right, scalar, scalar, scalar, scalar)
      } -> std::convertible_to<Real>;
    } ||
    requires(const Model model, const typename Model::State left, const typename Model::State right,
             Real scalar) {
      {
        model.contact_speed(left, right, scalar, scalar, scalar, scalar, Axis)
      } -> std::convertible_to<Real>;
    } ||
    requires(const Model model, const typename Model::State left, const typename Model::State right,
             Real scalar) {
      {
        model.template contact_speed<Axis>(left, right, scalar, scalar, scalar, scalar)
      } -> std::convertible_to<Real>;
    };

template <int Axis, class Model>
concept ModelStarStateAt =
    requires(const Model model, const typename Model::State state, Real scalar) {
      {
        model.template hllc_star_state<Axis>(state, scalar, scalar, scalar)
      } -> std::same_as<typename Model::State>;
    } || requires(const Model model, const typename Model::State state, Real scalar) {
      {
        model.hllc_star_state(state, scalar, scalar, scalar, Axis)
      } -> std::same_as<typename Model::State>;
    } || requires(const Model model, const typename Model::State state, Real scalar) {
      {
        model.template star_state<Axis>(state, scalar, scalar, scalar)
      } -> std::same_as<typename Model::State>;
    };

template <int Axis, class Model>
concept ModelRoeDissipationAt =
    requires(const Model model, const typename Model::State left,
             const BoundFluxProviders<Model> left_providers, const typename Model::State right,
             const BoundFluxProviders<Model> right_providers) {
      {
        model.template roe_dissipation<Axis>(left, left_providers, right, right_providers)
      } -> std::same_as<typename Model::State>;
    } ||
    requires(const Model model, const typename Model::State left,
             const BoundFluxProviders<Model> left_providers, const typename Model::State right,
             const BoundFluxProviders<Model> right_providers) {
      {
        model.roe_dissipation(left, left_providers, right, right_providers, Axis)
      } -> std::same_as<typename Model::State>;
    } ||
    requires(const Model model, const typename Model::State left,
             const typename Model::State right) {
      {
        model.template roe_dissipation<Axis>(left, right)
      } -> std::same_as<typename Model::State>;
    };

template <class Model, int Axis = 0>
consteval bool flux_all_axes() {
  if constexpr (!ModelFluxAt<Axis, Model> || !ModelMaximumWaveSpeedAt<Axis, Model>)
    return false;
  else if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return flux_all_axes<Model, Axis + 1>();
  return true;
}

template <class Model, int Axis = 0>
consteval bool wave_speeds_all_axes() {
  if constexpr (!ModelWaveSpeedsAt<Axis, Model>)
    return false;
  else if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return wave_speeds_all_axes<Model, Axis + 1>();
  return true;
}

template <class Model, int Axis = 0>
consteval bool hllc_all_axes() {
  if constexpr (!ModelWaveSpeedsAt<Axis, Model> || !ModelContactSpeedAt<Axis, Model> ||
                !ModelStarStateAt<Axis, Model>)
    return false;
  else if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return hllc_all_axes<Model, Axis + 1>();
  return true;
}

template <class Model, int Axis = 0>
consteval bool roe_all_axes() {
  if constexpr (!ModelRoeDissipationAt<Axis, Model>)
    return false;
  else if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return roe_all_axes<Model, Axis + 1>();
  return true;
}

template <class State>
POPS_HD State invalid_axis_state() {
  State result{};
  for (int component = 0; component < State::size(); ++component)
    result[component] = std::numeric_limits<Real>::quiet_NaN();
  return result;
}

template <int Axis, class Model>
  requires ModelFluxAt<Axis, Model>
POPS_HD typename Model::State model_flux_at(const Model& model, const typename Model::State& state,
                                            const BoundFluxProviders<Model>& providers) {
  if constexpr (requires { model.template flux<Axis>(state, providers); })
    return model.template flux<Axis>(state, providers);
  else if constexpr (requires { model.flux(state, providers, Axis); })
    return model.flux(state, providers, Axis);
  else
    return model.template flux<Axis>(state);
}

template <int Axis, class Model>
  requires ModelMaximumWaveSpeedAt<Axis, Model>
POPS_HD Real model_max_wave_speed_at(const Model& model, const typename Model::State& state,
                                     const BoundFluxProviders<Model>& providers) {
  if constexpr (requires { model.template max_wave_speed<Axis>(state, providers); })
    return model.template max_wave_speed<Axis>(state, providers);
  else if constexpr (requires { model.max_wave_speed(state, providers, Axis); })
    return model.max_wave_speed(state, providers, Axis);
  else
    return model.template max_wave_speed<Axis>(state);
}

template <int Axis = 0, class Model>
POPS_HD typename Model::State model_flux_at_runtime_axis(const Model& model,
                                                         const typename Model::State& state,
                                                         const BoundFluxProviders<Model>& providers,
                                                         int axis) {
  if (axis == Axis)
    return model_flux_at<Axis>(model, state, providers);
  if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return model_flux_at_runtime_axis<Axis + 1>(model, state, providers, axis);
  return invalid_axis_state<typename Model::State>();
}

template <int Axis = 0, class Model>
POPS_HD Real model_max_wave_speed_at_runtime_axis(const Model& model,
                                                  const typename Model::State& state,
                                                  const BoundFluxProviders<Model>& providers,
                                                  int axis) {
  if (axis == Axis)
    return model_max_wave_speed_at<Axis>(model, state, providers);
  if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return model_max_wave_speed_at_runtime_axis<Axis + 1>(model, state, providers, axis);
  return std::numeric_limits<Real>::quiet_NaN();
}

template <int Axis, class Model>
  requires ModelWaveSpeedsAt<Axis, Model>
POPS_HD void model_wave_speeds_at(const Model& model, const typename Model::State& state,
                                  const BoundFluxProviders<Model>& providers, Real& lower,
                                  Real& upper) {
  if constexpr (requires { model.template wave_speeds<Axis>(state, providers, lower, upper); })
    model.template wave_speeds<Axis>(state, providers, lower, upper);
  else if constexpr (requires { model.wave_speeds(state, providers, Axis, lower, upper); })
    model.wave_speeds(state, providers, Axis, lower, upper);
  else
    model.template wave_speeds<Axis>(state, lower, upper);
}

template <int Axis = 0, class Model>
  requires ModelWaveSpeedsAt<Axis, Model>
POPS_HD void model_wave_speeds_at_runtime_axis(const Model& model,
                                               const typename Model::State& state,
                                               const BoundFluxProviders<Model>& providers, int axis,
                                               Real& lower, Real& upper) {
  if (axis == Axis) {
    model_wave_speeds_at<Axis>(model, state, providers, lower, upper);
    return;
  }
  if constexpr (Axis + 1 < physical_model_dimension<Model>) {
    model_wave_speeds_at_runtime_axis<Axis + 1>(model, state, providers, axis, lower, upper);
    return;
  }
  lower = upper = std::numeric_limits<Real>::quiet_NaN();
}

template <int Axis, class Model>
  requires ModelContactSpeedAt<Axis, Model>
POPS_HD Real model_contact_speed_at(const Model& model, const typename Model::State& left,
                                    const typename Model::State& right, Real pressure_left,
                                    Real pressure_right, Real speed_left, Real speed_right) {
  if constexpr (requires {
                  model.template contact_speed<Axis>(left, right, pressure_left, pressure_right,
                                                     speed_left, speed_right);
                })
    return model.template contact_speed<Axis>(left, right, pressure_left, pressure_right,
                                              speed_left, speed_right);
  else
    return model.contact_speed(left, right, pressure_left, pressure_right, speed_left, speed_right,
                               Axis);
}

template <int Axis = 0, class Model>
  requires ModelContactSpeedAt<Axis, Model>
POPS_HD Real model_contact_speed_at_runtime_axis(const Model& model,
                                                 const typename Model::State& left,
                                                 const typename Model::State& right,
                                                 Real pressure_left, Real pressure_right,
                                                 Real speed_left, Real speed_right, int axis) {
  if (axis == Axis)
    return model_contact_speed_at<Axis>(model, left, right, pressure_left, pressure_right,
                                        speed_left, speed_right);
  if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return model_contact_speed_at_runtime_axis<Axis + 1>(
        model, left, right, pressure_left, pressure_right, speed_left, speed_right, axis);
  return std::numeric_limits<Real>::quiet_NaN();
}

template <int Axis, class Model>
  requires ModelStarStateAt<Axis, Model>
POPS_HD typename Model::State model_star_state_at(const Model& model,
                                                  const typename Model::State& state, Real pressure,
                                                  Real speed, Real contact) {
  if constexpr (requires { model.template hllc_star_state<Axis>(state, pressure, speed, contact); })
    return model.template hllc_star_state<Axis>(state, pressure, speed, contact);
  else if constexpr (requires { model.hllc_star_state(state, pressure, speed, contact, Axis); })
    return model.hllc_star_state(state, pressure, speed, contact, Axis);
  else
    return model.template star_state<Axis>(state, pressure, speed, contact);
}

template <int Axis = 0, class Model>
  requires ModelStarStateAt<Axis, Model>
POPS_HD typename Model::State model_star_state_at_runtime_axis(const Model& model,
                                                               const typename Model::State& state,
                                                               Real pressure, Real speed,
                                                               Real contact, int axis) {
  if (axis == Axis)
    return model_star_state_at<Axis>(model, state, pressure, speed, contact);
  if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return model_star_state_at_runtime_axis<Axis + 1>(model, state, pressure, speed, contact, axis);
  return invalid_axis_state<typename Model::State>();
}

template <int Axis, class Model>
  requires ModelRoeDissipationAt<Axis, Model>
POPS_HD typename Model::State model_roe_dissipation_at(
    const Model& model, const typename Model::State& left,
    const BoundFluxProviders<Model>& left_providers, const typename Model::State& right,
    const BoundFluxProviders<Model>& right_providers) {
  if constexpr (requires {
                  model.template roe_dissipation<Axis>(left, left_providers, right,
                                                       right_providers);
                })
    return model.template roe_dissipation<Axis>(left, left_providers, right, right_providers);
  else if constexpr (requires {
                       model.roe_dissipation(left, left_providers, right, right_providers, Axis);
                     })
    return model.roe_dissipation(left, left_providers, right, right_providers, Axis);
  else
    return model.template roe_dissipation<Axis>(left, right);
}

template <int Axis = 0, class Model>
  requires ModelRoeDissipationAt<Axis, Model>
POPS_HD typename Model::State model_roe_dissipation_at_runtime_axis(
    const Model& model, const typename Model::State& left,
    const BoundFluxProviders<Model>& left_providers, const typename Model::State& right,
    const BoundFluxProviders<Model>& right_providers, int axis) {
  if (axis == Axis)
    return model_roe_dissipation_at<Axis>(model, left, left_providers, right, right_providers);
  if constexpr (Axis + 1 < physical_model_dimension<Model>)
    return model_roe_dissipation_at_runtime_axis<Axis + 1>(model, left, left_providers, right,
                                                           right_providers, axis);
  return invalid_axis_state<typename Model::State>();
}

}  // namespace detail

/// Narrow physical constitutive interface over a bound provider pack.  Numerical-flux policies see
/// this value, never the complete runtime Model. Physical laws consume BoundFluxProviders directly;
/// no global Aux value is reconstructed on the finite-volume path.
template <class Model>
struct PhysicalFluxView {
  using State = typename Model::State;
  using ProviderPack = BoundFluxProviders<Model>;
  using Trace = FaceTrace<State, ProviderPack>;
  static constexpr int dimension = physical_model_dimension<Model>;
  static constexpr int n_vars = Model::n_vars;
  static_assert(detail::flux_all_axes<Model>(),
                "physical model must provide flux and wave speed on every ranked axis");

  Model physical;

  POPS_HD FluxDensity<State> evaluate(const Trace& trace, const FaceContext& face) const {
    State result =
        detail::model_flux_at_runtime_axis(physical, trace.state, trace.providers, face.axis);
    const Real sign = face.orientation_sign();
    if (sign < Real(0)) {
      for (int component = 0; component < n_vars; ++component)
        result[component] = -result[component];
    }
    return {result};
  }

  POPS_HD StabilityBound stability(const Trace& trace, const FaceContext& face) const {
    return {detail::model_max_wave_speed_at_runtime_axis(physical, trace.state, trace.providers,
                                                         face.axis),
            StabilityUnit::kLengthPerTime, StabilityConvention::kNormalSpectralRadius};
  }

  POPS_HD void signed_wave_speeds(const Trace& trace, const FaceContext& face, Real& lower,
                                  Real& upper) const
    requires(detail::wave_speeds_all_axes<Model>())
  {
    detail::model_wave_speeds_at_runtime_axis(physical, trace.state, trace.providers, face.axis,
                                              lower, upper);
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
    requires(detail::hllc_all_axes<Model>())
  {
    return detail::model_contact_speed_at_runtime_axis(
        physical, left, right, pressure_left, pressure_right, speed_left, speed_right, face.axis);
  }

  POPS_HD State star_state(const State& state, Real pressure, Real speed, Real contact,
                           const FaceContext& face) const
    requires(detail::hllc_all_axes<Model>())
  {
    return detail::model_star_state_at_runtime_axis(physical, state, pressure, speed, contact,
                                                    face.axis);
  }

  POPS_HD State roe_dissipation(const Trace& left, const Trace& right,
                                const FaceContext& face) const
    requires(detail::roe_all_axes<Model>())
  {
    return detail::model_roe_dissipation_at_runtime_axis(physical, left.state, left.providers,
                                                         right.state, right.providers, face.axis);
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
concept HasHLLCStructure = detail::hllc_all_axes<Model>() &&
                           requires(const Model& model, const typename Model::State& state) {
                             { model.pressure(state) } -> std::convertible_to<Real>;
                           };

template <class Model>
concept HasRoeDissipation = detail::roe_all_axes<Model>();

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
  auto result = numerical(physical, left, right, face);
  if (result.requested_solver != RiemannSolverId::kUnspecified)
    return result;
  constexpr RiemannSolverId solver = [] {
    if constexpr (requires { Numerical::solver_id; })
      return static_cast<RiemannSolverId>(Numerical::solver_id);
    return RiemannSolverId::kExternal;
  }();
  return result.with_single_solver(solver);
}

template <class Numerical, class Model, int Dim, class LeftStorage, class RightStorage>
POPS_HD FluxEvaluation<typename Model::State> evaluate_numerical_flux_at(
    const Numerical& numerical, const Model& model, const typename Model::State& left_state,
    const LeftStorage& left_providers, const Index<Dim>& left_index,
    const typename Model::State& right_state, const RightStorage& right_providers,
    const Index<Dim>& right_index, const FaceContext& face) {
  return evaluate_numerical_flux(
      numerical, model, left_state, bind_flux_providers_at<Model>(left_providers, left_index),
      right_state, bind_flux_providers_at<Model>(right_providers, right_index), face);
}

}  // namespace pops
