#pragma once

/// @file
/// @brief Prepared, ordered and transactional recovery of primitive/local variables.
///
/// A recovery plan is a compile-time chain of concrete methods.  It is device-callable and owns no
/// allocation, callback registry or mutable cache.  Every method returns an explicit action and
/// cause; the chain publishes only a finite candidate accepted by the plan-wide admissibility
/// provider.  Warm starts and accepted state remain caller-owned and are changed only through the
/// explicit publication transaction below.

#include <pops/core/foundation/types.hpp>
#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
#include <pops/physics/admissibility/admissibility.hpp>
#include <pops/physics/inversion/inversion.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {

enum class RecoveryMethodKind : int {
  kUnknown = 0,
  kClosedForm = 1,
  kPreparedLocalNonlinear = 2,
  kBracketed = 3,
  kRepair = 4,
  kCustom = 5,
};

enum class RecoveryMethodAction : int {
  kCandidate = 0,
  kContinueChain = 1,
  kReject = 2,
};

enum class RecoveryStatus : int {
  kRecovered = 0,
  kExhausted = 1,
  kRejected = 2,
  kInvalidContract = 3,
};

enum class RecoveryCause : int {
  kNone = 0,
  kClosedFormUnavailable = 1,
  kIterationLimit = 2,
  kSingularJacobian = 3,
  kInadmissibleCandidate = 4,
  kSafeguardFailure = 5,
  kInvalidEvaluation = 6,
  kUnsupportedCapability = 7,
  kEvaluationRetry = 8,
  kEvaluationReject = 9,
  kEvaluationFailed = 10,
  kExplicitRejection = 11,
  kNonFiniteCandidate = 12,
  kRepairPublicationForbidden = 13,
  kMissingFailureCause = 14,
  kInvalidMethodAction = 15,
};

POPS_HD inline RecoveryCause recovery_cause_from_local_nonlinear_status(
    LocalNonlinearStatus status) {
  switch (status) {
    case LocalNonlinearStatus::kConverged:
      return RecoveryCause::kNone;
    case LocalNonlinearStatus::kIterationLimit:
      return RecoveryCause::kIterationLimit;
    case LocalNonlinearStatus::kSingularJacobian:
      return RecoveryCause::kSingularJacobian;
    case LocalNonlinearStatus::kInadmissibleCandidate:
      return RecoveryCause::kInadmissibleCandidate;
    case LocalNonlinearStatus::kSafeguardFailure:
      return RecoveryCause::kSafeguardFailure;
    case LocalNonlinearStatus::kInvalidEvaluation:
      return RecoveryCause::kInvalidEvaluation;
    case LocalNonlinearStatus::kUnsupportedCapability:
      return RecoveryCause::kUnsupportedCapability;
    case LocalNonlinearStatus::kEvaluationRetry:
      return RecoveryCause::kEvaluationRetry;
    case LocalNonlinearStatus::kEvaluationReject:
      return RecoveryCause::kEvaluationReject;
    case LocalNonlinearStatus::kEvaluationFailed:
      return RecoveryCause::kEvaluationFailed;
  }
  return RecoveryCause::kInvalidEvaluation;
}

template <int N>
struct RecoveryMethodResult {
  static_assert(N > 0, "a recovery method needs at least one variable");

  Real value[N] = {};
  RecoveryMethodAction action = RecoveryMethodAction::kContinueChain;
  RecoveryCause cause = RecoveryCause::kMissingFailureCause;
  int iterations = 0;
  int evaluations = 0;
  Real residual_norm = std::numeric_limits<Real>::max();
  int failing_component = -1;
  std::uint32_t reason_code = 0;

  POPS_HD static RecoveryMethodResult candidate(const Real (&candidate_value)[N]) {
    RecoveryMethodResult result;
    for (int component = 0; component < N; ++component)
      result.value[component] = candidate_value[component];
    result.action = RecoveryMethodAction::kCandidate;
    result.cause = RecoveryCause::kNone;
    return result;
  }

  POPS_HD static RecoveryMethodResult continue_chain(RecoveryCause failure) {
    RecoveryMethodResult result;
    result.action = RecoveryMethodAction::kContinueChain;
    result.cause = failure;
    return result;
  }

  POPS_HD static RecoveryMethodResult reject(RecoveryCause failure) {
    RecoveryMethodResult result;
    result.action = RecoveryMethodAction::kReject;
    result.cause = failure;
    return result;
  }
};

template <int N>
struct RecoveryOutcome {
  static_assert(N > 0, "a recovery outcome needs at least one variable");

  Real value[N] = {};
  RecoveryStatus status = RecoveryStatus::kExhausted;
  RecoveryCause cause = RecoveryCause::kNone;
  int attempted_methods = 0;
  int selected_method = -1;
  int last_method = -1;
  RecoveryMethodKind selected_method_kind = RecoveryMethodKind::kUnknown;
  RecoveryMethodKind last_method_kind = RecoveryMethodKind::kUnknown;
  int total_iterations = 0;
  int total_evaluations = 0;
  Real residual_norm = std::numeric_limits<Real>::max();
  int failing_component = -1;
  std::uint32_t reason_code = 0;

  POPS_HD bool recovered() const { return status == RecoveryStatus::kRecovered; }
  POPS_HD bool publication_permitted() const { return recovered(); }
};

/// Fixed-width, type-erased summary carried across runtime/component seams.
///
/// RecoveryOutcome keeps the recovered value at its compile-time width.  Runtime block registries
/// erase that width, but must not erase the decision that controls publication.  RecoveryReport is
/// therefore the exact scalar control metadata of an outcome, without a candidate buffer.
struct RecoveryReport {
  RecoveryStatus status = RecoveryStatus::kExhausted;
  RecoveryCause cause = RecoveryCause::kNone;
  int attempted_methods = 0;
  int selected_method = -1;
  int last_method = -1;
  RecoveryMethodKind selected_method_kind = RecoveryMethodKind::kUnknown;
  RecoveryMethodKind last_method_kind = RecoveryMethodKind::kUnknown;
  int total_iterations = 0;
  int total_evaluations = 0;
  Real residual_norm = std::numeric_limits<Real>::max();
  int failing_component = -1;
  std::uint32_t reason_code = 0;

  POPS_HD bool recovered() const { return status == RecoveryStatus::kRecovered; }
  POPS_HD bool publication_permitted() const { return recovered(); }
};

static_assert(std::is_trivially_copyable_v<RecoveryReport>,
              "type-erased recovery reports must remain fixed-layout copyable values");

template <int N>
POPS_HD inline RecoveryReport recovery_report(const RecoveryOutcome<N>& outcome) {
  return RecoveryReport{outcome.status,
                        outcome.cause,
                        outcome.attempted_methods,
                        outcome.selected_method,
                        outcome.last_method,
                        outcome.selected_method_kind,
                        outcome.last_method_kind,
                        outcome.total_iterations,
                        outcome.total_evaluations,
                        outcome.residual_norm,
                        outcome.failing_component,
                        outcome.reason_code};
}

inline constexpr const char* recovery_status_name(RecoveryStatus status) {
  switch (status) {
    case RecoveryStatus::kRecovered:
      return "recovered";
    case RecoveryStatus::kExhausted:
      return "exhausted";
    case RecoveryStatus::kRejected:
      return "rejected";
    case RecoveryStatus::kInvalidContract:
      return "invalid_contract";
  }
  return "unknown";
}

inline constexpr const char* recovery_method_kind_name(RecoveryMethodKind kind) {
  switch (kind) {
    case RecoveryMethodKind::kUnknown:
      return "unknown";
    case RecoveryMethodKind::kClosedForm:
      return "closed_form";
    case RecoveryMethodKind::kPreparedLocalNonlinear:
      return "prepared_local_nonlinear";
    case RecoveryMethodKind::kBracketed:
      return "bracketed";
    case RecoveryMethodKind::kRepair:
      return "repair";
    case RecoveryMethodKind::kCustom:
      return "custom";
  }
  return "unknown";
}

inline constexpr const char* recovery_cause_name(RecoveryCause cause) {
  switch (cause) {
    case RecoveryCause::kNone:
      return "none";
    case RecoveryCause::kClosedFormUnavailable:
      return "closed_form_unavailable";
    case RecoveryCause::kIterationLimit:
      return "iteration_limit";
    case RecoveryCause::kSingularJacobian:
      return "singular_jacobian";
    case RecoveryCause::kInadmissibleCandidate:
      return "inadmissible_candidate";
    case RecoveryCause::kSafeguardFailure:
      return "safeguard_failure";
    case RecoveryCause::kInvalidEvaluation:
      return "invalid_evaluation";
    case RecoveryCause::kUnsupportedCapability:
      return "unsupported_capability";
    case RecoveryCause::kEvaluationRetry:
      return "evaluation_retry";
    case RecoveryCause::kEvaluationReject:
      return "evaluation_reject";
    case RecoveryCause::kEvaluationFailed:
      return "evaluation_failed";
    case RecoveryCause::kExplicitRejection:
      return "explicit_rejection";
    case RecoveryCause::kNonFiniteCandidate:
      return "non_finite_candidate";
    case RecoveryCause::kRepairPublicationForbidden:
      return "repair_publication_forbidden";
    case RecoveryCause::kMissingFailureCause:
      return "missing_failure_cause";
    case RecoveryCause::kInvalidMethodAction:
      return "invalid_method_action";
  }
  return "unknown";
}

struct EmptyRecoveryMethodList {
  static constexpr int size = 0;

  POPS_HD constexpr RecoveryMethodKind kind_at(int) const { return RecoveryMethodKind::kUnknown; }
};

template <class Head, class Tail>
struct RecoveryMethodList {
  Head head;
  Tail tail;
  static constexpr int size = 1 + Tail::size;

  POPS_HD constexpr RecoveryMethodKind kind_at(int index) const {
    return index == 0 ? Head::kind
                      : (index > 0 ? tail.kind_at(index - 1) : RecoveryMethodKind::kUnknown);
  }
};

template <class Head>
POPS_HD constexpr auto recovery_methods(Head head) {
  return RecoveryMethodList<Head, EmptyRecoveryMethodList>{head, {}};
}

template <class Head, class... Tail>
  requires(sizeof...(Tail) > 0)
POPS_HD constexpr auto recovery_methods(Head head, Tail... tail) {
  auto prepared_tail = recovery_methods(tail...);
  return RecoveryMethodList<Head, decltype(prepared_tail)>{head, prepared_tail};
}

template <int N, class Admissible, class Methods>
struct PreparedVariableRecoveryPlan {
  static_assert(N > 0, "a recovery plan needs at least one variable");
  static_assert(Methods::size > 0, "a recovery plan needs at least one method");

  Admissible admissible;
  Methods methods;

  POPS_HD static constexpr int method_count() { return Methods::size; }
  POPS_HD constexpr RecoveryMethodKind method_kind(int index) const {
    return methods.kind_at(index);
  }
};

template <int N, class Admissible, class Methods>
POPS_HD constexpr auto prepare_variable_recovery(Admissible admissible, Methods methods) {
  return PreparedVariableRecoveryPlan<N, Admissible, Methods>{admissible, methods};
}

namespace recovery_detail {

POPS_HD inline bool recovery_finite(Real value) {
  return value == value && value <= std::numeric_limits<Real>::max() &&
         value >= -std::numeric_limits<Real>::max();
}

template <int N>
POPS_HD inline bool finite_vector(const Real (&value)[N], int* failing_component) {
  for (int component = 0; component < N; ++component)
    if (!recovery_finite(value[component])) {
      if (failing_component != nullptr)
        *failing_component = component;
      return false;
    }
  if (failing_component != nullptr)
    *failing_component = -1;
  return true;
}

template <int N>
POPS_HD inline void copy_vector(const Real (&source)[N], Real (&destination)[N]) {
  for (int component = 0; component < N; ++component)
    destination[component] = source[component];
}

template <int MethodIndex, int N, class Admissible>
POPS_HD inline void execute_recovery_chain(const EmptyRecoveryMethodList&, const Admissible&,
                                           const Real (&)[N], const Real (&)[N],
                                           RecoveryOutcome<N>&) {}

template <int MethodIndex, int N, class Admissible, class Head, class Tail>
POPS_HD inline void execute_recovery_chain(const RecoveryMethodList<Head, Tail>& methods,
                                           const Admissible& admissible, const Real (&conserved)[N],
                                           const Real (&initial_guess)[N],
                                           RecoveryOutcome<N>& outcome) {
  const RecoveryMethodResult<N> method_result = methods.head(conserved, initial_guess);
  ++outcome.attempted_methods;
  outcome.last_method = MethodIndex;
  outcome.last_method_kind = Head::kind;
  outcome.total_iterations += method_result.iterations;
  outcome.total_evaluations += method_result.evaluations;
  outcome.residual_norm = method_result.residual_norm;
  outcome.failing_component = method_result.failing_component;
  outcome.reason_code = method_result.reason_code;
  outcome.cause = method_result.cause;

  if (method_result.action == RecoveryMethodAction::kReject) {
    outcome.status = method_result.cause == RecoveryCause::kNone ? RecoveryStatus::kInvalidContract
                                                                 : RecoveryStatus::kRejected;
    if (method_result.cause == RecoveryCause::kNone)
      outcome.cause = RecoveryCause::kMissingFailureCause;
    return;
  }
  if (method_result.action == RecoveryMethodAction::kContinueChain) {
    if (method_result.cause == RecoveryCause::kNone) {
      outcome.status = RecoveryStatus::kInvalidContract;
      outcome.cause = RecoveryCause::kMissingFailureCause;
      return;
    }
    execute_recovery_chain<MethodIndex + 1>(methods.tail, admissible, conserved, initial_guess,
                                            outcome);
    return;
  }
  if (method_result.action != RecoveryMethodAction::kCandidate ||
      method_result.cause != RecoveryCause::kNone) {
    outcome.status = RecoveryStatus::kInvalidContract;
    outcome.cause = RecoveryCause::kInvalidMethodAction;
    return;
  }

  if constexpr (Head::kind == RecoveryMethodKind::kRepair) {
    outcome.status = RecoveryStatus::kInvalidContract;
    outcome.cause = RecoveryCause::kRepairPublicationForbidden;
    return;
  }

  int failing_component = -1;
  if (!finite_vector<N>(method_result.value, &failing_component)) {
    outcome.status = RecoveryStatus::kInvalidContract;
    outcome.cause = RecoveryCause::kNonFiniteCandidate;
    outcome.failing_component = failing_component;
    return;
  }
  if (!admissible(method_result.value, &failing_component)) {
    outcome.cause = RecoveryCause::kInadmissibleCandidate;
    outcome.failing_component = failing_component;
    execute_recovery_chain<MethodIndex + 1>(methods.tail, admissible, conserved, initial_guess,
                                            outcome);
    return;
  }

  copy_vector<N>(method_result.value, outcome.value);
  outcome.status = RecoveryStatus::kRecovered;
  outcome.cause = RecoveryCause::kNone;
  outcome.selected_method = MethodIndex;
  outcome.selected_method_kind = Head::kind;
  outcome.failing_component = -1;
}

}  // namespace recovery_detail

template <int N, class Admissible, class Methods>
POPS_HD inline RecoveryOutcome<N> recover_prepared_variable(
    const PreparedVariableRecoveryPlan<N, Admissible, Methods>& plan, const Real (&conserved)[N],
    const Real (&initial_guess)[N]) {
  RecoveryOutcome<N> outcome;
  recovery_detail::execute_recovery_chain<0>(plan.methods, plan.admissible, conserved,
                                             initial_guess, outcome);
  return outcome;
}

/// Admissibility provider for model conversions that impose no additional physical policy.
/// recover_prepared_variable has already rejected every non-finite component before this provider is
/// called.  This preserves each model's historical closed-form conversion without adding a hidden
/// repair, floor or fallback.
template <int N>
struct FiniteModelRecoveryAdmissibility {
  POPS_HD bool operator()(const Real (&)[N], int* failing_component) const {
    if (failing_component != nullptr)
      *failing_component = -1;
    return true;
  }
};

/// Exact conservation-law conversion contract used by the rank-generic hyperbolic bricks.
///
/// New models expose a fallible ``recover`` result rather than the historical unchecked
/// ``to_primitive`` value.  Keep that refusal visible to the prepared recovery chain; treating such
/// a model as identity would publish conservative energy in the primitive pressure slot.
template <class Model>
concept HasDeclaredStateRecovery = requires(const Model model, const typename Model::State state) {
  typename Model::Primitive;
  { model.recover(state).succeeded() } -> std::same_as<bool>;
  { model.recover(state).value[0] } -> std::convertible_to<Real>;
};

template <HasDeclaredStateRecovery Model>
struct ClosedFormDeclaredRecoveryMethod {
  static constexpr RecoveryMethodKind kind = RecoveryMethodKind::kClosedForm;
  Model model;

  POPS_HD RecoveryMethodResult<Model::n_vars> operator()(const Real (&conserved)[Model::n_vars],
                                                         const Real (&)[Model::n_vars]) const {
    typename Model::State state{};
    for (int component = 0; component < Model::n_vars; ++component)
      state[component] = conserved[component];
    const auto recovered = model.recover(state);
    if (!recovered.succeeded())
      return RecoveryMethodResult<Model::n_vars>::reject(RecoveryCause::kExplicitRejection);
    Real candidate[Model::n_vars] = {};
    for (int component = 0; component < Model::n_vars; ++component)
      candidate[component] = recovered.value[component];
    return RecoveryMethodResult<Model::n_vars>::candidate(candidate);
  }
};

/// Adapter for a model-declared primitive admissibility predicate.
///
/// The plan owns a concrete model value, so the predicate remains allocation-free and
/// device-callable.  This adapter is selected only when HasRecoveryAdmissibility<Model> is true;
/// models without the optional contract retain the historical finite-only fast path above.
template <HasRecoveryAdmissibility Model>
struct DeclaredModelRecoveryAdmissibility {
  Model model;

  POPS_HD bool operator()(const Real (&value)[Model::n_vars], int* failing_component) const {
    typename Model::Prim primitive{};
    for (int component = 0; component < Model::n_vars; ++component)
      primitive[component] = value[component];
    return model.recovery_admissible(primitive, failing_component);
  }
};

/// One declared closed-form method around the model-owned conservative -> primitive formula.
template <HasPrimitiveVars Model>
struct ClosedFormModelRecoveryMethod {
  static constexpr RecoveryMethodKind kind = RecoveryMethodKind::kClosedForm;
  Model model;

  POPS_HD RecoveryMethodResult<Model::n_vars> operator()(const Real (&conserved)[Model::n_vars],
                                                         const Real (&)[Model::n_vars]) const {
    typename Model::State state{};
    for (int component = 0; component < Model::n_vars; ++component)
      state[component] = conserved[component];
    const typename Model::Prim primitive = model.to_primitive(state);
    Real candidate[Model::n_vars] = {};
    for (int component = 0; component < Model::n_vars; ++component)
      candidate[component] = primitive[component];
    return RecoveryMethodResult<Model::n_vars>::candidate(candidate);
  }
};

/// Identity is still an explicit prepared method for scalar/no-primitive models.  Consequently the
/// type-erased runtime consumer receives the same failure contract for every block.
template <int N>
struct IdentityModelRecoveryMethod {
  static constexpr RecoveryMethodKind kind = RecoveryMethodKind::kClosedForm;

  POPS_HD RecoveryMethodResult<N> operator()(const Real (&conserved)[N], const Real (&)[N]) const {
    return RecoveryMethodResult<N>::candidate(conserved);
  }
};

/// Prepare exactly one conservative -> primitive method.  There is deliberately no repair or
/// fallback method in this compatibility route.
template <class Model>
POPS_HD constexpr auto prepare_model_variable_recovery(const Model& model) {
  constexpr int N = Model::n_vars;
  if constexpr (HasDeclaredStateRecovery<Model>) {
    return prepare_variable_recovery<N>(
        FiniteModelRecoveryAdmissibility<N>{},
        recovery_methods(ClosedFormDeclaredRecoveryMethod<Model>{model}));
  } else if constexpr (HasPrimitiveVars<Model>) {
    const auto methods = recovery_methods(ClosedFormModelRecoveryMethod<Model>{model});
    if constexpr (HasRecoveryAdmissibility<Model>) {
      return prepare_variable_recovery<N>(DeclaredModelRecoveryAdmissibility<Model>{model},
                                          methods);
    } else {
      return prepare_variable_recovery<N>(FiniteModelRecoveryAdmissibility<N>{}, methods);
    }
  } else {
    return prepare_variable_recovery<N>(FiniteModelRecoveryAdmissibility<N>{},
                                        recovery_methods(IdentityModelRecoveryMethod<N>{}));
  }
}

/// Typed source failure at the boundary between a model conversion and the generic inversion
/// authority.  Detailed candidate validity never crosses this boundary: it is owned exclusively
/// by AdmissibleSet below.
enum class VariableRecoveryInversionFailure : std::uint8_t {
  kSourceRejected = 1,
};

/// Candidate space for the generic prepared authority.
///
/// A model with an explicit fallible recovery owns a distinct primitive candidate even when it
/// deliberately does not satisfy the wider PhysicalModel concept.  The latter concept is useful
/// to spatial operators, but must not erase a conservative-to-primitive refusal at this runtime
/// boundary.  Every model without either declared conversion has one state space, so its
/// conservative-to-recovery map is the typed identity on State.  A specialized trait is required
/// here: conditional_t would still form Model::Primitive for identity models during substitution.
template <class Model, bool = HasDeclaredStateRecovery<Model>, bool = HasPrimitiveVars<Model>>
struct ModelRecoveryCandidate {
  using type = typename Model::State;
};

template <class Model, bool HasPrimitiveContract>
struct ModelRecoveryCandidate<Model, true, HasPrimitiveContract> {
  using type = typename Model::Primitive;
};

template <class Model>
struct ModelRecoveryCandidate<Model, false, true> {
  using type = typename Model::Prim;
};

template <class Model>
using ModelRecoveryCandidateType = typename ModelRecoveryCandidate<Model>::type;

/// Concrete inversion source for a model's declared conversion or typed state identity.
///
/// This is an inversion source, not an admissibility adapter.  It deliberately reports only
/// whether the model produced a detached candidate; finite, physical and projected acceptance are
/// all decided later by the prepared generic authorities.
template <class Model>
struct ModelVariableInversionSource {
  static constexpr int dimension = Model::dimension;

  Model model;

  static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.runtime.model-variable-inversion", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::uint32_t{3})
        .scalar(std::int32_t{Model::n_vars})
        .scalar(static_cast<std::uint8_t>(HasDeclaredStateRecovery<Model> ? 1 : 0))
        .scalar(static_cast<std::uint8_t>(HasPrimitiveVars<Model> ? 1 : 0));
    if constexpr (requires(const Model& value, ExactContractBuilder& builder) {
                    value.serialize_exact_parameters(builder);
                  })
      model.serialize_exact_parameters(contract);
  }

  POPS_HD InversionResult<ModelRecoveryCandidateType<Model>, VariableRecoveryInversionFailure>
  operator()(const typename Model::State& state, const ProviderValues<0>&,
             InversionWorkspaceView) const {
    using Result =
        InversionResult<ModelRecoveryCandidateType<Model>, VariableRecoveryInversionFailure>;
    // A declared recovery owns its refusal: do not replace it with an identity merely because the
    // model is not a full PhysicalModel.  Once it produces a detached candidate, finite and model
    // predicates are still evaluated later by the one ordered AdmissibleSet authority.
    if constexpr (HasDeclaredStateRecovery<Model>) {
      const auto recovered = model.recover(state);
      if (!recovered.succeeded())
        return Result::fail(VariableRecoveryInversionFailure::kSourceRejected);
      return Result::success(recovered.value);
    } else if constexpr (HasPrimitiveVars<Model>) {
      return Result::success(model.to_primitive(state));
    } else {
      return Result::success(state);
    }
  }
};

/// Device-clean model-declared admissibility predicate.  This provider is only materialized as a
/// ModelInequality member of the ordered AdmissibleSet; recovery never invokes the model predicate
/// as a second validation authority.
template <HasRecoveryAdmissibility Model>
struct ModelRecoveryAdmissibilityProvider {
  Model model;

  static constexpr PreparedProviderIdentity provider_identity() noexcept {
    if constexpr (requires {
                    {
                      Model::provider_identity()
                    } noexcept -> std::same_as<PreparedProviderIdentity>;
                  })
      return Model::provider_identity();
    return {"pops.runtime.model-recovery-admissibility", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::uint32_t{1}).scalar(std::int32_t{Model::n_vars});
    if constexpr (requires(const Model& value, ExactContractBuilder& builder) {
                    value.serialize_exact_parameters(builder);
                  })
      model.serialize_exact_parameters(contract);
  }

  POPS_HD bool operator()(const typename Model::Prim& candidate) const {
    int ignored_failing_component = -1;
    return model.recovery_admissible(candidate, &ignored_failing_component);
  }
};

/// The sole ordered admissibility authority for a prepared model recovery.  The generic finite
/// declaration always runs first; a model declaration, when present, is a typed second constraint
/// with exact provider/model identity rather than an out-of-band recovery predicate.
template <class Model>
auto prepare_model_recovery_admissibility(const Model& model) {
  constexpr int N = Model::n_vars;
  if constexpr (HasRecoveryAdmissibility<Model>) {
    return AdmissibleSet(FiniteComponents<0, N>{1},
                         CustomInequality<ModelRecoveryAdmissibilityProvider<Model>>{
                             ModelRecoveryAdmissibilityProvider<Model>{model}, 2});
  } else {
    return AdmissibleSet(FiniteComponents<0, N>{1});
  }
}

/// Detached recovery decision with observable projection activity.
template <int N>
struct PreparedVariableRecoveryAttempt {
  RecoveryOutcome<N> outcome{};
  bool projection_attempted = false;
  bool projection_changed = false;
};

/// Uniform's prepared generic variable recovery authority.
///
/// The inversion owns its reusable backend allocation.  Candidates are detached until the ordered
/// AdmissibleSet accepts them at the scheduled acceptance phase; RecoveryPublicationTransaction
/// remains the only point that can publish either the accepted value or its warm-start image.
template <class Model>
class PreparedModelVariableInversionRecovery final {
 public:
  static_assert(Model::dimension >= 1 && Model::dimension <= 3,
                "prepared variable recovery requires an exact native dimension");

  static constexpr int N = Model::n_vars;
  using State = typename Model::State;
  using Candidate = ModelRecoveryCandidateType<Model>;
  using Inputs = ProviderValues<0>;
  using Problem = VariableInversionProblem<Model::dimension, State, Inputs, Candidate,
                                           VariableRecoveryInversionFailure>;
  using Inversion = PreparedVariableInversion<Problem, ModelVariableInversionSource<Model>>;
  using Admissibility =
      decltype(prepare_model_recovery_admissibility(std::declval<const Model&>()));
  using Enforcement = PreparedAdmissibilityEnforcement<Candidate, Inputs, Admissibility>;

  explicit PreparedModelVariableInversionRecovery(const Model& model)
      : inversion_(Problem{"conservative-state",
                           "recovery-inputs",
                           "primitive-candidate",
                           "variable-inversion-failure",
                           {sizeof(Real) * static_cast<std::size_t>(N), alignof(Real)}},
                   ModelVariableInversionSource<Model>{model}),
        enforcement_(prepare_model_recovery_admissibility(model), acceptance_schedule_()) {
    ExactContractBuilder contract;
    contract.text("pops.prepared-model-variable-recovery")
        .scalar(std::uint32_t{1})
        .bytes(inversion_.collective_contract())
        .bytes(enforcement_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  PreparedModelVariableInversionRecovery(const PreparedModelVariableInversionRecovery&) = delete;
  PreparedModelVariableInversionRecovery& operator=(const PreparedModelVariableInversionRecovery&) =
      delete;
  PreparedModelVariableInversionRecovery(PreparedModelVariableInversionRecovery&&) noexcept =
      default;
  PreparedModelVariableInversionRecovery& operator=(
      PreparedModelVariableInversionRecovery&&) noexcept = default;

  [[nodiscard]] PreparedVariableRecoveryAttempt<N> recover(const Real (&conserved)[N]) {
    PreparedVariableRecoveryAttempt<N> recovered;
    recovered.outcome.attempted_methods = 1;
    recovered.outcome.last_method = 0;
    recovered.outcome.last_method_kind = RecoveryMethodKind::kClosedForm;

    State state{};
    for (int component = 0; component < N; ++component)
      state[component] = conserved[component];

    auto inversion_outcome = inversion_.attempt(state, Inputs{});
    if (!inversion_outcome.succeeded()) {
      const auto consumed = std::move(inversion_outcome).consume();
      recovered.outcome.status = RecoveryStatus::kRejected;
      recovered.outcome.cause = RecoveryCause::kExplicitRejection;
      recovered.outcome.reason_code = static_cast<std::uint32_t>(consumed.failure());
      return recovered;
    }

    const Candidate candidate = std::move(inversion_outcome).consume().candidate();
    auto enforced = enforcement_.enforce(candidate, Inputs{}, EnforcementPhase::kAcceptance);
    recovered.projection_attempted = enforced.projection_attempted;
    recovered.projection_changed = enforced.projection_changed;
    if (!enforced.checked) {
      recovered.outcome.status = RecoveryStatus::kInvalidContract;
      recovered.outcome.cause = RecoveryCause::kInvalidMethodAction;
      return recovered;
    }
    if (!enforced.admissibility.accepted) {
      recovered.outcome.status = RecoveryStatus::kRejected;
      recovered.outcome.cause = RecoveryCause::kInadmissibleCandidate;
      recovered.outcome.reason_code = enforced.admissibility.diagnostic_code;
      return recovered;
    }

    for (int component = 0; component < N; ++component)
      recovered.outcome.value[component] = enforced.candidate[component];
    recovered.outcome.status = RecoveryStatus::kRecovered;
    recovered.outcome.cause = RecoveryCause::kNone;
    recovered.outcome.selected_method = 0;
    recovered.outcome.selected_method_kind = RecoveryMethodKind::kClosedForm;
    return recovered;
  }

  [[nodiscard]] const void* workspace_allocation_identity() const noexcept {
    return inversion_.workspace_allocation_identity();
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

 private:
  static EnforcementSchedule acceptance_schedule_() {
    std::array<EnforcementRule, EnforcementSchedule::phase_count> rules{};
    rules[static_cast<std::size_t>(EnforcementPhase::kAcceptance)] = {true, false};
    return EnforcementSchedule(rules);
  }

  Inversion inversion_;
  Enforcement enforcement_;
  std::string collective_contract_;
};

/// Adapter from the common ADC-750 prepared nonlinear provider to one explicit recovery method.
/// Recoverable numerical failures advance the declared chain; fatal evaluation failures reject the
/// attempt.  Both decisions remain visible in the final RecoveryOutcome.
template <int N, class ProblemFactory>
struct PreparedLocalNonlinearRecoveryMethod {
  static constexpr RecoveryMethodKind kind = RecoveryMethodKind::kPreparedLocalNonlinear;
  ProblemFactory problem_factory;

  POPS_HD RecoveryMethodResult<N> operator()(const Real (&conserved)[N],
                                             const Real (&initial_guess)[N]) const {
    const auto problem = problem_factory(conserved);
    const LocalNonlinearCellResult<N> local =
        solve_prepared_local_nonlinear(problem, initial_guess);
    RecoveryMethodResult<N> result;
    for (int component = 0; component < N; ++component)
      result.value[component] = local.value[component];
    result.iterations = local.iterations;
    result.evaluations = local.evaluations;
    result.residual_norm = local.residual_norm;
    result.failing_component = local.failing_component;
    result.reason_code = local.reason_code;
    result.cause = recovery_cause_from_local_nonlinear_status(local.status);

    switch (local.status) {
      case LocalNonlinearStatus::kConverged:
        result.action = RecoveryMethodAction::kCandidate;
        break;
      case LocalNonlinearStatus::kInvalidEvaluation:
      case LocalNonlinearStatus::kEvaluationReject:
      case LocalNonlinearStatus::kEvaluationFailed:
        result.action = RecoveryMethodAction::kReject;
        break;
      case LocalNonlinearStatus::kIterationLimit:
      case LocalNonlinearStatus::kSingularJacobian:
      case LocalNonlinearStatus::kInadmissibleCandidate:
      case LocalNonlinearStatus::kSafeguardFailure:
      case LocalNonlinearStatus::kUnsupportedCapability:
      case LocalNonlinearStatus::kEvaluationRetry:
        result.action = RecoveryMethodAction::kContinueChain;
        break;
    }
    return result;
  }
};

template <int N, class ProblemFactory>
POPS_HD constexpr auto prepared_local_nonlinear_recovery(ProblemFactory problem_factory) {
  return PreparedLocalNonlinearRecoveryMethod<N, ProblemFactory>{problem_factory};
}

/// One caller-owned, trivially copyable warm-start slot.  A topology/state-generation mismatch is
/// an explicit cache miss; reading a stale slot never mutates or silently refreshes it.
template <int N>
struct RecoveryWarmStartSlot {
  Real value[N] = {};
  std::uint64_t topology_generation = 0;
  std::uint64_t state_generation = 0;
  bool valid = false;

  POPS_HD bool current(std::uint64_t expected_topology, std::uint64_t expected_state) const {
    return valid && topology_generation == expected_topology && state_generation == expected_state;
  }

  POPS_HD bool load_if_current(std::uint64_t expected_topology, std::uint64_t expected_state,
                               Real (&destination)[N]) const {
    if (!current(expected_topology, expected_state))
      return false;
    recovery_detail::copy_vector<N>(value, destination);
    return true;
  }

  POPS_HD void store(const Real (&source)[N], std::uint64_t topology, std::uint64_t state) {
    recovery_detail::copy_vector<N>(source, value);
    topology_generation = topology;
    state_generation = state;
    valid = true;
  }

  POPS_HD void invalidate() { valid = false; }
};

enum class RecoveryPublicationState : int {
  kOpen = 0,
  kTentative = 1,
  kCommitted = 2,
  kRolledBack = 3,
};

/// Transaction for the only mutation point of accepted variables and their warm-start cache.
/// A failed/rejected outcome cannot enter the tentative state.  Rollback restores both snapshots
/// exactly; commit makes the already-staged candidate durable to the caller.
template <int N>
class RecoveryPublicationTransaction {
 public:
  POPS_HD RecoveryPublicationTransaction(Real (&accepted_value)[N], RecoveryWarmStartSlot<N>& cache)
      : accepted_value_(&accepted_value), cache_(&cache), cache_snapshot_(cache) {
    recovery_detail::copy_vector<N>(accepted_value, value_snapshot_);
  }

  RecoveryPublicationTransaction(const RecoveryPublicationTransaction&) = delete;
  RecoveryPublicationTransaction& operator=(const RecoveryPublicationTransaction&) = delete;
  RecoveryPublicationTransaction(RecoveryPublicationTransaction&&) = delete;
  RecoveryPublicationTransaction& operator=(RecoveryPublicationTransaction&&) = delete;

  /// A tentative publication is never allowed to escape merely because a caller returns early.
  /// Device code has no exception unwinding contract to lean on, so scope exit itself is the final
  /// fail-closed guard. Only an explicit commit makes the staged value and cache durable.
  POPS_HD ~RecoveryPublicationTransaction() {
    if (state_ == RecoveryPublicationState::kOpen || state_ == RecoveryPublicationState::kTentative)
      (void)rollback();
  }

  POPS_HD bool publish_tentative(const RecoveryOutcome<N>& outcome,
                                 std::uint64_t topology_generation,
                                 std::uint64_t state_generation) {
    if (state_ != RecoveryPublicationState::kOpen || !outcome.publication_permitted())
      return false;
    recovery_detail::copy_vector<N>(outcome.value, *accepted_value_);
    cache_->store(outcome.value, topology_generation, state_generation);
    state_ = RecoveryPublicationState::kTentative;
    return true;
  }

  POPS_HD bool commit() {
    if (state_ != RecoveryPublicationState::kTentative)
      return false;
    state_ = RecoveryPublicationState::kCommitted;
    return true;
  }

  POPS_HD bool rollback() {
    if (state_ == RecoveryPublicationState::kCommitted ||
        state_ == RecoveryPublicationState::kRolledBack)
      return false;
    recovery_detail::copy_vector<N>(value_snapshot_, *accepted_value_);
    *cache_ = cache_snapshot_;
    state_ = RecoveryPublicationState::kRolledBack;
    return true;
  }

  POPS_HD RecoveryPublicationState state() const { return state_; }

 private:
  Real (*accepted_value_)[N];
  RecoveryWarmStartSlot<N>* cache_;
  Real value_snapshot_[N] = {};
  RecoveryWarmStartSlot<N> cache_snapshot_;
  RecoveryPublicationState state_ = RecoveryPublicationState::kOpen;
};

static_assert(std::is_trivially_copyable_v<RecoveryWarmStartSlot<1>>,
              "warm-start slots must remain device-copyable PODs");

}  // namespace pops
