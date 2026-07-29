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
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

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
  int total_iterations = 0;
  int total_evaluations = 0;
  Real residual_norm = std::numeric_limits<Real>::max();
  int failing_component = -1;
  std::uint32_t reason_code = 0;

  POPS_HD bool recovered() const { return status == RecoveryStatus::kRecovered; }
  POPS_HD bool publication_permitted() const { return recovered(); }
};

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
