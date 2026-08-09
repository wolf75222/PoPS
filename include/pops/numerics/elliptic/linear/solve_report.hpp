#pragma once

/// @file
/// @brief SolveReport -- the authoritative result type shared by prepared linear and nonlinear
///        solves.
///
/// One definition shared by prepared matrix-free Krylov methods, cell-local nonlinear providers and
/// their runtime consumers, so generated and direct-native routes cannot drift into
/// hand-synchronised status contracts.

#include <pops/core/foundation/types.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops {

/// Exact spatial location attached to a failed solve when one cell is authoritative.
///
/// `SolveReport` is shared by spatial and non-spatial solvers, so its ABI owns a fixed maximum
/// rank while authenticating the active rank explicitly. Unused coordinates are canonical zeroes;
/// negative valid-cell coordinates remain representable because presence never relies on a sentinel.
struct SolveFailureLocation {
  static constexpr int maximum_rank = 3;

  bool found = false;
  int rank = 0;
  std::array<int, maximum_rank> index{};
  int component = -1;

  template <int Dim, class RankedIndex>
  static SolveFailureLocation from(const RankedIndex& source, int failed_component) {
    static_assert(Dim >= 1 && Dim <= maximum_rank);
    SolveFailureLocation location;
    location.found = true;
    location.rank = Dim;
    for (int axis = 0; axis < Dim; ++axis)
      location.index[static_cast<std::size_t>(axis)] = source[axis];
    location.component = failed_component;
    return location;
  }

  [[nodiscard]] bool valid() const noexcept {
    if (!found)
      return rank == 0 && index == std::array<int, maximum_rank>{} && component == -1;
    if (rank < 1 || rank > maximum_rank)
      return false;
    for (int axis = rank; axis < maximum_rank; ++axis)
      if (index[static_cast<std::size_t>(axis)] != 0)
        return false;
    return component >= -1;
  }
};

/// Explicit solve status. Only kSolved publishes a candidate; every other status is a failed solve
/// report that callers must consume while leaving the live state unchanged.
enum class SolveStatus {
  kSolved,
  kSingular,
  kBreakdown,
  kIterationLimit,
  kInvalidEvaluation,
  kCapabilityFailure,
  kInvalidInput,
  kIncompatibleRhs,
  kInadmissibleCandidate,
  kSafeguardFailure,
};

/// Runtime reaction requested by a solve report.
enum class SolveAction {
  kNone,
  kFailRun,
  kRejectAttempt,
};

inline const char* solve_status_name(SolveStatus status) {
  switch (status) {
    case SolveStatus::kSolved:
      return "solved";
    case SolveStatus::kSingular:
      return "singular";
    case SolveStatus::kBreakdown:
      return "breakdown";
    case SolveStatus::kIterationLimit:
      return "iteration_limit";
    case SolveStatus::kInvalidEvaluation:
      return "invalid_evaluation";
    case SolveStatus::kCapabilityFailure:
      return "capability_failure";
    case SolveStatus::kInvalidInput:
      return "invalid_input";
    case SolveStatus::kIncompatibleRhs:
      return "incompatible_rhs";
    case SolveStatus::kInadmissibleCandidate:
      return "inadmissible_candidate";
    case SolveStatus::kSafeguardFailure:
      return "safeguard_failure";
  }
  return "invalid_input";
}

inline const char* solve_action_name(SolveAction action) {
  switch (action) {
    case SolveAction::kNone:
      return "none";
    case SolveAction::kFailRun:
      return "fail_run";
    case SolveAction::kRejectAttempt:
      return "reject_attempt";
  }
  return "reject_attempt";
}

/// Outcome of a solve: iterations performed, the reference and final residual norms, their declared
/// ratio, and one authoritative status/action/reason triple. Callers query `solved()`; no mutable
/// boolean can contradict the status.
struct SolveReport {
  int iters = 0;                     ///< number of iterations performed
  int evaluations = 0;               ///< residual/Jacobian evaluations performed
  int safeguard_steps = 0;           ///< rejected backtracking trial steps
  Real rel_residual = 0;             ///< residual_norm / declared reference denominator
  Real reference_residual_norm = 0;  ///< exact reference norm of the owning solver contract
  Real residual_norm = 0;            ///< exact final norm tested for convergence
  Real step_norm = 0;                ///< final scaled update norm
  Real condition_evidence = 0;       ///< largest/smallest accepted pivot magnitude
  SolveFailureLocation failure{};    ///< exact-ranked failing cell, when available
  SolveStatus status = SolveStatus::kIterationLimit;
  SolveAction action = SolveAction::kFailRun;
  std::string reason = "iteration_limit";

  bool valid() const {
    return !reason.empty() && failure.valid() &&
           (status == SolveStatus::kSolved) == (action == SolveAction::kNone) &&
           (status != SolveStatus::kSolved || !failure.found);
  }
  bool solved() const { return valid() && status == SolveStatus::kSolved; }
  bool solved_value_available() const { return solved(); }
  bool failed() const { return !solved(); }
  const char* status_name() const { return solve_status_name(status); }
  const char* action_name() const { return solve_action_name(action); }

  void mark_solved(std::string solve_reason = {}) {
    failure = {};
    status = SolveStatus::kSolved;
    action = SolveAction::kNone;
    reason = solve_reason.empty() ? solve_status_name(status) : std::move(solve_reason);
  }
  void mark_failed(SolveStatus failed_status, SolveAction failed_action = SolveAction::kFailRun,
                   std::string failure_reason = {}) {
    if (failed_status == SolveStatus::kSolved)
      throw std::invalid_argument("SolveReport::mark_failed requires a failure status");
    if (failed_action == SolveAction::kNone)
      throw std::invalid_argument("SolveReport::mark_failed requires an explicit failure action");
    // An invalid evaluation means that one or more scientific measures are unavailable.  Keep
    // every trustworthy finite, non-negative witness, but represent unavailable evidence with the
    // one canonical value accepted by the publication contract.  This preserves the exact failure
    // location/action/reason while preventing NaN or infinity from turning a structured numerical
    // failure into a malformed outcome at the next runtime boundary.  Other failure statuses are
    // deliberately not repaired: a provider that labels non-finite evidence as breakdown,
    // singularity, or iteration exhaustion must still fail structural publication.
    if (failed_status == SolveStatus::kInvalidEvaluation) {
      const auto canonical_evidence = [](Real value) noexcept {
        return std::isfinite(value) && value >= Real(0) ? value : Real(0);
      };
      rel_residual = canonical_evidence(rel_residual);
      reference_residual_norm = canonical_evidence(reference_residual_norm);
      residual_norm = canonical_evidence(residual_norm);
      step_norm = canonical_evidence(step_norm);
      condition_evidence = canonical_evidence(condition_evidence);
    }
    status = failed_status;
    action = failed_action;
    reason = failure_reason.empty() ? solve_status_name(status) : std::move(failure_reason);
  }

  static SolveReport capability_failure() {
    SolveReport report;
    report.mark_failed(SolveStatus::kCapabilityFailure);
    return report;
  }
};

/// Structural validation shared by every boundary that publishes a report produced by a prepared
/// provider. Scientific convergence remains provider-owned because only that provider owns the
/// exact operator and residual contract; the runtime nevertheless rejects malformed status values,
/// contradictory publication state, impossible iteration counts and non-finite norms.
inline bool solve_report_is_publishable(const SolveReport& report,
                                        int maximum_iterations) noexcept {
  const auto known_status = [](SolveStatus status) noexcept {
    switch (status) {
      case SolveStatus::kSolved:
      case SolveStatus::kSingular:
      case SolveStatus::kBreakdown:
      case SolveStatus::kIterationLimit:
      case SolveStatus::kInvalidEvaluation:
      case SolveStatus::kCapabilityFailure:
      case SolveStatus::kInvalidInput:
      case SolveStatus::kIncompatibleRhs:
      case SolveStatus::kInadmissibleCandidate:
      case SolveStatus::kSafeguardFailure:
        return true;
    }
    return false;
  };
  const auto known_action = [](SolveAction action) noexcept {
    switch (action) {
      case SolveAction::kNone:
      case SolveAction::kFailRun:
      case SolveAction::kRejectAttempt:
        return true;
    }
    return false;
  };
  const auto finite_nonnegative = [](Real value) noexcept {
    return std::isfinite(value) && value >= Real(0);
  };
  return maximum_iterations >= 0 && report.iters >= 0 && report.iters <= maximum_iterations &&
         report.evaluations >= 0 && report.safeguard_steps >= 0 && known_status(report.status) &&
         known_action(report.action) && report.valid() &&
         report.solved_value_available() == (report.status == SolveStatus::kSolved) &&
         finite_nonnegative(report.rel_residual) &&
         finite_nonnegative(report.reference_residual_norm) &&
         finite_nonnegative(report.residual_norm) && finite_nonnegative(report.step_norm) &&
         finite_nonnegative(report.condition_evidence);
}

}  // namespace pops
