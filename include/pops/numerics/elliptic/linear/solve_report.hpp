#pragma once

/// @file
/// @brief SolveReport -- the authoritative result type of every iterative solve in
///        `include/pops/numerics/elliptic/linear`.
///
/// One definition shared by every prepared matrix-free Krylov method and its runtime consumers, so
/// generated and direct-native routes cannot drift into hand-synchronised status contracts.

#include <pops/core/foundation/types.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace pops {

/// Explicit status of a linear solve. Only kSolved publishes a solved value; every other status is a
/// failed solve report that callers must consume before using the mutated iterate.
enum class SolveStatus {
  kSolved,
  kSingular,
  kBreakdown,
  kIterationLimit,
  kInvalidEvaluation,
  kCapabilityFailure,
  kInvalidInput,
  kIncompatibleRhs,
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
  Real rel_residual = 0;             ///< residual_norm / declared reference denominator
  Real reference_residual_norm = 0;  ///< exact reference norm of the owning solver contract
  Real residual_norm = 0;            ///< exact final norm tested for convergence
  SolveStatus status = SolveStatus::kIterationLimit;
  SolveAction action = SolveAction::kFailRun;
  std::string reason = "iteration_limit";

  bool valid() const {
    return !reason.empty() && (status == SolveStatus::kSolved) == (action == SolveAction::kNone);
  }
  bool solved() const { return valid() && status == SolveStatus::kSolved; }
  bool solved_value_available() const { return solved(); }
  bool failed() const { return !solved(); }
  const char* status_name() const { return solve_status_name(status); }
  const char* action_name() const { return solve_action_name(action); }

  void mark_solved(std::string solve_reason = {}) {
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

}  // namespace pops
