#pragma once

/// @file
/// @brief SolveReport -- the authoritative result type of every iterative solve in
///        `include/pops/numerics/elliptic/linear`.
///
/// One definition shared by krylov_solver.hpp (the GeometricMG-coupled BiCGStab, TensorKrylovSolver)
/// and generic_krylov.hpp (the matrix-free richardson/cg/bicgstab loops), so the two never carry
/// hand-synchronised copies (a cross-TU ODR hazard if they ever drift).

#include <pops/core/foundation/types.hpp>

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

/// Outcome of a solve: iterations performed, final relative residual and one authoritative
/// status/action pair. Callers query `solved()`; no mutable boolean can contradict the status.
struct SolveReport {
  int iters = 0;          ///< number of iterations performed
  Real rel_residual = 0;  ///< ||r_final|| / ||b|| (global L2 norm; base 1 when ||b|| == 0)
  SolveStatus status = SolveStatus::kIterationLimit;
  SolveAction action = SolveAction::kFailRun;

  bool solved() const { return status == SolveStatus::kSolved; }
  bool solved_value_available() const { return solved(); }
  bool failed() const { return !solved(); }
  const char* status_name() const { return solve_status_name(status); }
  const char* action_name() const { return solve_action_name(action); }

  void mark_solved() {
    status = SolveStatus::kSolved;
    action = SolveAction::kNone;
  }
  void mark_failed(SolveStatus failed_status, SolveAction failed_action = SolveAction::kFailRun) {
    status = failed_status;
    action = failed_action;
  }

  static SolveReport capability_failure() {
    SolveReport report;
    report.mark_failed(SolveStatus::kCapabilityFailure);
    return report;
  }
};

}  // namespace pops
