#pragma once

#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/export.hpp>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pops::runtime::program {

/// All ranks validate the same scope transition before a snapshot or publication can mutate state.
inline void require_step_transaction_control(const ExecutionLane& lane, long operation, long depth,
                                             bool valid, const char* phase) {
  const long operation_min = all_reduce_min(operation, lane);
  const long operation_max = all_reduce_max(operation, lane);
  const long depth_min = all_reduce_min(depth, lane);
  const long depth_max = all_reduce_max(depth, lane);
  const long invalid = all_reduce_max(valid ? 0L : 1L, lane);
  if (invalid != 0 || operation_min != operation_max || depth_min != depth_max)
    throw std::runtime_error(std::string(phase) +
                             ": invalid or inconsistent collective transaction scope");
}

/// Attempt-level control requested by a recoverable native evaluation.  Retry asks an adaptive
/// controller to propose a smaller step; Reject leaves proposal policy to the caller.  Both retain
/// the same atomic rollback boundary and remain one Python-visible StepAttemptRejected type.
enum class StepAttemptDisposition : std::uint8_t { kRetry, kReject };

inline const char* step_attempt_disposition_name(StepAttemptDisposition disposition) noexcept {
  switch (disposition) {
    case StepAttemptDisposition::kRetry:
      return "retry";
    case StepAttemptDisposition::kReject:
      return "reject";
  }
  return "reject";
}

// StepAttemptRejected has two explicit, non-mixing compilation contracts:
//
// * ordinary pops::pops consumers leave POPS_RUNTIME_SHARED_EXCEPTION_ABI undefined and receive the
//   inline destructor below, preserving the header-only library contract;
// * the _pops host and every generated native loader define POPS_RUNTIME_SHARED_EXCEPTION_ABI. They
//   see the same declaration-only class body, while the host additionally defines
//   POPS_EXPORT_BUILDING_MODULE and provides the sole exported key function from pops_runtime_core.
//
// A final image must use one contract consistently. The in-class destructor declaration is stable in
// both modes; only the post-class header-only definition and the shared visibility annotation vary.
#if defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#define POPS_RUNTIME_EXCEPTION_ABI POPS_EXPORT
#else
#define POPS_RUNTIME_EXCEPTION_ABI
#endif

/// Typed control-flow signal emitted by a consumed SolveOutcome whose action is RejectAttempt.
/// Runtime step coordinators catch this exact type, restore the accepted snapshot and leave the
/// macro-step clock untouched.  FailRun remains an ordinary fatal exception.
class POPS_RUNTIME_EXCEPTION_ABI StepAttemptRejected final : public std::runtime_error {
 public:
  StepAttemptRejected(SolveStatus status, std::string phase, std::string detail = {})
      : std::runtime_error(message(status, phase, detail)),
        status_(status),
        phase_(std::move(phase)),
        detail_(std::move(detail)) {}
  StepAttemptRejected(SolveStatus status, StepAttemptDisposition disposition,
                      std::uint32_t reason_code, std::string phase, std::string detail = {})
      : std::runtime_error(message(status, disposition, reason_code, phase, detail)),
        status_(status),
        phase_(std::move(phase)),
        detail_(std::move(detail)),
        disposition_(disposition),
        reason_code_(reason_code) {}
  ~StepAttemptRejected() noexcept override;

  SolveStatus status() const noexcept { return status_; }
  const std::string& phase() const noexcept { return phase_; }
  const std::string& detail() const noexcept { return detail_; }
  StepAttemptDisposition disposition() const noexcept { return disposition_; }
  std::uint32_t reason_code() const noexcept { return reason_code_; }

 private:
  static std::string message(SolveStatus status, const std::string& phase,
                             const std::string& detail) {
    std::string out = "step attempt rejected during " + phase +
                      ": solve status=" + std::string(solve_status_name(status));
    if (!detail.empty())
      out += " (" + detail + ")";
    return out;
  }
  static std::string message(SolveStatus status, StepAttemptDisposition disposition,
                             std::uint32_t reason_code, const std::string& phase,
                             const std::string& detail) {
    std::string qualified = "attempt_action=";
    qualified += step_attempt_disposition_name(disposition);
    qualified += ", reason_code=" + std::to_string(reason_code);
    if (!detail.empty())
      qualified += ", " + detail;
    return message(status, phase, qualified);
  }

  SolveStatus status_;
  std::string phase_;
  std::string detail_;
  StepAttemptDisposition disposition_ = StepAttemptDisposition::kReject;
  std::uint32_t reason_code_ = 0;
};

#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
inline StepAttemptRejected::~StepAttemptRejected() noexcept = default;
#endif

#undef POPS_RUNTIME_EXCEPTION_ABI

/// Consume a previously reduced native evaluation status before any dependent computation.
/// The status categories are the existing EvaluationStatus order: ok, retry, reject, failed.
/// Explicit numerical evaluations retain their failure category without impersonating a solve.
inline void consume_native_evaluation_status(const ExecutionLane& lane, int program_block,
                                             int evaluation_id, double status,
                                             const char* operation_identity,
                                             std::uint32_t reason_code = 0) {
  const bool valid = std::isfinite(status) && status >= 0.0 && status <= 3.0 &&
                     status == std::floor(status) && program_block >= 0 && evaluation_id >= 0 &&
                     operation_identity != nullptr && operation_identity[0] != '\0';
  const long code = valid ? static_cast<long>(status) : 3L;
  const long minimum = all_reduce_min(code, lane);
  const long maximum = all_reduce_max(code, lane);
  const long invalid = all_reduce_max(valid ? 0L : 1L, lane);
  if (invalid != 0 || minimum != maximum)
    throw std::runtime_error("native evaluation status is invalid or differs between ranks");
  if (maximum == 0)
    return;
  const std::string detail = std::string(operation_identity) +
                             " block=" + std::to_string(program_block) +
                             " evaluation=" + std::to_string(evaluation_id) +
                             " reason_code=" + std::to_string(reason_code);
  if (maximum == 1 || maximum == 2)
    throw StepAttemptRejected(
        SolveStatus::kInvalidEvaluation,
        maximum == 1 ? StepAttemptDisposition::kRetry : StepAttemptDisposition::kReject,
        reason_code, "native_evaluation", detail);
  throw std::runtime_error("unrecoverable native evaluation: " + detail);
}

}  // namespace pops::runtime::program
