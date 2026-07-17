#pragma once

#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/export.hpp>

#include <stdexcept>
#include <string>

namespace pops::runtime::program {

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
  ~StepAttemptRejected() noexcept override;

  SolveStatus status() const noexcept { return status_; }
  const std::string& phase() const noexcept { return phase_; }
  const std::string& detail() const noexcept { return detail_; }

 private:
  static std::string message(SolveStatus status, const std::string& phase,
                             const std::string& detail) {
    std::string out = "step attempt rejected during " + phase +
                      ": solve status=" + std::string(solve_status_name(status));
    if (!detail.empty())
      out += " (" + detail + ")";
    return out;
  }

  SolveStatus status_;
  std::string phase_;
  std::string detail_;
};

#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
inline StepAttemptRejected::~StepAttemptRejected() noexcept = default;
#endif

#undef POPS_RUNTIME_EXCEPTION_ABI

}  // namespace pops::runtime::program
