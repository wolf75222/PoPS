#pragma once

#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <stdexcept>
#include <string>

namespace pops::runtime::program {

/// Typed control-flow signal emitted by a consumed SolveOutcome whose action is RejectAttempt.
/// Runtime step coordinators catch this exact type, restore the accepted snapshot and leave the
/// macro-step clock untouched.  FailRun remains an ordinary fatal exception.
class StepAttemptRejected final : public std::runtime_error {
 public:
  StepAttemptRejected(SolveStatus status, std::string phase, std::string detail = {})
      : std::runtime_error(message(status, phase, detail)), status_(status), phase_(std::move(phase)),
        detail_(std::move(detail)) {}

  SolveStatus status() const noexcept { return status_; }
  const std::string& phase() const noexcept { return phase_; }
  const std::string& detail() const noexcept { return detail_; }

 private:
  static std::string message(SolveStatus status, const std::string& phase,
                             const std::string& detail) {
    std::string out = "step attempt rejected during " + phase + ": solve status=" +
                      std::string(solve_status_name(status));
    if (!detail.empty())
      out += " (" + detail + ")";
    return out;
  }

  SolveStatus status_;
  std::string phase_;
  std::string detail_;
};

}  // namespace pops::runtime::program
