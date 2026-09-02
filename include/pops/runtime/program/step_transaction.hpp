#pragma once

#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/export.hpp>
#include <pops/runtime/program/program_abi.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <exception>
#include <string_view>

namespace pops::runtime::program {

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
class POPS_RUNTIME_EXCEPTION_ABI StepAttemptRejected final : public std::exception {
 public:
  StepAttemptRejected(SolveStatus status, std::string_view phase, std::string_view detail = {})
      : status_(status) {
    assign_text_(phase_, phase);
    assign_text_(detail_, detail);
    compose_what_();
  }
  StepAttemptRejected(SolveStatus status, StepAttemptDisposition disposition,
                      std::uint32_t reason_code, std::string_view phase,
                      std::string_view detail = {})
      : status_(status), disposition_(disposition), reason_code_(reason_code) {
    assign_text_(phase_, phase);
    assign_text_(detail_, detail);
    compose_what_();
  }
  ~StepAttemptRejected() noexcept override;

  [[nodiscard]] const char* what() const noexcept override { return what_.data(); }
  SolveStatus status() const noexcept { return status_; }
  std::string_view phase() const noexcept { return phase_.data(); }
  std::string_view detail() const noexcept { return detail_.data(); }
  StepAttemptDisposition disposition() const noexcept { return disposition_; }
  std::uint32_t reason_code() const noexcept { return reason_code_; }

 private:
  static constexpr std::size_t kTextCapacity = kProgramStepRejectTextCapacity;
  static constexpr std::size_t kWhatCapacity = 3 * kTextCapacity;

  template <std::size_t Capacity>
  static void append_(std::array<char, Capacity>& destination, std::size_t& cursor,
                      std::string_view text) noexcept {
    const std::size_t available = cursor < Capacity ? Capacity - cursor - 1 : 0;
    const std::size_t count = std::min(available, text.size());
    for (std::size_t index = 0; index < count; ++index)
      destination[cursor + index] = text[index];
    cursor += count;
    destination[std::min(cursor, Capacity - 1)] = '\0';
  }

  static void assign_text_(std::array<char, kTextCapacity>& destination,
                           std::string_view text) noexcept {
    std::size_t cursor = 0;
    append_(destination, cursor, text);
  }

  void compose_what_() noexcept {
    std::size_t cursor = 0;
    append_(what_, cursor, "step attempt rejected during ");
    append_(what_, cursor, phase());
    append_(what_, cursor, ": solve status=");
    append_(what_, cursor, solve_status_name(status_));
    append_(what_, cursor, ", attempt_action=");
    append_(what_, cursor, step_attempt_disposition_name(disposition_));
    append_(what_, cursor, ", reason_code=");
    char reason[16]{};
    const auto [end, error] = std::to_chars(reason, reason + sizeof(reason) - 1, reason_code_);
    if (error == std::errc{})
      append_(what_, cursor, std::string_view(reason, static_cast<std::size_t>(end - reason)));
    if (!detail().empty()) {
      append_(what_, cursor, ", ");
      append_(what_, cursor, detail());
    }
  }

  SolveStatus status_;
  std::array<char, kTextCapacity> phase_{};
  std::array<char, kTextCapacity> detail_{};
  std::array<char, kWhatCapacity> what_{};
  StepAttemptDisposition disposition_ = StepAttemptDisposition::kReject;
  std::uint32_t reason_code_ = 0;
};

#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
inline StepAttemptRejected::~StepAttemptRejected() noexcept = default;
#endif

#undef POPS_RUNTIME_EXCEPTION_ABI

}  // namespace pops::runtime::program
