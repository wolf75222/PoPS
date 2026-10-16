#pragma once

#include <pops/runtime/program/step_transaction.hpp>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops::runtime::program {
// A scalar Result's exact signed words survive the ordinary collective rejection
// envelope. This is control metadata, not a numerical or model-specific opcode.
inline StepAttemptRejected hierarchy_control_failure(std::int64_t code, std::int64_t origin) {
  if (code == 0)
    throw std::invalid_argument("successful Result cannot be raised");
  std::string payload;
  payload.reserve(16);
  for (const auto word : {code, origin}) {
    const auto bits = std::bit_cast<std::uint64_t>(word);
    for (unsigned byte = 0; byte < 8; ++byte)
      payload.push_back(static_cast<char>((bits >> (8 * byte)) & 255));
  }
  return StepAttemptRejected(SolveStatus::kInvalidEvaluation, StepAttemptDisposition::kReject, 0,
                             "structured_control", "recoverable hierarchy Result",
                             "pops.hierarchy.scalar-result@1", std::move(payload));
}

inline std::pair<std::int64_t, std::int64_t> hierarchy_control_failure_words(
    const StepAttemptRejected& failure) {
  if (failure.typed_failure_schema().empty())
    return {3, 2};
  if (failure.typed_failure_schema() != "pops.hierarchy.scalar-result@1" ||
      failure.typed_failure_payload().size() != 16)
    throw failure;  // A scalar catch must never erase a complete typed payload.
  std::int64_t words[2]{};
  for (unsigned word = 0; word < 2; ++word) {
    std::uint64_t bits = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
      bits |= std::uint64_t(
                  static_cast<unsigned char>(failure.typed_failure_payload()[word * 8 + byte]))
              << (8 * byte);
    words[word] = std::bit_cast<std::int64_t>(bits);
  }
  if (words[0] == 0)
    throw std::invalid_argument("raised hierarchy Result has a zero code");
  return {words[0], words[1]};
}
}  // namespace pops::runtime::program
