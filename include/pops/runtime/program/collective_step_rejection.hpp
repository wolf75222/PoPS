#pragma once

/// @file
/// @brief One typed rejection protocol for collective native execution phases.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::program {

/// Callers retain their wire schema and required diagnostic fields. These are fixed execution
/// contracts, not a relaxation of the status, disposition, reason, or exact-rank agreement checks.
struct StepRejectionContract {
  std::string_view schema;
  std::string_view agreement;
  bool require_phase;
  bool require_detail;
};

namespace step_rejection_detail {

struct Envelope {
  SolveStatus status;
  StepAttemptDisposition disposition;
  std::uint32_t reason_code;
  std::string phase;
  std::string detail;
};

inline void append_u64(std::string& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
}

inline std::uint64_t read_u64(std::string_view bytes, std::size_t& cursor) {
  if (cursor > bytes.size() || bytes.size() - cursor < 8)
    throw std::runtime_error("collective step rejection envelope is truncated");
  std::uint64_t value = 0;
  for (int byte = 0; byte < 8; ++byte)
    value = (value << 8u) | static_cast<unsigned char>(bytes[cursor++]);
  return value;
}

inline std::string encode(const StepAttemptRejected& rejected, StepRejectionContract contract) {
  std::string bytes(contract.schema);
  append_u64(bytes, static_cast<std::uint64_t>(rejected.status()));
  append_u64(bytes, static_cast<std::uint64_t>(rejected.disposition()));
  append_u64(bytes, rejected.reason_code());
  for (const std::string_view text : {rejected.phase(), rejected.detail()}) {
    append_u64(bytes, static_cast<std::uint64_t>(text.size()));
    bytes.append(text);
  }
  return bytes;
}

inline Envelope decode(std::string_view bytes, StepRejectionContract contract) {
  if (!bytes.starts_with(contract.schema))
    throw std::runtime_error("collective step rejection envelope has another schema");
  std::size_t cursor = contract.schema.size();
  const auto status = read_u64(bytes, cursor), disposition = read_u64(bytes, cursor);
  const auto reason = read_u64(bytes, cursor);
  if (status > static_cast<std::uint64_t>(SolveStatus::kSafeguardFailure) ||
      disposition > static_cast<std::uint64_t>(StepAttemptDisposition::kReject) ||
      reason > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("collective step rejection envelope has invalid typed fields");
  Envelope result{static_cast<SolveStatus>(status), static_cast<StepAttemptDisposition>(disposition),
                  static_cast<std::uint32_t>(reason), {}, {}};
  for (std::string* text : {&result.phase, &result.detail}) {
    const auto size = read_u64(bytes, cursor);
    if (size > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("collective step rejection text exceeds size_t");
    if (cursor > bytes.size() || size > bytes.size() - cursor)
      throw std::runtime_error("collective step rejection text is truncated");
    text->assign(bytes.substr(cursor, static_cast<std::size_t>(size)));
    cursor += static_cast<std::size_t>(size);
  }
  if (cursor != bytes.size() || (contract.require_phase && result.phase.empty()) ||
      (contract.require_detail && result.detail.empty()))
    throw std::runtime_error("collective step rejection envelope is incomplete");
  return result;
}

[[noreturn]] inline void rethrow(const CommunicatorView& communicator, std::string payload,
                                long rejected, StepRejectionContract contract) {
  std::string selected;
  if (all_reduce_min(rejected, communicator) != 0) {
    if (!all_ranks_agree_exact_ordered_byte_pairs({{contract.agreement, payload}}, communicator))
      throw std::runtime_error("collective step rejection fields differ between ranks");
    selected = std::move(payload);
  } else {
    const long root = all_reduce_min(rejected != 0 ? static_cast<long>(communicator.rank())
                                                   : static_cast<long>(communicator.size()),
                                      communicator);
    if (root < 0 || root >= static_cast<long>(communicator.size()))
      throw std::runtime_error("collective step rejection lost its typed envelope");
    const bool authoritative = communicator.rank() == root;
    const long invalid_length = authoritative && payload.size() >
        static_cast<std::size_t>(std::numeric_limits<long>::max()) ? 1L : 0L;
    if (all_reduce_max(invalid_length, communicator) != 0)
      throw std::length_error("collective step rejection envelope exceeds long capacity");
    const long length = all_reduce_max(authoritative ? static_cast<long>(payload.size()) : 0L,
                                       communicator);
    if (length <= 0)
      throw std::runtime_error("collective step rejection envelope is empty");
    long allocation_failed = 0;
    try {
      if (authoritative)
        selected = payload;
      selected.resize(static_cast<std::size_t>(length));
    } catch (...) { allocation_failed = 1; }
    if (all_reduce_max(allocation_failed, communicator) != 0)
      throw std::bad_alloc();
    broadcast_bytes_inplace(selected.data(), selected.size(), static_cast<int>(root), communicator);
    const long mismatch = rejected != 0 && payload != selected ? 1L : 0L;
    if (all_reduce_max(mismatch, communicator) != 0)
      throw std::runtime_error("collective step rejection fields differ between rejecting ranks");
  }
  // Decoding and exception construction allocate too. Converge those failures before any rank
  // leaves this phase with typed control that its enclosing transaction may handle differently.
  std::exception_ptr typed, error;
  try {
    auto envelope = decode(selected, contract);
    throw StepAttemptRejected(envelope.status, envelope.disposition, envelope.reason_code,
                              std::move(envelope.phase), std::move(envelope.detail));
  } catch (const StepAttemptRejected&) {
    typed = std::current_exception();
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, communicator, "collective rejection decoding failed");
  std::rethrow_exception(typed);
}

struct NoCompletion {
  void operator()() const noexcept {}
};

}  // namespace step_rejection_detail

/// Ordinary failures take precedence over retries from any rank. A completion operation always
/// runs, including after callback rejection; callers use it only when their fence must do so.
/// Success-only fences remain inside the operation supplied by the boundary and Tagger adapters.
template <class Operation, class Completion = step_rejection_detail::NoCompletion>
void collective_step_rejection_phase(const CommunicatorView& communicator,
                                      StepRejectionContract contract, std::string_view failure,
                                      Operation&& operation, Completion&& completion = {}) {
  std::exception_ptr error;
  std::string payload;
  long rejected = 0;
  try {
    std::forward<Operation>(operation)();
  } catch (const StepAttemptRejected& control) {
    try {
      payload = step_rejection_detail::encode(control, contract);
      rejected = 1;
    } catch (...) { error = std::current_exception(); }
  } catch (...) { error = std::current_exception(); }
  try {
    std::forward<Completion>(completion)();
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, communicator, failure);
  if (all_reduce_max(rejected, communicator) != 0)
    step_rejection_detail::rethrow(communicator, std::move(payload), rejected, contract);
}

}  // namespace pops::runtime::program
