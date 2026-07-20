/// @file
/// @brief Exact native MPI preflight for analytic-expression requests.

#pragma once

#include <pops/parallel/comm.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::analytic {

/// Named request metadata carried exactly as bytes. Names are sorted by the preflight, so callers do
/// not have to coordinate insertion order across independent frontends.
using AnalyticTextMetadata = std::pair<std::string_view, std::string_view>;

/// Named binary64 request metadata. The IEEE-754 bit pattern is part of the canonical request: in
/// particular +0 and -0 remain distinguishable and no decimal round-trip or probabilistic hash is
/// involved. Scientific range/finite checks remain the responsibility of the local validator.
using AnalyticRealMetadata = std::pair<std::string_view, double>;

namespace detail {

inline void append_analytic_u64(std::string& payload, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    payload.push_back(static_cast<char>((value >> shift) & UINT64_C(0xff)));
}

inline void append_analytic_size(std::string& payload, std::size_t value) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
      throw std::length_error("analytic request exceeds canonical uint64 length capacity");
  }
  append_analytic_u64(payload, static_cast<std::uint64_t>(value));
}

inline void append_analytic_bytes(std::string& payload, std::string_view value) {
  append_analytic_size(payload, value.size());
  if (!value.empty())
    payload.append(value.data(), value.size());
}

struct CanonicalAnalyticMetadata {
  std::string_view name;
  std::string_view text;
  double real = 0.0;
  bool is_real = false;
};

inline std::string canonical_analytic_request(
    std::string_view operation, std::span<const AnalyticTextMetadata> text_metadata,
    std::span<const AnalyticRealMetadata> real_metadata, const AnalyticOpcodeRows& opcodes,
    const AnalyticLiteralRows& literals) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559,
                "analytic request consensus requires IEEE-754 binary64");
  if (operation.empty())
    throw std::invalid_argument("analytic collective operation identity must be non-empty");

  std::vector<CanonicalAnalyticMetadata> metadata;
  metadata.reserve(text_metadata.size() + real_metadata.size());
  for (const auto& [name, value] : text_metadata)
    metadata.push_back(CanonicalAnalyticMetadata{name, value, 0.0, false});
  for (const auto& [name, value] : real_metadata)
    metadata.push_back(CanonicalAnalyticMetadata{name, {}, value, true});
  std::sort(metadata.begin(), metadata.end(), [](const auto& left, const auto& right) {
    return left.name < right.name;
  });
  for (std::size_t index = 0; index < metadata.size(); ++index) {
    if (metadata[index].name.empty())
      throw std::invalid_argument("analytic collective metadata name must be non-empty");
    if (index != 0 && metadata[index - 1].name == metadata[index].name)
      throw std::invalid_argument("analytic collective metadata names must be unique");
  }

  std::string payload;
  append_analytic_bytes(payload, "pops.analytic.request.v1");
  append_analytic_bytes(payload, operation);
  append_analytic_size(payload, metadata.size());
  for (const CanonicalAnalyticMetadata& field : metadata) {
    append_analytic_bytes(payload, field.name);
    payload.push_back(field.is_real ? char{1} : char{0});
    if (field.is_real)
      append_analytic_u64(payload, std::bit_cast<std::uint64_t>(field.real));
    else
      append_analytic_bytes(payload, field.text);
  }

  append_analytic_size(payload, opcodes.size());
  for (const auto& row : opcodes) {
    append_analytic_size(payload, row.size());
    for (const std::string& opcode : row)
      append_analytic_bytes(payload, opcode);
  }
  append_analytic_size(payload, literals.size());
  for (const auto& row : literals) {
    append_analytic_size(payload, row.size());
    for (double literal : row)
      append_analytic_u64(payload, std::bit_cast<std::uint64_t>(literal));
  }
  return payload;
}

}  // namespace detail

/// Run a non-mutating local validator/preparer on every rank, convert any rank-local exception into
/// one collective failure, then require exact equality of the complete canonical request. The
/// returned object may own prepared native programs or staged registry nodes; callers publish it
/// only after this function returns.
///
/// This is a control-plane collective. Every rank in @p communicator must call it in the same order.
/// With one rank the original validation exception is rethrown, preserving the serial API contract.
template <class LocalPrepare>
[[nodiscard]] auto collectively_prepare_analytic_request(
    std::string_view operation, std::span<const AnalyticTextMetadata> text_metadata,
    std::span<const AnalyticRealMetadata> real_metadata, const AnalyticOpcodeRows& opcodes,
    const AnalyticLiteralRows& literals, LocalPrepare&& local_prepare,
    const CommunicatorView& communicator = world_communicator_view())
    -> std::invoke_result_t<LocalPrepare> {
  using Result = std::invoke_result_t<LocalPrepare>;
  static_assert(!std::is_void_v<Result>);

  std::optional<Result> prepared;
  std::string canonical_payload;
  std::exception_ptr local_failure;
  try {
    prepared.emplace(std::invoke(std::forward<LocalPrepare>(local_prepare)));
    canonical_payload = detail::canonical_analytic_request(operation, text_metadata, real_metadata,
                                                            opcodes, literals);
  } catch (...) {
    local_failure = std::current_exception();
  }

  const long failure_count =
      all_reduce_sum(local_failure ? 1L : 0L, communicator);
  if (failure_count != 0) {
    if (communicator.size() == 1 && local_failure)
      std::rethrow_exception(local_failure);
    throw std::runtime_error(std::string(operation) +
                             ": rank-local analytic validation failed collectively on " +
                             std::to_string(failure_count) + " rank(s)");
  }

  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"pops.analytic.request.v1", canonical_payload}}, communicator))
    throw std::runtime_error(std::string(operation) +
                             ": analytic request differs across MPI ranks");
  return std::move(*prepared);
}

template <class LocalPrepare>
[[nodiscard]] auto collectively_prepare_analytic_request(
    std::string_view operation, std::initializer_list<AnalyticTextMetadata> text_metadata,
    std::initializer_list<AnalyticRealMetadata> real_metadata, const AnalyticOpcodeRows& opcodes,
    const AnalyticLiteralRows& literals, LocalPrepare&& local_prepare,
    const CommunicatorView& communicator = world_communicator_view())
    -> std::invoke_result_t<LocalPrepare> {
  return collectively_prepare_analytic_request(
      operation, std::span<const AnalyticTextMetadata>(text_metadata.begin(), text_metadata.size()),
      std::span<const AnalyticRealMetadata>(real_metadata.begin(), real_metadata.size()), opcodes,
      literals, std::forward<LocalPrepare>(local_prepare), communicator);
}

}  // namespace pops::analytic
