#pragma once

#include <pops/runtime/program/program_runtime_state.hpp>

namespace pops::runtime::program {

/// Opaque exact archive metadata. Names and levels prevent same-shaped foreign-ring rebinding;
/// the enclosing installed Program authenticates state, clock and interpolation descriptors.
inline std::vector<std::uint8_t> encode_history_sample_identity(
    std::string_view name, int level, std::span<const HistorySampleIdentity> samples) {
  if (samples.empty())
    throw std::invalid_argument("history sample archive requires a positive ring depth");
  constexpr std::size_t fixed = 32;
  if (name.size() > std::numeric_limits<std::size_t>::max() - fixed ||
      samples.size() > (std::numeric_limits<std::size_t>::max() - fixed - name.size()) / 32)
    throw std::overflow_error("history sample archive byte count overflow");
  std::vector<std::uint8_t> bytes{'P', 'O', 'P', 'S', 'H', 'I', 'D', '1'};
  bytes.reserve(fixed + name.size() + samples.size() * 32);
  const auto put = [&](std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  put(name.size());
  bytes.insert(bytes.end(), name.begin(), name.end());
  put(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(level)));
  put(samples.size());
  for (const auto& sample : samples) {
    sample.validate();
    put(static_cast<std::uint64_t>(sample.kind));
    put(sample.start_bits);
    put(sample.interval_bits);
    put(sample.ordinal);
  }
  return bytes;
}

inline std::vector<HistorySampleIdentity> decode_history_sample_identity(
    std::span<const std::uint8_t> bytes, std::string_view name, int level, std::size_t depth) {
  if (depth == 0)
    throw std::invalid_argument("history sample archive requires a positive ring depth");
  // Old archives did not record publication identities. Never infer them from dt or field values.
  if (bytes.empty())
    return std::vector<HistorySampleIdentity>(depth);
  constexpr std::array<std::uint8_t, 8> magic{'P', 'O', 'P', 'S', 'H', 'I', 'D', '1'};
  if (bytes.size() < magic.size() || !std::equal(magic.begin(), magic.end(), bytes.begin()))
    throw std::invalid_argument("history sample archive has an invalid version");
  std::size_t cursor = magic.size();
  const auto get = [&]() {
    if (bytes.size() - cursor < 8)
      throw std::invalid_argument("history sample archive is truncated");
    std::uint64_t result = 0;
    for (int shift = 0; shift < 64; shift += 8)
      result |= static_cast<std::uint64_t>(bytes[cursor++]) << shift;
    return result;
  };
  if (get() != name.size() || name.size() > bytes.size() - cursor ||
      !std::equal(name.begin(), name.end(), bytes.begin() + cursor))
    throw std::invalid_argument("history sample archive belongs to another history");
  cursor += name.size();
  if (std::bit_cast<std::int64_t>(get()) != level || get() != depth ||
      depth > (bytes.size() - cursor) / 32 || bytes.size() - cursor != depth * 32)
    throw std::invalid_argument("history sample archive level, depth or exact size differs");
  std::vector<HistorySampleIdentity> result(depth);
  for (auto& sample : result) {
    const auto kind = get();
    if (kind > static_cast<std::uint64_t>(HistorySampleKind::Publication))
      throw std::invalid_argument("history sample archive has an invalid kind");
    sample.kind = static_cast<HistorySampleKind>(kind);
    sample.start_bits = get();
    sample.interval_bits = get();
    sample.ordinal = get();
    sample.validate();
  }
  return result;
}

}  // namespace pops::runtime::program
