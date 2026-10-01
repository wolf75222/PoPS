#pragma once

#include <pops/core/foundation/types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {

/// Rank-local accepted diagnostics archive @1. Names are opaque native strings;
/// values preserve every native Real bit, including signed zero and NaN payloads.
/// Rank ownership is embedded here and checked against the authenticated outer offsets.
inline std::vector<std::uint8_t> checkpoint_program_diagnostics(
    const std::map<std::string, Real>& values, int rank = 0, int ranks = 1) {
  if (ranks <= 0 || rank < 0 || rank >= ranks)
    throw std::invalid_argument("Program diagnostic archive has invalid rank authority");
  std::vector<std::uint8_t> bytes{'P', 'O', 'P', 'S', 'D', 'I', 'A', '1'};
  const auto word = [&](std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
      bytes.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
  };
  word(sizeof(RealBits) * 8);
  word(rank);
  word(ranks);
  word(values.size());
  for (const auto& [name, value] : values) {
    if (name.rfind("pops.balance-term", 0) == 0)
      throw std::invalid_argument("Program diagnostic archive contains reserved balance namespace");
    word(name.size());
    bytes.insert(bytes.end(), name.begin(), name.end());
    word(std::bit_cast<RealBits>(value));
  }
  return bytes;
}

inline std::map<std::string, Real> read_program_diagnostics_checkpoint(
    std::span<const std::uint8_t> bytes, int rank = 0, int ranks = 1) {
  if (ranks <= 0 || rank < 0 || rank >= ranks)
    throw std::invalid_argument("Program diagnostic archive has invalid rank authority");
  constexpr std::array<std::uint8_t, 8> magic{'P', 'O', 'P', 'S', 'D', 'I', 'A', '1'};
  if (bytes.size() < 40 || !std::equal(magic.begin(), magic.end(), bytes.begin()))
    throw std::invalid_argument("Program diagnostic archive has invalid @1 header");
  std::size_t cursor = 8;
  const auto word = [&]() {
    if (bytes.size() - cursor < 8)
      throw std::invalid_argument("Program diagnostic archive is truncated");
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
      value |= std::uint64_t{bytes[cursor++]} << (8 * byte);
    return value;
  };
  if (word() != sizeof(RealBits) * 8)
    throw std::invalid_argument("Program diagnostic archive native Real width changed");
  if (word() != static_cast<std::uint64_t>(rank) || word() != static_cast<std::uint64_t>(ranks))
    throw std::invalid_argument("Program diagnostic archive belongs to another rank authority");
  const auto count = word();
  if (count > (bytes.size() - cursor) / 16)
    throw std::invalid_argument("Program diagnostic archive entry count exceeds actual bytes");
  std::map<std::string, Real> result;
  std::string previous;
  bool first = true;
  for (std::uint64_t entry = 0; entry < count; ++entry) {
    const auto length = word();
    if (bytes.size() - cursor < 8 || length > bytes.size() - cursor - 8)
      throw std::invalid_argument("Program diagnostic archive name exceeds actual bytes");
    std::string name(reinterpret_cast<const char*>(bytes.data() + cursor),
                     static_cast<std::size_t>(length));
    cursor += static_cast<std::size_t>(length);
    const auto bits = word();
    if ((!first && !(previous < name)) || name.rfind("pops.balance-term", 0) == 0)
      throw std::invalid_argument(
          "Program diagnostic archive has duplicate, unordered or reserved name");
    if (bits > std::numeric_limits<RealBits>::max())
      throw std::invalid_argument("Program diagnostic archive bits exceed native Real width");
    result.emplace(name, std::bit_cast<Real>(static_cast<RealBits>(bits)));
    previous = std::move(name);
    first = false;
  }
  if (cursor != bytes.size())
    throw std::invalid_argument("Program diagnostic archive has trailing bytes");
  return result;
}

}  // namespace pops::runtime::program
