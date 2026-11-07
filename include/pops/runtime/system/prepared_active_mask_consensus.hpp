#pragma once

/// @file
/// @brief One-collective consensus of a generated-Program active-mask query.

#include <pops/parallel/consensus_words.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pops::runtime::system {

/// What one rank knows of an active-mask query before any collective. The mask is set only for a
/// valid query of an active embedded boundary; mode is -1 without a prepared embedded boundary.
template <class Field>
struct PreparedActiveMaskLocal {
  const Field* mask = nullptr;
  bool foreign_lane = false;
  bool invalid = false;
  int mode = -1;
  std::uint64_t generation = 0;
};

/// The slots of one active-mask query in a ConsensusWords: twelve words.
struct PreparedActiveMaskSlots {
  std::size_t foreign_lane = 0;
  std::size_t invalid = 0;
  std::size_t mode = 0;
  std::array<std::size_t, 4> generation{};
};

inline constexpr std::size_t kPreparedActiveMaskWords = 12;

template <std::size_t Capacity, class Field>
PreparedActiveMaskSlots add_active_mask_consensus(ConsensusWords<Capacity>& words,
                                                  const PreparedActiveMaskLocal<Field>& local) {
  PreparedActiveMaskSlots slots;
  slots.foreign_lane = words.failure(local.foreign_lane);
  slots.invalid = words.failure(local.invalid);
  slots.mode = words.agreement(local.mode);
  for (std::size_t word = 0; word < slots.generation.size(); ++word)
    slots.generation[word] =
        words.agreement(static_cast<int>((local.generation >> (16U * word)) & 0xffffU));
  return slots;
}

/// Throws the first failure in the order of the separate collectives this consensus replaces.
template <std::size_t Capacity>
void require_active_mask_consensus(const ConsensusWords<Capacity>& words,
                                   const PreparedActiveMaskSlots& slots) {
  if (words.failed(slots.foreign_lane))
    throw std::invalid_argument(
        "System pointwise active-mask provider requires its authenticated execution lane");
  if (words.failed(slots.invalid))
    throw std::invalid_argument("System pointwise active-mask preparation failed collectively");
  if (words.differs(slots.mode))
    throw std::runtime_error("System pointwise active-mask mode differs across ranks");
  for (const std::size_t slot : slots.generation)
    if (words.differs(slot))
      throw std::runtime_error("System pointwise active-mask generation differs across ranks");
}

}  // namespace pops::runtime::system
