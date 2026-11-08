#pragma once

/// @file
/// @brief Decide the failure votes and agreement checks of one call site in one collective.

#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace pops {

/// Several failure votes and cross-rank agreement checks of one call site, decided by a single
/// maximum reduction instead of one collective each (spec 0.3 SV20, duties 2 and 3; PF8).
///
/// A failure slot holds the local flag. An agreement slot holds v and -v, whose lane maxima are
/// the largest v and the negated smallest v. A maximum slot holds a local value whose lane maximum
/// the caller reads back. Every rank reads the same reduced words, so a caller that tests its slots
/// in the order of the separate collectives they replace throws the same first failure on every
/// rank. Words are doubles: flags, int values and 16-bit chunks are exact, and a maximum keeps the
/// double reduction of the collective it replaces. A single-rank lane decides locally.
template <std::size_t Capacity>
class ConsensusWords {
 public:
  std::size_t failure(bool local_failure) { return push_(local_failure ? 1.0 : 0.0); }

  std::size_t agreement(int value) {
    const std::size_t slot = push_(static_cast<double>(value));
    push_(-static_cast<double>(value));
    return slot;
  }

  std::size_t maximum(double local) { return push_(local); }

  void reduce(const ExecutionLane& lane) {
    if (reduced_)
      throw std::logic_error("consensus words are reduced once");
    if (lane.size() > 1)
      all_reduce_max_inplace(words_.data(), static_cast<int>(size_), lane);
    reduced_ = true;
  }

  [[nodiscard]] bool failed(std::size_t slot) const { return word_(slot) != 0.0; }
  [[nodiscard]] bool differs(std::size_t slot) const { return word_(slot) != -word_(slot + 1); }
  [[nodiscard]] double value(std::size_t slot) const { return word_(slot); }

 private:
  std::size_t push_(double word) {
    if (reduced_ || size_ == Capacity)
      throw std::logic_error("consensus words exceed their capacity or were already reduced");
    words_[size_] = word;
    return size_++;
  }

  double word_(std::size_t slot) const {
    if (!reduced_ || slot >= size_)
      throw std::logic_error("consensus word read before its reduction or out of range");
    return words_[slot];
  }

  std::array<double, Capacity> words_{};
  std::size_t size_ = 0;
  bool reduced_ = false;
};

}  // namespace pops
