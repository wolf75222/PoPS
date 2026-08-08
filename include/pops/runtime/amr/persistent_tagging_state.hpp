/// @file
/// @brief Compile-time-ranked persistent AMR tagging hysteresis.

#pragma once

#include <pops/mesh/index/box.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

/// Accepted, topology-independent state for AMR minimum-cycle hysteresis.
///
/// Keys remain in the exact parent-level index space, so patch splits, merges, and ownership
/// changes do not remap the state.  Rank is a type property and is encoded in the checkpoint image;
/// a checkpoint from another spatial specialization therefore fails closed before any record is
/// published. Copies share an accepted image and detach only when a tagging cycle mutates it.
template <int Dim>
class PersistentTaggingState {
  static_assert(Dim >= 1 && Dim <= 3,
                "PersistentTaggingState only supports dimensions 1, 2, and 3");

 public:
  enum class Decision : std::uint8_t { Refine = 1, Coarsen = 2 };

  struct CellKey {
    std::int32_t parent_level = 0;
    Index<Dim> cell{};

    friend bool operator<(const CellKey& left, const CellKey& right) noexcept {
      if (left.parent_level != right.parent_level)
        return left.parent_level < right.parent_level;
      // The highest axis is the outermost canonical coordinate. Axis zero remains contiguous.
      for (int axis = Dim - 1; axis >= 0; --axis)
        if (left.cell[axis] != right.cell[axis])
          return left.cell[axis] < right.cell[axis];
      return false;
    }
  };

  struct Entry {
    std::uint64_t decision_cycle = 0;
    Decision decision = Decision::Refine;
  };

  [[nodiscard]] std::uint64_t cycle() const noexcept { return storage_->cycle; }
  [[nodiscard]] std::size_t active_entry_count() const noexcept { return storage_->entries.size(); }

  void clear() {
    if (storage_.use_count() != 1) {
      storage_ = std::make_shared<Storage>();
      return;
    }
    storage_->cycle = 0;
    storage_->entries.clear();
  }

  /// Open one accepted tagging evaluation cycle and discard entries at the inclusive boundary.
  void begin_cycle(std::int32_t min_cycles) {
    require_min_cycles_(min_cycles);
    if (min_cycles == 0) {
      clear();
      return;
    }
    ensure_unique_();
    if (storage_->cycle == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR tagging hysteresis cycle overflow");
    ++storage_->cycle;
    for (auto entry = storage_->entries.begin(); entry != storage_->entries.end();) {
      if (storage_->cycle - entry->second.decision_cycle >= static_cast<std::uint64_t>(min_cycles))
        entry = storage_->entries.erase(entry);
      else
        ++entry;
    }
  }

  [[nodiscard]] bool transition_allowed(const CellKey& key, std::int32_t min_cycles) const {
    require_min_cycles_(min_cycles);
    if (min_cycles == 0)
      return true;
    const auto entry = storage_->entries.find(key);
    return entry == storage_->entries.end() ||
           storage_->cycle - entry->second.decision_cycle >= static_cast<std::uint64_t>(min_cycles);
  }

  void record(const CellKey& key, Decision decision, std::int32_t min_cycles) {
    require_min_cycles_(min_cycles);
    if (min_cycles == 0)
      return;
    if (key.parent_level < 0)
      throw std::invalid_argument("AMR tagging hysteresis parent level must be non-negative");
    if (decision != Decision::Refine && decision != Decision::Coarsen)
      throw std::invalid_argument("AMR tagging hysteresis decision is invalid");
    ensure_unique_();
    storage_->entries[key] = Entry{storage_->cycle, decision};
  }

  /// Canonical rank-independent checkpoint image using little-endian fixed-width integers.
  [[nodiscard]] std::vector<std::uint8_t> encode(std::int32_t min_cycles,
                                                 const std::string& provider_identity) const {
    require_min_cycles_(min_cycles);
    if (min_cycles == 0) {
      if (!storage_->entries.empty() || storage_->cycle != 0)
        throw std::logic_error("disabled AMR tagging hysteresis retained persistent state");
      return {};
    }
    if (provider_identity.empty())
      throw std::invalid_argument("AMR tagging hysteresis has no provider identity");

    std::vector<std::uint8_t> result;
    constexpr std::array<std::uint8_t, 8> magic{'P', 'O', 'P', 'S', 'H', 'Y', 'S', '2'};
    result.insert(result.end(), magic.begin(), magic.end());
    append_unsigned_(result, static_cast<std::uint32_t>(Dim));
    append_unsigned_(result, static_cast<std::uint32_t>(min_cycles));
    append_unsigned_(result, storage_->cycle);
    append_unsigned_(result, static_cast<std::uint64_t>(provider_identity.size()));
    result.insert(result.end(), provider_identity.begin(), provider_identity.end());
    append_unsigned_(result, static_cast<std::uint64_t>(storage_->entries.size()));
    for (const auto& [key, entry] : storage_->entries) {
      append_signed_(result, key.parent_level);
      for (int axis = 0; axis < Dim; ++axis)
        append_signed_(result, key.cell[axis]);
      append_unsigned_(result, entry.decision_cycle);
      result.push_back(static_cast<std::uint8_t>(entry.decision));
    }
    return result;
  }

  static PersistentTaggingState decode(const std::vector<std::uint8_t>& payload,
                                       std::int32_t min_cycles,
                                       const std::string& provider_identity,
                                       const std::vector<Box<Dim>>& parent_domains) {
    require_min_cycles_(min_cycles);
    if (min_cycles == 0) {
      if (!payload.empty())
        throw std::invalid_argument(
            "AMR checkpoint carries hysteresis state for a graph with min_cycles=0");
      return {};
    }
    if (provider_identity.empty() || parent_domains.empty())
      throw std::invalid_argument("AMR tagging hysteresis restore lacks bound graph topology");

    std::size_t cursor = 0;
    constexpr std::array<std::uint8_t, 8> magic{'P', 'O', 'P', 'S', 'H', 'Y', 'S', '2'};
    if (payload.size() < magic.size() || !std::equal(magic.begin(), magic.end(), payload.begin()))
      throw std::invalid_argument("AMR checkpoint has an unsupported tagging hysteresis schema");
    cursor += magic.size();
    const auto encoded_dimension = read_unsigned_<std::uint32_t>(payload, cursor);
    const auto encoded_min_cycles = read_unsigned_<std::uint32_t>(payload, cursor);
    const auto cycle = read_unsigned_<std::uint64_t>(payload, cursor);
    const auto identity_size = read_unsigned_<std::uint64_t>(payload, cursor);
    if (identity_size > payload.size() - cursor)
      throw std::invalid_argument("AMR checkpoint tagging provider identity is truncated");
    const std::string encoded_identity(reinterpret_cast<const char*>(payload.data() + cursor),
                                       static_cast<std::size_t>(identity_size));
    cursor += static_cast<std::size_t>(identity_size);
    if (encoded_dimension != static_cast<std::uint32_t>(Dim) ||
        encoded_min_cycles != static_cast<std::uint32_t>(min_cycles) ||
        encoded_identity != provider_identity)
      throw std::invalid_argument(
          "AMR checkpoint tagging hysteresis does not match the bound predicate graph");

    PersistentTaggingState result;
    result.storage_->cycle = cycle;
    const auto count = read_unsigned_<std::uint64_t>(payload, cursor);
    constexpr std::size_t record_size =
        sizeof(std::int32_t) * (1u + Dim) + sizeof(std::uint64_t) + sizeof(std::uint8_t);
    if (count > (payload.size() - cursor) / record_size)
      throw std::invalid_argument("AMR checkpoint tagging hysteresis record count is invalid");
    for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal) {
      CellKey key;
      key.parent_level = read_signed_(payload, cursor);
      for (int axis = 0; axis < Dim; ++axis)
        key.cell[axis] = read_signed_(payload, cursor);
      const auto decision_cycle = read_unsigned_<std::uint64_t>(payload, cursor);
      if (cursor >= payload.size())
        throw std::invalid_argument("AMR checkpoint tagging hysteresis decision is truncated");
      const auto raw_decision = payload[cursor++];
      if (key.parent_level < 0 ||
          static_cast<std::size_t>(key.parent_level) >= parent_domains.size() ||
          !parent_domains[static_cast<std::size_t>(key.parent_level)].contains(key.cell) ||
          decision_cycle > cycle ||
          cycle - decision_cycle >= static_cast<std::uint64_t>(min_cycles) ||
          (raw_decision != static_cast<std::uint8_t>(Decision::Refine) &&
           raw_decision != static_cast<std::uint8_t>(Decision::Coarsen)))
        throw std::invalid_argument("AMR checkpoint tagging hysteresis record is invalid");
      const auto [ignored, inserted] = result.storage_->entries.emplace(
          key, Entry{decision_cycle, static_cast<Decision>(raw_decision)});
      (void)ignored;
      if (!inserted)
        throw std::invalid_argument("AMR checkpoint tagging hysteresis has duplicate cell state");
    }
    if (cursor != payload.size())
      throw std::invalid_argument("AMR checkpoint tagging hysteresis has trailing bytes");
    return result;
  }

 private:
  static void require_min_cycles_(std::int32_t min_cycles) {
    if (min_cycles < 0)
      throw std::invalid_argument("AMR tagging hysteresis min_cycles must be non-negative");
  }

  template <class UInt>
  static void append_unsigned_(std::vector<std::uint8_t>& output, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
      output.push_back(static_cast<std::uint8_t>(value >> (byte * 8u)));
  }

  static void append_signed_(std::vector<std::uint8_t>& output, std::int32_t value) {
    append_unsigned_(output, static_cast<std::uint32_t>(value));
  }

  template <class UInt>
  static UInt read_unsigned_(const std::vector<std::uint8_t>& payload, std::size_t& cursor) {
    static_assert(std::is_unsigned_v<UInt>);
    if (cursor > payload.size() || payload.size() - cursor < sizeof(UInt))
      throw std::invalid_argument("AMR checkpoint tagging hysteresis payload is truncated");
    UInt value = 0;
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
      value |= static_cast<UInt>(payload[cursor++]) << (byte * 8u);
    return value;
  }

  static std::int32_t read_signed_(const std::vector<std::uint8_t>& payload, std::size_t& cursor) {
    return static_cast<std::int32_t>(read_unsigned_<std::uint32_t>(payload, cursor));
  }

  struct Storage {
    std::uint64_t cycle = 0;
    std::map<CellKey, Entry> entries;
  };

  void ensure_unique_() {
    if (storage_.use_count() != 1)
      storage_ = std::make_shared<Storage>(*storage_);
  }

  std::shared_ptr<Storage> storage_ = std::make_shared<Storage>();
};

}  // namespace pops::runtime::amr
