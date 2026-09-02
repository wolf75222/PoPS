#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::program {

enum class ScheduleDomainKind { kAcceptedStep, kStage, kClockTick, kAmrLevel };

struct ScheduleCoordinate {
  std::int64_t value = 0;
};

struct ExactCoefficientTerm {
  int dt_power = 0;
  std::int64_t numerator = 0;
  std::int64_t denominator = 1;
};

/// Exception-safe runtime validation for generated nested logical-clock schedules.
class ClockScheduleState {
 public:
  struct Frame {
    std::uint32_t parent = 0;
    std::uint32_t child = 0;
    int count = 0;
    int next = 0;
  };

  class SubcycleScope {
   public:
    SubcycleScope(ClockScheduleState& owner, std::string_view parent, std::string_view child,
                  int count)
        : owner_(&owner), depth_(owner.begin_(parent, child, count)) {}
    SubcycleScope(const SubcycleScope&) = delete;
    SubcycleScope& operator=(const SubcycleScope&) = delete;
    SubcycleScope(SubcycleScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          depth_(other.depth_),
          finished_(other.finished_) {}
    ~SubcycleScope() {
      if (owner_ != nullptr && !finished_)
        owner_->abort_(depth_);
    }

    void iteration(int index) const {
      require_live_();
      owner_->iteration_(depth_, index);
    }
    void finish() {
      require_live_();
      owner_->finish_(depth_);
      finished_ = true;
    }

   private:
    void require_live_() const {
      if (owner_ == nullptr || finished_)
        throw std::runtime_error("logical-clock subcycle scope is no longer active");
    }
    ClockScheduleState* owner_ = nullptr;
    std::size_t depth_ = 0;
    bool finished_ = false;
  };

  SubcycleScope subcycle(std::string_view parent, std::string_view child, int count) {
    return SubcycleScope(*this, parent, child, count);
  }

  void configure_primary_clock(std::string clock) {
    if (sealed_)
      throw std::logic_error("logical-clock schedule changed after bind seal");
    if (clock.empty())
      throw std::runtime_error("logical-clock primary identity must be non-empty");
    if (!primary_.empty() && primary_ != clock)
      throw std::runtime_error("logical-clock primary identity changed after installation");
    primary_ = std::move(clock);
    ticks_per_macro_cache_.clear();
    ticks_per_macro_cache_.emplace(primary_, 1);
    for (const auto& [child, relation] : relations_) {
      (void)relation;
      ticks_per_macro_cache_.emplace(child, ticks_per_macro_(child, {}));
    }
  }

  void declare_relation(std::string parent, std::string child, int count) {
    if (sealed_)
      throw std::logic_error("logical-clock relation changed after bind seal");
    if (parent.empty() || child.empty() || parent == child || count <= 0)
      throw std::runtime_error("invalid logical-clock subcycle descriptor");
    const auto found = relations_.find(child);
    const Relation relation{std::move(parent), count};
    if (found != relations_.end() &&
        (found->second.parent != relation.parent || found->second.count != relation.count))
      throw std::runtime_error("logical child clock has conflicting parent/count declarations");
    relations_[child] = relation;
    // This recursive validation is cold configuration work.  Accepted-step refreshes read only
    // the sealed scalar cache below and therefore never construct a `std::set`.
    const std::int64_t ticks = ticks_per_macro_(child, {});
    ticks_per_macro_cache_[child] = ticks;
  }

  /// Freeze the finite clock graph before candidate execution.  Frame identities become compact
  /// ids and the stack capacity is exact: nested generated subcycles may not copy long clock
  /// strings or grow the vector after this boundary.
  void seal_for_execution() {
    if (sealed_)
      return;
    if (primary_.empty())
      throw std::logic_error("logical-clock schedule cannot seal without a primary clock");
    clock_ids_.clear();
    clock_ids_.reserve(relations_.size() + 1);
    clock_ids_.push_back(primary_);
    for (const auto& [child, relation] : relations_) {
      (void)relation;
      if (child == primary_)
        throw std::logic_error("logical-clock primary identity cannot be a child relation");
      clock_ids_.push_back(child);
    }
    parent_ids_.assign(relations_.size() + 1, kNoClock);
    relation_counts_.assign(relations_.size() + 1, 0);
    for (const auto& [child, relation] : relations_) {
      const auto child_id = clock_index_unsealed_(child);
      const auto parent = clock_index_unsealed_(relation.parent);
      if (!child_id || !parent)
        throw std::logic_error("logical-clock relation parent is absent from the bind graph");
      parent_ids_[*child_id] = *parent;
      relation_counts_[*child_id] = relation.count;
    }
    // The accepted checkpoint wire is a map, whose node order is fixed at bind.  Retain the
    // corresponding tick products in that exact order so accepted-step serialization never
    // performs a string/map lookup merely to refresh scalar values.
    accepted_tick_wire_values_.clear();
    accepted_tick_wire_values_.reserve(ticks_per_macro_cache_.size());
    for (const auto& [clock, ticks] : ticks_per_macro_cache_) {
      (void)clock;
      if (ticks <= 0)
        throw std::logic_error("logical-clock tick cache is invalid at bind seal");
      accepted_tick_wire_values_.push_back(ticks);
    }
    frames_.clear();
    frames_.reserve(relations_.size());
    sealed_ = true;
  }

  [[nodiscard]] bool sealed_for_execution() const noexcept { return sealed_; }

  std::optional<ScheduleCoordinate> coordinate(ScheduleDomainKind kind, const std::string& clock,
                                               const std::string& stage_identity,
                                               int required_level, int current_level,
                                               std::int64_t macro_step) const {
    const std::optional<std::int64_t> tick = active_tick_(clock, macro_step);
    if (!tick)
      return std::nullopt;
    switch (kind) {
      case ScheduleDomainKind::kAcceptedStep:
        if (!stage_identity.empty() || required_level != -1)
          throw std::runtime_error("accepted-step schedule carries foreign domain payload");
        return ScheduleCoordinate{macro_step};
      case ScheduleDomainKind::kStage:
        if (stage_identity.empty() || required_level != -1)
          throw std::runtime_error("stage schedule lacks its exact stage identity");
        // The generated site was statically proven equal to this exact StagePoint.  Runtime still
        // authenticates the active qualified clock before exposing the accepted-step coordinate.
        return ScheduleCoordinate{macro_step};
      case ScheduleDomainKind::kClockTick:
        if (!stage_identity.empty() || required_level != -1)
          throw std::runtime_error("clock-tick schedule carries foreign domain payload");
        return ScheduleCoordinate{*tick};
      case ScheduleDomainKind::kAmrLevel:
        if (!stage_identity.empty() || required_level < 0)
          throw std::runtime_error("AMR-level schedule lacks its exact level");
        if (current_level != required_level)
          return std::nullopt;
        return ScheduleCoordinate{*tick};
    }
    throw std::runtime_error("unknown native schedule domain");
  }

  std::map<std::string, std::int64_t> accepted_ticks(std::int64_t macro_step) const {
    if (primary_.empty())
      throw std::runtime_error("logical-clock schedule has no primary clock");
    std::map<std::string, std::int64_t> result;
    result.emplace(primary_, checked_multiply_(macro_step, cached_ticks_per_macro_(primary_)));
    for (const auto& [child, relation] : relations_) {
      (void)relation;
      result.emplace(child, checked_multiply_(macro_step, cached_ticks_per_macro_(child)));
    }
    return result;
  }

  /// Fill a cold-primed accepted-tick image without inserting map nodes or allocating strings.
  /// The schedule graph is immutable after Program bind, so a missing or foreign key is an
  /// authority change rather than a reason to rebuild checkpoint storage in an accepted step.
  void accepted_ticks_into(std::map<std::string, std::int64_t>& destination,
                           std::int64_t macro_step) const {
    if (primary_.empty())
      throw std::runtime_error("logical-clock schedule has no primary clock");
    if (destination.size() != relations_.size() + 1)
      throw std::logic_error("logical-clock accepted-tick image was not primed");
    const auto primary = destination.find(primary_);
    if (primary == destination.end())
      throw std::logic_error("logical-clock accepted-tick image lost its primary key");
    primary->second = checked_multiply_(macro_step, cached_ticks_per_macro_(primary_));
    for (const auto& [child, relation] : relations_) {
      (void)relation;
      const auto tick = destination.find(child);
      if (tick == destination.end())
        throw std::logic_error("logical-clock accepted-tick image changed its key set");
      tick->second = checked_multiply_(macro_step, cached_ticks_per_macro_(child));
    }
  }

  /// Fill pre-existing checkpoint-map nodes in their canonical map order.  This is the accepted
  /// hot path companion of `accepted_ticks_into`: identities were authenticated at bind, while
  /// this operation touches only compact scalar slots and never calls `map::find`.
  void accepted_ticks_in_wire_order_into(std::map<std::string, std::int64_t>& destination,
                                         std::int64_t macro_step) const {
    if (!sealed_)
      throw std::logic_error("logical-clock ordinal accepted-tick schedule was not sealed");
    if (destination.size() != accepted_tick_wire_values_.size())
      throw std::logic_error("logical-clock ordinal accepted-tick image was not primed");
    auto target = destination.begin();
    for (const std::int64_t ticks : accepted_tick_wire_values_) {
      if (target == destination.end() || ticks <= 0)
        throw std::logic_error("logical-clock ordinal accepted-tick image changed after bind");
      target->second = checked_multiply_(macro_step, ticks);
      ++target;
    }
    if (target != destination.end())
      throw std::logic_error("logical-clock ordinal accepted-tick image has a foreign node");
  }

  void restore_accepted_ticks(const std::map<std::string, std::int64_t>& ticks,
                              std::int64_t macro_step) {
    if (ticks != accepted_ticks(macro_step))
      throw std::runtime_error(
          "restored logical-clock ticks differ from the installed clock relations");
    accepted_ticks_ = ticks;
  }

  const std::map<std::string, std::int64_t>& restored_accepted_ticks() const {
    return accepted_ticks_;
  }

  std::int64_t ticks_per_macro(const std::string& clock) const {
    return ticks_per_macro_(clock, {});
  }

  /// Logical dynamic storage retained by this schedule.  The Program resource-plan contract
  /// counts the exact requested vector/map element and external-string capacities, but excludes
  /// object-inline representation and allocator bookkeeping.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    std::uint64_t total = 0;
    resident_checked_add_(total, external_string_storage_bytes_(primary_));
    resident_checked_add_(
        total, resident_allocation_bytes_(relations_.size(),
                                          sizeof(std::pair<const std::string, Relation>)));
    for (const auto& [child, relation] : relations_) {
      resident_checked_add_(total, external_string_storage_bytes_(child));
      resident_checked_add_(total, external_string_storage_bytes_(relation.parent));
    }
    resident_checked_add_(
        total, resident_allocation_bytes_(ticks_per_macro_cache_.size(),
                                          sizeof(std::pair<const std::string, std::int64_t>)));
    for (const auto& [clock, tick] : ticks_per_macro_cache_) {
      (void)tick;
      resident_checked_add_(total, external_string_storage_bytes_(clock));
    }
    resident_checked_add_(
        total, resident_allocation_bytes_(accepted_ticks_.size(),
                                          sizeof(std::pair<const std::string, std::int64_t>)));
    for (const auto& [clock, tick] : accepted_ticks_) {
      (void)tick;
      resident_checked_add_(total, external_string_storage_bytes_(clock));
    }
    resident_checked_add_(total, resident_allocation_bytes_(frames_.capacity(), sizeof(Frame)));
    resident_checked_add_(total, resident_allocation_bytes_(clock_ids_.capacity(), sizeof(std::string)));
    for (const std::string& clock : clock_ids_)
      resident_checked_add_(total, external_string_storage_bytes_(clock));
    resident_checked_add_(total, resident_allocation_bytes_(parent_ids_.capacity(), sizeof(std::uint32_t)));
    resident_checked_add_(total,
                          resident_allocation_bytes_(relation_counts_.capacity(), sizeof(int)));
    resident_checked_add_(total, resident_allocation_bytes_(accepted_tick_wire_values_.capacity(),
                                                            sizeof(std::int64_t)));
    return total;
  }

  [[nodiscard]] const std::string& primary_clock() const noexcept { return primary_; }

  /// Refresh a persistent transaction image without rebuilding its map/vector storage.  Logical
  /// clock declarations are immutable after Program installation, but accepted ticks and active
  /// subcycle frames are attempt state and must roll back exactly with the numerical hierarchy.
  void copy_into(ClockScheduleState& destination) const {
    destination.primary_ = primary_;
    destination.sealed_ = sealed_;
    destination.clock_ids_ = clock_ids_;
    destination.parent_ids_ = parent_ids_;
    destination.relation_counts_ = relation_counts_;
    destination.accepted_tick_wire_values_ = accepted_tick_wire_values_;
    copy_map_into_(destination.relations_, relations_);
    copy_map_into_(destination.ticks_per_macro_cache_, ticks_per_macro_cache_);
    copy_map_into_(destination.accepted_ticks_, accepted_ticks_);
    destination.frames_.resize(frames_.size());
    std::copy(frames_.begin(), frames_.end(), destination.frames_.begin());
  }

  /// Copy one resident accepted image without changing any container capacity.  The schedule graph
  /// is sealed at Program bind, so a different key/frame shape is an authority change rather than
  /// a reason to allocate while a transaction holds its accepted writer.
  void copy_into_preallocated(ClockScheduleState& destination) const {
    if (destination.primary_ != primary_)
      throw std::logic_error("logical-clock primary identity changed after transaction prime");
    if (destination.sealed_ != sealed_ || destination.clock_ids_ != clock_ids_ ||
        destination.parent_ids_ != parent_ids_ || destination.relation_counts_ != relation_counts_ ||
        destination.accepted_tick_wire_values_ != accepted_tick_wire_values_)
      throw std::logic_error("logical-clock bind-sealed frame graph changed after transaction prime");
    copy_relations_preallocated_(destination.relations_, relations_);
    copy_ticks_preallocated_(destination.ticks_per_macro_cache_, ticks_per_macro_cache_);
    copy_ticks_preallocated_(destination.accepted_ticks_, accepted_ticks_);
    if (destination.frames_.size() != frames_.size())
      throw std::logic_error("logical-clock frame shape changed after transaction prime");
    for (std::size_t index = 0; index < frames_.size(); ++index) {
      if (destination.frames_[index].parent != frames_[index].parent ||
          destination.frames_[index].child != frames_[index].child ||
          destination.frames_[index].count != frames_[index].count)
        throw std::logic_error("logical-clock frame identity changed after transaction prime");
      destination.frames_[index].next = frames_[index].next;
    }
  }

  void synchronize_sample_and_hold(const std::string& source, const std::string& target,
                                   int /*step*/, double offset) const {
    if (source.empty() || target.empty() || source == target)
      throw std::runtime_error(
          "sample-and-hold synchronization requires distinct qualified clocks");
    if (!std::isfinite(offset))
      throw std::runtime_error("sample-and-hold synchronization offset must be finite");
    (void)ticks_per_macro_(source, {});
    (void)ticks_per_macro_(target, {});
  }

 private:
  static void resident_checked_add_(std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("logical-clock resident storage overflows uint64");
    total += value;
  }

  static std::uint64_t resident_allocation_bytes_(std::size_t count, std::size_t element_size) {
    if (element_size != 0 && count > std::numeric_limits<std::uint64_t>::max() / element_size)
      throw std::overflow_error("logical-clock resident allocation overflows uint64");
    return static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(element_size);
  }

  static std::uint64_t external_string_storage_bytes_(const std::string& value) {
    const auto begin = reinterpret_cast<std::uintptr_t>(&value);
    const auto end = begin + sizeof(value);
    const auto data = reinterpret_cast<std::uintptr_t>(value.data());
    if (data >= begin && data < end)
      return 0;
    if (value.capacity() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("logical-clock resident string storage overflows uint64");
    return static_cast<std::uint64_t>(value.capacity()) + 1U;
  }

  static constexpr std::uint32_t kNoClock = std::numeric_limits<std::uint32_t>::max();

  std::optional<std::uint32_t> clock_index_unsealed_(std::string_view clock) const {
    for (std::size_t index = 0; index < clock_ids_.size(); ++index)
      if (clock_ids_[index] == clock)
        return static_cast<std::uint32_t>(index);
    if (clock == primary_)
      return std::uint32_t{0};
    return std::nullopt;
  }

  std::uint32_t clock_index_(std::string_view clock) const {
    if (!sealed_)
      throw std::logic_error("logical-clock schedule was not bind-sealed");
    const auto found = clock_index_unsealed_(clock);
    if (!found)
      throw std::runtime_error("logical clock is absent from the bind-sealed relation graph");
    return *found;
  }

  struct Relation {
    std::string parent;
    int count = 0;
  };

  template <class Map>
  static void copy_map_into_(Map& destination, const Map& source) {
    for (auto entry = destination.begin(); entry != destination.end();) {
      if (source.find(entry->first) == source.end())
        entry = destination.erase(entry);
      else
        ++entry;
    }
    for (const auto& [key, value] : source)
      destination.insert_or_assign(key, value);
  }

  static void copy_string_preallocated_(std::string& destination, const std::string& source) {
    if (source.size() > destination.capacity())
      throw std::logic_error("logical-clock string capacity was not primed");
    destination.assign(source.data(), source.size());
  }

  template <class Map>
  static void copy_relations_preallocated_(Map& destination, const Map& source) {
    if (destination.size() != source.size())
      throw std::logic_error("logical-clock relation graph changed after transaction prime");
    auto target = destination.begin();
    auto input = source.begin();
    for (; input != source.end(); ++input, ++target) {
      if (target->first != input->first)
        throw std::logic_error("logical-clock relation identity changed after transaction prime");
      if (target->second.parent != input->second.parent ||
          target->second.count != input->second.count)
        throw std::logic_error("logical-clock relation authority changed after transaction prime");
    }
  }

  static void copy_ticks_preallocated_(std::map<std::string, std::int64_t>& destination,
                                       const std::map<std::string, std::int64_t>& source) {
    if (destination.size() != source.size())
      throw std::logic_error("logical-clock accepted tick graph changed after transaction prime");
    auto target = destination.begin();
    auto input = source.begin();
    for (; input != source.end(); ++input, ++target) {
      if (target->first != input->first)
        throw std::logic_error(
            "logical-clock accepted tick identity changed after transaction prime");
      target->second = input->second;
    }
  }

  static std::int64_t checked_multiply_(std::int64_t a, std::int64_t b) {
    if (a < 0 || b <= 0 || (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a))
      throw std::runtime_error("logical-clock tick overflow");
    return a * b;
  }
  static std::int64_t checked_add_(std::int64_t a, std::int64_t b) {
    if (a < 0 || b < 0 || b > std::numeric_limits<std::int64_t>::max() - a)
      throw std::runtime_error("logical-clock tick overflow");
    return a + b;
  }

  std::int64_t ticks_per_macro_(const std::string& clock, std::set<std::string> visiting) const {
    if (clock.empty() || primary_.empty())
      throw std::runtime_error("logical-clock schedule is not configured");
    if (clock == primary_)
      return 1;
    if (!visiting.insert(clock).second)
      throw std::runtime_error("logical-clock relation cycle");
    const auto found = relations_.find(clock);
    if (found == relations_.end())
      throw std::runtime_error("logical clock is absent from the installed relation graph");
    return checked_multiply_(ticks_per_macro_(found->second.parent, std::move(visiting)),
                             found->second.count);
  }

  std::int64_t cached_ticks_per_macro_(const std::string& clock) const {
    const auto found = ticks_per_macro_cache_.find(clock);
    if (found == ticks_per_macro_cache_.end() || found->second <= 0)
      throw std::logic_error("logical-clock tick cache was not sealed at bind");
    return found->second;
  }

  std::optional<std::int64_t> active_tick_(std::string_view clock,
                                           std::int64_t macro_step) const {
    if (macro_step < 0 || primary_.empty())
      throw std::runtime_error("logical-clock coordinate requires a configured accepted step");
    const std::uint32_t requested = clock_index_(clock);
    std::uint32_t active = 0;
    std::int64_t tick = macro_step;
    for (const Frame& frame : frames_) {
      if (frame.parent != active || frame.next <= 0 || frame.next > frame.count)
        throw std::runtime_error("logical-clock active subcycle cursor is invalid");
      tick = checked_add_(checked_multiply_(tick, frame.count),
                          static_cast<std::int64_t>(frame.next - 1));
      active = frame.child;
    }
    if (active != requested)
      return std::nullopt;
    return tick;
  }

  std::size_t begin_(std::string_view parent, std::string_view child, int count) {
    if (parent.empty() || child.empty() || parent == child || count <= 0)
      throw std::runtime_error("invalid logical-clock subcycle descriptor");
    const std::uint32_t parent_id = clock_index_(parent);
    const std::uint32_t child_id = clock_index_(child);
    if (child_id == 0 || child_id >= parent_ids_.size() || parent_ids_[child_id] != parent_id ||
        relation_counts_[child_id] != count)
      throw std::runtime_error("logical-clock subcycle differs from the bind-sealed relation graph");
    const std::uint32_t active = frames_.empty() ? 0 : frames_.back().child;
    if (active != parent_id)
      throw std::runtime_error(
          "nested logical-clock subcycle parent does not match the active child clock");
    if (frames_.size() == frames_.capacity())
      throw std::logic_error("logical-clock frame stack was not primed at bind");
    frames_.push_back({parent_id, child_id, count, 0});
    return frames_.size() - 1;
  }

  Frame& frame_(std::size_t depth) {
    if (frames_.empty() || depth != frames_.size() - 1)
      throw std::runtime_error("logical-clock subcycle scopes must close in stack order");
    return frames_.back();
  }
  void iteration_(std::size_t depth, int index) {
    Frame& frame = frame_(depth);
    if (index != frame.next || index < 0 || index >= frame.count)
      throw std::runtime_error("logical-clock subcycle iteration cursor is not sequential");
    ++frame.next;
  }
  void finish_(std::size_t depth) {
    Frame& frame = frame_(depth);
    if (frame.next != frame.count)
      throw std::runtime_error("logical-clock subcycle ended before all child ticks completed");
    frames_.pop_back();
  }
  void abort_(std::size_t depth) noexcept {
    if (!frames_.empty() && depth == frames_.size() - 1)
      frames_.pop_back();
  }

  std::string primary_;
  bool sealed_ = false;
  std::vector<std::string> clock_ids_;
  std::vector<std::uint32_t> parent_ids_;
  std::vector<int> relation_counts_;
  std::vector<std::int64_t> accepted_tick_wire_values_;
  std::map<std::string, Relation> relations_;
  /// Cold-populated scalar products from the primary clock.  This is separate from accepted
  /// ticks so every hot accepted image can multiply/assign over pre-created nodes only.
  std::map<std::string, std::int64_t> ticks_per_macro_cache_;
  std::map<std::string, std::int64_t> accepted_ticks_;
  std::vector<Frame> frames_;
};

}  // namespace pops::runtime::program
