#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
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
    std::string parent;
    std::string child;
    int count = 0;
    int next = 0;
  };

  class SubcycleScope {
   public:
    SubcycleScope(ClockScheduleState& owner, std::string parent, std::string child, int count)
        : owner_(&owner), depth_(owner.begin_(std::move(parent), std::move(child), count)) {}
    SubcycleScope(const SubcycleScope&) = delete;
    SubcycleScope& operator=(const SubcycleScope&) = delete;
    SubcycleScope(SubcycleScope&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), depth_(other.depth_),
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

  SubcycleScope subcycle(std::string parent, std::string child, int count) {
    return SubcycleScope(*this, std::move(parent), std::move(child), count);
  }

  void configure_primary_clock(std::string clock) {
    if (clock.empty())
      throw std::runtime_error("logical-clock primary identity must be non-empty");
    if (!primary_.empty() && primary_ != clock)
      throw std::runtime_error("logical-clock primary identity changed after installation");
    primary_ = std::move(clock);
  }

  void declare_relation(std::string parent, std::string child, int count) {
    if (parent.empty() || child.empty() || parent == child || count <= 0)
      throw std::runtime_error("invalid logical-clock subcycle descriptor");
    const auto found = relations_.find(child);
    const Relation relation{std::move(parent), count};
    if (found != relations_.end() &&
        (found->second.parent != relation.parent || found->second.count != relation.count))
      throw std::runtime_error("logical child clock has conflicting parent/count declarations");
    relations_[child] = relation;
    (void)ticks_per_macro_(child, {});  // validates reachability and cycles immediately.
  }

  std::optional<ScheduleCoordinate> coordinate(
      ScheduleDomainKind kind, const std::string& clock, const std::string& stage_identity,
      int required_level, int current_level, std::int64_t macro_step) const {
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
    result.emplace(primary_, checked_multiply_(macro_step, ticks_per_macro_(primary_, {})));
    for (const auto& [child, relation] : relations_) {
      (void)relation;
      result.emplace(child, checked_multiply_(macro_step, ticks_per_macro_(child, {})));
    }
    return result;
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
  struct Relation {
    std::string parent;
    int count = 0;
  };

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

  std::optional<std::int64_t> active_tick_(const std::string& clock,
                                           std::int64_t macro_step) const {
    if (macro_step < 0 || primary_.empty())
      throw std::runtime_error("logical-clock coordinate requires a configured accepted step");
    std::string active = primary_;
    std::int64_t tick = macro_step;
    for (const Frame& frame : frames_) {
      if (frame.parent != active || frame.next <= 0 || frame.next > frame.count)
        throw std::runtime_error("logical-clock active subcycle cursor is invalid");
      tick = checked_add_(checked_multiply_(tick, frame.count),
                          static_cast<std::int64_t>(frame.next - 1));
      active = frame.child;
    }
    if (active != clock)
      return std::nullopt;
    return tick;
  }

  std::size_t begin_(std::string parent, std::string child, int count) {
    if (parent.empty() || child.empty() || parent == child || count <= 0)
      throw std::runtime_error("invalid logical-clock subcycle descriptor");
    declare_relation(parent, child, count);
    const std::string active = frames_.empty() ? primary_ : frames_.back().child;
    if (active != parent)
      throw std::runtime_error(
          "nested logical-clock subcycle parent does not match the active child clock");
    frames_.push_back({std::move(parent), std::move(child), count, 0});
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
  std::map<std::string, Relation> relations_;
  std::map<std::string, std::int64_t> accepted_ticks_;
  std::vector<Frame> frames_;
};

}  // namespace pops::runtime::program
