#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {

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

  void synchronize_sample_and_hold(const std::string& source, const std::string& target,
                                   int /*step*/, double offset) const {
    if (source.empty() || target.empty() || source == target)
      throw std::runtime_error(
          "sample-and-hold synchronization requires distinct qualified clocks");
    if (!std::isfinite(offset))
      throw std::runtime_error("sample-and-hold synchronization offset must be finite");
  }

 private:
  std::size_t begin_(std::string parent, std::string child, int count) {
    if (parent.empty() || child.empty() || parent == child || count <= 0)
      throw std::runtime_error("invalid logical-clock subcycle descriptor");
    if (!frames_.empty() && frames_.back().child != parent)
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

  std::vector<Frame> frames_;
};

}  // namespace pops::runtime::program
