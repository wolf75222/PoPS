#pragma once

#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

#include <cstdint>
#include <exception>
#include <map>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <utility>

namespace pops::runtime::program {

/// Context-owned numerical storage, separate from accepted state and evaluated data.
/// A Program node owns its resources until the context's layout generation changes;
/// different nodes/blocks/levels never share retained stage outputs. A caller must
/// reevaluate its law after acquire. Matches must be rank-local, including every
/// getter it calls: some ranks can have no resource and skip the predicate entirely.
/// Constructors containing collectives must make their internal allocations/failures
/// collective too, as prepared providers do. Replacement is published only after every
/// rank constructs successfully; a failed candidate leaves existing storage unchanged.
class PreparedResourceCache {
 private:
  struct Entry {
    std::shared_ptr<void> storage;
    std::shared_ptr<resource_lifetime_detail::Version> version;
  };
  using Key = std::tuple<std::type_index, std::int64_t, int, int>;
  struct State : resource_lifetime_detail::Authority {
    std::map<Key, Entry> entries;
    std::vector<std::shared_ptr<PreparedResourceTask::Control>> tasks;
    std::shared_ptr<resource_lifetime_detail::Attempt> attempt;
    std::uint64_t next_version = 0;
    std::uint64_t next_attempt = 0;
  };

 public:
  PreparedResourceCache() : state_(std::make_shared<State>()) {}
  PreparedResourceCache(const PreparedResourceCache&) = delete;
  PreparedResourceCache& operator=(const PreparedResourceCache&) = delete;
  PreparedResourceCache(PreparedResourceCache&&) = default;
  PreparedResourceCache& operator=(PreparedResourceCache&& other) noexcept {
    if (this != &other) {
      close_();
      state_ = std::move(other.state_);
    }
    return *this;
  }
  ~PreparedResourceCache() noexcept { close_(); }

  template <class Resource, class Matches, class... Args>
  Resource& acquire(std::int64_t node, int block, int level, const ExecutionLane& lane,
                    Matches&& matches, Args&&... args) {
    return acquire_lease<Resource>(node, block, level, lane, std::forward<Matches>(matches),
                                   std::forward<Args>(args)...)
        .get();
  }

  template <class Resource, class Matches, class... Args>
  PreparedResourceLease<Resource> acquire_lease(std::int64_t node, int block, int level,
                                                const ExecutionLane& lane, Matches&& matches,
                                                Args&&... args) {
    using Slot = std::optional<Resource>;
    std::shared_ptr<Slot> slot;
    Entry* entry_storage = nullptr;
    std::exception_ptr error;
    long disposition = 0;
    try {
      if (node < 0 || block < 0 || level < 0)
        throw std::invalid_argument("prepared resource requires exact nonnegative scope ids");
      if (!state_)
        throw std::logic_error("prepared resource cache was moved from");
      const Key key{std::type_index(typeid(Resource)), node, block, level};
      auto entry = state_->entries.try_emplace(key).first;
      entry_storage = &entry->second;
      drain_resource_(*state_, entry->second.version);
      // Allocate the empty holder before peers can enter a provider constructor.
      // Allocation failure leaves a retryable empty entry, never a readable resource.
      if (!entry->second.storage)
        entry->second.storage = std::make_shared<Slot>();
      slot = std::static_pointer_cast<Slot>(entry->second.storage);
      disposition = !slot->has_value() || !matches(slot->value()) ? 1L : 0L;
      if (disposition == 1 && state_->next_version == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("prepared resource version exhausted uint64_t");
    } catch (...) {
      error = std::current_exception();
      disposition = 2;
    }
    const long collective = all_reduce_max(disposition, lane);
    if (collective == 2) {
      if (error)
        std::rethrow_exception(error);
      throw std::runtime_error("prepared resource preflight failed on another rank");
    }
    if (collective == 1) {
      // Keep the current holder (and its numerical buffers) alive while building the
      // candidate. In-place optional::emplace would destroy it before construction.
      std::shared_ptr<Slot> candidate;
      std::shared_ptr<resource_lifetime_detail::Version> candidate_version;
      try {
        if (state_->next_version == std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("prepared resource version exhausted uint64_t");
        candidate = std::make_shared<Slot>();
        candidate_version = std::make_shared<resource_lifetime_detail::Version>();
        candidate_version->authority = state_;
        candidate_version->ordinal = state_->next_version + 1;
      } catch (...) {
        error = std::current_exception();
      }
      // A holder allocation must fail collectively before any provider constructor
      // can enter its own collectives.
      if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
        if (error)
          std::rethrow_exception(error);
        throw std::runtime_error("prepared resource candidate allocation failed on another rank");
      }
      try {
        candidate->emplace(std::forward<Args>(args)...);
      } catch (...) {
        error = std::current_exception();
      }
      if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
        if (error)
          std::rethrow_exception(error);
        throw std::runtime_error("prepared resource construction failed on another rank");
      }
      // Shared-holder publication cannot allocate or move Resource. Non-movable
      // resources and their retained lane/buffer addresses remain supported.
      if (entry_storage->version)
        entry_storage->version->current.store(false, std::memory_order_release);
      entry_storage->storage = candidate;
      entry_storage->version = std::move(candidate_version);
      ++state_->next_version;
      slot = std::move(candidate);
    }
    return PreparedResourceLease<Resource>(std::shared_ptr<Resource>(slot, &slot->value()),
                                           entry_storage->version);
  }

  /// Start a fresh exact attempt. Superseded work remains drainable, never consumable.
  PreparedResourceAttempt begin_attempt() {
    if (!state_)
      throw std::logic_error("prepared resource cache was moved from");
    if (state_->next_attempt == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("prepared resource attempt exhausted uint64_t");
    auto next = std::make_shared<resource_lifetime_detail::Attempt>();
    next->authority = state_;
    next->ordinal = state_->next_attempt + 1;
    reject_attempt();
    drain();  // Old scratch must not be overwritten while a rejected queue still uses it.
    state_->tasks.clear();
    state_->attempt = std::move(next);
    ++state_->next_attempt;
    return PreparedResourceAttempt(state_->attempt);
  }
  /// Every rank enters before a Program body can perform collective work.
  PreparedResourceAttempt begin_attempt(const ExecutionLane& lane) {
    PreparedResourceAttempt attempt;
    std::exception_ptr error;
    try {
      attempt = begin_attempt();
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      reject_attempt();
      drain();
      if (error)
        std::rethrow_exception(error);
      throw std::runtime_error("prepared resource attempt failed on another rank");
    }
    return attempt;
  }
  PreparedResourceAttempt current_attempt() const {
    if (!state_ || !state_->attempt || !state_->attempt->visible())
      throw std::logic_error("prepared resource cache has no live attempt");
    return PreparedResourceAttempt(state_->attempt);
  }
  void reject_attempt() noexcept {
    if (state_ && state_->attempt)
      state_->attempt->rejected.store(true, std::memory_order_release);
  }

  /// Logical invalidation does not claim device completion. Pending tickets own
  /// their old object and captured buffers until poll/wait acknowledges completion.
  void clear() noexcept {
    if (!state_)
      return;
    reject_attempt();
    for (auto& [key, entry] : state_->entries)
      if (entry.version)
        entry.version->current.store(false, std::memory_order_release);
    state_->entries.clear();
  }
  bool quiescent() const noexcept {
    if (!state_)
      return true;
    for (const auto& task : state_->tasks)
      if (!task->complete)
        return false;
    return true;
  }
  void drain() {
    if (state_)
      drain_(*state_);
  }
  /// All native work required by an attempt must complete successfully before its
  /// enclosing Program step can return to accepted-state publication.
  void finish_attempt() {
    if (state_)
      finish_(*state_);
  }

  /// Hook installed into the actual ProgramRuntimeState. It retains no cache or
  /// context pointer, and is safe after cache moves/destruction or snapshot copying.
  std::function<void(PreparedResourceAction)> lifetime_callback() const {
    return [weak = std::weak_ptr<State>(state_)](PreparedResourceAction action) {
      if (auto state = weak.lock()) {
        if (action == PreparedResourceAction::reject && state->attempt)
          state->attempt->rejected.store(true, std::memory_order_release);
        if (action == PreparedResourceAction::finish)
          finish_(*state);
        else
          drain_(*state);
      }
    };
  }

 private:
  static void finish_(State& state) {
    drain_(state);
    for (const auto& task : state.tasks) {
      if (task->attempt != state.attempt)
        continue;
      if (task->error)
        std::rethrow_exception(task->error);
      if (task->status() != PreparedResourceTask::Status::succeeded)
        throw std::runtime_error("prepared resource attempt has cancelled or failed native work");
    }
  }
  static void drain_(State& state) {
    std::exception_ptr error;
    for (const auto& task : state.tasks) {
      try {
        task->wait();
      } catch (...) {
        if (!error)
          error = std::current_exception();
      }
    }
    if (error)
      std::rethrow_exception(error);
  }

  static void drain_resource_(State& state,
                              const std::shared_ptr<resource_lifetime_detail::Version>& version) {
    if (!version)
      return;
    for (const auto& task : state.tasks) {
      if (task->complete)
        continue;
      for (const auto& retained : task->retained) {
        if (retained.version.lock() == version) {
          task->wait();
          break;
        }
      }
    }
  }

  template <class, class>
  friend class ::pops::runtime::accelerator::PreparedAcceleratorStreamExecutor;
  void register_task_(const std::shared_ptr<PreparedResourceTask::Control>& task) {
    if (!state_ || !task->attempt || !task->attempt->visible() ||
        task->attempt->authority.lock() != state_ || task->attempt != state_->attempt)
      throw std::logic_error("prepared submission requires this cache's exact live attempt");
    state_->tasks.push_back(
        task);  // All potentially throwing registration precedes queue submission.
  }
  void close_() noexcept {
    clear();
    try {
      drain();
    } catch (...) {
      std::terminate();
    }
  }
  std::shared_ptr<State> state_;
};

}  // namespace pops::runtime::program
