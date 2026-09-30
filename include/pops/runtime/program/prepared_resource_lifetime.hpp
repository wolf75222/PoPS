#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::runtime::accelerator {
template <class Scalar, class ExecutionSpace>
class PreparedAcceleratorStreamExecutor;
}

namespace pops::runtime::program {
class PreparedResourceCache;

enum class PreparedResourceAction { drain, finish, reject };

namespace resource_lifetime_detail {
struct Authority {};
struct Version {
  std::weak_ptr<Authority> authority;
  std::uint64_t ordinal = 0;
  std::atomic<bool> current{true};
};
struct Attempt {
  std::weak_ptr<Authority> authority;
  std::uint64_t ordinal = 0;
  std::atomic<bool> rejected{false};
  bool visible() const noexcept {
    return !authority.expired() && !rejected.load(std::memory_order_acquire);
  }
};
}  // namespace resource_lifetime_detail

/// Opaque identity of one cache-owned numerical attempt. Rejection is permanent.
/// This is lifetime/consumption authority, never a numerical acceptance predicate.
class PreparedResourceAttempt {
 public:
  PreparedResourceAttempt() = default;
  std::uint64_t ordinal() const noexcept { return state_ ? state_->ordinal : 0; }
  bool visible() const noexcept { return state_ && state_->visible(); }
  bool same_attempt(const PreparedResourceAttempt& other) const noexcept {
    return state_ && state_ == other.state_;
  }
  void reject() const noexcept {
    if (state_)
      state_->rejected.store(true, std::memory_order_release);
  }

 private:
  friend class PreparedResourceCache;
  friend class PreparedResourceTask;
  template <class, class>
  friend class ::pops::runtime::accelerator::PreparedAcceleratorStreamExecutor;
  explicit PreparedResourceAttempt(std::shared_ptr<resource_lifetime_detail::Attempt> state)
      : state_(std::move(state)) {}
  std::shared_ptr<resource_lifetime_detail::Attempt> state_;
};

/// Retains the actual prepared object, including its Kokkos allocations/lane owners.
/// Revocation affects new use and publication, not the physical lifetime of this lease.
template <class Resource>
class PreparedResourceLease {
 public:
  PreparedResourceLease() = default;
  Resource& get() const {
    if (!owner_)
      throw std::logic_error("empty prepared resource lease");
    return *owner_;
  }
  Resource* operator->() const { return &get(); }
  Resource& operator*() const { return get(); }
  bool current() const noexcept {
    return version_ && !version_->authority.expired() &&
           version_->current.load(std::memory_order_acquire);
  }
  std::uint64_t version() const noexcept { return version_ ? version_->ordinal : 0; }
  explicit operator bool() const noexcept { return static_cast<bool>(owner_); }

 private:
  friend class PreparedResourceCache;
  template <class, class>
  friend class ::pops::runtime::accelerator::PreparedAcceleratorStreamExecutor;
  PreparedResourceLease(std::shared_ptr<Resource> owner,
                        std::shared_ptr<resource_lifetime_detail::Version> version)
      : owner_(std::move(owner)), version_(std::move(version)) {}
  std::shared_ptr<Resource> owner_;
  std::shared_ptr<resource_lifetime_detail::Version> version_;
};

/// Move-only handle to real submitted work. A host coordinator serializes poll/wait;
/// native queues never mutate this control object. Dropping the final handle drains
/// before releasing owners. CPU completion can be acknowledged late, but is not GPU overlap.
class PreparedResourceTask {
 public:
  enum class Status { pending, succeeded, failed, cancel_requested, cancelled };

  PreparedResourceTask() = default;
  PreparedResourceTask(const PreparedResourceTask&) = delete;
  PreparedResourceTask& operator=(const PreparedResourceTask&) = delete;
  PreparedResourceTask(PreparedResourceTask&&) noexcept = default;
  PreparedResourceTask& operator=(PreparedResourceTask&&) noexcept = default;

  Status status() const { return control_().status(); }
  bool physically_complete() const { return control_().complete; }
  std::size_t retained_resource_count() const {
    std::size_t count = 0;
    for (const auto& lease : control_().retained)
      count += lease.owner ? 1 : 0;
    return count;
  }
  bool poll() { return control_().poll(); }
  void wait() { control_().wait(); }
  void request_cancel() noexcept {
    if (control)
      control->cancel_requested = true;
  }
  void require_consumable(const PreparedResourceAttempt& attempt) const {
    const auto& task = control_();
    if (!attempt.visible() || task.attempt != attempt.state_ || task.status() != Status::succeeded)
      throw std::logic_error("prepared task result is not consumable in this exact attempt");
  }
  void rethrow_failure() const {
    if (control_().error)
      std::rethrow_exception(control_().error);
  }

 private:
  friend class PreparedResourceCache;
  template <class, class>
  friend class ::pops::runtime::accelerator::PreparedAcceleratorStreamExecutor;

  struct Retained {
    std::shared_ptr<void> owner;
    std::weak_ptr<resource_lifetime_detail::Version> version;
  };
  struct Control {
    std::shared_ptr<resource_lifetime_detail::Attempt> attempt;
    std::vector<Retained> retained;
    // Supplied only by a native executor after creating its actual event/instance.
    // No public API accepts a caller-provided success/completion boolean.
    std::function<bool()> query;
    std::function<void()> drain;
    // Preparation can allocate/throw before a queue submission. The executor arms
    // this only after all owners, callbacks and cache registration are ready.
    bool complete = true;
    bool cancel_requested = false;
    std::exception_ptr error;

    ~Control() noexcept {
      if (!complete) {
        try {
          wait();
        } catch (...) {
          // A queue that cannot be drained must not outlive freed captured buffers.
          std::terminate();
        }
      }
    }
    bool revoked() const noexcept {
      if (cancel_requested || !attempt || !attempt->visible())
        return true;
      for (const auto& lease : retained) {
        const auto version = lease.version.lock();
        if (!version || version->authority.expired() ||
            !version->current.load(std::memory_order_acquire))
          return true;
      }
      return false;
    }
    Status status() const noexcept {
      if (revoked())
        return complete ? Status::cancelled : Status::cancel_requested;
      if (!complete)
        return Status::pending;
      return error ? Status::failed : Status::succeeded;
    }
    void finish() noexcept {
      // Latch revocation before dropping the last holder and expiring its weak tag.
      cancel_requested = revoked();
      complete = true;
      // Completion's execution-instance copy can wrap an externally owned native
      // stream. Destroy the event/callbacks before releasing the executor owner.
      query = nullptr;
      drain = nullptr;
      for (auto& lease : retained)
        lease.owner.reset();
    }
    bool poll() {
      if (complete)
        return true;
      try {
        if (!query())
          return false;
      } catch (...) {
        error = std::current_exception();
        // Query failure is not proof of completion. Drain must establish it.
        wait();
        return true;
      }
      finish();
      return true;
    }
    void wait() {
      if (complete)
        return;
      // On failure leave complete=false and keep EVERY owner for a later drain.
      try {
        drain();
      } catch (...) {
        if (!error)
          error = std::current_exception();
        throw;
      }
      finish();
    }
  };

  explicit PreparedResourceTask(std::shared_ptr<Control> state) : control(std::move(state)) {}
  Control& control_() const {
    if (!control)
      throw std::logic_error("empty prepared resource task");
    return *control;
  }
  std::shared_ptr<Control> control;
};

}  // namespace pops::runtime::program
