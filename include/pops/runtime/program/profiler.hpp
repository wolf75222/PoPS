#pragma once

// A lightweight per-node / per-brick profiler for the compiled Program step (Spec 3 section 29-30,
// ADC-459). It accumulates named wall-clock scopes -- one per Program node, native brick, or step
// phase -- plus a few integer counters (kernels, cache hits/misses, scheduled nodes due/skipped),
// and renders the report `sim.profile_report()` returns.
//
// Cost when disabled: record()/count()/count_max() are one atomic load and a predictable branch;
// they do not acquire the mutex. A disabled ProfileScope additionally constructs its name and reads
// that flag once, but does not read the clock. One Profiler may be shared by concurrent host sessions:
// enabled writes and all snapshots are serialized, and first-seen order is the order in which new
// names acquire that serialization point. Host only; on a device backend a Kokkos::fence() must
// precede the scope close so the timing reflects the kernel (the fence lives at the call site, not
// here -- this header has no Kokkos dependency). MPI: each rank profiles itself; the report is
// per-rank (any reduction belongs to the System integration).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {

class Profiler {
 public:
  struct Entry {
    std::uint64_t count = 0;
    double total_s = 0.0;
    double min_s = 0.0;
    double max_s = 0.0;
    double mean_s() const { return count != 0 ? total_s / static_cast<double>(count) : 0.0; }
  };

  struct ScopeSnapshot {
    std::string name;
    std::uint64_t count = 0;
    double total_s = 0.0;
    double mean_s = 0.0;
    double min_s = 0.0;
    double max_s = 0.0;
  };

  struct CounterSnapshot {
    std::string name;
    std::int64_t value = 0;
  };

  struct Snapshot {
    int schema_version = 1;
    bool enabled = false;
    double total_s = 0.0;
    std::vector<ScopeSnapshot> scopes;
    std::vector<CounterSnapshot> counters;
  };

  Profiler() = default;

  // Runtime step transactions snapshot and restore the profiler together with the numerical state.
  // Keep that value contract while making copies consistent with concurrent writers.
  Profiler(const Profiler& other) {
    std::lock_guard<std::mutex> lock(other.mutex_);
    copy_from_unlocked_(other);
  }

  Profiler& operator=(const Profiler& other) {
    if (this == &other) {
      return *this;
    }
    std::scoped_lock lock(mutex_, other.mutex_);
    copy_from_unlocked_(other);
    return *this;
  }

  Profiler(Profiler&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    move_from_unlocked_(std::move(other));
  }

  Profiler& operator=(Profiler&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    std::scoped_lock lock(mutex_, other.mutex_);
    move_from_unlocked_(std::move(other));
    return *this;
  }

  /// Refresh a bind-frozen profiler image without replacing its maps/vectors. Scope and counter
  /// identities are immutable once the Program has been primed; a late identity or an undersized
  /// string buffer is rejected instead of allocating in the accepted-step snapshot callback.
  void copy_from_preallocated(const Profiler& source) {
    if (this == &source)
      return;
    std::scoped_lock lock(mutex_, source.mutex_);
    if (order_.size() != source.order_.size() || entries_.size() != source.entries_.size() ||
        counter_order_.size() != source.counter_order_.size() ||
        counters_.size() != source.counters_.size())
      throw std::logic_error("Program profiler identity set changed after preparation");
    for (std::size_t index = 0; index < source.order_.size(); ++index) {
      if (order_[index] != source.order_[index])
        throw std::logic_error("Program profiler scope order changed after preparation");
    }
    for (const auto& [name, entry] : source.entries_) {
      const auto found = entries_.find(name);
      if (found == entries_.end())
        throw std::logic_error("Program profiler scope identity changed after preparation");
      found->second = entry;
    }
    for (std::size_t index = 0; index < source.counter_order_.size(); ++index) {
      if (counter_order_[index] != source.counter_order_[index])
        throw std::logic_error("Program profiler counter order changed after preparation");
    }
    for (const auto& [name, value] : source.counters_) {
      const auto found = counters_.find(name);
      if (found == counters_.end())
        throw std::logic_error("Program profiler counter identity changed after preparation");
      found->second = value;
    }
    enabled_.store(source.enabled_.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  /// Fully allocated rollback image. The live profiler lock is acquired during preparation and
  /// retained until publication/destruction, so publication cannot fail while replacing the exact
  /// accepted counters and first-seen ordering.
  class PreparedRestore {
   public:
    PreparedRestore(const PreparedRestore&) = delete;
    PreparedRestore& operator=(const PreparedRestore&) = delete;
    PreparedRestore(PreparedRestore&&) noexcept = default;
    PreparedRestore& operator=(PreparedRestore&&) noexcept = default;

   private:
    friend class Profiler;

    PreparedRestore(Profiler& owner, const Profiler& accepted)
        : owner_(&owner), owner_lock_(owner.mutex_) {
      std::lock_guard<std::mutex> accepted_lock(accepted.mutex_);
      enabled_ = accepted.enabled_.load(std::memory_order_relaxed);
      order_ = accepted.order_;
      entries_ = accepted.entries_;
      counter_order_ = accepted.counter_order_;
      counters_ = accepted.counters_;
    }

    Profiler* owner_ = nullptr;
    std::unique_lock<std::mutex> owner_lock_;
    bool enabled_ = false;
    std::vector<std::string> order_;
    std::map<std::string, Entry> entries_;
    std::vector<std::string> counter_order_;
    std::map<std::string, std::int64_t> counters_;
  };

  PreparedRestore prepare_restore(const Profiler& accepted) {
    if (this == &accepted)
      throw std::invalid_argument("Profiler restore requires an independent accepted image");
    return PreparedRestore(*this, accepted);
  }

  void publish_prepared_restore(PreparedRestore&& prepared) noexcept {
    if (prepared.owner_ != this || !prepared.owner_lock_.owns_lock())
      std::terminate();
    static_assert(noexcept(order_.swap(prepared.order_)));
    static_assert(noexcept(entries_.swap(prepared.entries_)));
    static_assert(noexcept(counter_order_.swap(prepared.counter_order_)));
    static_assert(noexcept(counters_.swap(prepared.counters_)));
    enabled_.store(prepared.enabled_, std::memory_order_release);
    order_.swap(prepared.order_);
    entries_.swap(prepared.entries_);
    counter_order_.swap(prepared.counter_order_);
    counters_.swap(prepared.counters_);
  }

  void enable() {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_.store(true, std::memory_order_release);
  }

  void disable() {
    // Serialize the transition with an in-flight enabled writer. Once disable() returns, no writer
    // from the preceding enabled interval can still publish a sample.
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_.store(false, std::memory_order_release);
  }

  bool enabled() const noexcept { return enabled_.load(std::memory_order_acquire); }

  // Drop accumulated values while retaining the bind-primed identities.  Accepted-step snapshots
  // copy the profiler into an already allocated image: clearing the maps here would make the next
  // snapshot either allocate or reject a perfectly valid reset as an identity drift.
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, entry] : entries_) {
      (void)name;
      entry = {};
    }
    for (auto& [name, value] : counters_) {
      (void)name;
      value = 0;
    }
  }

  // Record one timed sample of `name`, in seconds. No-op when disabled.
  void record(const std::string& name, double seconds) {
    if (!enabled()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_.load(std::memory_order_relaxed)) {
      return;
    }
    record_unlocked_(name, seconds);
  }

  // Bump a named integer counter (e.g. "kernels", "cache_hits", "nodes_skipped"). No-op when off.
  void count(const std::string& name, std::int64_t by = 1) {
    if (!enabled()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_.load(std::memory_order_relaxed)) {
      return;
    }
    count_unlocked_(name, by);
  }

  // Track a named PEAK counter: set it to max(current, value) instead of accumulating. Used for the
  // scratch peak memory (the largest single scratch allocation seen, in bytes), where the running sum
  // is meaningless. First-seen creates the counter at @p value. No-op when disabled.
  void count_max(const std::string& name, std::int64_t value) {
    if (!enabled()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_.load(std::memory_order_relaxed)) {
      return;
    }
    count_max_unlocked_(name, value);
  }

  // Record one authenticated scheduler decision and return it unchanged so generated code can
  // wrap every due primitive exactly once. All scheduled nodes contribute due/skipped counts;
  // cache hits/misses move only for policies that really own a STORE+RESTORE cache path.
  bool schedule_decision(bool due, bool cache_backed) {
    if (!enabled()) {
      return due;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_.load(std::memory_order_relaxed)) {
      return due;
    }
    count_unlocked_(due ? "nodes_due" : "nodes_skipped", 1);
    if (cache_backed) {
      count_unlocked_(due ? "cache_misses" : "cache_hits", 1);
    }
    return due;
  }

  // Legacy pointer view. The lookup itself is synchronized, but the returned Entry is owned by this
  // Profiler: callers must not retain or dereference it while another thread can record() or reset().
  // Use snapshot() for a value-owned read that is safe while writers remain active.
  const Entry* entry(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
  }

  std::int64_t counter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = counters_.find(name);
    return it == counters_.end() ? 0 : it->second;
  }

  // Sum of every scope's total time (the "total" line of the report).
  double total_s() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_s_unlocked_();
  }

  std::size_t scope_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return order_.size();
  }

  // Structured source of truth for inspection. profile_report() below is only a pretty view of this
  // accumulated data; Python does not need to parse the text path when this snapshot is exposed.
  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot out{};
    out.enabled = enabled_.load(std::memory_order_relaxed);
    out.total_s = total_s_unlocked_();
    out.scopes.reserve(order_.size());
    for (const auto& name : order_) {
      const Entry& e = entries_.at(name);
      out.scopes.push_back(ScopeSnapshot{.name = name,
                                         .count = e.count,
                                         .total_s = e.total_s,
                                         .mean_s = e.mean_s(),
                                         .min_s = e.min_s,
                                         .max_s = e.max_s});
    }
    out.counters.reserve(counter_order_.size());
    for (const auto& name : counter_order_) {
      out.counters.push_back(CounterSnapshot{.name = name, .value = counters_.at(name)});
    }
    return out;
  }

  // A human-readable report in first-seen order: one line per scope (count / total / mean / min /
  // max), then the counters. The exact text the Python `sim.profile_report()` returns.
  std::string report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(6);
    os << "Profiler report (total " << total_s_unlocked_() << " s, " << order_.size()
       << " scopes)\n";
    for (const auto& name : order_) {
      const Entry& e = entries_.at(name);
      os << "  " << name << "  count=" << e.count << "  total=" << e.total_s
         << "s  mean=" << e.mean_s() << "s  min=" << e.min_s << "s  max=" << e.max_s << "s\n";
    }
    if (!counter_order_.empty()) {
      os << "counters:";
      for (const auto& name : counter_order_) {
        os << "  " << name << "=" << counters_.at(name);
      }
      os << "\n";
    }
    return os.str();
  }

 private:
  void copy_from_unlocked_(const Profiler& other) {
    enabled_.store(other.enabled_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    order_ = other.order_;
    entries_ = other.entries_;
    counter_order_ = other.counter_order_;
    counters_ = other.counters_;
  }

  void move_from_unlocked_(Profiler&& other) {
    enabled_.store(other.enabled_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    order_ = std::move(other.order_);
    entries_ = std::move(other.entries_);
    counter_order_ = std::move(other.counter_order_);
    counters_ = std::move(other.counters_);
  }

  void record_unlocked_(const std::string& name, double seconds) {
    auto [it, inserted] = entries_.try_emplace(
        name, Entry{.count = 1, .total_s = seconds, .min_s = seconds, .max_s = seconds});
    if (inserted) {
      try {
        order_.push_back(name);
      } catch (...) {
        entries_.erase(it);
        throw;
      }
      return;
    }

    Entry& entry = it->second;
    entry.count += 1;
    entry.total_s += seconds;
    entry.min_s = std::min(entry.min_s, seconds);
    entry.max_s = std::max(entry.max_s, seconds);
  }

  void count_unlocked_(const std::string& name, std::int64_t by) {
    auto [it, inserted] = counters_.try_emplace(name, by);
    if (inserted) {
      try {
        counter_order_.push_back(name);
      } catch (...) {
        counters_.erase(it);
        throw;
      }
      return;
    }
    it->second += by;
  }

  void count_max_unlocked_(const std::string& name, std::int64_t value) {
    auto [it, inserted] = counters_.try_emplace(name, value);
    if (inserted) {
      try {
        counter_order_.push_back(name);
      } catch (...) {
        counters_.erase(it);
        throw;
      }
      return;
    }
    it->second = std::max(it->second, value);
  }

  double total_s_unlocked_() const {
    double total = 0.0;
    for (const auto& name : order_) {
      total += entries_.at(name).total_s;
    }
    return total;
  }

  mutable std::mutex mutex_;
  std::atomic<bool> enabled_{false};
  std::vector<std::string> order_;  // scope names, first-seen order (stable report)
  std::map<std::string, Entry> entries_;
  std::vector<std::string> counter_order_;  // counter names, first-seen order
  std::map<std::string, std::int64_t> counters_;
};

// RAII scope: times its own lifetime into `prof` under `name`. One per Program node / brick call.
// Construct it at the top of the work; its destructor records the elapsed wall-clock seconds.
class ProfileScope {
 public:
  ProfileScope(Profiler& prof, std::string name)
      : prof_(prof), name_(std::move(name)), active_(prof_.enabled()) {
    if (active_) {
      t0_ = std::chrono::steady_clock::now();
    }
  }

  ~ProfileScope() {
    if (!active_) {
      return;
    }
    const auto t1 = std::chrono::steady_clock::now();
    // A timing alloc failure must never abort the program nor escape this destructor: swallow any
    // exception from record() (the worst case is one lost sample).
    try {
      prof_.record(name_, std::chrono::duration<double>(t1 - t0_).count());
    } catch (...) {  // NOLINT(bugprone-empty-catch) -- a profiler never throws out of a scope
    }
  }

  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;
  ProfileScope(ProfileScope&&) = delete;
  ProfileScope& operator=(ProfileScope&&) = delete;

 private:
  Profiler& prof_;
  std::string name_;
  bool active_ = false;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace pops::runtime::program
