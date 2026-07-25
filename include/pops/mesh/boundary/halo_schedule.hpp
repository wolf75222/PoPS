/// @file
/// @brief HaloSchedule: memoized intra-level halo-exchange plan for fill_boundary (ADC-260).
///
/// fill_boundary_begin used to enumerate, on EVERY call, the neighbor-box job schedule: a BoxHash
/// build plus a local (and, under MPI, a global) enumeration over the BoxArray. That schedule is a
/// pure function of the LAYOUT (BoxArray, DistributionMapping, n_grow) and the per-call (Periodicity,
/// domain); only the copy/pack/MPI/unpack of the LIVE data must rerun. This header holds the
/// cacheable plan so the enumeration runs ONCE per (layout, Periodicity, domain). Jobs carry GLOBAL
/// box indices (resolved to local fabs at replay), so a plan is valid for any MultiFab over the same
/// layout. MPI-free: the in-flight buffers and MPI_Request stay in HaloExchange (fill_boundary.hpp).

#pragma once

#include <pops/mesh/index/box2d.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace pops {

struct HaloExchangeStorage;

/// One halo copy/transfer: the ghost @p region of box @p dst is filled from the shifted valid region
/// of box @p src (shift sx, sy in cells for the periodic wrap; 0 for an interior neighbor). @p src and
/// @p dst are GLOBAL box indices into the BoxArray (resolved to local fabs when the job is replayed).
struct HaloJob {
  int src = 0;
  int dst = 0;
  int sx = 0;
  int sy = 0;
  Box2D region{};
};

/// Memoized schedule for ONE (Periodicity, domain, communicator rank space) over a fixed layout.
/// @p local holds the copies
/// whose dst AND src are owned by this rank; @p send[r]/@p recv[r] hold the jobs exchanged with rank
/// r (both empty unless built under MPI with n_ranks() > 1). The fingerprint (per_x, per_y, domain)
/// identifies the (Periodicity, domain) it was built for; communicator size/rank distinguish a
/// rank-local replay from the process-world schedule. The exact layout is retained once in the plan
/// so replay and begin/end publication can fail closed without copying it on each exchange.
/// The jobs carry only Box2D regions, so the plan is INDEPENDENT of ncomp (the component count is
/// supplied at replay via mf.ncomp() to size buffers); ncomp is intentionally absent from the key.
struct HaloSchedule {
  bool per_x = false;
  bool per_y = false;
  Box2D domain{};
  int communicator_size = 1;
  int communicator_rank = 0;
  // Exact layout identity, captured once with the prepared schedule.  In-flight handles retain
  // this shared plan and compare the live MultiFab against it at publication, avoiding a fresh
  // boxes/ranks copy on every begin/end pair.
  std::vector<Box2D> boxes;
  std::vector<int> ranks;
  int ngrow = 0;
  std::vector<HaloJob> local;
  std::vector<std::vector<HaloJob>> send;  // [rank]; empty unless MPI && n_ranks() > 1
  std::vector<std::vector<HaloJob>> recv;  // [rank]
};

/// Small per-MultiFab cache of halo schedules, one entry per distinct
/// (Periodicity, domain, communicator size/rank). In
/// practice a MultiFab is filled with a single (Periodicity, domain) for its role, so this holds one
/// or two entries; lookup is a short linear scan. Entries are shared_ptr so an in-flight
/// HaloExchange can hold a stable handle to the plan it is replaying even if a later call appends a
/// new entry. The cache LIVES ON the MultiFab (multifab.hpp); it is dropped when the MultiFab is
/// reassigned (e.g. AMR regrid builds a fresh MultiFab and move-assigns it over the slot), which is
/// the only way the layout changes, so a stale schedule can never be served. NOT thread-safe (the
/// fill_boundary path is driven from a single host thread; Kokkos parallelism lives inside for_each).
class HaloScheduleCache {
 public:
  /// Existing schedule for (px, py, dom, communicator rank space), or nullptr if none is cached.
  std::shared_ptr<const HaloSchedule> find(bool px, bool py, const Box2D& dom,
                                           int communicator_size,
                                           int communicator_rank) const {
    for (const auto& s : entries_) {
      if (s->per_x == px && s->per_y == py && s->domain == dom &&
          s->communicator_size == communicator_size &&
          s->communicator_rank == communicator_rank) {
        return s;
      }
    }
    return nullptr;
  }

  /// Reserve publication capacity before building a schedule. publish_prepared() is then noexcept:
  /// a failed build can never leave a partially initialized entry visible to a later replay.
  void reserve_for_append() { entries_.reserve(entries_.size() + 1u); }

  void publish_prepared(std::shared_ptr<HaloSchedule> schedule) noexcept {
    entries_.push_back(std::move(schedule));
  }

  /// Drops every cached schedule, forcing a rebuild on the next fill_boundary. Used by tests to
  /// compare the cached path against a fresh rebuild; not needed in production (regrid drops the
  /// whole cache by reassigning the MultiFab).
  void clear() {
    entries_.clear();
    exchange_pool_.clear();
  }

  /// Number of cached schedules (test/instrumentation hook).
  std::size_t size() const { return entries_.size(); }
  std::size_t exchange_pool_size() const { return exchange_pool_.size(); }

  /// Borrow persistent communication storage for one schedule/component width. Multiple in-flight
  /// exchanges receive distinct leases; blocking repeated fills reuse capacities without allocating.
  std::shared_ptr<HaloExchangeStorage> acquire_exchange(
      const std::shared_ptr<const HaloSchedule>& schedule, int ncomp);

 private:
  std::vector<std::shared_ptr<HaloSchedule>> entries_;
  std::vector<std::shared_ptr<HaloExchangeStorage>> exchange_pool_;
};

namespace detail {
/// Process-wide count of halo-schedule (re)builds. Independent execution lanes may warm their
/// private caches concurrently, so even this instrumentation counter must not introduce a data race.
inline std::atomic<std::int64_t>& halo_schedule_build_counter() {
  static std::atomic<std::int64_t> n{0};
  return n;
}
}  // namespace detail

/// Number of times fill_boundary has BUILT (enumerated) a halo schedule. A reused (cached) schedule
/// does NOT increment it, so a stable layout filled K times reports 1. Test hook for cache
/// engagement; not part of the public numerical API.
inline std::int64_t halo_schedule_build_count() {
  return detail::halo_schedule_build_counter().load(std::memory_order_relaxed);
}

/// Resets the build counter (tests).
inline void reset_halo_schedule_build_count() {
  detail::halo_schedule_build_counter().store(0, std::memory_order_relaxed);
}

}  // namespace pops
