/// @file
/// @brief CopySchedule: memoized inter-layout redistribution plan for parallel_copy (ADC-607).
///
/// parallel_copy used to enumerate, on EVERY call, the src-box job schedule: a BoxHash build over
/// the SRC BoxArray plus a local (and, under MPI, a global) enumeration over the two BoxArrays. That
/// schedule is a pure function of the LAYOUT PAIR (dst BoxArray/DistributionMapping, src
/// BoxArray/DistributionMapping); only the copy/pack/MPI/unpack of the LIVE data must rerun. This
/// header holds the cacheable plan so the enumeration runs ONCE per (dst layout, src layout). Unlike
/// the intra-level halo schedule (halo_schedule.hpp, ADC-260) the plan depends on BOTH layouts, so
/// the cache LIVES ON the dst MultiFab (auto-dropped when regrid move-assigns a fresh dst) but each
/// entry is KEYED on a SRC-LAYOUT fingerprint (src BoxArray + src DistributionMapping): a fresh src
/// serves a fresh entry, never a stale one. Jobs carry GLOBAL box indices (resolved to local fabs at
/// replay), so a plan is valid for any MultiFab pair over the same layouts. MPI-free: the in-flight
/// buffers and MPI_Request stay in parallel_copy (refinement.hpp). Jobs replay in the SAME
/// deterministic order as the legacy inline enumeration (local: dst-local x sorted src candidates;
/// global: gd x sorted src candidates) so the packed buffers stay bit-identical.

#pragma once

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace pops {

struct CopyExchangeStorage;

/// One redistribution copy: the overlap @p region of dst box @p gd is filled from the SAME-index
/// region of src box @p gs (no shift: parallel_copy is a same-domain redistribution). @p gs and @p
/// gd are GLOBAL box indices (into the src / dst BoxArray respectively), resolved to local fabs when
/// the job is replayed.
struct CopyJob {
  int gs = 0;
  int gd = 0;
  Box2D region{};
};

/// A src-layout fingerprint: the src BoxArray boxes AND the src DistributionMapping ranks. The
/// schedule depends on both (which src box overlaps which dst box, and who owns each), so both are
/// part of the key. Compared by exact vector equality (not a hash) so a fingerprint collision can
/// never serve a wrong plan; the vectors are small (one entry per src box). Component width and
/// communicator identity live on CopySchedule itself so changing either cannot reuse communication
/// preparation from a different native contract.
struct SrcLayoutKey {
  std::vector<Box2D> boxes;  // src BoxArray boxes (order significant, = global index)
  std::vector<int> ranks;    // src DistributionMapping owner-rank per box

  bool matches(const BoxArray& sba, const DistributionMapping& sdm) const {
    return boxes == sba.boxes() && ranks == sdm.ranks();
  }
};

/// Memoized schedule for ONE (dst layout, src layout) pair. @p local holds the copies whose dst AND
/// src are owned by this rank; @p send[r]/@p recv[r] hold the jobs exchanged with rank r (both empty
/// unless built under MPI with n_ranks() > 1). @p key fingerprints the SRC layout it was built for;
/// the DST layout is implicit because the plan is stored on the dst MultiFab that owns the dst
/// BoxArray/DistributionMapping (see CopyScheduleCache). Although the geometry jobs do not depend
/// on ncomp, the cache identity deliberately does because its prepared buffers do.
struct CopySchedule {
  SrcLayoutKey key;
  int ncomp = 0;
  int communicator_size = 1;
  int communicator_rank = 0;
  std::int64_t communicator_identity = 0;
  int message_tag = 0;
  std::vector<CopyJob> local;
  std::vector<std::vector<CopyJob>> send;  // [rank]; empty unless MPI && n_ranks() > 1
  std::vector<std::vector<CopyJob>> recv;  // [rank]
  std::vector<std::int64_t> send_cells;    // [rank], before multiplication by ncomp
  std::vector<std::int64_t> recv_cells;    // [rank], before multiplication by ncomp
};

/// Small per-MultiFab cache of copy schedules, one entry per distinct SRC layout. In practice a dst
/// MultiFab is the target of parallel_copy from a handful of distinct src layouts (the fine-coarsen
/// grid, the replicated coarse), so this holds a few entries; lookup is a short linear scan.
/// Entries are shared_ptr so an in-flight copy can hold a stable handle to the plan it is replaying
/// even if a later call appends a new entry. The cache LIVES ON the dst MultiFab (multifab.hpp); it
/// is dropped when the dst MultiFab is reassigned (e.g. AMR regrid builds a fresh dst and
/// move-assigns it over the slot), which is the only way the DST layout changes, so a stale
/// dst-layout plan can never be served; a changed SRC layout is caught by the fingerprint key. NOT
/// thread-safe (the parallel_copy path is driven from a single host thread).
class CopyScheduleCache {
 public:
  /// Existing schedule whose SRC-layout fingerprint matches (sba, sdm), or nullptr if none is
  /// cached yet.
  std::shared_ptr<const CopySchedule> find(const BoxArray& sba, const DistributionMapping& sdm,
                                           int ncomp, int communicator_size,
                                           int communicator_rank,
                                           std::int64_t communicator_identity,
                                           int message_tag) const {
    for (const auto& s : entries_) {
      if (s->key.matches(sba, sdm) && s->ncomp == ncomp &&
          s->communicator_size == communicator_size &&
          s->communicator_rank == communicator_rank &&
          s->communicator_identity == communicator_identity &&
          s->message_tag == message_tag) {
        return s;
      }
    }
    return nullptr;
  }

  /// Reserve publication capacity during collective preparation. Once this succeeds on every
  /// rank, publish_prepared() cannot allocate and the freshly built plan becomes visible
  /// transactionally only after the common success witness.
  void reserve_for_append() { entries_.reserve(entries_.size() + 1u); }

  void publish_prepared(std::shared_ptr<CopySchedule> schedule) noexcept {
    entries_.push_back(std::move(schedule));
  }

  /// Drops every cached schedule, forcing a rebuild on the next parallel_copy. Used by tests to
  /// compare the cached path against a fresh rebuild; not needed in production (regrid drops the
  /// whole cache by reassigning the dst MultiFab).
  void clear() {
    entries_.clear();
    exchange_pool_.clear();
  }

  /// Number of cached schedules (test/instrumentation hook).
  std::size_t size() const { return entries_.size(); }
  std::size_t exchange_pool_size() const { return exchange_pool_.size(); }

  /// Borrow communication storage prepared for one exact schedule, provider width, and MPI
  /// communicator context. Concurrent calls receive distinct leases; sequential calls on a stable
  /// layout/lane reuse pinned buffer and MPI_Request capacity.
  std::shared_ptr<CopyExchangeStorage> acquire_exchange(
      const std::shared_ptr<const CopySchedule>& schedule, int ncomp,
      std::int64_t communicator_identity);

 private:
  std::vector<std::shared_ptr<CopySchedule>> entries_;
  std::vector<std::shared_ptr<CopyExchangeStorage>> exchange_pool_;
};

namespace detail {
/// Process-wide counters are atomic because independent execution lanes may warm private caches
/// concurrently. They remain instrumentation only and use relaxed ordering.
inline std::atomic<std::int64_t>& copy_schedule_build_counter() {
  static std::atomic<std::int64_t> n{0};
  return n;
}
/// Process-wide count of copy-schedule cache HITS (a parallel_copy that reused a memoized plan).
inline std::atomic<std::int64_t>& copy_schedule_hit_counter() {
  static std::atomic<std::int64_t> n{0};
  return n;
}
/// Process-wide count of copy-schedule cache MISSES (a parallel_copy that built a fresh plan). Equal
/// to the build counter by construction (a miss always builds); kept as a separate name so the
/// hit/miss pair reads symmetrically at the profiler seam.
inline std::atomic<std::int64_t>& copy_schedule_miss_counter() {
  static std::atomic<std::int64_t> n{0};
  return n;
}
}  // namespace detail

/// Number of times parallel_copy has BUILT (enumerated) a copy schedule. A reused (cached) schedule
/// does NOT increment it, so a stable layout pair copied K times reports 1. Test hook for cache
/// engagement; not part of the public numerical API.
inline std::int64_t copy_schedule_build_count() {
  return detail::copy_schedule_build_counter().load(std::memory_order_relaxed);
}

/// Number of parallel_copy calls served from the cache (hits) and rebuilt (misses). A stable layout
/// pair copied K times reports 1 miss and K-1 hits. Test / profiler hooks.
inline std::int64_t copy_schedule_hit_count() {
  return detail::copy_schedule_hit_counter().load(std::memory_order_relaxed);
}
inline std::int64_t copy_schedule_miss_count() {
  return detail::copy_schedule_miss_counter().load(std::memory_order_relaxed);
}

/// Resets the build / hit / miss counters (tests).
inline void reset_copy_schedule_build_count() {
  detail::copy_schedule_build_counter().store(0, std::memory_order_relaxed);
  detail::copy_schedule_hit_counter().store(0, std::memory_order_relaxed);
  detail::copy_schedule_miss_counter().store(0, std::memory_order_relaxed);
}

}  // namespace pops
