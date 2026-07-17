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

#include <cstdint>
#include <memory>
#include <vector>

namespace pops {

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
/// never serve a wrong plan; the vectors are small (one entry per src box). @p nc is deliberately
/// ABSENT: the region jobs are geometry-only and the component count is supplied at replay (via
/// min(dst.ncomp, src.ncomp)), exactly like HaloSchedule.
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
/// BoxArray/DistributionMapping (see CopyScheduleCache). The jobs carry only Box2D regions, so the
/// plan is INDEPENDENT of ncomp (the component count is supplied at replay).
struct CopySchedule {
  SrcLayoutKey key;
  std::vector<CopyJob> local;
  std::vector<std::vector<CopyJob>> send;  // [rank]; empty unless MPI && n_ranks() > 1
  std::vector<std::vector<CopyJob>> recv;  // [rank]
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
  std::shared_ptr<const CopySchedule> find(const BoxArray& sba,
                                           const DistributionMapping& sdm) const {
    for (const auto& s : entries_) {
      if (s->key.matches(sba, sdm)) {
        return s;
      }
    }
    return nullptr;
  }

  /// Appends a fresh, empty schedule and returns it for the caller to populate.
  std::shared_ptr<CopySchedule> add() {
    entries_.push_back(std::make_shared<CopySchedule>());
    return entries_.back();
  }

  /// Drops every cached schedule, forcing a rebuild on the next parallel_copy. Used by tests to
  /// compare the cached path against a fresh rebuild; not needed in production (regrid drops the
  /// whole cache by reassigning the dst MultiFab).
  void clear() { entries_.clear(); }

  /// Number of cached schedules (test/instrumentation hook).
  std::size_t size() const { return entries_.size(); }

 private:
  std::vector<std::shared_ptr<CopySchedule>> entries_;
};

namespace detail {
/// Process-wide count of copy-schedule (re)builds. A single instance across translation units (an
/// inline function with a function-local static). NOT thread-safe; instrumentation only.
inline std::int64_t& copy_schedule_build_counter() {
  static std::int64_t n = 0;
  return n;
}
/// Process-wide count of copy-schedule cache HITS (a parallel_copy that reused a memoized plan).
inline std::int64_t& copy_schedule_hit_counter() {
  static std::int64_t n = 0;
  return n;
}
/// Process-wide count of copy-schedule cache MISSES (a parallel_copy that built a fresh plan). Equal
/// to the build counter by construction (a miss always builds); kept as a separate name so the
/// hit/miss pair reads symmetrically at the profiler seam.
inline std::int64_t& copy_schedule_miss_counter() {
  static std::int64_t n = 0;
  return n;
}
}  // namespace detail

/// Number of times parallel_copy has BUILT (enumerated) a copy schedule. A reused (cached) schedule
/// does NOT increment it, so a stable layout pair copied K times reports 1. Test hook for cache
/// engagement; not part of the public numerical API.
inline std::int64_t copy_schedule_build_count() {
  return detail::copy_schedule_build_counter();
}

/// Number of parallel_copy calls served from the cache (hits) and rebuilt (misses). A stable layout
/// pair copied K times reports 1 miss and K-1 hits. Test / profiler hooks.
inline std::int64_t copy_schedule_hit_count() {
  return detail::copy_schedule_hit_counter();
}
inline std::int64_t copy_schedule_miss_count() {
  return detail::copy_schedule_miss_counter();
}

/// Resets the build / hit / miss counters (tests).
inline void reset_copy_schedule_build_count() {
  detail::copy_schedule_build_counter() = 0;
  detail::copy_schedule_hit_counter() = 0;
  detail::copy_schedule_miss_counter() = 0;
}

}  // namespace pops
