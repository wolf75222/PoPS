/// @file
/// @brief MultiFab: a field DISTRIBUTED over a level (equivalent of AMReX's MultiFab).
///
/// Carries the decomposition (BoxArray), the distribution (DistributionMapping), the component and
/// ghost counts, and allocates only the Fab2D OWNED by this rank. This is where data parallelism
/// lives; the physics layer never sees it. Iteration runs over the LOCAL fabs:
/// for (int li = 0; li < mf.local_size(); ++li) { auto a = mf.fab(li).array(); for_each_cell(...); }.
/// sync_host()/sync_device() encode the access intent (data residence, see for_each.hpp); under
/// unified memory sync_host = a targeted device_fence(), sync_device = no-op. sum() reduces over all
/// ranks (all_reduce): Kokkos::Sum reassociates per tile (deterministic/idempotent, not bit-identical
/// to a lexicographic sum).

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/foundation/validation.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>      // device_fence, sync_host, sync_device
#include <pops/mesh/boundary/halo_schedule.hpp>  // memoized fill_boundary schedule (ADC-260)
#include <pops/mesh/layout/copy_schedule.hpp>    // memoized parallel_copy schedule (ADC-607)
#include <pops/parallel/comm.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace pops {

/// Field distributed over a level: decomposition (BoxArray) + distribution (DistributionMapping) +
/// ncomp components + ngrow ghosts. Allocates only the fabs owned by THIS rank; iteration runs over
/// local_size() (LOCAL indices). global_index/local_index_of bridge local <-> global.
class MultiFab {
 public:
  MultiFab() = default;

  /// Builds the field: allocates one Fab2D (ncomp components, ngrow ghosts) for EACH box that this
  /// rank owns according to dm. Boxes belonging to other ranks are not allocated here.
  MultiFab(BoxArray ba, DistributionMapping dm, int ncomp, int ngrow)
      : ba_(std::move(ba)),
        dm_(std::move(dm)),
        ncomp_(ncomp),
        ngrow_(ngrow),
        local_index_(ba_.size(), -1) {
    validate_layout();
    const int me = my_rank();
    for (int i = 0; i < ba_.size(); ++i) {
      if (dm_[i] == me) {
        local_index_[i] = static_cast<int>(fabs_.size());
        global_of_local_.push_back(i);
        fabs_.emplace_back(ba_[i], ncomp_, ngrow_);
      }
    }
  }

  /// GLOBAL decomposition of the level (all boxes, all ranks).
  const BoxArray& box_array() const { return ba_; }
  /// GLOBAL distribution (owner rank per box).
  const DistributionMapping& dmap() const { return dm_; }
  /// Number of components.
  int ncomp() const { return ncomp_; }
  /// Number of ghost layers.
  int n_grow() const { return ngrow_; }

  /// Number of fabs OWNED by this rank (bound on local indices).
  int local_size() const { return static_cast<int>(fabs_.size()); }
  /// Local fab at index li (0 <= li < local_size()), for writing.
  Fab2D& fab(int li) {
    validate_local_index(li, "MultiFab::fab");
    return fabs_[li];
  }
  /// Local fab at index li, for reading.
  const Fab2D& fab(int li) const {
    validate_local_index(li, "MultiFab::fab const");
    return fabs_[li];
  }
  /// VALID box of local fab li.
  const Box2D& box(int li) const {
    validate_local_index(li, "MultiFab::box");
    return fabs_[li].box();
  }
  /// GLOBAL index (in box_array) of local fab li.
  int global_index(int li) const {
    validate_local_index(li, "MultiFab::global_index");
    return global_of_local_[li];
  }
  /// LOCAL index of the global box @p global, or -1 if it is not owned by this rank.
  int local_index_of(int global) const {
    if (global < 0 || global >= static_cast<int>(local_index_.size()))
      throw_validation_error("pops/mesh/storage/multifab.hpp: MultiFab::local_index_of",
                             "global box index in [0.." +
                                 std::to_string(static_cast<int>(local_index_.size()) - 1) + "]",
                             "global=" + std::to_string(global));
    return local_index_[global];
  }

  /// Makes the HOST residence valid (before a host access: operator(), loop, set_val). Under unified
  /// memory = a targeted device_fence().
  void sync_host() const { pops::sync_host(); }
  /// Marks a DEVICE residence (before a kernel). No-op under unified memory.
  void sync_device() { pops::sync_device(); }

  /// Fills all cells (valid + ghosts) of every local fab with v. Synchronizes host residence first
  /// (a kernel may have written these fabs).
  void set_val(Real v) {
    sync_host();  // a kernel may have written these fabs; make the host residence
                  // valid before the host fill (otherwise a host/kernel write
                  // race). Under unified memory = a device_fence().
    for (auto& f : fabs_)
      f.set_val(v);
  }

  /// Internal (ADC-260): memoized halo-exchange schedule used by fill_boundary. Lazily created on
  /// first use. The schedule is a pure function of (box_array, dmap, n_grow) for a given
  /// (Periodicity, domain); since none of ba_/dm_/ngrow_ has an in-place setter, the cache can only
  /// go stale through whole-object (re)assignment (e.g. AMR regrid builds a fresh MultiFab and
  /// move-assigns it over the level slot), which drops the cache with the object. It is shared on
  /// copy (a copy has the same layout), which keeps copies consistent. Not part of the public
  /// numerical API. Returned by reference so fill_boundary can populate it.
  HaloScheduleCache& halo_cache() const {
    if (!halo_cache_)
      halo_cache_ = std::make_shared<HaloScheduleCache>();
    return *halo_cache_;
  }

  /// Share the immutable-layout halo plan and its reusable communication-buffer pool. Prepared
  /// solver work vectors have the same box/distribution/ghost layout and execute exchanges
  /// sequentially, so one warmed cache removes lazy schedule and MPI-buffer allocation from the
  /// iteration without introducing a global registry.
  void share_halo_cache_from(const MultiFab& prototype) const {
    if (ba_.boxes() != prototype.ba_.boxes() || dm_.ranks() != prototype.dm_.ranks() ||
        ngrow_ != prototype.ngrow_)
      throw_validation_error("pops/mesh/storage/multifab.hpp: share_halo_cache_from",
                             "identical box, distribution, and ghost layout", "layout mismatch");
    prototype.halo_cache();
    halo_cache_ = prototype.halo_cache_;
  }

  /// Internal (ADC-607): memoized redistribution schedule used by parallel_copy when THIS MultiFab
  /// is the DESTINATION. Lazily created on first use. Unlike halo_cache_ the schedule depends on the
  /// SRC layout too, so each entry is keyed on a src-layout fingerprint (src BoxArray +
  /// DistributionMapping); the DST layout is implicit (this fab's ba_/dm_). Since none of ba_/dm_ has
  /// an in-place setter, the cache can only go stale for the DST through whole-object (re)assignment
  /// (e.g. AMR regrid builds a fresh dst and move-assigns it), which drops the cache with the object;
  /// a changed SRC is caught by the fingerprint. Shared on copy (a copy has the same dst layout, and
  /// the src-fingerprint keys still discriminate). Not part of the public numerical API. Returned by
  /// reference so parallel_copy can populate it.
  CopyScheduleCache& copy_cache() const {
    if (!copy_cache_)
      copy_cache_ = std::make_shared<CopyScheduleCache>();
    return *copy_cache_;
  }

 private:
  BoxArray ba_{};
  DistributionMapping dm_{};
  int ncomp_{1};
  int ngrow_{0};
  std::vector<Fab2D> fabs_{};           // locally owned fabs
  std::vector<int> local_index_{};      // global box -> local index (-1 otherwise)
  std::vector<int> global_of_local_{};  // local index -> global box
  // Memoized fill_boundary schedule (ADC-260). mutable: caching is logically const; lazily built.
  mutable std::shared_ptr<HaloScheduleCache> halo_cache_{};
  // Memoized parallel_copy schedule (ADC-607), keyed per src layout. mutable: caching is logically
  // const; lazily built. This MultiFab is the DST; the SRC layout rides in the entry key.
  mutable std::shared_ptr<CopyScheduleCache> copy_cache_{};

  void validate_layout() const {
    if (ncomp_ < 1)
      throw_validation_error("pops/mesh/storage/multifab.hpp: MultiFab",
                             "ncomp >= 1 for every allocated Fab2D",
                             "ncomp=" + std::to_string(ncomp_));
    if (ngrow_ < 0)
      throw_validation_error("pops/mesh/storage/multifab.hpp: MultiFab", "ghost width ngrow >= 0",
                             "ngrow=" + std::to_string(ngrow_));
    if (dm_.size() != ba_.size())
      throw_validation_error("pops/mesh/storage/multifab.hpp: MultiFab",
                             "DistributionMapping size equals BoxArray size",
                             "box_array.size=" + std::to_string(ba_.size()) +
                                 ", dmap.size=" + std::to_string(dm_.size()));
    const int nr = n_ranks();
    const std::vector<int>& ranks = dm_.ranks();
    for (int i = 0; i < static_cast<int>(ranks.size()); ++i) {
      if (ranks[static_cast<std::size_t>(i)] < 0 || ranks[static_cast<std::size_t>(i)] >= nr)
        throw_validation_error("pops/mesh/storage/multifab.hpp: MultiFab",
                               "owner rank in [0.." + std::to_string(nr - 1) + "] for every box",
                               "box=" + std::to_string(i) +
                                   ", owner=" + std::to_string(ranks[static_cast<std::size_t>(i)]) +
                                   ", n_ranks=" + std::to_string(nr));
    }
  }

  void validate_local_index(int li, const char* op) const {
    if (li < 0 || li >= static_cast<int>(fabs_.size()))
      throw_validation_error(
          "pops/mesh/storage/multifab.hpp: " + std::string(op),
          "local index in [0.." + std::to_string(static_cast<int>(fabs_.size()) - 1) + "]",
          "li=" + std::to_string(li) + ", local_size=" + std::to_string(fabs_.size()));
  }
};

/// Sum of the VALID cells of component comp, reduced over ALL ranks (all_reduce). COLLECTIVE under
/// MPI. FP NOTE: Kokkos::Sum reassociates per tile (deterministic/idempotent, not bit-identical to a
/// lexicographic sum).
inline Real sum(const MultiFab& mf, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    s += for_each_cell_reduce_sum(mf.box(li),
                                  [a, comp] POPS_HD(int i, int j) { return a(i, j, comp); });
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

}  // namespace pops
