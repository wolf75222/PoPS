#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/parallel/comm.hpp>  // all_reduce_sum_inplace (distributed multi-patch reflux)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

/// @file
/// @brief Named types of the multi-patch coarse-fine interface: PatchRange (coarse footprint
///        of a fine patch), FluxRegister (GLOBAL-indexed coarse buffer + all_reduce), CoverageMask
///        (cells shadowed by a patch), SubcyclingSchedule (Berger-Oliger cadence) and
///        CoarseFineInterface (coverage + reflux routing), with the multi-box fill/avgdown
///        helpers (mf_fill_fine_ghosts_multi, mf_average_down_multi, fill_periodic_local).
///
/// Layer: `include/pops/numerics/time`.
/// Role: promote to TYPES the roles previously inlined/duplicated in the multi-patch
///        subcycling (amr_subcycling.hpp). Centralization with strictly preserved arithmetic.
///
/// Invariants:
/// - PatchRange uses floor division in global index space, including negative/non-zero origins;
/// - FluxRegister / CoverageMask are built on the GLOBAL box_array (known to all ranks):
///   MPI-safe. Each rank fills its LOCAL contributions (0 elsewhere), gather() sums them via
///   all_reduce_sum_inplace; in serial all_reduce is the identity -> bit-for-bit identical;
/// - CoverageMask prevents double-reflux of a fine-fine joint (only true fine-coarse interfaces
///   are corrected);
/// - AvgDownMultiKernel / route_reflux are NAMED functors/functions (no generic lambda)
///   -> safe under nvcc;
/// - fill_periodic_local serves the REPLICATED coarse (per-rank dmap): purely local folding
///   without an MPI plan, reads valid / writes ghost (no race).

namespace pops {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// PatchRange (review, point 5: role promoted to a type). COARSE footprint [I0..I1]x[J0..J1]
// of a fine patch under ratio 2. Floor division is required for domains whose index origin is
// negative; ordinary C++ integer division would truncate toward zero and route the patch to the
// wrong parent cell.
struct PatchRange {
  int I0, I1, J0, J1;
  explicit PatchRange(const Box2D& fine) {
    if (fine.empty() || fine.coarsen(kAmrRefRatio).refine(kAmrRefRatio) != fine)
      throw std::invalid_argument(
          "PatchRange requires a non-empty fine box aligned to complete ratio-2 parent cells");
    const Box2D parent = fine.coarsen(kAmrRefRatio);
    I0 = parent.lo[0];
    I1 = parent.hi[0];
    J0 = parent.lo[1];
    J1 = parent.hi[1];
  }
  Box2D box() const { return Box2D{{I0, J0}, {I1, J1}}; }  // coarse footprint (cells)
};

/// Validate the structural contract shared by average-down and reflux before either path launches
/// a device kernel.  Ratio-aligned, disjoint fine boxes produce disjoint parent footprints, hence
/// every register/average destination has exactly one writer and needs no atomics.  Performing the
/// complete validation first also guarantees that a later malformed box cannot throw while an
/// earlier kernel still references pinned communication storage.
inline void validate_ratio_aligned_disjoint_fine_layout(
    const BoxArray& fine_boxes, const Box2D* coarse_domain = nullptr) {
  if (coarse_domain != nullptr && coarse_domain->empty())
    throw std::invalid_argument("coarse/fine layout requires a non-empty coarse domain");
  for (int current = 0; current < fine_boxes.size(); ++current) {
    const Box2D footprint = PatchRange(fine_boxes[current]).box();
    if (coarse_domain != nullptr && !coarse_domain->contains(footprint))
      throw std::invalid_argument("fine footprint lies outside the coarse domain");
    for (int previous = 0; previous < current; ++previous) {
      const Box2D previous_footprint = PatchRange(fine_boxes[previous]).box();
      if (!footprint.intersect(previous_footprint).empty())
        throw std::invalid_argument(
            "coarse/fine layout requires disjoint ratio-aligned fine footprints");
    }
  }
}

/// Pinned host storage is simultaneously device-accessible and safe to pass to a non-CUDA-aware
/// MPI.  Reflux strips and collective registers deliberately do not use Fab SharedSpace: an MPI
/// implementation must never classify these pointers as managed/device memory and attempt CUDA IPC
/// between ranks whose GPUs are isolated.  Every kernel touching this storage is fenced before a
/// collective or destruction.
template <class T>
using RefluxStorage = std::vector<T, comm_allocator<T>>;

// multi-box fine ghosts from the coarse (space+time interp), THEN fill_boundary
// (fine-fine) will overwrite the ghosts covered by a neighbor box. coarse mono-box.
inline void mf_fill_fine_ghosts_multi(MultiFab& Uf, const MultiFab& Uc_old, const MultiFab& Uc_new,
                                      const Box2D& coarse_domain, Real frac) {
  const int nc = Uf.ncomp();
  const ConstArray4 co = Uc_old.fab(0).const_array();
  const ConstArray4 cn = Uc_new.fab(0).const_array();
  const Box2D fine_domain = coarse_domain.refine(kAmrRefRatio);
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
    for_each_cell(g, CoarseFineTemporalGhostKernel{f, co, cn, v, coarse_domain, fine_domain, nc,
                                                   frac, Real(0), 0});
  }
}

namespace detail {
// NAMED device-clean functor (extended lambda -> trips nvcc cross-TU): fine -> coarse average
// (ratio 2) of a fine box over the PatchRange coarse footprint. Body bit-identical to the old
// lambda of mf_average_down_multi.
struct AvgDownMultiKernel {
  ConstArray4 f;
  Array4 c;
  int nc;
  POPS_HD void operator()(int I, int J) const {
    for (int k = 0; k < nc; ++k)
      c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                 f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
  }
};
}  // namespace detail

// fine -> coarse average over the footprint of EACH fine box (multi-box).
inline void mf_average_down_multi(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = Uc.ncomp();
  Array4 c = Uc.fab(0).array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    const PatchRange pr(Uf.box(li));
    for_each_cell(pr.box(), detail::AvgDownMultiKernel{f, c, nc});
  }
}

// PURELY LOCAL periodic fill of the ghosts of a mono-box coarse (self-folding).
// Equivalent to a periodic fill_boundary for a single box, but WITHOUT the MPI plan: serves
// the REPLICATED coarse (per-rank copy), whose per-rank DistributionMapping would violate the
// replicated-metadata assumption of fill_boundary. Reads valid cells (indices
// folded into [0,N)) and writes only ghosts: no read/write race.
namespace detail {
struct PeriodicLocalFillKernel {
  Array4 values;
  Box2D domain;
  Periodicity periodicity;
  int components;

  POPS_HD void operator()(int i, int j) const {
    const bool outside_x = i < domain.lo[0] || i > domain.hi[0];
    const bool outside_y = j < domain.lo[1] || j > domain.hi[1];
    if ((!outside_x && !outside_y) || (outside_x && !periodicity.x) ||
        (outside_y && !periodicity.y))
      return;
    const int source_i = outside_x
                             ? domain.lo[0] + (i - domain.lo[0]) -
                                   floor_div(i - domain.lo[0], domain.nx()) * domain.nx()
                             : i;
    const int source_j = outside_y
                             ? domain.lo[1] + (j - domain.lo[1]) -
                                   floor_div(j - domain.lo[1], domain.ny()) * domain.ny()
                             : j;
    for (int component = 0; component < components; ++component)
      values(i, j, component) = values(source_i, source_j, component);
  }
};
}  // namespace detail

inline void fill_periodic_local(MultiFab& mf, const Box2D& dom, Periodicity periodicity) {
  if (dom.empty())
    throw std::invalid_argument("fill_periodic_local requires a non-empty domain");
  for (int li = 0; li < mf.local_size(); ++li) {
    const Box2D grown = mf.fab(li).grown_box();
    for_each_cell(grown, detail::PeriodicLocalFillKernel{mf.fab(li).array(), dom, periodicity,
                                                         mf.ncomp()});
  }
}

namespace detail {

POPS_HD inline std::uint64_t sparse_cell_key(int I, int J) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(I)) << 32) |
         static_cast<std::uint32_t>(J);
}

POPS_HD inline std::uint64_t sparse_cell_hash(std::uint64_t key) {
  // SplitMix64 finalizer: deterministic on every backend and independent of process hash seeds.
  key ^= key >> 30;
  key *= UINT64_C(0xbf58476d1ce4e5b9);
  key ^= key >> 27;
  key *= UINT64_C(0x94d049bb133111eb);
  return key ^ (key >> 31);
}

}  // namespace detail

struct SparseCellLookupView {
  const std::uint64_t* keys = nullptr;
  const std::size_t* values = nullptr;
  std::size_t mask = 0;
  std::size_t maximum_probe = 0;
  std::size_t invalid_value = std::numeric_limits<std::size_t>::max();
  bool populated = false;

  POPS_HD bool locate(int I, int J, std::size_t& value) const {
    if (!populated)
      return false;
    const std::uint64_t key = detail::sparse_cell_key(I, J);
    const std::size_t home = static_cast<std::size_t>(detail::sparse_cell_hash(key)) & mask;
    for (std::size_t probe = 0; probe <= maximum_probe; ++probe) {
      const std::size_t slot = (home + probe) & mask;
      const std::size_t candidate = values[slot];
      if (candidate == invalid_value)
        return false;
      if (keys[slot] == key) {
        value = candidate;
        return true;
      }
    }
    return false;
  }
};

/// Deterministic open-addressed table used by registers and masks.  Capacity is the next power of
/// two above twice the number of populated cells, hence storage remains O(populated cells) even for
/// interfaces separated by an arbitrarily large index-space hole.  There are no deletions, so an
/// empty slot terminates lookup and the prepared maximum probe is a hard device-side bound.
class SparseCellLookup {
 public:
  void reserve(std::size_t entries) {
    const std::size_t requested = capacity_for_(entries);
    if (requested > capacity())
      rehash_(requested);
  }

  void insert(int I, int J, std::size_t value, bool reject_duplicate) {
    if (value == invalid_value_)
      throw std::invalid_argument("sparse cell lookup value collides with its sentinel");
    std::size_t existing = 0;
    if (view().locate(I, J, existing)) {
      if (reject_duplicate)
        throw std::invalid_argument("sparse cell lookup received a duplicate cell");
      return;
    }
    if (capacity() == 0 || size_ >= capacity() / 2)
      rehash_(capacity_for_(size_ + 1));
    insert_key_(detail::sparse_cell_key(I, J), value, reject_duplicate);
  }

  [[nodiscard]] SparseCellLookupView view() const {
    return {keys_.data(), values_.data(), capacity() == 0 ? 0 : capacity() - 1,
            maximum_probe_, invalid_value_, size_ != 0};
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return values_.size(); }

 private:
  static constexpr std::size_t invalid_value_ = std::numeric_limits<std::size_t>::max();

  static std::size_t capacity_for_(std::size_t entries) {
    if (entries == 0)
      return 0;
    if (entries > std::numeric_limits<std::size_t>::max() / 2)
      throw std::overflow_error("sparse cell lookup entry count overflow");
    const std::size_t required = std::max<std::size_t>(2, entries * 2);
    std::size_t result = 1;
    while (result < required) {
      if (result > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("sparse cell lookup capacity overflow");
      result *= 2;
    }
    return result;
  }

  void rehash_(std::size_t new_capacity) {
    if (new_capacity == 0)
      return;
    if ((new_capacity & (new_capacity - 1)) != 0)
      throw std::logic_error("sparse cell lookup capacity must be a power of two");
    RefluxStorage<std::uint64_t> old_keys = std::move(keys_);
    RefluxStorage<std::size_t> old_values = std::move(values_);
    keys_.assign(new_capacity, std::uint64_t(0));
    values_.assign(new_capacity, invalid_value_);
    size_ = 0;
    maximum_probe_ = 0;
    for (std::size_t slot = 0; slot < old_values.size(); ++slot)
      if (old_values[slot] != invalid_value_)
        insert_key_(old_keys[slot], old_values[slot], /*reject_duplicate=*/true);
  }

  void insert_key_(std::uint64_t key, std::size_t value, bool reject_duplicate) {
    const std::size_t table_capacity = capacity();
    if (table_capacity == 0)
      throw std::logic_error("sparse cell lookup insertion requires reserved storage");
    const std::size_t mask = table_capacity - 1;
    const std::size_t home = static_cast<std::size_t>(detail::sparse_cell_hash(key)) & mask;
    for (std::size_t probe = 0; probe < table_capacity; ++probe) {
      const std::size_t slot = (home + probe) & mask;
      if (values_[slot] == invalid_value_) {
        keys_[slot] = key;
        values_[slot] = value;
        ++size_;
        maximum_probe_ = std::max(maximum_probe_, probe);
        return;
      }
      if (keys_[slot] == key) {
        if (reject_duplicate)
          throw std::invalid_argument("sparse cell lookup received a duplicate cell");
        return;
      }
    }
    throw std::overflow_error("sparse cell lookup exhausted its bounded probe table");
  }

  RefluxStorage<std::uint64_t> keys_;
  RefluxStorage<std::size_t> values_;
  std::size_t size_ = 0;
  std::size_t maximum_probe_ = 0;
};

struct FluxRegisterView {
  SparseCellLookupView lookup;
  Real* values = nullptr;
  int components = 0;

  POPS_HD bool locate(int I, int J, std::size_t& offset) const {
    return lookup.locate(I, J, offset);
  }

  POPS_HD bool in(int I, int J) const {
    std::size_t ignored = 0;
    return locate(I, J, ignored);
  }

  POPS_HD void set(int I, int J, int component, Real value) const {
    std::size_t offset = 0;
    if (locate(I, J, offset))
      values[offset + static_cast<std::size_t>(component)] = value;
  }

  POPS_HD void add(int I, int J, int component, Real value) const {
    std::size_t offset = 0;
    if (locate(I, J, offset))
      values[offset + static_cast<std::size_t>(component)] += value;
  }

  POPS_HD Real at(int I, int J, int component) const {
    std::size_t offset = 0;
    return locate(I, J, offset)
               ? values[offset + static_cast<std::size_t>(component)]
               : Real(0);
  }
};

struct FluxRegisterConstView {
  SparseCellLookupView lookup;
  const Real* values = nullptr;
  int components = 0;

  POPS_HD bool locate(int I, int J, std::size_t& offset) const {
    return lookup.locate(I, J, offset);
  }

  POPS_HD bool in(int I, int J) const {
    std::size_t ignored = 0;
    return locate(I, J, ignored);
  }

  POPS_HD Real at(int I, int J, int component) const {
    std::size_t offset = 0;
    return locate(I, J, offset)
               ? values[offset + static_cast<std::size_t>(component)]
               : Real(0);
  }
};

namespace detail {
struct ClearRefluxStorageKernel {
  Real* values;
  POPS_HD void operator()(std::int64_t index) const { values[index] = Real(0); }
};
}  // namespace detail

// FluxRegister (review, point 2: role promoted to a type). Coarse register with GLOBAL indexing
// over a REGION (box, with origin), to lift average_down (overwrite of covered cells, set)
// and reflux (addition to bordering cells, bounded add) across ranks.
// Each rank fills its LOCAL contributions (0 elsewhere), gather() sums them via
// all_reduce_sum_inplace, then each rank reads the total via at(). In serial all_reduce is
// the identity -> bit-for-bit identical. One region preserves the historical contiguous path;
// disjoint compact regions retain the same global index formulas without materialising holes.
struct FluxRegister {
  int I0 = 0, J0 = 0;
  std::int64_t NX = 0, NY = 0;
  int nc = 0;
  std::vector<Box2D> regions;
  std::vector<std::size_t> region_offsets;
  SparseCellLookup cell_lookup;
  RefluxStorage<Real> buf;
  FluxRegister(const Box2D& region, int ncomp)
      : FluxRegister(std::vector<Box2D>{region}, ncomp) {}
  FluxRegister(std::vector<Box2D> compact_regions, int ncomp)
      : I0(0),
        J0(0),
        NX(0),
        NY(0),
        nc(ncomp),
        regions(compact_regions.begin(), compact_regions.end()) {
    if (regions.empty() || nc <= 0)
      throw std::invalid_argument("FluxRegister requires non-empty regions and components");
    Box2D bounds = regions.front();
    std::size_t cells = 0;
    for (std::size_t index = 0; index < regions.size(); ++index) {
      const Box2D& region = regions[index];
      if (region.empty())
        throw std::invalid_argument("FluxRegister region must be non-empty");
      for (std::size_t previous = 0; previous < index; ++previous)
        if (!region.intersect(regions[previous]).empty())
          throw std::invalid_argument("FluxRegister compact regions must be disjoint");
      if (cells > std::numeric_limits<std::size_t>::max() /
                      static_cast<std::size_t>(nc))
        throw std::overflow_error("FluxRegister component offset overflow");
      region_offsets.push_back(cells * static_cast<std::size_t>(nc));
      const std::size_t region_nx = checked_extent_(region, 0);
      const std::size_t region_ny = checked_extent_(region, 1);
      if (region_nx > std::numeric_limits<std::size_t>::max() / region_ny ||
          region_nx * region_ny > std::numeric_limits<std::size_t>::max() - cells)
        throw std::overflow_error("FluxRegister compact region size overflow");
      cells += region_nx * region_ny;
      bounds.lo[0] = std::min(bounds.lo[0], region.lo[0]);
      bounds.lo[1] = std::min(bounds.lo[1], region.lo[1]);
      bounds.hi[0] = std::max(bounds.hi[0], region.hi[0]);
      bounds.hi[1] = std::max(bounds.hi[1], region.hi[1]);
    }
    if (cells > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(nc))
      throw std::overflow_error("FluxRegister component storage size overflow");
    I0 = bounds.lo[0];
    J0 = bounds.lo[1];
    NX = static_cast<std::int64_t>(bounds.hi[0]) - bounds.lo[0] + 1;
    NY = static_cast<std::int64_t>(bounds.hi[1]) - bounds.lo[1] + 1;
    buf.assign(cells * static_cast<std::size_t>(nc), Real(0));
    build_sparse_lookup_(cells);
  }
  FluxRegisterView view() {
    return FluxRegisterView{cell_lookup.view(), buf.data(), nc};
  }
  FluxRegisterConstView view() const {
    return FluxRegisterConstView{cell_lookup.view(), buf.data(), nc};
  }
  void clear_on_device() {
    detail::ensure_kokkos_initialized();
    Kokkos::parallel_for(
        "pops_clear_reflux_register",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(
            0, static_cast<std::int64_t>(buf.size())),
        detail::ClearRefluxStorageKernel{buf.data()});
  }
  std::size_t idx(int I, int J, int k) const {
    if (k < 0 || k >= nc)
      throw std::out_of_range("FluxRegister component is out of range");
    std::size_t offset = 0;
    if (cell_lookup.view().locate(I, J, offset))
      return offset + static_cast<std::size_t>(k);
    throw std::out_of_range("FluxRegister index lies outside compact regions");
  }
  bool in(int I, int J) const {
    std::size_t ignored = 0;
    return cell_lookup.view().locate(I, J, ignored);
  }
  void set(int I, int J, int k, Real v) { buf[idx(I, J, k)] = v; }  // overwrite (average_down)
  void add(int I, int J, int k, Real v) {                           // bordering addition (reflux)
    if (in(I, J))
      buf[idx(I, J, k)] += v;
  }
  Real at(int I, int J, int k) const { return buf[idx(I, J, k)]; }
  void gather() {
    device_fence();
    all_reduce_sum_inplace(buf.data(), buf.size());
  }
  void gather(const CommunicatorView& communicator) {
    device_fence();
    all_reduce_sum_inplace(buf.data(), buf.size(), communicator);
  }
  [[nodiscard]] std::size_t lookup_capacity() const noexcept {
    return cell_lookup.capacity();
  }
  [[nodiscard]] std::size_t covered_cell_count() const noexcept {
    return cell_lookup.size();
  }

 private:
  static std::size_t checked_extent_(const Box2D& region, int axis) {
    if (region.empty())
      return 0;
    const std::int64_t extent = static_cast<std::int64_t>(region.hi[axis]) -
                                static_cast<std::int64_t>(region.lo[axis]) + 1;
    if (extent <= 0 || static_cast<std::uint64_t>(extent) >
                           std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("FluxRegister extent exceeds addressable range");
    return static_cast<std::size_t>(extent);
  }

  void build_sparse_lookup_(std::size_t cells) {
    cell_lookup.reserve(cells);
    for (std::size_t region_index = 0; region_index < regions.size(); ++region_index) {
      const Box2D& region = regions[region_index];
      const std::size_t region_nx = checked_extent_(region, 0);
      for (int J = region.lo[1];;) {
        for (int I = region.lo[0];;) {
          const std::size_t local_cell =
              static_cast<std::size_t>(static_cast<std::int64_t>(J) - region.lo[1]) *
                  region_nx +
              static_cast<std::size_t>(static_cast<std::int64_t>(I) - region.lo[0]);
          cell_lookup.insert(I, J,
                             region_offsets[region_index] +
                                 local_cell * static_cast<std::size_t>(nc),
                             /*reject_duplicate=*/true);
          if (I == region.hi[0])
            break;
          ++I;
        }
        if (J == region.hi[1])
          break;
        ++J;
      }
    }
  }
};

// CoverageMask (review, point 2: "coverage" part of CoarseFineInterface). Coarse mask
// over a REGION saying which cells are SHADOWED by a fine patch. Built on the
// GLOBAL box_array (known to all ranks) -> MPI-safe. mark(box) marks the coarse footprint
// of a patch (intersected with the region); covered(I,J) is bounded (false outside region).
// This is what prevents the double-reflux of a fine-fine joint. Same cells as before.
struct CoverageMaskView {
  SparseCellLookupView lookup;
  POPS_HD bool covered(int I, int J) const {
    std::size_t ignored = 0;
    return lookup.locate(I, J, ignored);
  }
};

struct CoverageMask {
  int I0 = 0, J0 = 0;
  std::int64_t NX = 0, NY = 0;

  explicit CoverageMask(const Box2D& region)
      : I0(region.lo[0]),
        J0(region.lo[1]),
        NX(static_cast<std::int64_t>(region.hi[0]) - region.lo[0] + 1),
        NY(static_cast<std::int64_t>(region.hi[1]) - region.lo[1] + 1),
        region_(region) {
    if (region.empty())
      throw std::invalid_argument("CoverageMask requires a non-empty region");
  }
  void mark(const Box2D& b) {  // marks the cells of b intersected with the region
    const int i0 = std::max(b.lo[0], region_.lo[0]);
    const int i1 = std::min(b.hi[0], region_.hi[0]);
    const int j0 = std::max(b.lo[1], region_.lo[1]);
    const int j1 = std::min(b.hi[1], region_.hi[1]);
    if (i0 > i1 || j0 > j1)
      return;
    for (int J = j0;;) {
      for (int I = i0;;) {
        lookup_.insert(I, J, 0, /*reject_duplicate=*/false);
        if (I == i1)
          break;
        ++I;
      }
      if (J == j1)
        break;
      ++J;
    }
  }
  bool covered(int I, int J) const {
    std::size_t ignored = 0;
    return lookup_.view().locate(I, J, ignored);
  }
  CoverageMaskView view() const { return {lookup_.view()}; }
  [[nodiscard]] std::size_t lookup_capacity() const noexcept { return lookup_.capacity(); }
  [[nodiscard]] std::size_t covered_cell_count() const noexcept { return lookup_.size(); }

 private:
  Box2D region_{};
  SparseCellLookup lookup_;
};

/// POD view over one coarse/fine interface strip.  The owning RegMP/EdgeStrip keeps the pinned
/// buffers alive; kernels capture only these raw pointers and integer bounds.
struct RefluxStripView {
  int I0 = 0, I1 = -1, J0 = 0, J1 = -1;
  Real* cL = nullptr;
  Real* cR = nullptr;
  Real* cB = nullptr;
  Real* cT = nullptr;
  Real* fL = nullptr;
  Real* fR = nullptr;
  Real* fB = nullptr;
  Real* fT = nullptr;
  int components = 0;
};

struct RefluxStripConstView {
  int I0 = 0, I1 = -1, J0 = 0, J1 = -1;
  const Real* cL = nullptr;
  const Real* cR = nullptr;
  const Real* cB = nullptr;
  const Real* cT = nullptr;
  const Real* fL = nullptr;
  const Real* fR = nullptr;
  const Real* fB = nullptr;
  const Real* fT = nullptr;
  int components = 0;
};

template <class Strip>
inline RefluxStripView reflux_strip_view(Strip& strip, int components) {
  return {strip.I0,      strip.I1,      strip.J0,      strip.J1,
          strip.cL.data(), strip.cR.data(), strip.cB.data(), strip.cT.data(),
          strip.fL.data(), strip.fR.data(), strip.fB.data(), strip.fT.data(), components};
}

template <class Strip>
inline RefluxStripConstView reflux_strip_const_view(const Strip& strip, int components) {
  return {strip.I0,      strip.I1,      strip.J0,      strip.J1,
          strip.cL.empty() ? nullptr : strip.cL.data(),
          strip.cR.empty() ? nullptr : strip.cR.data(),
          strip.cB.empty() ? nullptr : strip.cB.data(),
          strip.cT.empty() ? nullptr : strip.cT.data(),
          strip.fL.empty() ? nullptr : strip.fL.data(),
          strip.fR.empty() ? nullptr : strip.fR.data(),
          strip.fB.empty() ? nullptr : strip.fB.data(),
          strip.fT.empty() ? nullptr : strip.fT.data(), components};
}

namespace detail {

struct SampleCoarseXStripKernel {
  ConstArray4 left;
  ConstArray4 right;
  RefluxStripView strip;
  POPS_HD void operator()(int, int J) const {
    const std::size_t base =
        static_cast<std::size_t>(J - strip.J0) * static_cast<std::size_t>(strip.components);
    for (int component = 0; component < strip.components; ++component) {
      strip.cL[base + static_cast<std::size_t>(component)] = left(strip.I0, J, component);
      strip.cR[base + static_cast<std::size_t>(component)] = right(strip.I1 + 1, J, component);
    }
  }
};

struct SampleCoarseYStripKernel {
  ConstArray4 bottom;
  ConstArray4 top;
  RefluxStripView strip;
  POPS_HD void operator()(int I, int) const {
    const std::size_t base =
        static_cast<std::size_t>(I - strip.I0) * static_cast<std::size_t>(strip.components);
    for (int component = 0; component < strip.components; ++component) {
      strip.cB[base + static_cast<std::size_t>(component)] = bottom(I, strip.J0, component);
      strip.cT[base + static_cast<std::size_t>(component)] = top(I, strip.J1 + 1, component);
    }
  }
};

struct AccumulateFineXStripKernel {
  ConstArray4 flux;
  RefluxStripView strip;
  Real scale;
  POPS_HD void operator()(int, int J) const {
    const std::size_t base =
        static_cast<std::size_t>(J - strip.J0) * static_cast<std::size_t>(strip.components);
    for (int component = 0; component < strip.components; ++component) {
      strip.fL[base + static_cast<std::size_t>(component)] +=
          Real(0.5) * (flux(2 * strip.I0, 2 * J, component) +
                       flux(2 * strip.I0, 2 * J + 1, component)) *
          scale;
      strip.fR[base + static_cast<std::size_t>(component)] +=
          Real(0.5) * (flux(2 * strip.I1 + 2, 2 * J, component) +
                       flux(2 * strip.I1 + 2, 2 * J + 1, component)) *
          scale;
    }
  }
};

struct AccumulateFineYStripKernel {
  ConstArray4 flux;
  RefluxStripView strip;
  Real scale;
  POPS_HD void operator()(int I, int) const {
    const std::size_t base =
        static_cast<std::size_t>(I - strip.I0) * static_cast<std::size_t>(strip.components);
    for (int component = 0; component < strip.components; ++component) {
      strip.fB[base + static_cast<std::size_t>(component)] +=
          Real(0.5) * (flux(2 * I, 2 * strip.J0, component) +
                       flux(2 * I + 1, 2 * strip.J0, component)) *
          scale;
      strip.fT[base + static_cast<std::size_t>(component)] +=
          Real(0.5) * (flux(2 * I, 2 * strip.J1 + 2, component) +
                       flux(2 * I + 1, 2 * strip.J1 + 2, component)) *
          scale;
    }
  }
};

struct AverageDownRegisterKernel {
  ConstArray4 fine;
  FluxRegisterView average;
  int components;
  POPS_HD void operator()(int I, int J) const {
    for (int component = 0; component < components; ++component)
      average.set(I, J, component,
                  Real(0.25) * (fine(2 * I, 2 * J, component) +
                                fine(2 * I + 1, 2 * J, component) +
                                fine(2 * I, 2 * J + 1, component) +
                                fine(2 * I + 1, 2 * J + 1, component)));
  }
};

struct ApplyAverageDownRegisterKernel {
  Array4 coarse;
  FluxRegisterView average;
  CoverageMaskView coverage;
  int components;
  POPS_HD void operator()(int I, int J) const {
    if (!coverage.covered(I, J) || !average.in(I, J))
      return;
    for (int component = 0; component < components; ++component)
      coarse(I, J, component) = average.at(I, J, component);
  }
};

struct ApplyRefluxRegisterKernel {
  Array4 coarse;
  FluxRegisterView correction;
  int components;
  POPS_HD void operator()(int I, int J) const {
    if (!correction.in(I, J))
      return;
    for (int component = 0; component < components; ++component)
      coarse(I, J, component) += correction.at(I, J, component);
  }
};

struct ApplyRefluxThenAverageKernel {
  Array4 coarse;
  FluxRegisterView correction;
  FluxRegisterView average;
  CoverageMaskView coverage;
  int components;
  POPS_HD void operator()(int I, int J) const {
    if (!correction.in(I, J))
      return;
    for (int component = 0; component < components; ++component) {
      coarse(I, J, component) += correction.at(I, J, component);
      if (coverage.covered(I, J))
        coarse(I, J, component) = average.at(I, J, component);
    }
  }
};

/// Deterministic one-strip reflux route.  A single device work item walks the four faces in the
/// canonical left/right/bottom/top order, so multiple physical faces targeting one coarse cell are
/// never concurrent within a patch.  Callers enqueue patches in stable local-patch order on the same
/// execution space; the gather fence closes the ordered queue before MPI reads the register.
struct RouteRefluxStripKernel {
  RefluxStripConstView coarse;
  RefluxStripConstView fine;
  FluxRegisterView correction;
  CoverageMaskView coverage;
  Box2D coarse_domain;
  Periodicity periodicity;
  Real inverse_dx;
  Real inverse_dy;
  Real coarse_scale;

  POPS_HD static int wrap_index(int value, int lo, int extent) {
    const std::int64_t relative = static_cast<std::int64_t>(value) - lo;
    std::int64_t quotient = relative / extent;
    if (relative % extent < 0)
      --quotient;
    return static_cast<int>(static_cast<std::int64_t>(lo) + relative - quotient * extent);
  }

  POPS_HD bool canonicalize(int& I, int& J) const {
    if (I < coarse_domain.lo[0] || I > coarse_domain.hi[0]) {
      if (!periodicity.x)
        return false;
      I = wrap_index(I, coarse_domain.lo[0], coarse_domain.nx());
    }
    if (J < coarse_domain.lo[1] || J > coarse_domain.hi[1]) {
      if (!periodicity.y)
        return false;
      J = wrap_index(J, coarse_domain.lo[1], coarse_domain.ny());
    }
    return true;
  }

  POPS_HD Real value(const Real* values, std::size_t index) const {
    return values == nullptr ? Real(0) : values[index];
  }

  POPS_HD void add_if_uncovered(int I, int J, int component, Real amount) const {
    if (!canonicalize(I, J) || coverage.covered(I, J))
      return;
    correction.add(I, J, component, amount);
  }

  POPS_HD void operator()(int, int) const {
    const RefluxStripConstView shape =
        (fine.fL != nullptr || fine.fB != nullptr) ? fine : coarse;
    for (int J = shape.J0; J <= shape.J1; ++J)
      for (int component = 0; component < shape.components; ++component) {
        const std::size_t index = static_cast<std::size_t>(J - shape.J0) *
                                      static_cast<std::size_t>(shape.components) +
                                  static_cast<std::size_t>(component);
        add_if_uncovered(shape.I0 - 1, J, component,
                         -(value(fine.fL, index) - coarse_scale * value(coarse.cL, index)) *
                             inverse_dx);
        add_if_uncovered(shape.I1 + 1, J, component,
                         +(value(fine.fR, index) - coarse_scale * value(coarse.cR, index)) *
                             inverse_dx);
      }
    for (int I = shape.I0; I <= shape.I1; ++I)
      for (int component = 0; component < shape.components; ++component) {
        const std::size_t index = static_cast<std::size_t>(I - shape.I0) *
                                      static_cast<std::size_t>(shape.components) +
                                  static_cast<std::size_t>(component);
        add_if_uncovered(I, shape.J0 - 1, component,
                         -(value(fine.fB, index) - coarse_scale * value(coarse.cB, index)) *
                             inverse_dy);
        add_if_uncovered(I, shape.J1 + 1, component,
                         +(value(fine.fT, index) - coarse_scale * value(coarse.cT, index)) *
                             inverse_dy);
      }
  }
};

}  // namespace detail

inline void sample_coarse_x_strip(const ConstArray4& left, const ConstArray4& right,
                                  RefluxStripView strip, int begin, int end) {
  if (begin > end)
    return;
  for_each_cell(Box2D{{0, begin}, {0, end}},
                detail::SampleCoarseXStripKernel{left, right, strip});
}

inline void sample_coarse_y_strip(const ConstArray4& bottom, const ConstArray4& top,
                                  RefluxStripView strip, int begin, int end) {
  if (begin > end)
    return;
  for_each_cell(Box2D{{begin, 0}, {end, 0}},
                detail::SampleCoarseYStripKernel{bottom, top, strip});
}

inline void sample_coarse_strip(const ConstArray4& left, const ConstArray4& right,
                                const ConstArray4& bottom, const ConstArray4& top,
                                RefluxStripView strip) {
  sample_coarse_x_strip(left, right, strip, strip.J0, strip.J1);
  sample_coarse_y_strip(bottom, top, strip, strip.I0, strip.I1);
}

inline void accumulate_fine_strip(const ConstArray4& Fx, const ConstArray4& Fy,
                                  RefluxStripView strip, Real scale) {
  for_each_cell(Box2D{{0, strip.J0}, {0, strip.J1}},
                detail::AccumulateFineXStripKernel{Fx, strip, scale});
  for_each_cell(Box2D{{strip.I0, 0}, {strip.I1, 0}},
                detail::AccumulateFineYStripKernel{Fy, strip, scale});
}

// SubcyclingSchedule (review, point 5: role promoted to a type). Berger-Oliger cadence of a
// level: temporal refinement ratio r, substep dt/r, and temporal position frac(s)
// = s/r of substep s in the parent step. Centralizes the `const int r = kAmrRefRatio`, `dt / r` and
// `Real(s) / r` scattered across the subcycling loops. Arithmetic strictly preserved:
// dt_sub(dt) == dt / r and frac(s) == Real(s) / r at the same types, thus bit-identical.
struct SubcyclingSchedule {
  amr::ParentChildClockRelation clocks;

  SubcyclingSchedule(int parent_level, int child_level, amr::Rational temporal_ratio,
                     amr::RemainderPolicy remainder_policy)
      : clocks(parent_level, child_level, temporal_ratio, remainder_policy) {
    if (!temporal_ratio.integral())
      throw std::invalid_argument(
          "SubcyclingSchedule scalar loop requires an integral temporal ratio; use the explicit "
          "clock partition for a declared remainder");
  }
  int count() const { return static_cast<int>(clocks.temporal_ratio().numerator); }
  Real dt_sub(Real dt) const { return dt / static_cast<Real>(count()); }
  Real frac(int s) const { return Real(s) / static_cast<Real>(count()); }
};

// CoarseFineInterface (review, point 2). The coarse-fine interface of a level: coverage
// (which coarse cells are shadowed by a fine patch, via CoverageMask) + bordering ROUTING
// of the reflux (which coarse cell borders which fine-patch face, and the conservative
// correction poured into it). Centralizes the two inline logics previously duplicated
// in amr_step_2level_multipatch and subcycle_level_mp. Builds the mask on the GLOBAL
// box_array() of the fine patches (MPI-safe). route_reflux is a template on the register
// type (Reg / RegMP, same field layout): named function (no generic lambda), thus
// safe under nvcc. Arithmetic bit-identical to the previous bodies.
struct CoarseFineInterface {
  CoverageMask cmask;
  Box2D coarse_region;
  Periodicity periodicity;
  // region = coarse footprint of the level (origin (0,0), dims NX x NY); fine_ba = GLOBAL
  // fine patches (all the boxes, known to all ranks). We mark the coarse PatchRange footprint
  // of each patch.
  CoarseFineInterface(const Box2D& region, const BoxArray& fine_ba, Periodicity per)
      : cmask(region), coarse_region(region), periodicity(per) {
    if (region.empty())
      throw std::invalid_argument("CoarseFineInterface requires a non-empty coarse domain");
    if (region.lo[0] == std::numeric_limits<int>::min() ||
        region.lo[1] == std::numeric_limits<int>::min() ||
        region.hi[0] == std::numeric_limits<int>::max() ||
        region.hi[1] == std::numeric_limits<int>::max())
      throw std::overflow_error(
          "CoarseFineInterface domain leaves no integer room for bordering reflux cells");
    validate_ratio_aligned_disjoint_fine_layout(fine_ba, &region);
    for (int g = 0; g < fine_ba.size(); ++g) {
      const Box2D footprint = PatchRange(fine_ba[g]).box();
      cmask.mark(footprint);
    }
  }
  bool canonicalize(int& I, int& J) const {
    const auto wrap = [](int value, int lo, int extent) {
      const std::int64_t relative = static_cast<std::int64_t>(value) - lo;
      const std::int64_t quotient = relative >= 0 ? relative / extent : -((-relative + extent - 1) / extent);
      const std::int64_t wrapped = static_cast<std::int64_t>(lo) + relative - quotient * extent;
      if (wrapped < std::numeric_limits<int>::min() || wrapped > std::numeric_limits<int>::max())
        throw std::overflow_error("coarse/fine periodic index overflow");
      return static_cast<int>(wrapped);
    };
    if (I < coarse_region.lo[0] || I > coarse_region.hi[0]) {
      if (!periodicity.x)
        return false;
      I = wrap(I, coarse_region.lo[0], coarse_region.nx());
    }
    if (J < coarse_region.lo[1] || J > coarse_region.hi[1]) {
      if (!periodicity.y)
        return false;
      J = wrap(J, coarse_region.lo[1], coarse_region.ny());
    }
    return true;
  }
  bool covered(int I, int J) const {
    return canonicalize(I, J) && cmask.covered(I, J);
  }

  std::vector<Box2D> reflux_register_regions(const Box2D& fine_parent_footprint) const {
    const Box2D grown = fine_parent_footprint.grow(1);
    const auto axis_segments = [](int lo, int hi, int domain_lo, int domain_hi, bool periodic) {
      std::vector<std::array<int, 2>> segments;
      if (!periodic) {
        const int clipped_lo = std::max(lo, domain_lo);
        const int clipped_hi = std::min(hi, domain_hi);
        if (clipped_lo <= clipped_hi)
          segments.push_back({clipped_lo, clipped_hi});
        return segments;
      }
      const int extent = domain_hi - domain_lo + 1;
      const std::int64_t width = static_cast<std::int64_t>(hi) - lo + 1;
      if (width >= extent) {
        segments.push_back({domain_lo, domain_hi});
        return segments;
      }
      const std::int64_t relative = static_cast<std::int64_t>(lo) - domain_lo;
      const std::int64_t quotient = relative >= 0 ? relative / extent : -((-relative + extent - 1) / extent);
      const std::int64_t start64 = static_cast<std::int64_t>(domain_lo) + relative - quotient * extent;
      const std::int64_t end64 = start64 + width - 1;
      if (start64 < std::numeric_limits<int>::min() || end64 > std::numeric_limits<int>::max())
        throw std::overflow_error("coarse/fine reflux region index overflow");
      const int start = static_cast<int>(start64);
      const int end = static_cast<int>(end64);
      if (end <= domain_hi)
        segments.push_back({start, end});
      else {
        segments.push_back({start, domain_hi});
        segments.push_back({domain_lo, domain_lo + (end - domain_hi) - 1});
      }
      return segments;
    };
    const auto xs = axis_segments(grown.lo[0], grown.hi[0], coarse_region.lo[0],
                                  coarse_region.hi[0], periodicity.x);
    const auto ys = axis_segments(grown.lo[1], grown.hi[1], coarse_region.lo[1],
                                  coarse_region.hi[1], periodicity.y);
    std::vector<Box2D> regions_out;
    regions_out.reserve(xs.size() * ys.size());
    for (const auto& y : ys)
      for (const auto& x : xs)
        regions_out.push_back(Box2D{{x[0], y[0]}, {x[1], y[1]}});
    return regions_out;
  }

  /// Exact compact union for an arbitrary fine layout.  Building one register from the bounding
  /// box of all patches makes two tiny, distant patches consume storage proportional to the hole
  /// between them.  This cold-path catalogue instead unions each grown footprint, canonicalizes
  /// periodic images through the single-footprint routine above, then compresses cells into
  /// disjoint one-row runs.  FluxRegister storage is therefore O(covered cells), independent of
  /// global index span.
  std::vector<Box2D> reflux_register_regions(const BoxArray& fine_boxes) const {
    validate_ratio_aligned_disjoint_fine_layout(fine_boxes, &coarse_region);
    std::vector<std::pair<int, int>> cells;
    for (const Box2D& fine_box : fine_boxes.boxes()) {
      const Box2D footprint = PatchRange(fine_box).box();
      for (const Box2D& region : reflux_register_regions(footprint)) {
        const std::int64_t count = region.num_cells();
        if (count < 0 || static_cast<std::uint64_t>(count) >
                             std::numeric_limits<std::size_t>::max() - cells.size())
          throw std::overflow_error("coarse/fine reflux catalogue size overflow");
        cells.reserve(cells.size() + static_cast<std::size_t>(count));
        for (int j = region.lo[1];;) {
          for (int i = region.lo[0];;) {
            cells.emplace_back(j, i);
            if (i == region.hi[0])
              break;
            ++i;
          }
          if (j == region.hi[1])
            break;
          ++j;
        }
      }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    if (cells.empty())
      throw std::invalid_argument("coarse/fine reflux catalogue is empty");
    std::vector<Box2D> compact;
    compact.reserve(cells.size());
    std::size_t begin = 0;
    while (begin < cells.size()) {
      const int row = cells[begin].first;
      int lo = cells[begin].second;
      int hi = lo;
      std::size_t end = begin + 1;
      while (end < cells.size() && cells[end].first == row &&
             static_cast<std::int64_t>(cells[end].second) ==
                 static_cast<std::int64_t>(hi) + 1) {
        hi = cells[end].second;
        ++end;
      }
      compact.push_back(Box2D{{lo, row}, {hi, row}});
      begin = end;
    }
    return compact;
  }

  template <class Value>
  void add_if_uncovered(FluxRegister& ref, int I, int J, int component, Value correction) const {
    if (!canonicalize(I, J) || cmask.covered(I, J))
      return;
    ref.add(I, J, component, static_cast<Real>(correction));
  }

  // Pours the reflux correction of ONE fine patch (register g, parent coords) into the
  // coarse register ref: on each BORDERING coarse cell not covered by another patch,
  // (time-integrated fine flux - coarse flux x dt) / dx|dy. Same formulas, same order
  // (left/right in x, bottom/top in y) as the original inline bodies.
  template <class Reg>
  void route_reflux(const Reg& g, Real dx, Real dy, Real dt, FluxRegister& ref, int nc) const {
    validate_route_inputs_(g, g, dx, dy, dt, nc, /*allow_empty_roles=*/false);
    const RefluxStripConstView strip = reflux_strip_const_view(g, nc);
    for_each_cell(Box2D{{0, 0}, {0, 0}},
                  detail::RouteRefluxStripKernel{strip, strip, ref.view(), cmask.view(),
                                                 coarse_region, periodicity, Real(1) / dx,
                                                 Real(1) / dy, dt});
  }

  // ADC-639 variant of route_reflux for the compiled-Program driver: BOTH the coarse side (g.c*) and the
  // fine side (g.f*) are ALREADY dt-integrated (the effective-flux ledger carries dt through the Program's
  // linear combination -- g.cL = dt*Feff_coarse, g.fL = dt*Feff_fine), so the correction is
  // -(g.fL - g.cL)/dx with NO *dt. Dropping the /dt*dt round-trip keeps the coarse-fine cancellation exact
  // to round-off (a *dt then implicit /dt would re-introduce a rounding step). Same coverage guard, same
  // face order (left/right in x, bottom/top in y), same FluxRegister.add as route_reflux -- only the *dt is
  // gone. @c Reg is EdgeStrip / RegMP-shaped (I0..J1 + the eight flat strip arrays).
  template <class Reg>
  void route_reflux_integrated(const Reg& g, Real dx, Real dy, FluxRegister& ref, int nc) const {
    validate_route_inputs_(g, g, dx, dy, Real(1), nc, /*allow_empty_roles=*/false);
    const RefluxStripConstView strip = reflux_strip_const_view(g, nc);
    for_each_cell(Box2D{{0, 0}, {0, 0}},
                  detail::RouteRefluxStripKernel{strip, strip, ref.view(), cmask.view(),
                                                 coarse_region, periodicity, Real(1) / dx,
                                                 Real(1) / dy, Real(1)});
  }

  /// Allocation-free Program variant: the coarse and fine roles remain in their independently
  /// owned strips instead of being copied into a temporary merged strip at every synchronization.
  /// Missing role buffers mean an exact zero contribution.  If both roles exist, their qualified
  /// patch footprint must agree; a topology/ledger mismatch is never coerced.
  template <class Reg>
  void route_reflux_integrated_pair(const Reg& coarse, const Reg& fine, Real dx, Real dy,
                                    FluxRegister& ref, int nc) const {
    const bool coarse_present = coarse.I1 >= coarse.I0 && coarse.J1 >= coarse.J0 &&
                                (!coarse.cL.empty() || !coarse.cB.empty());
    const bool fine_present = fine.I1 >= fine.I0 && fine.J1 >= fine.J0 &&
                              (!fine.fL.empty() || !fine.fB.empty());
    if (!coarse_present && !fine_present)
      return;
    const Reg& shape = fine_present ? fine : coarse;
    if (coarse_present && fine_present &&
        (coarse.I0 != fine.I0 || coarse.I1 != fine.I1 || coarse.J0 != fine.J0 ||
         coarse.J1 != fine.J1))
      throw std::runtime_error("coarse/fine reflux strips have different patch footprints");
    validate_route_inputs_(coarse, fine, dx, dy, Real(1), nc,
                           /*allow_empty_roles=*/true);
    for_each_cell(
        Box2D{{0, 0}, {0, 0}},
        detail::RouteRefluxStripKernel{
            reflux_strip_const_view(coarse, nc), reflux_strip_const_view(fine, nc), ref.view(),
            cmask.view(), coarse_region, periodicity, Real(1) / dx, Real(1) / dy, Real(1)});
  }

  template <class Reg>
  static void validate_route_inputs_(const Reg& coarse, const Reg& fine, Real dx, Real dy,
                                     Real coarse_scale, int nc, bool allow_empty_roles) {
    if (nc <= 0 || !std::isfinite(dx) || !std::isfinite(dy) || dx <= Real(0) ||
        dy <= Real(0) || !std::isfinite(coarse_scale))
      throw std::invalid_argument("reflux route requires finite positive spacing and components");
    const Reg& shape = (!fine.fL.empty() || !fine.fB.empty()) ? fine : coarse;
    if (shape.I1 < shape.I0 || shape.J1 < shape.J0)
      throw std::invalid_argument("reflux strip has an empty footprint");
    const std::size_t nJ = static_cast<std::size_t>(shape.J1 - shape.J0 + 1) *
                           static_cast<std::size_t>(nc);
    const std::size_t nI = static_cast<std::size_t>(shape.I1 - shape.I0 + 1) *
                           static_cast<std::size_t>(nc);
    const bool coarse_present = !coarse.cL.empty() || !coarse.cR.empty() ||
                                !coarse.cB.empty() || !coarse.cT.empty();
    const bool fine_present = !fine.fL.empty() || !fine.fR.empty() || !fine.fB.empty() ||
                              !fine.fT.empty();
    const bool coarse_exact = coarse.cL.size() == nJ && coarse.cR.size() == nJ &&
                              coarse.cB.size() == nI && coarse.cT.size() == nI;
    const bool fine_exact = fine.fL.size() == nJ && fine.fR.size() == nJ &&
                            fine.fB.size() == nI && fine.fT.size() == nI;
    if ((!allow_empty_roles && (!coarse_exact || !fine_exact)) ||
        (allow_empty_roles && ((coarse_present && !coarse_exact) ||
                               (fine_present && !fine_exact))))
      throw std::runtime_error("coarse/fine reflux strip width disagrees with its footprint");
  }
};

}  // namespace pops
