/// @file
/// @brief Fab2D: single-grid data on a Box2D (in-house equivalent of AMReX's FArrayBox);
///        Array4 / ConstArray4: lightweight POD device-copyable handles over its buffer.
///
/// Contiguous buffer covering the VALID box grown by n_ghost layers, with n_comp components.
/// Component-SLOW layout (like AMReX Array4): for a given component the (i, j) plane is
/// contiguous with i as the fast index -> each variable is a contiguous SoA slice (good
/// per-variable vectorization). Index: c*(nx_tot*ny_tot) + (j-jg0)*nx_tot + (i-ig0). Array4 /
/// ConstArray4 are lightweight handles (raw pointer + strides), trivially copyable and
/// capturable BY VALUE in a functor (Kokkos view semantics): you capture the handle, not the
/// Fab; operator() is POPS_HD (device-callable). Storage lives in UNIFIED memory (cf.
/// allocator.hpp).

#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/foundation/validation.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box2d.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pops {

/// WRITE POD handle (raw pointer + strides) over a Fab2D buffer, indexed by (i, j, c)
/// IN GLOBAL INDICES (ig0/jg0 = lower corner of the grown box). Trivially copyable, capturable by
/// value in a device kernel. INVARIANT: owns NOTHING; valid as long as the source Fab is.
struct Array4 {
  Real* p{nullptr};
  int nx_tot{0};
  std::int64_t comp_stride{0};
  int ig0{0}, jg0{0};  // global indices of the lower corner of the grown box

  /// Reference to cell (i, j) of component c (global indices). POPS_HD. No bounds checking
  /// (hot path / device): the caller guarantees (i, j, c) is inside the grown box.
  POPS_HD Real& operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride +
             (static_cast<std::int64_t>(j) - static_cast<std::int64_t>(jg0)) * nx_tot +
             (static_cast<std::int64_t>(i) - static_cast<std::int64_t>(ig0))];
  }
};

/// READ-only handle (const counterpart of Array4): same layout and same contract (POD
/// device-copyable, global indices, no bounds checking). POPS_HD.
struct ConstArray4 {
  const Real* p{nullptr};
  int nx_tot{0};
  std::int64_t comp_stride{0};
  int ig0{0}, jg0{0};

  /// Value of cell (i, j) of component c (global indices). POPS_HD, no bounds checking.
  POPS_HD Real operator()(int i, int j, int c = 0) const {
    return p[c * comp_stride +
             (static_cast<std::int64_t>(j) - static_cast<std::int64_t>(jg0)) * nx_tot +
             (static_cast<std::int64_t>(i) - static_cast<std::int64_t>(ig0))];
  }
};

namespace detail {

/// Fill every component of one Fab cell.  The same named functor is used by Fab2D and MultiFab so
/// Kokkos Serial, OpenMP and device builds all take the canonical execution seam.  In particular,
/// a CUDA build never faults every SharedSpace page back to the host merely to clear a scratch Fab.
struct SetFabValueKernel {
  Array4 values;
  int components;
  Real value;

  POPS_HD void operator()(int i, int j) const {
    for (int component = 0; component < components; ++component)
      values(i, j, component) = value;
  }
};

}  // namespace detail

/// Single-grid data on a Box2D: VALID box + ng ghost layers, ncomp components, component-slow
/// layout. OWNS its buffer (unified memory). Exposes Array4 / ConstArray4 handles to kernels
/// (capture by value), never the Fab itself.
class Fab2D {
 public:
  Fab2D() = default;

  /// Allocates the valid box grown by ng ghosts, ncomp components, initialized to 0.
  Fab2D(const Box2D& valid, int ncomp, int ng) : valid_(valid) {
    if (ncomp < 1)
      throw_validation_error("pops/mesh/storage/fab2d.hpp: Fab2D",
                             "ncomp >= 1 for a field component layout",
                             "ncomp=" + std::to_string(ncomp));
    if (ng < 0)
      throw_validation_error("pops/mesh/storage/fab2d.hpp: Fab2D", "ghost width ng >= 0",
                             "ng=" + std::to_string(ng));
    ng_ = ng;
    ncomp_ = ncomp;
    gbox_ = valid.grow(ng);
    if (gbox_.empty())
      return;

    const std::int64_t width = gbox_.length64(0);
    const std::int64_t height = gbox_.length64(1);
    if (width > std::numeric_limits<int>::max() || height > std::numeric_limits<int>::max())
      throw_validation_error("pops/mesh/storage/fab2d.hpp: Fab2D",
                             "each grown-box extent representable by the native signed-int kernel "
                             "index type",
                             "grown_box=" + box_bounds(gbox_));

    // All PoPS cell kernels currently form half-open upper bounds as hi + 1 and several legacy
    // host traversals increment a signed-int cell index.  A Box2D remains able to describe a cell
    // at INT_MAX, but materializing such a box would make those iteration contracts overflow.
    // Refuse before allocation or kernel launch instead of claiming partial support.
    if (gbox_.hi[0] == std::numeric_limits<int>::max() ||
        gbox_.hi[1] == std::numeric_limits<int>::max())
      throw_validation_error("pops/mesh/storage/fab2d.hpp: Fab2D",
                             "grown-box upper bounds <= INT_MAX - 1 for native iteration",
                             "grown_box=" + box_bounds(gbox_));

    nx_tot_ = static_cast<int>(width);
    ny_tot_ = static_cast<int>(height);
    const std::int64_t cells = width * height;  // both factors are positive and <= INT_MAX
    if (cells > std::numeric_limits<std::int64_t>::max() / ncomp_)
      throw_validation_error(
          "pops/mesh/storage/fab2d.hpp: Fab2D", "ncomp * grown-box cells representable by int64_t",
          "ncomp=" + std::to_string(ncomp_) + ", grown_box=" + box_bounds(gbox_));
    const std::int64_t elements = cells * ncomp_;
    if (static_cast<std::uint64_t>(elements) > static_cast<std::uint64_t>(data_.max_size()))
      throw_validation_error("pops/mesh/storage/fab2d.hpp: Fab2D",
                             "allocation element count <= allocator max_size",
                             "elements=" + std::to_string(elements) +
                                 ", allocator.max_size=" + std::to_string(data_.max_size()));
    data_.assign(static_cast<std::size_t>(elements), Real{0});
  }

  /// VALID box (without ghosts).
  const Box2D& box() const { return valid_; }
  /// Grown box (valid + ng ghosts) = actual memory footprint.
  const Box2D& grown_box() const { return gbox_; }
  /// Number of components.
  int ncomp() const { return ncomp_; }
  /// Number of ghost layers.
  int n_ghost() const { return ng_; }
  /// Buffer size (nx_tot * ny_tot * ncomp).
  std::int64_t size() const { return static_cast<std::int64_t>(data_.size()); }

  /// HOST write access (i, j, c) (bounds assert in debug). Do not call inside a device kernel:
  /// go through array() (POD handle).
  Real& operator()(int i, int j, int c = 0) { return data_[idx(i, j, c)]; }
  /// HOST read access (i, j, c) (bounds assert in debug).
  Real operator()(int i, int j, int c = 0) const { return data_[idx(i, j, c)]; }

  /// WRITE handle (POD device-copyable) over this Fab. Valid as long as the Fab lives.
  Array4 array() {
    return Array4{data_.data(), nx_tot_, static_cast<std::int64_t>(nx_tot_) * ny_tot_, gbox_.lo[0],
                  gbox_.lo[1]};
  }
  /// READ handle (POD device-copyable) over this Fab. Valid as long as the Fab lives.
  ConstArray4 const_array() const {
    return ConstArray4{data_.data(), nx_tot_, static_cast<std::int64_t>(nx_tot_) * ny_tot_,
                       gbox_.lo[0], gbox_.lo[1]};
  }

  /// Raw pointer to the buffer (passed directly to MPI in unified memory, for instance).
  Real* data() { return data_.data(); }
  const Real* data() const { return data_.data(); }
  /// Fills the whole buffer (valid + ghosts) with value v through the canonical Kokkos execution
  /// seam.  Completion is synchronous, preserving the historical host-observable contract of this
  /// low-level operation; MultiFab batches every local Fab behind one final fence instead.
  void set_val(Real v) {
    if (gbox_.empty())
      return;
    for_each_cell(gbox_, detail::SetFabValueKernel{array(), ncomp_, v});
    device_fence();
  }

 private:
  static std::string box_bounds(const Box2D& b) {
    return "[" + std::to_string(b.lo[0]) + ".." + std::to_string(b.hi[0]) + "]x[" +
           std::to_string(b.lo[1]) + ".." + std::to_string(b.hi[1]) + "]";
  }

  // linear index (i, j, c) in the component-slow layout; release-active bounds validation.
  std::int64_t idx(int i, int j, int c) const {
    if (!gbox_.contains(i, j) || c < 0 || c >= ncomp_)
      throw_validation_error("pops/mesh/storage/fab2d.hpp: Fab2D::operator()",
                             "cell index inside grown box " + box_bounds(gbox_) +
                                 " and component in [0.." + std::to_string(ncomp_ - 1) + "]",
                             "i=" + std::to_string(i) + ", j=" + std::to_string(j) +
                                 ", component=" + std::to_string(c));
    return c * static_cast<std::int64_t>(nx_tot_) * ny_tot_ +
           (static_cast<std::int64_t>(j) - static_cast<std::int64_t>(gbox_.lo[1])) * nx_tot_ +
           (static_cast<std::int64_t>(i) - static_cast<std::int64_t>(gbox_.lo[0]));
  }

  Box2D valid_{};
  int ng_{0};
  int ncomp_{1};
  Box2D gbox_{};
  int nx_tot_{0}, ny_tot_{0};
  // storage: host (std::allocator) or CUDA unified memory (cf. allocator.hpp).
  std::vector<Real, fab_allocator<Real>> data_{};
};

}  // namespace pops
