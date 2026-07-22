/// @file
/// @brief Box2D: the integer index space of a 2D cell-centered Cartesian grid.
///
/// Building block of the AMR stack, inspired by AMReX's Box. Corners lo / hi INCLUSIVE (AMReX
/// convention); box EMPTY if hi < lo along a direction. Pure integer arithmetic: no data, no
/// parallelism, fully testable. Indices may be NEGATIVE (ghost layers), hence the FLOOR division
/// in coarsen (consistent on both sides of zero). length/nx/ny are POPS_HD (called from
/// Geometry::dx()/dy() inside a device kernel). Concrete 2D to match the physical targets: 2D is an
/// official, introspectable invariant of the core (pops.capabilities()["dimension"] == 2, ADR-0001
/// Decision 1; see docs/sphinx/reference/known-limitations.md), not an oversight. The move to a
/// Dim-template (BoxND) is a generalization deferred to a future milestone (Option B).

#pragma once

#include <pops/core/foundation/types.hpp>  // POPS_HD: nx/ny/length called from Geometry::dx() inside a device kernel

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Abort.hpp>
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pops {

namespace detail {

/// A Box2D is deliberately an aggregate, so callers may construct every interval representable by
/// two signed-int corners.  Operations returning an int still have to reject an exact extent which
/// does not fit that legacy result type.  Host code gets a catchable error; a device call cannot
/// throw and therefore terminates the kernel instead of narrowing or continuing with corrupt bounds.
[[noreturn]] POPS_HD inline void box2d_integer_failure(const char* message) {
#if defined(POPS_HAS_KOKKOS)
  KOKKOS_IF_ON_DEVICE((Kokkos::abort(message);))
  KOKKOS_IF_ON_HOST((throw std::overflow_error(message);))
#else
  throw std::overflow_error(message);
#endif
}

[[noreturn]] POPS_HD inline void box2d_domain_failure(const char* message) {
#if defined(POPS_HAS_KOKKOS)
  KOKKOS_IF_ON_DEVICE((Kokkos::abort(message);))
  KOKKOS_IF_ON_HOST((throw std::invalid_argument(message);))
#else
  throw std::invalid_argument(message);
#endif
}

POPS_HD inline int checked_box2d_int(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    box2d_integer_failure(operation);
  return static_cast<int>(value);
}

inline int checked_box2d_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline void require_box2d_direction(int direction, const char* operation) {
  if (direction < 0 || direction > 1)
    throw std::invalid_argument(operation);
}

inline void require_positive_box2d_ratio(int ratio, const char* operation) {
  if (ratio <= 0)
    throw std::invalid_argument(operation);
}

}  // namespace detail

// Integer division rounded down (toward -inf), consistent on both sides of zero: the only correct
// division for NEGATIVE indices (ghost layers) during coarsen / spatial hash. C++ division truncates
// toward zero; we subtract 1 when the remainder is non-zero and of opposite sign to the divisor (the
// truncated quotient was then rounded up). Shared low-level building block (Box2D coarsen, BoxHash bin
// hashing, coarse->fine indices in refinement.hpp).
/// Integer division of a by b rounded down (handles a < 0 AND b < 0). POPS_HD (kernels).
/// A zero divisor and the sole non-representable int quotient (INT_MIN / -1) fail closed.
POPS_HD inline int floor_div(int a, int b) {
  if (b == 0)
    detail::box2d_domain_failure("pops::floor_div: divisor must be non-zero");
  if (a == std::numeric_limits<int>::min() && b == -1)
    detail::box2d_integer_failure(
        "pops::floor_div: quotient is outside the signed-int index range");
  const int q = a / b;
  const int rem = a % b;
  return (rem != 0 && ((rem < 0) != (b < 0))) ? q - 1 : q;
}

/// 2D integer index space, cell-centered. Corners lo/hi INCLUSIVE; box empty if hi < lo.
/// Pure POD (no field data): trivially copyable, capturable by value inside a kernel.
/// INVARIANT: indices may be negative (ghosts); refine/coarsen are block-wise bijections
/// (refine then coarsen gives back the box, but coarsen then refine rounds it to the block).
struct Box2D {
  int lo[2]{0, 0};
  int hi[2]{-1, -1};  // empty by default (hi < lo)

  /// Box [0, nx-1] x [0, ny-1] covering nx*ny cells from the index origin.
  /// A zero extent creates an empty box; negative extents are invalid.
  static Box2D from_extents(int nx, int ny) {
    if (nx < 0 || ny < 0)
      throw std::invalid_argument("pops::Box2D::from_extents: extents must be non-negative");
    return Box2D{{0, 0},
                 {detail::checked_box2d_index(static_cast<std::int64_t>(nx) - 1,
                                              "pops::Box2D::from_extents: x extent overflow"),
                  detail::checked_box2d_index(static_cast<std::int64_t>(ny) - 1,
                                              "pops::Box2D::from_extents: y extent overflow")}};
  }

  /// Exact signed extent in direction d. Unlike length(), all differences between int corners fit.
  POPS_HD std::int64_t length64(int d) const {
    if (d < 0 || d > 1)
      detail::box2d_domain_failure("pops::Box2D::length64: direction must be 0 or 1");
    return static_cast<std::int64_t>(hi[d]) - static_cast<std::int64_t>(lo[d]) + 1;
  }

  // POPS_HD: Geometry::dx()/dy() (themselves POPS_HD) read domain.nx()/ny(); a device kernel that
  // calls geom.x_cell(i) descends down to here. Without POPS_HD this is a __host__ from __device__ ->
  // nvcc yields GARBAGE (often 0) with no error. Pure integer arithmetic, device-safe, host unchanged.
  /// Number of cells in direction d (= hi[d] - lo[d] + 1); negative if the box is empty. POPS_HD.
  /// Fails if the exact result cannot be represented by the historical signed-int return type.
  POPS_HD int length(int d) const {
    return detail::checked_box2d_int(length64(d),
                                     "pops::Box2D::length: extent is outside the signed-int range");
  }
  /// Width (direction 0). POPS_HD (called from Geometry::dx() in a device kernel).
  POPS_HD int nx() const { return length(0); }
  /// Height (direction 1). POPS_HD (called from Geometry::dy() in a device kernel).
  POPS_HD int ny() const { return length(1); }
  /// Total number of cells (nx*ny, floored at 0 per direction): 0 if the box is empty.
  std::int64_t num_cells() const {
    const std::int64_t width = length64(0);
    const std::int64_t height = length64(1);
    if (width <= 0 || height <= 0)
      return 0;
    if (width > std::numeric_limits<std::int64_t>::max() / height)
      throw std::overflow_error("pops::Box2D::num_cells: cell count exceeds int64_t");
    return width * height;
  }
  /// true if the box contains no cell (hi < lo in one direction).
  bool empty() const { return hi[0] < lo[0] || hi[1] < lo[1]; }

  /// true if cell (i, j) is inside the box (lo/hi bounds inclusive).
  bool contains(int i, int j) const { return i >= lo[0] && i <= hi[0] && j >= lo[1] && j <= hi[1]; }
  /// true if box b (non-empty) is entirely contained in *this.
  bool contains(const Box2D& b) const {
    return !b.empty() && b.lo[0] >= lo[0] && b.hi[0] <= hi[0] && b.lo[1] >= lo[1] &&
           b.hi[1] <= hi[1];
  }

  /// Grows the box by n cells in ALL directions (uniform ghost layer).
  Box2D grow(int n) const {
    if (empty())
      return *this;
    return {{detail::checked_box2d_index(static_cast<std::int64_t>(lo[0]) - n,
                                         "pops::Box2D::grow: x lower bound overflow"),
             detail::checked_box2d_index(static_cast<std::int64_t>(lo[1]) - n,
                                         "pops::Box2D::grow: y lower bound overflow")},
            {detail::checked_box2d_index(static_cast<std::int64_t>(hi[0]) + n,
                                         "pops::Box2D::grow: x upper bound overflow"),
             detail::checked_box2d_index(static_cast<std::int64_t>(hi[1]) + n,
                                         "pops::Box2D::grow: y upper bound overflow")}};
  }
  /// Grows by n cells in the SINGLE direction d (n may be negative to shrink).
  Box2D grow(int d, int n) const {
    detail::require_box2d_direction(d, "pops::Box2D::grow: direction must be 0 or 1");
    if (empty())
      return *this;
    Box2D b = *this;
    b.lo[d] = detail::checked_box2d_index(static_cast<std::int64_t>(b.lo[d]) - n,
                                          "pops::Box2D::grow: lower bound overflow");
    b.hi[d] = detail::checked_box2d_index(static_cast<std::int64_t>(b.hi[d]) + n,
                                          "pops::Box2D::grow: upper bound overflow");
    return b;
  }
  /// Translates the box by s cells in direction d (lo and hi shifted by the same s).
  Box2D shift(int d, int s) const {
    detail::require_box2d_direction(d, "pops::Box2D::shift: direction must be 0 or 1");
    if (empty())
      return *this;
    Box2D b = *this;
    b.lo[d] = detail::checked_box2d_index(static_cast<std::int64_t>(b.lo[d]) + s,
                                          "pops::Box2D::shift: lower bound overflow");
    b.hi[d] = detail::checked_box2d_index(static_cast<std::int64_t>(b.hi[d]) + s,
                                          "pops::Box2D::shift: upper bound overflow");
    return b;
  }

  /// Refines by a ratio r: each cell becomes an r x r block ([lo, hi] -> [lo*r, hi*r + r-1]).
  Box2D refine(int r) const {
    detail::require_positive_box2d_ratio(r, "pops::Box2D::refine: ratio must be strictly positive");
    if (empty())
      return *this;
    return {{detail::checked_box2d_index(static_cast<std::int64_t>(lo[0]) * r,
                                         "pops::Box2D::refine: x lower bound overflow"),
             detail::checked_box2d_index(static_cast<std::int64_t>(lo[1]) * r,
                                         "pops::Box2D::refine: y lower bound overflow")},
            {detail::checked_box2d_index(static_cast<std::int64_t>(hi[0]) * r + r - 1,
                                         "pops::Box2D::refine: x upper bound overflow"),
             detail::checked_box2d_index(static_cast<std::int64_t>(hi[1]) * r + r - 1,
                                         "pops::Box2D::refine: y upper bound overflow")}};
  }
  /// Coarsens by a ratio r via FLOOR division of each corner (handles the negative ghost indices).
  Box2D coarsen(int r) const {
    detail::require_positive_box2d_ratio(r,
                                         "pops::Box2D::coarsen: ratio must be strictly positive");
    if (empty())
      return *this;
    return {{floor_div(lo[0], r), floor_div(lo[1], r)}, {floor_div(hi[0], r), floor_div(hi[1], r)}};
  }

  /// Intersection of the two boxes (possibly empty: hi < lo if they do not overlap).
  Box2D intersect(const Box2D& o) const {
    return {{std::max(lo[0], o.lo[0]), std::max(lo[1], o.lo[1])},
            {std::min(hi[0], o.hi[0]), std::min(hi[1], o.hi[1])}};
  }

  bool operator==(const Box2D&) const = default;
};

}  // namespace pops
