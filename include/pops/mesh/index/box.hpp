/// @file
/// @brief Compile-time-ranked inclusive integer index boxes.

#pragma once

#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/index/index.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pops {

namespace detail {

inline int checked_box_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline int floor_div_index(int numerator, int denominator) {
  if (denominator <= 0)
    throw std::invalid_argument("pops::Box::coarsen: ratio must be strictly positive");
  if (numerator == std::numeric_limits<int>::min() && denominator == -1)
    throw std::overflow_error("pops::Box::coarsen: quotient is outside the signed index range");
  const int quotient = numerator / denominator;
  const int remainder = numerator % denominator;
  return remainder < 0 ? quotient - 1 : quotient;
}

}  // namespace detail

/// Inclusive integer box over a compile-time spatial rank.  A box is empty when any upper bound
/// is below its lower bound; empty boxes are preserved by geometric transforms.
template <int Dim>
struct Box {
  static_assert(Dim >= 1 && Dim <= 3, "pops::Box only supports dimensions 1, 2, and 3");

  static constexpr int rank = Dim;
  Index<Dim> lo;
  Index<Dim> hi;

  POPS_HD constexpr Box() : lo{}, hi{} {
    for (int axis = 0; axis < Dim; ++axis)
      hi[axis] = -1;
  }

  POPS_HD constexpr Box(Index<Dim> lower, Index<Dim> upper) : lo(lower), hi(upper) {}

  /// Box covering the half-open extent [0, extents) with inclusive upper bounds.
  static Box from_extents(const Extent<Dim>& extents) {
    Box result;
    for (int axis = 0; axis < Dim; ++axis) {
      if (extents[axis] < 0)
        throw std::invalid_argument("pops::Box::from_extents: extents must be non-negative");
      result.lo[axis] = 0;
      result.hi[axis] = detail::checked_box_index(
          extents[axis] - 1, "pops::Box::from_extents: extent exceeds signed index range");
    }
    return result;
  }

  POPS_HD constexpr bool empty() const {
    for (int axis = 0; axis < Dim; ++axis)
      if (hi[axis] < lo[axis])
        return true;
    return false;
  }

  /// Exact extent along an axis; empty boxes report zero along every axis.
  POPS_HD constexpr std::int64_t length(int axis) const {
    return empty() ? 0 : static_cast<std::int64_t>(hi[axis]) - lo[axis] + 1;
  }

  POPS_HD constexpr Extent<Dim> extent() const {
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = length(axis);
    return result;
  }

  /// Number of points with host-side overflow detection.
  std::int64_t numPts() const {
    if (empty())
      return 0;
    std::int64_t count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t axis_extent = length(axis);
      if (count > std::numeric_limits<std::int64_t>::max() / axis_extent)
        throw std::overflow_error("pops::Box::numPts: point count exceeds int64_t");
      count *= axis_extent;
    }
    return count;
  }

  POPS_HD constexpr bool contains(const Index<Dim>& index) const {
    if (empty())
      return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (index[axis] < lo[axis] || index[axis] > hi[axis])
        return false;
    return true;
  }

  POPS_HD constexpr bool contains(const Box& other) const {
    if (other.empty())
      return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (other.lo[axis] < lo[axis] || other.hi[axis] > hi[axis])
        return false;
    return true;
  }

  POPS_HD constexpr Box intersect(const Box& other) const {
    Box result{};
    for (int axis = 0; axis < Dim; ++axis) {
      result.lo[axis] = lo[axis] < other.lo[axis] ? other.lo[axis] : lo[axis];
      result.hi[axis] = hi[axis] < other.hi[axis] ? hi[axis] : other.hi[axis];
    }
    return result;
  }

  Box grow(int amount) const {
    if (empty())
      return *this;
    Box result = *this;
    for (int axis = 0; axis < Dim; ++axis) {
      result.lo[axis] = detail::checked_box_index(static_cast<std::int64_t>(lo[axis]) - amount,
                                                  "pops::Box::grow: lower bound overflow");
      result.hi[axis] = detail::checked_box_index(static_cast<std::int64_t>(hi[axis]) + amount,
                                                  "pops::Box::grow: upper bound overflow");
    }
    return result;
  }

  Box grow(int axis, int amount) const {
    if (axis < 0 || axis >= Dim)
      throw std::invalid_argument("pops::Box::grow: axis is outside the compile-time rank");
    if (empty())
      return *this;
    Box result = *this;
    result.lo[axis] = detail::checked_box_index(static_cast<std::int64_t>(lo[axis]) - amount,
                                                "pops::Box::grow: lower bound overflow");
    result.hi[axis] = detail::checked_box_index(static_cast<std::int64_t>(hi[axis]) + amount,
                                                "pops::Box::grow: upper bound overflow");
    return result;
  }

  /// Checked host-side translation used by future periodic-image construction.
  Box shift(const Index<Dim>& offset) const {
    if (empty())
      return *this;
    Box result = *this;
    for (int axis = 0; axis < Dim; ++axis) {
      result.lo[axis] =
          detail::checked_box_index(static_cast<std::int64_t>(lo[axis]) + offset[axis],
                                    "pops::Box::shift: lower bound overflow");
      result.hi[axis] =
          detail::checked_box_index(static_cast<std::int64_t>(hi[axis]) + offset[axis],
                                    "pops::Box::shift: upper bound overflow");
    }
    return result;
  }

  Box refine(int ratio) const {
    if (ratio <= 0)
      throw std::invalid_argument("pops::Box::refine: ratio must be strictly positive");
    if (empty())
      return *this;
    Box result{};
    for (int axis = 0; axis < Dim; ++axis) {
      result.lo[axis] = detail::checked_box_index(static_cast<std::int64_t>(lo[axis]) * ratio,
                                                  "pops::Box::refine: lower bound overflow");
      result.hi[axis] =
          detail::checked_box_index(static_cast<std::int64_t>(hi[axis]) * ratio + ratio - 1,
                                    "pops::Box::refine: upper bound overflow");
    }
    return result;
  }

  Box coarsen(int ratio) const {
    if (ratio <= 0)
      throw std::invalid_argument("pops::Box::coarsen: ratio must be strictly positive");
    if (empty())
      return *this;
    Box result{};
    for (int axis = 0; axis < Dim; ++axis) {
      result.lo[axis] = detail::floor_div_index(lo[axis], ratio);
      result.hi[axis] = detail::floor_div_index(hi[axis], ratio);
    }
    return result;
  }

  POPS_HD constexpr bool operator==(const Box&) const = default;
};

}  // namespace pops
