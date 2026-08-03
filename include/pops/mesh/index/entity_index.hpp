/// @file
/// @brief Typed compile-time-ranked coordinates for cell and face kernels.

#pragma once

#include <pops/mesh/index/index.hpp>

namespace pops {

/// Cell kernels use the canonical signed compile-time-ranked coordinate.
template <int Dim>
using CellIndex = Index<Dim>;

/// Coordinate of a face whose normal axis is selected at compile time.  Keeping Axis in the type
/// lets a numerical functor specialize flux/metric access without a per-face direction branch.
template <int Dim, int Axis>
struct FaceIndex {
  static_assert(Dim >= 1 && Dim <= 3, "pops::FaceIndex only supports dimensions 1, 2, and 3");
  static_assert(Axis >= 0 && Axis < Dim,
                "pops::FaceIndex normal axis must lie inside the compile-time rank");

  static constexpr int rank = Dim;
  static constexpr int normal_axis = Axis;

  Index<Dim> coordinate{};

  POPS_HD constexpr FaceIndex() = default;
  POPS_HD constexpr explicit FaceIndex(Index<Dim> value) : coordinate(value) {}

  POPS_HD constexpr int& operator[](int axis) { return coordinate[axis]; }
  POPS_HD constexpr int operator[](int axis) const { return coordinate[axis]; }

  POPS_HD constexpr bool operator==(const FaceIndex&) const = default;
};

}  // namespace pops
