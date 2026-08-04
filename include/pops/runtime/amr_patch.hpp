/// @file
/// @brief Exact ranked index-space footprint of one materialized AMR patch.

#pragma once

#include <pops/mesh/index/box.hpp>

namespace pops {

/// Read-only topology projection used by checkpointing, visualization, and ownership reports.
/// The box carries inclusive native bounds in axis order; no axis is encoded as a named scalar.
template <int Dim>
struct AmrPatch {
  static_assert(Dim >= 1 && Dim <= 3, "AmrPatch only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  int level = 0;
  Box<Dim> box{};

  bool operator==(const AmrPatch&) const = default;
};

}  // namespace pops
