#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <atomic>
#include <cstdint>
#include <limits>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/index.hpp>
#include <pops/mesh/storage/field_view.hpp>

namespace pops::runtime::program {

/// Bytes that regrid_read copied from one hierarchy instance onto the next (DM6). It counts the
/// scratch of every relayout, halo cells included, once per process.
inline std::atomic<std::uint64_t>& relayout_bytes() noexcept {
  static std::atomic<std::uint64_t> bytes{0};
  return bytes;
}

namespace detail {

/// Floor division: the low halo of a target patch has negative indices, and C++ `/` truncates.
POPS_HD inline int regrid_floor_div(int numerator, int denominator) {
  const int quotient = numerator / denominator;
  return (numerator % denominator != 0 && ((numerator < 0) != (denominator < 0))) ? quotient - 1
                                                                                  : quotient;
}

/// target(c) = scratch(floor(c / r)) over the grown target box. The scratch holds, on the
/// coarsening of that box, the values of the source cells that are parents under Parent(r).
template <int Dim>
struct RegridParentKernel {
  FieldView<Real, Dim> target{};
  FieldView<const Real, Dim> scratch{};
  Index<Dim> ratio{};
  int components = 0;

  POPS_HD void operator()(const Index<Dim>& cell) const {
    Index<Dim> parent{};
    for (int axis = 0; axis < Dim; ++axis)
      parent[axis] = regrid_floor_div(cell[axis], ratio[axis]);
    for (int component = 0; component < components; ++component)
      target(cell, component) = scratch(parent, component);
  }
};

/// One component of a field read by regrid_read; CompositeIntegralKernel owns the measure.
template <int Dim>
struct RegridComponentIntegrand {
  FieldView<const Real, Dim> values{};
  int component = 0;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real value = values(index, component);
    return Kokkos::isfinite(value) ? value : std::numeric_limits<Real>::infinity();
  }
};

}  // namespace detail
}  // namespace pops::runtime::program
