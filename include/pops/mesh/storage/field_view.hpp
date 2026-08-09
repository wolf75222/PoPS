/// @file
/// @brief Non-owning compile-time-ranked field descriptor for device kernels.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>

#include <cstdint>

namespace pops {

/// Device-copyable non-owning view.  Axis 0 is contiguous and components are slowest.
template <class T, int Dim>
struct FieldView {
  static_assert(Dim >= 1 && Dim <= 3, "pops::FieldView only supports dimensions 1, 2, and 3");

  static constexpr int rank = Dim;
  T* data{nullptr};
  Index<Dim> origin{};
  Extent<Dim> extents{};
  std::int64_t strides[Dim]{};
  int ncomp{0};
  std::int64_t component_stride{0};

  POPS_HD T& operator()(const Index<Dim>& index, int component = 0) const {
    std::int64_t offset = static_cast<std::int64_t>(component) * component_stride;
    for (int axis = 0; axis < Dim; ++axis)
      offset += (static_cast<std::int64_t>(index[axis]) - origin[axis]) * strides[axis];
    return data[offset];
  }
};

}  // namespace pops
