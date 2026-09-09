/// Generic tensor-product support reduction / constant extension for Transfer ABI v1.
#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace pops::component {

/// Resolved numerical data. Axis indices refer to the actual native storage views.
/// Weights include measure and authored moment factors, so signed weights are valid.
struct PhysicalSupportTransfer {
  int dimension = 0;
  int operation = 0;
  int source_to_target[3] = {-1, -1, -1};
  int source_active[3] = {0, 0, 0};
  int target_active[3] = {0, 0, 0};
  std::size_t reduction_cells[3] = {0, 0, 0};
  std::size_t weight_offsets[3] = {0, 0, 0};
  const double* weights = nullptr;
  std::size_t weight_count = 0;
};

inline int apply_physical_support_transfer(const PhysicalSupportTransfer& map,
                                           const PopsTransferRequestV1* request,
                                           PopsComponentStatusV1* status) {
  if (!request || !status || request->struct_size < sizeof(PopsTransferRequestV1) ||
      map.dimension < 1 || map.dimension > 3 || request->dimension != map.dimension ||
      request->operation != map.operation || (map.operation != 2 && map.operation != 3))
    return 2;
  const auto s = request->source;
  const auto d = request->destination;
  if (s.struct_size < sizeof(PopsConstFieldViewV1) || d.struct_size < sizeof(PopsFieldViewV1) ||
      s.dimension != map.dimension || d.dimension != map.dimension || !s.data || !d.data ||
      s.memory_space != POPS_MEMORY_SPACE_HOST_V1 || d.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      s.scalar_type != POPS_SCALAR_FLOAT64_V1 || d.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
      !s.component_count || s.component_count != d.component_count || s.component_stride <= 0 ||
      d.component_stride <= 0)
    return 2;
  const auto offset_fits = [](const auto& view, int dimension) {
    const auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    std::size_t maximum = 0;
    const auto add_extent = [&](std::size_t extent, std::ptrdiff_t stride) {
      if (!extent || stride <= 0 ||
          extent - 1 > (limit - maximum) / static_cast<std::size_t>(stride))
        return false;
      maximum += (extent - 1) * static_cast<std::size_t>(stride);
      return true;
    };
    if (!add_extent(view.component_count, view.component_stride))
      return false;
    for (int axis = 0; axis < dimension; ++axis)
      if (!add_extent(view.extents[axis], view.axis_strides[axis]))
        return false;
    return true;
  };
  if (!offset_fits(s, map.dimension) || !offset_fits(d, map.dimension))
    return 3;
  std::size_t cells = 1, reductions = 1;
  bool used[3] = {false, false, false};
  bool changes_support = false;
  for (int axis = 0; axis < map.dimension; ++axis) {
    if (!s.extents[axis] || !d.extents[axis] || s.axis_strides[axis] <= 0 ||
        d.axis_strides[axis] <= 0 || s.ghost_lower[axis] || s.ghost_upper[axis] ||
        d.ghost_lower[axis] || d.ghost_upper[axis] ||
        (map.source_active[axis] != 0 && map.source_active[axis] != 1) ||
        (map.target_active[axis] != 0 && map.target_active[axis] != 1))
      return 3;
    if (cells > std::numeric_limits<std::size_t>::max() / d.extents[axis])
      return 3;
    cells *= d.extents[axis];
    const int destination_axis = map.source_to_target[axis];
    if (destination_axis >= 0) {
      if (destination_axis >= map.dimension || used[destination_axis] || !map.source_active[axis] ||
          !map.target_active[destination_axis] || s.extents[axis] != d.extents[destination_axis] ||
          map.reduction_cells[axis])
        return 3;
      used[destination_axis] = true;
    } else {
      if (destination_axis != -1)
        return 3;
      if (map.source_active[axis]) {
        if (map.operation != 2 || map.reduction_cells[axis] != s.extents[axis] || !map.weights ||
            map.weight_offsets[axis] > map.weight_count ||
            s.extents[axis] > map.weight_count - map.weight_offsets[axis])
          return 3;
        if (reductions > std::numeric_limits<std::size_t>::max() / s.extents[axis])
          return 3;
        reductions *= s.extents[axis];
        changes_support = true;
      } else if (s.extents[axis] != 1 || map.reduction_cells[axis])
        return 3;
    }
    if (!map.target_active[axis] && d.extents[axis] != 1)
      return 3;
  }
  for (int axis = 0; axis < map.dimension; ++axis) {
    if (map.target_active[axis] && !used[axis]) {
      if (map.operation != 3)
        return 3;
      changes_support = true;
    }
  }
  if (!changes_support || cells > std::numeric_limits<std::size_t>::max() / d.component_count)
    return 3;
  for (std::size_t i = 0; i < map.weight_count; ++i)
    if (!map.weights || !std::isfinite(map.weights[i]))
      return 3;
  const auto* source = static_cast<const double*>(s.data);
  auto* destination = static_cast<double*>(d.data);
  const std::size_t count = cells * d.component_count;
  using Policy =
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::IndexType<std::size_t>>;
  Kokkos::parallel_for(
      "pops_explicit_physical_map", Policy(0, count), KOKKOS_LAMBDA(const std::size_t linear) {
        std::size_t coordinates[3] = {0, 0, 0};
        std::size_t index = linear % cells;
        const std::size_t component = linear / cells;
        std::size_t destination_offset = component * d.component_stride;
        std::size_t source_offset = component * s.component_stride;
        for (int axis = 0; axis < map.dimension; ++axis) {
          coordinates[axis] = index % d.extents[axis];
          index /= d.extents[axis];
          destination_offset += coordinates[axis] * d.axis_strides[axis];
        }
        for (int axis = 0; axis < map.dimension; ++axis)
          if (map.source_to_target[axis] >= 0)
            source_offset += coordinates[map.source_to_target[axis]] * s.axis_strides[axis];
        double value = 0.0;
        for (std::size_t reduction = 0; reduction < reductions; ++reduction) {
          std::size_t remainder = reduction;
          std::size_t offset = source_offset;
          double weight = 1.0;
          for (int axis = 0; axis < map.dimension; ++axis) {
            if (map.reduction_cells[axis]) {
              const auto cell = remainder % map.reduction_cells[axis];
              remainder /= map.reduction_cells[axis];
              offset += cell * s.axis_strides[axis];
              weight *= map.weights[map.weight_offsets[axis] + cell];
            }
          }
          value += source[offset] * weight;
        }
        destination[destination_offset] = value;
      });
  Kokkos::fence();
  int invalid = 0;
  Kokkos::parallel_reduce(
      "pops_physical_map_finite", Policy(0, count),
      KOKKOS_LAMBDA(const std::size_t linear, int& bad) {
        std::size_t index = linear % cells;
        std::size_t offset = (linear / cells) * d.component_stride;
        for (int axis = 0; axis < map.dimension; ++axis) {
          offset += (index % d.extents[axis]) * d.axis_strides[axis];
          index /= d.extents[axis];
        }
        if (!Kokkos::isfinite(destination[offset]))
          bad = 1;
        else if (bad < 0)
          bad = 0;
      },
      Kokkos::Max<int>(invalid));
  if (invalid)
    return 4;
  *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}
}  // namespace pops::component
