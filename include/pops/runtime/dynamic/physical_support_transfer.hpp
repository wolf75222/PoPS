/// Generic tensor-product support kernels for the negotiated Transfer interface.
#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

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

namespace physical_transfer_detail {
template <class MemorySpace>
constexpr PopsMemorySpaceV1 memory_kind() noexcept {
  if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
    return POPS_MEMORY_SPACE_HOST_V1;
  else if constexpr (Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                                                MemorySpace>::accessible)
    return POPS_MEMORY_SPACE_MANAGED_V1;
  else
    return POPS_MEMORY_SPACE_DEVICE_V1;
}
/// Owning backend copy. Borrowed authored host weights never enter a device kernel.
/// The invocation owns this View through its final fence; callers vote allocation/copy errors.
template <class ExecutionSpace>
class OwnedWeights {
 public:
  using View = Kokkos::View<double*, typename ExecutionSpace::memory_space>;
  OwnedWeights(const ExecutionSpace& execution, const double* source, std::size_t count)
      : execution_(execution), values_(Kokkos::view_alloc(Kokkos::WithoutInitializing, "pops_physical_weights"), count) {
    if (count) {
      using Host = Kokkos::View<const double*, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      try {
        Kokkos::deep_copy(execution, values_, Host(source, count));
        execution.fence();
      } catch (...) {
        try { execution.fence(); } catch (...) {}
        throw;
      }
    }
  }
  ~OwnedWeights() noexcept {
    // Fence before releasing backend allocation, including exception unwinding. Normal apply
    // already completed both kernel fences; an execution error is then reported by its caller.
    try { execution_.fence(); } catch (...) {}
  }
  const double* data() const noexcept { return values_.data(); }
 private:
  ExecutionSpace execution_;
  View values_;
};

template <class ExecutionSpace>
inline constexpr bool borrows_native_stream = false
#if defined(KOKKOS_ENABLE_CUDA)
    || std::is_same_v<ExecutionSpace, Kokkos::Cuda>
#endif
#if defined(KOKKOS_ENABLE_HIP)
    || std::is_same_v<ExecutionSpace, Kokkos::HIP>
#endif
    ;

/// The native caller authenticates backend/device/lane identities. Here memory dispatch must
/// additionally agree with the provider's compiled execution/memory space and context policy.
inline bool supports_device_context(const PopsExecutionContextV1& context,
                                    PopsMemorySpaceV1 memory) noexcept {
  if (memory != POPS_MEMORY_SPACE_DEVICE_V1 && memory != POPS_MEMORY_SPACE_MANAGED_V1)
    return false;
  if constexpr (std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
                               Kokkos::HostSpace>)
    return false;
  if (context.memory_space != memory ||
      context.compute_precision != POPS_PRECISION_FLOAT64_V1 ||
      context.accumulation_precision != POPS_PRECISION_FLOAT64_V1 ||
      context.reduction_precision != POPS_PRECISION_FLOAT64_V1)
    return false;
  try { validate_execution_context(context); } catch (...) { return false; }
  // CUDA/HIP wrap the exact borrowed native stream. Other backends use their authenticated
  // default instance only; an opaque foreign stream cannot silently become the default queue.
  if constexpr (!borrows_native_stream<Kokkos::DefaultExecutionSpace>)
    if (context.stream_handle != 0) return false;
  return true;
}

template <class ExecutionSpace>
ExecutionSpace execution_instance(const PopsExecutionContextV1& context) {
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
    return ExecutionSpace(reinterpret_cast<cudaStream_t>(context.stream_handle));
#endif
#if defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
    return ExecutionSpace(reinterpret_cast<hipStream_t>(context.stream_handle));
#endif
  (void)context;
  return ExecutionSpace{};
}

template <class ExecutionSpace>
int apply_kernel(const PhysicalSupportTransfer& map, const PopsConstFieldViewV1& s,
                 const PopsFieldViewV1& d, std::size_t cells, std::size_t reductions,
                 const ExecutionSpace& execution, PopsComponentStatusV1* status) {
  const auto* source = static_cast<const double*>(s.data);
  auto* destination = static_cast<double*>(d.data);
  const std::size_t count = cells * d.component_count;
  *status = {sizeof(PopsComponentStatusV1), 4, POPS_COMPONENT_ABORT_RUN_V1,
             "physical transfer produced a non-finite value"};
  using Policy =
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>;
  Kokkos::parallel_for(
      "pops_explicit_physical_map", Policy(execution, 0, count), KOKKOS_LAMBDA(const std::size_t linear) {
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
  execution.fence();
  int invalid = 0;
  Kokkos::parallel_reduce(
      "pops_physical_map_finite", Policy(execution, 0, count),
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
  execution.fence();
  if (invalid)
    return 4;
  *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}
}  // namespace physical_transfer_detail

inline int apply_physical_support_transfer(const PhysicalSupportTransfer& map,
                                           const PopsTransferRequestV1* request,
                                           PopsComponentStatusV1* status) {
  if (status)
    *status = {sizeof(PopsComponentStatusV1), 2, POPS_COMPONENT_ABORT_RUN_V1,
               "physical transfer has an invalid request or field contract"};
  if (!request || !status || request->struct_size < sizeof(PopsTransferRequestV1) ||
      map.dimension < 1 || map.dimension > 3 || request->dimension != map.dimension ||
      request->operation != map.operation || (map.operation != 2 && map.operation != 3))
    return 2;
  const auto s = request->source;
  const auto d = request->destination;
  if (s.struct_size < sizeof(PopsConstFieldViewV1) || d.struct_size < sizeof(PopsFieldViewV1) ||
      s.dimension != map.dimension || d.dimension != map.dimension || !s.data || !d.data ||
      s.memory_space != d.memory_space ||
      (s.memory_space != POPS_MEMORY_SPACE_HOST_V1 && s.memory_space != POPS_MEMORY_SPACE_DEVICE_V1 &&
       s.memory_space != POPS_MEMORY_SPACE_MANAGED_V1) ||
      s.scalar_type != POPS_SCALAR_FLOAT64_V1 || d.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
      !s.component_count || s.component_count != d.component_count || s.component_stride <= 0 ||
      d.component_stride <= 0)
    return 2;
  if (s.memory_space != POPS_MEMORY_SPACE_HOST_V1 &&
      !physical_transfer_detail::supports_device_context(request->execution, s.memory_space))
    return 2;
  *status = {sizeof(PopsComponentStatusV1), 3, POPS_COMPONENT_ABORT_RUN_V1,
             "physical transfer has invalid extents, strides or support weights"};
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
  if (map.weight_count > static_cast<std::size_t>(
          std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double))
    return 3;
  for (std::size_t i = 0; i < map.weight_count; ++i)
    if (!map.weights || !std::isfinite(map.weights[i]))
      return 3;
  if (s.memory_space == POPS_MEMORY_SPACE_HOST_V1) {
    // Historical Host callback accepts the original zero execution projection too.
    return physical_transfer_detail::apply_kernel(map, s, d, cells, reductions,
                                                   Kokkos::DefaultHostExecutionSpace{}, status);
  }
  try {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    const auto execution = physical_transfer_detail::execution_instance<ExecutionSpace>(request->execution);
    physical_transfer_detail::OwnedWeights<ExecutionSpace> weights(execution, map.weights, map.weight_count);
    auto owned_map = map;
    owned_map.weights = weights.data();
    return physical_transfer_detail::apply_kernel(owned_map, s, d, cells, reductions,
                                                   execution, status);
  } catch (...) {
    // No exception crosses the component callback. Existing native caller performs the matching
    // failure vote and rejects its candidate; no new collective occurs in this patch-local hook.
    *status = {sizeof(PopsComponentStatusV1), 5, POPS_COMPONENT_ABORT_RUN_V1,
               "physical transfer backend allocation, copy or execution failed"};
    return 5;
  }
}

/// One tensor-product intersection integral. Every source axis is integrated into one
/// destination cell; axis weights already contain retained-coordinate overlap fractions and
/// eliminated-coordinate measure. This is the same authoritative reduction used by a physical
/// support map, exposed explicitly for composite AMR contributions rather than relabeling a map.
struct PhysicalSupportIntegral {
  int dimension = 0;
  std::size_t weight_offsets[3] = {0, 0, 0};
  const double* weights = nullptr;
  std::size_t weight_count = 0;
};

inline int apply_physical_support_integral(const PhysicalSupportIntegral& integral,
                                           const PopsConstFieldViewV1& source,
                                           const PopsFieldViewV1& destination,
                                           PopsComponentStatusV1* status,
                                           const PopsExecutionContextV1* execution = nullptr) {
  if (status)
    *status = {sizeof(PopsComponentStatusV1), 2, POPS_COMPONENT_ABORT_RUN_V1,
               "physical integral has an invalid dimension"};
  if (integral.dimension < 1 || integral.dimension > 3)
    return 2;
  PhysicalSupportTransfer reduction{};
  reduction.dimension = integral.dimension;
  reduction.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  reduction.weights = integral.weights;
  reduction.weight_count = integral.weight_count;
  const std::int32_t identity_ratios[3] = {1, 1, 1};
  for (int axis = 0; axis < integral.dimension; ++axis) {
    if (destination.extents[axis] != 1) {
      if (status)
        *status = {sizeof(PopsComponentStatusV1), 3, POPS_COMPONENT_ABORT_RUN_V1,
                   "physical integral requires one destination cell"};
      return 3;
    }
    reduction.source_active[axis] = 1;
    reduction.reduction_cells[axis] = source.extents[axis];
    reduction.weight_offsets[axis] = integral.weight_offsets[axis];
  }
  PopsTransferRequestV1 request{};
  request.struct_size = sizeof(request);
  request.source = source;
  request.destination = destination;
  request.refinement_ratio = identity_ratios;
  request.dimension = integral.dimension;
  request.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  if (execution) request.execution = *execution;
  return apply_physical_support_transfer(reduction, &request, status);
}

}  // namespace pops::component
