#pragma once

/// @file
/// @brief Prepared, fail-closed accelerator stream partition with lane-private scratch.

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/foundation/types.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::accelerator {

/// Raised when a caller asks PoPS to claim independent accelerator streams without proof.
class PreparedStreamPartitionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {

template <class ExecutionSpace>
inline constexpr bool authentic_partitioned_stream_backend =
#if defined(KOKKOS_ENABLE_CUDA)
    std::is_same_v<ExecutionSpace, Kokkos::Cuda> ||
#endif
#if defined(KOKKOS_ENABLE_HIP)
    std::is_same_v<ExecutionSpace, Kokkos::HIP> ||
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL> ||
#endif
    false;

template <class ExecutionSpace>
[[nodiscard]] constexpr const char* stream_backend_name() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
    return "cuda";
#endif
#if defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
    return "hip";
#endif
#if defined(KOKKOS_ENABLE_SYCL)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>)
    return "sycl";
#endif
  return "unsupported";
}

template <class ExecutionSpace>
concept InstanceIdentifiedExecutionSpace = requires(const ExecutionSpace& instance) {
  { instance.impl_instance_id() } -> std::convertible_to<std::uint32_t>;
};

}  // namespace detail

/// Reviewable facts established while the stream/workspace partition is prepared.
///
/// ``independent_streams`` is deliberately narrower than "partition_space returned N objects": it
/// is true only for a Kokkos accelerator backend that creates native queues/streams and only after
/// every returned instance identifier has been proved distinct.  Runtime overlap is not claimed
/// here; it must be measured by the out-of-CI hardware campaign.
struct PreparedStreamPartitionEvidence {
  std::string backend;
  std::vector<std::string> stream_identities;
  bool independent_streams = false;
  bool disjoint_workspaces = false;
  std::size_t workspace_values_per_stream = 0;
};

/// Prepared authority for concurrent accelerator kernels.
///
/// The authority owns every execution-space instance and one device-memory workspace per lane.
/// Both are materialized before execution. ``launch_for`` takes a lane explicitly and performs no
/// PoPS allocation or global fence, allowing independent lanes to overlap.  Callers synchronize
/// with ``fence(lane)`` or ``fence_all()`` only at their dependency boundary.
template <class Scalar = Real, class ExecutionSpace = Kokkos::DefaultExecutionSpace>
class PreparedAcceleratorStreamExecutor {
 public:
  using scalar_type = Scalar;
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using workspace_type = Kokkos::View<scalar_type*, memory_space>;

  static_assert(Kokkos::is_execution_space<execution_space>::value,
                "PreparedAcceleratorStreamExecutor requires a Kokkos execution space");

  PreparedAcceleratorStreamExecutor(const PreparedAcceleratorStreamExecutor&) = delete;
  PreparedAcceleratorStreamExecutor& operator=(const PreparedAcceleratorStreamExecutor&) = delete;
  PreparedAcceleratorStreamExecutor(PreparedAcceleratorStreamExecutor&&) noexcept = default;
  PreparedAcceleratorStreamExecutor& operator=(PreparedAcceleratorStreamExecutor&&) noexcept =
      default;

  /// Materialize an exact stream partition and all lane-private workspaces.
  ///
  /// CPU execution spaces and accelerator backends for which Kokkos does not create independent
  /// native queues are refused.  Returning aliased instance identities is also a hard error.
  [[nodiscard]] static PreparedAcceleratorStreamExecutor prepare(
      std::size_t stream_count, std::size_t workspace_values_per_stream,
      std::vector<double> weights = {}) {
    if (stream_count < 2)
      throw std::invalid_argument("accelerator stream partition requires at least two streams");
    if (workspace_values_per_stream == 0)
      throw std::invalid_argument("accelerator stream workspaces must be non-empty");
    if (workspace_values_per_stream > std::numeric_limits<std::size_t>::max() / sizeof(scalar_type))
      throw std::overflow_error("accelerator stream workspace byte extent overflows size_t");
    if (stream_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("accelerator stream count exceeds the supported integer range");
    if (!weights.empty() && weights.size() != stream_count)
      throw std::invalid_argument("accelerator stream weights must match the stream count");
    if (weights.empty())
      weights.assign(stream_count, 1.0);
    if (std::any_of(weights.begin(), weights.end(), [](double weight) { return !(weight > 0.0); }))
      throw std::invalid_argument("accelerator stream weights must be strictly positive");

    if constexpr (!detail::authentic_partitioned_stream_backend<execution_space>) {
      throw PreparedStreamPartitionError(std::string("Kokkos execution space '") +
                                         execution_space::name() +
                                         "' cannot prove independent accelerator streams");
    } else {
      static_assert(detail::InstanceIdentifiedExecutionSpace<execution_space>,
                    "authenticated stream backends must expose an instance identifier");
      pops::detail::ensure_kokkos_initialized();
      const execution_space base_instance{};
      std::vector<execution_space> instances =
          Kokkos::Experimental::partition_space(base_instance, weights);
      if (instances.size() != stream_count)
        throw PreparedStreamPartitionError(
            "Kokkos returned an incomplete accelerator stream partition");
      return PreparedAcceleratorStreamExecutor(std::move(instances), workspace_values_per_stream);
    }
  }

  [[nodiscard]] static constexpr bool backend_can_partition_authentic_streams() noexcept {
    return detail::authentic_partitioned_stream_backend<execution_space>;
  }

  [[nodiscard]] std::size_t size() const noexcept { return lanes_.size(); }
  [[nodiscard]] std::size_t workspace_values_per_stream() const noexcept {
    return evidence_.workspace_values_per_stream;
  }
  [[nodiscard]] const PreparedStreamPartitionEvidence& evidence() const noexcept {
    return evidence_;
  }

  [[nodiscard]] const execution_space& instance(std::size_t lane) const {
    return lane_(lane).instance;
  }
  [[nodiscard]] const workspace_type& workspace(std::size_t lane) const {
    return lane_(lane).workspace;
  }
  [[nodiscard]] scalar_type* workspace_data(std::size_t lane) const {
    return lane_(lane).workspace.data();
  }
  [[nodiscard]] std::uintptr_t workspace_address(std::size_t lane) const {
    return reinterpret_cast<std::uintptr_t>(workspace_data(lane));
  }
  [[nodiscard]] const std::string& stream_identity(std::size_t lane) const {
    return lane_(lane).identity;
  }

  /// Submit a kernel to one exact prepared lane.  This call intentionally does not fence.
  template <class Functor>
  void launch_for(std::size_t lane, const char* label, std::int64_t count, Functor functor) const {
    if (label == nullptr || *label == '\0')
      throw std::invalid_argument("accelerator stream kernel label must be non-empty");
    if (count < 0)
      throw std::invalid_argument("accelerator stream kernel extent must be non-negative");
    if (count == 0)
      return;
    const Lane& selected = lane_(lane);
    using policy_type = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::int64_t>>;
    Kokkos::parallel_for(label, policy_type(selected.instance, 0, count), std::move(functor));
  }

  void fence(std::size_t lane, const std::string& label = "PoPS prepared stream fence") const {
    lane_(lane).instance.fence(label);
  }
  void fence_all() const {
    for (std::size_t lane = 0; lane < lanes_.size(); ++lane)
      fence(lane, "PoPS prepared stream partition fence");
  }

 private:
  struct Lane {
    execution_space instance;
    workspace_type workspace;
    std::string identity;
  };

  PreparedAcceleratorStreamExecutor(std::vector<execution_space> instances,
                                    std::size_t workspace_values_per_stream) {
    lanes_.reserve(instances.size());
    evidence_.backend = detail::stream_backend_name<execution_space>();
    evidence_.workspace_values_per_stream = workspace_values_per_stream;
    evidence_.stream_identities.reserve(instances.size());

    std::vector<std::uint32_t> instance_ids;
    instance_ids.reserve(instances.size());
    for (std::size_t lane = 0; lane < instances.size(); ++lane) {
      const std::uint32_t instance_id =
          static_cast<std::uint32_t>(instances[lane].impl_instance_id());
      const std::string identity = evidence_.backend + ":instance=" + std::to_string(instance_id) +
                                   ":lane=" + std::to_string(lane);
      const std::string workspace_label = "pops_prepared_stream_workspace_" + std::to_string(lane);
      workspace_type workspace(workspace_label, workspace_values_per_stream);
      Kokkos::deep_copy(instances[lane], workspace, scalar_type{});
      lanes_.push_back({std::move(instances[lane]), std::move(workspace), identity});
      instance_ids.push_back(instance_id);
      evidence_.stream_identities.push_back(identity);
    }
    fence_all();

    std::sort(instance_ids.begin(), instance_ids.end());
    evidence_.independent_streams =
        std::adjacent_find(instance_ids.begin(), instance_ids.end()) == instance_ids.end();
    evidence_.disjoint_workspaces = workspaces_are_disjoint_();
    if (!evidence_.independent_streams)
      throw PreparedStreamPartitionError(
          "Kokkos partition_space returned aliased accelerator instances");
    if (!evidence_.disjoint_workspaces)
      throw PreparedStreamPartitionError("prepared accelerator stream workspaces overlap");
  }

  [[nodiscard]] const Lane& lane_(std::size_t lane) const {
    if (lane >= lanes_.size())
      throw std::out_of_range("accelerator stream lane is out of range");
    return lanes_[lane];
  }

  [[nodiscard]] bool workspaces_are_disjoint_() const noexcept {
    for (std::size_t lhs = 0; lhs < lanes_.size(); ++lhs)
      for (std::size_t rhs = lhs + 1; rhs < lanes_.size(); ++rhs) {
        const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lanes_[lhs].workspace.data());
        const auto rhs_begin = reinterpret_cast<std::uintptr_t>(lanes_[rhs].workspace.data());
        const std::size_t bytes = evidence_.workspace_values_per_stream * sizeof(scalar_type);
        const auto lhs_end = lhs_begin + static_cast<std::uintptr_t>(bytes);
        const auto rhs_end = rhs_begin + static_cast<std::uintptr_t>(bytes);
        if (lhs_begin < rhs_end && rhs_begin < lhs_end)
          return false;
      }
    return true;
  }

  std::vector<Lane> lanes_;
  PreparedStreamPartitionEvidence evidence_;
};

}  // namespace pops::runtime::accelerator
