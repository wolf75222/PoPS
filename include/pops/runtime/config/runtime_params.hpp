#pragma once

#include <pops/core/foundation/types.hpp>  // Real, POPS_HD (device-callable)

#include <cstddef>
#include <pops/core/foundation/device_capture.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

/// @file
/// Runtime parameter values are an immutable, owned host/device incarnation.
/// Python assigns exact, stable indices; generated equations use get(index).
/// Declarations carry cardinality only and must be bound before execution.
/// A complete carrier is allocated only after its explicit byte-capacity check.
/// Copies share its read-only managed views; rebinding creates new storage.
/// The public index is int, but there is no fixed parameter-count capacity.

namespace pops {

class RuntimeParams {
 public:
  using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
  using host_storage = Kokkos::View<const Real*, Kokkos::HostSpace>;
  using device_storage = Kokkos::View<const Real*, memory_space>;

  using pops_device_capture_members =
      DeviceCaptureMembers<host_storage, device_storage, int, std::uint64_t>;

  int count = 0;

  RuntimeParams() = default;

  /// Declaration-only bricks do not allocate before the host has authenticated
  /// BindSchema cardinality and the caller's storage budget. They cannot
  /// execute until bind_runtime_params replaces this declaration with a bound
  /// carrier.
  static RuntimeParams declaration(int declared_count) {
    if (declared_count < 0)
      throw std::invalid_argument("runtime parameter declaration has negative cardinality");
    RuntimeParams result;
    result.count = declared_count;
    return result;
  }

  /// Account for both payloads and the owning descriptor before allocation.
  /// Allocator-specific bookkeeping is outside this declared storage budget.
  static std::uint64_t checked_storage_bytes(std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error(
          "runtime parameter cardinality exceeds the native index representation");
    if (size == 0)
      return 0;
    constexpr std::uint64_t width = 2 * sizeof(Real);
    if (size > (std::numeric_limits<std::uint64_t>::max() - sizeof(RuntimeParams)) / width)
      throw std::length_error("runtime parameter storage byte count overflows");
    return sizeof(RuntimeParams) + static_cast<std::uint64_t>(size) * width;
  }

  static RuntimeParams from_host(std::span<const double> values, std::uint64_t byte_capacity) {
    const auto bytes = checked_storage_bytes(values.size());
    if (bytes > byte_capacity)
      throw std::length_error("runtime parameter carrier exceeds its bound byte capacity");
    RuntimeParams result;
    result.count = static_cast<int>(values.size());
    result.byte_capacity_ = byte_capacity;
    if (values.empty())
      return result;
    detail::ensure_kokkos_initialized();
    Kokkos::View<Real*, Kokkos::HostSpace> host("pops_runtime_params_host", values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
      host(index) = static_cast<Real>(values[index]);
    Kokkos::View<Real*, memory_space> device("pops_runtime_params_device", values.size());
    Kokkos::deep_copy(device, host);
    Kokkos::fence();
    result.host_ = std::move(host);
    result.device_ = std::move(device);
    return result;
  }

  bool bound() const noexcept {
    return count >= 0 && host_.extent(0) == static_cast<std::size_t>(count) &&
           device_.extent(0) == static_cast<std::size_t>(count);
  }
  void require_bound() const {
    if (!bound())
      throw std::invalid_argument(
          "runtime parameter carrier is not bound to its declared cardinality");
    if (checked_storage_bytes(static_cast<std::size_t>(count)) > byte_capacity_)
      throw std::length_error("runtime parameter carrier lost its bound byte capacity");
  }
  std::uint64_t byte_capacity() const noexcept { return byte_capacity_; }
  std::uint64_t storage_bytes() const {
    require_bound();
    return checked_storage_bytes(static_cast<std::size_t>(count));
  }

  POPS_HD Real get(int index) const {
    KOKKOS_ASSERT(index >= 0 && index < count);
    KOKKOS_IF_ON_DEVICE((KOKKOS_ASSERT(device_.extent(0) == static_cast<std::size_t>(count));
                         return device_(index);))
    KOKKOS_IF_ON_HOST(
        (KOKKOS_ASSERT(host_.extent(0) == static_cast<std::size_t>(count)); return host_(index);))
  }

 private:
  host_storage host_;
  device_storage device_;
  std::uint64_t byte_capacity_ = 0;
};

}  // namespace pops
