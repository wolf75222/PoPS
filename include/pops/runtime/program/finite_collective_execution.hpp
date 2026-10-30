#pragma once

#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/finite_device_array.hpp>
#include <pops/runtime/program/finite_kokkos.hpp>
#include <pops/runtime/program/finite_structural_storage.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <any>
#include <bit>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

// These functions factor the existing host staging/collective protocol. They
// neither evaluate equations nor change a value's ownership or numerical work.
#if defined(__GNUC__) || defined(__clang__)
#define POPS_FINITE_HOST_NOINLINE __attribute__((noinline))
#else
#define POPS_FINITE_HOST_NOINLINE
#endif

namespace pops::runtime::program::finite_collective {

POPS_FINITE_HOST_NOINLINE inline void reject_if(bool invalid, const ExecutionLane& lane,
                                                const char* message) {
  if (all_reduce_max(invalid ? 1L : 0L, lane) != 0)
    throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, "structured_control", message);
}

POPS_FINITE_HOST_NOINLINE inline void uniform_int(std::int64_t value, const ExecutionLane& lane,
                                                  const char* message) {
  static_assert(std::numeric_limits<long>::digits >= std::numeric_limits<std::int64_t>::digits,
                "native collective long must represent every Int64 value");
  if (all_reduce_min(static_cast<long>(value), lane) !=
      all_reduce_max(static_cast<long>(value), lane))
    throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, "structured_control", message);
}

POPS_FINITE_HOST_NOINLINE inline void require_domain(const FiniteDomainRef& domain,
                                                     std::int64_t capacity, std::int64_t family,
                                                     const ExecutionLane& lane) {
  reject_if(!domain || !domain->visible() || domain->size < 0 || domain->size > capacity ||
                domain->capacity != capacity || domain->family != family,
            lane, "invalid finite domain instance");
  uniform_int(domain->size, lane, "replicated finite domain differs across execution ranks");
  uniform_int(domain->version, lane, "replicated finite domain differs across execution ranks");
}

// This template is instantiated by the small typed operations below, not by
// every generated SSA operation. Preserve exception classification and the
// exact collective order, including the exhaustion vote after the error vote.
template <class Operation>
POPS_FINITE_HOST_NOINLINE void allocation_phase(Operation&& operation, const ExecutionLane& lane) {
  bool exhausted = false;
  std::exception_ptr error;
  try {
    operation();
  } catch (const std::bad_alloc&) {
    exhausted = true;
  } catch (const std::length_error&) {
    exhausted = true;
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "finite array allocation");
  reject_if(exhausted, lane, "finite array allocation capacity exhausted");
}

template <class T>
POPS_FINITE_HOST_NOINLINE void allocate_view(finite_kokkos::View<T>& target, std::int64_t count,
                                             const char* label, const ExecutionLane& lane) {
  allocation_phase([&] { target = finite_kokkos::allocate<T>(count, label); }, lane);
}

template <class HostView, class DeviceView>
POPS_FINITE_HOST_NOINLINE void mirror(HostView& target, const DeviceView& source,
                                      const ExecutionLane& lane) {
  allocation_phase([&] { target = Kokkos::create_mirror_view(source); }, lane);
}

template <class HostView, class DeviceView>
POPS_FINITE_HOST_NOINLINE void mirror_copy(HostView& target, const DeviceView& source,
                                           const ExecutionLane& lane) {
  allocation_phase(
      [&] { target = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source); }, lane);
}

POPS_FINITE_HOST_NOINLINE inline void allocate_values(
    std::shared_ptr<std::vector<std::any>>& target, std::size_t count, const ExecutionLane& lane) {
  allocation_phase([&] { target = finite_structural::make_values(count); }, lane);
}

template <class T>
POPS_FINITE_HOST_NOINLINE void assign_any(std::any& target, const T& value,
                                          const ExecutionLane& lane) {
  allocation_phase([&] { target = std::any(value); }, lane);
}

// Device-resident Arrays (finite_device_array.hpp). Each local device phase
// converges its errors before the next collective; only O(1) control scalars
// (a failure lane, a fingerprint, one element on an explicit host read) reach
// the host.
template <class Operation>
POPS_FINITE_HOST_NOINLINE void device_phase(Operation&& operation, const ExecutionLane& lane,
                                            const char* message) {
  std::exception_ptr error;
  try {
    operation();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, message);
}

template <class T>
POPS_FINITE_HOST_NOINLINE void capture(finite_kokkos::View<const T>& target,
                                       const finite_device::Array& array, std::size_t column,
                                       const ExecutionLane& lane) {
  device_phase([&] { target = finite_device::column<T>(array, column); }, lane,
               "finite device Array capture");
}

template <class... T>
POPS_FINITE_HOST_NOINLINE void publish(finite_device::Array& target, const FiniteDomainRef& domain,
                                       const ExecutionLane& lane,
                                       const finite_kokkos::View<T>&... columns) {
  allocation_phase(
      [&] {
        target = finite_device::publish(
            domain, std::vector<finite_device::Column>{
                        finite_device::Column(finite_kokkos::View<const T>(columns))...});
      },
      lane);
}

template <class T>
POPS_FINITE_HOST_NOINLINE void allocate_host(finite_device::HostView<T>& target, std::int64_t count,
                                             const char* label, const ExecutionLane& lane) {
  allocation_phase(
      [&] {
        if (count < 0 ||
            static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / sizeof(T))
          throw std::length_error("finite host column exceeds addressable memory");
        target = finite_device::HostView<T>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                               std::string(finite_device::kHostLabel) + label),
            static_cast<std::size_t>(count));
      },
      lane);
}

// Publication of a host phase: one counted copy per column.
template <class... T>
POPS_FINITE_HOST_NOINLINE void publish_host(finite_device::Array& target,
                                            const FiniteDomainRef& domain,
                                            finite_device::Residency* residency,
                                            const ExecutionLane& lane,
                                            const finite_device::HostView<T>&... host) {
  allocation_phase(
      [&] {
        target = finite_device::from_host(
            domain, std::vector<finite_device::HostColumn>{finite_device::HostColumn(host)...},
            residency);
      },
      lane);
}

template <class T>
POPS_FINITE_HOST_NOINLINE T read(const finite_device::Array& array, std::size_t column,
                                 std::int64_t offset, finite_device::Residency* residency,
                                 const ExecutionLane& lane) {
  T value{};
  device_phase([&] { value = finite_device::read<T>(array, column, offset, residency); }, lane,
               "finite device Array read");
  return value;
}

POPS_FINITE_HOST_NOINLINE inline std::int64_t first_failure(
    const finite_kokkos::View<unsigned char>& flags, std::int64_t count,
    const ExecutionLane& lane) {
  std::int64_t first = count;
  device_phase([&] { first = finite_device::first_failure(flags, count); }, lane,
               "finite Kokkos failure reduction");
  return all_reduce_min(static_cast<long>(first), lane);
}

// Replication of a device Array: the domain, then one fingerprint per column
// family, O(1) collectives instead of one per element (finite_device_array.hpp).
POPS_FINITE_HOST_NOINLINE inline void uniform_device(const finite_device::Array& array,
                                                     std::int64_t capacity, std::int64_t family,
                                                     std::size_t leaves,
                                                     const ExecutionLane& lane) {
  require_domain(array.domain, capacity, family, lane);
  finite_device::Fingerprint print;
  device_phase(
      [&] {
        finite_device::require_columns(array, leaves);
        print = finite_device::fingerprint(array);
      },
      lane, "finite device Array fingerprint");
  reject_if(print.nonfinite != 0, lane, "replicated array has a nonfinite element");
  for (const auto word : {print.real_first, print.real_second})
    uniform_int(std::bit_cast<std::int64_t>(word), lane,
                "replicated array Float64 bits differ across execution ranks");
  for (const auto word : {print.integer_first, print.integer_second})
    uniform_int(std::bit_cast<std::int64_t>(word), lane,
                "replicated array element differs across execution ranks");
}

}  // namespace pops::runtime::program::finite_collective
