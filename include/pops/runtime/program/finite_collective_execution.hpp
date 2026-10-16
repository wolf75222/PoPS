#pragma once

#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/finite_kokkos.hpp>
#include <pops/runtime/program/finite_structural_storage.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <any>
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

}  // namespace pops::runtime::program::finite_collective
