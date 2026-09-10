#pragma once

#include <pops/numerics/fv/flux_interfaces.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <limits>
#include <type_traits>

namespace pops {

template <bool Host, bool Device>
inline constexpr bool native_call_execution_supported =
    std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>
        ? Host
        : Device;

/// Device-copyable result for a statically compiled model-library function.
/// EvaluationStatus is the existing PoPS status contract, not another failure authority.
template <int N>
struct NativeCallResult {
  static_assert(N > 0);
  EvaluationStatus status = EvaluationStatus::kFailed;
  unsigned int reason = 0;
  Kokkos::Array<Real, N> values{};

  POPS_HD static NativeCallResult rejected(unsigned int why = 1) {
    NativeCallResult result;
    result.status = EvaluationStatus::kReject;
    result.reason = why;
    return result;
  }

  POPS_HD Real read(int component) const {
    return status == EvaluationStatus::kOk && component >= 0 && component < N
               ? values[component]
               : std::numeric_limits<Real>::quiet_NaN();
  }
};

}  // namespace pops
