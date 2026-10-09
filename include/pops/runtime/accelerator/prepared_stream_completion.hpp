#pragma once

#include <Kokkos_Core.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace pops::runtime::accelerator::detail {

/// A real queue event on CUDA/HIP. Other execution spaces fence at record(),
/// including CPU: they make no asynchronous-device completion claim.
template <class ExecutionSpace>
class PreparedStreamCompletion {
 public:
  explicit PreparedStreamCompletion(const ExecutionSpace& instance, bool synchronous)
      : instance_(instance), synchronous_(synchronous) {
    if (synchronous_)
      return;
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      cudaEvent_t event = nullptr;
      check_cuda_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      handle_ = reinterpret_cast<std::uintptr_t>(event);
      return;
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      hipEvent_t event = nullptr;
      check_hip_(hipEventCreateWithFlags(&event, hipEventDisableTiming));
      handle_ = reinterpret_cast<std::uintptr_t>(event);
      return;
    }
#endif
    synchronous_ = true;
  }
  PreparedStreamCompletion(const PreparedStreamCompletion&) = delete;
  PreparedStreamCompletion& operator=(const PreparedStreamCompletion&) = delete;
  ~PreparedStreamCompletion() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      if (handle_ && cudaEventDestroy(reinterpret_cast<cudaEvent_t>(handle_)) != cudaSuccess)
        std::terminate();
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      if (handle_ && hipEventDestroy(reinterpret_cast<hipEvent_t>(handle_)) != hipSuccess)
        std::terminate();
    }
#endif
  }
  void record() {
    if (synchronous_) {
      wait();
      return;
    }
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
      check_cuda_(cudaEventRecord(reinterpret_cast<cudaEvent_t>(handle_), instance_.cuda_stream()));
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
      check_hip_(hipEventRecord(reinterpret_cast<hipEvent_t>(handle_), instance_.hip_stream()));
#endif
  }
  bool query() const {
    if (synchronous_)
      return true;
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      const auto status = cudaEventQuery(reinterpret_cast<cudaEvent_t>(handle_));
      if (status == cudaErrorNotReady)
        return false;
      check_cuda_(status);
      return true;
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      const auto status = hipEventQuery(reinterpret_cast<hipEvent_t>(handle_));
      if (status == hipErrorNotReady)
        return false;
      check_hip_(status);
      return true;
    }
#endif
    throw std::logic_error("prepared completion has no native event implementation");
  }
  void wait() const {
    // Unlike querying/recording an event, a successful queue fence also covers
    // work that was submitted before event recording failed.
    instance_.fence("PoPS prepared resource physical completion");
  }

 private:
#if defined(KOKKOS_ENABLE_CUDA)
  static void check_cuda_(cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("prepared CUDA completion: ") +
                               cudaGetErrorString(status));
  }
#endif
#if defined(KOKKOS_ENABLE_HIP)
  static void check_hip_(hipError_t status) {
    if (status != hipSuccess)
      throw std::runtime_error(std::string("prepared HIP completion: ") +
                               hipGetErrorString(status));
  }
#endif
  ExecutionSpace instance_;
  bool synchronous_;
  std::uintptr_t handle_ = 0;
};

}  // namespace pops::runtime::accelerator::detail
