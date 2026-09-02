/// @file
/// @brief Shared Kokkos lifecycle: lazy init + device barrier. Required since the
///        unified allocator (kokkos_malloc<SharedSpace>) is called AS SOON AS a Fab is built,
///        BEFORE any for_each. Init can therefore no longer be left to the first kernel alone: this
///        same guard applies to allocation AND to kernels, so that the Kokkos build
///        (Serial/OpenMP/Cuda) works without an explicit Kokkos::initialize in each main.
///
/// SEQUENCING INVARIANT: detail::ensure_kokkos_initialized() is called by ManagedArena
/// before any kokkos_malloc; device_fence() is called by the host before any access to unified
/// memory after a kernel. These two entry points are the ONLY places where the Kokkos lifecycle
/// is driven; do not call Kokkos::initialize/finalize anywhere else.

#pragma once

#include <string>

#ifdef POPS_HAS_KOKKOS
#include <Kokkos_Core.hpp>

#include <cstdlib>  // std::atexit
#endif

namespace pops {

#ifdef POPS_HAS_KOKKOS
namespace detail {
inline const std::string& kokkos_device_fence_label() {
  static const std::string value("pops.device-fence");
  return value;
}

inline bool& kokkos_initialized_by_pops_flag() {
  static bool value = false;
  return value;
}

inline bool& kokkos_atexit_finalize_registered_flag() {
  static bool value = false;
  return value;
}

/// Process-lifetime default execution instance.  Kokkos Serial stores its singleton through a
/// shared control block, so creating a temporary execution space is a heap allocation even
/// though dispatch itself is synchronous.  Construct this once during lifecycle priming and
/// retain it through Kokkos finalization; individual Program steps must only borrow it.
inline const Kokkos::DefaultExecutionSpace& default_execution_space() {
  static const auto* value = new Kokkos::DefaultExecutionSpace();
  return *value;
}

/// Initializes Kokkos on FIRST need (Fab allocation OR first kernel), finalizes via atexit.
/// No-op if the caller already did its own Kokkos::initialize / ScopeGuard, or if Kokkos is already
/// finalized. A single atexit is registered (subsequent calls see is_initialized()). Destruction
/// sequence: LOCAL MultiFabs are destroyed at the end of main, hence BEFORE the atexit finalize.
inline void ensure_kokkos_initialized() {
  // Prime the owning label before any prepared execution window can begin.  Kokkos' no-argument
  // fence constructs its default std::string at each call, which is an otherwise hidden hot-path
  // heap allocation.
  (void)kokkos_device_fence_label();
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::initialize();
    kokkos_initialized_by_pops_flag() = true;
    std::atexit([] {
      if (Kokkos::is_initialized())
        Kokkos::finalize();
    });
    kokkos_atexit_finalize_registered_flag() = true;
  }
  if (Kokkos::is_initialized())
    (void)default_execution_space();
}
}  // namespace detail
#endif

inline bool kokkos_initialized_by_pops() {
#ifdef POPS_HAS_KOKKOS
  return detail::kokkos_initialized_by_pops_flag();
#else
  return false;
#endif
}

inline bool kokkos_atexit_finalize_registered() {
#ifdef POPS_HAS_KOKKOS
  return detail::kokkos_atexit_finalize_registered_flag();
#else
  return false;
#endif
}

/// Device barrier: waits for in-flight kernels to finish before a HOST access to unified memory.
/// No-op outside Kokkos (and if nothing has been launched).
#ifdef POPS_HAS_KOKKOS
inline void device_fence(const std::string& label = detail::kokkos_device_fence_label()) {
  if (Kokkos::is_initialized()) {
    // OpenMP and Serial submit synchronously. Kokkos' process-wide OpenMP fence still builds a
    // profiling payload, so it allocates even though there is no asynchronous work to drain.
    // Keep the process-wide fence whenever a potentially asynchronous execution space is
    // compiled: explicit CUDA/HIP streams must be complete before host-visible publication.
#if defined(KOKKOS_ENABLE_THREADS) || defined(KOKKOS_ENABLE_HPX) || defined(KOKKOS_ENABLE_CUDA) || \
    defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC) || \
    defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_NEXTSILICON)
    // This is deliberately the process-wide Kokkos fence, not a fence on the retained default
    // instance.  Prepared CUDA/HIP paths may submit work on explicit execution-space instances;
    // publication must wait for those streams as well before exposing host-visible state.  A
    // mixed CPU/accelerator build also takes this branch, regardless of its default execution
    // space.
    Kokkos::fence(label);
#else
    (void)label;
#endif
  }
}
#else
/// Preserve the labelled public fence API in builds without Kokkos.
inline void device_fence(const std::string& = {}) {}
#endif

}  // namespace pops
