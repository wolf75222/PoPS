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

#ifdef POPS_HAS_KOKKOS
#include <Kokkos_Core.hpp>

#include <cstdlib>  // std::atexit
#endif

namespace pops {

#ifdef POPS_HAS_KOKKOS
namespace detail {
inline bool& kokkos_initialized_by_pops_flag() {
  static bool value = false;
  return value;
}

inline bool& kokkos_atexit_finalize_registered_flag() {
  static bool value = false;
  return value;
}

/// Initializes Kokkos on FIRST need (Fab allocation OR first kernel), finalizes via atexit.
/// No-op if the caller already did its own Kokkos::initialize / ScopeGuard, or if Kokkos is already
/// finalized. A single atexit is registered (subsequent calls see is_initialized()). Destruction
/// sequence: LOCAL MultiFabs are destroyed at the end of main, hence BEFORE the atexit finalize.
inline void ensure_kokkos_initialized() {
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::initialize();
    kokkos_initialized_by_pops_flag() = true;
    std::atexit([] {
      if (Kokkos::is_initialized())
        Kokkos::finalize();
    });
    kokkos_atexit_finalize_registered_flag() = true;
  }
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
inline void device_fence() {
#ifdef POPS_HAS_KOKKOS
  if (Kokkos::is_initialized())
    Kokkos::fence();
#endif
}

}  // namespace pops
