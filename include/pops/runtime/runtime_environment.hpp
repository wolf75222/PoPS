#pragma once

/// @file
/// @brief Runtime environment facts: Kokkos lifecycle, MPI communicator, precision and allocator.
///
/// This report is deliberately descriptive. It does not initialize Kokkos, MPI, or an allocator;
/// it only exposes the global assumptions that affect binding and runtime behaviour.

#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/parallel/comm.hpp>

#include <string>

#ifdef POPS_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace pops {

inline constexpr int kNativeDimension = 2;
inline constexpr int kNativeAmrRefinementRatio = kAmrRefRatio;

struct RuntimeEnvironmentReport {
  int dimension = kNativeDimension;
  int amr_refinement_ratio = kNativeAmrRefinementRatio;

  std::string precision = "double";
  int real_bytes = static_cast<int>(sizeof(Real));
  bool supports_single_precision = false;
  bool supports_mixed_precision = false;

  bool has_kokkos = false;
  bool kokkos_initialized = false;
  bool kokkos_finalized = false;
  bool kokkos_initialized_by_pops = false;
  bool kokkos_atexit_finalize_registered = false;
  std::string kokkos_backend = "none";
  std::string kokkos_ownership = "not-built";
  std::string kokkos_lifecycle = "not-built";

  bool mpi_compiled = false;
  bool mpi_active = false;
  int mpi_rank = 0;
  int mpi_ranks = 1;
  std::string communicator = "serial";
  bool supports_custom_communicator = false;
  bool mpi_initialized_by_pops = false;
  bool mpi_atexit_finalize_registered = false;
  int mpi_thread_level = 0;
  std::string mpi_ownership = "not-built";

  std::string allocator_mode = "std_allocator";
  std::string comm_allocator_mode = "std_allocator";
  std::string allocator_lifetime = "standard library allocator lifetime";
};

inline RuntimeEnvironmentReport runtime_environment_report() {
  RuntimeEnvironmentReport report{};
#ifdef POPS_HAS_KOKKOS
  report.has_kokkos = true;
  report.kokkos_initialized = Kokkos::is_initialized();
  report.kokkos_finalized = Kokkos::is_finalized();
  report.kokkos_initialized_by_pops = kokkos_initialized_by_pops();
  report.kokkos_atexit_finalize_registered = kokkos_atexit_finalize_registered();
  report.kokkos_backend = Kokkos::DefaultExecutionSpace::name();
  if (report.kokkos_initialized_by_pops) {
    report.kokkos_ownership = "pops-owned-lazy";
    report.kokkos_lifecycle =
        "PoPS lazily initialized Kokkos and registered an atexit finalize hook";
  } else if (report.kokkos_initialized) {
    report.kokkos_ownership = "external";
    report.kokkos_lifecycle =
        "Kokkos was already initialized by the caller; PoPS attaches and does not finalize it";
  } else if (report.kokkos_finalized) {
    report.kokkos_ownership = "finalized";
    report.kokkos_lifecycle = "Kokkos has already been finalized";
  } else {
    report.kokkos_ownership = "lazy";
    report.kokkos_lifecycle =
        "PoPS will lazily initialize Kokkos on first allocation or kernel launch";
  }
  report.allocator_mode = "kokkos_shared_space_managed_arena";
  report.comm_allocator_mode = "kokkos_shared_host_pinned_space";
  report.allocator_lifetime =
      "process-lifetime ManagedArena; blocks are released by a Kokkos finalize hook, "
      "the arena tables are intentionally never destroyed";
#endif
#ifdef POPS_HAS_MPI
  report.mpi_compiled = true;
  report.mpi_active = comm_active();
  report.mpi_rank = my_rank();
  report.mpi_ranks = n_ranks();
  report.communicator = "MPI_COMM_WORLD";
  report.supports_custom_communicator = false;
  report.mpi_initialized_by_pops = pops::mpi_initialized_by_pops();
  report.mpi_atexit_finalize_registered = pops::mpi_atexit_finalize_registered();
  report.mpi_thread_level = pops::mpi_thread_level();
  report.mpi_ownership =
      report.mpi_initialized_by_pops ? "pops-owned" : (report.mpi_active ? "external" : "inactive");
#endif
  return report;
}

}  // namespace pops
