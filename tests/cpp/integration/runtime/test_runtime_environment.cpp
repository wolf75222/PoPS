#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/runtime/runtime_environment.hpp>

#include "test_harness.hpp"

#include <string>

using namespace pops;

static int pops_run_test_runtime_environment() {
  pops::test::Checker chk;
  const RuntimeEnvironmentReport report = runtime_environment_report();

  chk(report.dimension == 2, "dimension_2");
  chk(report.amr_refinement_ratio == kAmrRefRatio, "amr_ratio_2");
  chk(report.precision == "double", "precision_double");
  chk(report.real_bytes == static_cast<int>(sizeof(Real)), "real_bytes");
  chk(!report.supports_single_precision, "no_single_precision");
  chk(!report.supports_mixed_precision, "no_mixed_precision");
  chk(!report.supports_custom_communicator, "no_custom_communicator");

#ifdef POPS_HAS_MPI
  chk(report.mpi_compiled, "mpi_compiled");
  chk(report.communicator == "MPI_COMM_WORLD", "mpi_world_communicator");
#else
  chk(!report.mpi_compiled, "serial_mpi_flag");
  chk(report.communicator == "serial", "serial_communicator");
#endif

#ifdef POPS_HAS_KOKKOS
  chk(report.has_kokkos, "has_kokkos");
  chk(!report.kokkos_backend.empty(), "kokkos_backend_named");
  chk(report.allocator_mode == "kokkos_shared_space_managed_arena", "managed_arena");
  chk(report.comm_allocator_mode == "kokkos_shared_host_pinned_space", "pinned_comm_allocator");
  chk(report.allocator_lifetime.find("process-lifetime") != std::string::npos,
      "allocator_lifetime_reported");
#else
  chk(!report.has_kokkos, "no_kokkos");
  chk(report.allocator_mode == "std_allocator", "std_allocator");
#endif

  if (chk.fails() == 0)
    std::printf("OK test_runtime_environment\n");
  return chk.failed();
}

TEST(test_runtime_environment, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_runtime_environment, "test_runtime_environment"), 0);
}
