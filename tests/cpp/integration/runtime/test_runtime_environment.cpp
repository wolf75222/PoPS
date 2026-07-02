#include <gtest/gtest.h>

#include <pops/runtime/runtime_environment.hpp>

#include <string>

using namespace pops;

TEST(RuntimeEnvironment, ReportsDimensionPrecisionAndBackends) {
  const RuntimeEnvironmentReport report = runtime_environment_report();

  EXPECT_TRUE(report.dimension == 2) << "dimension_2";
  EXPECT_TRUE(report.amr_refinement_ratio == kAmrRefRatio) << "amr_ratio_2";
  EXPECT_TRUE(report.precision == "double") << "precision_double";
  EXPECT_TRUE(report.real_bytes == static_cast<int>(sizeof(Real))) << "real_bytes";
  EXPECT_TRUE(!report.supports_single_precision) << "no_single_precision";
  EXPECT_TRUE(!report.supports_mixed_precision) << "no_mixed_precision";
  EXPECT_TRUE(!report.supports_custom_communicator) << "no_custom_communicator";

#ifdef POPS_HAS_MPI
  EXPECT_TRUE(report.mpi_compiled) << "mpi_compiled";
  EXPECT_TRUE(report.communicator == "MPI_COMM_WORLD") << "mpi_world_communicator";
#else
  EXPECT_TRUE(!report.mpi_compiled) << "serial_mpi_flag";
  EXPECT_TRUE(report.communicator == "serial") << "serial_communicator";
#endif

#ifdef POPS_HAS_KOKKOS
  EXPECT_TRUE(report.has_kokkos) << "has_kokkos";
  EXPECT_TRUE(!report.kokkos_backend.empty()) << "kokkos_backend_named";
  EXPECT_TRUE(report.allocator_mode == "kokkos_shared_space_managed_arena") << "managed_arena";
  EXPECT_TRUE(report.comm_allocator_mode == "kokkos_shared_host_pinned_space")
      << "pinned_comm_allocator";
  EXPECT_TRUE(report.allocator_lifetime.find("process-lifetime") != std::string::npos)
      << "allocator_lifetime_reported";
#else
  EXPECT_TRUE(!report.has_kokkos) << "no_kokkos";
  EXPECT_TRUE(report.allocator_mode == "std_allocator") << "std_allocator";
#endif
}
