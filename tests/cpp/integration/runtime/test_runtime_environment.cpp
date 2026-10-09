#include <gtest/gtest.h>

#include <pops/mesh/storage/fab.hpp>
#include <pops/runtime/runtime_environment.hpp>

#include <string>

using namespace pops;

TEST(RuntimeEnvironment, ReportsActualFabMemoryResidence) {
  using FieldMemorySpace = typename Fab<kNativeDimension>::memory_space;
  static_assert(std::is_same_v<FieldMemorySpace,
                              Kokkos::DefaultExecutionSpace::memory_space>);
  const RuntimeEnvironmentReport report = runtime_environment_report();
  if constexpr (std::is_same_v<FieldMemorySpace, Kokkos::HostSpace>) {
    EXPECT_EQ(report.field_memory_space, "host");
  } else if constexpr (Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                                                 FieldMemorySpace>::accessible) {
    EXPECT_EQ(report.field_memory_space, "managed");
  } else {
    EXPECT_EQ(report.field_memory_space, "device");
  }

  Box<kNativeDimension> valid{};
  for (int axis = 0; axis < kNativeDimension; ++axis)
    valid.hi[axis] = 1;
  Fab<kNativeDimension> fab(valid, 2);
  const auto storage = fab.storage();
  Kokkos::parallel_for(
      "runtime_environment_actual_field_residence",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, fab.size()),
      KOKKOS_LAMBDA(const std::size_t index) {
        storage(index) = static_cast<Real>(index) + Real{0.25};
      });
  Kokkos::DefaultExecutionSpace{}.fence("runtime_environment_field_residence");
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  for (std::size_t index = 0; index < fab.size(); ++index)
    EXPECT_EQ(host(index), static_cast<Real>(index) + Real{0.25});
}

TEST(RuntimeEnvironment, ReportsDimensionPrecisionAndBackends) {
  const RuntimeEnvironmentReport report = runtime_environment_report();

  EXPECT_TRUE(report.dimension == kNativeDimension) << "native_dimension";
  EXPECT_TRUE(report.amr_refinement_ratio_selection == "hierarchy_exact_rank")
      << "amr_ratio_authority";
  EXPECT_TRUE(report.amr_refinement_ratio_rank == kNativeDimension) << "amr_ratio_rank";
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
  if (report.kokkos_initialized) {
    EXPECT_TRUE(report.kokkos_concurrency == Kokkos::DefaultExecutionSpace{}.concurrency())
        << "initialized_kokkos_concurrency";
  } else {
    EXPECT_TRUE(report.kokkos_concurrency == 0) << "inactive_kokkos_concurrency";
  }
  EXPECT_TRUE(report.allocator_mode == "kokkos_shared_space_managed_arena") << "managed_arena";
  EXPECT_TRUE(report.comm_allocator_mode == "kokkos_shared_host_pinned_space")
      << "pinned_comm_allocator";
  EXPECT_TRUE(report.allocator_lifetime.find("process-lifetime") != std::string::npos)
      << "allocator_lifetime_reported";

  if (!report.kokkos_initialized && !report.kokkos_finalized) {
    detail::ensure_kokkos_initialized();
    const RuntimeEnvironmentReport initialized_report = runtime_environment_report();
    EXPECT_TRUE(initialized_report.kokkos_initialized) << "kokkos_initialized_for_exact_probe";
    EXPECT_TRUE(initialized_report.kokkos_concurrency ==
                Kokkos::DefaultExecutionSpace{}.concurrency())
        << "exact_default_execution_space_concurrency";
  }
#else
  EXPECT_TRUE(!report.has_kokkos) << "no_kokkos";
  EXPECT_TRUE(report.kokkos_concurrency == 0) << "no_kokkos_concurrency";
  EXPECT_TRUE(report.allocator_mode == "std_allocator") << "std_allocator";
#endif
}
