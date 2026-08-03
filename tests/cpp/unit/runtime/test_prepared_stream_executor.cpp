#include <gtest/gtest.h>

#include <pops/runtime/accelerator/prepared_stream_executor.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using pops::runtime::accelerator::PreparedAcceleratorStreamExecutor;
using pops::runtime::accelerator::PreparedStreamPartitionError;

namespace {

using Executor = PreparedAcceleratorStreamExecutor<double>;

TEST(PreparedStreamExecutor, InvalidPreparationIsRejectedBeforeBackendSelection) {
  EXPECT_THROW((void)Executor::prepare(1, 64), std::invalid_argument);
  EXPECT_THROW((void)Executor::prepare(2, 0), std::invalid_argument);
  EXPECT_THROW((void)Executor::prepare(2, 64, {1.0}), std::invalid_argument);
  EXPECT_THROW((void)Executor::prepare(2, 64, {1.0, 0.0}), std::invalid_argument);
  EXPECT_THROW((void)Executor::prepare(2, 64, {1.0, std::numeric_limits<double>::quiet_NaN()}),
               std::invalid_argument);
}

TEST(PreparedStreamExecutor, CpuBackendsCannotClaimIndependentAcceleratorStreams) {
  if constexpr (!Executor::backend_can_partition_authentic_streams()) {
    EXPECT_THROW((void)Executor::prepare(2, 64), PreparedStreamPartitionError);
  } else {
    GTEST_SKIP() << "This assertion is the fail-closed CPU half of the backend matrix";
  }
}

TEST(PreparedStreamExecutor, AcceleratorInstancesLaunchOnExplicitDisjointLanes) {
  if constexpr (!Executor::backend_can_partition_authentic_streams()) {
    GTEST_SKIP() << "requires a Kokkos CUDA, HIP, or SYCL execution space";
  } else {
    constexpr std::int64_t values = 4096;
    Executor executor = Executor::prepare(2, static_cast<std::size_t>(values));

    ASSERT_EQ(executor.size(), 2u);
    EXPECT_EQ(executor.workspace_values_per_stream(), static_cast<std::size_t>(values));
    EXPECT_TRUE(executor.evidence().independent_streams);
    EXPECT_TRUE(executor.evidence().disjoint_workspaces);
    EXPECT_NE(executor.workspace_address(0), executor.workspace_address(1));
    EXPECT_EQ(std::set<std::string>(executor.evidence().stream_identities.begin(),
                                    executor.evidence().stream_identities.end())
                  .size(),
              2u);

    double* lane_zero = executor.workspace_data(0);
    double* lane_one = executor.workspace_data(1);
    executor.launch_for(
        0, "pops_test_prepared_stream_lane_zero", values, KOKKOS_LAMBDA(std::int64_t index) {
          lane_zero[index] = 2.0 * static_cast<double>(index) + 1.0;
        });
    executor.launch_for(
        1, "pops_test_prepared_stream_lane_one", values, KOKKOS_LAMBDA(std::int64_t index) {
          lane_one[index] = 3.0 * static_cast<double>(index) - 2.0;
        });
    executor.fence_all();

    const auto zero_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(0));
    const auto one_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(1));
    for (std::int64_t index = 0; index < values; ++index) {
      EXPECT_DOUBLE_EQ(zero_host(index), 2.0 * static_cast<double>(index) + 1.0);
      EXPECT_DOUBLE_EQ(one_host(index), 3.0 * static_cast<double>(index) - 2.0);
    }

    EXPECT_THROW((void)executor.workspace(2), std::out_of_range);
    EXPECT_THROW(executor.launch_for(0, "", 1, KOKKOS_LAMBDA(std::int64_t){}),
                 std::invalid_argument);
    EXPECT_THROW(executor.launch_for(0, "negative", -1, KOKKOS_LAMBDA(std::int64_t){}),
                 std::invalid_argument);
  }
}

}  // namespace
