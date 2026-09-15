#include <gtest/gtest.h>
#include "mapped_disk_fac_witness.hpp"
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>

TEST(MappedDiskTensorFAC, RejectsBoundaryMasksThatDisagreeWithNativeTopology) {
  const auto lane = pops::ExecutionLane::duplicate_world_collectively("tests.mapped-disk-fac-boundary");
  using Options = pops::elliptic::nd::CartesianTensorStencilOptions;
  for (const auto options : {Options{16u, 2u, true}, Options{1u, 1u, true},
                            Options{4u, 2u, true}, Options{2u, 1u, true}})
    EXPECT_THROW((void)pops::test::mapped_disk_fac_witness(8, 0, false, lane, options),
                 std::invalid_argument);
}

TEST(MappedDiskTensorFAC, RejectsInvalidAuthoredDampingInNativeWireContract) {
  using namespace pops::runtime::program::tensor_elliptic_detail;
  for (const double damping : {0.0, -0.5, 1.5, std::numeric_limits<double>::infinity()}) {
    auto options = default_options();
    options.values.emplace("fac.correction_damping", damping);
    EXPECT_THROW((void)decode_controls(options), std::invalid_argument);
  }
  auto options = default_options();
  options.values.emplace("fac.correction_damping", std::int64_t{1});
  EXPECT_THROW((void)decode_controls(options), std::invalid_argument);
  options.values["fac.correction_damping"] = 0.5;
  EXPECT_DOUBLE_EQ(*decode_controls(options).correction_damping, 0.5);
}

TEST(MappedDiskTensorFAC, MappedFullDiskConvergesWithPartialPeriodicAMR) {
  const auto lane = pops::ExecutionLane::duplicate_world_collectively("tests.mapped-disk-fac-mms");
  const auto coarse = pops::test::mapped_disk_fac_witness(8, 48, false, lane,
                                                       {1u, 2u, true}, 4, 64, 512, 200, 0.5);
  const auto fine = pops::test::mapped_disk_fac_witness(16, 96, false, lane,
                                                     {1u, 2u, true}, 4, 64, 512, 200, 0.5);
  ASSERT_TRUE(coarse.report.solved()) << coarse.report.reason;
  ASSERT_TRUE(fine.report.solved()) << fine.report.reason;
  EXPECT_LT(fine.maximum_error, coarse.maximum_error / 3.0);
}
