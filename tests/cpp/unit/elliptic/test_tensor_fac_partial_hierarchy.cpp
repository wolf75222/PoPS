#include <gtest/gtest.h>

#include "partial_mapped_disk_fac_witness.hpp"

namespace {
using namespace pops;

void expect_partial_disk(int levels) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.partial-mapped-disk");
  if (lane.size() != 1 && lane.size() != 2)
    GTEST_SKIP() << "partial disk witness exercises MPI worlds 1 and 2";
  const auto result = test::partial_mapped_disk_fac_witness(levels, lane);
  SCOPED_TRACE(result.report.reason);
  ASSERT_TRUE(result.all_values_finite);
  EXPECT_EQ(result.active_cells, levels == 3 ? (std::vector<long>{512, 768, 5120})
                                             : (std::vector<long>{512, 2048}));
  EXPECT_EQ(result.remote_parent_gather, lane.size() == 2);
  EXPECT_EQ(result.reference, result.report.reference_residual_norm);
  EXPECT_LE(result.coefficient_ghost_difference,
            Real(64) * std::numeric_limits<Real>::epsilon() * result.coefficient_scale);
  ASSERT_TRUE(result.report.solved()) << "iterations=" << result.report.iters
      << " original residual=" << result.independent_residual;
  const Real target = std::max(Real(1e-12), Real(1e-10) * result.reference);
  EXPECT_LE(result.independent_residual, target);
  EXPECT_LE(result.maximum_error, Real(1e-10));
}

TEST(TensorFacPartialHierarchy, TwoLevelDiscreteManufacturedControl) { expect_partial_disk(2); }

// With the old cycle, covered intermediate values retained the pre-correction iterate while
// children had received the correction; the next fine post-smoother consumed stale parents.
TEST(TensorFacPartialHierarchy, ThreeLevelCorrectionKeepsParentGhostDataConsistent) {
  expect_partial_disk(3);
}
}  // namespace
