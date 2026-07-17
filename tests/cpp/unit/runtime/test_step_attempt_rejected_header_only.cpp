#include <pops/runtime/program/step_transaction.hpp>

#include <gtest/gtest.h>

#include <string>

#if defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI) || defined(POPS_EXPORT_BUILDING_MODULE)
#error "a direct pops::pops consumer must retain the header-only exception contract"
#endif

namespace {

using pops::SolveStatus;
using pops::runtime::program::StepAttemptRejected;

TEST(StepAttemptRejectedHeaderOnly, DirectPopsTargetThrowsAndCatchesWithoutRuntimeLibrary) {
  try {
    throw StepAttemptRejected(SolveStatus::kBreakdown, "solve", "header-only");
  } catch (const StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.status(), SolveStatus::kBreakdown);
    EXPECT_EQ(rejected.phase(), "solve");
    EXPECT_EQ(rejected.detail(), "header-only");
    EXPECT_NE(std::string(rejected.what()).find("step attempt rejected during solve"),
              std::string::npos);
    return;
  }
  FAIL() << "header-only typed rejection was not caught";
}

}  // namespace
