#include <pops/runtime/program/step_transaction.hpp>

#include <gtest/gtest.h>

#include <string>
#include <typeinfo>

#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI) || !defined(POPS_EXPORT_BUILDING_MODULE)
#error "an AMR host consumer must inherit the shared exception ABI producer contract"
#endif

namespace {

using pops::SolveStatus;
using pops::runtime::program::StepAttemptRejected;

TEST(StepAttemptRejectedAmrLink, AmrFacadeCarriesCanonicalRuntimeTypeinfoTransitively) {
  try {
    throw StepAttemptRejected(SolveStatus::kIterationLimit, "guard", "retry");
  } catch (const StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.status(), SolveStatus::kIterationLimit);
    EXPECT_EQ(rejected.phase(), "guard");
    EXPECT_EQ(rejected.detail(), "retry");
    EXPECT_EQ(typeid(rejected), typeid(StepAttemptRejected));
    EXPECT_NE(std::string(rejected.what()).find("step attempt rejected during guard"),
              std::string::npos);
    return;
  }
  FAIL() << "typed rejection was not caught through its canonical RTTI";
}

}  // namespace
