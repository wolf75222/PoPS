// Exact-ranked ProgramContext compile-fire plus grid-free cadence transaction proofs.

#include <gtest/gtest.h>

#include <pops/runtime/program/program_context.hpp>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

static_assert(std::is_class_v<pops::runtime::program::ProgramContext<1>>);
static_assert(std::is_class_v<pops::runtime::program::ProgramContext<2>>);
static_assert(std::is_class_v<pops::runtime::program::ProgramContext<3>>);
static_assert(!std::is_trivially_constructible_v<pops::runtime::program::ProgramContext<2>>);

TEST(ProgramContextSchurFree, ExactRankedHeaderIsSelfContainedAndRejectsNullRuntime) {
  EXPECT_THROW((void)pops::runtime::program::ProgramContext<2>(nullptr), std::invalid_argument);
}

TEST(ProgramRuntimeStateCadence, SharedDispatcherOwnsHoldSubstepAndCursorCommit) {
  pops::runtime::program::ProgramRuntimeState<2> state;
  struct Dispatch {
    double start = 0.0;
    double dt = 0.0;
    int macro_step = -1;
  };
  std::vector<Dispatch> dispatches;
  double physical_time = 2.0;
  int macro_step = 4;
  state.install_unverified_step(
      [&](double dt) { dispatches.push_back({physical_time, dt, macro_step}); });
  state.set_cadence(/*substeps=*/2, /*stride=*/2, "Fixture");

  state.dispatch_cadence_step(physical_time, macro_step, 0.1, "Fixture");
  EXPECT_TRUE(dispatches.empty());
  EXPECT_DOUBLE_EQ(physical_time, 2.1);
  EXPECT_EQ(macro_step, 5);

  state.dispatch_cadence_step(physical_time, macro_step, 0.3, "Fixture");
  ASSERT_EQ(dispatches.size(), 2);
  EXPECT_DOUBLE_EQ(dispatches[0].start, 2.0);
  EXPECT_EQ(dispatches[0].macro_step, 4);
  EXPECT_DOUBLE_EQ(dispatches[1].start, dispatches[0].start + dispatches[0].dt);
  EXPECT_EQ(macro_step, 6);
}

TEST(ProgramRuntimeStateCadence, DispatchFailureRestoresCursorAndReentrancyLease) {
  pops::runtime::program::ProgramRuntimeState<3> state;
  double physical_time = 1.0;
  int macro_step = 0;
  int calls = 0;
  bool fail_second_substep = true;
  state.install_unverified_step([&](double) {
    ++calls;
    if (fail_second_substep && calls == 2)
      throw std::runtime_error("injected cadence substep failure");
  });
  state.set_cadence(/*substeps=*/2, /*stride=*/1, "Fixture");

  EXPECT_THROW(state.dispatch_cadence_step(physical_time, macro_step, 0.4, "Fixture"),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(physical_time, 1.0);
  EXPECT_EQ(macro_step, 0);
  EXPECT_FALSE(state.cadence_dispatch_active_);

  calls = 0;
  fail_second_substep = false;
  EXPECT_NO_THROW(state.dispatch_cadence_step(physical_time, macro_step, 0.4, "Fixture"));
  EXPECT_EQ(calls, 2);
  EXPECT_DOUBLE_EQ(physical_time, 1.4);
  EXPECT_EQ(macro_step, 1);
}

TEST(ProgramRuntimeStateCadence, MacroStepOverflowFailsBeforeProgramDispatch) {
  pops::runtime::program::ProgramRuntimeState<1> state;
  double physical_time = 0.0;
  int macro_step = std::numeric_limits<int>::max();
  int calls = 0;
  state.install_unverified_step([&](double) { ++calls; });

  EXPECT_THROW(state.dispatch_cadence_step(physical_time, macro_step, 0.1, "Fixture"),
               std::overflow_error);
  EXPECT_EQ(calls, 0);
  EXPECT_DOUBLE_EQ(physical_time, 0.0);
  EXPECT_EQ(macro_step, std::numeric_limits<int>::max());
}
