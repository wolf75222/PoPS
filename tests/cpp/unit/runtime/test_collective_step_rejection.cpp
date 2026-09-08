#include <gtest/gtest.h>

#include <pops/runtime/program/collective_step_rejection.hpp>

#include <array>
#include <limits>

namespace {
namespace program = pops::runtime::program;
namespace wire = program::step_rejection_detail;
constexpr std::array<program::StepRejectionContract, 3> contracts{{
    {"pops.step-rejection.v1", "step-rejection", false, false},
    {"pops.boundary-step-rejection.v1", "boundary-step-rejection", true, false},
    {"pops.amr-tagger.step-rejection.v1", "amr-tagger-step-rejection", true, true},
}};

program::StepAttemptRejected control(std::string phase = "stage", std::string detail = "retry") {
  return {pops::SolveStatus::kInvalidEvaluation, program::StepAttemptDisposition::kRetry,
          std::numeric_limits<std::uint32_t>::max(), std::move(phase), std::move(detail)};
}

void check_control(const program::StepAttemptRejected& caught) {
  EXPECT_EQ(caught.status(), pops::SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(caught.disposition(), program::StepAttemptDisposition::kRetry);
  EXPECT_EQ(caught.reason_code(), std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(caught.phase(), "stage");
  EXPECT_EQ(caught.detail(), "retry");
}
}  // namespace

TEST(CollectiveStepRejection, WireRetainsEveryTypedFieldAndCallerDiagnosticRequirements) {
  for (const auto contract : contracts) {
    const auto encoded = wire::encode(control(), contract);
    // Frozen legacy layout: three big-endian integers followed by two length-prefixed texts.
    const std::string body("\0\0\0\0\0\0\0\4" "\0\0\0\0\0\0\0\0"
                           "\0\0\0\0\xff\xff\xff\xff" "\0\0\0\0\0\0\0\5" "stage"
                           "\0\0\0\0\0\0\0\5" "retry", 50);
    EXPECT_EQ(encoded, std::string(contract.schema) + body);
    const auto restored = wire::decode(encoded, contract);
    EXPECT_EQ(restored.status, pops::SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(restored.disposition, program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(restored.reason_code, std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(restored.phase, "stage");
    EXPECT_EQ(restored.detail, "retry");
    EXPECT_THROW(wire::decode(encoded + "x", contract), std::runtime_error);
    for (std::size_t size = 0; size < encoded.size(); ++size)
      EXPECT_THROW(wire::decode(std::string_view(encoded).substr(0, size), contract),
                   std::runtime_error);
    auto corrupt = encoded;
    for (int field = 0; field < 3; ++field) {
      corrupt = encoded;
      corrupt[contract.schema.size() + static_cast<std::size_t>(field) * 8] = '\x7f';
      EXPECT_THROW(wire::decode(corrupt, contract), std::runtime_error);
    }
    const auto no_phase = wire::encode(control("", "retry"), contract);
    const auto no_detail = wire::encode(control("stage", ""), contract);
    if (contract.require_phase)
      EXPECT_THROW(wire::decode(no_phase, contract), std::runtime_error);
    else
      EXPECT_NO_THROW(wire::decode(no_phase, contract));
    if (contract.require_detail)
      EXPECT_THROW(wire::decode(no_detail, contract), std::runtime_error);
    else
      EXPECT_NO_THROW(wire::decode(no_detail, contract));
  }
  EXPECT_THROW(wire::decode(wire::encode(control(), contracts[0]), contracts[1]),
               std::runtime_error);
}

TEST(CollectiveStepRejection, SubsetAndAllRankRejectionPreserveControlOnEveryRank) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.collective-rejection.control");
  for (const auto contract : contracts) {
    for (bool all : {false, true}) {
      try {
        program::collective_step_rejection_phase(lane.communicator(), contract, "fatal", [&] {
          if (all || lane.rank() == 0)
            throw control();
        });
        FAIL() << "rejection was lost";
      } catch (const program::StepAttemptRejected& caught) {
        check_control(caught);
      }
    }
  }
}

TEST(CollectiveStepRejection, DisagreeingRejectingRanksFailInsteadOfSelectingOneReason) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.collective-rejection.exact");
  if (lane.size() < 2)
    GTEST_SKIP() << "requires disagreeing ranks";
  try {
    program::collective_step_rejection_phase(lane.communicator(), contracts[0], "fatal", [&] {
      throw control("stage", std::to_string(lane.rank()));
    });
    FAIL() << "different typed envelopes were accepted";
  } catch (const program::StepAttemptRejected&) {
    FAIL() << "selected one of the disagreeing reasons";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("fields differ"), std::string::npos);
  }
}

TEST(CollectiveStepRejection, OrdinaryErrorHasPriorityOverAnotherRanksRetry) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.collective-rejection.fatal");
  try {
    program::collective_step_rejection_phase(lane.communicator(), contracts[0], "fatal phase", [&] {
      if (lane.rank() == 0)
        throw std::logic_error("physical provider failed");
      throw control();
    });
    FAIL() << "ordinary failure was lost";
  } catch (const program::StepAttemptRejected&) {
    FAIL() << "ordinary failure was downgraded to retry";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("physical provider failed"), std::string::npos);
    if (lane.size() == 1)
      EXPECT_NE(dynamic_cast<const std::logic_error*>(&error), nullptr);
  }
}

TEST(CollectiveStepRejection, CallerControlsSuccessOnlyAndUnconditionalCompletion) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.collective-rejection.fences");
  for (bool reject : {false, true}) {
    int success_fence = 0, unconditional_fence = 0;
    try {
      program::collective_step_rejection_phase(
          lane.communicator(), contracts[0], "fatal",
          [&] {
            if (reject)
              throw control();
            ++success_fence;
          },
          [&] { ++unconditional_fence; });
      EXPECT_FALSE(reject);
    } catch (const program::StepAttemptRejected& caught) {
      EXPECT_TRUE(reject);
      check_control(caught);
    }
    EXPECT_EQ(success_fence, reject ? 0 : 1);
    EXPECT_EQ(unconditional_fence, 1);
  }
  try {
    program::collective_step_rejection_phase(
        lane.communicator(), contracts[0], "completion phase", [] { throw control(); },
        [] { throw std::logic_error("completion failed"); });
    FAIL() << "completion failure was lost";
  } catch (const program::StepAttemptRejected&) {
    FAIL() << "completion failure was downgraded to retry";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("completion failed"), std::string::npos);
  }
}

TEST(CollectiveStepRejection, MalformedTypedDiagnosticFailsCollectivelyBeforeRetry) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.collective-rejection.decode");
  try {
    program::collective_step_rejection_phase(lane.communicator(), contracts[2], "fatal", [&] {
      if (lane.rank() == 0)
        throw control("stage", "");
    });
    FAIL() << "missing required detail was accepted";
  } catch (const program::StepAttemptRejected&) {
    FAIL() << "malformed control escaped validation";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("incomplete"), std::string::npos);
  }
}
