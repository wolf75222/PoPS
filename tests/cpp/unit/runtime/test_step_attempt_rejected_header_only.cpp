#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>

#include <gtest/gtest.h>

#include <string>

#if defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI) || defined(POPS_EXPORT_BUILDING_MODULE)
#error "a direct pops::pops consumer must retain the header-only exception contract"
#endif

namespace {

using pops::SolveStatus;
using pops::runtime::program::StepAttemptDisposition;
using pops::runtime::program::StepAttemptRejected;
using pops::runtime::program::ProgramStepRejectMailbox;
using pops::runtime::program::ProgramStepRejectRecord;

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

TEST(StepAttemptRejectedHeaderOnly, FluxAttemptDispositionAndReasonRemainStructured) {
  try {
    throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, StepAttemptDisposition::kRetry,
                              0x1234u, "stage", "external flux requested retry");
  } catch (const StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.status(), SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(rejected.disposition(), StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), 0x1234u);
    EXPECT_EQ(rejected.phase(), "stage");
    EXPECT_NE(std::string(rejected.what()).find("attempt_action=retry"), std::string::npos);
    return;
  }
  FAIL() << "flux-driven typed rejection was not caught";
}

TEST(StepAttemptRejectedHeaderOnly, MailboxRejectsNonCanonicalRawRecordsBeforeAdoption) {
  ProgramStepRejectMailbox mailbox;
  mailbox.arm(7, 11);
  ProgramStepRejectRecord record{};
  record.generation = 7;
  record.attempt = 11;
  record.status = static_cast<std::uint32_t>(SolveStatus::kIterationLimit);
  record.disposition = static_cast<std::uint32_t>(StepAttemptDisposition::kRetry);
  record.reserved = 1;
  EXPECT_FALSE(mailbox.adopt(record));

  record.reserved = 0;
  record.disposition = 2;
  EXPECT_FALSE(mailbox.adopt(record));

  record.disposition = static_cast<std::uint32_t>(StepAttemptDisposition::kRetry);
  ASSERT_TRUE(mailbox.adopt(record));
  ProgramStepRejectRecord consumed{};
  ASSERT_EQ(mailbox.consume(7, 11, consumed), ProgramStepRejectMailbox::ConsumeResult::valid);
  EXPECT_EQ(consumed.reason_code, 0u);
  EXPECT_EQ(consumed.disposition, static_cast<std::uint32_t>(StepAttemptDisposition::kRetry));
}

TEST(StepAttemptRejectedHeaderOnly, PublishedInvalidRecordFailsClosedAndIsConsumedOnce) {
  ProgramStepRejectMailbox mailbox;
  mailbox.arm(17, 23);
  ProgramStepRejectRecord published{};
  ASSERT_TRUE(mailbox.publish(static_cast<SolveStatus>(UINT32_C(0xffffffff)),
                              StepAttemptDisposition::kRetry, 0, "stage", "bad", published));
  ProgramStepRejectRecord consumed{};
  EXPECT_EQ(mailbox.consume(17, 23, consumed), ProgramStepRejectMailbox::ConsumeResult::invalid);
  EXPECT_EQ(mailbox.consume(17, 23, consumed), ProgramStepRejectMailbox::ConsumeResult::none);
}

}  // namespace
