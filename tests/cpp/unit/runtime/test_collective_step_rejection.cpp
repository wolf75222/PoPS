#include <gtest/gtest.h>

#include <pops/runtime/program/collective_step_rejection.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/program/typed_failure.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
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
    const std::string body(
        "\0\0\0\0\0\0\0\4"
        "\0\0\0\0\0\0\0\0"
        "\0\0\0\0\xff\xff\xff\xff"
        "\0\0\0\0\0\0\0\5"
        "stage"
        "\0\0\0\0\0\0\0\5"
        "retry",
        50);
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

TEST(CollectiveStepRejection, TypedPayloadKeepsExactBytesAndCannotBecomeLegacyOnTruncation) {
  const std::string payload("\0\xff\x80payload", 10);
  program::StepAttemptRejected original(pops::SolveStatus::kInvalidEvaluation,
                                        program::StepAttemptDisposition::kReject, 42, "stage",
                                        "typed", "closed-type", payload);
  for (const auto contract : contracts) {
    const auto encoded = wire::encode(original, contract);
    const auto restored = wire::decode(encoded, contract);
    EXPECT_EQ(restored.typed_failure_schema, "closed-type");
    EXPECT_EQ(restored.typed_failure_payload, payload);
    for (std::size_t size = 0; size < encoded.size(); ++size)
      EXPECT_THROW(wire::decode(std::string_view(encoded).substr(0, size), contract),
                   std::runtime_error);
    EXPECT_THROW(wire::decode(encoded + "x", contract), std::runtime_error);
  }
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.typed-failure.wire");
  try {
    program::collective_step_rejection_phase(lane.communicator(), contracts[0], "fatal",
                                             [&] { throw original; });
    FAIL() << "typed error was lost";
  } catch (const program::StepAttemptRejected& caught) {
    EXPECT_EQ(caught.typed_failure_schema(), "closed-type");
    EXPECT_EQ(caught.typed_failure_payload(), payload);
  }
}

TEST(CollectiveStepRejection, SubsetAndAllRankRejectionPreserveControlOnEveryRank) {
  auto lane =
      pops::ExecutionLane::duplicate_world_collectively("test.collective-rejection.control");
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

TEST(CollectiveStepRejection, TypedFailureAggregateCapacityIncludesEveryEncodedByte) {
  constexpr int Dim = pops::kNativeDimension;
  using Kind = program::ContinuationKind;
  using Value = program::ContinuationValue<Dim>;
  using Codec = program::ContinuationCodec<Dim>;
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.typed-failure.capacity");
  constexpr std::uint64_t ample = 1024 * 1024;
  const Codec::Prototypes prototypes = [](int) -> const pops::MultiFab<Dim>& {
    throw std::logic_error("this capacity test contains no field leaves");
  };
  // Independent framing oracle: never call typed_failure_detail::word or reproduce its
  // capacity arithmetic. Derive the boundary from the payload actually published below.
  const auto append_word = [](std::string& bytes, std::uint64_t value) {
    for (unsigned byte = 0; byte < sizeof(value); ++byte)
      bytes.push_back(static_cast<char>((value >> (8 * (sizeof(value) - 1 - byte))) & 0xff));
  };
  for (bool record : {false, true}) {
    SCOPED_TRACE(record ? "Product(Boolean, Int64)" : "Boolean");
    program::ContinuationType type;
    type.kind = Kind::Boolean;
    if (record) {
      type.kind = Kind::Product;
      program::ContinuationType boolean, integer;
      boolean.kind = Kind::Boolean;
      integer.kind = Kind::Int64;
      type.children = {boolean, integer};
      type.names = {"flag", "exact"};
    }
    const std::string schema =
        record ? "test.typed-capacity.product.v1" : "test.typed-capacity.boolean.v1";
    const auto value_at = [record](int rank) {
      auto flag = Value::boolean(rank % 2 == 0);
      if (!record)
        return flag;
      return Value::product({flag, Value::integer((INT64_C(1) << 60) + rank)});
    };
    const auto prepare = [&] { return std::make_tuple(type, value_at(lane.rank()), schema); };
    const auto capture = [&](std::uint64_t aggregate_capacity) {
      std::string bytes;
      bool caught_typed = false;
      try {
        program::reject_typed_failure<Dim>(lane, prepare, prototypes, {ample, aggregate_capacity});
      } catch (const program::StepAttemptRejected& failure) {
        caught_typed = true;
        EXPECT_EQ(failure.status(), pops::SolveStatus::kInvalidEvaluation);
        EXPECT_EQ(failure.disposition(), program::StepAttemptDisposition::kReject);
        EXPECT_EQ(failure.typed_failure_schema(), schema);
        bytes = failure.typed_failure_payload();
      } catch (const std::exception& error) {
        ADD_FAILURE() << "complete typed payload was rejected: " << error.what();
      }
      EXPECT_EQ(pops::all_reduce_min(caught_typed ? 1L : 0L, lane.communicator()), 1);
      return bytes;
    };
    const auto observed = capture(ample);
    const long have_payload = observed.empty() ? 0L : 1L;
    ASSERT_EQ(pops::all_reduce_min(have_payload, lane.communicator()), 1);

    // Construct every rank shard from its independent known data, retaining exact Int64.
    // The magic is a string, so its size is never replaced with an assumed word width.
    std::string expected = "POPSFAIL1";
    append_word(expected, static_cast<std::uint64_t>(lane.size()));
    Codec codec(type, schema, ample, prototypes);
    for (int rank = 0; rank < lane.size(); ++rank) {
      const auto shard = codec.encode(value_at(rank), 0);
      append_word(expected, shard.size());
      expected.append(reinterpret_cast<const char*>(shard.data()), shard.size());
    }
    EXPECT_EQ(observed, expected);
    const auto exact_capacity = static_cast<std::uint64_t>(observed.size());
    EXPECT_EQ(capture(exact_capacity), expected);

    // Both an everywhere-small bound and a single-rank bound must fail collectively,
    // before any rank publishes a retry/rejection carrying an oversized aggregate.
    for (bool rank_local : {false, true}) {
      const auto capacity = exact_capacity - ((!rank_local || lane.rank() == 0) ? 1 : 0);
      bool fatal_capacity = false;
      try {
        program::reject_typed_failure<Dim>(lane, prepare, prototypes, {ample, capacity});
      } catch (const program::StepAttemptRejected&) {
        ADD_FAILURE() << "one byte below actual aggregate size published a typed rejection";
      } catch (const std::exception& error) {
        fatal_capacity =
            std::string(error.what()).find("aggregate diagnostic capacity") != std::string::npos;
        EXPECT_TRUE(fatal_capacity) << error.what();
      }
      EXPECT_EQ(pops::all_reduce_sum(fatal_capacity ? 1L : 0L, lane.communicator()),
                static_cast<long>(lane.size()));
    }
  }
}

TEST(CollectiveStepRejection, DeepTypedFailureKeepsCompletePayloadAndRejectsInsufficientWorkspace) {
  constexpr int Dim = pops::kNativeDimension;
  using Value = program::ContinuationValue<Dim>;
  using Kind = program::ContinuationKind;
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.typed-failure.deep");
  program::ContinuationType type{Kind::Int64};
  auto value = Value::integer((INT64_C(1) << 60) + lane.rank());
  const auto leaf = value;
  for (int depth = 0; depth < 2048; ++depth) {
    program::ContinuationType parent{Kind::Product};
    parent.names.push_back("detail");
    parent.children.push_back(std::move(type));
    type = std::move(parent);
    value = Value::product({std::move(value)});
  }
  constexpr std::uint64_t ample = 8 * 1024 * 1024;
  const std::string schema = "test.deep-typed-failure@1";
  const program::ContinuationCodec<Dim>::Prototypes prototypes =
      [](int) -> const pops::MultiFab<Dim>& { throw std::logic_error("no fields"); };
  const auto prepare = [&] { return std::make_tuple(type, value, schema); };
  bool complete = false;
  try {
    program::reject_typed_failure<Dim>(lane, prepare, prototypes, {ample, ample});
  } catch (const program::StepAttemptRejected& error) {
    complete = true;
    EXPECT_EQ(error.typed_failure_schema(), schema);
    const auto& bytes = error.typed_failure_payload();
    ASSERT_GT(bytes.size(), 25u);
    std::size_t cursor = 17;
    program::ContinuationCodec<Dim> codec(type, schema, ample, prototypes);
    for (int rank = 0; rank < lane.size(); ++rank) {
      const auto size =
          program::typed_failure_detail::word(std::string_view(bytes).substr(cursor, 8));
      cursor += 8;
      const auto* begin = reinterpret_cast<const std::uint8_t*>(bytes.data() + cursor);
      auto restored = codec.decode(std::vector<std::uint8_t>(begin, begin + size)).root;
      for (int depth = 0; depth < 2048; ++depth)
        restored = restored->members().at(0);
      EXPECT_EQ(restored->integer(), (INT64_C(1) << 60) + rank);
      cursor += size;
    }
    EXPECT_EQ(cursor, bytes.size());
    EXPECT_NE(std::string(error.what()).find("...="), std::string::npos);
  }
  EXPECT_TRUE(complete);
  bool refused_before_typed_publication = false;
  try {
    program::reject_typed_failure<Dim>(lane, prepare, prototypes, {1, ample});
  } catch (const program::StepAttemptRejected&) {
    ADD_FAILURE() << "insufficient capacity published a truncated typed failure";
  } catch (const std::exception&) {
    refused_before_typed_publication = true;
  }
  EXPECT_TRUE(refused_before_typed_publication);
  EXPECT_EQ(leaf->integer(), (INT64_C(1) << 60) + lane.rank());
}
