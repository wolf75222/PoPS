#include <pops/runtime/program/program_transaction.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/core/foundation/allocator.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace pops::runtime::program;

struct State final {
  int accepted = 1;
  int candidate = 1;
  bool fail_snapshot = false;
  bool fail_publish = false;
  int snapshots = 0;
  int restores = 0;
  int publishes = 0;

  static bool snapshot(void* object, void* image, std::size_t bytes) noexcept {
    auto& state = *static_cast<State*>(object);
    ++state.snapshots;
    if (state.fail_snapshot || bytes != sizeof(state.accepted))
      return false;
    *static_cast<int*>(image) = state.accepted;
    return true;
  }

  static void restore(void* object, const void* image, std::size_t bytes) noexcept {
    auto& state = *static_cast<State*>(object);
    ++state.restores;
    if (bytes == sizeof(state.accepted))
      state.accepted = *static_cast<const int*>(image);
  }

  static bool publish(void* object) noexcept {
    auto& state = *static_cast<State*>(object);
    ++state.publishes;
    if (state.fail_publish)
      return false;
    state.accepted = state.candidate;
    return true;
  }

  static void* candidate_view(void* object) noexcept {
    // Candidate and accepted values live in one typed participant object in this fixture.  A
    // production participant can return a detached T carrier here; the callback contract remains
    // type-preserving for ProvisionalView<T>.
    return object;
  }
};

struct MethodParticipantState final {
  int accepted = 0;

  bool snapshot(void* image, std::size_t bytes) noexcept {
    if (bytes != sizeof(accepted))
      return false;
    *static_cast<int*>(image) = accepted;
    return true;
  }
  void restore(const void* image, std::size_t bytes) noexcept {
    if (bytes == sizeof(accepted))
      accepted = *static_cast<const int*>(image);
  }
  bool publish() noexcept { return true; }
};

struct EffectLog final {
  std::mutex mutex;
  std::vector<int> events;
  std::vector<int>* sequence = nullptr;
  int identity = 0;
  bool prepare_ok = true;
  bool publish_ok = true;
  bool finalize_ok = true;
  int prepare_calls = 0;
  int publish_calls = 0;
  int compensate_calls = 0;
  int discard_calls = 0;
  int finalize_calls = 0;

  static bool prepare(void* context) noexcept {
    auto& effect = *static_cast<EffectLog*>(context);
    ++effect.prepare_calls;
    return effect.prepare_ok;
  }
  static bool publish(void* context) noexcept {
    auto& effect = *static_cast<EffectLog*>(context);
    ++effect.publish_calls;
    // Model a publisher that can discover failure only after its external mutation.  The
    // transaction must compensate this attempted publication as well as earlier successful ones.
    std::lock_guard<std::mutex> lock(effect.mutex);
    effect.events.push_back(effect.identity);
    if (effect.sequence != nullptr)
      effect.sequence->push_back(effect.identity);
    return effect.publish_ok;
  }
  static void compensate(void* context) noexcept {
    auto& effect = *static_cast<EffectLog*>(context);
    ++effect.compensate_calls;
    std::lock_guard<std::mutex> lock(effect.mutex);
    effect.events.push_back(-effect.identity);
    if (effect.sequence != nullptr)
      effect.sequence->push_back(-effect.identity);
  }
  static void discard(void* context) noexcept { ++static_cast<EffectLog*>(context)->discard_calls; }
  static bool finalize(void* context) noexcept {
    auto& effect = *static_cast<EffectLog*>(context);
    ++effect.finalize_calls;
    return effect.finalize_ok;
  }

  PreparedCompensableEffect prepared() noexcept {
    return PreparedCompensableEffect(this, &EffectLog::publish, &EffectLog::compensate,
                                     &EffectLog::finalize, &EffectLog::discard, &EffectLog::prepare,
                                     static_cast<std::uint64_t>(identity));
  }
};

struct ConsensusWordLog final {
  std::array<std::uint32_t, 32> words{};
  std::array<std::uint32_t, 32> phases{};
  std::size_t count = 0;
  std::uint32_t rejected_word = std::numeric_limits<std::uint32_t>::max();

  static bool agree(void* context, std::uint32_t phase, std::uint32_t status) noexcept {
    auto& log = *static_cast<ConsensusWordLog*>(context);
    if (log.count < log.words.size()) {
      log.phases[log.count] = phase;
      log.words[log.count] = status;
    }
    ++log.count;
    return status != log.rejected_word;
  }
};

ProgramParticipantOps state_ops() noexcept {
  ProgramParticipantOps ops;
  ops.snapshot = &State::snapshot;
  ops.restore = &State::restore;
  ops.publish = &State::publish;
  ops.candidate = &State::candidate_view;
  return ops;
}

TEST(ProgramTransaction, BindFreezesTypedOrderBudgetsAndHotCapacity) {
  State first;
  State second;
  ProgramTransactionRegistry registry({2, 2 * sizeof(int), 2 * sizeof(int), 1});
  const auto first_handle =
      registry.register_participant(first, state_ops(), {sizeof(int), sizeof(int)});
  const auto second_handle =
      registry.register_participant(second, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(1);
  EXPECT_EQ(first_handle.index, 0u);
  EXPECT_EQ(second_handle.index, 1u);
  ASSERT_TRUE(effect_handle);
  EXPECT_EQ(registry.effect_info(0).status, ProgramEffectSlotStatus::kDeclared);
  EXPECT_THROW((void)registry.register_effect(1), std::invalid_argument);
  EXPECT_FALSE(registry.try_register_effect(1));
  EXPECT_THROW((void)registry.register_effect(2), std::length_error);
  EXPECT_FALSE(registry.try_register_effect(2));

  registry.bind();
  EXPECT_TRUE(registry.bound());
  EXPECT_EQ(registry.participant_count(), 2u);
  EXPECT_EQ(registry.restore_bytes(), 2 * sizeof(int));
  EXPECT_EQ(registry.candidate_bytes(), 2 * sizeof(int));
  EXPECT_EQ(registry.effect_capacity(), 1u);
  EXPECT_EQ(registry.effect_info(0).identity, 1u);
  EXPECT_EQ(registry.effect_info(0).order, 0u);
  EXPECT_EQ(registry.effect_info(0).status, ProgramEffectSlotStatus::kFrozen);

  EXPECT_THROW((void)registry.register_participant(first, state_ops(), {sizeof(int), sizeof(int)}),
               std::logic_error);
  EXPECT_FALSE(registry.try_register_participant(first, state_ops(), {sizeof(int), sizeof(int)}));
  EXPECT_THROW((void)registry.register_effect(2), std::logic_error);
  EXPECT_FALSE(registry.try_register_effect(2));
  EXPECT_THROW(registry.set_consensus({}), std::logic_error);
  EXPECT_FALSE(registry.try_set_consensus({}));

  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  EffectLog effect;
  effect.identity = 1;
  ASSERT_TRUE(transaction.prepare_effect(effect_handle, effect.prepared()));
  EXPECT_FALSE(transaction.prepare_effect(effect_handle, EffectLog{}.prepared()));
  EXPECT_EQ(transaction.effect_count(), 1u);
  transaction.rollback();
  EXPECT_EQ(registry.effect_info(0).status, ProgramEffectSlotStatus::kFrozen);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
}

TEST(ProgramTransaction, ZeroBudgetsAreExactAndNonzeroBudgetsAdmitOnlyTheirDeclaredCapacity) {
  State state;
  const ProgramParticipantBudget state_budget{sizeof(int), sizeof(int)};

  ProgramTransactionRegistry zero_participants({0, sizeof(int), sizeof(int), 0});
  EXPECT_THROW((void)zero_participants.register_participant(state, state_ops(), state_budget),
               std::length_error);
  EXPECT_FALSE(zero_participants.try_register_participant(state, state_ops(), state_budget));
  zero_participants.bind();
  EXPECT_EQ(zero_participants.participant_count(), 0u);

  ProgramTransactionRegistry zero_restore({1, 0, sizeof(int), 0});
  EXPECT_THROW((void)zero_restore.register_participant(state, state_ops(), state_budget),
               std::length_error);
  EXPECT_FALSE(zero_restore.try_register_participant(state, state_ops(), state_budget));
  zero_restore.bind();
  EXPECT_EQ(zero_restore.restore_bytes(), 0u);

  ProgramTransactionRegistry zero_candidate({1, sizeof(int), 0, 0});
  EXPECT_THROW((void)zero_candidate.register_participant(state, state_ops(), state_budget),
               std::length_error);
  EXPECT_FALSE(zero_candidate.try_register_participant(state, state_ops(), state_budget));
  zero_candidate.bind();
  EXPECT_EQ(zero_candidate.candidate_bytes(), 0u);

  ProgramTransactionRegistry zero_effects({1, sizeof(int), sizeof(int), 0});
  EXPECT_THROW((void)zero_effects.register_effect(17), std::length_error);
  EXPECT_FALSE(zero_effects.try_register_effect(17));
  zero_effects.bind();
  EXPECT_EQ(zero_effects.effect_capacity(), 0u);
}

TEST(ProgramTransaction, MethodParticipantDefaultBudgetIsInferredButExplicitZeroStaysExact) {
  MethodParticipantState inferred;
  ProgramTransactionRegistry inferred_registry({1, sizeof(inferred), sizeof(inferred), 0});
  const auto inferred_handle = inferred_registry.register_participant(inferred);
  ASSERT_TRUE(inferred_handle);
  EXPECT_EQ(inferred_registry.participant_info(inferred_handle.index).budget.restore_bytes,
            sizeof(inferred));
  EXPECT_EQ(inferred_registry.participant_info(inferred_handle.index).budget.candidate_bytes,
            sizeof(inferred));
  inferred_registry.bind();

  MethodParticipantState explicit_zero;
  ProgramTransactionRegistry zero_byte_registry({1, 0, 0, 0});
  EXPECT_THROW((void)zero_byte_registry.register_participant(explicit_zero), std::length_error);
  const auto zero_handle =
      zero_byte_registry.register_participant(explicit_zero, ProgramParticipantBudget{0, 0});
  ASSERT_TRUE(zero_handle);
  EXPECT_EQ(zero_byte_registry.participant_info(zero_handle.index).budget.restore_bytes, 0u);
  EXPECT_EQ(zero_byte_registry.participant_info(zero_handle.index).budget.candidate_bytes, 0u);
  zero_byte_registry.bind();
}

TEST(ProgramTransaction, FrozenEffectSlotsRefuseOutOfOrderAndMissingSubmissionsCollectively) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 2});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto first_handle = registry.register_effect(11);
  const auto second_handle = registry.register_effect(12);
  registry.bind();

  EffectLog out_of_order_first;
  EffectLog out_of_order_second;
  out_of_order_first.identity = 12;
  out_of_order_second.identity = 11;
  auto out_of_order = registry.begin();
  ASSERT_TRUE(out_of_order.begin_candidate());
  ASSERT_TRUE(out_of_order.begin_solve_guard_effect_prepare());
  ASSERT_TRUE(out_of_order.prepare_effect(second_handle, out_of_order_first.prepared()));
  ASSERT_TRUE(out_of_order.prepare_effect(first_handle, out_of_order_second.prepared()));
  EXPECT_FALSE(out_of_order.publish());
  EXPECT_EQ(out_of_order.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(out_of_order.fault().failure, ProgramTransactionFailure::kEffectPrepare);
  EXPECT_EQ(out_of_order_first.publish_calls, 0);
  EXPECT_EQ(out_of_order_second.publish_calls, 0);
  EXPECT_EQ(out_of_order_first.discard_calls, 1);
  EXPECT_EQ(out_of_order_second.discard_calls, 1);

  EffectLog only_first;
  only_first.identity = 11;
  auto missing = registry.begin();
  ASSERT_TRUE(missing.begin_candidate());
  ASSERT_TRUE(missing.begin_solve_guard_effect_prepare());
  ASSERT_TRUE(missing.prepare_effect(first_handle, only_first.prepared()));
  EXPECT_FALSE(missing.publish());
  EXPECT_EQ(missing.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(missing.fault().failure, ProgramTransactionFailure::kEffectPrepare);
  EXPECT_EQ(only_first.publish_calls, 0);
  EXPECT_EQ(only_first.discard_calls, 1);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
}

TEST(ProgramTransaction, EffectPreparationConsensusAuthenticatesEveryIdentityWord) {
  constexpr std::uint64_t identity = 0x1122334455667788ULL;
  State state;
  ConsensusWordLog log;
  log.rejected_word = 0x55667788U;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 1},
                                      ProgramTransactionConsensus{&ConsensusWordLog::agree, &log});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(identity);
  registry.bind();

  EffectLog effect;
  effect.identity = 42;
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  // The callback fixture has an int display identity, so construct the effect with the exact
  // frozen 64-bit authority while retaining its counters/context.
  PreparedCompensableEffect prepared(&effect, &EffectLog::publish, &EffectLog::compensate,
                                     &EffectLog::finalize, &EffectLog::discard, &EffectLog::prepare,
                                     identity);
  ASSERT_TRUE(transaction.prepare_effect(effect_handle, std::move(prepared)));
  EXPECT_FALSE(transaction.publish());
  EXPECT_EQ(transaction.fault().failure, ProgramTransactionFailure::kEffectPrepare);
  EXPECT_EQ(effect.prepare_calls, 0);
  EXPECT_EQ(effect.publish_calls, 0);
  EXPECT_EQ(effect.discard_calls, 1);

  // Snapshot, Candidate and SolveGuard each contribute a leading zero. The frozen validation then
  // emits count, submission-protocol status, immutable slot status, ordinal, low identity, high
  // identity, submission status and the skipped-prepare status. Every reduction still runs after
  // the injected low-word refusal.
  ASSERT_EQ(log.count, 11u);
  EXPECT_EQ(log.words[3], 1u);
  EXPECT_EQ(log.words[4], 0u);
  EXPECT_EQ(log.words[5], static_cast<std::uint32_t>(ProgramEffectSlotStatus::kFrozen));
  EXPECT_EQ(log.words[6], 0u);
  EXPECT_EQ(log.words[7], 0x55667788U);
  EXPECT_EQ(log.words[8], 0x11223344U);
  EXPECT_EQ(log.words[9], 0u);
  EXPECT_EQ(log.words[10], 7u);
}

TEST(ProgramTransaction, IgnoredEffectSubmissionProtocolFaultStillFailsCollectively) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 1});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(91);
  registry.bind();

  EffectLog early;
  early.identity = 91;
  auto early_fault = registry.begin();
  ASSERT_TRUE(early_fault.begin_candidate());
  EXPECT_FALSE(early_fault.prepare_effect(effect_handle, early.prepared()));
  ASSERT_TRUE(early_fault.begin_solve_guard_effect_prepare());
  EffectLog valid_after_early;
  valid_after_early.identity = 91;
  ASSERT_TRUE(early_fault.prepare_effect(effect_handle, valid_after_early.prepared()));
  EXPECT_FALSE(early_fault.publish());
  EXPECT_EQ(early_fault.fault().failure, ProgramTransactionFailure::kEffectPrepare);
  EXPECT_EQ(early.prepare_calls, 0);
  EXPECT_EQ(early.discard_calls, 1);
  EXPECT_EQ(valid_after_early.prepare_calls, 0);
  EXPECT_EQ(valid_after_early.discard_calls, 1);

  EffectLog valid;
  valid.identity = 91;
  EffectLog excess;
  excess.identity = 91;
  auto budget_fault = registry.begin();
  ASSERT_TRUE(budget_fault.begin_candidate());
  ASSERT_TRUE(budget_fault.begin_solve_guard_effect_prepare());
  ASSERT_TRUE(budget_fault.prepare_effect(effect_handle, valid.prepared()));
  EXPECT_FALSE(budget_fault.prepare_effect(effect_handle, excess.prepared()));
  EXPECT_FALSE(budget_fault.publish());
  EXPECT_EQ(budget_fault.fault().failure, ProgramTransactionFailure::kBudget);
  EXPECT_EQ(valid.prepare_calls, 0);
  EXPECT_EQ(valid.discard_calls, 1);
  EXPECT_EQ(excess.prepare_calls, 0);
  EXPECT_EQ(excess.discard_calls, 1);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
}

TEST(ProgramTransaction, SnapshotFailureDoesNotTouchAcceptedState) {
  State state;
  state.accepted = 17;
  state.candidate = 29;
  state.fail_snapshot = true;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  EXPECT_THROW((void)registry.begin(), std::runtime_error);
  EXPECT_EQ(state.accepted, 17);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
  EXPECT_EQ(registry.last_fault().phase, ProgramTransactionPhase::kSnapshot);
  EXPECT_EQ(registry.last_fault().failure, ProgramTransactionFailure::kSnapshot);
}

TEST(ProgramTransaction, CandidateAndPrepareFaultsLeaveGenerationUnchanged) {
  State state;
  bool reject = true;
  ProgramTransactionConsensus consensus;
  consensus.context = &reject;
  consensus.function = [](void* context, std::uint32_t phase, std::uint32_t) noexcept {
    return phase != static_cast<std::uint32_t>(ProgramTransactionPhase::kCandidate) ||
           !*static_cast<bool*>(context);
  };
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 1}, consensus);
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(4);
  registry.bind();

  auto candidate_fault = registry.begin();
  EXPECT_FALSE(candidate_fault.begin_candidate());
  EXPECT_EQ(candidate_fault.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(candidate_fault.fault().phase, ProgramTransactionPhase::kCandidate);
  EXPECT_FALSE(candidate_fault.active());
  EXPECT_EQ(state.restores, 0);
  candidate_fault.rollback();

  reject = false;
  auto prepare_fault = registry.begin();
  ASSERT_TRUE(prepare_fault.begin_candidate());
  ASSERT_TRUE(prepare_fault.begin_solve_guard_effect_prepare());
  EffectLog effect;
  effect.identity = 4;
  effect.prepare_ok = false;
  EXPECT_TRUE(prepare_fault.prepare_effect(effect_handle, effect.prepared()));
  EXPECT_FALSE(prepare_fault.publish());
  EXPECT_EQ(effect.discard_calls, 1);
  EXPECT_EQ(prepare_fault.fault().failure, ProgramTransactionFailure::kEffectPrepare);
  prepare_fault.rollback();
  EXPECT_EQ(registry.accepted_generation().value, 0u);
}

TEST(ProgramTransaction, SolveGuardAndAtomicSealConsensusFaultsRollback) {
  State state;
  std::uint32_t rejected_phase =
      static_cast<std::uint32_t>(ProgramTransactionPhase::kSolveGuardEffectPrepare);
  ProgramTransactionConsensus consensus;
  consensus.context = &rejected_phase;
  consensus.function = [](void* context, std::uint32_t phase, std::uint32_t) noexcept {
    return phase != *static_cast<std::uint32_t*>(context);
  };
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0}, consensus);
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  auto solve_fault = registry.begin();
  ASSERT_TRUE(solve_fault.begin_candidate());
  EXPECT_FALSE(solve_fault.begin_solve_guard_effect_prepare());
  EXPECT_EQ(solve_fault.phase(), ProgramTransactionPhase::kSolveGuardEffectPrepare);
  EXPECT_EQ(solve_fault.fault().phase, ProgramTransactionPhase::kSolveGuardEffectPrepare);
  EXPECT_EQ(solve_fault.fault().failure, ProgramTransactionFailure::kSolve);
  solve_fault.rollback();

  rejected_phase = static_cast<std::uint32_t>(ProgramTransactionPhase::kAtomicSeal);
  auto seal_fault = registry.begin();
  ASSERT_TRUE(seal_fault.begin_candidate());
  ASSERT_TRUE(seal_fault.begin_solve_guard_effect_prepare());
  ASSERT_TRUE(seal_fault.publish());
  const auto rejected = seal_fault.seal();
  EXPECT_FALSE(rejected);
  EXPECT_EQ(seal_fault.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(seal_fault.fault().phase, ProgramTransactionPhase::kAtomicSeal);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
  EXPECT_EQ(state.accepted, 1);
}

TEST(ProgramTransaction, HiddenPublishParticipantFailureRestoresAllPublishedState) {
  State state;
  state.accepted = 13;
  state.candidate = 31;
  state.fail_publish = true;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 31;
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  const auto receipt = transaction.publish();

  EXPECT_FALSE(receipt);
  EXPECT_EQ(transaction.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(transaction.fault().phase, ProgramTransactionPhase::kHiddenPublish);
  EXPECT_EQ(transaction.fault().failure, ProgramTransactionFailure::kHiddenPublish);
  EXPECT_EQ(state.accepted, 13);
  EXPECT_EQ(state.restores, 1);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
}

TEST(ProgramTransaction, PublicationConsensusAuthenticatesTheExactFailureOrdinal) {
  const auto observed = [](const ConsensusWordLog& log, ProgramTransactionPhase phase,
                           std::uint32_t status) {
    for (std::size_t index = 0; index < std::min(log.count, log.words.size()); ++index)
      if (log.phases[index] == static_cast<std::uint32_t>(phase) && log.words[index] == status)
        return true;
    return false;
  };

  {
    State first;
    State second;
    second.fail_publish = true;
    ConsensusWordLog log;
    ProgramTransactionRegistry registry({2, 2 * sizeof(int), 2 * sizeof(int), 0},
                                        {&ConsensusWordLog::agree, &log});
    (void)registry.register_participant(first, state_ops(), {sizeof(int), sizeof(int)});
    (void)registry.register_participant(second, state_ops(), {sizeof(int), sizeof(int)});
    registry.bind();
    auto transaction = registry.begin();
    ASSERT_TRUE(transaction.begin_candidate());
    ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
    EXPECT_FALSE(transaction.publish());
    EXPECT_TRUE(observed(log, ProgramTransactionPhase::kHiddenPublish, 2U));
  }

  {
    State state;
    ConsensusWordLog log;
    ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 2},
                                        {&ConsensusWordLog::agree, &log});
    (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
    const auto first_handle = registry.register_effect(71);
    const auto second_handle = registry.register_effect(72);
    registry.bind();
    EffectLog first;
    EffectLog second;
    first.identity = 71;
    second.identity = 72;
    second.publish_ok = false;
    auto transaction = registry.begin();
    ASSERT_TRUE(transaction.begin_candidate());
    ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
    ASSERT_TRUE(transaction.prepare_effect(first_handle, first.prepared()));
    ASSERT_TRUE(transaction.prepare_effect(second_handle, second.prepared()));
    EXPECT_FALSE(transaction.publish());
    EXPECT_TRUE(observed(log, ProgramTransactionPhase::kCompensableEffects, 2U));
  }
}

TEST(ProgramTransaction, PublishFaultCompensatesEffectsInReverseOrderExactlyOnce) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 3});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto first_handle = registry.register_effect(1);
  const auto second_handle = registry.register_effect(2);
  const auto third_handle = registry.register_effect(3);
  registry.bind();
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());

  EffectLog first;
  EffectLog second;
  EffectLog third;
  first.identity = 1;
  second.identity = 2;
  third.identity = 3;
  third.publish_ok = false;
  std::vector<int> sequence;
  first.sequence = &sequence;
  second.sequence = &sequence;
  third.sequence = &sequence;
  ASSERT_TRUE(transaction.prepare_effect(first_handle, first.prepared()));
  ASSERT_TRUE(transaction.prepare_effect(second_handle, second.prepared()));
  ASSERT_TRUE(transaction.prepare_effect(third_handle, third.prepared()));

  const auto receipt = transaction.publish();
  EXPECT_FALSE(receipt.published);
  EXPECT_EQ(transaction.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(transaction.fault().failure, ProgramTransactionFailure::kCompensation);
  EXPECT_EQ(first.compensate_calls, 1);
  EXPECT_EQ(second.compensate_calls, 1);
  EXPECT_EQ(third.compensate_calls, 1);
  EXPECT_EQ(third.discard_calls, 0);
  {
    std::lock_guard<std::mutex> first_lock(first.mutex);
    std::lock_guard<std::mutex> second_lock(second.mutex);
    EXPECT_EQ(first.events, std::vector<int>({1, -1}));
    EXPECT_EQ(second.events, std::vector<int>({2, -2}));
  }
  {
    std::lock_guard<std::mutex> third_lock(third.mutex);
    EXPECT_EQ(third.events, std::vector<int>({3, -3}));
  }
  EXPECT_EQ(sequence, std::vector<int>({1, 2, 3, -3, -2, -1}));
  ASSERT_NE(transaction.effect_receipt(0), nullptr);
  ASSERT_NE(transaction.effect_receipt(1), nullptr);
  ASSERT_NE(transaction.effect_receipt(2), nullptr);
  EXPECT_TRUE(transaction.effect_receipt(0)->compensated);
  EXPECT_TRUE(transaction.effect_receipt(1)->compensated);
  EXPECT_TRUE(transaction.effect_receipt(2)->compensated);
  EXPECT_EQ(state.accepted, 1);
  EXPECT_EQ(registry.accepted_generation().value, 0u);
  transaction.rollback();
  EXPECT_EQ(first.compensate_calls, 1);
  EXPECT_EQ(second.compensate_calls, 1);
  EXPECT_EQ(third.compensate_calls, 1);
}

TEST(ProgramTransaction, ForeignAcceptedReaderRejectsWriterWithoutBlockingCollectiveProgress) {
  const auto hold_reader = [](ProgramTransactionRegistry& registry, std::promise<void>& acquired,
                              std::shared_future<void> release) {
    auto lease = registry.acquire_read();
    acquired.set_value();
    release.wait();
  };

  {
    State state;
    ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
    (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
    registry.set_candidate_visibility_lock(true);
    registry.bind();
    std::promise<void> acquired;
    auto ready = acquired.get_future();
    std::promise<void> release;
    auto release_signal = release.get_future().share();
    std::thread reader(hold_reader, std::ref(registry), std::ref(acquired), release_signal);
    EXPECT_EQ(ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto transaction = registry.begin();
    EXPECT_FALSE(transaction.begin_candidate());
    EXPECT_EQ(transaction.fault().failure, ProgramTransactionFailure::kCandidate);
    EXPECT_EQ(transaction.fault().reason_code, 1u);
    transaction.rollback();
    EXPECT_EQ(state.restores, 0);
    release.set_value();
    reader.join();
    EXPECT_EQ(state.restores, 0);
  }

  {
    State state;
    ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
    (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
    registry.bind();
    std::promise<void> acquired;
    auto ready = acquired.get_future();
    std::promise<void> release;
    auto release_signal = release.get_future().share();
    std::thread reader(hold_reader, std::ref(registry), std::ref(acquired), release_signal);
    EXPECT_EQ(ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto transaction = registry.begin();
    const bool candidate_started = transaction.begin_candidate();
    EXPECT_TRUE(candidate_started);
    const bool prepare_started =
        candidate_started && transaction.begin_solve_guard_effect_prepare();
    EXPECT_TRUE(prepare_started);
    if (prepare_started) {
      EXPECT_FALSE(transaction.publish());
      EXPECT_EQ(transaction.fault().failure, ProgramTransactionFailure::kHiddenPublish);
      EXPECT_EQ(transaction.fault().reason_code, 1u);
    }
    release.set_value();
    reader.join();
  }
}

TEST(ProgramTransaction, HiddenPublishConsensusRollbackRetainsWriterAndUnlinksMarker) {
  State state;
  state.accepted = 7;
  state.candidate = 19;
  std::uint32_t rejected_phase =
      static_cast<std::uint32_t>(ProgramTransactionPhase::kHiddenPublish);
  ProgramTransactionConsensus consensus;
  consensus.context = &rejected_phase;
  consensus.function = [](void* context, std::uint32_t phase, std::uint32_t) noexcept {
    return phase != *static_cast<std::uint32_t*>(context);
  };
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0}, consensus);
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.set_candidate_visibility_lock(true);
  registry.bind();

  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 19;
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  EXPECT_FALSE(transaction.publish());
  EXPECT_EQ(transaction.phase(), ProgramTransactionPhase::kRolledBack);
  EXPECT_EQ(transaction.fault().phase, ProgramTransactionPhase::kHiddenPublish);
  EXPECT_EQ(state.accepted, 7);
  EXPECT_EQ(state.restores, 1);

  {
    auto reader = registry.acquire_read();
    ASSERT_TRUE(reader);
    ASSERT_NE(reader.read(handle), nullptr);
    EXPECT_EQ(reader.read(handle)->accepted, 7);
  }

  rejected_phase = std::numeric_limits<std::uint32_t>::max();
  auto retry = registry.begin();
  ASSERT_TRUE(retry.begin_candidate());
  retry.rollback();
}

TEST(ProgramTransaction, ReaderSeesOldGenerationAndNewReadersBlockUntilSeal) {
  State state;
  state.accepted = 3;
  state.candidate = 9;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.set_candidate_visibility_lock(true);
  registry.bind();

  {
    auto old_reader = registry.acquire_read();
    ASSERT_TRUE(old_reader);
    ASSERT_EQ(old_reader.generation().value, 0u);
    ASSERT_NE(old_reader.read(handle), nullptr);
    EXPECT_EQ(old_reader.read(handle)->accepted, 3);
  }
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 9;
  EXPECT_THROW((void)registry.acquire_read(), std::logic_error);
  {
    auto provisional_scope = registry.acquire_provisional_read();
    ASSERT_TRUE(provisional_scope);
    auto candidate_reader = registry.acquire_read();
    ASSERT_TRUE(candidate_reader);
    ASSERT_NE(candidate_reader.read(handle), nullptr);
    EXPECT_EQ(candidate_reader.read(handle)->candidate, 9);
  }
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  ASSERT_TRUE(transaction.publish());

  std::promise<void> started;
  std::future<void> ready = started.get_future();
  std::promise<int> observed;
  std::future<int> result = observed.get_future();
  std::thread reader([&] {
    started.set_value();
    auto lease = registry.acquire_read();
    const State* accepted = lease.read(handle);
    observed.set_value(accepted == nullptr ? -1 : accepted->accepted);
  });
  ASSERT_EQ(ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  EXPECT_TRUE(transaction.seal());
  EXPECT_TRUE(transaction.finalize());
  reader.join();
  EXPECT_EQ(result.get(), 9);
  EXPECT_EQ(registry.accepted_generation().value, 1u);
}

TEST(ProgramTransaction, BorrowedWriterDefersGenerationUntilExternalSeal) {
  State state;
  state.accepted = 3;
  state.candidate = 9;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.set_candidate_visibility_lock(true);
  registry.bind();

  auto external_writer = registry.acquire_write();
  ASSERT_TRUE(external_writer);
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 9;
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  ASSERT_TRUE(transaction.publish());
  ASSERT_TRUE(transaction.seal());
  ASSERT_TRUE(transaction.finalize());
  EXPECT_EQ(state.accepted, 9);
  EXPECT_EQ(registry.accepted_generation().value, 0u);

  auto foreign_reader = std::async(std::launch::async, [&] {
    auto lease = registry.acquire_read();
    const State* accepted = lease.read(handle);
    return accepted == nullptr ? -1 : accepted->accepted;
  });
  EXPECT_EQ(foreign_reader.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  EXPECT_TRUE(registry.seal_external_writer());
  EXPECT_EQ(registry.accepted_generation().value, 1u);
  external_writer = AcceptedWriteLease{};
  EXPECT_EQ(foreign_reader.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(foreign_reader.get(), 9);
}

TEST(ProgramTransaction, NestedAcceptedReadLeasesBorrowTheAuthenticatedRoot) {
  State state;
  state.accepted = 13;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  auto outer_source = registry.acquire_read();
  auto outer = std::move(outer_source);
  EXPECT_FALSE(outer_source);
  ASSERT_TRUE(outer);
  ASSERT_EQ(outer.generation().value, 0u);
  ASSERT_NE(outer.read(handle), nullptr);
  EXPECT_EQ(outer.read(handle)->accepted, 13);
  {
    auto inner_source = registry.acquire_read();
    auto inner = std::move(inner_source);
    EXPECT_FALSE(inner_source);
    ASSERT_TRUE(inner);
    EXPECT_EQ(inner.generation(), outer.generation());
    ASSERT_NE(inner.read(handle), nullptr);
    EXPECT_EQ(inner.read(handle)->accepted, 13);
    EXPECT_THROW((void)registry.acquire_write(), std::logic_error);
  }
  EXPECT_TRUE(outer.valid());
}

TEST(ProgramTransaction, NestedAcceptedReadLeasesKeepRegistriesIndependent) {
  State first_state;
  first_state.accepted = 17;
  State second_state;
  second_state.accepted = 29;
  ProgramTransactionRegistry first_registry({1, sizeof(int), sizeof(int), 0});
  ProgramTransactionRegistry second_registry({1, sizeof(int), sizeof(int), 0});
  const auto first_handle =
      first_registry.register_participant(first_state, state_ops(), {sizeof(int), sizeof(int)});
  const auto second_handle =
      second_registry.register_participant(second_state, state_ops(), {sizeof(int), sizeof(int)});
  first_registry.bind();
  second_registry.bind();

  auto first_outer = first_registry.acquire_read();
  ASSERT_TRUE(first_outer);
  {
    auto second_outer = second_registry.acquire_read();
    ASSERT_TRUE(second_outer);
    auto first_inner = first_registry.acquire_read();
    auto second_inner = second_registry.acquire_read();
    ASSERT_TRUE(first_inner);
    ASSERT_TRUE(second_inner);
    ASSERT_NE(first_inner.read(first_handle), nullptr);
    ASSERT_NE(second_inner.read(second_handle), nullptr);
    EXPECT_EQ(first_inner.read(first_handle)->accepted, 17);
    EXPECT_EQ(second_inner.read(second_handle)->accepted, 29);
    EXPECT_THROW((void)first_registry.acquire_write(), std::logic_error);
    EXPECT_THROW((void)second_registry.acquire_write(), std::logic_error);
  }
  EXPECT_TRUE(first_outer.valid());
  auto second_writer = second_registry.acquire_write();
  ASSERT_TRUE(second_writer);
  EXPECT_THROW((void)first_registry.acquire_write(), std::logic_error);
}

TEST(ProgramTransaction, AcceptedReadRefusesWriterBeforeLockAttempt) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  {
    auto reader = registry.acquire_read();
    ASSERT_TRUE(reader);
    EXPECT_THROW((void)registry.acquire_write(), std::logic_error);
    EXPECT_THROW((void)registry.begin(), std::logic_error);
  }
  auto writer = registry.acquire_write();
  ASSERT_TRUE(writer);
}

TEST(ProgramTransaction, TryAcquireWriterRefusesReadLeaseAndRetriesAfterRelease) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  {
    auto reader = registry.acquire_read();
    ASSERT_TRUE(reader);
    EXPECT_FALSE(registry.try_acquire_write());
  }

  auto writer = registry.try_acquire_write();
  ASSERT_TRUE(writer);
}

TEST(ProgramTransaction, TryAcquireWriterRefusesActiveTransactionBeforeCandidate) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  (void)registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.bind();

  auto transaction = registry.begin();
  EXPECT_FALSE(registry.try_acquire_write());
  transaction.rollback();
}

TEST(ProgramTransaction, CandidateStillRefusesPublicReadWithoutProvisionalLease) {
  State state;
  state.accepted = 3;
  state.candidate = 9;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.set_candidate_visibility_lock(true);
  registry.bind();

  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 9;
  EXPECT_THROW((void)registry.acquire_read(), std::logic_error);
  EXPECT_THROW((void)registry.acquire_write(), std::logic_error);
  transaction.rollback();
}

TEST(ProgramTransaction, ProvisionalReadLeaseAuthenticatesPhaseOwnerAndNestedScopes) {
  State state;
  state.accepted = 3;
  state.candidate = 9;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 0});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  registry.set_candidate_visibility_lock(true);
  registry.bind();

  EXPECT_THROW((void)registry.acquire_provisional_read(), std::logic_error);
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 9;

  auto outer = registry.acquire_provisional_read();
  ASSERT_TRUE(outer);
  auto inner = registry.acquire_provisional_read();
  ASSERT_TRUE(inner);
  {
    auto candidate_reader = registry.acquire_read();
    ASSERT_TRUE(candidate_reader);
    ASSERT_NE(candidate_reader.read(handle), nullptr);
    EXPECT_EQ(candidate_reader.read(handle)->candidate, 9);
  }

  std::promise<bool> wrong_thread_release;
  std::thread wrong_thread([&] {
    try {
      outer.release();
      wrong_thread_release.set_value(false);
    } catch (const std::logic_error&) {
      wrong_thread_release.set_value(true);
    }
  });
  EXPECT_TRUE(wrong_thread_release.get_future().get());
  wrong_thread.join();
  EXPECT_TRUE(outer.valid());

  inner.release();
  EXPECT_FALSE(inner.valid());
  EXPECT_THROW(inner.release(), std::logic_error);

  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  EXPECT_TRUE(outer.valid());
  {
    auto prepare_reader = registry.acquire_read();
    ASSERT_TRUE(prepare_reader);
    ASSERT_NE(prepare_reader.read(handle), nullptr);
    EXPECT_EQ(prepare_reader.read(handle)->candidate, 9);
  }
  ASSERT_TRUE(transaction.publish());
  EXPECT_FALSE(outer.valid());
  EXPECT_THROW((void)registry.acquire_provisional_read(), std::logic_error);
  EXPECT_THROW((void)registry.acquire_read(), std::logic_error);
  ASSERT_TRUE(transaction.seal());
  EXPECT_FALSE(outer.valid());
  outer.release();
  EXPECT_THROW((void)registry.acquire_provisional_read(), std::logic_error);
  ASSERT_TRUE(transaction.finalize());
  EXPECT_THROW((void)registry.acquire_provisional_read(), std::logic_error);
}

TEST(ProgramTransaction, NestedRegistriesKeepWriterAndScopeMarkersIndependent) {
  State first_state;
  first_state.accepted = 1;
  first_state.candidate = 11;
  State second_state;
  second_state.accepted = 2;
  second_state.candidate = 22;
  ProgramTransactionRegistry first_registry({1, sizeof(int), sizeof(int), 0});
  ProgramTransactionRegistry second_registry({1, sizeof(int), sizeof(int), 0});
  const auto first_handle =
      first_registry.register_participant(first_state, state_ops(), {sizeof(int), sizeof(int)});
  const auto second_handle =
      second_registry.register_participant(second_state, state_ops(), {sizeof(int), sizeof(int)});
  first_registry.set_candidate_visibility_lock(true);
  second_registry.set_candidate_visibility_lock(true);
  first_registry.bind();
  second_registry.bind();

  auto first_transaction = first_registry.begin();
  ASSERT_TRUE(first_transaction.begin_candidate());
  first_transaction.provisional(first_handle)->candidate = 11;
  auto first_scope = first_registry.acquire_provisional_read();
  ASSERT_TRUE(first_scope);

  auto second_transaction = second_registry.begin();
  ASSERT_TRUE(second_transaction.begin_candidate());
  second_transaction.provisional(second_handle)->candidate = 22;
  auto second_scope = second_registry.acquire_provisional_read();
  ASSERT_TRUE(second_scope);
  EXPECT_EQ(second_registry.acquire_read().read(second_handle)->candidate, 22);
  EXPECT_EQ(first_registry.acquire_read().read(first_handle)->candidate, 11);

  second_scope.release();
  second_transaction.rollback();
  EXPECT_TRUE(first_scope.valid());
  EXPECT_EQ(first_registry.acquire_read().read(first_handle)->candidate, 11);
  first_scope.release();
  first_transaction.rollback();
}

TEST(ProgramTransaction, SealIsAtomicAndFinalizeFailureIsFailStopWithoutScientificRollback) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 1});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(7);
  registry.bind();
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 23;
  ASSERT_TRUE(transaction.begin_solve_guard_effect_prepare());
  EffectLog effect;
  effect.identity = 7;
  effect.finalize_ok = false;
  ASSERT_TRUE(transaction.prepare_effect(effect_handle, effect.prepared()));
  ASSERT_TRUE(transaction.publish());
  const auto sealed = transaction.seal();
  ASSERT_TRUE(sealed);
  EXPECT_EQ(sealed.generation.value, 1u);
  ASSERT_TRUE(transaction.sealed());
  const auto finalized = transaction.finalize();
  EXPECT_FALSE(finalized);
  EXPECT_TRUE(finalized.fail_stop);
  EXPECT_TRUE(registry.fail_stop());
  EXPECT_EQ(state.accepted, 23);
  EXPECT_EQ(effect.compensate_calls, 0);
  EXPECT_EQ(effect.finalize_calls, 1);
  EXPECT_THROW((void)registry.begin(), std::logic_error);
  (void)transaction.finalize();
  EXPECT_EQ(effect.finalize_calls, 1);
}

TEST(ProgramTransaction, SuccessfulEffectsAndReceiptsAreExactOnce) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 1});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(8);
  registry.bind();
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 41;
  ASSERT_TRUE(transaction.begin_prepare());
  EffectLog effect;
  effect.identity = 8;
  ASSERT_TRUE(transaction.prepare_effect(effect_handle, effect.prepared()));
  const auto published = transaction.publish();
  ASSERT_TRUE(published);
  ASSERT_EQ(transaction.effect_count(), 1u);
  ASSERT_NE(transaction.effect_receipt(0), nullptr);
  EXPECT_TRUE(transaction.effect_receipt(0)->published);
  const auto sealed = transaction.atomic_seal();
  ASSERT_TRUE(sealed);
  EXPECT_EQ(transaction.effect_receipt(0)->generation, sealed.generation);
  const auto finalized = transaction.irreversible_finalize();
  ASSERT_TRUE(finalized);
  EXPECT_TRUE(transaction.effect_receipt(0)->finalized);
  EXPECT_EQ(effect.publish_calls, 1);
  EXPECT_EQ(effect.finalize_calls, 1);
  (void)transaction.irreversible_finalize();
  EXPECT_EQ(effect.finalize_calls, 1);
}

using RuntimeTransactionState = ProgramRuntimeState<1>;

std::string balance_route(char digit) {
  return "pops.balance-ledger-route.v1:sha256:" + std::string(64, digit);
}

void declare_runtime_transaction_shape(RuntimeTransactionState& state, const std::string& route,
                                       bool second_projection = true) {
  state.declare_diagnostic("residual");
  state.declare_balance_route(route);
  state.declare_automatic_balance_term(2, 0, 1, "projection");
  state.declare_automatic_balance_term(2, 1, 1, "projection");
  state.declare_automatic_balance_term(2, 0, 1, "reflux");
  state.declare_step_projection("positivity");
  if (second_projection)
    state.declare_step_projection("realizability");
  state.bind_transaction_authorities();
}

void record_complete_balance(RuntimeTransactionState& state, const std::string& route,
                             double scale) {
  static constexpr std::array<std::string_view, 5> terms{"storage_change", "outward_boundary_flux",
                                                         "sources", "reflux", "projection"};
  for (std::size_t index = 0; index < terms.size(); ++index)
    state.record_balance_term(route, terms[index], scale * static_cast<double>(index + 1), "test");
}

TEST(ProgramRuntimeStateTransaction,
     BindSealsDiagnosticsBalanceAndProjectionWithoutCandidateGrowth) {
  const std::string route = balance_route('4');
  RuntimeTransactionState state;
  EXPECT_THROW(state.record_diagnostic("residual", 1.0), std::logic_error);
  declare_runtime_transaction_shape(state, route);

  const std::size_t diagnostic_size = state.diagnostics_.size();
  const std::size_t balance_size = state.step_balance_terms_.size();
  const std::size_t automatic_size = state.automatic_balance_terms_.size();
  const std::size_t projection_size = state.step_projections_.size();
  const std::size_t projection_capacity = state.step_projections_.capacity();
  const auto* diagnostic_slot = std::addressof(state.diagnostics_.begin()->second);
  const auto* balance_slot = std::addressof(state.step_balance_terms_.begin()->second);
  const auto* automatic_slot = std::addressof(state.automatic_balance_terms_.begin()->second);
  const auto* projection_storage = state.step_projections_.data();

  EXPECT_THROW(state.declare_diagnostic("late"), std::logic_error);
  EXPECT_THROW(state.declare_balance_route(balance_route('5')), std::logic_error);
  EXPECT_THROW(state.declare_step_projection("late"), std::logic_error);

  state.begin_step_projection_report();
  state.record_diagnostic("residual", 2.5);
  record_complete_balance(state, route, 1.0);
  state.run_balance_due_window(0, "test", [&] {
    state.note_automatic_balance_capture_due(true, "test");
    state.record_automatic_balance_term(2, 0, 1, "projection", 0.25, "test");
    state.record_automatic_balance_term(2, 1, 1, "projection", 0.75, "test");
    state.record_automatic_balance_term(2, 0, 1, "reflux", 0.5, "test");
  });
  state.note_step_projection("positivity");
  state.note_step_projection("positivity");

  EXPECT_THROW(state.record_diagnostic("late", 3.0), std::logic_error);
  EXPECT_THROW(state.record_balance_term(balance_route('5'), "sources", 1.0, "test"),
               std::logic_error);
  EXPECT_THROW(state.record_automatic_balance_term(3, 0, 1, "projection", 1.0, "test"),
               std::logic_error);
  EXPECT_THROW(state.note_step_projection("late"), std::logic_error);

  EXPECT_EQ(state.diagnostics_.size(), diagnostic_size);
  EXPECT_EQ(state.step_balance_terms_.size(), balance_size);
  EXPECT_EQ(state.automatic_balance_terms_.size(), automatic_size);
  EXPECT_EQ(state.step_projections_.size(), projection_size);
  EXPECT_EQ(state.step_projections_.capacity(), projection_capacity);
  EXPECT_EQ(std::addressof(state.diagnostics_.begin()->second), diagnostic_slot);
  EXPECT_EQ(std::addressof(state.step_balance_terms_.begin()->second), balance_slot);
  EXPECT_EQ(std::addressof(state.automatic_balance_terms_.begin()->second), automatic_slot);
  EXPECT_EQ(state.step_projections_.data(), projection_storage);
  EXPECT_DOUBLE_EQ(state.diagnostic("residual", "test"), 2.5);
  EXPECT_EQ(state.accepted_balance_terms(route, "test").at("projection"), 5.0);
  EXPECT_EQ(state.consume_step_projections(), (std::vector<std::string>{"positivity"}));
  EXPECT_TRUE(state.consume_step_projections().empty());
}

TEST(ProgramRuntimeStateTransaction,
     PreallocatedCopyIncludesProjectionActivityAndRejectsShapeDriftWithoutClobber) {
  const std::string route = balance_route('6');
  RuntimeTransactionState source;
  RuntimeTransactionState destination;
  declare_runtime_transaction_shape(source, route);
  declare_runtime_transaction_shape(destination, route);

  source.begin_step_projection_report();
  source.record_diagnostic("residual", 8.0);
  source.note_step_projection("realizability");
  destination.begin_step_projection_report();
  destination.record_diagnostic("residual", 1.0);
  destination.note_step_projection("positivity");
  destination.copy_from_preallocated(source);

  EXPECT_DOUBLE_EQ(destination.diagnostic("residual", "test"), 8.0);
  EXPECT_EQ(destination.consume_step_projections(), (std::vector<std::string>{"realizability"}));

  RuntimeTransactionState wrong_shape;
  declare_runtime_transaction_shape(wrong_shape, route, false);
  wrong_shape.begin_step_projection_report();
  wrong_shape.record_diagnostic("residual", 3.0);
  wrong_shape.note_step_projection("positivity");
  EXPECT_THROW(wrong_shape.copy_from_preallocated(source), std::logic_error);
  EXPECT_DOUBLE_EQ(wrong_shape.diagnostic("residual", "test"), 3.0);
  EXPECT_EQ(wrong_shape.consume_step_projections(), (std::vector<std::string>{"positivity"}));
}

TEST(ProgramRuntimeStateTransaction,
     PreparedRestoreRollsBackDiagnosticsBalanceAndProjectionActivityExactly) {
  const std::string route = balance_route('7');
  RuntimeTransactionState accepted;
  declare_runtime_transaction_shape(accepted, route);
  accepted.begin_step_projection_report();
  accepted.record_diagnostic("residual", 2.0);
  record_complete_balance(accepted, route, 1.0);
  accepted.note_step_projection("positivity");
  accepted.complete_balance_step(true);

  RuntimeTransactionState live = accepted;
  auto restore = live.prepare_accepted_restore(accepted);
  live.begin_step_projection_report();
  live.record_diagnostic("residual", 9.0);
  record_complete_balance(live, route, 10.0);
  live.note_step_projection("realizability");
  live.complete_balance_step(true);

  live.publish_prepared_accepted_restore(std::move(restore));
  EXPECT_DOUBLE_EQ(live.diagnostic("residual", "test"), 2.0);
  const auto balance = live.accepted_balance_terms(route, "test");
  EXPECT_DOUBLE_EQ(balance.at("storage_change"), 1.0);
  EXPECT_DOUBLE_EQ(balance.at("projection"), 5.0);
  EXPECT_EQ(live.consume_step_projections(), (std::vector<std::string>{"positivity"}));
  EXPECT_EQ(live.diagnostics_.size(), accepted.diagnostics_.size());
  EXPECT_EQ(live.step_balance_terms_.size(), accepted.step_balance_terms_.size());
  EXPECT_EQ(live.automatic_balance_terms_.size(), accepted.automatic_balance_terms_.size());
  EXPECT_EQ(live.step_projections_.size(), accepted.step_projections_.size());
}

TEST(ProgramTransaction, FrozenHotPathDoesNotRecordPoPSAllocations) {
  State state;
  ProgramTransactionRegistry registry({1, sizeof(int), sizeof(int), 1});
  const auto handle = registry.register_participant(state, state_ops(), {sizeof(int), sizeof(int)});
  const auto effect_handle = registry.register_effect(9);
  registry.set_candidate_visibility_lock(true);
  registry.bind();
  const auto before = pops::allocation_event_stats();
  auto transaction = registry.begin();
  ASSERT_TRUE(transaction.begin_candidate());
  transaction.provisional(handle)->candidate = 51;
  auto provisional_scope = registry.acquire_provisional_read();
  ASSERT_TRUE(provisional_scope);
  auto candidate_reader = registry.acquire_read();
  ASSERT_TRUE(candidate_reader);
  ASSERT_NE(candidate_reader.read(handle), nullptr);
  EXPECT_EQ(candidate_reader.read(handle)->candidate, 51);
  provisional_scope.release();
  ASSERT_TRUE(transaction.begin_prepare());
  EffectLog effect;
  effect.identity = 9;
  ASSERT_TRUE(transaction.prepare_effect(effect_handle, effect.prepared()));
  ASSERT_TRUE(transaction.publish());
  ASSERT_TRUE(transaction.seal());
  ASSERT_TRUE(transaction.finalize());
  const auto after = pops::allocation_event_stats();
  EXPECT_EQ(after.fab_calls, before.fab_calls);
  EXPECT_EQ(after.fab_bytes, before.fab_bytes);
  EXPECT_EQ(after.communication_calls, before.communication_calls);
  EXPECT_EQ(after.communication_bytes, before.communication_bytes);
  EXPECT_EQ(registry.effect_capacity(), 1u);
}

}  // namespace
