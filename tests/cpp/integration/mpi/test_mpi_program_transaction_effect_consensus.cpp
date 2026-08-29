#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/program/program_transaction.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace {

using pops::runtime::program::EffectHandle;
using pops::runtime::program::AcceptedReadLease;
using pops::runtime::program::kInvalidProgramTransactionIndex;
using pops::runtime::program::PreparedCompensableEffect;
using pops::runtime::program::ProgramParticipantBudget;
using pops::runtime::program::ProgramParticipantOps;
using pops::runtime::program::ProgramTransaction;
using pops::runtime::program::ProgramTransactionConsensus;
using pops::runtime::program::ProgramTransactionFailure;
using pops::runtime::program::ProgramTransactionPhase;
using pops::runtime::program::ProgramTransactionRegistry;

struct ConsensusContext final {
  const pops::ExecutionLane* lane = nullptr;

  static bool agree(void* opaque, std::uint32_t phase, std::uint32_t status) noexcept {
    const auto& context = *static_cast<const ConsensusContext*>(opaque);
    if (context.lane == nullptr)
      return false;
    try {
      const long local_phase = static_cast<long>(phase);
      const long local_status = static_cast<long>(status);
      const long phase_min = pops::all_reduce_min(local_phase, *context.lane);
      const long phase_max = pops::all_reduce_max(local_phase, *context.lane);
      const long status_min = pops::all_reduce_min(local_status, *context.lane);
      const long status_max = pops::all_reduce_max(local_status, *context.lane);
      return phase_min == phase_max && status_min == status_max;
    } catch (...) {
      return false;
    }
  }
};

struct ScientificState final {
  int accepted = 17;
  int candidate = 29;
  int restores = 0;
  int publish_calls = 0;
  bool fail_publish = false;

  static bool snapshot(void* opaque, void* image, std::size_t bytes) noexcept {
    if (opaque == nullptr || image == nullptr || bytes != sizeof(int))
      return false;
    const auto& state = *static_cast<const ScientificState*>(opaque);
    std::memcpy(image, &state.accepted, sizeof(int));
    return true;
  }
  static void restore(void* opaque, const void* image, std::size_t bytes) noexcept {
    if (opaque == nullptr || image == nullptr || bytes != sizeof(int))
      std::terminate();
    auto& state = *static_cast<ScientificState*>(opaque);
    std::memcpy(&state.accepted, image, sizeof(int));
    ++state.restores;
  }
  static bool publish(void* opaque) noexcept {
    auto& state = *static_cast<ScientificState*>(opaque);
    ++state.publish_calls;
    if (state.fail_publish)
      return false;
    state.accepted = state.candidate;
    return true;
  }
  static void* candidate_view(void* opaque) noexcept { return opaque; }
};

struct CompensationTrace final {
  int labels[8] = {};
  int count = 0;

  void record(int label) noexcept {
    if (count < static_cast<int>(sizeof(labels) / sizeof(labels[0])))
      labels[count++] = label;
  }
};

struct RankLocalEffect final {
  bool fail_prepare = false;
  bool fail_publish = false;
  bool fail_finalize = false;
  int prepare_calls = 0;
  int publish_calls = 0;
  int compensate_calls = 0;
  int discard_calls = 0;
  int finalize_calls = 0;
  CompensationTrace* compensation_trace = nullptr;
  int compensation_label = 0;

  static bool prepare(void* opaque) noexcept {
    auto& effect = *static_cast<RankLocalEffect*>(opaque);
    ++effect.prepare_calls;
    return !effect.fail_prepare;
  }
  static bool publish(void* opaque) noexcept {
    auto& effect = *static_cast<RankLocalEffect*>(opaque);
    ++effect.publish_calls;
    return !effect.fail_publish;
  }
  static void compensate(void* opaque) noexcept {
    auto& effect = *static_cast<RankLocalEffect*>(opaque);
    ++effect.compensate_calls;
    if (effect.compensation_trace != nullptr)
      effect.compensation_trace->record(effect.compensation_label);
  }
  static void discard(void* opaque) noexcept {
    ++static_cast<RankLocalEffect*>(opaque)->discard_calls;
  }
  static bool finalize(void* opaque) noexcept {
    auto& effect = *static_cast<RankLocalEffect*>(opaque);
    ++effect.finalize_calls;
    return !effect.fail_finalize;
  }

  PreparedCompensableEffect prepared(std::uint64_t identity) noexcept {
    return {this,
            &RankLocalEffect::publish,
            &RankLocalEffect::compensate,
            &RankLocalEffect::finalize,
            &RankLocalEffect::discard,
            &RankLocalEffect::prepare,
            identity};
  }
};

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool collectively_equal(long value, const pops::ExecutionLane& lane) {
  return pops::all_reduce_min(value, lane) == pops::all_reduce_max(value, lane);
}

void require_collectively_equal(long value, const pops::ExecutionLane& lane, const char* message) {
  require(collectively_equal(value, lane), message);
}

void require_collectively_true(bool value, const pops::ExecutionLane& lane, const char* message) {
  require(pops::all_reduce_sum(value ? 1L : 0L, lane) == lane.size(), message);
}

void require_phase(const ProgramTransaction& transaction, ProgramTransactionPhase expected,
                   const pops::ExecutionLane& lane, const char* message) {
  require_collectively_true(transaction.phase() == expected, lane, message);
  require_collectively_equal(static_cast<long>(transaction.phase()), lane, message);
}

void require_transaction_fault(const ProgramTransaction& transaction, ProgramTransactionPhase phase,
                               ProgramTransactionFailure failure, const pops::ExecutionLane& lane,
                               const char* message) {
  const bool local_contract = transaction.phase() == ProgramTransactionPhase::kRolledBack &&
                              transaction.fault().phase == phase &&
                              transaction.fault().failure == failure;
  require_collectively_true(local_contract, lane, message);
  require_collectively_equal(static_cast<long>(transaction.phase()), lane, message);
  require_collectively_equal(static_cast<long>(transaction.fault().phase), lane, message);
  require_collectively_equal(static_cast<long>(transaction.fault().failure), lane, message);
  require_collectively_equal(static_cast<long>(transaction.fault().ordinal), lane, message);
}

int run_mpi_program_transaction_effect_consensus(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int failure = 0;
  {
#if defined(POPS_HAS_KOKKOS)
    Kokkos::ScopeGuard kokkos(argc, argv);
#else
    (void)argc;
    (void)argv;
#endif
    try {
      require(pops::n_ranks() == 2,
              "Program transaction effect-consensus proof requires exactly two MPI ranks");
      const pops::ExecutionLane lane =
          pops::ExecutionLane::world("tests.mpi.program-transaction-effect-consensus/lane");
      ConsensusContext consensus_context{&lane};
      ProgramTransactionRegistry registry(
          {1, sizeof(int), sizeof(int), 1},
          ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
      ScientificState state;
      ProgramParticipantOps participant_ops;
      participant_ops.snapshot = &ScientificState::snapshot;
      participant_ops.restore = &ScientificState::restore;
      participant_ops.publish = &ScientificState::publish;
      participant_ops.candidate = &ScientificState::candidate_view;
      (void)registry.register_participant(state, participant_ops,
                                          ProgramParticipantBudget{sizeof(int), sizeof(int)});
      constexpr std::uint64_t effect_identity = 0x4d50494546465831ULL;
      const EffectHandle effect_handle = registry.register_effect(effect_identity);
      registry.bind();

      RankLocalEffect effect;
      effect.fail_prepare = lane.rank() == 0;
      auto transaction = registry.begin();
      require(transaction.begin_candidate(), "candidate phase failed");
      require(transaction.begin_solve_guard_effect_prepare(), "solve/guard phase failed");
      require(transaction.prepare_effect(effect_handle, effect.prepared(effect_identity)),
              "frozen effect submission failed");
      const auto publication = transaction.publish();

      const bool local_ok =
          !publication && transaction.phase() == ProgramTransactionPhase::kRolledBack &&
          transaction.fault().failure == ProgramTransactionFailure::kEffectPrepare &&
          transaction.fault().reason_code == 7 && state.accepted == 17 && state.restores == 1 &&
          effect.prepare_calls == 1 && effect.publish_calls == 0 && effect.compensate_calls == 0 &&
          effect.discard_calls == 1 && registry.accepted_generation().value == 0;
      require(pops::all_reduce_sum(local_ok ? 1L : 0L, lane) == lane.size(),
              "rank-local effect preparation did not roll back collectively and exactly");

      // A successful transaction is the positive generation/phase witness for the same frozen
      // registry.  Each transition is checked on every rank before the next collective phase, so
      // a rank-local phase drift cannot be hidden by a later all-reduce.
      {
        ProgramTransactionRegistry success_registry(
            {1, sizeof(int), sizeof(int), 1},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState success_state;
        ProgramParticipantOps success_participant_ops;
        success_participant_ops.snapshot = &ScientificState::snapshot;
        success_participant_ops.restore = &ScientificState::restore;
        success_participant_ops.publish = &ScientificState::publish;
        success_participant_ops.candidate = &ScientificState::candidate_view;
        const auto success_participant = success_registry.register_participant(
            success_state, success_participant_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        const auto success_effect_handle = success_registry.register_effect(effect_identity);
        success_registry.bind();

        RankLocalEffect success_effect;
        auto accepted = success_registry.begin();
        require_phase(accepted, ProgramTransactionPhase::kSnapshot, lane,
                      "successful transaction did not start in snapshot phase");
        require_collectively_true(accepted.begin_candidate(), lane,
                                  "successful transaction candidate phase was not collective");
        require_phase(accepted, ProgramTransactionPhase::kCandidate, lane,
                      "successful transaction candidate phase drifted");
        require_collectively_true(accepted.begin_solve_guard_effect_prepare(), lane,
                                  "successful transaction solve/guard phase was not collective");
        require_phase(accepted, ProgramTransactionPhase::kSolveGuardEffectPrepare, lane,
                      "successful transaction solve/guard phase drifted");
        require_collectively_true(accepted.prepare_effect(success_effect_handle,
                                                          success_effect.prepared(effect_identity)),
                                  lane,
                                  "successful effect submission was not accepted collectively");

        const auto successful_publish = accepted.publish();
        require_collectively_true(static_cast<bool>(successful_publish), lane,
                                  "successful transaction hidden publish was not collective");
        require_phase(accepted, ProgramTransactionPhase::kCompensableEffects, lane,
                      "successful transaction did not reach compensable effects");
        require_collectively_equal(static_cast<long>(successful_publish.phase), lane,
                                   "successful publish phase receipt drifted");
        require_collectively_equal(static_cast<long>(successful_publish.generation.value), lane,
                                   "successful publish generation drifted");
        require_collectively_true(successful_publish.generation.value == 0, lane,
                                  "hidden publish exposed a generation before atomic seal");

        const auto sealed = accepted.seal();
        require_collectively_true(static_cast<bool>(sealed), lane,
                                  "successful transaction atomic seal was not collective");
        require_phase(accepted, ProgramTransactionPhase::kAtomicSeal, lane,
                      "successful transaction did not reach atomic seal");
        require_collectively_equal(static_cast<long>(sealed.phase), lane,
                                   "atomic seal phase receipt drifted");
        require_collectively_equal(static_cast<long>(sealed.generation.value), lane,
                                   "atomic seal generation drifted");
        require_collectively_true(
            sealed.generation.value == 1, lane,
            "successful transaction did not advance the accepted generation exactly once");

        const auto finalized = accepted.finalize();
        require_collectively_true(static_cast<bool>(finalized), lane,
                                  "successful transaction finalization was not collective");
        require_phase(accepted, ProgramTransactionPhase::kAccepted, lane,
                      "successful transaction did not reach accepted phase");
        require_collectively_equal(static_cast<long>(finalized.phase), lane,
                                   "finalize phase receipt drifted");
        require_collectively_equal(static_cast<long>(finalized.generation.value), lane,
                                   "finalize generation receipt drifted");
        require_collectively_true(
            finalized.generation.value == 1 && finalized.finalized_effects == 1 &&
                finalized.failed_effects == 0 && success_state.accepted == 29 &&
                success_effect.prepare_calls == 1 && success_effect.publish_calls == 1 &&
                success_effect.compensate_calls == 0 && success_effect.discard_calls == 0 &&
                accepted.effect_receipt(0) != nullptr && accepted.effect_receipt(0)->valid() &&
                accepted.effect_receipt(0)->generation.value == 1 &&
                accepted.effect_receipt(0)->finalized,
            lane, "successful transaction did not publish the exact accepted image");
        require_collectively_equal(static_cast<long>(success_registry.accepted_generation().value),
                                   lane, "accepted generation differs between ranks");
        require_collectively_true(success_registry.accepted_generation().value == 1, lane,
                                  "registry accepted generation did not match the sealed receipt");
        {
          auto reader = success_registry.acquire_read();
          require_collectively_true(reader.valid() && reader.generation().value == 1, lane,
                                    "accepted reader did not observe the sealed generation");
          const auto* visible = reader.read(success_participant);
          require_collectively_true(visible != nullptr && visible->accepted == 29, lane,
                                    "accepted reader did not observe the published participant");
        }
      }

      // A rank may ignore a failed out-of-budget submission, but the subsequent publish still
      // performs the fixed effect-slot protocol on every rank and fails before prepare/publish.
      {
        ProgramTransactionRegistry budget_registry(
            {1, sizeof(int), sizeof(int), 1},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState budget_state;
        ProgramParticipantOps budget_participant_ops;
        budget_participant_ops.snapshot = &ScientificState::snapshot;
        budget_participant_ops.restore = &ScientificState::restore;
        budget_participant_ops.publish = &ScientificState::publish;
        budget_participant_ops.candidate = &ScientificState::candidate_view;
        (void)budget_registry.register_participant(
            budget_state, budget_participant_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        const auto budget_effect_handle = budget_registry.register_effect(effect_identity);
        budget_registry.bind();

        RankLocalEffect budget_effect;
        RankLocalEffect excess_effect;
        auto budget_transaction = budget_registry.begin();
        require_collectively_true(budget_transaction.begin_candidate(), lane,
                                  "out-of-budget transaction candidate phase failed");
        require_collectively_true(budget_transaction.begin_solve_guard_effect_prepare(), lane,
                                  "out-of-budget transaction solve/guard phase failed");
        require_collectively_true(
            budget_transaction.prepare_effect(budget_effect_handle,
                                              budget_effect.prepared(effect_identity)),
            lane, "in-budget effect submission failed before out-of-budget probe");
        if (lane.rank() == 0) {
          // The return value is intentionally ignored: it is the rank-local protocol witness.
          (void)budget_transaction.prepare_effect(budget_effect_handle,
                                                  excess_effect.prepared(effect_identity));
        }
        const auto budget_publication = budget_transaction.publish();
        require_collectively_true(!budget_publication, lane,
                                  "out-of-budget submission unexpectedly published");
        require_transaction_fault(budget_transaction,
                                  ProgramTransactionPhase::kSolveGuardEffectPrepare,
                                  ProgramTransactionFailure::kEffectPrepare, lane,
                                  "out-of-budget submission did not fail in effect preparation");
        require_collectively_true(budget_transaction.fault().reason_code == 8, lane,
                                  "out-of-budget protocol status was not authenticated exactly");
        require_collectively_true(
            budget_state.accepted == 17 && budget_state.restores == 1 &&
                budget_effect.prepare_calls == 0 && budget_effect.publish_calls == 0 &&
                budget_effect.compensate_calls == 0 && budget_effect.discard_calls == 1 &&
                excess_effect.prepare_calls == 0 && excess_effect.publish_calls == 0 &&
                excess_effect.compensate_calls == 0 &&
                excess_effect.discard_calls == (lane.rank() == 0 ? 1 : 0) &&
                budget_registry.accepted_generation().value == 0,
            lane, "out-of-budget submission mutated state before collective refusal");
      }

      // Holding an accepted reader on one rank makes the candidate writer try-lock disagree
      // across ranks.  All ranks still execute the same consensus and return in bounded time.
      {
        ProgramTransactionRegistry candidate_lock_registry(
            {1, sizeof(int), sizeof(int), 0},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState candidate_lock_state;
        ProgramParticipantOps candidate_lock_ops;
        candidate_lock_ops.snapshot = &ScientificState::snapshot;
        candidate_lock_ops.restore = &ScientificState::restore;
        candidate_lock_ops.publish = &ScientificState::publish;
        candidate_lock_ops.candidate = &ScientificState::candidate_view;
        (void)candidate_lock_registry.register_participant(
            candidate_lock_state, candidate_lock_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        candidate_lock_registry.set_candidate_visibility_lock(true);
        candidate_lock_registry.bind();

        auto candidate_lock_transaction = candidate_lock_registry.begin();
        AcceptedReadLease retained_reader;
        if (lane.rank() == 0)
          retained_reader = candidate_lock_registry.acquire_read();
        require_collectively_true(
            lane.rank() == 0 ? retained_reader.valid() : !retained_reader.valid(), lane,
            "candidate-lock reader setup diverged");
        const bool candidate_started = candidate_lock_transaction.begin_candidate();
        require_collectively_true(!candidate_started, lane,
                                  "candidate writer did not refuse a retained accepted reader");
        // begin_candidate() reports the collective refusal while retaining the active lease so
        // the caller can choose its normal rollback path.  Close that path explicitly before
        // checking the terminal transaction phase.
        candidate_lock_transaction.rollback();
        require_transaction_fault(candidate_lock_transaction, ProgramTransactionPhase::kCandidate,
                                  ProgramTransactionFailure::kCandidate, lane,
                                  "candidate writer refusal was not collective");
        require_collectively_true(
            candidate_lock_transaction.fault().reason_code == (lane.rank() == 0 ? 1U : 3U), lane,
            "candidate writer refusal status was not preserved per rank");
        require_collectively_true(
            lane.rank() == 0 ? retained_reader.valid() && retained_reader.generation().value == 0
                             : !retained_reader.valid(),
            lane, "retained accepted reader was lost during candidate refusal");
        require_collectively_equal(
            static_cast<long>(candidate_lock_registry.accepted_generation().value), lane,
            "candidate refusal changed accepted generation");
      }

      // The same retained reader can be introduced after Candidate and before HiddenPublish when
      // candidate visibility locking is disabled.  Hidden publication must reject collectively,
      // before any participant/effect is published, without waiting on a rank-local mutex.
      {
        ProgramTransactionRegistry hidden_lock_registry(
            {1, sizeof(int), sizeof(int), 1},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState hidden_lock_state;
        ProgramParticipantOps hidden_lock_ops;
        hidden_lock_ops.snapshot = &ScientificState::snapshot;
        hidden_lock_ops.restore = &ScientificState::restore;
        hidden_lock_ops.publish = &ScientificState::publish;
        hidden_lock_ops.candidate = &ScientificState::candidate_view;
        (void)hidden_lock_registry.register_participant(
            hidden_lock_state, hidden_lock_ops, ProgramParticipantBudget{sizeof(int), sizeof(int)});
        const auto hidden_effect_handle = hidden_lock_registry.register_effect(effect_identity);
        hidden_lock_registry.bind();

        RankLocalEffect hidden_effect;
        auto hidden_transaction = hidden_lock_registry.begin();
        require_collectively_true(hidden_transaction.begin_candidate(), lane,
                                  "hidden-publish lock transaction candidate phase failed");
        require_collectively_true(hidden_transaction.begin_solve_guard_effect_prepare(), lane,
                                  "hidden-publish lock transaction solve/guard phase failed");
        require_collectively_true(
            hidden_transaction.prepare_effect(hidden_effect_handle,
                                              hidden_effect.prepared(effect_identity)),
            lane, "hidden-publish lock effect submission failed");
        AcceptedReadLease retained_reader;
        if (lane.rank() == 0)
          retained_reader = hidden_lock_registry.acquire_read();
        require_collectively_true(
            lane.rank() == 0 ? retained_reader.valid() : !retained_reader.valid(), lane,
            "hidden-publish reader setup diverged");

        const auto hidden_publication = hidden_transaction.publish();
        require_collectively_true(!hidden_publication, lane,
                                  "hidden publication unexpectedly bypassed retained reader");
        require_transaction_fault(hidden_transaction, ProgramTransactionPhase::kHiddenPublish,
                                  ProgramTransactionFailure::kHiddenPublish, lane,
                                  "hidden publication writer refusal was not collective");
        require_collectively_true(
            hidden_transaction.fault().reason_code == (lane.rank() == 0 ? 1U : 3U), lane,
            "hidden publication writer refusal status was not preserved per rank");
        require_collectively_true(
            (lane.rank() == 0 ? retained_reader.valid() && retained_reader.generation().value == 0
                              : !retained_reader.valid()) &&
                hidden_lock_state.accepted == 17 && hidden_lock_state.restores == 1 &&
                hidden_effect.prepare_calls == 1 && hidden_effect.publish_calls == 0 &&
                hidden_effect.compensate_calls == 0 && hidden_effect.discard_calls == 1 &&
                hidden_lock_registry.accepted_generation().value == 0,
            lane, "hidden publication mutated state before writer refusal");
      }

      // Participant publication is also authenticated one frozen slot at a time.  If the second
      // participant fails on one rank, every rank stops before invoking the third slot, then the
      // participant images are restored in reverse order.
      {
        ProgramTransactionRegistry participant_registry(
            {3, 3 * sizeof(int), 3 * sizeof(int), 0},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState first_participant;
        ScientificState second_participant;
        ScientificState third_participant;
        second_participant.fail_publish = lane.rank() == 0;
        ProgramParticipantOps participant_slot_ops;
        participant_slot_ops.snapshot = &ScientificState::snapshot;
        participant_slot_ops.restore = &ScientificState::restore;
        participant_slot_ops.publish = &ScientificState::publish;
        participant_slot_ops.candidate = &ScientificState::candidate_view;
        (void)participant_registry.register_participant(
            first_participant, participant_slot_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        (void)participant_registry.register_participant(
            second_participant, participant_slot_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        (void)participant_registry.register_participant(
            third_participant, participant_slot_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        participant_registry.bind();

        auto participant_transaction = participant_registry.begin();
        require_collectively_true(participant_transaction.begin_candidate(), lane,
                                  "participant publication transaction candidate phase failed");
        require_collectively_true(participant_transaction.begin_solve_guard_effect_prepare(), lane,
                                  "participant publication transaction solve phase failed");
        const auto participant_publication = participant_transaction.publish();
        require_collectively_true(!participant_publication, lane,
                                  "rank-local participant publication unexpectedly succeeded");
        require_transaction_fault(participant_transaction, ProgramTransactionPhase::kHiddenPublish,
                                  ProgramTransactionFailure::kHiddenPublish, lane,
                                  "participant publication failure was not collective");
        require_collectively_true(participant_transaction.fault().reason_code == 2, lane,
                                  "participant publication consensus status drifted");
        require_collectively_true(
            first_participant.accepted == 17 && second_participant.accepted == 17 &&
                third_participant.accepted == 17 && first_participant.publish_calls == 1 &&
                second_participant.publish_calls == 1 && third_participant.publish_calls == 0 &&
                first_participant.restores == 1 && second_participant.restores == 1 &&
                third_participant.restores == 1 &&
                participant_registry.accepted_generation().value == 0,
            lane, "participant publication failure did not restore the exact scientific image");
      }

      // A failure in the second compensable effect proves both fixed-slot short-circuiting and
      // reverse-order compensation.  The third effect is prepared but never published anywhere;
      // effects zero and one are both compensated in the order 1, 0 on every rank.
      {
        ProgramTransactionRegistry effect_registry(
            {1, sizeof(int), sizeof(int), 3},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState effect_state;
        ProgramParticipantOps effect_participant_ops;
        effect_participant_ops.snapshot = &ScientificState::snapshot;
        effect_participant_ops.restore = &ScientificState::restore;
        effect_participant_ops.publish = &ScientificState::publish;
        effect_participant_ops.candidate = &ScientificState::candidate_view;
        (void)effect_registry.register_participant(
            effect_state, effect_participant_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        const auto first_effect_handle = effect_registry.register_effect(effect_identity + 1);
        const auto second_effect_handle = effect_registry.register_effect(effect_identity + 2);
        const auto third_effect_handle = effect_registry.register_effect(effect_identity + 3);
        effect_registry.bind();

        CompensationTrace compensation_trace;
        RankLocalEffect first_effect;
        first_effect.compensation_trace = &compensation_trace;
        first_effect.compensation_label = 0;
        RankLocalEffect second_effect;
        second_effect.fail_publish = lane.rank() == 0;
        second_effect.compensation_trace = &compensation_trace;
        second_effect.compensation_label = 1;
        RankLocalEffect third_effect;
        third_effect.compensation_trace = &compensation_trace;
        third_effect.compensation_label = 2;

        auto effect_transaction = effect_registry.begin();
        require_collectively_true(effect_transaction.begin_candidate(), lane,
                                  "effect publication transaction candidate phase failed");
        require_collectively_true(effect_transaction.begin_solve_guard_effect_prepare(), lane,
                                  "effect publication transaction solve phase failed");
        require_collectively_true(
            effect_transaction.prepare_effect(first_effect_handle,
                                              first_effect.prepared(effect_identity + 1)),
            lane, "first effect preparation failed");
        require_collectively_true(
            effect_transaction.prepare_effect(second_effect_handle,
                                              second_effect.prepared(effect_identity + 2)),
            lane, "second effect preparation failed");
        require_collectively_true(
            effect_transaction.prepare_effect(third_effect_handle,
                                              third_effect.prepared(effect_identity + 3)),
            lane, "third effect preparation failed");

        const auto effect_publication = effect_transaction.publish();
        require_collectively_true(!effect_publication, lane,
                                  "rank-local effect publication unexpectedly succeeded");
        require_transaction_fault(effect_transaction, ProgramTransactionPhase::kCompensableEffects,
                                  ProgramTransactionFailure::kCompensation, lane,
                                  "effect publication failure was not collective");
        require_collectively_true(effect_transaction.fault().reason_code == 1, lane,
                                  "effect publication consensus status drifted");
        require_collectively_true(
            effect_state.accepted == 17 && effect_state.restores == 1 &&
                first_effect.publish_calls == 1 && second_effect.publish_calls == 1 &&
                third_effect.publish_calls == 0 && first_effect.compensate_calls == 1 &&
                second_effect.compensate_calls == 1 && third_effect.compensate_calls == 0 &&
                first_effect.discard_calls == 0 && second_effect.discard_calls == 0 &&
                third_effect.discard_calls == 1 && compensation_trace.count == 2 &&
                compensation_trace.labels[0] == 1 && compensation_trace.labels[1] == 0 &&
                effect_registry.accepted_generation().value == 0,
            lane, "effect publication rollback did not preserve LIFO and no-next-slot rules");
      }

      // Finalization is irreversible after seal.  A rank-local finalizer failure therefore enters
      // fail-stop while retaining the accepted scientific image and the sealed generation; it
      // never compensates or restores the already accepted state.
      {
        ProgramTransactionRegistry finalizer_registry(
            {1, sizeof(int), sizeof(int), 2},
            ProgramTransactionConsensus{&ConsensusContext::agree, &consensus_context});
        ScientificState finalizer_state;
        ProgramParticipantOps finalizer_participant_ops;
        finalizer_participant_ops.snapshot = &ScientificState::snapshot;
        finalizer_participant_ops.restore = &ScientificState::restore;
        finalizer_participant_ops.publish = &ScientificState::publish;
        finalizer_participant_ops.candidate = &ScientificState::candidate_view;
        (void)finalizer_registry.register_participant(
            finalizer_state, finalizer_participant_ops,
            ProgramParticipantBudget{sizeof(int), sizeof(int)});
        const auto first_finalizer_handle = finalizer_registry.register_effect(effect_identity + 4);
        const auto second_finalizer_handle =
            finalizer_registry.register_effect(effect_identity + 5);
        finalizer_registry.bind();

        RankLocalEffect first_finalizer;
        RankLocalEffect second_finalizer;
        second_finalizer.fail_finalize = lane.rank() == 0;
        auto finalizer_transaction = finalizer_registry.begin();
        require_collectively_true(finalizer_transaction.begin_candidate(), lane,
                                  "finalizer transaction candidate phase failed");
        require_collectively_true(finalizer_transaction.begin_solve_guard_effect_prepare(), lane,
                                  "finalizer transaction solve phase failed");
        require_collectively_true(
            finalizer_transaction.prepare_effect(first_finalizer_handle,
                                                 first_finalizer.prepared(effect_identity + 4)),
            lane, "first finalizer effect preparation failed");
        require_collectively_true(
            finalizer_transaction.prepare_effect(second_finalizer_handle,
                                                 second_finalizer.prepared(effect_identity + 5)),
            lane, "second finalizer effect preparation failed");
        require_collectively_true(static_cast<bool>(finalizer_transaction.publish()), lane,
                                  "finalizer transaction publication failed");
        require_phase(finalizer_transaction, ProgramTransactionPhase::kCompensableEffects, lane,
                      "finalizer transaction did not reach compensable effects");
        const auto finalizer_seal = finalizer_transaction.seal();
        require_collectively_true(static_cast<bool>(finalizer_seal), lane,
                                  "finalizer transaction seal failed");
        require_collectively_true(finalizer_seal.generation.value == 1, lane,
                                  "finalizer transaction sealed the wrong generation");

        const auto finalizer_result = finalizer_transaction.finalize();
        require_collectively_true(
            !finalizer_result && finalizer_result.fail_stop && !finalizer_result.finalized &&
                finalizer_result.generation.value == 1 && finalizer_result.finalized_effects == 1 &&
                finalizer_result.failed_effects == 1,
            lane, "rank-local finalizer failure did not enter fail-stop exactly");
        require_phase(finalizer_transaction, ProgramTransactionPhase::kFailStop, lane,
                      "rank-local finalizer failure did not preserve fail-stop phase");
        const bool finalizer_fault =
            finalizer_transaction.fault().phase == ProgramTransactionPhase::kIrreversibleFinalize &&
            finalizer_transaction.fault().failure == ProgramTransactionFailure::kFinalize &&
            finalizer_transaction.fault().ordinal == kInvalidProgramTransactionIndex &&
            finalizer_transaction.fault().reason_code == 1;
        require_collectively_true(finalizer_fault, lane,
                                  "rank-local finalizer failure fault witness drifted");
        require_collectively_equal(static_cast<long>(finalizer_transaction.fault().phase), lane,
                                   "finalizer fault phase differed between ranks");
        require_collectively_equal(static_cast<long>(finalizer_transaction.fault().failure), lane,
                                   "finalizer fault failure differed between ranks");
        require_collectively_equal(static_cast<long>(finalizer_transaction.fault().ordinal), lane,
                                   "finalizer fault ordinal differed between ranks");
        require_collectively_equal(static_cast<long>(finalizer_transaction.fault().reason_code),
                                   lane, "finalizer fault status differed between ranks");
        require_collectively_true(
            finalizer_state.accepted == 29 && finalizer_state.publish_calls == 1 &&
                finalizer_state.restores == 0 && first_finalizer.finalize_calls == 1 &&
                second_finalizer.finalize_calls == 1 && first_finalizer.compensate_calls == 0 &&
                second_finalizer.compensate_calls == 0 && first_finalizer.discard_calls == 0 &&
                second_finalizer.discard_calls == 0 &&
                finalizer_transaction.effect_receipt(0) != nullptr &&
                finalizer_transaction.effect_receipt(1) != nullptr &&
                finalizer_transaction.effect_receipt(0)->published &&
                finalizer_transaction.effect_receipt(1)->published &&
                finalizer_transaction.effect_receipt(0)->finalized &&
                !finalizer_transaction.effect_receipt(1)->finalized &&
                finalizer_transaction.effect_receipt(0)->finalize_attempted &&
                finalizer_transaction.effect_receipt(1)->finalize_attempted &&
                finalizer_transaction.effect_receipt(0)->generation.value == 1 &&
                finalizer_transaction.effect_receipt(1)->generation.value == 1 &&
                finalizer_registry.fail_stop() &&
                finalizer_registry.accepted_generation().value == 1,
            lane, "fail-stop finalization rolled back accepted scientific state");
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d Program effect-consensus proof failed: %s\n", pops::my_rank(),
                   error.what());
      failure = 1;
    }
    failure = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(failure || ::testing::Test::HasFailure())));
  }
  pops::comm_finalize();
  return failure;
}

}  // namespace

TEST(test_mpi_program_transaction_effect_consensus, RankLocalPrepareFailureRollsBackEveryRank) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_program_transaction_effect_consensus,
                                    "test_mpi_program_transaction_effect_consensus"),
            0);
}
