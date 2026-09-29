/// @file
/// @brief Accepted-boundary temporal partition and exact-ranked checkpoint proofs.

#include <gtest/gtest.h>

#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace program = pops::runtime::program;

program::CellTemporalPartitionAcceptedState cell_local_state(std::uint64_t topology_epoch = 7) {
  program::CellTemporalPartitionAcceptedState state;
  state.kind = program::TemporalPartitionKind::CellLocal;
  state.provider_identity = "test.temporal-partition.batched-cells@1";
  state.topology_epoch = topology_epoch;
  state.synchronization_tick = 8;
  state.tick_denominator = 16;
  state.cells = {{0, 10, 0, 8}, {0, 11, 1, 8}, {1, 20, 2, 8}};
  return state;
}

template <int Dim>
program::AmrProgramAcceptedState<Dim> accepted_state() {
  program::AmrProgramAcceptedState<Dim> state;
  state.spatial_contract = "test.temporal-partition.dim-" + std::to_string(Dim);
  state.topology_epoch = 7;
  state.materialization_generation = 2;
  state.level_clocks = {{0, 5, pops::amr::Rational(0, 1), 1.25},
                        {1, 5, pops::amr::Rational(0, 1), 1.25}};
  state.logical_clock_ticks.emplace("clock.macro", 20);
  state.temporal_partition = cell_local_state(state.topology_epoch);
  state.tagging_hysteresis_state = {3, 1, 4};
  return state;
}

template <int Dim>
void prove_exact_ranked_round_trip() {
  const auto accepted = accepted_state<Dim>();
  const std::vector<std::uint8_t> encoded = program::serialize_amr_program_accepted_state(accepted);
  const auto decoded = program::deserialize_amr_program_accepted_state<Dim>(encoded);
  EXPECT_EQ(decoded.spatial_contract, accepted.spatial_contract);
  EXPECT_EQ(decoded.topology_epoch, accepted.topology_epoch);
  EXPECT_EQ(decoded.materialization_generation, accepted.materialization_generation);
  EXPECT_EQ(decoded.logical_clock_ticks, accepted.logical_clock_ticks);
  EXPECT_EQ(decoded.temporal_partition, accepted.temporal_partition);
  EXPECT_EQ(decoded.tagging_hysteresis_state, accepted.tagging_hysteresis_state);
  EXPECT_EQ(program::serialize_amr_program_accepted_state(decoded), encoded);
}

program::AmrProgramAcceptedState<2> pending_history_state() {
  program::AmrProgramAcceptedState<2> state;
  state.spatial_contract = "tests.pending-remap.qualification";
  state.topology_epoch = 11;
  state.materialization_generation = 21;
  state.level_clocks = {{0, 3, {0, 1}, .75}, {1, 3, {0, 1}, .75}, {2, 3, {0, 1}, .75}};
  state.logical_clock_ticks = {{"clock", 3}};
  state.histories = {{"prior", 0, "state", "cell", "clock", "linear", 2, 1}};
  program::HistorySampleIdentity sample{std::bit_cast<std::uint64_t>(.5),
      std::bit_cast<std::uint64_t>(.25), 17, program::HistorySampleKind::Publication};
  for (int level = 0; level < 3; ++level)
    for (int slot = 0; slot < 2; ++slot)
      state.history_slots.push_back({"prior", level, slot, .25, true, 2, sample});
  program::AmrProgramPendingHistoryRemap marker{
      "pops.amr.level-history.v1/1/5:prior", 0, 1, 10, 20, 11, 21, 3, 2, 1, .25, .125, false};
  marker.qualified_topology_epoch = 11;
  marker.qualified_materialization_generation = 21;
  marker.source_sample = sample;
  marker.retained_ring_contract = program::checkpoint_detail::pending_ring_contract(
      state.histories, state.history_slots, "prior", 1, state.level_clocks[1]);
  state.pending_history_remaps = {marker};
  state.flux_budget_contract = "empty-flux";
  state.coupling_contract = "no-coupling";
  return state;
}

TEST(test_temporal_partition_restart, PendingRemapKeepsOriginAcrossTwoUnchangedQualifications) {
  auto state = pending_history_state();
  const auto origin = state.pending_history_remaps.front();
  for (std::uint64_t epoch : {12u, 13u}) {
    program::AmrProgramHistoryRemapDescriptor descriptor;
    descriptor.parent_level = 1;
    descriptor.child_level = 2;
    descriptor.prior_topology_epoch = epoch-1;
    descriptor.prior_materialization_generation = epoch+9;
    descriptor.published_topology_epoch = epoch;
    descriptor.published_materialization_generation = epoch+10;
    auto& marker = state.pending_history_remaps.front();
    program::checkpoint_detail::requalify_retained_pending_history(
        marker, marker.key, descriptor, marker.retained_ring_contract, marker.source_sample, false);
    state.topology_epoch = epoch;
    state.materialization_generation = epoch+10;
    EXPECT_EQ(marker.prior_topology_epoch, origin.prior_topology_epoch);
    EXPECT_EQ(marker.published_topology_epoch, origin.published_topology_epoch);
    EXPECT_EQ(marker.prior_materialization_generation, origin.prior_materialization_generation);
    EXPECT_EQ(marker.published_materialization_generation, origin.published_materialization_generation);
    EXPECT_EQ(marker.source_sample, origin.source_sample);
    EXPECT_EQ(marker.retained_ring_contract, origin.retained_ring_contract);
    const auto bytes = program::serialize_amr_program_accepted_state(state);
    EXPECT_EQ(bytes[7], '9');
    EXPECT_EQ(program::serialized_amr_program_accepted_state_size(state), bytes.size());
    EXPECT_EQ(program::deserialize_amr_program_accepted_state<2>(bytes).pending_history_remaps,
              state.pending_history_remaps);
  }
  auto bytes = program::serialize_amr_program_accepted_state(state);
  bytes[7] = '8';
  EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<2>(bytes), std::runtime_error);
  program::AmrProgramAcceptedStateCapacity<2> capacity;
  capacity.spatial_contract_characters = state.spatial_contract.size();
  capacity.level_count = state.level_clocks.size();
  capacity.logical_clock_identities = {"clock"};
  capacity.histories = state.histories;
  capacity.temporal_provider_identity = state.temporal_partition.provider_identity;
  capacity.pending_history_remap_count = 1;
  capacity.pending_history_remap_key_characters = origin.key.size();
  capacity.flux_budget_contract_characters = state.flux_budget_contract.size();
  capacity.coupling_contract_characters = state.coupling_contract.size();
  EXPECT_EQ(program::serialized_amr_program_accepted_state_capacity(capacity),
            bytes.size() + 32 + state.spatial_contract.size());
}

TEST(test_temporal_partition_restart, PendingRemapRefusesForgedQualificationAndChangedSample) {
  const auto accepted = pending_history_state();
  const auto bytes = program::serialize_amr_program_accepted_state(accepted);
  for (int failure = 0; failure < 5; ++failure) {
    auto forged = accepted;
    auto& marker = forged.pending_history_remaps.front();
    if (failure == 0) ++marker.qualified_topology_epoch;
    if (failure == 1) --marker.qualified_materialization_generation;
    if (failure == 2) ++marker.source_sample.ordinal;
    if (failure == 3) forged.history_slots[3].sample.ordinal += 1;
    if (failure == 4) marker.retained_ring_contract.back() ^= 1;
    EXPECT_THROW((void)program::serialize_amr_program_accepted_state(forged), std::invalid_argument);
    EXPECT_EQ(program::serialize_amr_program_accepted_state(accepted), bytes);
  }
}

TEST(test_temporal_partition_restart, PendingRemapRejectedSecondTransitionDoesNotMutateCandidate) {
  const auto accepted = pending_history_state();
  const auto origin = accepted.pending_history_remaps.front();
  for (int failure = 0; failure < 8; ++failure) {
    auto candidate = origin;
    program::AmrProgramHistoryRemapDescriptor descriptor;
    descriptor.parent_level = 1;
    descriptor.child_level = 2;
    descriptor.prior_topology_epoch = 11;
    descriptor.prior_materialization_generation = 21;
    descriptor.published_topology_epoch = 12;
    descriptor.published_materialization_generation = 22;
    auto sample = origin.source_sample;
    auto contract = origin.retained_ring_contract;
    bool stored = false;
    if (failure == 0) descriptor.parent_level = 0;
    if (failure == 1) descriptor.history_plan.push_back({origin.key, {}, program::AmrProgramHistoryRemapSource::RetainedChild});
    if (failure == 2) descriptor.history_plan.push_back({origin.key, "parent", program::AmrProgramHistoryRemapSource::ParentDeferred});
    if (failure == 3) ++descriptor.prior_topology_epoch;
    if (failure == 4) ++descriptor.published_materialization_generation;
    if (failure == 5) ++sample.ordinal;
    if (failure == 6) contract.back() ^= 1;
    if (failure == 7) stored = true;
    EXPECT_THROW(program::checkpoint_detail::requalify_retained_pending_history(
        candidate, candidate.key, descriptor, contract, sample, stored), std::runtime_error);
    EXPECT_EQ(candidate, origin);
  }
}

TEST(test_temporal_partition_restart, BatchedAttemptCommitRollbackAndCheckpointAreExact) {
  const auto accepted = cell_local_state();
  program::BatchedCellTemporalPartition partition(accepted);

  partition.begin_attempt(16);
  partition.advance_batch(0, {0}, 12);
  partition.advance_batch(0, {0}, 16);
  EXPECT_THROW(partition.require_barrier("field solve"), std::logic_error);
  EXPECT_THROW((void)partition.checkpoint(), std::logic_error);
  partition.rollback();
  EXPECT_EQ(partition.checkpoint(), accepted);

  partition.begin_attempt(16);
  partition.advance_batch(0, {0}, 16);
  partition.advance_batch(1, {1}, 16);
  partition.advance_batch(2, {2}, 16);
  EXPECT_NO_THROW(partition.require_barrier("output"));
  partition.commit();

  const auto committed = partition.checkpoint();
  EXPECT_EQ(committed.synchronization_tick, 16);
  for (const auto& cell : committed.cells)
    EXPECT_EQ(cell.accepted_tick, 16);
  const auto manifest = partition.manifest();
  ASSERT_EQ(manifest.size(), 4U);
  EXPECT_EQ(manifest[0][1], "cell_local");
  EXPECT_EQ(manifest[0][6], "3");
}

TEST(test_temporal_partition_restart, MalformedStateAndBatchesFailBeforeMutation) {
  const auto accepted = cell_local_state();
  program::BatchedCellTemporalPartition partition(accepted);

  auto unsynchronized = accepted;
  unsynchronized.cells[1].accepted_tick = 6;
  EXPECT_THROW(partition.restore(unsynchronized), std::invalid_argument);
  EXPECT_EQ(partition.checkpoint(), accepted);

  partition.begin_attempt(16);
  EXPECT_THROW(partition.advance_batch(0, {0, 0}, 12), std::invalid_argument);
  EXPECT_THROW(partition.advance_batch(0, {1}, 12), std::invalid_argument);
  EXPECT_THROW(partition.advance_batch(2, {2}, 10), std::invalid_argument);
  EXPECT_THROW(partition.restore(accepted), std::logic_error);
  partition.rollback();
  EXPECT_EQ(partition.checkpoint(), accepted);

  EXPECT_THROW(partition.require_global_execution_route(), std::logic_error);
  EXPECT_THROW(partition.require_prepared_execution_route("test.temporal-partition.other@1"),
               std::logic_error);
  EXPECT_NO_THROW(
      partition.require_prepared_execution_route("test.temporal-partition.batched-cells@1"));
  EXPECT_NO_THROW(program::BatchedCellTemporalPartition().require_global_execution_route());
}

TEST(test_temporal_partition_restart, AcceptedImageIsCanonicalInOneTwoAndThreeDimensions) {
  prove_exact_ranked_round_trip<1>();
  prove_exact_ranked_round_trip<2>();
  prove_exact_ranked_round_trip<3>();

  const auto encoded = program::serialize_amr_program_accepted_state(accepted_state<2>());
  EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<1>(encoded),
               std::runtime_error);

  auto duplicate = accepted_state<2>();
  duplicate.temporal_partition.cells[1].cell = duplicate.temporal_partition.cells[0].cell;
  EXPECT_THROW(program::serialize_amr_program_accepted_state(duplicate), std::invalid_argument);

  auto wrong_topology = accepted_state<2>();
  ++wrong_topology.temporal_partition.topology_epoch;
  EXPECT_THROW(program::serialize_amr_program_accepted_state(wrong_topology),
               std::invalid_argument);
}

TEST(test_temporal_partition_restart, RegridRequiresAnExactRematerializablePartition) {
  const auto cell_local = cell_local_state();
  EXPECT_THROW(program::require_regrid_rematerializable_temporal_partition(cell_local),
               std::runtime_error);

  program::CellTemporalPartitionAcceptedState global;
  EXPECT_NO_THROW(program::require_regrid_rematerializable_temporal_partition(global));
}

TEST(test_temporal_partition_restart, LegacyAndTruncatedImagesAreRefused) {
  std::vector<std::uint8_t> legacy = {'P', 'O', 'P', 'S', 'A', 'S', 'T', '4'};
  legacy.resize(17 * sizeof(std::uint64_t), 0);
  EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<2>(legacy),
               std::runtime_error);

  auto truncated = program::serialize_amr_program_accepted_state(accepted_state<2>());
  truncated.resize(truncated.size() / 2);
  EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<2>(truncated),
               std::runtime_error);
}

}  // namespace
