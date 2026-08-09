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
