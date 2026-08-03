#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using namespace pops::runtime::program;

namespace {

CellTemporalPartitionAcceptedState cell_local_state(std::uint64_t topology_epoch = 7) {
  CellTemporalPartitionAcceptedState state;
  state.kind = TemporalPartitionKind::CellLocal;
  state.provider_identity = "test.temporal-partition.batched-cells@1";
  state.topology_epoch = topology_epoch;
  state.synchronization_tick = 8;
  state.tick_denominator = 16;
  state.cells = {{0, 10, 0, 8}, {0, 11, 1, 8}, {1, 20, 2, 8}};
  return state;
}

CellTemporalPartitionAcceptedState single_level_cell_local_state(std::uint64_t topology_epoch) {
  CellTemporalPartitionAcceptedState state = cell_local_state(topology_epoch);
  for (CellTemporalPartitionRecord& cell : state.cells)
    cell.level = 0;
  return state;
}

ModelSpec exb_spec() {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  return spec;
}

}  // namespace

TEST(test_temporal_partition_restart, batched_attempt_commit_and_rollback_are_exact) {
  const CellTemporalPartitionAcceptedState accepted = cell_local_state();
  BatchedCellTemporalPartition partition(accepted);

  partition.begin_attempt(16);
  partition.advance_batch(0, {0}, 12);
  partition.advance_batch(0, {0}, 16);
  EXPECT_THROW(partition.require_barrier("field solve"), std::logic_error);
  partition.rollback();
  EXPECT_EQ(partition.checkpoint(), accepted);

  partition.begin_attempt(16);
  partition.advance_batch(0, {0}, 16);
  partition.advance_batch(1, {1}, 16);
  partition.advance_batch(2, {2}, 16);
  EXPECT_NO_THROW(partition.require_barrier("output"));
  partition.commit();

  const CellTemporalPartitionAcceptedState committed = partition.checkpoint();
  EXPECT_EQ(committed.synchronization_tick, 16);
  for (const CellTemporalPartitionRecord& cell : committed.cells)
    EXPECT_EQ(cell.accepted_tick, 16);
  const auto manifest = partition.manifest();
  ASSERT_EQ(manifest.size(), 4u);
  EXPECT_EQ(manifest[0][1], "cell_local");
  EXPECT_EQ(manifest[0][6], "3");
  EXPECT_EQ(manifest[1], (std::vector<std::string>{"rung", "0", "1"}));
  EXPECT_EQ(manifest[2], (std::vector<std::string>{"rung", "1", "1"}));
  EXPECT_EQ(manifest[3], (std::vector<std::string>{"rung", "2", "1"}));
}

TEST(test_temporal_partition_restart, malformed_state_and_batches_fail_before_mutation) {
  const CellTemporalPartitionAcceptedState accepted = cell_local_state();
  BatchedCellTemporalPartition partition(accepted);

  CellTemporalPartitionAcceptedState unsynchronized = accepted;
  unsynchronized.cells[1].accepted_tick = 6;
  EXPECT_THROW(partition.restore(unsynchronized), std::invalid_argument);
  EXPECT_EQ(partition.checkpoint(), accepted);

  partition.begin_attempt(16);
  EXPECT_THROW(partition.advance_batch(0, {0, 0}, 12), std::invalid_argument);
  EXPECT_THROW(partition.advance_batch(0, {1}, 12), std::invalid_argument);
  EXPECT_THROW(partition.advance_batch(2, {2}, 10), std::invalid_argument);
  partition.rollback();
  EXPECT_EQ(partition.checkpoint(), accepted);

  EXPECT_THROW(partition.require_global_execution_route(), std::logic_error);
  EXPECT_THROW(partition.require_prepared_execution_route("test.temporal-partition.other@1"),
               std::logic_error);
  EXPECT_NO_THROW(
      partition.require_prepared_execution_route("test.temporal-partition.batched-cells@1"));
  EXPECT_NO_THROW(BatchedCellTemporalPartition().require_global_execution_route());
}

TEST(test_temporal_partition_restart, accepted_image_round_trips_canonically) {
  AmrProgramAcceptedState accepted;
  accepted.temporal_partition = cell_local_state();

  const std::vector<std::uint8_t> encoded = serialize_amr_program_accepted_state(accepted);
  const AmrProgramAcceptedState decoded = deserialize_amr_program_accepted_state(encoded);
  EXPECT_EQ(decoded.temporal_partition, accepted.temporal_partition);
  EXPECT_EQ(serialize_amr_program_accepted_state(decoded), encoded);

  CellTemporalPartitionAcceptedState duplicate = accepted.temporal_partition;
  duplicate.cells[1].cell = duplicate.cells[0].cell;
  accepted.temporal_partition = duplicate;
  EXPECT_THROW(serialize_amr_program_accepted_state(accepted), std::invalid_argument);
}

TEST(test_temporal_partition_restart, legacy_image_without_temporal_authority_is_refused) {
  std::vector<std::uint8_t> legacy = {'P', 'O', 'P', 'S', 'A', 'S', 'T', '4'};
  legacy.resize(17 * sizeof(std::uint64_t), 0);

  try {
    static_cast<void>(deserialize_amr_program_accepted_state(legacy));
    FAIL() << "accepted-state v4 silently invents a global temporal partition";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(),
                 "invalid AMR Program accepted-state payload: unsupported magic/version");
  }
}

TEST(test_temporal_partition_restart,
     strict_amr_restore_consumes_manifest_and_refuses_global_step_bypass) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.regrid_every = 0;
  config.periodicity = {true, true};

  AmrSystem system(config);
  system.add_block("tracer", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
  test::install_forward_euler_program(system);
  system.step(0.01);
  ASSERT_EQ(system.engine()->nlev(), 1);

  AmrProgramAcceptedState accepted =
      deserialize_amr_program_accepted_state(system.program_accepted_state());
  accepted.temporal_partition = single_level_cell_local_state(system.engine()->topology_epoch());
  const std::vector<std::uint8_t> cell_local_image = serialize_amr_program_accepted_state(accepted);
  system.restore_checkpoint_accepted_state(cell_local_image);

  const auto manifest = system.program_temporal_partition_manifest();
  ASSERT_EQ(manifest.size(), 4u);
  EXPECT_EQ(manifest[0][1], "cell_local");
  EXPECT_EQ(manifest[0][2], "test.temporal-partition.batched-cells@1");

  const double time_before = system.time();
  const int step_before = system.macro_step();
  const std::vector<std::uint8_t> bytes_before = system.program_accepted_state();
  try {
    system.step(0.01);
    FAIL() << "an authenticated cell-local schedule degraded to the global AMR driver";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string(error.what()).find("local-stage and time-integrated flux-ledger"),
              std::string::npos);
  }
  EXPECT_DOUBLE_EQ(system.time(), time_before);
  EXPECT_EQ(system.macro_step(), step_before);
  EXPECT_EQ(system.program_accepted_state(), bytes_before);
  EXPECT_EQ(system.program_temporal_partition_manifest(), manifest);

  AmrProgramAcceptedState wrong_topology = accepted;
  ++wrong_topology.temporal_partition.topology_epoch;
  EXPECT_THROW(system.restore_checkpoint_accepted_state(
                   serialize_amr_program_accepted_state(wrong_topology)),
               std::runtime_error);
  EXPECT_EQ(system.program_accepted_state(), bytes_before)
      << "rejected restore must not replace the accepted image";

  AmrProgramAcceptedState wrong_level = accepted;
  wrong_level.temporal_partition.cells.back().level = system.engine()->nlev();
  EXPECT_THROW(
      system.restore_checkpoint_accepted_state(serialize_amr_program_accepted_state(wrong_level)),
      std::runtime_error);
  EXPECT_EQ(system.program_accepted_state(), bytes_before)
      << "an inactive-level partition must not replace the accepted image";
}
