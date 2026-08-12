#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/runtime/program/cell_temporal_partition_executor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#define POPS_TEST_CELL_TEMPORAL_INLINE KOKKOS_INLINE_FUNCTION
#else
#define POPS_TEST_CELL_TEMPORAL_INLINE inline
#endif

using namespace pops;
using namespace pops::runtime::program;

namespace {

CellTemporalPartitionAcceptedState prepared_state() {
  CellTemporalPartitionAcceptedState state;
  state.kind = TemporalPartitionKind::CellLocal;
  state.provider_identity = "pops.test.temporal-stage-flux@1";
  state.topology_epoch = 17;
  state.synchronization_tick = 8;
  state.tick_denominator = 32;
  state.cells = {{0, 10, 0, 8}, {0, 11, 0, 8}, {0, 12, 1, 8},
                 {1, 20, 0, 8}, {1, 21, 1, 8}, {1, 22, 2, 8}};
  return state;
}

template <class T>
using DeviceVector = std::vector<T, fab_allocator<T>>;

struct StageFluxProbe {
  explicit StageFluxProbe(std::size_t cells)
      : last_begin(cells, -1),
        last_end(cells, -1),
        visits(cells, 0),
        scratch_flux(cells, 0),
        committed_flux(cells, 0) {}

  DeviceVector<std::int64_t> last_begin;
  DeviceVector<std::int64_t> last_end;
  DeviceVector<std::uint32_t> visits;
  DeviceVector<std::uint32_t> scratch_flux;
  DeviceVector<std::uint32_t> committed_flux;
  std::size_t fail_record = std::numeric_limits<std::size_t>::max();
  std::int64_t fail_end_tick = -1;
  std::uint32_t fail_reason = 0;
  bool reject_begin = false;
  bool reject_commit = false;
  int begins = 0;
  int commits = 0;
  int rollbacks = 0;
};

struct ProbeStageFluxDeviceView {
  std::int64_t* last_begin = nullptr;
  std::int64_t* last_end = nullptr;
  std::uint32_t* visits = nullptr;
  std::uint32_t* scratch_flux = nullptr;
  std::size_t cell_count = 0;
  std::size_t fail_record = std::numeric_limits<std::size_t>::max();
  std::int64_t fail_end_tick = -1;
  std::uint32_t fail_reason = 0;

  [[nodiscard]] POPS_TEST_CELL_TEMPORAL_INLINE CellTemporalStageOutcome
  evaluate_local_stage_and_record_space_time_flux(CellTemporalStagePoint point) const noexcept {
    if (point.record_index >= cell_count)
      return CellTemporalStageOutcome::failed(9001);
    last_begin[point.record_index] = point.begin_tick;
    last_end[point.record_index] = point.end_tick;
    ++visits[point.record_index];
    if (point.record_index == fail_record && point.end_tick == fail_end_tick)
      return CellTemporalStageOutcome::rejected(fail_reason);
    ++scratch_flux[point.record_index];
    return CellTemporalStageOutcome::accepted();
  }
};

static_assert(CellTemporalStageFluxDeviceView<ProbeStageFluxDeviceView>);

class ProbeStageFluxProvider {
 public:
  ProbeStageFluxProvider(std::shared_ptr<StageFluxProbe> probe, const ExecutionLane& lane)
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        probe_(std::move(probe)),
        local_indices_(probe_->last_begin.size()) {
    std::iota(local_indices_.begin(), local_indices_.end(), std::size_t{0});
  }

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.temporal-stage-flux", 1};
  }
  [[nodiscard]] static constexpr PreparedCellTemporalStageFluxContractV1
  stage_flux_contract() noexcept {
    return {};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("probe-stage-flux")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint64_t>(probe_->last_begin.size()))
        .scalar(static_cast<std::uint64_t>(probe_->fail_record))
        .scalar(probe_->fail_end_tick)
        .scalar(probe_->fail_reason);
  }
  [[nodiscard]] PreparedProviderSupport begin_attempt(
      CellTemporalAttemptDescriptor attempt) noexcept {
    ++probe_->begins;
    std::fill(probe_->last_begin.begin(), probe_->last_begin.end(), -1);
    std::fill(probe_->last_end.begin(), probe_->last_end.end(), -1);
    std::fill(probe_->visits.begin(), probe_->visits.end(), 0);
    std::fill(probe_->scratch_flux.begin(), probe_->scratch_flux.end(), 0);
    if (probe_->reject_begin)
      return PreparedProviderSupport::reject(41, "probe rejected attempt preparation");
    if (attempt.topology_epoch != 17 || attempt.begin_tick != 8 || attempt.target_tick <= 8 ||
        attempt.tick_denominator != 32 || attempt.cell_count != probe_->last_begin.size())
      return PreparedProviderSupport::reject(42, "probe received the wrong attempt authority");
    return PreparedProviderSupport::accept();
  }
  [[nodiscard]] PreparedProviderSupport prepare_commit_attempt() noexcept {
    if (probe_->reject_commit)
      return PreparedProviderSupport::reject(43, "probe rejected accepted publication");
    return PreparedProviderSupport::accept();
  }
  void commit_attempt() noexcept {
    ++probe_->commits;
    std::copy(probe_->scratch_flux.begin(), probe_->scratch_flux.end(),
              probe_->committed_flux.begin());
  }
  void rollback_attempt() noexcept {
    ++probe_->rollbacks;
    std::fill(probe_->scratch_flux.begin(), probe_->scratch_flux.end(), 0);
  }
  [[nodiscard]] ProbeStageFluxDeviceView device_view() const noexcept {
    return {probe_->last_begin.data(),   probe_->last_end.data(),   probe_->visits.data(),
            probe_->scratch_flux.data(), probe_->last_begin.size(), probe_->fail_record,
            probe_->fail_end_tick,       probe_->fail_reason};
  }
  [[nodiscard]] const ExecutionLane& execution_lane() const noexcept { return *lane_; }
  [[nodiscard]] std::span<const std::size_t> local_record_indices() const noexcept {
    return local_indices_;
  }

 private:
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::shared_ptr<StageFluxProbe> probe_;
  std::vector<std::size_t> local_indices_;
};

static_assert(CellTemporalStageFluxProvider<ProbeStageFluxProvider>);

}  // namespace

TEST(test_cell_temporal_partition_executor,
     executes_bounded_rung_batches_and_commits_exact_local_clocks) {
  const CellTemporalPartitionAcceptedState accepted = prepared_state();
  const ExecutionLane provider_lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.executor.success");
  const ExecutionLane executor_lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.executor.success");
  const auto probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  PreparedBatchedCellTemporalExecutor executor{
      accepted, ProbeStageFluxProvider(probe, provider_lane), executor_lane};

  EXPECT_NE(&provider_lane, &executor_lane);
  ASSERT_EQ(executor.prepared_rung_count(), 3u);
  EXPECT_EQ(executor.provider_identity(), accepted.provider_identity);
  EXPECT_FALSE(executor.exact_contract().empty());
  const AllocationEventStats allocations_before = allocation_event_stats();

  executor.begin_attempt(16);
  executor.advance_to_barrier();
  EXPECT_TRUE(executor.attempt_active());
  executor.commit();

  EXPECT_EQ(allocation_event_stats(), allocations_before)
      << "the prepared attempt and rung loop must not allocate PoPS storage";
  EXPECT_FALSE(executor.attempt_active());
  EXPECT_EQ(probe->begins, 1);
  EXPECT_EQ(probe->commits, 1);
  EXPECT_EQ(probe->rollbacks, 0);
  const CellTemporalPartitionAcceptedState committed = executor.checkpoint();
  EXPECT_EQ(committed.synchronization_tick, 16);
  for (const CellTemporalPartitionRecord& cell : committed.cells)
    EXPECT_EQ(cell.accepted_tick, 16);

  EXPECT_EQ(executor.stats().rung_batch_launches, 14u);
  EXPECT_EQ(executor.stats().stage_evaluations, 34u);
  const std::vector<std::uint32_t> expected_visits{8, 8, 4, 8, 4, 2};
  const std::vector<std::int64_t> expected_last_begin{15, 15, 14, 15, 14, 12};
  for (std::size_t index = 0; index < accepted.cells.size(); ++index) {
    EXPECT_EQ(probe->visits[index], expected_visits[index]);
    EXPECT_EQ(probe->committed_flux[index], expected_visits[index]);
    EXPECT_EQ(probe->last_begin[index], expected_last_begin[index]);
    EXPECT_EQ(probe->last_end[index], 16);
  }
}

TEST(test_cell_temporal_partition_executor,
     stage_rejection_rolls_back_clocks_and_attempt_local_flux_ledger) {
  const CellTemporalPartitionAcceptedState accepted = prepared_state();
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.executor.rejection");
  const auto probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  probe->fail_record = 2;
  probe->fail_end_tick = 10;
  probe->fail_reason = 73;
  PreparedBatchedCellTemporalExecutor executor{accepted, ProbeStageFluxProvider(probe, lane), lane};

  executor.begin_attempt(16);
  try {
    executor.advance_to_barrier();
    FAIL() << "a rejected local stage advanced the accepted clock";
  } catch (const CellTemporalStageFailure& failure) {
    EXPECT_EQ(failure.disposition(), CellTemporalStageDisposition::Rejected);
    EXPECT_EQ(failure.reason_code(), 73u);
  }

  EXPECT_FALSE(executor.attempt_active());
  EXPECT_EQ(executor.checkpoint(), accepted);
  EXPECT_EQ(probe->begins, 1);
  EXPECT_EQ(probe->commits, 0);
  EXPECT_EQ(probe->rollbacks, 1);
  EXPECT_TRUE(std::all_of(probe->scratch_flux.begin(), probe->scratch_flux.end(),
                          [](std::uint32_t value) { return value == 0; }));
  EXPECT_TRUE(std::all_of(probe->committed_flux.begin(), probe->committed_flux.end(),
                          [](std::uint32_t value) { return value == 0; }));
  EXPECT_EQ(executor.stats().rung_batch_launches, 3u)
      << "the executor batches every same-rung cell into one launch";
  EXPECT_EQ(executor.stats().stage_evaluations, 8u);
}

TEST(test_cell_temporal_partition_executor,
     provider_preparation_and_identity_fail_closed_before_any_accepted_mutation) {
  const CellTemporalPartitionAcceptedState accepted = prepared_state();
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.executor.preparation");
  const auto rejected_probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  rejected_probe->reject_begin = true;
  PreparedBatchedCellTemporalExecutor rejected{accepted,
                                               ProbeStageFluxProvider(rejected_probe, lane), lane};
  EXPECT_THROW(rejected.begin_attempt(16), std::runtime_error);
  EXPECT_EQ(rejected.checkpoint(), accepted);
  EXPECT_EQ(rejected_probe->begins, 1);
  EXPECT_EQ(rejected_probe->commits, 0);
  EXPECT_EQ(rejected_probe->rollbacks, 1);

  CellTemporalPartitionAcceptedState wrong_identity = accepted;
  wrong_identity.provider_identity = "pops.test.different-temporal-stage-flux@1";
  const auto wrong_probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  EXPECT_THROW((PreparedBatchedCellTemporalExecutor(
                   wrong_identity, ProbeStageFluxProvider(wrong_probe, lane), lane)),
               std::logic_error);
  EXPECT_EQ(wrong_probe->begins, 0);
  EXPECT_EQ(wrong_probe->commits, 0);
  EXPECT_EQ(wrong_probe->rollbacks, 0);
}

TEST(test_cell_temporal_partition_executor,
     provider_commit_preflight_rolls_back_clocks_and_attempt_local_ledger) {
  const CellTemporalPartitionAcceptedState accepted = prepared_state();
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.executor.commit");
  const auto probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  PreparedBatchedCellTemporalExecutor executor{accepted, ProbeStageFluxProvider(probe, lane), lane};

  executor.begin_attempt(16);
  executor.advance_to_barrier();
  probe->reject_commit = true;
  EXPECT_THROW(executor.commit(), std::runtime_error);

  EXPECT_FALSE(executor.attempt_active());
  EXPECT_EQ(executor.checkpoint(), accepted);
  EXPECT_EQ(probe->commits, 0);
  EXPECT_EQ(probe->rollbacks, 1);
  EXPECT_TRUE(std::all_of(probe->committed_flux.begin(), probe->committed_flux.end(),
                          [](std::uint32_t value) { return value == 0; }));
}

#undef POPS_TEST_CELL_TEMPORAL_INLINE
