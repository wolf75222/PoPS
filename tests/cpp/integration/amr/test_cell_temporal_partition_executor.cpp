#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/cell_temporal_partition_executor.hpp>
#include <pops/runtime/program/same_level_cell_temporal_provider.hpp>

#include "load_balance_test_authority.hpp"

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
  explicit ProbeStageFluxProvider(std::shared_ptr<StageFluxProbe> probe)
      : probe_(std::move(probe)) {}

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

 private:
  std::shared_ptr<StageFluxProbe> probe_;
};

static_assert(CellTemporalStageFluxProvider<ProbeStageFluxProvider>);

struct LinearTransportModel {
  using State = StateVec<1>;
  using Prim = State;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  Real velocity_x = Real(0.7);
  Real velocity_y = Real(-0.2);

  POPS_HD State flux(const State& state, const auto&, int axis) const {
    return State{(axis == 0 ? velocity_x : velocity_y) * state[0]};
  }
  POPS_HD Real max_wave_speed(const State&, const auto&, int axis) const {
    const Real velocity = axis == 0 ? velocity_x : velocity_y;
    return velocity < Real(0) ? -velocity : velocity;
  }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }

  [[nodiscard]] static constexpr PreparedProviderIdentity
  transport_model_provider_identity() noexcept {
    return {"pops.test.linear-transport-model", 1};
  }
  void serialize_exact_transport_parameters(ExactContractBuilder& contract) const {
    contract.scalar(velocity_x).scalar(velocity_y);
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
};

static_assert(PhysicalModel<LinearTransportModel>);
static_assert(detail::ExactAmrTransportModelProvider<LinearTransportModel>);

std::unique_ptr<AmrRuntime> make_linear_transport_runtime() {
  constexpr int n = 4;
  AmrBuildParams build;
  build.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  build.mesh.n = n;
  build.mesh.L = 1.0;
  build.mesh.periodicity = Periodicity{true, true};
  build.mesh.regrid_every = 0;
  build.poisson.bc = BCRec{};
  detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(build, 1);
  std::vector<double> initial(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      initial[static_cast<std::size_t>(j) * n + i] =
          Real(1) + Real(0.05) * static_cast<Real>(i + 2 * j);
  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::build_amr_block<LinearTransportModel, NoSlope, RusanovFlux>(
      LinearTransportModel{}, layout, "tracer", initial, true, 1.4, 1, false));
  blocks.back().state_identity = "test://cell-temporal/tracer/U";
  return std::make_unique<AmrRuntime>(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                                      std::move(blocks), layout.base_per,
                                      layout.replicated_coarse, layout.wall);
}

std::shared_ptr<SameLevelCellIntegratedFluxLedger> make_scientific_flux_ledger(
    AmrRuntime& runtime, const CellTemporalPartitionAcceptedState& partition) {
  return std::make_shared<SameLevelCellIntegratedFluxLedger>(
      runtime.topology_epoch(), runtime.topology_materialization_generation(), 0, 0,
      partition.cells.size(), runtime.level_state(0, 0).ncomp());
}

}  // namespace

TEST(test_cell_temporal_partition_executor,
     executes_bounded_rung_batches_and_commits_exact_local_clocks) {
  const CellTemporalPartitionAcceptedState accepted = prepared_state();
  const auto probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  PreparedBatchedCellTemporalExecutor executor{accepted, ProbeStageFluxProvider(probe)};

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
  const auto probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  probe->fail_record = 2;
  probe->fail_end_tick = 10;
  probe->fail_reason = 73;
  PreparedBatchedCellTemporalExecutor executor{accepted, ProbeStageFluxProvider(probe)};

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
  const auto rejected_probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  rejected_probe->reject_begin = true;
  PreparedBatchedCellTemporalExecutor rejected{accepted, ProbeStageFluxProvider(rejected_probe)};
  EXPECT_THROW(rejected.begin_attempt(16), std::runtime_error);
  EXPECT_EQ(rejected.checkpoint(), accepted);
  EXPECT_EQ(rejected_probe->begins, 1);
  EXPECT_EQ(rejected_probe->commits, 0);
  EXPECT_EQ(rejected_probe->rollbacks, 1);

  CellTemporalPartitionAcceptedState wrong_identity = accepted;
  wrong_identity.provider_identity = "pops.test.different-temporal-stage-flux@1";
  const auto wrong_probe = std::make_shared<StageFluxProbe>(accepted.cells.size());
  EXPECT_THROW(
      (PreparedBatchedCellTemporalExecutor(wrong_identity, ProbeStageFluxProvider(wrong_probe))),
      std::logic_error);
  EXPECT_EQ(wrong_probe->begins, 0);
  EXPECT_EQ(wrong_probe->commits, 0);
  EXPECT_EQ(wrong_probe->rollbacks, 0);
}

TEST(test_cell_temporal_partition_executor,
     production_same_level_provider_commits_real_state_and_integrated_face_fluxes) {
  auto runtime = make_linear_transport_runtime();
  constexpr Real seconds_per_tick = Real(0.01);
  const CellTemporalPartitionAcceptedState partition =
      prepare_same_level_transport_euler_partition(*runtime, 0, 100, 0);
  auto ledger = make_scientific_flux_ledger(*runtime, partition);

  MultiFab expected = runtime->level_state(0, 0);
  MultiFab residual(expected.box_array(), expected.dmap(), expected.ncomp(), 0);
  MultiFab flux_x(same_level_cell_temporal_detail::face_boxes(expected.box_array(), true),
                  expected.dmap(), expected.ncomp(), 0);
  MultiFab flux_y(same_level_cell_temporal_detail::face_boxes(expected.box_array(), false),
                  expected.dmap(), expected.ncomp(), 0);
  runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock = "test.clock.cell-local";
  point.tick = 0;
  point.level = 0;
  point.substep = 0;
  point.stage = 0;
  point.stage_fraction = amr::Rational(0, 1);
  point.dt = seconds_per_tick;
  point.physical_time = 0.0;
  runtime->level_neg_div_flux_capture_into(0, 0, point, expected, residual, flux_x, flux_y);
  lincomb(expected, Real(1), expected, seconds_per_tick, residual);
  device_fence();

  PreparedSameLevelTransportEulerStageFluxProvider provider(
      *runtime, partition, ledger, "test.clock.cell-local");
  PreparedBatchedCellTemporalExecutor executor{partition, std::move(provider)};
  EXPECT_NE(executor.exact_contract().find("pops.amr.compiled-transport-flux"),
            std::string::npos);
  const std::string initial_contract = executor.exact_contract();
  executor.begin_attempt(1);
  executor.advance_to_barrier();
  executor.commit();
  device_fence();

  const MultiFab& actual = runtime->level_state(0, 0);
  const ConstArray4 want = expected.fab(0).const_array();
  const ConstArray4 got = actual.fab(0).const_array();
  const ConstArray4 fx = flux_x.fab(0).const_array();
  const ConstArray4 fy = flux_y.fab(0).const_array();
  const Box2D box = actual.box(0);
  std::size_t linear = 0;
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    for (int i = box.lo[0]; i <= box.hi[0]; ++i, ++linear) {
      EXPECT_DOUBLE_EQ(got(i, j), want(i, j));
      EXPECT_DOUBLE_EQ(ledger->integrated_flux(linear, SameLevelCellFace::XLow, 0),
                       seconds_per_tick * fx(i, j));
      EXPECT_DOUBLE_EQ(ledger->integrated_flux(linear, SameLevelCellFace::XHigh, 0),
                       seconds_per_tick * fx(i + 1, j));
      EXPECT_DOUBLE_EQ(ledger->integrated_flux(linear, SameLevelCellFace::YLow, 0),
                       seconds_per_tick * fy(i, j));
      EXPECT_DOUBLE_EQ(ledger->integrated_flux(linear, SameLevelCellFace::YHigh, 0),
                       seconds_per_tick * fy(i, j + 1));
    }
  EXPECT_EQ(ledger->publication_generation(), 1u);
  EXPECT_EQ(ledger->begin_tick(), 0);
  EXPECT_EQ(ledger->end_tick(), 1);
  EXPECT_EQ(ledger->tick_denominator(), 100);
  EXPECT_EQ(executor.checkpoint().synchronization_tick, 1);
  EXPECT_NE(executor.exact_contract(), initial_contract);

  const std::vector<double> after_first_commit = runtime->density(0);
  executor.begin_attempt(2);
  executor.advance_to_barrier();
  executor.commit();
  EXPECT_NE(runtime->density(0), after_first_commit);
  EXPECT_EQ(ledger->publication_generation(), 2u);
  EXPECT_EQ(ledger->begin_tick(), 1);
  EXPECT_EQ(ledger->end_tick(), 2);
  EXPECT_EQ(executor.checkpoint().synchronization_tick, 2);
}

TEST(test_cell_temporal_partition_executor,
     production_same_level_provider_rolls_back_and_refuses_unproved_envelopes) {
  auto runtime = make_linear_transport_runtime();
  const std::vector<double> accepted_state = runtime->density(0);
  const CellTemporalPartitionAcceptedState partition =
      prepare_same_level_transport_euler_partition(*runtime, 0, 100, 0);
  auto ledger = make_scientific_flux_ledger(*runtime, partition);
  PreparedSameLevelTransportEulerStageFluxProvider provider(*runtime, partition, ledger,
                                                            "test.clock.cell-local");
  PreparedBatchedCellTemporalExecutor executor{partition, std::move(provider)};
  executor.begin_attempt(1);
  executor.advance_to_barrier();
  executor.rollback();
  EXPECT_EQ(runtime->density(0), accepted_state);
  EXPECT_EQ(ledger->publication_generation(), 0u);

  CellTemporalPartitionAcceptedState mixed_rungs = partition;
  mixed_rungs.cells.back().rung = 1;
  auto mixed_ledger = make_scientific_flux_ledger(*runtime, mixed_rungs);
  EXPECT_THROW((PreparedSameLevelTransportEulerStageFluxProvider(
                   *runtime, mixed_rungs, mixed_ledger, Real(0.01), "test.clock.cell-local")),
               std::invalid_argument);

  auto stale_ledger = make_scientific_flux_ledger(*runtime, partition);
  PreparedSameLevelTransportEulerStageFluxProvider stale_provider(
      *runtime, partition, stale_ledger, "test.clock.cell-local");
  PreparedBatchedCellTemporalExecutor stale_executor{partition, std::move(stale_provider)};
  runtime->restore_checkpoint_counters(runtime->regrid_count(), runtime->topology_epoch() + 1);
  EXPECT_THROW(stale_executor.begin_attempt(1), std::runtime_error);
  EXPECT_EQ(runtime->density(0), accepted_state);
  EXPECT_EQ(stale_ledger->publication_generation(), 0u);
}

#undef POPS_TEST_CELL_TEMPORAL_INLINE
