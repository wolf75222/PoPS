#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/same_level_cell_temporal_provider.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

template <int Dim>
struct ScalarAdvectionModel {
  using Law = nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;

  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.cell-temporal-program.scalar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, Real& lower, Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

template <int Dim>
ScalarAdvectionModel<Dim> scalar_advection_model(int transport_axis) {
  RealVector<Dim> velocity{};
  velocity[transport_axis] = Real(1);
  return {nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim, class MemorySpace>
Real copied_host_value(const Fab<Dim, MemorySpace>& fab, const Index<Dim>& index, int component) {
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const Box<Dim>& grown = fab.grown_box();
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return host(static_cast<std::size_t>(component) * stride + offset);
}

struct CollectiveRungFailureState {
  std::vector<std::uint32_t, fab_allocator<std::uint32_t>> accepted{11u, 22u};
  std::vector<std::uint32_t, fab_allocator<std::uint32_t>> scratch{11u, 22u};
  std::vector<std::uint32_t, fab_allocator<std::uint32_t>> device_evaluations{0u};
  bool fail_next_begin = false;
  bool failure_consumed = false;
  int commits = 0;
  int rollbacks = 0;
  int restores = 0;
};

struct CollectiveRungFailureDeviceView {
  std::uint32_t* scratch = nullptr;
  std::uint32_t* evaluations = nullptr;
  std::size_t size = 0;

  [[nodiscard]] POPS_HD runtime::program::CellTemporalStageOutcome
  evaluate_local_stage_and_record_space_time_flux(
      runtime::program::CellTemporalStagePoint point) const noexcept {
    if (point.record_index >= size)
      return runtime::program::CellTemporalStageOutcome::failed(9101);
    ++*evaluations;
    ++scratch[point.record_index];
    return runtime::program::CellTemporalStageOutcome::accepted();
  }
};

class CollectiveRungFailureProvider {
 public:
  CollectiveRungFailureProvider(std::shared_ptr<CollectiveRungFailureState> state,
                                const ExecutionLane& lane)
      : lane_(&lane), lane_borrow_(lane.borrow_immutably()), state_(std::move(state)) {
    if (lane.rank() == 0)
      local_indices_.push_back(0);
  }

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.cell-temporal-program.collective-rung-failure", 1};
  }
  [[nodiscard]] static constexpr runtime::program::PreparedCellTemporalStageFluxContractV1
  stage_flux_contract() noexcept {
    return {};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("collective-rung-failure")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint64_t>(state_->accepted.size()));
  }
  [[nodiscard]] PreparedProviderSupport begin_attempt(
      runtime::program::CellTemporalAttemptDescriptor) noexcept {
    std::copy(state_->accepted.begin(), state_->accepted.end(), state_->scratch.begin());
    return PreparedProviderSupport::accept();
  }
  void prepare_rung_batch_local(runtime::program::CellTemporalRungBatchDescriptor) {
    if (!local_indices_.empty()) {
      const std::size_t local = local_indices_.front();
      state_->scratch[local] = std::uint32_t{0xA5000000u} + static_cast<std::uint32_t>(local);
    }
    if (state_->fail_next_begin && !state_->failure_consumed) {
      state_->failure_consumed = true;
      throw std::runtime_error("rank-local injected begin-rung failure");
    }
  }
  void materialize_rung_batch_snapshot(runtime::program::CellTemporalRungBatchDescriptor) {}
  void finalize_rung_batch_candidate(runtime::program::CellTemporalRungBatchDescriptor) {}
  void complete_rung_batch(runtime::program::CellTemporalRungBatchDescriptor) noexcept {}
  [[nodiscard]] PreparedProviderSupport prepare_commit_attempt() noexcept {
    return PreparedProviderSupport::accept();
  }
  void commit_attempt() noexcept {
    std::copy(state_->scratch.begin(), state_->scratch.end(), state_->accepted.begin());
    ++state_->commits;
  }
  void rollback_attempt() noexcept {
    std::copy(state_->accepted.begin(), state_->accepted.end(), state_->scratch.begin());
    ++state_->rollbacks;
  }
  void restore_accepted_boundary(
      const runtime::program::CellTemporalPartitionAcceptedState&) noexcept {
    ++state_->restores;
  }
  [[nodiscard]] CollectiveRungFailureDeviceView device_view() const noexcept {
    return {state_->scratch.data(), state_->device_evaluations.data(), state_->scratch.size()};
  }
  [[nodiscard]] const ExecutionLane& execution_lane() const noexcept { return *lane_; }
  [[nodiscard]] std::span<const std::size_t> local_record_indices() const noexcept {
    return local_indices_;
  }

 private:
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::shared_ptr<CollectiveRungFailureState> state_;
  std::vector<std::size_t> local_indices_;
};

static_assert(runtime::program::CellTemporalStageFluxProvider<CollectiveRungFailureProvider>);
static_assert(runtime::program::CellTemporalRungBatchLifecycle<CollectiveRungFailureProvider>);
static_assert(
    runtime::program::CellTemporalAcceptedBoundaryLifecycle<CollectiveRungFailureProvider>);
static_assert(
    runtime::program::DistributedCellTemporalStageFluxProvider<CollectiveRungFailureProvider>);

int run_rank_local_begin_rung_failure_regression(const ExecutionLane& lane) {
  using namespace runtime::program;
  if (lane.size() != 2)
    return 30;

  CellTemporalPartitionAcceptedState accepted;
  accepted.kind = TemporalPartitionKind::CellLocal;
  accepted.provider_identity = "test.mpi.cell-temporal-program.collective-rung-failure@1";
  accepted.topology_epoch = 1;
  accepted.synchronization_tick = 0;
  accepted.tick_denominator = 4;
  accepted.cells = {{0, 0, 0, 0}};

  const auto state = std::make_shared<CollectiveRungFailureState>();
  CellTemporalPartitionAcceptedState divergent_plan = accepted;
  if (lane.rank() == 1)
    divergent_plan.cells.push_back({0, 1, 0, 0});
  bool divergent_plan_rejected = false;
  try {
    PreparedBatchedCellTemporalExecutor divergent_executor{
        divergent_plan, CollectiveRungFailureProvider(state, lane), lane};
  } catch (const std::invalid_argument& error) {
    divergent_plan_rejected =
        std::string(error.what()).find("fixed plan envelope differs") != std::string::npos;
  }
  if (all_reduce_min(divergent_plan_rejected ? 1L : 0L, lane) != 1)
    return 33;

  state->fail_next_begin = lane.rank() == 0;
  const auto accepted_bytes_before = state->accepted;
  PreparedBatchedCellTemporalExecutor executor{accepted, CollectiveRungFailureProvider(state, lane),
                                               lane};
  const CellTemporalPartitionAcceptedState checkpoint_before = executor.checkpoint();

  bool collectively_rejected = false;
  std::string failure_message;
  try {
    executor.begin_attempt(1);
    executor.advance_to_barrier();
  } catch (const std::exception& error) {
    collectively_rejected = true;
    failure_message = error.what();
  }
  bool checkpoint_restored = false;
  if (!executor.attempt_active()) {
    try {
      checkpoint_restored = executor.checkpoint() == checkpoint_before;
    } catch (const std::exception&) {
      checkpoint_restored = false;
    }
  }
  const bool byte_exact = std::memcmp(state->accepted.data(), accepted_bytes_before.data(),
                                      accepted_bytes_before.size() * sizeof(std::uint32_t)) == 0 &&
                          std::memcmp(state->scratch.data(), accepted_bytes_before.data(),
                                      accepted_bytes_before.size() * sizeof(std::uint32_t)) == 0;
  const bool collective_diagnostic =
      lane.rank() == 0
          ? failure_message == "rank-local injected begin-rung failure"
          : failure_message == "cell-local temporal rung-batch preparation failed on another rank";
  const long failed_rollback = !collectively_rejected || executor.attempt_active() ||
                                       !collective_diagnostic || !checkpoint_restored ||
                                       !byte_exact || state->commits != 0 ||
                                       state->rollbacks != 1 || state->device_evaluations[0] != 0
                                   ? 1L
                                   : 0L;
  if (all_reduce_max(failed_rollback, lane) != 0)
    return 31;

  bool retry_failed = false;
  CellTemporalPartitionAcceptedState retry_checkpoint;
  try {
    executor.begin_attempt(1);
    executor.advance_to_barrier();
    executor.commit();
    retry_checkpoint = executor.checkpoint();
  } catch (const std::exception&) {
    retry_failed = true;
  }
  const long failed_retry = retry_failed || executor.attempt_active() ||
                                    retry_checkpoint.synchronization_tick != 1 ||
                                    state->commits != 1 || state->rollbacks != 1 ||
                                    state->device_evaluations[0] != (lane.rank() == 0 ? 1u : 0u)
                                ? 1L
                                : 0L;
  if (all_reduce_max(failed_retry, lane) != 0)
    return 32;

  CellTemporalPartitionAcceptedState divergent_restore = retry_checkpoint;
  if (lane.rank() == 0) {
    divergent_restore.synchronization_tick = 0;
    divergent_restore.cells.front().accepted_tick = 0;
  }
  bool divergent_restore_rejected = false;
  try {
    executor.restore_accepted_boundary(std::move(divergent_restore));
  } catch (const std::invalid_argument& error) {
    divergent_restore_rejected =
        std::string(error.what()).find("restore target differs") != std::string::npos;
  }
  const long failed_divergent_restore = !divergent_restore_rejected || executor.attempt_active() ||
                                                executor.checkpoint() != retry_checkpoint ||
                                                state->restores != 0
                                            ? 1L
                                            : 0L;
  if (all_reduce_max(failed_divergent_restore, lane) != 0)
    return 34;

  CellTemporalPartitionAcceptedState restored = retry_checkpoint;
  restored.synchronization_tick = 0;
  restored.cells.front().accepted_tick = 0;
  try {
    executor.restore_accepted_boundary(restored);
    executor.begin_attempt(1);
    executor.advance_to_barrier();
    executor.commit();
  } catch (...) {
    return 35;
  }
  const long failed_restore_retry = executor.attempt_active() ||
                                            executor.checkpoint().synchronization_tick != 1 ||
                                            state->restores != 1 || state->commits != 2
                                        ? 1L
                                        : 0L;
  return all_reduce_max(failed_restore_retry, lane) == 0 ? 0 : 36;
}

int run_collective_program_route(int split_axis, bool prove_collective_rollback) {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = axis == split_axis ? 6 : 2;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }
  config.level_count = 1;
  config.regrid_every = 0;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  config.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis)
    config.coarse_max_grid[axis] = axis == split_axis ? 2 : config.shape[axis];
  const Box<Dim> domain = config.index_domain();
  for (int patch = 0; patch < 3; ++patch) {
    Box<Dim> box = domain;
    box.lo[split_axis] = 2 * patch;
    box.hi[split_axis] = 2 * patch + 1;
    config.boxes.push_back(box);
  }

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.mpi.cell-temporal-program/runtime@1");
  system.install_block_state_route("tracer", "test.mpi.cell-temporal-program/tracer/state@1");
  add_compiled_model<Dim>(system, "tracer", scalar_advection_model<Dim>(split_axis), "minmod",
                          "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          "test.mpi.cell-temporal-program/physical-flux");
  if (!system.uses_runtime_engine())
    return 1;

  std::size_t cells = 1;
  std::size_t varying_stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    cells *= static_cast<std::size_t>(config.shape[axis]);
    if (axis < split_axis)
      varying_stride *= static_cast<std::size_t>(config.shape[axis]);
  }
  std::vector<double> initial_state(cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell)
    initial_state[cell] = static_cast<double>((cell / varying_stride) %
                                              static_cast<std::size_t>(config.shape[split_axis]));
  system.set_conservative_state("tracer", initial_state);
  system.set_program_block_map({0});
  const std::string clock = "test.clock.cell-local-mpi-program.axis" + std::to_string(split_axis);
  std::vector<Box<Dim>> initial_boxes;
  std::vector<Index<Dim>> initial_owners;
  long local_boxes = 0;
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto initial_view = system.prepared_amr_block_state(0, 0);
    if (!initial_view)
      return 20;
    initial_boxes = initial_view->layout().boxes();
    initial_owners.reserve(initial_boxes.size());
    for (std::size_t index = 0; index < initial_boxes.size(); ++index)
      initial_owners.push_back(initial_view->distribution().owner(index));
    local_boxes = static_cast<long>(initial_view->local_size());
    state_components = static_cast<std::uint32_t>(initial_view->ncomp());
    state_ghosts = static_cast<std::uint32_t>(initial_view->ghosts()[0]);
  }
  if (initial_boxes.size() < 2)
    return 21;
  const ExecutionLane lane = ExecutionLane::world("test.mpi.cell-temporal-program/lane");
  if (all_reduce_min(local_boxes, lane) == all_reduce_max(local_boxes, lane))
    return 22;
  if (all_reduce_sum(local_boxes, lane) != static_cast<long>(initial_boxes.size()))
    return 23;

  std::size_t interface_patch = initial_boxes.size();
  Index<Dim> interface_cell{};
  for (std::size_t left = 0; left < initial_boxes.size(); ++left) {
    for (std::size_t right = 0; right < initial_boxes.size(); ++right) {
      if (left == right || initial_owners[left] == initial_owners[right])
        continue;
      const Box<Dim>& left_box = initial_boxes[left];
      const Box<Dim>& right_box = initial_boxes[right];
      bool same_transverse_extent = true;
      for (int axis = 0; axis < Dim; ++axis)
        if (axis != split_axis &&
            (left_box.lo[axis] != right_box.lo[axis] || left_box.hi[axis] != right_box.hi[axis]))
          same_transverse_extent = false;
      if (same_transverse_extent && left_box.hi[split_axis] + 1 == right_box.lo[split_axis]) {
        interface_patch = left;
        interface_cell = left_box.hi;
        break;
      }
    }
    if (interface_patch != initial_boxes.size())
      break;
  }
  if (interface_patch == initial_boxes.size())
    return 25;

  using Resource = test::program_v5::CallbackProgramResource;
  const std::vector<Resource> resources{
      {Resource::Kind::rhs, 0, 0, 0, 0, state_components, state_ghosts}};
  struct CallbackState {
    bool prepared = false;
    bool inject_failure = false;
    bool failure_consumed = false;
    int dispatches = 0;
    int regression_result = 0;
  };
  const auto callback_state = std::make_shared<CallbackState>();
  const std::array route{runtime::program::SameLevelCellTemporalForwardEulerRoute{0, 0, 0}};
  test::install_explicit_amr_callback_program<Dim>(
      system, "test.mpi.cell-temporal-program/program@1", clock, resources, {},
      [callback_state, clock, route, prove_collective_rollback, split_axis](auto& context,
                                                                            double dt) mutable {
        context.begin_step(dt);
        if (!callback_state->prepared) {
          context.prepare_same_level_cell_temporal_execution(clock, 100, 0, route);
          callback_state->prepared = true;
          if (prove_collective_rollback && split_axis == 0)
            callback_state->regression_result =
                run_rank_local_begin_rung_failure_regression(context.prepared_execution_lane());
          if (callback_state->regression_result != 0)
            throw std::runtime_error("cell-local MPI collective rollback regression failed");
        }
        context.advance_same_level_cell_temporal(dt);
        if (callback_state->inject_failure && !callback_state->failure_consumed) {
          callback_state->failure_consumed = true;
          throw runtime::program::StepAttemptRejected(
              SolveStatus::kIterationLimit, runtime::program::StepAttemptDisposition::kRetry,
              0x43454C4Cu, "cell-local-mpi-candidate", "injected-cell-local-retry");
        }
        ++callback_state->dispatches;
      });
  if (system.accepted_transaction_generation_() != 0)
    return 27;
  try {
    system.step(0.01);
  } catch (...) {
    return 3;
  }
  if (callback_state->dispatches != 1 || !callback_state->prepared)
    return 24;
  long remote_neighbour_proof_failed = 0;
  {
    const auto live = system.prepared_amr_block_state(0, 0);
    remote_neighbour_proof_failed = live ? 0L : 1L;
    if (live && live->contains_local(interface_patch)) {
      const std::size_t local = live->local_index_of(interface_patch);
      const Real expected = Real(interface_cell[split_axis]) - Real(0.01);
      if (copied_host_value(std::as_const(live->fab(local)), interface_cell, 0) != expected)
        remote_neighbour_proof_failed = 1;
    }
  }
  if (all_reduce_max(remote_neighbour_proof_failed, lane) != 0)
    return 4;
  if (!prove_collective_rollback)
    return 0;

  const auto accepted_before_reject = system.block_level_state_global("tracer", 0);
  callback_state->inject_failure = true;
  bool rejected = false;
  try {
    system.step(0.01);
  } catch (const runtime::program::StepAttemptRejected&) {
    rejected = true;
  }
  callback_state->inject_failure = false;
  const auto accepted_after_reject = system.block_level_state_global("tracer", 0);
  if (all_reduce_min(rejected ? 1L : 0L, lane) != 1 ||
      accepted_after_reject != accepted_before_reject || system.macro_step() != 1 ||
      system.accepted_transaction_generation_() != 1)
    return 6;

  try {
    system.step(0.01);
  } catch (...) {
    return 8;
  }
  if (callback_state->dispatches != 2 || system.macro_step() != 2 ||
      system.accepted_transaction_generation_() != 2)
    return 9;
  return 0;
}

int pops_run_test_mpi_cell_temporal_program(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int result = 0;
  const int split_axis_count =
#if defined(POPS_CELL_TEMPORAL_MULTIBOX_EVERY_AXIS_PROOF) || \
    defined(POPS_CELL_TEMPORAL_COLLECTIVE_ROLLBACK_PROOF)
      kNativeDimension;
#else
      1;
#endif
  for (int split_axis = 0; split_axis < split_axis_count && result == 0; ++split_axis)
    result = run_collective_program_route(split_axis,
#if defined(POPS_CELL_TEMPORAL_COLLECTIVE_ROLLBACK_PROOF)
                                          true
#else
                                          false
#endif
    );
  comm_finalize();
  return result;
}

}  // namespace

#if defined(POPS_CELL_TEMPORAL_COLLECTIVE_ROLLBACK_PROOF)
#define POPS_CELL_TEMPORAL_MPI_SUITE test_mpi_cell_temporal_program_collective_rollback
#define POPS_CELL_TEMPORAL_MPI_CASE RankLocalFailureRollsBackCollectivelyAndRetrySucceeds
#elif defined(POPS_CELL_TEMPORAL_MULTIBOX_EVERY_AXIS_PROOF)
#define POPS_CELL_TEMPORAL_MPI_SUITE test_mpi_cell_temporal_program_multibox
#define POPS_CELL_TEMPORAL_MPI_CASE MultirankMultiboxExecutesEveryAxis
#else
#define POPS_CELL_TEMPORAL_MPI_SUITE test_mpi_cell_temporal_program
#define POPS_CELL_TEMPORAL_MPI_CASE DirectPreparedSubengineExecutesNeighbourFluxExactly
#endif

TEST(POPS_CELL_TEMPORAL_MPI_SUITE, POPS_CELL_TEMPORAL_MPI_CASE) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_cell_temporal_program,
                                    "test_mpi_cell_temporal_program"),
            0);
}

#undef POPS_CELL_TEMPORAL_MPI_CASE
#undef POPS_CELL_TEMPORAL_MPI_SUITE
