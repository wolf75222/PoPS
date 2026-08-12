#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
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

template <int Dim>
struct SetFirstCell {
  FieldView<Real, Dim> values{};
  Index<Dim> cell{};
  Real value = Real(0);
  int components = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (index == cell)
      for (int component = 0; component < components; ++component)
        values(index, component) = value;
  }
};

template <int Dim>
struct SetLinearCellState {
  FieldView<Real, Dim> values{};
  int varying_axis = 0;
  int components = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < components; ++component)
      values(index, component) = Real(index[varying_axis]) + Real(100 * component);
  }
};

template <int Dim>
class DirectPreparedCellTemporalRuntime {
 public:
  // This lot proves the prepared same-level subengine directly. Wiring these contracts through
  // AmrProgramContext remains the separate facade lot; no registered positive calls its
  // deliberately unavailable temporal entry points here.
  static constexpr int dimension = Dim;
  using context_type = runtime::program::AmrProgramContext<Dim>;
  using runtime_type = typename context_type::runtime_type;
  using field_type = MultiFab<Dim>;

  DirectPreparedCellTemporalRuntime(context_type& context, runtime_type& runtime, int varying_axis,
                                    const ExecutionLane& lane)
      : runtime_(&runtime),
        geometry_(context.geometry()),
        lane_(&lane),
        varying_axis_(varying_axis) {
    periodicity_.fill(true);
    boundary_ = std::make_unique<runtime::program::PreparedScalarBoundarySession<Dim>>(
        geometry_, BoundaryTopology<Dim>::axis_periodic(periodicity_),
        runtime_->hierarchy().state(0), lane, 1);
  }

  [[nodiscard]] std::uint64_t topology_epoch() const noexcept { return runtime_->topology_epoch(); }
  [[nodiscard]] std::uint64_t materialization_generation() const noexcept {
    return runtime_->materialization_generation();
  }
  [[nodiscard]] std::size_t same_level_cell_block_count() const noexcept { return 1; }
  [[nodiscard]] int same_level_cell_level_count() const noexcept { return 1; }
  [[nodiscard]] field_type& same_level_cell_state() noexcept {
    return runtime_->hierarchy().state(0);
  }
  [[nodiscard]] const Geometry<Dim>& same_level_cell_geometry() const noexcept { return geometry_; }
  [[nodiscard]] const std::array<bool, Dim>& same_level_cell_periodicity() const noexcept {
    return periodicity_;
  }
  [[nodiscard]] std::string_view same_level_cell_state_identity() const noexcept {
    return "test://mpi/direct-prepared-cell-temporal/state";
  }
  [[nodiscard]] std::string_view same_level_cell_flux_provider_identity() const noexcept {
    return "test://mpi/direct-prepared-cell-temporal/central-neighbour-flux";
  }
  [[nodiscard]] std::string_view same_level_cell_flux_parameter_contract() const noexcept {
    return "test.mpi.direct-cell-temporal.central-neighbour-flux.v1";
  }
  [[nodiscard]] std::string_view same_level_cell_stage_snapshot_contract() const noexcept {
    return "test.mpi.direct-cell-temporal.amr-halo-snapshot.v1";
  }

  void prepare_same_level_cell_stage_snapshot(const runtime::multiblock::BoundaryEvaluationPoint&,
                                              field_type& snapshot, const ExecutionLane& lane) {
    if (lane.identity() != lane_->identity() || !lane.congruent_with(*lane_))
      throw std::invalid_argument("direct temporal snapshot received another execution lane");
    boundary_->fill(snapshot);
  }

  void capture_same_level_negative_flux_divergence(
      const runtime::multiblock::BoundaryEvaluationPoint&, const field_type& snapshot,
      field_type& residual, const std::array<field_type*, Dim>& fluxes) {
    const int components = snapshot.ncomp();
    for (int axis = 0; axis < Dim; ++axis) {
      fluxes[axis]->set_val(Real(0));
      for (std::size_t local = 0; local < fluxes[axis]->local_size(); ++local) {
        const auto state = snapshot.fab(local).view();
        const auto flux = fluxes[axis]->fab(local).view();
        for_each_cell(fluxes[axis]->box(local), [=] POPS_HD(const Index<Dim>& face) {
          Index<Dim> left = face;
          --left[axis];
          for (int component = 0; component < components; ++component)
            flux(face, component) = Real(0.5) * (state(left, component) + state(face, component));
        });
      }
    }
    device_fence();
    residual.set_val(Real(0));
    for (std::size_t local = 0; local < residual.local_size(); ++local) {
      const auto output = residual.fab(local).view();
      std::array<FieldView<const Real, Dim>, Dim> prepared_fluxes{};
      for (int axis = 0; axis < Dim; ++axis)
        prepared_fluxes[axis] = std::as_const(*fluxes[axis]).fab(local).view();
      for_each_cell(residual.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component) {
          Real value = Real(0);
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> high = cell;
            ++high[axis];
            value -=
                prepared_fluxes[axis](high, component) - prepared_fluxes[axis](cell, component);
          }
          output(cell, component) = value;
        }
      });
    }
    device_fence();
    ++capture_count_;
  }

  [[nodiscard]] int varying_axis() const noexcept { return varying_axis_; }
  [[nodiscard]] int capture_count() const noexcept { return capture_count_; }

 private:
  runtime_type* runtime_ = nullptr;
  Geometry<Dim> geometry_;
  std::array<bool, Dim> periodicity_{};
  const ExecutionLane* lane_ = nullptr;
  std::unique_ptr<runtime::program::PreparedScalarBoundarySession<Dim>> boundary_;
  int varying_axis_ = 0;
  int capture_count_ = 0;
};

using DirectPreparedNativeRuntime = DirectPreparedCellTemporalRuntime<kNativeDimension>;
using DirectPreparedNativeProvider =
    runtime::program::PreparedSameLevelTransportEulerStageFluxProvider<kNativeDimension,
                                                                       DirectPreparedNativeRuntime>;
static_assert(
    runtime::program::SameLevelCellTemporalRuntime<kNativeDimension, DirectPreparedNativeRuntime>);
static_assert(
    runtime::program::DistributedCellTemporalStageFluxProvider<DirectPreparedNativeProvider>);

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
  system.install_block_state_route("tracer", "test.mpi.cell-temporal-program/tracer/state@1");
  add_compiled_model<Dim>(system, "tracer", scalar_advection_model<Dim>(split_axis), "minmod",
                          "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          "test.mpi.cell-temporal-program/physical-flux");
  auto* const engine = system.engine();
  if (!system.uses_runtime_engine() || engine == nullptr)
    return 1;
  system.set_program_block_map({0});
  auto context = std::make_shared<runtime::program::AmrProgramContext<Dim>>(engine, &system);
  const std::string clock = "test.clock.cell-local-mpi-program.axis" + std::to_string(split_axis);
  context->configure_primary_clock(clock);
  const ExecutionLane& lane = context->prepared_execution_lane();
  const MultiFab<Dim>& initial = engine->hierarchy().state(0);
  const long local_boxes = static_cast<long>(initial.local_size());
  if (initial.layout().size() < 2)
    return 21;
  if (all_reduce_min(local_boxes, lane) == all_reduce_max(local_boxes, lane))
    return 22;
  if (all_reduce_sum(local_boxes, lane) != static_cast<long>(initial.layout().size()))
    return 23;
  if (prove_collective_rollback && split_axis == 0) {
    const int phase_regression = run_rank_local_begin_rung_failure_regression(lane);
    if (phase_regression != 0)
      return phase_regression;
  }
  MultiFab<Dim>& live = engine->hierarchy().state(0);
  for (std::size_t local = 0; local < live.local_size(); ++local)
    for_each_cell(live.box(local),
                  SetLinearCellState<Dim>{live.fab(local).view(), split_axis, live.ncomp()});
  device_fence();

  DirectPreparedCellTemporalRuntime<Dim> direct_runtime(*context, *engine, split_axis, lane);
  using DirectProvider = runtime::program::PreparedSameLevelTransportEulerStageFluxProvider<
      Dim, DirectPreparedCellTemporalRuntime<Dim>>;
  const runtime::program::CellTemporalPartitionAcceptedState partition =
      runtime::program::prepare_same_level_transport_euler_partition<Dim>(direct_runtime, 0, 100, 0,
                                                                          lane);
  std::size_t local_cells = 0;
  for (std::size_t local = 0; local < live.local_size(); ++local)
    local_cells += static_cast<std::size_t>(live.box(local).numPts());
  auto ledger = std::make_shared<runtime::program::SameLevelCellIntegratedFluxLedger<Dim>>(
      direct_runtime.topology_epoch(), direct_runtime.materialization_generation(), 0, 0,
      local_cells, live.ncomp());
  DirectProvider provider(direct_runtime, partition, ledger, clock, lane);
  runtime::program::PreparedBatchedCellTemporalExecutor executor(partition, std::move(provider),
                                                                 lane);
  if (executor.provider_identity() != runtime::program::kSameLevelTransportEulerStageFluxProvider ||
      executor.execution_lane().identity() != lane.identity() ||
      !executor.execution_lane().congruent_with(lane))
    return 24;

  std::size_t interface_patch = initial.layout().size();
  Index<Dim> interface_cell{};
  for (std::size_t left = 0; left < initial.layout().size(); ++left) {
    for (std::size_t right = 0; right < initial.layout().size(); ++right) {
      if (left == right ||
          initial.distribution().owner(left) == initial.distribution().owner(right))
        continue;
      const Box<Dim>& left_box = initial.layout()[left];
      const Box<Dim>& right_box = initial.layout()[right];
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
    if (interface_patch != initial.layout().size())
      break;
  }
  if (interface_patch == initial.layout().size())
    return 25;

  try {
    executor.begin_attempt(1);
    executor.advance_to_barrier();
    executor.commit();
  } catch (...) {
    return 3;
  }
  const auto first_checkpoint = executor.checkpoint();
  long remote_neighbour_proof_failed = first_checkpoint.synchronization_tick != 1 ||
                                               direct_runtime.capture_count() != 1 ||
                                               ledger->publication_generation() != 1
                                           ? 1L
                                           : 0L;
  if (live.contains_local(interface_patch)) {
    const std::size_t local = live.local_index_of(interface_patch);
    const Real expected = Real(interface_cell[split_axis]) - Real(0.01);
    if (copied_host_value(std::as_const(live).fab(local), interface_cell, 0) != expected)
      remote_neighbour_proof_failed = 1;
  }
  if (all_reduce_max(remote_neighbour_proof_failed, lane) != 0)
    return 4;
  if (!prove_collective_rollback)
    return 0;

  bool injection_failed = false;
  try {
    if (lane.rank() == 0) {
      if (live.local_size() == 0)
        throw std::logic_error("rank zero lost its local coarse patch");
      const Index<Dim> first = live.box(0).lo;
      for_each_cell(live.box(0),
                    SetFirstCell<Dim>{live.fab(0).view(), first,
                                      std::numeric_limits<Real>::quiet_NaN(), live.ncomp()});
      device_fence();
    }
  } catch (const std::exception&) {
    injection_failed = true;
  }
  if (all_reduce_max(injection_failed ? 1L : 0L, lane) != 0)
    return 5;

  bool rejected = false;
  try {
    executor.begin_attempt(2);
    executor.advance_to_barrier();
    executor.commit();
  } catch (const std::exception&) {
    rejected = true;
  }
  if (all_reduce_min(rejected ? 1L : 0L, lane) != 1 || executor.attempt_active() ||
      executor.checkpoint().synchronization_tick != 1 || ledger->publication_generation() != 1)
    return 6;

  if (lane.rank() == 0) {
    const Index<Dim> first = live.box(0).lo;
    for_each_cell(live.box(0), SetFirstCell<Dim>{live.fab(0).view(), first, Real(1), live.ncomp()});
    device_fence();
  }
  try {
    executor.begin_attempt(2);
    executor.advance_to_barrier();
    executor.commit();
  } catch (...) {
    return 8;
  }
  if (executor.checkpoint().synchronization_tick != 2 || ledger->publication_generation() != 2)
    return 9;

  const runtime::program::CellTemporalPartitionAcceptedState off_grid_partition =
      runtime::program::prepare_same_level_transport_euler_partition<Dim>(direct_runtime, 2, 100, 1,
                                                                          lane);
  auto off_grid_ledger = std::make_shared<runtime::program::SameLevelCellIntegratedFluxLedger<Dim>>(
      direct_runtime.topology_epoch(), direct_runtime.materialization_generation(), 0, 0,
      local_cells, live.ncomp());
  DirectProvider off_grid_provider(direct_runtime, off_grid_partition, off_grid_ledger, clock,
                                   lane);
  runtime::program::PreparedBatchedCellTemporalExecutor off_grid_executor(
      off_grid_partition, std::move(off_grid_provider), lane);
  bool off_grid_tick_refused = false;
  try {
    off_grid_executor.begin_attempt(3);
  } catch (const std::invalid_argument&) {
    off_grid_tick_refused = true;
  }
  const long off_grid_failure = !off_grid_tick_refused || off_grid_executor.attempt_active() ||
                                        off_grid_executor.checkpoint() != off_grid_partition ||
                                        off_grid_ledger->publication_generation() != 0
                                    ? 1L
                                    : 0L;
  return all_reduce_max(off_grid_failure, lane) == 0 ? 0 : 10;
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
