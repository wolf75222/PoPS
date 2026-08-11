#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/program/cell_temporal_partition_executor.hpp>

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

struct CollectiveRungFailureState {
  std::array<std::uint32_t, 2> accepted{11u, 22u};
  std::array<std::uint32_t, 2> scratch{11u, 22u};
  bool fail_next_begin = false;
  bool failure_consumed = false;
  std::uint32_t device_evaluations = 0;
  int commits = 0;
  int rollbacks = 0;
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
                                CommunicatorView communicator)
      : state_(std::move(state)),
        communicator_(communicator),
        local_indices_{static_cast<std::size_t>(communicator.rank())} {}

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
    state_->scratch = state_->accepted;
    return PreparedProviderSupport::accept();
  }
  void begin_rung_batch(runtime::program::CellTemporalRungBatchDescriptor) {
    const std::size_t local = local_indices_.front();
    state_->scratch[local] = std::uint32_t{0xA5000000u} + static_cast<std::uint32_t>(local);
    if (state_->fail_next_begin && !state_->failure_consumed) {
      state_->failure_consumed = true;
      throw std::runtime_error("rank-local injected begin-rung failure");
    }
  }
  void complete_rung_batch(runtime::program::CellTemporalRungBatchDescriptor) noexcept {}
  [[nodiscard]] PreparedProviderSupport prepare_commit_attempt() noexcept {
    return PreparedProviderSupport::accept();
  }
  void commit_attempt() noexcept {
    state_->accepted = state_->scratch;
    ++state_->commits;
  }
  void rollback_attempt() noexcept {
    state_->scratch = state_->accepted;
    ++state_->rollbacks;
  }
  [[nodiscard]] CollectiveRungFailureDeviceView device_view() const noexcept {
    return {state_->scratch.data(), &state_->device_evaluations, state_->scratch.size()};
  }
  [[nodiscard]] CommunicatorView communicator() const noexcept { return communicator_; }
  [[nodiscard]] std::span<const std::size_t> local_record_indices() const noexcept {
    return local_indices_;
  }

 private:
  std::shared_ptr<CollectiveRungFailureState> state_;
  CommunicatorView communicator_{};
  std::vector<std::size_t> local_indices_;
};

static_assert(runtime::program::CellTemporalStageFluxProvider<CollectiveRungFailureProvider>);
static_assert(runtime::program::CellTemporalRungBatchLifecycle<CollectiveRungFailureProvider>);
static_assert(
    runtime::program::DistributedCellTemporalStageFluxProvider<CollectiveRungFailureProvider>);

int run_rank_local_begin_rung_failure_regression() {
  using namespace runtime::program;
  const CommunicatorView communicator = world_communicator_view();
  if (communicator.size() != 2)
    return 30;

  CellTemporalPartitionAcceptedState accepted;
  accepted.kind = TemporalPartitionKind::CellLocal;
  accepted.provider_identity = "test.mpi.cell-temporal-program.collective-rung-failure@1";
  accepted.topology_epoch = 1;
  accepted.synchronization_tick = 0;
  accepted.tick_denominator = 4;
  accepted.cells = {{0, 0, 0, 0}, {0, 1, 0, 0}};

  const auto state = std::make_shared<CollectiveRungFailureState>();
  state->fail_next_begin = communicator.rank() == 0;
  const auto accepted_bytes_before = state->accepted;
  PreparedBatchedCellTemporalExecutor executor{accepted,
                                               CollectiveRungFailureProvider(state, communicator)};
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
                                      sizeof(accepted_bytes_before)) == 0 &&
                          std::memcmp(state->scratch.data(), accepted_bytes_before.data(),
                                      sizeof(accepted_bytes_before)) == 0;
  const bool collective_diagnostic =
      communicator.rank() == 0
          ? failure_message == "rank-local injected begin-rung failure"
          : failure_message == "cell-local temporal rung-batch preparation failed on another rank";
  const long failed_rollback = !collectively_rejected || executor.attempt_active() ||
                                       !collective_diagnostic || !checkpoint_restored ||
                                       !byte_exact || state->commits != 0 ||
                                       state->rollbacks != 1 || state->device_evaluations != 0
                                   ? 1L
                                   : 0L;
  if (all_reduce_max(failed_rollback, communicator) != 0)
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
  const long failed_retry =
      retry_failed || executor.attempt_active() || retry_checkpoint.synchronization_tick != 1 ||
              state->commits != 1 || state->rollbacks != 1 || state->device_evaluations != 1
          ? 1L
          : 0L;
  return all_reduce_max(failed_retry, communicator) == 0 ? 0 : 32;
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
  const MultiFab<Dim>& initial = engine->hierarchy().state(0);
  const long local_boxes = static_cast<long>(initial.local_size());
  if (initial.layout().size() < 2)
    return 21;
  if (all_reduce_min(local_boxes) == all_reduce_max(local_boxes))
    return 22;
  if (all_reduce_sum(local_boxes) != static_cast<long>(initial.layout().size()))
    return 23;
  system.set_program_block_map({0});
  auto context = std::make_shared<runtime::program::AmrProgramContext<Dim>>(engine, &system);
  const std::string clock = "test.clock.cell-local-mpi-program.axis" + std::to_string(split_axis);
  context->configure_primary_clock(clock);
  context->prepare_same_level_cell_temporal_execution(clock, 100, 0);
  context->install([context](double dt) { context->advance_same_level_cell_temporal(dt); },
                   context);
  system.set_program_block_map({0});

  try {
    system.step(0.01);
  } catch (const std::exception& error) {
    (void)error;
    return 3;
  }
  if (system.time() != 0.01)
    return 4;
  if (!prove_collective_rollback)
    return 0;

  MultiFab<Dim>& live = engine->hierarchy().state(0);
  bool injection_failed = false;
  try {
    if (world_communicator_view().rank() == 0) {
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
  if (all_reduce_max(injection_failed ? 1L : 0L) != 0)
    return 5;

  bool rejected = false;
  try {
    system.step(0.01);
  } catch (const std::exception&) {
    rejected = true;
  }
  if (all_reduce_min(rejected ? 1L : 0L) != 1 || system.time() != 0.01)
    return 6;

  auto* const retry_engine = system.engine();
  if (retry_engine == nullptr)
    return 7;
  MultiFab<Dim>& retry_state = retry_engine->hierarchy().state(0);
  if (world_communicator_view().rank() == 0) {
    const Index<Dim> first = retry_state.box(0).lo;
    for_each_cell(retry_state.box(0), SetFirstCell<Dim>{retry_state.fab(0).view(), first, Real(1),
                                                        retry_state.ncomp()});
    device_fence();
  }
  try {
    system.step(0.01);
  } catch (...) {
    return 8;
  }
  if (system.time() != 0.02)
    return 9;

  bool fractional_tick_refused = false;
  try {
    system.step(0.015);
  } catch (const std::invalid_argument&) {
    fractional_tick_refused = true;
  }
  return all_reduce_min(fractional_tick_refused ? 1L : 0L) == 1 && system.time() == 0.02 ? 0 : 10;
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
#if defined(POPS_CELL_TEMPORAL_COLLECTIVE_ROLLBACK_PROOF)
  result = run_rank_local_begin_rung_failure_regression();
#endif
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
#define POPS_CELL_TEMPORAL_MPI_CASE BoundedProgramContextExecutesForwardEulerExactly
#endif

TEST(POPS_CELL_TEMPORAL_MPI_SUITE, POPS_CELL_TEMPORAL_MPI_CASE) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_cell_temporal_program,
                                    "test_mpi_cell_temporal_program"),
            0);
}

#undef POPS_CELL_TEMPORAL_MPI_CASE
#undef POPS_CELL_TEMPORAL_MPI_SUITE
