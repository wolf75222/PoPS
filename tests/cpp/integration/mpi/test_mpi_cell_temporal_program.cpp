#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/same_level_cell_temporal_provider.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

// This executable owns the global new/delete replacements.  They are opt-in at runtime so setup,
// diagnostics, and assertions cannot pollute the transaction allocation witness.
std::atomic<bool> g_heap_measurement_enabled{false};
std::atomic<std::uint64_t> g_measured_heap_allocations{0};

void note_measured_heap_allocation() noexcept {
  if (g_heap_measurement_enabled.load(std::memory_order_relaxed))
    g_measured_heap_allocations.fetch_add(1, std::memory_order_relaxed);
}

void* measured_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_allocate_nothrow(std::size_t size) noexcept {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate_nothrow(std::size_t size, std::size_t alignment) noexcept {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

class HeapAllocationWindow {
 public:
  HeapAllocationWindow() : before_(g_measured_heap_allocations.load(std::memory_order_relaxed)) {
    g_heap_measurement_enabled.store(true, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t close() noexcept {
    g_heap_measurement_enabled.store(false, std::memory_order_relaxed);
    return g_measured_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_ = 0;
};

}  // namespace

void* operator new(std::size_t size) {
  return measured_allocate(size);
}
void* operator new[](std::size_t size) {
  return measured_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

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

class RankLocalBeginRungFailure final : public std::exception {
 public:
  const char* what() const noexcept override { return "rank-local injected begin-rung failure"; }
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
      throw RankLocalBeginRungFailure{};
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

class PreparedCollectiveRungFailureRegression {
 public:
  explicit PreparedCollectiveRungFailureRegression(const ExecutionLane& lane) : lane_(&lane) {
    if (lane.size() != 2)
      throw std::invalid_argument("collective rung-failure regression requires two ranks");
    accepted_.kind = runtime::program::TemporalPartitionKind::CellLocal;
    accepted_.provider_identity = "test.mpi.cell-temporal-program.collective-rung-failure@1";
    accepted_.topology_epoch = 1;
    accepted_.synchronization_tick = 0;
    accepted_.tick_denominator = 4;
    accepted_.cells = {{0, 0, 0, 0}};
    state_ = std::make_shared<CollectiveRungFailureState>();

    auto divergent_plan = accepted_;
    if (lane.rank() == 1)
      divergent_plan.cells.push_back({0, 1, 0, 0});
    bool divergent_plan_rejected = false;
    try {
      runtime::program::PreparedBatchedCellTemporalExecutor divergent_executor{
          std::move(divergent_plan), CollectiveRungFailureProvider(state_, lane), lane};
    } catch (const std::invalid_argument& error) {
      divergent_plan_rejected =
          std::string_view(error.what()).find("fixed plan envelope differs") !=
          std::string_view::npos;
    }
    if (all_reduce_min(divergent_plan_rejected ? 1L : 0L, lane) != 1)
      throw std::runtime_error("collective rung-failure divergent preparation was accepted");

    executor_ = std::make_unique<
        runtime::program::PreparedBatchedCellTemporalExecutor<CollectiveRungFailureProvider>>(
        accepted_, CollectiveRungFailureProvider(state_, lane), lane);
    checkpoint_before_ = executor_->checkpoint();
    retry_checkpoint_ = checkpoint_before_;
    divergent_restore_ = checkpoint_before_;
    restored_ = checkpoint_before_;
    accepted_bytes_before_ = state_->accepted;
    failure_message_.reserve(128);
  }

  int run() {
    state_->fail_next_begin = lane_->rank() == 0;
    state_->failure_consumed = false;
    accepted_bytes_before_ = state_->accepted;
    bool collectively_rejected = false;
    failure_message_.clear();
    try {
      executor_->begin_attempt(1);
      executor_->advance_to_barrier();
    } catch (const std::exception& error) {
      collectively_rejected = true;
      failure_message_.assign(error.what());
    }
    const bool checkpoint_restored =
        !executor_->attempt_active() && executor_->accepted_state() == checkpoint_before_;
    const bool byte_exact =
        std::memcmp(state_->accepted.data(), accepted_bytes_before_.data(),
                    accepted_bytes_before_.size() * sizeof(std::uint32_t)) == 0 &&
        std::memcmp(state_->scratch.data(), accepted_bytes_before_.data(),
                    accepted_bytes_before_.size() * sizeof(std::uint32_t)) == 0;
    const bool collective_diagnostic =
        lane_->rank() == 0
            ? failure_message_ == "rank-local injected begin-rung failure"
            : failure_message_ ==
                  "cell-local temporal rung-batch preparation failed on another rank";
    if (all_reduce_max(!collectively_rejected || executor_->attempt_active() ||
                               !collective_diagnostic || !checkpoint_restored || !byte_exact ||
                               state_->commits != 0 || state_->rollbacks != 1 ||
                               state_->device_evaluations[0] != 0
                           ? 1L
                           : 0L,
                       *lane_) != 0)
      return 31;

    bool retry_failed = false;
    try {
      executor_->begin_attempt(1);
      executor_->advance_to_barrier();
      executor_->commit();
      retry_checkpoint_ = executor_->accepted_state();
    } catch (const std::exception&) {
      retry_failed = true;
    }
    if (all_reduce_max(retry_failed || executor_->attempt_active() ||
                               retry_checkpoint_.synchronization_tick != 1 ||
                               state_->commits != 1 || state_->rollbacks != 1 ||
                               state_->device_evaluations[0] != (lane_->rank() == 0 ? 1u : 0u)
                           ? 1L
                           : 0L,
                       *lane_) != 0)
      return 32;

    divergent_restore_ = retry_checkpoint_;
    if (lane_->rank() == 0) {
      divergent_restore_.synchronization_tick = 0;
      divergent_restore_.cells.front().accepted_tick = 0;
    }
    bool divergent_restore_rejected = false;
    try {
      executor_->restore_accepted_boundary(std::move(divergent_restore_));
    } catch (const std::exception& error) {
      divergent_restore_rejected =
          std::string_view(error.what()).find("restore target differs") != std::string_view::npos;
    }
    if (all_reduce_max(!divergent_restore_rejected || executor_->attempt_active() ||
                               executor_->accepted_state() != retry_checkpoint_ ||
                               state_->restores != 0
                           ? 1L
                           : 0L,
                       *lane_) != 0)
      return 34;

    restored_ = retry_checkpoint_;
    restored_.synchronization_tick = 0;
    restored_.cells.front().accepted_tick = 0;
    try {
      executor_->restore_accepted_boundary(restored_);
      executor_->begin_attempt(1);
      executor_->advance_to_barrier();
      executor_->commit();
    } catch (...) {
      return 35;
    }
    return all_reduce_max(executor_->attempt_active() ||
                                  executor_->accepted_state().synchronization_tick != 1 ||
                                  state_->restores != 1 || state_->commits != 2
                              ? 1L
                              : 0L,
                          *lane_) == 0
               ? 0
               : 36;
  }

 private:
  const ExecutionLane* lane_ = nullptr;
  std::shared_ptr<CollectiveRungFailureState> state_;
  std::unique_ptr<
      runtime::program::PreparedBatchedCellTemporalExecutor<CollectiveRungFailureProvider>>
      executor_;
  runtime::program::CellTemporalPartitionAcceptedState accepted_;
  runtime::program::CellTemporalPartitionAcceptedState checkpoint_before_;
  runtime::program::CellTemporalPartitionAcceptedState retry_checkpoint_;
  runtime::program::CellTemporalPartitionAcceptedState divergent_restore_;
  runtime::program::CellTemporalPartitionAcceptedState restored_;
  std::vector<std::uint32_t, fab_allocator<std::uint32_t>> accepted_bytes_before_;
  std::string failure_message_;
};

int run_collective_program_route(int split_axis, bool prove_collective_rollback) {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = axis == split_axis ? 6 : 2;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }
  config.level_count = 2;
  constexpr std::int64_t kCellTemporalTickDenominator = 100;
  // The authored finest rung is zero. With the exact 2:1 level relation the coarsest rung has
  // stride two, so one ordinary Program FE batch spans two ticks.
  constexpr double kCellTemporalStepDt = 2.0 / kCellTemporalTickDenominator;
  config.regrid_every = 0;
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
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "test.mpi.cell-temporal-program/tracer/state@1");
  add_compiled_model<Dim>(system, "tracer", scalar_advection_model<Dim>(split_axis), "minmod",
                          "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          "test.mpi.cell-temporal-program/physical-flux");
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
  test::install_prepared_threshold_union(
      system,
      {{"tracer", "u", 0.0, test::PreparedThresholdRelation::Above,
        "test.mpi.cell-temporal-program/tracer/state@1"}},
      "test.mpi.cell-temporal-program/tagging@1");
  system.refresh_prepared_amr_levels();
  if (!system.uses_runtime_engine())
    return 1;
  if (system.n_levels() != 2)
    return 29;
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
  const test::program_v5::CallbackProgramCellTemporalAuthority cell_temporal{
      clock, kCellTemporalTickDenominator, 0, {{0, -1, 0}}};
  struct CallbackState {
    bool inject_failure = false;
    bool failure_consumed = false;
    bool regression_executed = false;
    int dispatches = 0;
    int regression_result = 0;
  };
  const auto callback_state = std::make_shared<CallbackState>();
  const auto rollback_regression_fixture =
      prove_collective_rollback ? std::make_unique<PreparedCollectiveRungFailureRegression>(lane)
                                : std::unique_ptr<PreparedCollectiveRungFailureRegression>{};
  PreparedCollectiveRungFailureRegression* const rollback_regression =
      rollback_regression_fixture.get();
  test::install_explicit_amr_callback_program<Dim>(
      system, "test.mpi.cell-temporal-program/program@1", clock, resources, {},
      [callback_state, rollback_regression, prove_collective_rollback, split_axis](
          auto& context, double dt) mutable {
        context.begin_step(dt);
        if (prove_collective_rollback && split_axis == 0 && !callback_state->regression_executed) {
          callback_state->regression_result = rollback_regression->run();
          callback_state->regression_executed = true;
        }
        if (callback_state->regression_result != 0)
          throw std::runtime_error("cell-local MPI collective rollback regression failed");
        context.advance_same_level_cell_temporal(dt);
        if (callback_state->inject_failure && !callback_state->failure_consumed) {
          callback_state->failure_consumed = true;
          throw runtime::program::StepAttemptRejected(
              SolveStatus::kIterationLimit, runtime::program::StepAttemptDisposition::kRetry,
              0x43454C4Cu, "cell-local-mpi-candidate", "injected-cell-local-retry");
        }
        ++callback_state->dispatches;
      },
      std::vector<test::program_v5::CallbackProgramHistory>{},
      std::vector<test::program_v5::CallbackProgramClockRelation>{}, std::nullopt,
      test::program_v5::CallbackProgramTransactionAuthorities{}, cell_temporal);
  // Installation publishes the complete POPSAND5/capacity pair before owner-last.  Re-reading the
  // public capacity at bind therefore validates the sealed candidate and must not synthesize a
  // second checkpoint image or change its revision.
  const auto accepted_bootstrap = system.program_accepted_state();
  const auto capacity_bootstrap = system.checkpoint_program_state_capacity();
  const auto accepted_after_capacity = system.program_accepted_state();
  const auto capacity_after_capacity = system.checkpoint_program_state_capacity();
  if (accepted_bootstrap.empty() || accepted_after_capacity != accepted_bootstrap ||
      capacity_bootstrap != capacity_after_capacity ||
      accepted_bootstrap.size() > capacity_bootstrap.first)
    return 26;
  if (system.accepted_transaction_generation_() != 0)
    return 27;
  // The first accepted full step is measured immediately after the candidate has prepared its
  // cell-local executor and bootstrap image.  Construction of the counters, error handling, and
  // collective comparison are deliberately outside begin -> step -> commit -> finalize.
  const AllocationEventStats first_full_step_before = allocation_event_stats();
  HeapAllocationWindow first_full_step_heap;
  bool first_full_step_accepted = true;
  try {
    system.begin_step_transaction();
    system.step(kCellTemporalStepDt);
    system.commit_step_transaction();
    system.finalize_step_transaction();
  } catch (...) {
    first_full_step_accepted = false;
  }
  const std::uint64_t first_full_step_heap_allocations = first_full_step_heap.close();
  const AllocationEventStats first_full_step_after = allocation_event_stats();
  const bool first_full_step_allocation_free = first_full_step_accepted &&
                                               first_full_step_heap_allocations == 0 &&
                                               first_full_step_after == first_full_step_before;
  if (all_reduce_min(first_full_step_allocation_free ? 1L : 0L, lane) != 1 ||
      all_reduce_max(first_full_step_allocation_free ? 1L : 0L, lane) != 1) {
    return 3;
  }
  if (callback_state->dispatches != 1)
    return 24;
  long remote_neighbour_proof_failed = 0;
  {
    const auto live = system.prepared_amr_block_state(0, 0);
    remote_neighbour_proof_failed = live ? 0L : 1L;
    if (live && live->contains_local(interface_patch)) {
      const std::size_t local = live->local_index_of(interface_patch);
      const Real dx = (config.upper[split_axis] - config.lower[split_axis]) /
                      static_cast<Real>(config.shape[split_axis]);
      // The seeded ramp is expressed in index coordinates, so one upwind FE step changes it by
      // dt / dx rather than dt.
      const Real expected = Real(interface_cell[split_axis]) - Real(kCellTemporalStepDt) / dx;
      if (copied_host_value(std::as_const(live->fab(local)), interface_cell, 0) != expected)
        remote_neighbour_proof_failed = 1;
    }
  }
  if (all_reduce_max(remote_neighbour_proof_failed, lane) != 0)
    return 4;

  // Repeat the same complete transaction after the first accepted publication.  This keeps the
  // resident executor witness independent from the first-use path without using an unmeasured
  // warmup step.
  const AllocationEventStats repeated_full_step_before = allocation_event_stats();
  HeapAllocationWindow repeated_full_step_heap;
  bool repeated_full_step_accepted = true;
  try {
    system.begin_step_transaction();
    system.step(kCellTemporalStepDt);
    system.commit_step_transaction();
    system.finalize_step_transaction();
  } catch (...) {
    repeated_full_step_accepted = false;
  }
  const std::uint64_t repeated_full_step_heap_allocations = repeated_full_step_heap.close();
  const AllocationEventStats repeated_full_step_after = allocation_event_stats();
  const bool repeated_full_step_allocation_free =
      repeated_full_step_accepted && repeated_full_step_heap_allocations == 0 &&
      repeated_full_step_after == repeated_full_step_before;
  if (all_reduce_min(repeated_full_step_allocation_free ? 1L : 0L, lane) != 1 ||
      all_reduce_max(repeated_full_step_allocation_free ? 1L : 0L, lane) != 1)
    return 5;
  if (callback_state->dispatches != 2 || system.macro_step() != 2 ||
      system.accepted_transaction_generation_() != 2)
    return 30;
  if (!prove_collective_rollback)
    return 0;

  const auto accepted_before_reject = system.block_level_state_global("tracer", 0);
  callback_state->inject_failure = true;
  bool rejected = false;
  try {
    system.step(kCellTemporalStepDt);
  } catch (const runtime::program::StepAttemptRejected&) {
    rejected = true;
  }
  callback_state->inject_failure = false;
  const auto accepted_after_reject = system.block_level_state_global("tracer", 0);
  if (all_reduce_min(rejected ? 1L : 0L, lane) != 1 ||
      accepted_after_reject != accepted_before_reject || system.macro_step() != 2 ||
      system.accepted_transaction_generation_() != 2)
    return 6;

  const AllocationEventStats warmed_retry_allocations = allocation_event_stats();
  try {
    system.step(kCellTemporalStepDt);
  } catch (...) {
    return 8;
  }
  if (callback_state->dispatches != 3 || system.macro_step() != 3 ||
      system.accepted_transaction_generation_() != 3)
    return 9;
  if (allocation_event_stats() != warmed_retry_allocations)
    return 28;
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
