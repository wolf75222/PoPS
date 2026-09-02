#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

using namespace pops;

namespace {

std::atomic<std::uint64_t> g_heap_allocations{0};
std::atomic<bool> g_observe_heap_allocations{false};

void* counted_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  if (g_observe_heap_allocations.load(std::memory_order_relaxed))
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

void* counted_allocate_nothrow(std::size_t size) noexcept {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer != nullptr && g_observe_heap_allocations.load(std::memory_order_relaxed))
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

void* counted_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  if (g_observe_heap_allocations.load(std::memory_order_relaxed))
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

void* counted_aligned_allocate_nothrow(std::size_t size, std::size_t alignment) noexcept {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer != nullptr && g_observe_heap_allocations.load(std::memory_order_relaxed))
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

class HeapAllocationWindow final {
 public:
  HeapAllocationWindow() : before_(g_heap_allocations.load(std::memory_order_relaxed)) {
    g_observe_heap_allocations.store(true, std::memory_order_relaxed);
  }
  ~HeapAllocationWindow() { g_observe_heap_allocations.store(false, std::memory_order_relaxed); }
  std::uint64_t close() {
    g_observe_heap_allocations.store(false, std::memory_order_relaxed);
    return g_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_;
};

template <int Dim>
Extent<Dim> uniform_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
MultiFab<Dim> one_patch_field(const Box<Dim>& box, int ncomp) {
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{box});
  const mesh::RankSpace<Dim> ranks(Index<Dim>{}, uniform_extent<Dim>(1));
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  return MultiFab<Dim>(layout, distribution, Index<Dim>{}, ncomp, Extent<Dim>{});
}

template <int Dim>
std::size_t host_offset(const Box<Dim>& storage, const Index<Dim>& index, int component) {
  std::int64_t linear = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + linear);
}

template <int Dim>
void expect_valid_value(const Fab<Dim>& field, Real expected) {
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  for (std::int64_t linear = 0; linear < field.box().numPts(); ++linear) {
    std::int64_t remaining = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = field.box().lo[axis] + static_cast<int>(remaining % field.box().length(axis));
      remaining /= field.box().length(axis);
    }
    EXPECT_NEAR(host(host_offset(field.grown_box(), index, 0)), expected,
                Real(64) * std::numeric_limits<Real>::epsilon());
  }
}

template <int Dim>
struct RankedLinearImplicitModel {
  using State = StateVec<1>;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  POPS_HD State source(const State& state, const auto&) const {
    State result{};
    result[0] = -state[0];
    return result;
  }

  POPS_HD void source_jacobian(const State&, const auto&, Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(-1);
  }
};

enum class FallibleSourceMode { kConverged, kRejectAttempt, kFailRun };

template <int Dim>
struct FallibleImplicitModel {
  using State = StateVec<1>;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  FallibleSourceMode mode = FallibleSourceMode::kConverged;

  POPS_HD ImplicitEvaluationResult evaluate_source(const State& state, const ProviderValues<0>&,
                                                   State& output) const {
    if (mode == FallibleSourceMode::kRejectAttempt)
      return ImplicitEvaluationResult::reject(0x3412u);
    if (mode == FallibleSourceMode::kFailRun)
      return ImplicitEvaluationResult::failed(0x7856u);
    output[0] = -state[0];
    return ImplicitEvaluationResult::ok();
  }

  POPS_HD State source(const State& state, const ProviderValues<0>&) const {
    return State{-state[0]};
  }

  POPS_HD void source_jacobian(const State&, const ProviderValues<0>&,
                               Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(-1);
  }
};

template <int Dim>
struct ProviderBoundImplicitModel {
  using State = StateVec<1>;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 1;

  POPS_HD State source(const State& state, const ProviderValues<1>&) const {
    return State{-state[0]};
  }

  POPS_HD void source_jacobian(const State&, const ProviderValues<1>&,
                               Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(-1);
  }
};

template <int Dim>
SolveOutcome prepared_outcome(const FallibleImplicitModel<Dim>& model, MultiFab<Dim>& state,
                              PreparedImplicitSourceWorkspace<Dim>& workspace,
                              std::uint64_t generation, const ExecutionLane& lane) {
  return backward_euler_source(
      model, [](std::size_t) { return ProviderStorageView<Dim, 0>{}; }, state, workspace,
      generation, Real(0.25), NewtonOptions{}, lane);
}

template <int Dim>
void check_ranked_implicit_provider() {
  using Model = RankedLinearImplicitModel<Dim>;
  const Box<Dim> box = Box<Dim>::from_extents(uniform_extent<Dim>(2));
  auto state = one_patch_field(box, Model::n_vars);
  state.set_val(Real(2));

  NewtonOptions options{};
  const auto provider_at = [](std::size_t) { return ProviderStorageView<Dim, 0>{}; };
  const ExecutionLane lane = ExecutionLane::world("pops.test.implicit-source-nd.provider");
  auto outcome = backward_euler_source(Model{}, provider_at, state, Real(0.25), options, lane);
  ASSERT_TRUE(outcome.report().solved()) << outcome.report().reason;
  const SolveReport accepted = outcome.consume(SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  expect_valid_value(state.fab(0), Real(1.6));
}

template <int Dim>
void check_ranked_failure_collective() {
  Index<Dim> lower{};
  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -3 - axis;
    upper[axis] = 2 + axis;
  }
  const Box<Dim> box{lower, upper};
  auto statistics = one_patch_field(box, 13);
  statistics.set_val(Real(0));

  Index<Dim> selected = upper;
  selected[Dim - 1] = lower[Dim - 1];
  Index<Dim> competitor = lower;
  competitor[Dim - 1] = upper[Dim - 1];

  auto host = statistics.fab(0).create_host_mirror();
  statistics.fab(0).copy_to_host(host);
  host(host_offset(statistics.fab(0).grown_box(), selected, 12)) = Real(7);
  host(host_offset(statistics.fab(0).grown_box(), selected, 8)) = Real(4);
  host(host_offset(statistics.fab(0).grown_box(), competitor, 12)) = Real(7);
  host(host_offset(statistics.fab(0).grown_box(), competitor, 8)) = Real(1);
  statistics.fab(0).copy_from_host(host);

  const ExecutionLane lane = ExecutionLane::world("pops.test.implicit-source-nd.failure");
  const auto location = collective_first_local_nonlinear_failure(statistics, 7, 12, 8, lane);
  ASSERT_TRUE(location.found);
  EXPECT_EQ(location.priority, 7);
  EXPECT_EQ(location.component, 4);
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_EQ(location.index[axis], selected[axis]);

  const SolveFailureLocation reported =
      SolveFailureLocation::from<Dim>(location.index, location.component);
  const SolveReport solve = local_nonlinear_solve_report(
      local_nonlinear_status_code(LocalNonlinearStatus::kInvalidEvaluation), 2, 3, Real(1),
      Real(0.5), Real(0.25), Real(1), 0, reported, SolveAction::kFailRun);
  ASSERT_TRUE(solve.valid());
  ASSERT_TRUE(solve.failure.found);
  EXPECT_EQ(solve.failure.rank, Dim);
  EXPECT_EQ(solve.failure.component, 4);
  for (int axis = 0; axis < SolveFailureLocation::maximum_rank; ++axis)
    EXPECT_EQ(solve.failure.index[static_cast<std::size_t>(axis)], axis < Dim ? selected[axis] : 0);
}

}  // namespace

TEST(test_implicit_source_nd, provider_instantiates_one_algorithm_in_1d_2d_3d) {
  check_ranked_implicit_provider<1>();
  check_ranked_implicit_provider<2>();
  check_ranked_implicit_provider<3>();
}

TEST(test_implicit_source_nd, collective_selects_exact_signed_indices_in_1d_2d_3d) {
  check_ranked_failure_collective<1>();
  check_ranked_failure_collective<2>();
  check_ranked_failure_collective<3>();
}

TEST(test_implicit_source_nd,
     prepared_workspace_first_repeat_accept_reject_and_failrun_allocate_nothing) {
  constexpr int Dim = 2;
  const Box<Dim> box = Box<Dim>::from_extents(uniform_extent<Dim>(2));
  auto state = one_patch_field(box, 1);
  const ExecutionLane lane = ExecutionLane::world("pops.test.implicit-source-nd.prepared");
  PreparedImplicitSourceWorkspace<Dim> workspace;
  workspace.bind(state, 41);
  ASSERT_EQ(workspace.allocation_count(), std::size_t{3});

  state.set_val(Real(2));
  FallibleImplicitModel<Dim> model{};
  const AllocationEventStats first_events = allocation_event_stats();
  {
    HeapAllocationWindow heap;
    auto outcome = prepared_outcome(model, state, workspace, 41, lane);
    ASSERT_TRUE(outcome.report().solved());
    EXPECT_TRUE(outcome.consume(SolveConsumption::kAccept).solved());
    EXPECT_EQ(heap.close(), std::uint64_t{0});
  }
  EXPECT_EQ(allocation_event_stats(), first_events);
  expect_valid_value(state.fab(0), Real(1.6));
  EXPECT_FALSE(workspace.in_flight());

  // Program owns a detached stage carrier, not the level state used to cold-bind this workspace.
  // The exact shape/generation contract authenticates that target without a warm rebind.
  auto detached_stage = one_patch_field(box, 1);
  detached_stage.set_val(Real(2));
  const AllocationEventStats detached_events = allocation_event_stats();
  {
    HeapAllocationWindow heap;
    auto outcome = prepared_outcome(model, detached_stage, workspace, 41, lane);
    EXPECT_TRUE(outcome.consume(SolveConsumption::kAccept).solved());
    EXPECT_EQ(heap.close(), std::uint64_t{0});
  }
  EXPECT_EQ(allocation_event_stats(), detached_events);
  expect_valid_value(state.fab(0), Real(1.6));
  expect_valid_value(detached_stage.fab(0), Real(1.6));
  EXPECT_FALSE(workspace.in_flight());

  state.set_val(Real(2));
  model.mode = FallibleSourceMode::kRejectAttempt;
  const AllocationEventStats reject_events = allocation_event_stats();
  {
    HeapAllocationWindow heap;
    auto outcome = prepared_outcome(model, state, workspace, 41, lane);
    EXPECT_EQ(outcome.report().action, SolveAction::kRejectAttempt);
    EXPECT_EQ(outcome.consume(SolveConsumption::kRejectAttempt).action,
              SolveAction::kRejectAttempt);
    EXPECT_EQ(heap.close(), std::uint64_t{0});
  }
  EXPECT_EQ(allocation_event_stats(), reject_events);
  expect_valid_value(state.fab(0), Real(2));
  EXPECT_EQ(workspace.reason_code(), 0x3412u);
  EXPECT_FALSE(workspace.in_flight());

  model.mode = FallibleSourceMode::kFailRun;
  const AllocationEventStats fail_events = allocation_event_stats();
  {
    HeapAllocationWindow heap;
    auto outcome = prepared_outcome(model, state, workspace, 41, lane);
    EXPECT_EQ(outcome.report().action, SolveAction::kFailRun);
    EXPECT_EQ(outcome.consume(SolveConsumption::kFailRun).action, SolveAction::kFailRun);
    EXPECT_EQ(heap.close(), std::uint64_t{0});
  }
  EXPECT_EQ(allocation_event_stats(), fail_events);
  expect_valid_value(state.fab(0), Real(2));
  EXPECT_EQ(workspace.reason_code(), 0x7856u);

  model.mode = FallibleSourceMode::kConverged;
  state.set_val(Real(2));
  const AllocationEventStats repeated_events = allocation_event_stats();
  {
    HeapAllocationWindow heap;
    auto outcome = prepared_outcome(model, state, workspace, 41, lane);
    EXPECT_TRUE(outcome.consume(SolveConsumption::kAccept).solved());
    EXPECT_EQ(heap.close(), std::uint64_t{0});
  }
  EXPECT_EQ(allocation_event_stats(), repeated_events);
  expect_valid_value(state.fab(0), Real(1.6));
}

TEST(test_implicit_source_nd, prepared_workspace_refuses_stale_generation_shape_and_reentrancy) {
  constexpr int Dim = 1;
  const Box<Dim> box = Box<Dim>::from_extents(uniform_extent<Dim>(2));
  auto state = one_patch_field(box, 1);
  state.set_val(Real(2));
  const ExecutionLane lane = ExecutionLane::world("pops.test.implicit-source-nd.refusal");
  PreparedImplicitSourceWorkspace<Dim> workspace;
  workspace.bind(state, 7);
  FallibleImplicitModel<Dim> model{};

  auto stale = prepared_outcome(model, state, workspace, 8, lane);
  EXPECT_EQ(stale.report().status, SolveStatus::kInvalidInput);
  EXPECT_EQ(stale.consume(SolveConsumption::kFailRun).reason, "stale");
  expect_valid_value(state.fab(0), Real(2));

  auto first = prepared_outcome(model, state, workspace, 7, lane);
  ASSERT_TRUE(first.report().solved());
  EXPECT_TRUE(workspace.in_flight());
  auto second = prepared_outcome(model, state, workspace, 7, lane);
  EXPECT_EQ(second.report().status, SolveStatus::kCapabilityFailure);
  EXPECT_EQ(second.consume(SolveConsumption::kFailRun).reason, "busy");
  expect_valid_value(state.fab(0), Real(2));
  EXPECT_TRUE(first.consume(SolveConsumption::kAccept).solved());
  EXPECT_FALSE(workspace.in_flight());

  state = one_patch_field(Box<Dim>::from_extents(uniform_extent<Dim>(3)), 1);
  state.set_val(Real(2));
  auto shape = prepared_outcome(model, state, workspace, 7, lane);
  EXPECT_EQ(shape.report().status, SolveStatus::kInvalidInput);
  EXPECT_EQ(shape.consume(SolveConsumption::kFailRun).reason, "stale");
  expect_valid_value(state.fab(0), Real(2));
}

TEST(test_implicit_source_nd, prepared_workspace_collectively_releases_a_provider_fault) {
  constexpr int Dim = 1;
  const Box<Dim> box = Box<Dim>::from_extents(uniform_extent<Dim>(2));
  auto state = one_patch_field(box, 1);
  state.set_val(Real(2));
  const ExecutionLane lane = ExecutionLane::world("pops.test.implicit-source-nd.provider-fault");
  PreparedImplicitSourceWorkspace<Dim> workspace;
  workspace.bind(state, 11);

  auto outcome = backward_euler_source(
      ProviderBoundImplicitModel<Dim>{},
      [](std::size_t) -> ProviderStorageView<Dim, 1> {
        throw std::runtime_error("injected prepared implicit provider fault");
      },
      state, workspace, 11, Real(0.25), NewtonOptions{}, lane);
  EXPECT_EQ(outcome.report().status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(outcome.consume(SolveConsumption::kFailRun).reason, "kernel");
  EXPECT_FALSE(workspace.in_flight());
  expect_valid_value(state.fab(0), Real(2));
}

void* operator new(std::size_t size) {
  return counted_allocate(size);
}
void* operator new[](std::size_t size) {
  return counted_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return counted_allocate_nothrow(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return counted_allocate_nothrow(size);
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
  return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return counted_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return counted_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
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
