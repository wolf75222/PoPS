#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>
#include <pops/runtime/accelerator/prepared_stream_executor.hpp>

using namespace pops;
using runtime::program::PreparedResourceCache;
using runtime::program::PreparedResourceAction;

namespace {
struct Resource {
  int shape;
  int evaluated = 0;
  Resource(int requested, int& constructions) : shape(requested) { ++constructions; }
};

// Actual numerical storage: a failed candidate must neither replace nor release the
// accepted buffer on any rank. No move/copy is needed to publish prepared resources.
struct BufferedResource {
  Kokkos::View<double*> values;
  BufferedResource(int size, bool fail, int& constructions)
      : values("prepared-resource-candidate", size) {
    ++constructions;
    Kokkos::deep_copy(values, 42.0);
    if (fail)
      throw std::runtime_error("rank-local candidate allocation failure");
  }
  BufferedResource(const BufferedResource&) = delete;
  BufferedResource(BufferedResource&&) = delete;
};
}  // namespace

TEST(PreparedResourceCache, ReusesStorageWithoutAliasingNodesBlocksOrLevels) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-resource-lifetime");
  PreparedResourceCache cache;
  int constructions = 0;
  auto acquire = [&](int node, int block, int level, int shape) -> Resource& {
    return cache.acquire<Resource>(
        node, block, level, lane, [=](const Resource& resource) { return resource.shape == shape; },
        shape, constructions);
  };
  Resource& stage = acquire(1, 0, 0, 8);
  stage.evaluated = 17;
  EXPECT_EQ(&acquire(1, 0, 0, 8), &stage);
  EXPECT_EQ(acquire(1, 0, 0, 8).evaluated, 17);
  EXPECT_EQ(constructions, 1);
  acquire(2, 0, 0, 8).evaluated = 23;
  acquire(1, 1, 0, 8).evaluated = 29;
  acquire(1, 0, 1, 8).evaluated = 31;
  EXPECT_EQ(stage.evaluated, 17);
  EXPECT_EQ(constructions, 4);
  EXPECT_EQ(acquire(1, 0, 0, 16).evaluated, 0);
  EXPECT_EQ(constructions, 5);
  cache.clear();
  EXPECT_EQ(acquire(1, 0, 0, 16).evaluated, 0);
  EXPECT_EQ(constructions, 6);
}

TEST(PreparedResourceCache, RankLocalInvalidationAndPreflightFailureAreCollective) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-resource-consensus");
  PreparedResourceCache cache;
  int constructions = 0;
  cache.acquire<Resource>(3, 0, 0, lane, [](const Resource&) { return true; }, 8, constructions);
  cache.acquire<Resource>(
      3, 0, 0, lane, [&](const Resource&) { return lane.rank() != 0; }, 8, constructions);
  EXPECT_EQ(constructions, 2);
  EXPECT_THROW(cache.acquire<Resource>(
                   3, 0, 0, lane,
                   [&](const Resource&) {
                     if (lane.rank() == 0)
                       throw std::runtime_error("rank-local invalid preparation");
                     return true;
                   },
                   8, constructions),
               std::runtime_error);
  EXPECT_EQ(constructions, 2);
  EXPECT_EQ(
      cache.acquire<Resource>(
               3, 0, 0, lane, [](const Resource&) { return true; }, 8, constructions)
          .shape,
      8);
  EXPECT_EQ(constructions, 2);
}

TEST(PreparedResourceCache, FailedConstructionPreservesAcceptedBufferOnEveryRank) {
  comm_init();
  detail::ensure_kokkos_initialized();
  auto lane = ExecutionLane::world("prepared-resource-construction-rollback");
  PreparedResourceCache cache;
  int constructions = 0;
  auto acquire = [&](int size, bool fail) -> BufferedResource& {
    return cache.acquire<BufferedResource>(
        4, 0, 0, lane,
        [=](const BufferedResource& resource) {
          return resource.values.extent(0) == static_cast<std::size_t>(size);
        },
        size, fail, constructions);
  };
  auto& accepted = acquire(8, false);
  const auto* accepted_resource = &accepted;
  const auto* accepted_buffer = accepted.values.data();
  Kokkos::deep_copy(accepted.values, 17.0);

  EXPECT_THROW(acquire(16, lane.rank() == 0), std::runtime_error);

  // Reacquiring the old shape must be a cache hit, with its original content and
  // identity. A copied buffer or a successful candidate on a peer is insufficient.
  auto& restored = acquire(8, false);
  EXPECT_EQ(constructions, 2);
  EXPECT_EQ(&restored, accepted_resource);
  EXPECT_EQ(restored.values.data(), accepted_buffer);
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, restored.values);
  EXPECT_DOUBLE_EQ(host(0), 17.0);
  EXPECT_DOUBLE_EQ(host(7), 17.0);

  auto& replacement = acquire(16, false);
  EXPECT_EQ(constructions, 3);
  EXPECT_EQ(replacement.values.extent(0), 16u);
  const auto next = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, replacement.values);
  EXPECT_DOUBLE_EQ(next(15), 42.0);
}

TEST(PreparedResourceCache, FailedInitialConstructionRemainsCollectivelyRetryable) {
  comm_init();
  detail::ensure_kokkos_initialized();
  auto lane = ExecutionLane::world("prepared-resource-first-construction");
  PreparedResourceCache cache;
  int constructions = 0;
  EXPECT_THROW(cache.acquire<BufferedResource>(
                   5, 0, 0, lane, [](const BufferedResource&) { return true; },
                   8, lane.rank() == 0, constructions),
               std::runtime_error);
  auto& result = cache.acquire<BufferedResource>(
      5, 0, 0, lane, [](const BufferedResource&) { return true; }, 8, false, constructions);
  EXPECT_EQ(constructions, 2);
  EXPECT_EQ(result.values.extent(0), 8u);
}

namespace {
using CpuExecutor =
    runtime::accelerator::PreparedAcceleratorStreamExecutor<double,
                                                            Kokkos::DefaultHostExecutionSpace>;
using ResourceTask = runtime::program::PreparedResourceTask;

struct TrackedBuffer {
  Kokkos::View<double*, Kokkos::HostSpace> values;
  std::shared_ptr<int> destructions;
  TrackedBuffer(int size, std::shared_ptr<int> released, bool fail = false)
      : values("leased-native-buffer", size), destructions(std::move(released)) {
    Kokkos::deep_copy(values, 17.0);
    if (fail)
      throw std::runtime_error("rank-local leased buffer preparation failure");
  }
  TrackedBuffer(const TrackedBuffer&) = delete;
  TrackedBuffer(TrackedBuffer&&) = delete;
  ~TrackedBuffer() { ++*destructions; }
};

auto prepare_cpu_executor(PreparedResourceCache& cache, const ExecutionLane& lane) {
  return cache.acquire_lease<CpuExecutor>(
      91, 0, 0, lane, [](const CpuExecutor&) { return true; }, CpuExecutor::prepare_synchronous(8));
}

auto prepare_buffer(PreparedResourceCache& cache, const ExecutionLane& lane, int node, int size,
                    const std::shared_ptr<int>& released, bool fail = false) {
  return cache.acquire_lease<TrackedBuffer>(
      node, 0, 0, lane,
      [=](const TrackedBuffer& buffer) { return buffer.values.extent(0) == std::size_t(size); },
      size, released, fail);
}
}  // namespace

TEST(PreparedResourceCache, LeaseRetainsActualBufferAcrossReplacementAndClear) {
  comm_init();
  detail::ensure_kokkos_initialized();
  auto lane = ExecutionLane::world("prepared-lease-replacement");
  PreparedResourceCache cache;
  auto released = std::make_shared<int>(0);
  auto old = prepare_buffer(cache, lane, 92, 8, released);
  const auto version = old.version();
  auto* pointer = old->values.data();
  auto replacement = prepare_buffer(cache, lane, 92, 16, released);
  EXPECT_FALSE(old.current());
  EXPECT_TRUE(replacement.current());
  EXPECT_GT(replacement.version(), version);
  EXPECT_NE(pointer, replacement->values.data());
  EXPECT_DOUBLE_EQ(old->values(7), 17.0);
  cache.clear();
  EXPECT_FALSE(replacement.current());
  EXPECT_EQ(*released, 0);
  old = {};
  EXPECT_EQ(*released, 1);
  replacement = {};
  EXPECT_EQ(*released, 2);
}

TEST(PreparedResourceCache, FailedCandidateDoesNotRevokeAnExistingLeaseOnAnyRank) {
  comm_init();
  detail::ensure_kokkos_initialized();
  auto lane = ExecutionLane::world("prepared-lease-construction-failure");
  PreparedResourceCache cache;
  auto released = std::make_shared<int>(0);
  auto accepted = prepare_buffer(cache, lane, 92, 8, released);
  const auto version = accepted.version();
  EXPECT_THROW(prepare_buffer(cache, lane, 92, 16, released, lane.rank() == 0), std::runtime_error);
  auto hit = prepare_buffer(cache, lane, 92, 8, released);
  EXPECT_TRUE(accepted.current());
  EXPECT_EQ(hit.version(), version);
  EXPECT_EQ(hit->values.data(), accepted->values.data());
  EXPECT_DOUBLE_EQ(hit->values(7), 17.0);
}

TEST(PreparedResourceCache, CpuSubmissionRetainsCancelledWorkUntilCompletionAcknowledgement) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-cpu-cancel-late-ack");
  PreparedResourceCache cache;
  auto executor = prepare_cpu_executor(cache, lane);
  EXPECT_FALSE(executor->evidence().independent_streams);
  auto released = std::make_shared<int>(0);
  auto buffer = prepare_buffer(cache, lane, 92, 8, released);
  auto attempt = cache.begin_attempt();
  double* output = buffer->values.data();
  double* scratch = executor->workspace_data(0);
  auto task = executor->submit_for(
      cache, attempt, executor, 0, "leased-cpu-write", 8,
      KOKKOS_LAMBDA(std::int64_t i) {
        scratch[i] = 2 * output[i];
        output[i] = scratch[i] + 1;
      },
      buffer);
  // The CPU backend has already fenced. Late acknowledgement below exercises
  // lifetime semantics, not an invented pending GPU event or CPU/GPU overlap.
  EXPECT_DOUBLE_EQ(buffer->values(7), 35.0);
  EXPECT_EQ(task.status(), ResourceTask::Status::pending);
  EXPECT_EQ(task.retained_resource_count(), 2u);
  EXPECT_FALSE(cache.quiescent());
  executor = {};
  buffer = {};
  cache.clear();
  EXPECT_EQ(*released, 0);
  EXPECT_EQ(task.status(), ResourceTask::Status::cancel_requested);
  EXPECT_THROW(task.require_consumable(attempt), std::logic_error);
  EXPECT_TRUE(task.poll());
  EXPECT_TRUE(task.physically_complete());
  EXPECT_TRUE(cache.quiescent());
  EXPECT_EQ(task.status(), ResourceTask::Status::cancelled);
  EXPECT_EQ(task.retained_resource_count(), 0u);
  EXPECT_EQ(*released, 1);
  EXPECT_THROW(task.require_consumable(attempt), std::logic_error);
}

TEST(PreparedResourceCache, CompletionCannotRestoreReplacedVersionOrRejectedAttempt) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-version-attempt-revocation");
  PreparedResourceCache cache;
  auto executor = prepare_cpu_executor(cache, lane);
  auto released = std::make_shared<int>(0);
  auto old = prepare_buffer(cache, lane, 92, 8, released);
  auto attempt = cache.begin_attempt();
  auto task = executor->submit_for(cache, attempt, executor, 0, "leased-version-write", 1,
                                   KOKKOS_LAMBDA(std::int64_t){}, old);
  auto replacement = prepare_buffer(cache, lane, 92, 16, released);
  EXPECT_TRUE(attempt.visible());  // Replacement revokes only the touched resource.
  task.wait();
  EXPECT_EQ(task.status(), ResourceTask::Status::cancelled);
  EXPECT_THROW(task.require_consumable(attempt), std::logic_error);

  auto good = executor->submit_for(cache, attempt, executor, 0, "leased-good-write", 1,
                                   KOKKOS_LAMBDA(std::int64_t){}, replacement);
  good.wait();
  EXPECT_NO_THROW(good.require_consumable(attempt));
  auto next_attempt = cache.begin_attempt();
  EXPECT_GT(next_attempt.ordinal(), attempt.ordinal());
  EXPECT_FALSE(attempt.visible());
  EXPECT_EQ(good.status(), ResourceTask::Status::cancelled);
  EXPECT_THROW(good.require_consumable(attempt), std::logic_error);
  EXPECT_THROW(good.require_consumable(next_attempt), std::logic_error);
}

TEST(PreparedResourceCache, InvalidSubmissionCannotWriteOrLeaveUndrainableOwners) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-invalid-submission");
  PreparedResourceCache cache, other;
  auto executor = prepare_cpu_executor(cache, lane);
  auto released = std::make_shared<int>(0);
  auto buffer = prepare_buffer(cache, lane, 92, 8, released);
  auto attempt = cache.begin_attempt();
  auto foreign = other.begin_attempt();
  double* output = buffer->values.data();
  const auto kernel = KOKKOS_LAMBDA(std::int64_t i) {
    output[i] = -99;
  };
  EXPECT_THROW(
      (void)executor->submit_for(cache, foreign, executor, 0, "foreign", 8, kernel, buffer),
      std::logic_error);
  EXPECT_THROW(
      (void)executor->submit_for(cache, attempt, executor, 0, "negative", -1, kernel, buffer),
      std::invalid_argument);
  auto replacement = prepare_buffer(cache, lane, 92, 16, released);
  EXPECT_THROW(
      (void)executor->submit_for(cache, attempt, executor, 0, "revoked", 8, kernel, buffer),
      std::logic_error);
  EXPECT_DOUBLE_EQ(buffer->values(7), 17.0);
  EXPECT_DOUBLE_EQ(replacement->values(15), 17.0);
  EXPECT_TRUE(cache.quiescent());
}

TEST(PreparedResourceCache, DestructionDrainsAnExternallyRetainedNativeTask) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-cache-destructor-drain");
  auto released = std::make_shared<int>(0);
  ResourceTask task;
  runtime::program::PreparedResourceAttempt attempt;
  {
    PreparedResourceCache cache;
    auto executor = prepare_cpu_executor(cache, lane);
    auto buffer = prepare_buffer(cache, lane, 92, 8, released);
    attempt = cache.begin_attempt();
    double* output = buffer->values.data();
    task = executor->submit_for(
        cache, attempt, executor, 0, "destructor-drain", 8,
        KOKKOS_LAMBDA(std::int64_t i) { output[i] = 5; }, buffer);
  }
  EXPECT_TRUE(task.physically_complete());
  EXPECT_EQ(task.status(), ResourceTask::Status::cancelled);
  EXPECT_EQ(task.retained_resource_count(), 0u);
  EXPECT_EQ(*released, 1);
  EXPECT_THROW(task.require_consumable(attempt), std::logic_error);
}

TEST(PreparedResourceCache, LegacyRawCaptureLaunchCompletesBeforeReturning) {
  auto executor = CpuExecutor::prepare_synchronous(8);
  Kokkos::View<double*, Kokkos::HostSpace> output("legacy-raw-capture", 8);
  double* raw = output.data();
  executor.launch_for(
      0, "unowned-launch-completes", 8, KOKKOS_LAMBDA(std::int64_t i) { raw[i] = 3.0 * i; });
  // No fence is inserted by this caller before reading/releasing captured storage.
  EXPECT_DOUBLE_EQ(output(7), 21.0);
}

TEST(PreparedResourceCache, LifetimeHookSurvivesMoveAndExpiresWithNativeOwners) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-weak-native-lifetime-hook");
  std::function<void(PreparedResourceAction)> hook;
  ResourceTask task;
  runtime::program::PreparedResourceAttempt attempt;
  {
    PreparedResourceCache original;
    auto executor = prepare_cpu_executor(original, lane);
    attempt = original.begin_attempt(lane);
    task = executor->submit_for(original, attempt, executor, 0, "moved-cache", 1,
                                KOKKOS_LAMBDA(std::int64_t){});
    hook = original.lifetime_callback();
    PreparedResourceCache moved(std::move(original));
    EXPECT_NO_THROW(hook(PreparedResourceAction::finish));
    EXPECT_TRUE(task.physically_complete());
    EXPECT_NO_THROW(task.require_consumable(attempt));
    hook(PreparedResourceAction::reject);
    EXPECT_FALSE(attempt.visible());
    EXPECT_THROW(task.require_consumable(attempt), std::logic_error);
    const auto next = moved.begin_attempt(lane);
    EXPECT_GT(next.ordinal(), attempt.ordinal());
  }
  EXPECT_NO_THROW(hook(PreparedResourceAction::finish));
  EXPECT_NO_THROW(hook(PreparedResourceAction::reject));
}

TEST(PreparedResourceCache, DroppingTaskHandleCannotHideCancellationFromStepCompletion) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-lost-handle-failure");
  PreparedResourceCache cache;
  auto executor = prepare_cpu_executor(cache, lane);
  const auto attempt = cache.begin_attempt(lane);
  {
    auto task = executor->submit_for(cache, attempt, executor, 0, "dropped-cancelled", 1,
                                     KOKKOS_LAMBDA(std::int64_t){});
    task.request_cancel();
  }
  EXPECT_THROW(cache.lifetime_callback()(PreparedResourceAction::finish), std::runtime_error);
  EXPECT_TRUE(cache.quiescent());
  EXPECT_NO_THROW(cache.lifetime_callback()(PreparedResourceAction::reject));
  const auto retry = cache.begin_attempt(lane);
  EXPECT_GT(retry.ordinal(), attempt.ordinal());
  EXPECT_NO_THROW(cache.finish_attempt());
}

TEST(PreparedResourceCache, ForeignBufferCannotEscapeItsOwnersDrainRegistry) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-foreign-buffer-owner");
  PreparedResourceCache cache, other;
  auto executor = prepare_cpu_executor(cache, lane);
  auto buffer = prepare_buffer(other, lane, 92, 8, std::make_shared<int>(0));
  const auto attempt = cache.begin_attempt(lane);
  double* raw = buffer->values.data();
  EXPECT_THROW((void)executor->submit_for(
                   cache, attempt, executor, 0, "foreign-buffer", 8,
                   KOKKOS_LAMBDA(std::int64_t i) { raw[i] = -2; }, buffer),
               std::logic_error);
  EXPECT_DOUBLE_EQ(buffer->values(0), 17);
  EXPECT_TRUE(cache.quiescent());
}

TEST(PreparedResourceCache, DestructionDrainsRegistryWithoutAnyExternalTaskHandle) {
  comm_init();
  auto lane = ExecutionLane::world("prepared-orphan-task-destructor");
  auto released = std::make_shared<int>(0);
  runtime::program::PreparedResourceAttempt attempt;
  {
    PreparedResourceCache cache;
    auto executor = prepare_cpu_executor(cache, lane);
    auto buffer = prepare_buffer(cache, lane, 92, 8, released);
    attempt = cache.begin_attempt(lane);
    double* raw = buffer->values.data();
    (void)executor->submit_for(
        cache, attempt, executor, 0, "registry-only-owner", 8,
        KOKKOS_LAMBDA(std::int64_t i) { raw[i] = 19; }, buffer);
    buffer = {};
    executor = {};
    cache.clear();
    EXPECT_FALSE(cache.quiescent());
    EXPECT_EQ(*released, 0);
  }
  EXPECT_EQ(*released, 1);
  EXPECT_FALSE(attempt.visible());
}
