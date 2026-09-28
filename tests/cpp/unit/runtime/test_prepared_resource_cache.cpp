#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>

using namespace pops;
using runtime::program::PreparedResourceCache;

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
