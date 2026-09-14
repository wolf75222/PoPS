#include <gtest/gtest.h>
#include <pops/runtime/program/prepared_resource_cache.hpp>

using namespace pops;
using runtime::program::PreparedResourceCache;

namespace {
struct Resource {
  int shape;
  int evaluated = 0;
  Resource(int requested, int& constructions) : shape(requested) { ++constructions; }
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
