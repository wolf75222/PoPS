/// Exact ownership migration with rank-local pre-collective faults and candidate-only publication.
#include <gtest/gtest.h>
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/parallel/region_transfer.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace {
constexpr int Dim = pops::kNativeDimension;
using Field = pops::MultiFab<Dim>;
using Box = pops::Box<Dim>;
using Index = pops::Index<Dim>;
using Runtime = pops::runtime::amr::AmrRuntime<Dim>;
namespace transport = pops::mesh::parallel;

pops::Extent<Dim> extent(int n) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = n;
  return result;
}
Index owner(int rank) {
  Index result{};
  result[0] = rank;
  return result;
}

// Global patch and component encoding is injective and exactly representable in binary64.
pops::Real value(std::size_t patch, int component) {
  return pops::Real(17 * patch + 3 * component + 1) / 8;
}

std::vector<std::vector<pops::Real>> snapshot(const Field& field) {
  std::vector<std::vector<pops::Real>> result;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    std::vector<pops::Real> values(host.size());
    for (std::size_t i = 0; i < host.size(); ++i)
      values[i] = host(i);
    result.push_back(std::move(values));
  }
  return result;
}

void prove_transitions() {
  const int ranks = pops::n_ranks(), rank = pops::my_rank();
  ASSERT_TRUE(ranks == 2 || ranks == 4);
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.m7.transition.invariants");
  auto rank_extent = extent(1);
  rank_extent[0] = ranks;
  const pops::mesh::RankSpace<Dim> rank_space(Index{}, rank_extent);
  std::vector<Box> boxes;
  for (int patch = 0; patch < 2 * ranks; ++patch) {
    Index coordinate{};
    coordinate[0] = patch - 3;
    boxes.emplace_back(coordinate, coordinate);
  }
  const pops::mesh::BoxArray<Dim> patches(boxes);
  const pops::mesh::BoxArrayValidationBudget layout_budget{32, 512};
  const pops::parallel::LoadBalancePreparationBudget balance_budget{
      32, 4, std::numeric_limits<std::int64_t>::max()};
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(
      patches, rank_space, std::vector<Index>(patches.size(), owner(0)));
  auto authority = std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.m7.ownership",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
  pops::amr::hierarchy::LevelLayout<Dim> layout(0, patches.bounding_box(), patches, distribution,
                                                pops::amr::RefinementRatio<Dim>{}, layout_budget);
  Field state(patches, distribution, owner(rank), 2, extent(1));
  state.set_val(pops::Real(-99));
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    auto& fab = state.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    // Every patch is one cell: the center offset is the sum of axis strides.
    std::size_t center = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      center += stride;
      stride *= 3;
    }
    for (int c = 0; c < 2; ++c)
      host(center + c * stride) = value(state.global_index(local), c);
    fab.copy_from_host(host);
  }
  Runtime runtime(
      pops::amr::hierarchy::AmrHierarchy<Dim>::from_coarse(layout, std::move(state), {1, 64}),
      authority, "test.m7.transition.spatial");
  EXPECT_EQ(pops::all_reduce_sum(runtime.hierarchy().state(0).local_size() == 0 ? 1L : 0L, lane),
            ranks - 1);
  const auto accepted = snapshot(runtime.hierarchy().state(0));
  const auto accepted_contract = runtime.hierarchy().spatial_contract();
  const auto accepted_epoch = runtime.topology_epoch();
  const auto accepted_generation = runtime.materialization_generation();

  std::vector<std::int64_t> weights(patches.size(), 1);
  if (rank == ranks - 1)
    weights.back() = -1;
  bool rejected = false;
  try {
    (void)authority->prepare(patches, rank_space, balance_budget, weights, lane);
  } catch (const std::invalid_argument& error) {
    rejected =
        std::string(error.what()).find("prepared load-balance source failed") != std::string::npos;
  }
  EXPECT_EQ(pops::all_reduce_sum(rejected ? 1L : 0L, lane), ranks);
  weights.back() = 1;
  const auto healthy = authority->prepare(patches, rank_space, balance_budget, weights, lane);
  EXPECT_EQ(healthy.plan().total_weight(), static_cast<std::int64_t>(patches.size()));

  std::vector<pops::ResourceEstimate> estimates(patches.size());
  for (auto& estimate : estimates) {
    estimate.topology_epoch = accepted_epoch;
    estimate.materialization_generation = accepted_generation;
    estimate.samples = 1;
    estimate.cell_updates = 1;
    estimate.compute_nanoseconds = 1000000;
    estimate.memory_bytes = 16;
    estimate.resident_bytes = 16;
  }
  pops::RebalancePolicy policy;
  policy.minimum_improvement_ppm = 0;
  policy.amortization_steps = 100;
  policy.per_patch_migration_latency_nanoseconds = 0;
  auto stale = estimates;
  if (rank == ranks - 1)
    ++stale.back().materialization_generation;
  rejected = false;
  try {
    (void)runtime.prepare_rebalance(0, stale, balance_budget, policy, lane);
  } catch (const std::invalid_argument& error) {
    rejected =
        std::string(error.what()).find("prepared rebalance request failed") != std::string::npos;
  }
  EXPECT_EQ(pops::all_reduce_sum(rejected ? 1L : 0L, lane), ranks);
  auto decision = runtime.prepare_rebalance(0, estimates, balance_budget, policy, lane);
  ASSERT_TRUE(decision.accepted);
  Field candidate(patches, decision.proposed.plan().distribution(), owner(rank), 2, extent(1));
  candidate.set_val(pops::Real(-777));
  const auto untouched_candidate = snapshot(candidate);
  std::vector<transport::RegionTransferJob<Dim>> jobs;
  for (std::size_t patch = 0; patch < patches.size(); ++patch)
    jobs.push_back({patch, patch, distribution.owner(patch), candidate.distribution().owner(patch),
                    patches[patch], patches[patch]});
  const auto elements = patches.size() * 2;
  transport::RegionTransport<Dim> exchange(transport::RegionTransferPlan<Dim>{
      rank_space,
      owner(rank),
      2,
      jobs,
      {jobs.size(), static_cast<std::size_t>(ranks), elements, elements, elements}});
  exchange.prepare_collectively(lane);
  const auto destination = [&](const auto& job) {
    return candidate.fab_global(job.destination_patch).view();
  };
  for (int fault : {1, 2}) {
    std::vector<int> accesses(patches.size(), 0);
    rejected = false;
    try {
      exchange.execute(
          [&](const auto& job) -> pops::FieldView<const pops::Real, Dim> {
            // First access is binding validation; second is native packing before MPI posts.
            if (rank == 0 && ++accesses[job.source_patch] == fault)
              throw std::runtime_error("injected native transition source failure");
            return std::as_const(runtime.hierarchy().state(0)).fab_global(job.source_patch).view();
          },
          destination);
    } catch (const std::exception& error) {
      const std::string expected =
          fault == 1 ? "binding failed collectively" : "packing failed collectively";
      rejected = std::string(error.what()).find(expected) != std::string::npos;
    }
    EXPECT_EQ(pops::all_reduce_sum(rejected ? 1L : 0L, lane), ranks);
    EXPECT_FALSE(exchange.sealed());
    EXPECT_EQ(snapshot(candidate), untouched_candidate);
    EXPECT_EQ(snapshot(runtime.hierarchy().state(0)), accepted);
    EXPECT_EQ(runtime.hierarchy().spatial_contract(), accepted_contract);
  }
  exchange.execute(
      [&](const auto& job) -> pops::FieldView<const pops::Real, Dim> {
        return std::as_const(runtime.hierarchy().state(0)).fab_global(job.source_patch).view();
      },
      destination);
  Kokkos::fence();
  long local_cells = 0;
  long double local_sum = 0;
  for (std::size_t local = 0; local < candidate.local_size(); ++local) {
    const auto& fab = candidate.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    std::size_t center = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      center += stride;
      stride *= 3;
    }
    for (int c = 0; c < 2; ++c) {
      EXPECT_EQ(host(center + c * stride), value(candidate.global_index(local), c));
      local_sum += host(center + c * stride);
      for (std::size_t i = 0; i < stride; ++i)
        if (i != center)
          EXPECT_EQ(host(i + c * stride), pops::Real(-777));
    }
    ++local_cells;
  }
  EXPECT_EQ(local_cells, 2);
  EXPECT_EQ(pops::all_reduce_sum(local_cells, lane), static_cast<long>(patches.size()));
  long double expected_sum = 0;
  for (std::size_t patch = 0; patch < patches.size(); ++patch)
    for (int c = 0; c < 2; ++c)
      expected_sum += value(patch, c);
  EXPECT_EQ(pops::all_reduce_sum(static_cast<double>(local_sum), lane),
            static_cast<double>(expected_sum));
  auto invalid_decision = decision;
  ++invalid_decision.materialization_generation;
  EXPECT_THROW(runtime.apply_rebalance(0, std::move(invalid_decision), Field(candidate)),
               std::invalid_argument);
  EXPECT_EQ(snapshot(runtime.hierarchy().state(0)), accepted);
  EXPECT_EQ(runtime.hierarchy().spatial_contract(), accepted_contract);
  EXPECT_EQ(runtime.topology_epoch(), accepted_epoch);
  EXPECT_EQ(runtime.materialization_generation(), accepted_generation);
  const auto transferred_candidate = snapshot(candidate);
  runtime.apply_rebalance(0, std::move(decision), std::move(candidate));
  EXPECT_EQ(snapshot(runtime.hierarchy().state(0)), transferred_candidate);
  EXPECT_NE(runtime.hierarchy().spatial_contract(), accepted_contract);
  EXPECT_EQ(runtime.hierarchy().layout(0).distribution(), healthy.plan().distribution());
  EXPECT_GT(runtime.materialization_generation(), accepted_generation);
}

int run(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
      prove_transitions();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "M7 transition invariant rank %d: %s\n", pops::my_rank(), error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
  }
  pops::comm_finalize();
  return result;
}
}  // namespace

TEST(test_mpi_transition_invariants, NativePrecollectiveFailureRetryAndExactOwnershipPublication) {
  EXPECT_EQ(pops::test::RunTestBody(&run, "test_mpi_transition_invariants"), 0);
}
