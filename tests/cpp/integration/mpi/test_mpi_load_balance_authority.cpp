// Native MPI contract for prepared exact-ranked AMR ownership: deterministic weighted plans,
// collective fail-closed provenance, and a distributed prepared-regrid publication.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace regridding = pops::amr::regridding;
namespace tagging = pops::amr::tagging;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{128, 8'192};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{2, 8'192};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    128, 64, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::mesh::RankSpace<Dim> rank_space_for(int ranks) {
  pops::Extent<Dim> extent = filled_extent<Dim>(1);
  int remaining = ranks;
  int factor_slot = 0;
  for (int factor = 2; factor <= remaining / factor; ++factor) {
    while (remaining % factor == 0) {
      extent[factor_slot % Dim] *= factor;
      remaining /= factor;
      ++factor_slot;
    }
  }
  if (remaining > 1)
    extent[factor_slot % Dim] *= remaining;
  return {pops::Index<Dim>{}, extent};
}

template <int Dim>
pops::amr::RefinementRatio<Dim> ratio_two() {
  std::array<int, Dim> values{};
  values.fill(2);
  return pops::amr::RefinementRatio<Dim>(values);
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance_authority(
    std::string identity) {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", std::move(identity),
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::mesh::BoxArray<Dim> partitioned_boxes(const pops::mesh::RankSpace<Dim>& ranks) {
  pops::Extent<Dim> domain_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    domain_extent[axis] = 4 * ranks.extent()[axis];
  return pops::mesh::BoxArray<Dim>::from_domain(pops::Box<Dim>::from_extents(domain_extent),
                                                  filled_extent<Dim>(4));
}

template <int Dim>
regridding::RegridPreparationBudget regrid_budget() {
  return {.clustered_parent_layout = kLayoutBudget,
          .fine_layout = kLayoutBudget,
          .load_balance = kLoadBalanceBudget};
}

template <int Dim>
tagging::ClusterResult<Dim> clustered_parent_boxes(const hierarchy::LevelLayout<Dim>& parent) {
  tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[axis] = 1;
    options.max_box_size[axis] = 64;
  }
  options.budget = {128, 8'192, 1U << 20, 128, 1U << 22};
  const pops::mesh::BoxArray<Dim> boxes(parent.patches().boxes());
  tagging::ClusterResultIdentity<Dim> identity{
      "test.mpi-load-balance-authority.clustered-parent@1", parent.exact_identity(), options, {},
      boxes.boxes()};
  return {boxes, std::move(identity)};
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_partitioned_runtime(
    const pops::mesh::RankSpace<Dim>& ranks, const pops::Index<Dim>& local_rank,
    std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> authority) {
  const pops::mesh::BoxArray<Dim> patches = partitioned_boxes(ranks);
  const auto prepared = authority->prepare(patches, ranks, kLoadBalanceBudget);
  const auto& distribution = prepared.plan().distribution();
  const pops::Box<Dim> domain = patches.bounding_box();
  hierarchy::LevelLayout<Dim> coarse(0, domain, patches, distribution,
                                     pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  pops::MultiFab<Dim> state(patches, distribution, local_rank, 1, filled_extent<Dim>(1));
  state.set_val(pops::Real(1));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>::from_coarse(coarse, std::move(state), kHierarchyBudget),
      std::move(authority), "test.mpi-load-balance-authority/state-qid@1");
}

template <int Dim>
bool hierarchy_regrid_is_collective_and_owner_authenticated(
    const pops::mesh::RankSpace<Dim>& ranks, const pops::Index<Dim>& local_rank,
    const std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>>& authority) {
  auto runtime = make_partitioned_runtime(ranks, local_rank, authority);
  const auto stable_contract = runtime.spatial_contract();
  const auto stable_epoch = runtime.topology_epoch();

  auto rejected_prepared = runtime.prepare_regrid(
      0, ratio_two<Dim>(), clustered_parent_boxes(runtime.hierarchy().layout(0)), regrid_budget<Dim>());
  if (!rejected_prepared.fine_layout())
    return false;
  const auto& expected = rejected_prepared.fine_layout()->distribution();
  std::vector<pops::Index<Dim>> rotated;
  rotated.reserve(expected.owners().size());
  for (const pops::Index<Dim>& owner : expected.owners())
    rotated.push_back(ranks.coordinate((ranks.linear_rank(owner) + 1) % ranks.size()));
  pops::MultiFab<Dim> wrong(rejected_prepared.fine_layout()->patches(),
                             pops::mesh::Distribution<Dim>::partitioned(
                                 rejected_prepared.fine_layout()->patches(), ranks, std::move(rotated)),
                             local_rank, 1, filled_extent<Dim>(1));
  bool rejected = false;
  try {
    runtime.publish_regrid(0, std::move(rejected_prepared), std::move(wrong));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || runtime.spatial_contract() != stable_contract || runtime.topology_epoch() != stable_epoch)
    return false;

  auto accepted_prepared = runtime.prepare_regrid(
      0, ratio_two<Dim>(), clustered_parent_boxes(runtime.hierarchy().layout(0)), regrid_budget<Dim>());
  if (!accepted_prepared.fine_layout())
    return false;
  pops::MultiFab<Dim> accepted(accepted_prepared.fine_layout()->patches(),
                                accepted_prepared.fine_layout()->distribution(), local_rank, 1,
                                filled_extent<Dim>(1));
  accepted.set_val(pops::Real(2));
  const auto expected_owners = accepted_prepared.fine_layout()->distribution().owners();
  runtime.publish_regrid(0, std::move(accepted_prepared), std::move(accepted));
  return runtime.hierarchy().num_levels() == 2 &&
         runtime.hierarchy().layout(1).distribution().owners() == expected_owners &&
         runtime.topology_epoch() == stable_epoch + 1;
}

template <int Dim>
bool authority_decisions_are_exact_and_collective(const pops::mesh::RankSpace<Dim>& ranks,
                                                  const pops::Index<Dim>& local_rank,
                                                  int rank) {
  const pops::mesh::BoxArray<Dim> boxes = partitioned_boxes(ranks);
  std::vector<std::int64_t> weights(boxes.size(), 1);
  weights.front() = 17;
  const auto authority = load_balance_authority<Dim>("test.mpi.weighted-sfc.semantic-identity@1");
  const auto first = authority->prepare(boxes, ranks, kLoadBalanceBudget, weights);
  const auto second = authority->prepare(boxes, ranks, kLoadBalanceBudget, weights);
  if (first.plan().distribution() != second.plan().distribution() ||
      first.plan().distribution().owners().size() != boxes.size())
    return false;
  for (const pops::Index<Dim>& owner : first.plan().distribution().owners())
    if (!ranks.contains(owner))
      return false;

  std::vector<pops::ResourceEstimate> estimates(boxes.size());
  for (pops::ResourceEstimate& estimate : estimates) {
    estimate.topology_epoch = 10;
    estimate.materialization_generation = 20;
    estimate.samples = 1;
    estimate.cell_updates = 1;
    estimate.compute_nanoseconds = 1000;
    estimate.memory_bytes = 64;
    estimate.resident_bytes = 64;
  }
  pops::RebalancePolicy policy;
  policy.minimum_improvement_ppm = 0;
  policy.amortization_steps = 100;
  policy.migration_bandwidth_bytes_per_second = 1'000'000'000'000LL;
  policy.per_patch_migration_latency_nanoseconds = 0;
  const pops::mesh::Distribution<Dim> concentrated =
      pops::mesh::Distribution<Dim>::partitioned(
          boxes, ranks, std::vector<pops::Index<Dim>>(boxes.size(), ranks.coordinate(0)));
  const auto beneficial = authority->decide_rebalance(1, boxes, concentrated, 10, 20, estimates,
                                                       kLoadBalanceBudget, policy);
  if (!beneficial.accepted || beneficial.reason != pops::RebalanceReason::NetBenefit ||
      beneficial.moved_patches <= 0 || beneficial.exact_contract.empty())
    return false;
  const auto unchanged = authority->decide_rebalance(
      1, boxes, beneficial.proposed.plan().distribution(), 10, 20, estimates, kLoadBalanceBudget,
      policy);
  if (unchanged.accepted || unchanged.reason != pops::RebalanceReason::MappingUnchanged ||
      unchanged.moved_patches != 0 || unchanged.exact_contract.empty())
    return false;

  if (ranks.size() > 1) {
    bool rejected = false;
    try {
      const auto divergent = load_balance_authority<Dim>(
          rank == 1 ? "test.mpi.other-sfc.semantic-identity@1"
                    : "test.mpi.weighted-sfc.semantic-identity@1");
      (void)divergent->prepare(boxes, ranks, kLoadBalanceBudget, weights);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      return false;
  }
  (void)local_rank;
  return true;
}

int run_mpi_load_balance_authority(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int failure = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
      constexpr int Dim = pops::kNativeDimension;
      const auto ranks = rank_space_for<Dim>(pops::n_ranks());
      const auto local_rank = ranks.coordinate(static_cast<std::size_t>(pops::my_rank()));
      const auto hierarchy_authority =
          load_balance_authority<Dim>("test.mpi.hierarchy-sfc.semantic-identity@1");
      if (!authority_decisions_are_exact_and_collective<Dim>(ranks, local_rank, pops::my_rank()) ||
          !hierarchy_regrid_is_collective_and_owner_authenticated<Dim>(ranks, local_rank,
                                                                       hierarchy_authority))
        failure = 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact load-balance authority proof failed: %s\n", pops::my_rank(),
                   error.what());
      failure = 1;
    }
    failure = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(failure || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && failure == 0)
      std::printf("OK test_mpi_load_balance_authority np=%d dim=%d exact-ranked\n", pops::n_ranks(),
                  pops::kNativeDimension);
  }
  pops::comm_finalize();
  return failure;
}

}  // namespace

TEST(test_mpi_load_balance_authority, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_mpi_load_balance_authority, "test_mpi_load_balance_authority"),
      0);
}
