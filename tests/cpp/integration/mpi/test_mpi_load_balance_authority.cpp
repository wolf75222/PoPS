/// @file
/// @brief MPI proof for exact-rank ownership, tagging, regrid and migration authority.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/tagging/berger_rigoutsos.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/coupling/amr/amr_regrid_coupler.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace tagging = pops::amr::tagging;
namespace transfer = pops::amr::transfer;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{128, 8'192};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{2, 16'384};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    128, 128, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> rank_coordinate(int rank) {
  pops::Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
pops::mesh::RankSpace<Dim> mpi_rank_space(int ranks) {
  pops::Extent<Dim> extent = filled_extent<Dim>(1);
  extent[0] = ranks;
  return {pops::Index<Dim>{}, extent};
}

template <int Dim>
pops::amr::RefinementRatio<Dim> ratio_two() {
  std::array<int, Dim> values{};
  values.fill(2);
  return pops::amr::RefinementRatio<Dim>(values);
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.mpi-load-balance-authority.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_partitioned_runtime(
    int ranks, int rank,
    const std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>>& authority) {
  pops::Index<Dim> upper{};
  upper[0] = 4 * ranks - 1;
  for (int axis = 1; axis < Dim; ++axis)
    upper[axis] = 3;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, upper};
  std::vector<pops::Box<Dim>> patch_values;
  std::vector<pops::Index<Dim>> owners;
  patch_values.reserve(static_cast<std::size_t>(ranks));
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int owner = 0; owner < ranks; ++owner) {
    pops::Box<Dim> patch = domain;
    patch.lo[0] = 4 * owner;
    patch.hi[0] = patch.lo[0] + 3;
    patch_values.push_back(patch);
    owners.push_back(rank_coordinate<Dim>(owner));
  }
  const pops::mesh::BoxArray<Dim> patches(std::move(patch_values));
  const auto rank_space = mpi_rank_space<Dim>(ranks);
  const auto distribution =
      pops::mesh::Distribution<Dim>::partitioned(patches, rank_space, std::move(owners));
  hierarchy::LevelLayout<Dim> coarse(0, domain, patches, distribution,
                                     pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  pops::MultiFab<Dim> state(patches, distribution, rank_coordinate<Dim>(rank), 1,
                            filled_extent<Dim>(2));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>::from_coarse(coarse, std::move(state), kHierarchyBudget),
      authority, "test.mpi-load-balance-authority.spatial");
}

template <int Dim>
POPS_HD pops::Real affine_parent(const pops::Index<Dim>& cell,
                                 const transfer::IndexMapping<Dim>& mapping) {
  pops::Real value = pops::Real(2.5);
  for (int axis = 0; axis < Dim; ++axis)
    value += pops::Real(0.125 * (axis + 1)) * pops::Real(cell[axis] - mapping.coarse_origin[axis]);
  return value;
}

template <int Dim>
POPS_HD pops::Real affine_child(const pops::Index<Dim>& cell,
                                const pops::amr::RefinementRatio<Dim>& ratio,
                                const transfer::IndexMapping<Dim>& mapping) {
  pops::Real value = pops::Real(2.5);
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real relative = pops::Real(cell[axis] - mapping.fine_origin[axis]);
    const pops::Real parent_coordinate =
        (relative + pops::Real(0.5)) / pops::Real(ratio[axis]) - pops::Real(0.5);
    value += pops::Real(0.125 * (axis + 1)) * parent_coordinate;
  }
  return value;
}

template <int Dim>
struct FillAffineParent {
  pops::FieldView<pops::Real, Dim> values{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    values(cell) = affine_parent(cell, mapping);
  }
};

template <int Dim>
std::size_t host_offset(const pops::Fab<Dim>& fab, const pops::Index<Dim>& cell,
                        int component = 0) {
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(cell[axis] - fab.grown_box().lo[axis]) * stride;
    stride *= static_cast<std::size_t>(fab.grown_box().length(axis));
  }
  return static_cast<std::size_t>(component) * stride + offset;
}

template <int Dim, class Function>
void for_each_host_index(const pops::Box<Dim>& box, Function&& function) {
  if (box.empty())
    return;
  pops::Index<Dim> cell = box.lo;
  while (true) {
    function(cell);
    int axis = 0;
    for (; axis < Dim; ++axis) {
      if (cell[axis] < box.hi[axis]) {
        ++cell[axis];
        break;
      }
      cell[axis] = box.lo[axis];
    }
    if (axis == Dim)
      return;
  }
}

template <int Dim>
bool exact_field_equal(const pops::MultiFab<Dim>& left, const pops::MultiFab<Dim>& right) {
  if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
      left.local_rank() != right.local_rank() || left.ncomp() != right.ncomp() ||
      left.ghosts() != right.ghosts() ||
      left.local_global_indices() != right.local_global_indices())
    return false;
  for (std::size_t local = 0; local < left.local_size(); ++local) {
    const auto& left_fab = left.fab(local);
    const auto& right_fab = right.fab(local);
    auto left_host = left_fab.create_host_mirror();
    auto right_host = right_fab.create_host_mirror();
    left_fab.copy_to_host(left_host);
    right_fab.copy_to_host(right_host);
    if (left_host.size() != right_host.size())
      return false;
    for (std::size_t entry = 0; entry < left_host.size(); ++entry)
      if (left_host(entry) != right_host(entry))
        return false;
  }
  return true;
}

template <int Dim>
tagging::ClusterOptions<Dim> cluster_options() {
  tagging::ClusterOptions<Dim> result;
  result.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    result.min_box_size[static_cast<std::size_t>(axis)] = 1;
    result.max_box_size[static_cast<std::size_t>(axis)] = 16;
  }
  result.budget = {128, 4'096, 1'000'000, 4'096, 1U << 22};
  return result;
}

template <int Dim>
struct GatheredTagShards {
  std::vector<tagging::TagMask<Dim>> shards{};
  pops::Index<Dim> tagged{};
  int author = -1;
  bool remote_gather_observed = false;
};

template <int Dim>
GatheredTagShards<Dim> gather_rank_local_tag_shards(const hierarchy::LevelLayout<Dim>& level,
                                                    int ranks, int rank) {
  const int author = ranks - 1;
  pops::Index<Dim> tagged = level.patches()[static_cast<std::size_t>(author)].lo;
  ++tagged[0];
  for (int axis = 1; axis < Dim; ++axis)
    ++tagged[axis];
  const std::size_t cells = static_cast<std::size_t>(level.patches()[0].numPts());
  const tagging::TagMaskBudget budget{
      static_cast<std::size_t>(ranks), 1, cells, cells, cells, 1U << 22};
  tagging::TagMask<Dim> local(level, rank_coordinate<Dim>(rank), budget);
  if (rank == author)
    local.set(tagged);

  std::string payload;
  if (local.count() == 1) {
    payload = "tag";
    for (int axis = 0; axis < Dim; ++axis)
      payload += ":" + std::to_string(tagged[axis]);
  }
  const auto gathered = pops::WorldCommunicator::world().allgather_bytes(payload);
  const std::string authored_payload = [&] {
    std::string value = "tag";
    for (int axis = 0; axis < Dim; ++axis)
      value += ":" + std::to_string(tagged[axis]);
    return value;
  }();
  if (gathered.size() != static_cast<std::size_t>(ranks))
    throw std::runtime_error("rank-local tag gather returned another rank cardinality");
  for (int source = 0; source < ranks; ++source) {
    const std::string& expected = source == author ? authored_payload : std::string{};
    if (gathered[static_cast<std::size_t>(source)] != expected)
      throw std::runtime_error("rank-local tag gather changed its exact source payload");
  }

  std::vector<tagging::TagMask<Dim>> shards;
  shards.reserve(static_cast<std::size_t>(ranks));
  for (int source = 0; source < ranks; ++source) {
    shards.emplace_back(level, rank_coordinate<Dim>(source), budget);
    if (!gathered[static_cast<std::size_t>(source)].empty())
      shards.back().set(tagged);
  }
  return {std::move(shards), tagged, author,
          rank != author && gathered[static_cast<std::size_t>(author)] == authored_payload};
}

template <int Dim>
bool prove_tagging_prolongation_and_owner_rollback(int ranks, int rank) {
  if (ranks < 2)
    return false;
  const auto authority = load_balance<Dim>();
  auto runtime = make_partitioned_runtime<Dim>(ranks, rank, authority);
  tagging::BergerRigoutsosProvider<Dim> cluster_provider;
  pops::coupling::amr::AmrRegridCoupler<Dim> coupler(runtime, cluster_provider);
  const auto ratio = ratio_two<Dim>();
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = kLayoutBudget,
      .fine_layout = kLayoutBudget,
      .load_balance = kLoadBalanceBudget,
  };
  const auto gathered = gather_rank_local_tag_shards(runtime.hierarchy().layout(0), ranks, rank);
  if (pops::all_reduce_sum(gathered.remote_gather_observed ? 1L : 0L) != ranks - 1)
    return false;
  auto prepared = coupler.prepare(0, ratio, gathered.shards, cluster_options<Dim>(), budget);
  if (prepared.removes_fine_level() || !prepared.fine_layout())
    return false;
  const auto& clustering = prepared.clustering();
  if (clustering.canonical_shards.size() != static_cast<std::size_t>(ranks) ||
      clustering.boxes !=
          std::vector<pops::Box<Dim>>{pops::Box<Dim>{gathered.tagged, gathered.tagged}} ||
      prepared.fine_layout()->distribution().owners() !=
          std::vector<pops::Index<Dim>>{rank_coordinate<Dim>(0)})
    return false;
  std::size_t materialized_tags = 0;
  for (const auto& patch :
       clustering.canonical_shards[static_cast<std::size_t>(gathered.author)].patches)
    materialized_tags +=
        static_cast<std::size_t>(std::count(patch.tags.begin(), patch.tags.end(), std::uint8_t{1}));
  if (materialized_tags != 1 ||
      clustering.canonical_shards[static_cast<std::size_t>(gathered.author)].local_rank !=
          rank_coordinate<Dim>(gathered.author))
    return false;

  const pops::Index<Dim> local_rank = rank_coordinate<Dim>(rank);
  pops::MultiFab<Dim> child(prepared.fine_layout()->patches(),
                            prepared.fine_layout()->distribution(), local_rank, 1,
                            filled_extent<Dim>(2));
  coupler.publish(std::move(prepared), std::move(child));
  if (runtime.hierarchy().num_levels() != 2)
    return false;

  auto& coarse = runtime.hierarchy().state(0);
  auto& fine = runtime.hierarchy().state(1);
  const transfer::IndexMapping<Dim> mapping{runtime.hierarchy().layout(0).domain().lo,
                                            runtime.hierarchy().layout(1).domain().lo};
  for (std::size_t local = 0; local < coarse.local_size(); ++local)
    pops::for_each_cell(coarse.fab(local).box(),
                        FillAffineParent<Dim>{coarse.fab(local).view(), mapping});
  constexpr pops::Real kGhostSentinel = pops::Real(-777);
  fine.set_val(kGhostSentinel);
  pops::runtime::amr::AmrGhostFillPreparation<Dim> request{};
  request.fine_level = 1;
  request.coarse_domain = runtime.hierarchy().layout(0).domain();
  request.fine_domain = runtime.hierarchy().layout(1).domain();
  request.ratio = ratio;
  request.topology = pops::BoundaryTopology<Dim>::physical();
  request.topology_generation = 7;
  request.materialization_generation = 11;
  request.field_identity = "test.mpi-load-balance-authority.remote-fine-state";
  request.budget = {{128, 4'096, 16'384, 16'384, 128, 1U << 20, 1U << 20, 1U << 20},
                    {{128, 8'192}, 16'384, 16'384, 256, 128, 1U << 20, 1U << 20, 1U << 20}};
  const pops::ExecutionLane lane = pops::ExecutionLane::world();
  const auto ghost_fill = pops::runtime::amr::prepare_amr_ghost_fill(coarse, fine, request, lane);
  const long remote_schedule_ranks =
      pops::all_reduce_sum(ghost_fill.has_remote_parent_jobs() ? 1L : 0L);
  pops::runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  ghost_fill(fine, point);

  std::size_t local_materialized = 0;
  pops::Real local_error = pops::Real(0);
  for (std::size_t local = 0; local < fine.local_size(); ++local) {
    const auto& fab = fine.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for_each_host_index(
        fab.grown_box().intersect(request.fine_domain), [&](const pops::Index<Dim>& cell) {
          if (fab.box().contains(cell))
            return;
          const pops::Real value = host(host_offset(fab, cell));
          if (value == kGhostSentinel)
            return;
          ++local_materialized;
          const pops::Real difference = value - affine_child(cell, ratio, mapping);
          local_error =
              std::max(local_error, difference < pops::Real(0) ? -difference : difference);
        });
  }
  const bool remote_materialized =
      rank == 0 && ghost_fill.has_remote_parent_jobs() && local_materialized > 0;
  if (remote_schedule_ranks < 2 || pops::all_reduce_sum(remote_materialized ? 1L : 0L) != 1 ||
      pops::all_reduce_max(static_cast<double>(local_error)) >= 2e-13)
    return false;

  constexpr pops::Real kRollbackSentinel = pops::Real(913.25);
  fine.set_val(kRollbackSentinel);
  const pops::MultiFab<Dim> stable_fine(fine);
  const std::string stable_contract(runtime.spatial_contract());
  const auto stable_spatial = runtime.hierarchy().spatial_contract();
  auto replacement = coupler.prepare(0, ratio, gathered.shards, cluster_options<Dim>(), budget);
  if (!replacement.fine_layout())
    return false;
  const auto& expected = replacement.fine_layout()->distribution();
  std::vector<pops::Index<Dim>> rotated = expected.owners();
  for (auto& owner : rotated) {
    const std::size_t linear = expected.rank_space().linear_rank(owner);
    owner = expected.rank_space().coordinate((linear + 1) % expected.rank_space().size());
  }
  const auto wrong_distribution = pops::mesh::Distribution<Dim>::partitioned(
      replacement.fine_layout()->patches(), expected.rank_space(), std::move(rotated));
  pops::MultiFab<Dim> wrong(replacement.fine_layout()->patches(), wrong_distribution, local_rank, 1,
                            filled_extent<Dim>(2));
  bool rejected = false;
  try {
    coupler.publish(std::move(replacement), std::move(wrong));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  return rejected && runtime.hierarchy().num_levels() == 2 &&
         runtime.spatial_contract() == stable_contract &&
         runtime.hierarchy().spatial_contract() == stable_spatial &&
         exact_field_equal(runtime.hierarchy().state(1), stable_fine);
}

template <int Dim>
pops::mesh::BoxArray<Dim> load_balance_boxes(int count) {
  std::vector<pops::Box<Dim>> boxes;
  boxes.reserve(static_cast<std::size_t>(count));
  for (int patch = 0; patch < count; ++patch) {
    pops::Index<Dim> coordinate{};
    coordinate[0] = patch;
    boxes.emplace_back(coordinate, coordinate);
  }
  return pops::mesh::BoxArray<Dim>(std::move(boxes));
}

template <int Dim>
bool plan_retains_explicit_mapping(const pops::parallel::OwnershipPlan<Dim>& plan,
                                   const pops::mesh::RankSpace<Dim>& rank_space,
                                   const std::vector<std::int64_t>& weights) {
  if (plan.linear_owners().size() != weights.size() ||
      plan.distribution().owners().size() != weights.size() ||
      plan.linear_rank_loads().size() != rank_space.size())
    return false;
  std::vector<std::int64_t> loads(rank_space.size(), 0);
  for (std::size_t patch = 0; patch < weights.size(); ++patch) {
    const std::size_t owner = plan.linear_owners()[patch];
    if (owner >= rank_space.size() ||
        plan.distribution().owners()[patch] != rank_space.coordinate(owner))
      return false;
    loads[owner] += weights[patch];
  }
  return plan.weights() == weights && plan.linear_rank_loads() == loads;
}

template <int Dim>
bool prove_stable_collective_mapping(int ranks, int rank) {
  const int box_count = std::max(4, 4 * ranks);
  const auto boxes = load_balance_boxes<Dim>(box_count);
  const auto rank_space = mpi_rank_space<Dim>(ranks);
  const auto authority = pops::prepare_load_balance_authority<Dim>(
      "space_filling_curve", "test.mpi-load-balance-authority.weighted-sfc",
      pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}});
  const std::vector<std::int64_t> uniform_weights(static_cast<std::size_t>(box_count), 1);
  auto skewed_weights = uniform_weights;
  skewed_weights.front() = 17;
  const auto uniform = authority.prepare(boxes, rank_space, kLoadBalanceBudget, uniform_weights);
  const auto skewed = authority.prepare(boxes, rank_space, kLoadBalanceBudget, skewed_weights);
  const auto skewed_again =
      authority.prepare(boxes, rank_space, kLoadBalanceBudget, skewed_weights);
  std::vector<std::size_t> expected_uniform_owners(static_cast<std::size_t>(box_count));
  for (int patch = 0; patch < box_count; ++patch)
    expected_uniform_owners[static_cast<std::size_t>(patch)] = static_cast<std::size_t>(patch / 4);
  const std::vector<std::int64_t> expected_uniform_loads(static_cast<std::size_t>(ranks), 4);
  if (!plan_retains_explicit_mapping(uniform.plan(), rank_space, uniform_weights) ||
      !plan_retains_explicit_mapping(skewed.plan(), rank_space, skewed_weights) ||
      uniform.plan().linear_owners() != expected_uniform_owners ||
      uniform.plan().linear_rank_loads() != expected_uniform_loads ||
      uniform.plan().linear_owners() == skewed.plan().linear_owners() ||
      uniform.plan().linear_rank_loads() == skewed.plan().linear_rank_loads() ||
      skewed.plan().linear_owners() != skewed_again.plan().linear_owners() ||
      skewed.plan().linear_rank_loads() != skewed_again.plan().linear_rank_loads() ||
      skewed.plan().distribution() != skewed_again.plan().distribution() ||
      skewed.exact_contract() != skewed_again.exact_contract())
    return false;

  if (ranks > 1) {
    auto divergent = skewed_weights;
    if (rank == 1)
      ++divergent.back();
    bool rejected = false;
    try {
      (void)authority.prepare(boxes, rank_space, kLoadBalanceBudget, divergent);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      return false;
  }

  constexpr std::uint64_t topology_epoch = 10;
  constexpr std::uint64_t materialization_generation = 20;
  std::vector<pops::ResourceEstimate> estimates(static_cast<std::size_t>(box_count));
  for (auto& estimate : estimates) {
    estimate.topology_epoch = topology_epoch;
    estimate.materialization_generation = materialization_generation;
    estimate.samples = 1;
    estimate.cell_updates = 1;
    estimate.compute_nanoseconds = 1'000;
    estimate.memory_bytes = 64;
    estimate.resident_bytes = 64;
  }
  pops::RebalancePolicy policy;
  policy.minimum_improvement_ppm = 0;
  policy.amortization_steps = 100;
  policy.migration_bandwidth_bytes_per_second = 1'000'000'000'000LL;
  policy.per_patch_migration_latency_nanoseconds = 0;
  const pops::mesh::Distribution<Dim> concentrated = pops::mesh::Distribution<Dim>::partitioned(
      boxes, rank_space,
      std::vector<pops::Index<Dim>>(static_cast<std::size_t>(box_count), rank_space.coordinate(0)));
  const auto beneficial =
      authority.decide_rebalance(1, boxes, concentrated, topology_epoch, materialization_generation,
                                 estimates, kLoadBalanceBudget, policy);
  const std::vector<std::int64_t> measured_weights(static_cast<std::size_t>(box_count), 1'000);
  if (!beneficial.accepted || beneficial.reason != pops::RebalanceReason::NetBenefit ||
      beneficial.moved_patches <= 0 || beneficial.proposed.plan().distribution() == concentrated ||
      !plan_retains_explicit_mapping(beneficial.proposed.plan(), rank_space, measured_weights) ||
      beneficial.proposed.plan().linear_owners() != expected_uniform_owners ||
      beneficial.proposed.plan().linear_rank_loads() !=
          std::vector<std::int64_t>(static_cast<std::size_t>(ranks), 4'000) ||
      beneficial.current_max_nanoseconds_per_step != box_count * 1'000 ||
      beneficial.proposed_max_nanoseconds_per_step != 4'000 ||
      beneficial.predicted_net_speedup <= 1.0 ||
      beneficial.exact_contract != pops::detail::exact_rebalance_decision(beneficial))
    return false;
  const auto unchanged = authority.decide_rebalance(
      1, boxes, beneficial.proposed.plan().distribution(), topology_epoch,
      materialization_generation, estimates, kLoadBalanceBudget, policy);
  return !unchanged.accepted && unchanged.reason == pops::RebalanceReason::MappingUnchanged &&
         unchanged.moved_patches == 0 &&
         unchanged.exact_contract == pops::detail::exact_rebalance_decision(unchanged);
}

int run_mpi_load_balance_authority(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
      constexpr int Dim = pops::kNativeDimension;
      const bool stable = prove_stable_collective_mapping<Dim>(pops::n_ranks(), pops::my_rank());
      const bool hierarchy_ok =
          prove_tagging_prolongation_and_owner_rollback<Dim>(pops::n_ranks(), pops::my_rank());
      if (!stable || !hierarchy_ok)
        result = 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact load-balance authority proof failed: %s\n",
                   pops::my_rank(), error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_load_balance_authority np=%d dim=%d exact-ranked-authority\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_load_balance_authority, ExactRankedAuthorityIsCollectiveAndTransactional) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_mpi_load_balance_authority, "test_mpi_load_balance_authority"),
      0);
}
