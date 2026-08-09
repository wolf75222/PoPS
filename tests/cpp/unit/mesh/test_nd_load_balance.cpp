#include <gtest/gtest.h>

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/parallel/load_balance.hpp>
#include <pops/parallel/prepared_load_balance.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using pops::Box;
using pops::ExecutionLane;
using pops::Extent;
using pops::Index;
using pops::mesh::BoxArray;
using pops::mesh::RankSpace;
using pops::PreparedLoadBalanceAuthority;
using pops::PreparedLoadBalanceProvider;
using pops::PreparedLoadBalanceResult;
using pops::PreparedRebalanceDecision;
using pops::parallel::LoadBalancePreparationBudget;
using pops::parallel::LoadBalanceProvider;
using pops::parallel::LoadBalanceStrategy;

constexpr LoadBalancePreparationBudget budget() {
  return {128, 64, std::numeric_limits<std::int64_t>::max()};
}

template <int Dim>
Box<Dim> point_box(Index<Dim> coordinate) {
  return {coordinate, coordinate};
}

template <int Dim>
std::vector<std::size_t> reconstructed_linear_owners(
    const pops::mesh::Distribution<Dim>& distribution) {
  std::vector<std::size_t> result;
  result.reserve(distribution.owners().size());
  for (const auto& owner : distribution.owners())
    result.push_back(distribution.rank_space().linear_rank(owner));
  return result;
}

template <int Dim>
BoxArray<Dim> prepared_point_patches(int seed = -7) {
  std::vector<Box<Dim>> patches;
  for (int patch = 0; patch < 3; ++patch) {
    Index<Dim> coordinate{};
    for (int axis = 0; axis < Dim; ++axis)
      coordinate[axis] = seed + patch * (axis + 1);
    patches.push_back(point_box(coordinate));
  }
  return BoxArray<Dim>(std::move(patches));
}

template <int Dim>
RankSpace<Dim> one_rank_space(int seed = 5) {
  Index<Dim> origin{};
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis) {
    origin[axis] = seed + axis * 3;
    extent[axis] = 1;
  }
  return RankSpace<Dim>(origin, extent);
}

template <int Dim>
PreparedLoadBalanceResult<Dim> prepare_builtin_sfc() {
  const auto authority = pops::prepare_load_balance_authority<Dim>(
      "space_filling_curve", "test.nd.prepared-sfc",
      pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}});
  const std::vector<std::int64_t> weights{9, 4, 2};
  return authority.prepare(prepared_point_patches<Dim>(), one_rank_space<Dim>(), budget(), weights);
}

struct ExplicitGenericRoundRobin {
  int* invocations = nullptr;

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.load_balance.explicit_generic_round_robin", 1};
  }

  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.text("explicit-generic-round-robin").scalar(std::uint32_t{1});
  }

  template <int Dim>
  pops::parallel::OwnershipPlan<Dim> operator()(const BoxArray<Dim>& patches,
                                                const RankSpace<Dim>& rank_space,
                                                LoadBalancePreparationBudget preparation_budget,
                                                pops::LoadBalanceWeights weights) const {
    ++*invocations;
    return LoadBalanceProvider<Dim>::round_robin().prepare(patches, rank_space, preparation_budget,
                                                           weights);
  }
};

struct WrongSourceGenericProvider {
  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.load_balance.wrong_source_generic", 1};
  }

  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.text("wrong-source-generic").scalar(std::uint32_t{1});
  }

  template <int Dim>
  pops::parallel::OwnershipPlan<Dim> operator()(const BoxArray<Dim>& patches,
                                                const RankSpace<Dim>& rank_space,
                                                LoadBalancePreparationBudget preparation_budget,
                                                pops::LoadBalanceWeights weights) const {
    std::vector<Box<Dim>> changed = patches.boxes();
    changed.front().lo[0] += 100;
    changed.front().hi[0] += 100;
    return LoadBalanceProvider<Dim>::round_robin().prepare(BoxArray<Dim>(std::move(changed)),
                                                           rank_space, preparation_budget, weights);
  }
};

struct LegacyTwoDimensionalProvider {
  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.load_balance.legacy_two_dimensional", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder&) const {}
  std::vector<int> operator()(const std::vector<Box<2>>& boxes, int ranks,
                              pops::LoadBalanceWeights) const {
    return std::vector<int>(boxes.size(), ranks - 1);
  }
};

static_assert(std::is_copy_constructible_v<PreparedLoadBalanceProvider<1>>);
static_assert(std::is_nothrow_move_constructible_v<PreparedLoadBalanceProvider<2>>);
static_assert(std::is_copy_constructible_v<PreparedLoadBalanceAuthority<3>>);
static_assert(!std::is_aggregate_v<PreparedLoadBalanceResult<1>>);
static_assert(!std::is_default_constructible_v<PreparedLoadBalanceResult<2>>);
static_assert(!std::is_copy_assignable_v<PreparedLoadBalanceResult<2>>);
static_assert(!std::is_move_assignable_v<PreparedLoadBalanceResult<2>>);
static_assert(!std::is_default_constructible_v<PreparedRebalanceDecision<2>>);
static_assert(!std::is_copy_assignable_v<PreparedRebalanceDecision<2>>);
static_assert(std::is_same_v<decltype(std::declval<const PreparedLoadBalanceResult<3>&>().plan()),
                             const pops::parallel::OwnershipPlan<3>&>);
static_assert(
    !std::is_constructible_v<PreparedLoadBalanceResult<2>, pops::parallel::OwnershipPlan<2>,
                             std::string, std::string, std::string, std::string>);
static_assert(
    !std::is_constructible_v<PreparedLoadBalanceProvider<1>, LegacyTwoDimensionalProvider>);
static_assert(
    !std::is_constructible_v<PreparedLoadBalanceProvider<2>, LegacyTwoDimensionalProvider>);
static_assert(
    !std::is_constructible_v<PreparedLoadBalanceProvider<3>, LegacyTwoDimensionalProvider>);

}  // namespace

TEST(test_nd_load_balance, one_dimensional_morton_sfc_preserves_negative_origins_and_contiguity) {
  const BoxArray<1> patches(std::vector<Box<1>>{point_box(Index<1>{10}), point_box(Index<1>{0}),
                                                point_box(Index<1>{5}), point_box(Index<1>{15})});
  const RankSpace<1> ranks(Index<1>{7}, Extent<1>{2});
  const auto plan = LoadBalanceProvider<1>::space_filling_curve().prepare(patches, ranks, budget());

  EXPECT_EQ(plan.strategy(), LoadBalanceStrategy::space_filling_curve);
  EXPECT_EQ(plan.traversal(), (std::vector<std::size_t>{1, 2, 0, 3}));
  EXPECT_EQ(plan.linear_owners(), (std::vector<std::size_t>{1, 0, 0, 1}));
  EXPECT_EQ(reconstructed_linear_owners(plan.distribution()), plan.linear_owners());
  EXPECT_EQ(plan.distribution().owners(),
            (std::vector<Index<1>>{Index<1>{8}, Index<1>{7}, Index<1>{7}, Index<1>{8}}));
  EXPECT_EQ(plan.linear_rank_loads(), (std::vector<std::int64_t>{2, 2}));
  EXPECT_EQ(plan.total_weight(), 4);
  EXPECT_EQ(plan.max_load(), 2);
  EXPECT_DOUBLE_EQ(plan.imbalance(), 1.0);
}

TEST(test_nd_load_balance, three_dimensional_morton_uses_all_axes_and_spatial_rank_coordinates) {
  const Index<3> base{-10, -20, -30};
  const BoxArray<3> patches(
      std::vector<Box<3>>{point_box(Index<3>{base[0] + 1, base[1], base[2]}),
                          point_box(Index<3>{base[0], base[1] + 1, base[2]}),
                          point_box(Index<3>{base[0], base[1], base[2] + 1}), point_box(base),
                          point_box(Index<3>{base[0] + 1, base[1] + 1, base[2] + 1})});
  const RankSpace<3> ranks(Index<3>{4, -2, 7}, Extent<3>{2, 1, 2});
  const auto plan = LoadBalanceProvider<3>::space_filling_curve().prepare(patches, ranks, budget());

  EXPECT_EQ(plan.traversal(), (std::vector<std::size_t>{3, 0, 1, 2, 4}));
  EXPECT_EQ(plan.linear_owners(), (std::vector<std::size_t>{0, 1, 2, 0, 3}));
  EXPECT_EQ(plan.distribution().owners(),
            (std::vector<Index<3>>{Index<3>{4, -2, 7}, Index<3>{5, -2, 7}, Index<3>{4, -2, 8},
                                   Index<3>{4, -2, 7}, Index<3>{5, -2, 8}}));
  EXPECT_EQ(plan.linear_rank_loads(), (std::vector<std::int64_t>{2, 1, 1, 1}));
}

TEST(test_nd_load_balance, morton_offsets_retain_the_complete_signed_coordinate_range) {
  constexpr int low = std::numeric_limits<int>::min();
  constexpr int high = std::numeric_limits<int>::max();
  const BoxArray<2> patches(
      std::vector<Box<2>>{point_box(Index<2>{high, high}), point_box(Index<2>{low, high}),
                          point_box(Index<2>{high, low}), point_box(Index<2>{low, low})});
  const RankSpace<2> ranks(Index<2>{0, 0}, Extent<2>{2, 1});
  const auto plan = LoadBalanceProvider<2>::space_filling_curve().prepare(patches, ranks, budget());

  EXPECT_EQ(plan.traversal(), (std::vector<std::size_t>{3, 2, 1, 0}));
  EXPECT_EQ(plan.linear_owners(), (std::vector<std::size_t>{1, 1, 0, 0}));
}

TEST(test_nd_load_balance, weighted_lpt_reduces_max_load_with_deterministic_ties) {
  std::vector<Box<2>> boxes;
  for (int patch = 0; patch < 6; ++patch)
    boxes.push_back(point_box(Index<2>{patch * 10, 0}));
  const BoxArray<2> patches(std::move(boxes));
  const RankSpace<2> ranks(Index<2>{0, 0}, Extent<2>{3, 1});
  const std::vector<std::int64_t> weights{5, 4, 3, 2, 2, 2};

  const auto sfc =
      LoadBalanceProvider<2>::space_filling_curve().prepare(patches, ranks, budget(), weights);
  const auto lpt =
      LoadBalanceProvider<2>::weighted_lpt().prepare(patches, ranks, budget(), weights);
  EXPECT_EQ(sfc.linear_owners(), (std::vector<std::size_t>{0, 0, 1, 2, 2, 2}));
  EXPECT_EQ(sfc.linear_rank_loads(), (std::vector<std::int64_t>{9, 3, 6}));
  EXPECT_EQ(lpt.traversal(), (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(lpt.linear_owners(), (std::vector<std::size_t>{0, 1, 2, 2, 1, 0}));
  EXPECT_EQ(lpt.linear_rank_loads(), (std::vector<std::int64_t>{7, 6, 5}));
  EXPECT_LT(lpt.max_load(), sfc.max_load());
  EXPECT_LT(lpt.imbalance(), sfc.imbalance());
}

TEST(test_nd_load_balance, sfc_seeds_every_rank_when_patch_count_is_sufficient) {
  const BoxArray<1> patches(
      std::vector<Box<1>>{point_box(Index<1>{0}), point_box(Index<1>{1}), point_box(Index<1>{2})});
  const RankSpace<1> ranks(Index<1>{4}, Extent<1>{3});
  const std::vector<std::int64_t> weights{1, 1, 100};

  const auto plan =
      LoadBalanceProvider<1>::space_filling_curve().prepare(patches, ranks, budget(), weights);
  EXPECT_EQ(plan.linear_owners(), (std::vector<std::size_t>{0, 1, 2}));
  EXPECT_EQ(plan.linear_rank_loads(), (std::vector<std::int64_t>{1, 1, 100}));
}

TEST(test_nd_load_balance, sfc_uses_every_available_patch_when_ranks_outnumber_patches) {
  const BoxArray<1> patches(std::vector<Box<1>>{point_box(Index<1>{0}), point_box(Index<1>{1})});
  const RankSpace<1> ranks(Index<1>{4}, Extent<1>{4});

  const auto plan = LoadBalanceProvider<1>::space_filling_curve().prepare(patches, ranks, budget());
  EXPECT_EQ(plan.linear_owners(), (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(plan.linear_rank_loads(), (std::vector<std::int64_t>{1, 1, 0, 0}));
}

TEST(test_nd_load_balance, round_robin_is_explicit_and_linearizes_nonzero_rank_spaces) {
  const BoxArray<2> patches(std::vector<Box<2>>{
      point_box(Index<2>{0, 0}), point_box(Index<2>{1, 0}), point_box(Index<2>{2, 0}),
      point_box(Index<2>{3, 0}), point_box(Index<2>{4, 0})});
  const RankSpace<2> ranks(Index<2>{10, -3}, Extent<2>{2, 2});
  const std::vector<std::int64_t> weights{9, 8, 7, 6, 5};
  const auto plan =
      LoadBalanceProvider<2>::round_robin().prepare(patches, ranks, budget(), weights);

  EXPECT_EQ(plan.strategy(), LoadBalanceStrategy::round_robin);
  EXPECT_EQ(plan.traversal(), (std::vector<std::size_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(plan.linear_owners(), (std::vector<std::size_t>{0, 1, 2, 3, 0}));
  EXPECT_EQ(plan.distribution().owners(),
            (std::vector<Index<2>>{Index<2>{10, -3}, Index<2>{11, -3}, Index<2>{10, -2},
                                   Index<2>{11, -2}, Index<2>{10, -3}}));
  EXPECT_EQ(plan.linear_rank_loads(), (std::vector<std::int64_t>{14, 8, 7, 6}));
}

TEST(test_nd_load_balance, empty_layout_is_an_exact_partitioned_plan) {
  const BoxArray<3> patches;
  const RankSpace<3> ranks(Index<3>{2, 4, 6}, Extent<3>{2, 1, 1});
  const auto plan = LoadBalanceProvider<3>::weighted_lpt().prepare(patches, ranks, budget());

  EXPECT_EQ(plan.distribution().box_count(), 0u);
  EXPECT_FALSE(plan.distribution().replicated());
  EXPECT_TRUE(plan.distribution().owners().empty());
  EXPECT_TRUE(plan.weights().empty());
  EXPECT_TRUE(plan.linear_owners().empty());
  EXPECT_TRUE(plan.traversal().empty());
  EXPECT_EQ(plan.linear_rank_loads(), (std::vector<std::int64_t>{0, 0}));
  EXPECT_EQ(plan.total_weight(), 0);
  EXPECT_EQ(plan.max_load(), 0);
  EXPECT_DOUBLE_EQ(plan.imbalance(), 1.0);
}

TEST(test_nd_load_balance, budgets_weights_geometry_and_strategy_fail_closed) {
  const BoxArray<1> patches(std::vector<Box<1>>{point_box(Index<1>{0}), point_box(Index<1>{1})});
  const RankSpace<1> ranks(Index<1>{0}, Extent<1>{2});
  const auto provider = LoadBalanceProvider<1>::space_filling_curve();

  EXPECT_THROW((void)provider.prepare(patches, ranks, LoadBalancePreparationBudget{0, 1, 1}),
               std::invalid_argument);
  EXPECT_THROW((void)provider.prepare(patches, ranks, LoadBalancePreparationBudget{1, 2, 2}),
               std::length_error);
  EXPECT_THROW((void)provider.prepare(patches, ranks, LoadBalancePreparationBudget{2, 1, 2}),
               std::length_error);
  EXPECT_THROW((void)provider.prepare(patches, ranks, LoadBalancePreparationBudget{2, 2, 1}),
               std::length_error);
  EXPECT_THROW((void)provider.prepare(patches, ranks, budget(), std::vector<std::int64_t>{1}),
               std::invalid_argument);
  EXPECT_THROW((void)provider.prepare(patches, ranks, budget(), std::vector<std::int64_t>{1, 0}),
               std::invalid_argument);
  EXPECT_THROW((void)provider.prepare(
                   patches, ranks, budget(),
                   std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max(), 1}),
               std::overflow_error);

  const BoxArray<1> empty_patch(std::vector<Box<1>>{Box<1>{}});
  EXPECT_THROW((void)provider.prepare(empty_patch, ranks, budget()), std::invalid_argument);
  const RankSpace<1> empty_ranks(Index<1>{0}, Extent<1>{0});
  EXPECT_THROW((void)provider.prepare(patches, empty_ranks, budget()), std::invalid_argument);
  EXPECT_THROW((void)LoadBalanceProvider<1>(static_cast<LoadBalanceStrategy>(255)),
               std::invalid_argument);
}

TEST(test_nd_load_balance, produced_distribution_constructs_a_level_layout_without_translation) {
  const Box<2> domain{Index<2>{-2, 4}, Index<2>{1, 7}};
  const auto patches = BoxArray<2>::from_domain(domain, std::array<int, 2>{2, 2});
  const RankSpace<2> ranks(Index<2>{3, -1}, Extent<2>{2, 1});
  const auto ownership =
      LoadBalanceProvider<2>::space_filling_curve().prepare(patches, ranks, budget());
  const auto& distribution = ownership.distribution();

  const pops::amr::hierarchy::LevelLayout<2> level(
      0, domain, patches, distribution, pops::amr::RefinementRatio<2>{1, 1},
      pops::mesh::BoxArrayValidationBudget{patches.size(),
                                           patches.size() * (patches.size() - 1) / 2});
  EXPECT_EQ(level.distribution(), distribution);
  EXPECT_EQ(level.exact_identity().owners, distribution.owners());
}

TEST(test_nd_load_balance, prepared_authority_retains_authenticated_plans_in_every_dimension) {
  const auto one = prepare_builtin_sfc<1>();
  const auto two = prepare_builtin_sfc<2>();
  const auto three = prepare_builtin_sfc<3>();
  const std::vector<std::int64_t> expected_weights{9, 4, 2};
  const auto round_robin_authority = pops::prepare_load_balance_authority<2>(
      "round_robin", "test.nd.prepared-sfc",
      pops::PreparedProviderOptions{"pops.amr.load-balance.round-robin@1", {}});
  const auto round_robin = round_robin_authority.prepare(
      prepared_point_patches<2>(), one_rank_space<2>(), budget(), expected_weights);

  EXPECT_EQ(one.plan().strategy(), LoadBalanceStrategy::space_filling_curve);
  EXPECT_EQ(two.plan().strategy(), LoadBalanceStrategy::space_filling_curve);
  EXPECT_EQ(three.plan().strategy(), LoadBalanceStrategy::space_filling_curve);
  EXPECT_EQ(one.plan().weights(), expected_weights);
  EXPECT_EQ(two.plan().weights(), expected_weights);
  EXPECT_EQ(three.plan().weights(), expected_weights);
  EXPECT_EQ(one.plan().distribution().rank_space(), one_rank_space<1>());
  EXPECT_EQ(two.plan().distribution().rank_space(), one_rank_space<2>());
  EXPECT_EQ(three.plan().distribution().rank_space(), one_rank_space<3>());
  EXPECT_FALSE(one.collective_contract().empty());
  EXPECT_FALSE(one.collective_context_contract().empty());
  EXPECT_FALSE(one.source_contract().empty());
  EXPECT_FALSE(one.exact_contract().empty());

  EXPECT_NE(one.collective_contract(), two.collective_contract());
  EXPECT_NE(two.collective_contract(), three.collective_contract());
  EXPECT_NE(one.source_contract(), two.source_contract());
  EXPECT_NE(two.source_contract(), three.source_contract());
  EXPECT_NE(one.exact_contract(), two.exact_contract());
  EXPECT_NE(two.exact_contract(), three.exact_contract());
  EXPECT_NE(two.collective_contract(), round_robin.collective_contract());
  EXPECT_NE(two.exact_contract(), round_robin.exact_contract());
}

TEST(test_nd_load_balance, default_rebalance_policy_is_part_of_the_exact_collective_identity) {
  const auto collective_for = [](std::optional<pops::RebalancePolicy> policy) {
    const PreparedLoadBalanceAuthority<2> authority(
        "test.nd.policy-identity",
        PreparedLoadBalanceProvider<2>(ExplicitGenericRoundRobin{nullptr}), std::move(policy));
    return std::string(authority.collective_contract());
  };

  const pops::RebalancePolicy reference{
      .minimum_improvement_ppm = 125'000,
      .amortization_steps = 40,
      .migration_bandwidth_bytes_per_second = 25'000'000'000,
      .per_patch_migration_latency_nanoseconds = 2'500,
  };
  pops::RebalancePolicy changed_improvement = reference;
  ++changed_improvement.minimum_improvement_ppm;
  pops::RebalancePolicy changed_steps = reference;
  ++changed_steps.amortization_steps;
  pops::RebalancePolicy changed_bandwidth = reference;
  ++changed_bandwidth.migration_bandwidth_bytes_per_second;
  pops::RebalancePolicy changed_latency = reference;
  ++changed_latency.per_patch_migration_latency_nanoseconds;

  const std::string exact = collective_for(reference);
  EXPECT_EQ(exact, collective_for(reference));
  EXPECT_NE(exact, collective_for(std::nullopt));
  EXPECT_NE(exact, collective_for(changed_improvement));
  EXPECT_NE(exact, collective_for(changed_steps));
  EXPECT_NE(exact, collective_for(changed_bandwidth));
  EXPECT_NE(exact, collective_for(changed_latency));
}

TEST(test_nd_load_balance, prepared_result_is_bound_to_a_stable_execution_lane_identity) {
  int invocations = 0;
  const PreparedLoadBalanceAuthority<2> authority(
      "test.nd.collective-context",
      PreparedLoadBalanceProvider<2>(ExplicitGenericRoundRobin{&invocations}));
  const auto first_lane = ExecutionLane::world("test.nd.collective-context/first");
  const auto same_lane_identity = ExecutionLane::world("test.nd.collective-context/first");
  const auto second_lane = ExecutionLane::world("test.nd.collective-context/second");
  const std::vector<std::int64_t> weights{8, 5, 3};

  const auto first = authority.prepare(prepared_point_patches<2>(), one_rank_space<2>(), budget(),
                                       weights, first_lane);
  const auto same = authority.prepare(prepared_point_patches<2>(), one_rank_space<2>(), budget(),
                                      weights, same_lane_identity);
  const auto second = authority.prepare(prepared_point_patches<2>(), one_rank_space<2>(), budget(),
                                        weights, second_lane);

  EXPECT_EQ(invocations, 3);
  EXPECT_EQ(first.collective_contract(), same.collective_contract());
  EXPECT_EQ(first.collective_context_contract(), same.collective_context_contract());
  EXPECT_EQ(first.source_contract(), same.source_contract());
  EXPECT_EQ(first.exact_contract(), same.exact_contract());
  EXPECT_EQ(first.collective_contract(), second.collective_contract());
  EXPECT_NE(first.collective_context_contract(), second.collective_context_contract());
  EXPECT_NE(first.source_contract(), second.source_contract());
  EXPECT_NE(first.exact_contract(), second.exact_contract());
}

TEST(test_nd_load_balance, prepared_source_authenticates_rank_shape_origin_budget_and_weights) {
  const BoxArray<2> patches = prepared_point_patches<2>();
  const RankSpace<2> x_major(Index<2>{4, -9}, Extent<2>{2, 1});
  const RankSpace<2> y_major(Index<2>{4, -9}, Extent<2>{1, 2});
  const RankSpace<2> shifted(Index<2>{5, -9}, Extent<2>{2, 1});
  const LoadBalancePreparationBudget first_budget{8, 2, 100};
  const LoadBalancePreparationBudget second_budget{9, 2, 100};
  const std::vector<std::int64_t> first_weights{3, 4, 5};
  const std::vector<std::int64_t> second_weights{3, 4, 6};
  const BoxArray<2> shifted_patches = prepared_point_patches<2>(-6);
  const std::string collective = pops::detail::exact_load_balance_collective<2>(
      "test.nd.contract", "test.nd.provider-contract", std::nullopt);
  const std::string other_collective = pops::detail::exact_load_balance_collective<2>(
      "test.nd.contract", "test.nd.other-provider-contract", std::nullopt);
  const std::string collective_context = "test.nd.collective-context";
  const std::string other_collective_context = "test.nd.other-collective-context";

  const std::string reference = pops::detail::exact_load_balance_source<2>(
      collective, collective_context, patches, x_major, first_budget, first_weights);
  EXPECT_NE(reference,
            pops::detail::exact_load_balance_source<2>(collective, collective_context, patches,
                                                       y_major, first_budget, first_weights));
  EXPECT_NE(reference,
            pops::detail::exact_load_balance_source<2>(collective, collective_context, patches,
                                                       shifted, first_budget, first_weights));
  EXPECT_NE(reference,
            pops::detail::exact_load_balance_source<2>(collective, collective_context, patches,
                                                       x_major, second_budget, first_weights));
  EXPECT_NE(reference,
            pops::detail::exact_load_balance_source<2>(collective, collective_context, patches,
                                                       x_major, first_budget, second_weights));
  EXPECT_NE(reference, pops::detail::exact_load_balance_source<2>(collective, collective_context,
                                                                  shifted_patches, x_major,
                                                                  first_budget, first_weights));
  EXPECT_NE(reference, pops::detail::exact_load_balance_source<2>(
                           collective, other_collective_context, patches, x_major, first_budget,
                           first_weights));
  EXPECT_NE(collective, other_collective);
}

TEST(test_nd_load_balance, explicit_generic_provider_is_invoked_once_without_a_legacy_fallback) {
  int invocations = 0;
  const PreparedLoadBalanceAuthority<3> authority(
      "test.nd.explicit-generic",
      PreparedLoadBalanceProvider<3>(ExplicitGenericRoundRobin{&invocations}));
  const std::vector<std::int64_t> weights{8, 5, 3};

  const auto result =
      authority.prepare(prepared_point_patches<3>(), one_rank_space<3>(), budget(), weights);

  EXPECT_EQ(invocations, 1);
  EXPECT_EQ(result.plan().strategy(), LoadBalanceStrategy::round_robin);
  EXPECT_EQ(result.plan().weights(), weights);
  EXPECT_EQ(result.plan().distribution().rank_space(), one_rank_space<3>());
}

TEST(test_nd_load_balance, prepared_authority_rejects_a_plan_materialized_from_another_source) {
  const PreparedLoadBalanceAuthority<1> authority(
      "test.nd.wrong-source", PreparedLoadBalanceProvider<1>(WrongSourceGenericProvider{}));

  EXPECT_THROW((void)authority.prepare(prepared_point_patches<1>(), one_rank_space<1>(), budget()),
               std::invalid_argument);
}

TEST(test_nd_load_balance, unregistered_legacy_only_route_fails_closed_for_nd) {
  EXPECT_THROW((void)pops::prepare_load_balance_authority<1>(
                   "legacy_two_dimensional_only", "test.nd.no-legacy-fallback",
                   pops::PreparedProviderOptions{"pops.test.load-balance.legacy@1", {}}),
               std::invalid_argument);
  EXPECT_THROW((void)pops::prepare_load_balance_authority<3>(
                   "legacy_two_dimensional_only", "test.nd.no-legacy-fallback",
                   pops::PreparedProviderOptions{"pops.test.load-balance.legacy@1", {}}),
               std::invalid_argument);
}

TEST(test_nd_load_balance, prepared_rebalance_retains_proposal_and_reports_unchanged_mapping) {
  const BoxArray<3> patches = prepared_point_patches<3>();
  const RankSpace<3> ranks = one_rank_space<3>();
  const std::vector<std::int64_t> measured_weights{9, 4, 2};
  const auto authority = pops::prepare_load_balance_authority<3>(
      "measured_knapsack", "test.nd.measured-rebalance",
      pops::PreparedProviderOptions{
          "pops.amr.load-balance.measured-knapsack@1",
          {{"minimum_improvement_ppm", std::int64_t{50'000}},
           {"amortization_steps", std::int64_t{20}},
           {"migration_bandwidth_bytes_per_second", std::int64_t{1'000'000'000}},
           {"per_patch_migration_latency_nanoseconds", std::int64_t{0}}}});
  const auto current_plan =
      LoadBalanceProvider<3>::weighted_lpt().prepare(patches, ranks, budget(), measured_weights);
  std::vector<pops::ResourceEstimate> estimates;
  for (const std::int64_t weight : measured_weights) {
    estimates.push_back(pops::ResourceEstimate{
        .topology_epoch = 7,
        .materialization_generation = 11,
        .samples = 1,
        .cell_updates = 1,
        .compute_nanoseconds = weight,
        .memory_bytes = 1,
        .communication_bytes = 0,
        .communication_nanoseconds = 0,
        .resident_bytes = 64,
    });
  }

  const PreparedRebalanceDecision<3> decision = authority.decide_rebalance(
      2, patches, current_plan.distribution(), 7, 11, estimates, budget());

  EXPECT_FALSE(decision.accepted);
  EXPECT_EQ(decision.reason, pops::RebalanceReason::MappingUnchanged);
  EXPECT_EQ(decision.moved_patches, 0);
  EXPECT_EQ(decision.proposed.plan().weights(), measured_weights);
  EXPECT_EQ(decision.proposed.plan().distribution(), current_plan.distribution());
  EXPECT_FALSE(decision.source_contract.empty());
  EXPECT_FALSE(decision.exact_contract.empty());
  EXPECT_EQ(decision.exact_contract, pops::detail::exact_rebalance_decision(decision));
}

TEST(test_nd_load_balance, prepared_rebalance_rejects_stale_measurements_before_provider_work) {
  const BoxArray<1> patches = prepared_point_patches<1>();
  const RankSpace<1> ranks = one_rank_space<1>();
  const auto current = LoadBalanceProvider<1>::weighted_lpt().prepare(patches, ranks, budget());
  const auto authority = pops::prepare_load_balance_authority<1>(
      "measured_knapsack", "test.nd.stale-rebalance",
      pops::PreparedProviderOptions{
          "pops.amr.load-balance.measured-knapsack@1",
          {{"minimum_improvement_ppm", std::int64_t{50'000}},
           {"amortization_steps", std::int64_t{20}},
           {"migration_bandwidth_bytes_per_second", std::int64_t{1'000'000'000}},
           {"per_patch_migration_latency_nanoseconds", std::int64_t{0}}}});
  const std::vector<pops::ResourceEstimate> stale(
      patches.size(), pops::ResourceEstimate{.topology_epoch = 4,
                                             .materialization_generation = 8,
                                             .samples = 1,
                                             .cell_updates = 1,
                                             .compute_nanoseconds = 1,
                                             .memory_bytes = 1,
                                             .resident_bytes = 1});

  EXPECT_THROW(
      (void)authority.decide_rebalance(0, patches, current.distribution(), 5, 8, stale, budget()),
      std::invalid_argument);
}
