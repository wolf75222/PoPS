#include <gtest/gtest.h>

#include <pops/amr/hierarchy/nd/level_layout.hpp>
#include <pops/parallel/nd/load_balance_provider.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::mesh::BoxArray;
using pops::mesh::RankSpace;
using pops::parallel::nd::LoadBalancePreparationBudget;
using pops::parallel::nd::LoadBalanceProvider;
using pops::parallel::nd::LoadBalanceStrategy;

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
  const auto distribution =
      LoadBalanceProvider<2>::space_filling_curve().distribute(patches, ranks, budget());

  const pops::amr::hierarchy::nd::LevelLayout<2> level(
      0, domain, patches, distribution, pops::amr::hierarchy::nd::RefinementRatio<2>{1, 1},
      pops::mesh::BoxArrayValidationBudget{patches.size(),
                                           patches.size() * (patches.size() - 1) / 2});
  EXPECT_EQ(level.distribution(), distribution);
  EXPECT_EQ(level.exact_identity().owners, distribution.owners());
}
