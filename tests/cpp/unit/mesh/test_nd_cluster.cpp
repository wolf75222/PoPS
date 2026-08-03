#include <gtest/gtest.h>

#include <pops/amr/hierarchy/nd/berger_rigoutsos.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <vector>

namespace nd = pops::amr::hierarchy::nd;
namespace mesh = pops::mesh;

using pops::Box;
using pops::Extent;
using pops::Index;

namespace {

constexpr mesh::BoxArrayValidationBudget kLayoutBudget{64, 2016};

template <int Dim>
nd::ClusterOptions<Dim> options(std::array<int, Dim> minimum, std::array<int, Dim> maximum,
                                double efficiency = 0.7) {
  return nd::ClusterOptions<Dim>{efficiency, minimum, maximum,
                                 nd::ClusterWorkBudget{16, 1024, 100000, 1024}};
}

template <int Dim>
nd::LevelLayout<Dim> replicated_level(const Box<Dim>& domain, const mesh::BoxArray<Dim>& patches,
                                      const mesh::RankSpace<Dim>& ranks) {
  nd::RefinementRatio<Dim> ratio{};
  ratio.fill(1);
  return nd::LevelLayout<Dim>(0, domain, patches,
                              mesh::Distribution<Dim>::replicated(patches, ranks), ratio,
                              kLayoutBudget);
}

bool box_less(const Box<3>& left, const Box<3>& right) {
  for (int axis = 0; axis < 3; ++axis) {
    if (left.lo[axis] != right.lo[axis])
      return left.lo[axis] < right.lo[axis];
    if (left.hi[axis] != right.hi[axis])
      return left.hi[axis] < right.hi[axis];
  }
  return false;
}

Index<3> permute(const Index<3>& index, const std::array<int, 3>& axes) {
  return Index<3>{index[axes[0]], index[axes[1]], index[axes[2]]};
}

Box<3> permute(const Box<3>& box, const std::array<int, 3>& axes) {
  return Box<3>{permute(box.lo, axes), permute(box.hi, axes)};
}

}  // namespace

TEST(test_nd_cluster, one_dimensional_holes_split_into_deterministic_boxes) {
  const Box<1> domain{Index<1>{-8}, Index<1>{7}};
  const mesh::BoxArray<1> patches(std::vector<Box<1>>{domain});
  const mesh::RankSpace<1> ranks{Index<1>{3}, Extent<1>{1}};
  const auto level = replicated_level(domain, patches, ranks);
  nd::TagMask<1> mask(level, Index<1>{3}, nd::TagMaskBudget{1, 16, 16, 16});
  for (const int coordinate : {-6, -5, 4, 5})
    mask.set(Index<1>{coordinate});

  const nd::BergerRigoutsosProvider<1> provider;
  const std::array<nd::TagMask<1>, 1> shards{mask};
  const auto first = provider.cluster(shards, options<1>({1}, {16}));
  const auto second = provider.cluster(shards, options<1>({1}, {16}));
  EXPECT_EQ(first.boxes.boxes(), (std::vector<Box<1>>{Box<1>{Index<1>{-6}, Index<1>{-5}},
                                                      Box<1>{Index<1>{4}, Index<1>{5}}}));
  EXPECT_EQ(first.identity, second.identity);
  EXPECT_EQ(first.identity.provider, nd::BergerRigoutsosProvider<1>::kIdentity);
}

TEST(test_nd_cluster, anisotropic_final_chop_is_axis_indexed) {
  const Box<2> domain{Index<2>{-2, 5}, Index<2>{1, 10}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{2, -1}, Extent<2>{1, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  nd::TagMask<2> mask(level, Index<2>{2, -1}, nd::TagMaskBudget{1, 24, 24, 24});
  for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
    for (int i = domain.lo[0]; i <= domain.hi[0]; ++i)
      mask.set(Index<2>{i, j});

  const nd::BergerRigoutsosProvider<2> provider;
  const std::array<nd::TagMask<2>, 1> shards{mask};
  const auto clustered = provider.cluster(shards, options<2>({1, 1}, {2, 3}));
  ASSERT_EQ(clustered.boxes.size(), 4U);
  for (const Box<2>& box : clustered.boxes.boxes()) {
    EXPECT_LE(box.length(0), 2);
    EXPECT_LE(box.length(1), 3);
  }
}

TEST(test_nd_cluster, three_dimensional_axis_permutation_maps_to_the_same_clusters) {
  const Box<3> domain{Index<3>{0, 0, 0}, Index<3>{7, 4, 2}};
  const mesh::BoxArray<3> patches(std::vector<Box<3>>{domain});
  const mesh::RankSpace<3> ranks{Index<3>{0, 0, 0}, Extent<3>{1, 1, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  nd::TagMask<3> mask(level, Index<3>{0, 0, 0}, nd::TagMaskBudget{1, 120, 120, 120});
  for (int z = 0; z <= 0; ++z)
    for (int y = 0; y <= 1; ++y)
      for (int x = 0; x <= 1; ++x)
        mask.set(Index<3>{x, y, z});
  for (int z = 2; z <= 2; ++z)
    for (int y = 3; y <= 4; ++y)
      for (int x = 6; x <= 7; ++x)
        mask.set(Index<3>{x, y, z});

  const nd::BergerRigoutsosProvider<3> provider;
  const std::array<nd::TagMask<3>, 1> shards{mask};
  const auto original = provider.cluster(shards, options<3>({1, 1, 1}, {8, 5, 3}));

  const std::array<int, 3> axes{2, 1, 0};
  const Box<3> transposed_domain = permute(domain, axes);
  const mesh::BoxArray<3> transposed_patches(std::vector<Box<3>>{transposed_domain});
  const auto transposed_level = replicated_level(transposed_domain, transposed_patches, ranks);
  nd::TagMask<3> transposed(transposed_level, Index<3>{0, 0, 0},
                            nd::TagMaskBudget{1, 120, 120, 120});
  mask.for_each_tagged_in(domain,
                          [&](const Index<3>& index) { transposed.set(permute(index, axes)); });
  const std::array<nd::TagMask<3>, 1> transposed_shards{transposed};
  const auto mapped = provider.cluster(transposed_shards, options<3>({1, 1, 1}, {3, 5, 8}));

  std::vector<Box<3>> expected;
  for (const Box<3>& box : original.boxes.boxes())
    expected.push_back(permute(box, axes));
  std::sort(expected.begin(), expected.end(), box_less);
  EXPECT_EQ(mapped.boxes.boxes(), expected);
}

TEST(test_nd_cluster, partitioned_shards_are_canonicalized_and_exactly_authenticated) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{7, 3}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
                                                      Box<2>{Index<2>{4, 0}, Index<2>{7, 3}}});
  const mesh::RankSpace<2> ranks{Index<2>{10, -2}, Extent<2>{2, 1}};
  const auto distribution =
      mesh::Distribution<2>::partitioned(patches, ranks, {Index<2>{10, -2}, Index<2>{11, -2}});
  const nd::LevelLayout<2> level(0, domain, patches, distribution, {1, 1}, kLayoutBudget);
  nd::TagMask<2> left(level, Index<2>{10, -2}, nd::TagMaskBudget{1, 16, 16, 16});
  nd::TagMask<2> right(level, Index<2>{11, -2}, nd::TagMaskBudget{1, 16, 16, 16});
  left.set(Index<2>{1, 1});
  right.set(Index<2>{6, 2});

  const nd::BergerRigoutsosProvider<2> provider;
  const std::vector<nd::TagMask<2>> ordered{left, right};
  const std::vector<nd::TagMask<2>> reversed{right, left};
  const auto first = provider.cluster(ordered, options<2>({1, 1}, {4, 4}));
  const auto second = provider.cluster(reversed, options<2>({1, 1}, {4, 4}));
  EXPECT_EQ(first.boxes, second.boxes);
  EXPECT_EQ(first.identity, second.identity);
  EXPECT_EQ(first.boxes.boxes(), (std::vector<Box<2>>{Box<2>{Index<2>{1, 1}, Index<2>{1, 1}},
                                                      Box<2>{Index<2>{6, 2}, Index<2>{6, 2}}}));

  const std::vector<nd::TagMask<2>> missing{left};
  const std::vector<nd::TagMask<2>> duplicate{left, left};
  EXPECT_THROW((void)provider.cluster(missing, options<2>({1, 1}, {4, 4})), std::invalid_argument);
  EXPECT_THROW((void)provider.cluster(duplicate, options<2>({1, 1}, {4, 4})),
               std::invalid_argument);

  const auto reversed_distribution =
      mesh::Distribution<2>::partitioned(patches, ranks, {Index<2>{11, -2}, Index<2>{10, -2}});
  const nd::LevelLayout<2> other_level(0, domain, patches, reversed_distribution, {1, 1},
                                       kLayoutBudget);
  nd::TagMask<2> other(other_level, Index<2>{10, -2}, nd::TagMaskBudget{1, 16, 16, 16});
  const std::vector<nd::TagMask<2>> mismatched{left, other};
  EXPECT_THROW((void)provider.cluster(mismatched, options<2>({1, 1}, {4, 4})),
               std::invalid_argument);
}

TEST(test_nd_cluster, replicated_multi_shard_and_invalid_or_exhausted_budgets_fail_closed) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 3}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{2, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  nd::TagMask<2> first(level, Index<2>{0, 0}, nd::TagMaskBudget{1, 16, 16, 16});
  nd::TagMask<2> second(level, Index<2>{1, 0}, nd::TagMaskBudget{1, 16, 16, 16});
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i)
      first.set(Index<2>{i, j});
  const nd::BergerRigoutsosProvider<2> provider;
  const std::vector<nd::TagMask<2>> duplicated{first, second};
  EXPECT_THROW((void)provider.cluster(duplicated, options<2>({1, 1}, {4, 4})),
               std::invalid_argument);

  const std::array<nd::TagMask<2>, 1> shard{first};
  auto invalid_efficiency = options<2>({1, 1}, {4, 4});
  invalid_efficiency.min_efficiency = 0.0;
  EXPECT_THROW((void)provider.cluster(shard, invalid_efficiency), std::invalid_argument);
  auto invalid_size = options<2>({2, 1}, {1, 4});
  EXPECT_THROW((void)provider.cluster(shard, invalid_size), std::invalid_argument);
  auto invalid_budget = options<2>({1, 1}, {4, 4});
  invalid_budget.budget.recursion_nodes = 0;
  EXPECT_THROW((void)provider.cluster(shard, invalid_budget), std::invalid_argument);

  auto cells_exhausted = options<2>({1, 1}, {4, 4});
  cells_exhausted.budget.cell_visits = 15;
  EXPECT_THROW((void)provider.cluster(shard, cells_exhausted), std::length_error);
  auto output_exhausted = options<2>({1, 1}, {1, 1});
  output_exhausted.budget.output_boxes = 2;
  EXPECT_THROW((void)provider.cluster(shard, output_exhausted), std::length_error);
  auto shard_exhausted = options<2>({1, 1}, {4, 4});
  shard_exhausted.budget.shards = 0;
  EXPECT_THROW((void)provider.cluster(shard, shard_exhausted), std::invalid_argument);
}
