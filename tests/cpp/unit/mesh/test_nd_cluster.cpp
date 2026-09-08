#include <gtest/gtest.h>

#include <pops/amr/tagging/berger_rigoutsos.hpp>
#include <pops/amr/regridding/regrid.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace hierarchy = pops::amr::hierarchy;
namespace tagging = pops::amr::tagging;
namespace mesh = pops::mesh;

using pops::Box;
using pops::Extent;
using pops::Index;

namespace {

constexpr mesh::BoxArrayValidationBudget kLayoutBudget{64, 2016};
constexpr std::size_t kIdentityBudget = 1U << 20;

constexpr tagging::TagMaskBudget tag_budget(std::size_t global_patches, std::size_t owned_patches,
                                            std::size_t cells_per_patch, std::size_t owned_cells) {
  return tagging::TagMaskBudget{global_patches, owned_patches, cells_per_patch,
                                owned_cells,    owned_cells,   kIdentityBudget};
}

template <int Dim>
tagging::ClusterOptions<Dim> options(std::array<int, Dim> minimum, std::array<int, Dim> maximum,
                                     double efficiency = 0.7) {
  return tagging::ClusterOptions<Dim>{
      efficiency, minimum, maximum,
      tagging::ClusterWorkBudget{16, 1024, 100000, 1024, kIdentityBudget}};
}

template <int Dim>
hierarchy::LevelLayout<Dim> replicated_level(const Box<Dim>& domain,
                                             const mesh::BoxArray<Dim>& patches,
                                             const mesh::RankSpace<Dim>& ranks) {
  pops::amr::RefinementRatio<Dim> ratio{};
  return hierarchy::LevelLayout<Dim>(0, domain, patches,
                                     mesh::Distribution<Dim>::replicated(patches, ranks), ratio,
                                     kLayoutBudget);
}

template <int Dim>
bool box_less(const Box<Dim>& left, const Box<Dim>& right) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (left.lo[axis] != right.lo[axis])
      return left.lo[axis] < right.lo[axis];
    if (left.hi[axis] != right.hi[axis])
      return left.hi[axis] < right.hi[axis];
  }
  return false;
}

template <int Dim, std::size_t AxisCount>
  requires(AxisCount == static_cast<std::size_t>(Dim))
Index<Dim> permute(const Index<Dim>& index, const std::array<int, AxisCount>& axes) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = index[axes[axis]];
  return result;
}

template <int Dim, std::size_t AxisCount>
  requires(AxisCount == static_cast<std::size_t>(Dim))
Box<Dim> permute(const Box<Dim>& box, const std::array<int, AxisCount>& axes) {
  return Box<Dim>{permute(box.lo, axes), permute(box.hi, axes)};
}

}  // namespace

TEST(test_nd_cluster, one_dimensional_holes_split_into_deterministic_boxes) {
  const Box<1> domain{Index<1>{-8}, Index<1>{7}};
  const mesh::BoxArray<1> patches(std::vector<Box<1>>{domain});
  const mesh::RankSpace<1> ranks{Index<1>{3}, Extent<1>{1}};
  const auto level = replicated_level(domain, patches, ranks);
  tagging::TagMask<1> mask(level, Index<1>{3}, tag_budget(1, 1, 16, 16));
  for (const int coordinate : {-6, -5, 4, 5})
    mask.set(Index<1>{coordinate});

  const tagging::BergerRigoutsosProvider<1> provider;
  const std::array<tagging::TagMask<1>, 1> shards{mask};
  const auto first = provider.cluster(shards, options<1>({1}, {16}));
  const auto second = provider.cluster(shards, options<1>({1}, {16}));
  EXPECT_EQ(first.boxes.boxes(), (std::vector<Box<1>>{Box<1>{Index<1>{-6}, Index<1>{-5}},
                                                      Box<1>{Index<1>{4}, Index<1>{5}}}));
  EXPECT_EQ(first.identity, second.identity);
  EXPECT_EQ(first.identity.provider, tagging::BergerRigoutsosProvider<1>::kIdentity);
}

TEST(test_nd_cluster, anisotropic_final_chop_is_axis_indexed) {
  const Box<2> domain{Index<2>{-2, 5}, Index<2>{1, 10}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{2, -1}, Extent<2>{1, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  tagging::TagMask<2> mask(level, Index<2>{2, -1}, tag_budget(1, 1, 24, 24));
  for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
    for (int i = domain.lo[0]; i <= domain.hi[0]; ++i)
      mask.set(Index<2>{i, j});

  const tagging::BergerRigoutsosProvider<2> provider;
  const std::array<tagging::TagMask<2>, 1> shards{mask};
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
  tagging::TagMask<3> mask(level, Index<3>{0, 0, 0}, tag_budget(1, 1, 120, 120));
  for (int z = 0; z <= 0; ++z)
    for (int y = 0; y <= 1; ++y)
      for (int x = 0; x <= 1; ++x)
        mask.set(Index<3>{x, y, z});
  for (int z = 2; z <= 2; ++z)
    for (int y = 3; y <= 4; ++y)
      for (int x = 6; x <= 7; ++x)
        mask.set(Index<3>{x, y, z});

  const tagging::BergerRigoutsosProvider<3> provider;
  const std::array<tagging::TagMask<3>, 1> shards{mask};
  const auto original = provider.cluster(shards, options<3>({1, 1, 1}, {8, 5, 3}));

  const std::array<int, 3> axes{2, 1, 0};
  const Box<3> transposed_domain = permute(domain, axes);
  const mesh::BoxArray<3> transposed_patches(std::vector<Box<3>>{transposed_domain});
  const auto transposed_level = replicated_level(transposed_domain, transposed_patches, ranks);
  tagging::TagMask<3> transposed(transposed_level, Index<3>{0, 0, 0}, tag_budget(1, 1, 120, 120));
  mask.for_each_tagged_in(domain,
                          [&](const Index<3>& index) { transposed.set(permute(index, axes)); });
  const std::array<tagging::TagMask<3>, 1> transposed_shards{transposed};
  const auto mapped = provider.cluster(transposed_shards, options<3>({1, 1, 1}, {3, 5, 8}));

  std::vector<Box<3>> expected;
  for (const Box<3>& box : original.boxes.boxes())
    expected.push_back(permute(box, axes));
  std::sort(expected.begin(), expected.end(), box_less<3>);
  EXPECT_EQ(mapped.boxes.boxes(), expected);
}

TEST(test_nd_cluster, equal_length_axis_ties_are_permutation_equivariant) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 3}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{1, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  tagging::TagMask<2> mask(level, Index<2>{0, 0}, tag_budget(1, 1, 16, 16));
  for (const int y : {0, 1, 3})
    for (const int x : {0, 2, 3})
      mask.set(Index<2>{x, y});

  const tagging::BergerRigoutsosProvider<2> provider;
  const std::array<tagging::TagMask<2>, 1> shards{mask};
  const auto original = provider.cluster(shards, options<2>({1, 1}, {4, 4}, 0.6));

  const std::array<int, 2> axes{1, 0};
  const auto transposed_level = replicated_level(permute(domain, axes), patches, ranks);
  tagging::TagMask<2> transposed(transposed_level, Index<2>{0, 0}, tag_budget(1, 1, 16, 16));
  mask.for_each_tagged_in(domain,
                          [&](const Index<2>& index) { transposed.set(permute(index, axes)); });
  const std::array<tagging::TagMask<2>, 1> transposed_shards{transposed};
  const auto mapped = provider.cluster(transposed_shards, options<2>({1, 1}, {4, 4}, 0.6));

  std::vector<Box<2>> expected;
  for (const Box<2>& box : original.boxes.boxes())
    expected.push_back(permute(box, axes));
  std::sort(expected.begin(), expected.end(), box_less<2>);
  EXPECT_GT(expected.size(), 1U);
  EXPECT_EQ(mapped.boxes.boxes(), expected);
}

TEST(test_nd_cluster, partitioned_shards_are_canonicalized_and_exactly_authenticated) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{7, 3}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
                                                      Box<2>{Index<2>{4, 0}, Index<2>{7, 3}}});
  const mesh::RankSpace<2> ranks{Index<2>{10, -2}, Extent<2>{3, 1}};
  const auto distribution =
      mesh::Distribution<2>::partitioned(patches, ranks, {Index<2>{10, -2}, Index<2>{11, -2}});
  const hierarchy::LevelLayout<2> level(0, domain, patches, distribution,
                                        pops::amr::RefinementRatio<2>{1, 1}, kLayoutBudget);
  tagging::TagMask<2> left(level, Index<2>{10, -2}, tag_budget(2, 1, 16, 16));
  tagging::TagMask<2> right(level, Index<2>{11, -2}, tag_budget(2, 1, 16, 16));
  tagging::TagMask<2> empty_rank(level, Index<2>{12, -2}, tag_budget(2, 0, 16, 0));
  left.set(Index<2>{1, 1});
  right.set(Index<2>{6, 2});

  const tagging::BergerRigoutsosProvider<2> provider;
  const std::vector<tagging::TagMask<2>> ordered{left, right, empty_rank};
  const std::vector<tagging::TagMask<2>> reversed{empty_rank, right, left};
  const auto first = provider.cluster(ordered, options<2>({1, 1}, {4, 4}));
  const auto second = provider.cluster(reversed, options<2>({1, 1}, {4, 4}));
  EXPECT_EQ(first.boxes, second.boxes);
  EXPECT_EQ(first.identity, second.identity);
  EXPECT_EQ(first.boxes.boxes(), (std::vector<Box<2>>{Box<2>{Index<2>{1, 1}, Index<2>{1, 1}},
                                                      Box<2>{Index<2>{6, 2}, Index<2>{6, 2}}}));

  const std::vector<tagging::TagMask<2>> missing{left};
  const std::vector<tagging::TagMask<2>> duplicate{left, left, empty_rank};
  EXPECT_THROW((void)provider.cluster(missing, options<2>({1, 1}, {4, 4})), std::invalid_argument);
  EXPECT_THROW((void)provider.cluster(duplicate, options<2>({1, 1}, {4, 4})),
               std::invalid_argument);

  const auto reversed_distribution =
      mesh::Distribution<2>::partitioned(patches, ranks, {Index<2>{11, -2}, Index<2>{10, -2}});
  const hierarchy::LevelLayout<2> other_level(0, domain, patches, reversed_distribution,
                                              pops::amr::RefinementRatio<2>{1, 1}, kLayoutBudget);
  tagging::TagMask<2> other(other_level, Index<2>{10, -2}, tag_budget(2, 1, 16, 16));
  const std::vector<tagging::TagMask<2>> mismatched{left, other, empty_rank};
  EXPECT_THROW((void)provider.cluster(mismatched, options<2>({1, 1}, {4, 4})),
               std::invalid_argument);
}

TEST(test_nd_cluster, replicated_shards_are_authenticated_and_canonicalized) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 3}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{2, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  tagging::TagMask<2> first(level, Index<2>{0, 0}, tag_budget(1, 1, 16, 16));
  tagging::TagMask<2> second(level, Index<2>{1, 0}, tag_budget(1, 1, 16, 16));
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) {
      first.set(Index<2>{i, j});
      second.set(Index<2>{i, j});
    }
  const tagging::BergerRigoutsosProvider<2> provider;
  const std::vector<tagging::TagMask<2>> ordered{first, second};
  const std::vector<tagging::TagMask<2>> reversed{second, first};
  const auto canonical = provider.cluster(ordered, options<2>({1, 1}, {4, 4}));
  const auto reordered = provider.cluster(reversed, options<2>({1, 1}, {4, 4}));
  EXPECT_EQ(canonical.identity, reordered.identity);
  ASSERT_EQ(canonical.identity.canonical_shards.size(), 2U);
  EXPECT_FALSE(canonical.identity.canonical_shards[0].replicated_alias);
  EXPECT_EQ(canonical.identity.canonical_shards[0].patches.size(), 1U);
  EXPECT_TRUE(canonical.identity.canonical_shards[1].replicated_alias);
  EXPECT_TRUE(canonical.identity.canonical_shards[1].patches.empty());

  const std::array<tagging::TagMask<2>, 1> missing{first};
  EXPECT_THROW((void)provider.cluster(missing, options<2>({1, 1}, {4, 4})), std::invalid_argument);

  tagging::TagMask<2> divergent = second;
  divergent.set(Index<2>{0, 0}, false);
  const std::vector<tagging::TagMask<2>> disagreement{first, divergent};
  EXPECT_THROW((void)provider.cluster(disagreement, options<2>({1, 1}, {4, 4})),
               std::invalid_argument);
}

TEST(test_nd_cluster, invalid_or_exhausted_work_budgets_fail_closed) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 3}};
  const mesh::BoxArray<2> patches(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{2, 1}};
  const auto level = replicated_level(domain, patches, ranks);
  tagging::TagMask<2> first(level, Index<2>{0, 0}, tag_budget(1, 1, 16, 16));
  tagging::TagMask<2> second(level, Index<2>{1, 0}, tag_budget(1, 1, 16, 16));
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) {
      first.set(Index<2>{i, j});
      second.set(Index<2>{i, j});
    }
  const tagging::BergerRigoutsosProvider<2> provider;
  const std::vector<tagging::TagMask<2>> shards{first, second};

  auto invalid_efficiency = options<2>({1, 1}, {4, 4});
  invalid_efficiency.min_efficiency = 0.0;
  EXPECT_THROW((void)provider.cluster(shards, invalid_efficiency), std::invalid_argument);
  auto invalid_size = options<2>({2, 1}, {1, 4});
  EXPECT_THROW((void)provider.cluster(shards, invalid_size), std::invalid_argument);
  auto invalid_budget = options<2>({1, 1}, {4, 4});
  invalid_budget.budget.recursion_nodes = 0;
  EXPECT_THROW((void)provider.cluster(shards, invalid_budget), std::invalid_argument);
  auto invalid_identity_budget = options<2>({1, 1}, {4, 4});
  invalid_identity_budget.budget.identity_bytes = 0;
  EXPECT_THROW((void)provider.cluster(shards, invalid_identity_budget), std::invalid_argument);

  auto cells_exhausted = options<2>({1, 1}, {4, 4});
  cells_exhausted.budget.cell_visits = 15;
  EXPECT_THROW((void)provider.cluster(shards, cells_exhausted), std::length_error);
  auto output_exhausted = options<2>({1, 1}, {1, 1});
  output_exhausted.budget.output_boxes = 2;
  EXPECT_THROW((void)provider.cluster(shards, output_exhausted), std::length_error);
  auto shard_exhausted = options<2>({1, 1}, {4, 4});
  shard_exhausted.budget.shards = 1;
  EXPECT_THROW((void)provider.cluster(shards, shard_exhausted), std::length_error);

  tagging::TagMask<2> sparse_first(level, Index<2>{0, 0}, tag_budget(1, 1, 16, 16));
  tagging::TagMask<2> sparse_second(level, Index<2>{1, 0}, tag_budget(1, 1, 16, 16));
  for (const Index<2> index : {Index<2>{0, 0}, Index<2>{3, 3}}) {
    sparse_first.set(index);
    sparse_second.set(index);
  }
  const std::vector<tagging::TagMask<2>> sparse_shards{sparse_first, sparse_second};
  auto recursion_exhausted = options<2>({1, 1}, {4, 4});
  recursion_exhausted.budget.recursion_nodes = 1;
  EXPECT_THROW((void)provider.cluster(sparse_shards, recursion_exhausted), std::length_error);

  auto identity_exhausted = options<2>({1, 1}, {4, 4});
  identity_exhausted.budget.identity_bytes = 1;
  EXPECT_THROW((void)provider.cluster(shards, identity_exhausted), std::length_error);
}

namespace {

template <int Dim>
void prove_nesting_preserves_patch_union_and_periodic_images() {
  Index<Dim> lower{};
  Index<Dim> upper{};
  Extent<Dim> rank_extent{};
  std::array<int, Dim> minimum{};
  std::array<int, Dim> maximum{};
  std::array<int, Dim> ratios{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = 15;
    rank_extent[axis] = 1;
    minimum[axis] = 1;
    maximum[axis] = 16;
    ratios[axis] = 2;
  }
  const Box<Dim> domain{lower, upper};
  const mesh::RankSpace<Dim> ranks{lower, rank_extent};
  auto controls = options<Dim>(minimum, maximum);
  controls.nesting_buffer.fill(2);
  controls.budget.recursion_nodes = static_cast<std::size_t>(2 * domain.numPts());
  controls.budget.cell_visits =
      static_cast<std::size_t>(domain.numPts()) * controls.budget.recursion_nodes;
  const auto cluster = [&](std::vector<Box<Dim>> boxes, bool periodic) {
    const mesh::BoxArray<Dim> patches(std::move(boxes));
    const hierarchy::LevelLayout<Dim> level(1, domain, patches,
                                            mesh::Distribution<Dim>::replicated(patches, ranks),
                                            pops::amr::RefinementRatio<Dim>(ratios), kLayoutBudget);
    tagging::TagMask<Dim> mask(
        level, lower, tag_budget(patches.size(), patches.size(), domain.numPts(), domain.numPts()));
    for (const auto& patch : mask.patches())
      for (std::size_t ordinal = 0; ordinal < patch.tags.size(); ++ordinal) {
        std::size_t rest = ordinal;
        Index<Dim> cell{};
        for (int axis = 0; axis < Dim; ++axis) {
          cell[axis] = patch.box.lo[axis] + static_cast<int>(rest % patch.box.length(axis));
          rest /= static_cast<std::size_t>(patch.box.length(axis));
        }
        mask.set(patch.global_patch, cell);
      }
    controls.periodic_axes[0] = periodic;
    return tagging::BergerRigoutsosProvider<Dim>{}.cluster(
        std::array<tagging::TagMask<Dim>, 1>{mask}, controls);
  };
  const auto cells = [](const auto& result) {
    std::int64_t count = 0;
    for (const auto& box : result.boxes.boxes())
      count += box.numPts();
    return count;
  };
  Box<Dim> first = domain;
  Box<Dim> second = domain;
  first.hi[0] = 7;
  second.lo[0] = 8;
  EXPECT_EQ(cells(cluster({first, second}, false)), domain.numPts());
  first.hi[0] = 3;
  second.lo[0] = 12;
  const auto wrapped = cluster({first, second}, true);
  EXPECT_EQ(cells(wrapped), domain.numPts() / 4);
  for (const auto& box : wrapped.boxes.boxes())
    EXPECT_TRUE(box.hi[0] <= 1 || box.lo[0] >= 14);
  EXPECT_TRUE(cluster({first}, true).boxes.empty());
  const auto node_budget = controls.budget.recursion_nodes;
  controls.budget.recursion_nodes = 1;
  EXPECT_THROW((void)cluster({first}, true), std::length_error);
  controls.budget.recursion_nodes = node_budget;

  // A complete union produces identical boxes under these policies; the exact preparation
  // identity must nevertheless retain both the guaranteed padding and periodic topology.
  first = domain;
  const auto physical = cluster({first}, false);
  const auto periodic = cluster({first}, true);
  controls.nesting_buffer.fill(0);
  const auto no_padding = cluster({first}, false);
  EXPECT_EQ(physical.boxes.boxes(), periodic.boxes.boxes());
  EXPECT_EQ(physical.boxes.boxes(), no_padding.boxes.boxes());
  const auto exact = [&](const auto& result) {
    return pops::amr::regridding::detail::exact_regrid_contract<Dim>(
        result.identity.source_level, pops::amr::RefinementRatio<Dim>(ratios), result.identity, {},
        {}, {});
  };
  EXPECT_NE(exact(physical), exact(periodic));
  EXPECT_NE(exact(physical), exact(no_padding));
}

}  // namespace

TEST(test_nd_cluster, proper_nesting_preserves_parent_seams_and_authenticates_periodic_images) {
  prove_nesting_preserves_patch_union_and_periodic_images<1>();
  prove_nesting_preserves_patch_union_and_periodic_images<2>();
  prove_nesting_preserves_patch_union_and_periodic_images<3>();
}


TEST(test_nd_cluster, refinement_coverage_is_independent_of_parent_tiles_and_rank_owners) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{7, 3}};
  const mesh::BoxArray<2> mono(std::vector<Box<2>>{domain});
  const mesh::BoxArray<2> tiled(std::vector<Box<2>>{
      Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
      Box<2>{Index<2>{4, 0}, Index<2>{7, 3}}});
  const mesh::RankSpace<2> one_rank{Index<2>{0, 0}, Extent<2>{1, 1}};
  const auto mono_level = replicated_level(domain, mono, one_rank);
  const auto tiled_level = replicated_level(domain, tiled, one_rank);
  tagging::TagMask<2> mono_mask(mono_level, Index<2>{0, 0}, tag_budget(1, 1, 32, 32));
  tagging::TagMask<2> tiled_mask(tiled_level, Index<2>{0, 0}, tag_budget(2, 2, 16, 32));

  // The global rectangle has efficiency 4/6 >= 0.6; clustering its left tile
  // independently sees only 2/4 and discards the two legitimate untagged cells.
  const std::array<Index<2>, 4> tags{
      Index<2>{2, 1}, Index<2>{3, 2}, Index<2>{4, 1}, Index<2>{4, 2}};
  for (const auto& cell : tags) {
    mono_mask.set(cell);
    tiled_mask.set(cell);
  }
  const auto controls = options<2>({1, 1}, {8, 4}, 0.6);
  const tagging::BergerRigoutsosProvider<2> provider;
  const auto expected = provider.cluster(std::array{mono_mask}, controls);
  const auto local_tiled = provider.cluster(std::array{tiled_mask}, controls);
  EXPECT_EQ(expected.boxes.boxes(),
            (std::vector<Box<2>>{Box<2>{Index<2>{2, 1}, Index<2>{4, 2}}}));
  EXPECT_EQ(local_tiled.boxes, expected.boxes);
  EXPECT_NE(local_tiled.identity.source_level, expected.identity.source_level);

  const mesh::RankSpace<2> ranks{Index<2>{5, -1}, Extent<2>{3, 1}};
  const hierarchy::LevelLayout<2> partitioned(
      0, domain, tiled,
      mesh::Distribution<2>::partitioned(tiled, ranks, {Index<2>{6, -1}, Index<2>{5, -1}}),
      pops::amr::RefinementRatio<2>{1, 1}, kLayoutBudget);
  tagging::TagMask<2> right(partitioned, Index<2>{5, -1}, tag_budget(2, 1, 16, 16));
  tagging::TagMask<2> left(partitioned, Index<2>{6, -1}, tag_budget(2, 1, 16, 16));
  tagging::TagMask<2> empty(partitioned, Index<2>{7, -1}, tag_budget(2, 0, 16, 0));
  for (const auto& cell : tags)
    (cell[0] < 4 ? left : right).set(cell);
  EXPECT_EQ(provider.cluster(std::array{empty, left, right}, controls).boxes, expected.boxes);
}

TEST(test_nd_cluster, unbuffered_global_clustering_does_not_refine_parent_coverage_holes) {
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const mesh::BoxArray<1> patches(std::vector<Box<1>>{
      Box<1>{Index<1>{0}, Index<1>{2}}, Box<1>{Index<1>{5}, Index<1>{7}}});
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto level = replicated_level(domain, patches, ranks);
  tagging::TagMask<1> mask(level, Index<1>{0}, tag_budget(2, 2, 3, 6));
  for (int cell : {0, 1, 2, 5, 6, 7})
    mask.set(Index<1>{cell});
  const auto result = tagging::BergerRigoutsosProvider<1>{}.cluster(
      std::array{mask}, options<1>({1}, {8}, 0.5));
  EXPECT_EQ(result.boxes.boxes(), patches.boxes());
}
