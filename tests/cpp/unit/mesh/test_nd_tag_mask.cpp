#include <gtest/gtest.h>

#include <pops/amr/tagging/tag_mask.hpp>

#include <array>
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
hierarchy::LevelLayout<Dim> make_partitioned_level(int level, const Box<Dim>& domain,
                                                   const mesh::BoxArray<Dim>& patches,
                                                   const mesh::RankSpace<Dim>& ranks,
                                                   const std::vector<Index<Dim>>& owners,
                                                   const pops::amr::RefinementRatio<Dim>& ratio) {
  return hierarchy::LevelLayout<Dim>(level, domain, patches,
                                     mesh::Distribution<Dim>::partitioned(patches, ranks, owners),
                                     ratio, kLayoutBudget);
}

}  // namespace

TEST(test_nd_tag_mask, partitioned_storage_contains_only_owned_patches) {
  const Box<1> domain{Index<1>{-4}, Index<1>{3}};
  const mesh::BoxArray<1> patches = mesh::BoxArray<1>::from_domain(domain, std::array<int, 1>{2});
  const mesh::RankSpace<1> ranks{Index<1>{10}, Extent<1>{2}};
  const auto level = make_partitioned_level<1>(
      0, domain, patches, ranks, {Index<1>{10}, Index<1>{11}, Index<1>{10}, Index<1>{11}},
      pops::amr::RefinementRatio<1>{1});
  tagging::TagMask<1> mask(level, Index<1>{10}, tag_budget(4, 2, 2, 4));

  ASSERT_EQ(mask.local_patch_count(), 2U);
  EXPECT_EQ(mask.local_cell_count(), 4U);
  EXPECT_EQ(mask.patches()[0].global_patch, 0U);
  EXPECT_EQ(mask.patches()[1].global_patch, 2U);
  mask.set(Index<1>{-4});
  mask.set(2, Index<1>{0});
  EXPECT_TRUE(mask.tagged(0, Index<1>{-4}));
  EXPECT_TRUE(mask.tagged(2, Index<1>{0}));
  EXPECT_EQ(mask.count(), 2U);
  EXPECT_THROW(mask.set(Index<1>{-2}), std::out_of_range);
  EXPECT_THROW(mask.set(1, Index<1>{-2}), std::out_of_range);
  EXPECT_THROW((void)mask.tagged(0, Index<1>{3}), std::out_of_range);
}

TEST(test_nd_tag_mask, all_storage_dimensions_honor_nonzero_origins_and_axis_zero_order) {
  const Box<2> plane{Index<2>{-2, 5}, Index<2>{0, 6}};
  const mesh::BoxArray<2> plane_patches(std::vector<Box<2>>{plane});
  const mesh::RankSpace<2> plane_ranks{Index<2>{3, -1}, Extent<2>{1, 1}};
  const auto plane_level = make_partitioned_level<2>(
      0, plane, plane_patches, plane_ranks, {Index<2>{3, -1}}, pops::amr::RefinementRatio<2>{1, 1});
  tagging::TagMask<2> plane_mask(plane_level, Index<2>{3, -1}, tag_budget(1, 1, 6, 6));
  plane_mask.set(Index<2>{-2, 5});
  plane_mask.set(Index<2>{0, 5});
  plane_mask.set(Index<2>{-1, 6});
  std::vector<Index<2>> plane_tags;
  plane_mask.for_each_tagged_in(plane, [&](const Index<2>& index) { plane_tags.push_back(index); });
  EXPECT_EQ(plane_tags, (std::vector<Index<2>>{Index<2>{-2, 5}, Index<2>{0, 5}, Index<2>{-1, 6}}));

  const Box<3> volume{Index<3>{4, -2, 7}, Index<3>{5, 0, 8}};
  const mesh::BoxArray<3> volume_patches(std::vector<Box<3>>{volume});
  const mesh::RankSpace<3> volume_ranks{Index<3>{-3, 2, 1}, Extent<3>{1, 1, 1}};
  const auto volume_level =
      make_partitioned_level<3>(0, volume, volume_patches, volume_ranks, {Index<3>{-3, 2, 1}},
                                pops::amr::RefinementRatio<3>{1, 1, 1});
  tagging::TagMask<3> volume_mask(volume_level, Index<3>{-3, 2, 1}, tag_budget(1, 1, 12, 12));
  volume_mask.set(Index<3>{5, -1, 8});
  EXPECT_EQ(volume_mask.count(), 1U);
  EXPECT_TRUE(volume_mask.tagged(0, Index<3>{5, -1, 8}));
}

TEST(test_nd_tag_mask, explicit_metadata_cell_byte_and_identity_budgets_fail_closed) {
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const mesh::BoxArray<1> patches = mesh::BoxArray<1>::from_domain(domain, std::array<int, 1>{4});
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto level = make_partitioned_level<1>(
      0, domain, patches, ranks, {Index<1>{0}, Index<1>{0}}, pops::amr::RefinementRatio<1>{1});

  EXPECT_THROW((void)tagging::TagMask<1>(level, Index<1>{0},
                                         tagging::TagMaskBudget{1, 2, 4, 8, 8, kIdentityBudget}),
               std::length_error);
  EXPECT_THROW((void)tagging::TagMask<1>(level, Index<1>{0},
                                         tagging::TagMaskBudget{2, 1, 4, 8, 8, kIdentityBudget}),
               std::length_error);
  EXPECT_THROW((void)tagging::TagMask<1>(level, Index<1>{0},
                                         tagging::TagMaskBudget{2, 2, 3, 8, 8, kIdentityBudget}),
               std::length_error);
  EXPECT_THROW((void)tagging::TagMask<1>(level, Index<1>{0},
                                         tagging::TagMaskBudget{2, 2, 4, 7, 8, kIdentityBudget}),
               std::length_error);
  EXPECT_THROW((void)tagging::TagMask<1>(level, Index<1>{0},
                                         tagging::TagMaskBudget{2, 2, 4, 8, 7, kIdentityBudget}),
               std::length_error);
  EXPECT_THROW(
      (void)tagging::TagMask<1>(level, Index<1>{0}, tagging::TagMaskBudget{2, 2, 4, 8, 8, 1}),
      std::length_error);
  EXPECT_THROW((void)tagging::TagMask<1>(level, Index<1>{2}, tag_budget(2, 2, 4, 8)),
               std::out_of_range);
}

TEST(test_nd_tag_mask, exact_identity_tracks_rank_patch_topology_and_tag_bits) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const mesh::BoxArray<1> patches = mesh::BoxArray<1>::from_domain(domain, std::array<int, 1>{2});
  const mesh::RankSpace<1> ranks{Index<1>{4}, Extent<1>{2}};
  const auto level = make_partitioned_level<1>(
      0, domain, patches, ranks, {Index<1>{4}, Index<1>{5}}, pops::amr::RefinementRatio<1>{1});
  tagging::TagMask<1> first(level, Index<1>{4}, tag_budget(2, 1, 2, 2));
  tagging::TagMask<1> same(level, Index<1>{4}, tag_budget(2, 1, 2, 2));
  EXPECT_EQ(first.exact_identity(), same.exact_identity());
  first.set(Index<1>{0});
  EXPECT_NE(first.exact_identity(), same.exact_identity());

  tagging::TagMask<1> other_rank(level, Index<1>{5}, tag_budget(2, 1, 2, 2));
  EXPECT_NE(first.exact_identity(), other_rank.exact_identity());
}
