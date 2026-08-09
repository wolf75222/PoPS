#include <gtest/gtest.h>

#include <pops/amr/hierarchy/hierarchy_plan.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hierarchy = pops::amr::hierarchy;
namespace mesh = pops::mesh;

using pops::Box;
using pops::Extent;
using pops::Index;

namespace {

constexpr mesh::BoxArrayValidationBudget kLayoutBudget{64, 2016};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{8, 4096};

template <int Dim>
hierarchy::LevelLayout<Dim> make_level(int level, const Box<Dim>& domain,
                                       const mesh::BoxArray<Dim>& patches,
                                       const mesh::RankSpace<Dim>& ranks,
                                       const std::vector<Index<Dim>>& owners,
                                       const pops::amr::RefinementRatio<Dim>& ratio) {
  return hierarchy::LevelLayout<Dim>(level, domain, patches,
                                     mesh::Distribution<Dim>::partitioned(patches, ranks, owners),
                                     ratio, kLayoutBudget);
}

}  // namespace

TEST(test_nd_hierarchy_plan, one_dimensional_nonzero_origin_and_ratio_are_exact) {
  const Box<1> coarse_domain{Index<1>{-3}, Index<1>{4}};
  const mesh::BoxArray<1> coarse_patches =
      mesh::BoxArray<1>::from_domain(coarse_domain, std::array<int, 1>{4});
  const mesh::RankSpace<1> ranks{Index<1>{-2}, Extent<1>{2}};
  const auto coarse = make_level<1>(0, coarse_domain, coarse_patches, ranks,
                                    {Index<1>{-2}, Index<1>{-1}}, pops::amr::RefinementRatio<1>{1});

  const pops::amr::RefinementRatio<1> ratio{3};
  const Box<1> fine_domain = hierarchy::refine_box(coarse_domain, ratio);
  const Box<1> fine_patch = hierarchy::refine_box(Box<1>{Index<1>{-3}, Index<1>{-1}}, ratio);
  const mesh::BoxArray<1> fine_patches(std::vector<Box<1>>{fine_patch});
  const auto fine = make_level<1>(1, fine_domain, fine_patches, ranks, {Index<1>{-1}}, ratio);

  const hierarchy::HierarchyPlan<1> plan({coarse, fine}, kHierarchyBudget);
  ASSERT_EQ(plan.num_levels(), 2U);
  EXPECT_EQ(plan.level(1).domain(), (Box<1>{Index<1>{-9}, Index<1>{14}}));
  EXPECT_EQ(hierarchy::coarsen_box(fine_patch, ratio), (Box<1>{Index<1>{-3}, Index<1>{-1}}));
  EXPECT_EQ(plan.exact_identity(),
            hierarchy::HierarchyPlan<1>({coarse, fine}, kHierarchyBudget).exact_identity());
}

TEST(test_nd_hierarchy_plan, anisotropic_two_and_three_dimensional_levels_are_validated) {
  const Box<2> plane_domain{Index<2>{-2, 4}, Index<2>{1, 7}};
  const mesh::BoxArray<2> plane_patches =
      mesh::BoxArray<2>::from_domain(plane_domain, std::array<int, 2>{2, 2});
  const mesh::RankSpace<2> plane_ranks{Index<2>{5, -2}, Extent<2>{2, 1}};
  const auto plane_coarse =
      make_level<2>(0, plane_domain, plane_patches, plane_ranks,
                    {Index<2>{5, -2}, Index<2>{6, -2}, Index<2>{5, -2}, Index<2>{6, -2}},
                    pops::amr::RefinementRatio<2>{1, 1});
  const pops::amr::RefinementRatio<2> plane_ratio{2, 3};
  const Box<2> plane_fine_patch =
      hierarchy::refine_box(Box<2>{Index<2>{-1, 5}, Index<2>{0, 6}}, plane_ratio);
  const mesh::BoxArray<2> plane_fine_patches(std::vector<Box<2>>{plane_fine_patch});
  const auto plane_fine =
      make_level<2>(1, hierarchy::refine_box(plane_domain, plane_ratio), plane_fine_patches,
                    plane_ranks, {Index<2>{6, -2}}, plane_ratio);
  const hierarchy::HierarchyPlan<2> plane({plane_coarse, plane_fine}, kHierarchyBudget);
  EXPECT_EQ(plane.level(1).domain(), (Box<2>{Index<2>{-4, 12}, Index<2>{3, 23}}));
  EXPECT_EQ(plane_fine_patch, (Box<2>{Index<2>{-2, 15}, Index<2>{1, 20}}));

  const Box<3> volume_domain{Index<3>{-2, 3, -1}, Index<3>{1, 4, 1}};
  const mesh::BoxArray<3> volume_patches =
      mesh::BoxArray<3>::from_domain(volume_domain, std::array<int, 3>{2, 2, 3});
  const mesh::RankSpace<3> volume_ranks{Index<3>{7, -3, 2}, Extent<3>{2, 1, 1}};
  const auto volume_coarse = make_level<3>(0, volume_domain, volume_patches, volume_ranks,
                                           {Index<3>{7, -3, 2}, Index<3>{8, -3, 2}},
                                           pops::amr::RefinementRatio<3>{1, 1, 1});
  const pops::amr::RefinementRatio<3> volume_ratio{2, 1, 3};
  const Box<3> volume_fine_patch =
      hierarchy::refine_box(Box<3>{Index<3>{-2, 3, 0}, Index<3>{-1, 4, 1}}, volume_ratio);
  const mesh::BoxArray<3> volume_fine_patches(std::vector<Box<3>>{volume_fine_patch});
  const auto volume_fine =
      make_level<3>(1, hierarchy::refine_box(volume_domain, volume_ratio), volume_fine_patches,
                    volume_ranks, {Index<3>{7, -3, 2}}, volume_ratio);
  const hierarchy::HierarchyPlan<3> volume({volume_coarse, volume_fine}, kHierarchyBudget);
  EXPECT_EQ(volume.level(1).domain(), (Box<3>{Index<3>{-4, 3, -3}, Index<3>{3, 4, 5}}));
  EXPECT_EQ(hierarchy::coarsen_box(volume_fine_patch, volume_ratio),
            (Box<3>{Index<3>{-2, 3, 0}, Index<3>{-1, 4, 1}}));
}

TEST(test_nd_hierarchy_plan, layout_and_hierarchy_refuse_invalid_contracts) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const mesh::BoxArray<1> full(std::vector<Box<1>>{domain});
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto distribution = mesh::Distribution<1>::partitioned(full, ranks, {Index<1>{0}});

  EXPECT_THROW((void)hierarchy::LevelLayout<1>(0, domain, full, distribution,
                                               pops::amr::RefinementRatio<1>{2}, kLayoutBudget),
               std::invalid_argument);
  EXPECT_THROW((void)hierarchy::LevelLayout<1>(1, domain, full, distribution,
                                               pops::amr::RefinementRatio<1>{1}, kLayoutBudget),
               std::invalid_argument);
  EXPECT_THROW(
      (void)hierarchy::LevelLayout<1>(
          0, domain, mesh::BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{2}}}),
          distribution, pops::amr::RefinementRatio<1>{1}, kLayoutBudget),
      std::invalid_argument);
  EXPECT_THROW((void)hierarchy::LevelLayout<1>(
                   0, domain, full,
                   mesh::Distribution<1>::partitioned(
                       mesh::BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}},
                                                             Box<1>{Index<1>{2}, Index<1>{3}}}),
                       ranks, {Index<1>{0}, Index<1>{0}}),
                   pops::amr::RefinementRatio<1>{1}, kLayoutBudget),
               std::invalid_argument);
  EXPECT_THROW((void)hierarchy::LevelLayout<1>(0, domain, full, distribution,
                                               pops::amr::RefinementRatio<1>{1},
                                               mesh::BoxArrayValidationBudget{0, 0}),
               std::length_error);

  const auto coarse =
      make_level<1>(0, domain, full, ranks, {Index<1>{0}}, pops::amr::RefinementRatio<1>{1});
  const Box<1> fine_domain = hierarchy::refine_box(domain, pops::amr::RefinementRatio<1>{2});
  const mesh::BoxArray<1> unaligned(std::vector<Box<1>>{Box<1>{Index<1>{1}, Index<1>{4}}});
  const auto unaligned_level = make_level<1>(1, fine_domain, unaligned, ranks, {Index<1>{0}},
                                             pops::amr::RefinementRatio<1>{2});
  EXPECT_THROW((void)hierarchy::HierarchyPlan<1>({coarse, unaligned_level}, kHierarchyBudget),
               std::invalid_argument);

  const mesh::RankSpace<1> changed_ranks{Index<1>{1}, Extent<1>{1}};
  const mesh::BoxArray<1> aligned(std::vector<Box<1>>{
      hierarchy::refine_box(Box<1>{Index<1>{0}, Index<1>{1}}, pops::amr::RefinementRatio<1>{2})});
  const auto changed_space = make_level<1>(1, fine_domain, aligned, changed_ranks, {Index<1>{1}},
                                           pops::amr::RefinementRatio<1>{2});
  EXPECT_THROW((void)hierarchy::HierarchyPlan<1>({coarse, changed_space}, kHierarchyBudget),
               std::invalid_argument);
  EXPECT_THROW(
      (void)hierarchy::HierarchyPlan<1>({coarse}, hierarchy::HierarchyValidationBudget{0, 0}),
      std::length_error);
  EXPECT_THROW((void)hierarchy::HierarchyPlan<1>({coarse, changed_space},
                                                 hierarchy::HierarchyValidationBudget{2, 0}),
               std::invalid_argument);
  EXPECT_THROW((void)hierarchy::refine_box(Box<1>{Index<1>{std::numeric_limits<int>::max()},
                                                  Index<1>{std::numeric_limits<int>::max()}},
                                           pops::amr::RefinementRatio<1>{2}),
               std::overflow_error);
  EXPECT_THROW((void)hierarchy::refine_box(Box<1>{Index<1>{std::numeric_limits<int>::min()},
                                                  Index<1>{std::numeric_limits<int>::min()}},
                                           pops::amr::RefinementRatio<1>{2}),
               std::overflow_error);
}

TEST(test_nd_hierarchy_plan, sparse_parent_coverage_and_nonconsecutive_levels_fail_closed) {
  const Box<1> coarse_domain{Index<1>{0}, Index<1>{3}};
  const mesh::BoxArray<1> coarse_patches(std::vector<Box<1>>{coarse_domain});
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto coarse = make_level<1>(0, coarse_domain, coarse_patches, ranks, {Index<1>{0}},
                                    pops::amr::RefinementRatio<1>{1});

  const Box<1> level_one_domain =
      hierarchy::refine_box(coarse_domain, pops::amr::RefinementRatio<1>{2});
  const mesh::BoxArray<1> sparse_one(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{3}}});
  const auto level_one = make_level<1>(1, level_one_domain, sparse_one, ranks, {Index<1>{0}},
                                       pops::amr::RefinementRatio<1>{2});
  const Box<1> level_two_domain =
      hierarchy::refine_box(level_one_domain, pops::amr::RefinementRatio<1>{2});
  const mesh::BoxArray<1> uncovered(std::vector<Box<1>>{Box<1>{Index<1>{8}, Index<1>{11}}});
  const auto level_two = make_level<1>(2, level_two_domain, uncovered, ranks, {Index<1>{0}},
                                       pops::amr::RefinementRatio<1>{2});
  EXPECT_THROW((void)hierarchy::HierarchyPlan<1>({coarse, level_one, level_two}, kHierarchyBudget),
               std::invalid_argument);

  const auto mislabeled = make_level<1>(2, level_one_domain, sparse_one, ranks, {Index<1>{0}},
                                        pops::amr::RefinementRatio<1>{2});
  EXPECT_THROW((void)hierarchy::HierarchyPlan<1>({coarse, mislabeled}, kHierarchyBudget),
               std::invalid_argument);
  EXPECT_THROW((void)hierarchy::HierarchyPlan<1>({coarse, level_one},
                                                 hierarchy::HierarchyValidationBudget{2, 0}),
               std::length_error);
}

TEST(test_nd_hierarchy_plan, exact_identity_tracks_order_ownership_and_replacement) {
  const Box<1> domain{Index<1>{-2}, Index<1>{1}};
  const mesh::BoxArray<1> patches = mesh::BoxArray<1>::from_domain(domain, std::array<int, 1>{2});
  const mesh::RankSpace<1> ranks{Index<1>{4}, Extent<1>{2}};
  const auto left_owned = make_level<1>(0, domain, patches, ranks, {Index<1>{4}, Index<1>{5}},
                                        pops::amr::RefinementRatio<1>{1});
  const auto right_owned = make_level<1>(0, domain, patches, ranks, {Index<1>{5}, Index<1>{4}},
                                         pops::amr::RefinementRatio<1>{1});
  const hierarchy::HierarchyPlan<1> left_plan({left_owned}, kHierarchyBudget);
  const hierarchy::HierarchyPlan<1> right_plan({right_owned}, kHierarchyBudget);
  EXPECT_NE(left_plan.exact_identity(), right_plan.exact_identity());
  const mesh::BoxArray<1> reordered_patches(std::vector<Box<1>>{patches[1], patches[0]});
  const auto reordered =
      make_level<1>(0, domain, reordered_patches, ranks, {Index<1>{5}, Index<1>{4}},
                    pops::amr::RefinementRatio<1>{1});
  const hierarchy::HierarchyPlan<1> reordered_plan({reordered}, kHierarchyBudget);
  EXPECT_NE(left_plan.exact_identity(), reordered_plan.exact_identity());

  const hierarchy::HierarchyValidationBudget append_forbidden{1, 4096};
  const hierarchy::HierarchyPlan<1> limited_plan({left_owned}, append_forbidden);
  EXPECT_NE(left_plan.exact_identity(), limited_plan.exact_identity());

  const Box<1> fine_domain = hierarchy::refine_box(domain, pops::amr::RefinementRatio<1>{2});
  const mesh::BoxArray<1> fine_patches(std::vector<Box<1>>{
      hierarchy::refine_box(Box<1>{Index<1>{-2}, Index<1>{-1}}, pops::amr::RefinementRatio<1>{2})});
  const auto fine = make_level<1>(1, fine_domain, fine_patches, ranks, {Index<1>{4}},
                                  pops::amr::RefinementRatio<1>{2});
  const hierarchy::HierarchyPlan<1> appended = left_plan.with_level(fine);
  ASSERT_EQ(appended.num_levels(), 2U);
  EXPECT_EQ(appended.level(0).exact_identity(), left_owned.exact_identity());
  EXPECT_NE(appended.exact_identity(), left_plan.exact_identity());
  EXPECT_THROW((void)limited_plan.with_level(fine), std::length_error);
  EXPECT_THROW((void)left_plan.level(1), std::out_of_range);

  const Box<1> finer_domain = hierarchy::refine_box(fine_domain, pops::amr::RefinementRatio<1>{2});
  const mesh::BoxArray<1> finer_patches(std::vector<Box<1>>{
      hierarchy::refine_box(fine_patches[0], pops::amr::RefinementRatio<1>{2})});
  const auto finer = make_level<1>(2, finer_domain, finer_patches, ranks, {Index<1>{4}},
                                   pops::amr::RefinementRatio<1>{2});
  const hierarchy::HierarchyPlan<1> three_levels({left_owned, fine, finer}, kHierarchyBudget);

  const mesh::BoxArray<1> replacement_patches(std::vector<Box<1>>{
      hierarchy::refine_box(Box<1>{Index<1>{0}, Index<1>{1}}, pops::amr::RefinementRatio<1>{2})});
  const auto replacement = make_level<1>(1, fine_domain, replacement_patches, ranks, {Index<1>{5}},
                                         pops::amr::RefinementRatio<1>{2});
  const hierarchy::HierarchyPlan<1> truncated = three_levels.with_level(replacement);
  ASSERT_EQ(truncated.num_levels(), 2U);
  EXPECT_EQ(truncated.level(1).exact_identity(), replacement.exact_identity());
}
