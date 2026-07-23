// AmrHierarchy : construction du niveau grossier, ajout d'un niveau fin imbrique,
// et interpolation grossier->fin sur la region raffinee.

#include <gtest/gtest.h>

#include "load_balance_test_authority.hpp"

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/refinement.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace pops;

namespace {
bool close(Real x, Real y) {
  return std::fabs(x - y) < 1e-9;
}
}  // namespace

TEST(test_amr_hierarchy, Runs) {
  Box2D cdom = Box2D::from_extents(8, 8);  // [0..7]
  AmrHierarchy h(cdom, /*max_grid_size=*/8, /*ncomp=*/1, /*ngrow=*/1,
                 test::prepare_test_space_filling_curve_load_balance(), /*ref_ratio=*/2);

  EXPECT_EQ(h.num_levels(), 1) << "lev0_count";
  EXPECT_EQ(h.ref_ratio(), 2) << "ref_ratio";
  EXPECT_TRUE(h.domain(0) == cdom) << "lev0_domain";
  EXPECT_EQ(h.data(0).local_size(), 1) << "lev0_one_box";

  // niveau fin imbrique : cellules grossieres [2..5]^2 raffinees -> [4..11]^2
  BoxArray fba(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  h.add_level(fba);

  EXPECT_EQ(h.num_levels(), 2) << "lev1_count";
  EXPECT_TRUE(h.domain(1) == cdom.refine(2)) << "lev1_domain";  // [0..15]
  EXPECT_TRUE(h.boxes(1)[0] == (Box2D{{4, 4}, {11, 11}})) << "lev1_box";
  EXPECT_EQ(h.data(1).n_grow(), 1) << "lev1_ghost";

  auto fine_data = [&]() {
    return MultiFab(fba, DistributionMapping(fba.size(), n_ranks()), /*ncomp=*/1, /*ngrow=*/1);
  };
  EXPECT_THROW(h.install_level(0, fba, fine_data(), world_communicator_view()), std::out_of_range)
      << "install_level_rejects_level0";
  EXPECT_THROW(h.install_level(h.num_levels() + 1, fba, fine_data(), world_communicator_view()),
               std::out_of_range)
      << "install_level_rejects_gap";
  EXPECT_EQ(h.num_levels(), 2) << "install_level_invalid_keeps_levels";

  // remplir le grossier puis interpoler vers le fin imbrique
  Array4 ac = h.data(0).fab(0).array();
  for_each_cell(cdom, [ac](int I, int J) { ac(I, J, 0) = I + 100.0 * J; });
  interpolate(h.data(0), h.data(1), h.ref_ratio());

  // fine(i,j) = gc(i/2, j/2)
  EXPECT_TRUE(close(h.data(1).fab(0)(4, 4, 0), 202.0)) << "interp_44";      // gc(2,2)
  EXPECT_TRUE(close(h.data(1).fab(0)(11, 11, 0), 505.0)) << "interp_1111";  // gc(5,5)
}

TEST(test_amr_hierarchy, RejectsInvalidConstructionAndBoundsEveryPublicAccess) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  EXPECT_THROW(AmrHierarchy(Box2D{}, 8, 1, 0, load_balance), std::invalid_argument);
  EXPECT_THROW(AmrHierarchy(domain, 0, 1, 0, load_balance), std::invalid_argument);
  EXPECT_THROW(AmrHierarchy(domain, 8, 0, 0, load_balance), std::invalid_argument);
  EXPECT_THROW(AmrHierarchy(domain, 8, 1, -1, load_balance), std::invalid_argument);
  EXPECT_THROW(AmrHierarchy(domain, 8, 1, 0, {}), std::invalid_argument);

  AmrHierarchy hierarchy(domain, 8, 1, 0, load_balance);
  EXPECT_THROW((void)hierarchy.domain(-1), std::out_of_range);
  EXPECT_THROW((void)hierarchy.domain(1), std::out_of_range);
  EXPECT_THROW((void)hierarchy.boxes(-1), std::out_of_range);
  EXPECT_THROW((void)hierarchy.boxes(1), std::out_of_range);
  EXPECT_THROW((void)hierarchy.data(-1), std::out_of_range);
  EXPECT_THROW((void)hierarchy.data(1), std::out_of_range);
  const AmrHierarchy& constant = hierarchy;
  EXPECT_THROW((void)constant.data(-1), std::out_of_range);
  EXPECT_THROW((void)constant.data(1), std::out_of_range);
  EXPECT_THROW(hierarchy.clear_above(-1), std::out_of_range);
  EXPECT_THROW(hierarchy.clear_above(1), std::out_of_range);
  EXPECT_EQ(hierarchy.num_levels(), 1);
}

TEST(test_amr_hierarchy, FineLayoutValidationIsAtomic) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  AmrHierarchy hierarchy(domain, 8, 1, 1, load_balance);

  const auto expect_unchanged = [&](const BoxArray& candidate) {
    EXPECT_THROW(hierarchy.add_level(candidate), std::invalid_argument);
    EXPECT_EQ(hierarchy.num_levels(), 1);
    EXPECT_TRUE(hierarchy.domain(0) == domain);
  };
  expect_unchanged(BoxArray{});
  expect_unchanged(BoxArray({Box2D{}}));
  expect_unchanged(BoxArray({Box2D{{1, 0}, {4, 3}}}));
  expect_unchanged(BoxArray({Box2D{{16, 0}, {19, 3}}}));
  expect_unchanged(
      BoxArray({Box2D{{0, 0}, {7, 7}}, Box2D{{4, 4}, {11, 11}}}));

  const BoxArray partial_parent({Box2D{{0, 0}, {7, 15}}});
  hierarchy.add_level(partial_parent);
  ASSERT_EQ(hierarchy.num_levels(), 2);
  const BoxArray outside_parent_coverage({Box2D{{16, 0}, {19, 3}}});
  EXPECT_THROW(hierarchy.add_level(outside_parent_coverage), std::invalid_argument);
  EXPECT_EQ(hierarchy.num_levels(), 2);
  EXPECT_EQ(hierarchy.boxes(1).boxes(), partial_parent.boxes());
}

TEST(test_amr_hierarchy, InstallValidatesCompleteCandidateBeforeTruncatingFinerLevels) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  AmrHierarchy hierarchy(domain, 8, 1, 1, load_balance);
  const BoxArray level_one({Box2D{{0, 0}, {7, 15}}});
  const BoxArray level_two({Box2D{{0, 0}, {7, 15}}});
  hierarchy.add_level(level_one);
  hierarchy.add_level(level_two);
  ASSERT_EQ(hierarchy.num_levels(), 3);

  auto field = [](const BoxArray& boxes, int components, int ghosts) {
    return MultiFab(boxes, DistributionMapping(boxes.size(), n_ranks()), components, ghosts);
  };
  const BoxArray different_layout({Box2D{{8, 0}, {15, 15}}});
  EXPECT_THROW(hierarchy.install_level(1, level_one, field(different_layout, 1, 1),
                                       world_communicator_view()),
               std::invalid_argument);
  EXPECT_THROW(hierarchy.install_level(1, level_one, field(level_one, 2, 1),
                                       world_communicator_view()),
               std::invalid_argument);
  EXPECT_THROW(hierarchy.install_level(1, level_one, field(level_one, 1, 2),
                                       world_communicator_view()),
               std::invalid_argument);
  EXPECT_EQ(hierarchy.num_levels(), 3);
  EXPECT_EQ(hierarchy.boxes(1).boxes(), level_one.boxes());
  EXPECT_EQ(hierarchy.boxes(2).boxes(), level_two.boxes());

  const BoxArray replacement({Box2D{{8, 0}, {15, 15}}});
  hierarchy.install_level(1, replacement, field(replacement, 1, 1),
                          world_communicator_view());
  EXPECT_EQ(hierarchy.num_levels(), 2);
  EXPECT_EQ(hierarchy.boxes(1).boxes(), replacement.boxes());
}
