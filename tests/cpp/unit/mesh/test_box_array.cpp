// BoxArray::from_domain : decoupage en tuiles disjointes et couvrantes,
// reparties egalement (cas divisible et non divisible).

#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>

#include <limits>

using namespace pops;

TEST(test_box_array, divisible_domain_tiles_evenly) {
  // cas divisible : 8x8, max_grid_size 4 -> 2x2 = 4 boxes de 4x4
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);
  EXPECT_EQ(ba.size(), 4) << "div_count";
  for (int b = 0; b < ba.size(); ++b)
    EXPECT_TRUE(ba[b].nx() == 4 && ba[b].ny() == 4) << "div_tile_size";
  EXPECT_TRUE(ba.tiles_exactly(dom)) << "div_tiles_exactly";
  EXPECT_EQ(ba.num_cells(), 64) << "div_num_cells";
  EXPECT_EQ(ba.bounding_box(), dom) << "div_bbox";
}

TEST(test_box_array, nondivisible_domain_splits_evenly) {
  // cas non divisible : 10x10, max_grid_size 4 -> 3x3 = 9 boxes, tailles 4,3,3
  Box2D dom2 = Box2D::from_extents(10, 10);
  BoxArray ba2 = BoxArray::from_domain(dom2, 4);
  EXPECT_EQ(ba2.size(), 9) << "ndiv_count";
  EXPECT_TRUE(ba2.tiles_exactly(dom2)) << "ndiv_tiles_exactly";
  EXPECT_EQ(ba2.num_cells(), 100) << "ndiv_num_cells";
  EXPECT_EQ(ba2.bounding_box(), dom2) << "ndiv_bbox";
  // premiere tuile en x doit faire 4 (base 3 + reste 1), les suivantes 3
  EXPECT_TRUE(ba2[0].nx() == 4 && ba2[1].nx() == 3 && ba2[2].nx() == 3) << "ndiv_even_split";
}

TEST(test_box_array, rectangular_domain_accepts_independent_axis_tile_limits) {
  const Box2D domain = Box2D::from_extents(12, 6);
  const BoxArray boxes = BoxArray::from_domain(domain, 6, 2);
  ASSERT_EQ(boxes.size(), 6);
  EXPECT_TRUE(boxes.tiles_exactly(domain));
  for (int box = 0; box < boxes.size(); ++box) {
    EXPECT_EQ(boxes[box].nx(), 6);
    EXPECT_EQ(boxes[box].ny(), 2);
  }
}

TEST(test_box_array, accepts_irregular_exact_tiling_and_shared_edges) {
  const Box2D domain{{-2, -3}, {3, 2}};
  const BoxArray boxes({Box2D{{-2, -3}, {-1, 2}}, Box2D{{0, -3}, {3, -1}}, Box2D{{0, 0}, {3, 2}}});
  EXPECT_TRUE(boxes.tiles_exactly(domain));
}

TEST(test_box_array, rejects_gap_overlap_empty_and_outside_boxes) {
  const Box2D domain{{0, 0}, {3, 1}};

  EXPECT_FALSE(BoxArray({Box2D{{0, 0}, {2, 1}}}).tiles_exactly(domain));
  // Area equals the domain, but the second box overlaps the first and leaves x=3 uncovered.
  EXPECT_FALSE(BoxArray({Box2D{{0, 0}, {2, 1}}, Box2D{{2, 0}, {2, 1}}}).tiles_exactly(domain));
  EXPECT_FALSE(BoxArray({Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {1, 1}}, Box2D{{2, 0}, {3, 1}}})
                   .tiles_exactly(domain));
  EXPECT_FALSE(BoxArray({Box2D{{0, 0}, {2, 1}}, Box2D{{3, 0}, {4, 1}}}).tiles_exactly(domain));
}

TEST(test_box_array, handles_full_int_coordinate_range_without_overflow) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();
  const Box2D domain{{lo, lo}, {hi, hi}};

  EXPECT_TRUE(
      BoxArray({Box2D{{lo, lo}, {-1, hi}}, Box2D{{0, lo}, {hi, hi}}}).tiles_exactly(domain));
  EXPECT_FALSE(BoxArray({domain, domain}).tiles_exactly(domain));
}

TEST(test_box_array, empty_layout_tiles_only_empty_domain) {
  EXPECT_TRUE(BoxArray{}.tiles_exactly(Box2D{}));
  EXPECT_FALSE(BoxArray{}.tiles_exactly(Box2D::from_extents(1, 1)));
  EXPECT_FALSE(BoxArray({Box2D{}}).tiles_exactly(Box2D{}));
}

TEST(test_box_array, from_domain_validates_tile_count_before_allocation) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();
  const Box2D full{{lo, lo}, {hi, hi}};

  EXPECT_THROW((void)BoxArray::from_domain(Box2D::from_extents(1, 1), 0), std::invalid_argument);
  EXPECT_THROW((void)BoxArray::from_domain(full, 1), std::length_error);

  // A large coordinate interval remains a valid geometric object when it can be represented by a
  // bounded number of tiles.  No hi + 1 signed-int arithmetic is used while splitting it.
  const BoxArray tiled = BoxArray::from_domain(full, hi);
  EXPECT_EQ(tiled.size(), 9);
  EXPECT_TRUE(tiled.tiles_exactly(full));
}

TEST(test_box_array, num_cells_rejects_single_box_and_sum_overflow) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();
  EXPECT_THROW((void)BoxArray({Box2D{{lo, lo}, {hi, hi}}}).num_cells(), std::overflow_error);

  const Box2D large = Box2D::from_extents(hi, hi);
  EXPECT_THROW((void)BoxArray({large, large, large}).num_cells(), std::overflow_error);
}
