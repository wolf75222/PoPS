// Box2D : arithmetique entiere de l'espace d'indices (grow, refine/coarsen,
// intersect, contains), y compris la division plancher du coarsen sur les
// indices negatifs des ghosts.

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>

using namespace pops;

TEST(test_box2d, extents_and_contains) {
  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  EXPECT_TRUE(b.nx() == 4 && b.ny() == 3) << "extents";
  EXPECT_EQ(b.num_cells(), 12) << "num_cells";
  EXPECT_TRUE(b.contains(3, 2) && !b.contains(4, 0)) << "contains";
}

TEST(test_box2d, grow) {
  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Box2D g = b.grow(1);                  // [-1..4] x [-1..3]
  EXPECT_TRUE(g.lo[0] == -1 && g.hi[0] == 4 && g.lo[1] == -1 && g.hi[1] == 3) << "grow";
  EXPECT_TRUE(g.nx() == 6 && g.ny() == 5) << "grow_extents";
}

TEST(test_box2d, refine_and_coarsen_roundtrip) {
  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Box2D r = b.refine(2);                // [0..7] x [0..5]
  EXPECT_TRUE(r.lo[0] == 0 && r.hi[0] == 7 && r.hi[1] == 5) << "refine";
  EXPECT_EQ(r.coarsen(2), b) << "coarsen_roundtrip";
}

TEST(test_box2d, coarsen_floors_negative_indices) {
  // coarsen avec indices negatifs : floor(-1/2) = -1, floor(2/2) = 1
  Box2D neg{{-1, -1}, {2, 2}};
  Box2D c = neg.coarsen(2);
  EXPECT_TRUE(c.lo[0] == -1 && c.hi[0] == 1) << "coarsen_floor";
}

TEST(test_box2d, intersection_clips_to_overlap) {
  Box2D a{{0, 0}, {5, 5}};
  Box2D d{{3, 3}, {9, 9}};
  Box2D in = a.intersect(d);  // [3..5] x [3..5]
  EXPECT_TRUE(in.lo[0] == 3 && in.hi[0] == 5 && in.lo[1] == 3 && in.hi[1] == 5) << "intersect";
  EXPECT_TRUE(a.intersect(Box2D{{10, 10}, {12, 12}}).empty()) << "intersect_empty";
  EXPECT_TRUE(a.contains(in) && !a.contains(d)) << "contains_box";
}
