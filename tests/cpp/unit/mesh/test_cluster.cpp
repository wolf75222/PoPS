// Berger-Rigoutsos : bloc plein -> une box, deux blocs separes -> deux boxes,
// gros bloc -> chop par max_box_size, et couverture complete d'une forme en L.

#include <gtest/gtest.h>

#include <pops/amr/tagging/cluster.hpp>
#include <pops/amr/tagging/tag_box.hpp>
#include <pops/mesh/index/box2d.hpp>

#include <algorithm>
#include <vector>

using namespace pops;

namespace {

void tag_block(TagBox& tb, const Box2D& b) {
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i)
      tb(i, j) = 1;
}

// toutes les cellules taguees sont-elles couvertes par au moins une box ?
bool covers_all_tags(const TagBox& tb, const std::vector<Box2D>& boxes) {
  for (int j = tb.box.lo[1]; j <= tb.box.hi[1]; ++j)
    for (int i = tb.box.lo[0]; i <= tb.box.hi[0]; ++i)
      if (tb(i, j)) {
        bool in = false;
        for (const auto& b : boxes)
          if (b.contains(i, j)) {
            in = true;
            break;
          }
        if (!in)
          return false;
      }
  return true;
}

bool has_box(const std::vector<Box2D>& v, const Box2D& b) {
  return std::find(v.begin(), v.end(), b) != v.end();
}

}  // namespace

TEST(test_cluster, solid_block_yields_one_box) {
  TagBox tb(Box2D::from_extents(10, 10));
  tag_block(tb, Box2D{{2, 3}, {5, 6}});
  auto boxes = berger_rigoutsos(tb, ClusterParams{});
  EXPECT_EQ(boxes.size(), 1u) << "solid_count";
  EXPECT_TRUE(has_box(boxes, Box2D{{2, 3}, {5, 6}})) << "solid_box";
}

TEST(test_cluster, two_separate_blocks_yield_two_boxes) {
  TagBox tb(Box2D::from_extents(10, 10));
  tag_block(tb, Box2D{{1, 1}, {3, 3}});
  tag_block(tb, Box2D{{6, 6}, {8, 8}});
  auto boxes = berger_rigoutsos(tb, ClusterParams{});
  EXPECT_EQ(boxes.size(), 2u) << "two_count";
  EXPECT_TRUE(has_box(boxes, Box2D{{1, 1}, {3, 3}})) << "two_box1";
  EXPECT_TRUE(has_box(boxes, Box2D{{6, 6}, {8, 8}})) << "two_box2";
  EXPECT_TRUE(covers_all_tags(tb, boxes)) << "two_cover";
}

TEST(test_cluster, large_block_chopped_by_max_box_size) {
  TagBox tb(Box2D::from_extents(16, 16));
  tag_block(tb, Box2D::from_extents(16, 16));
  ClusterParams p;
  p.max_box_size = 8;
  auto boxes = berger_rigoutsos(tb, p);
  EXPECT_EQ(boxes.size(), 4u) << "chop_count";
  for (const auto& b : boxes)
    EXPECT_TRUE(b.nx() <= 8 && b.ny() <= 8) << "chop_size";
  EXPECT_TRUE(covers_all_tags(tb, boxes)) << "chop_cover";
}

TEST(test_cluster, l_shape_fully_covered_within_domain) {
  Box2D dom = Box2D::from_extents(8, 8);
  TagBox tb(dom);
  tag_block(tb, Box2D{{0, 0}, {1, 7}});  // colonne gauche
  tag_block(tb, Box2D{{0, 0}, {7, 1}});  // ligne basse
  auto boxes = berger_rigoutsos(tb, ClusterParams{});
  EXPECT_TRUE(covers_all_tags(tb, boxes)) << "L_cover";
  for (const auto& b : boxes)
    EXPECT_TRUE(dom.contains(b)) << "L_in_domain";
}
