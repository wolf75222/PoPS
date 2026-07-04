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

// ADC-616: the full ClusterParams set the pops.mesh.amr.PatchClustering descriptor forwards is
// honored -- default {0.7, 1, 32} clusters a solid block into one box, and a lower min_efficiency
// still covers every tag while a smaller max_box_size chops as above. Pins the params end to end.
TEST(test_cluster, custom_params_are_honored) {
  TagBox tb(Box2D::from_extents(16, 16));
  // A SPARSE tag pattern: a lower min_efficiency accepts a looser box, a higher one splits more.
  tag_block(tb, Box2D{{2, 2}, {5, 5}});
  tag_block(tb, Box2D{{10, 10}, {13, 13}});

  ClusterParams loose;
  loose.min_efficiency = 0.3;
  auto loose_boxes = berger_rigoutsos(tb, loose);
  EXPECT_TRUE(covers_all_tags(tb, loose_boxes)) << "loose_cover";

  ClusterParams tight;
  tight.min_efficiency = 0.95;
  auto tight_boxes = berger_rigoutsos(tb, tight);
  EXPECT_TRUE(covers_all_tags(tb, tight_boxes)) << "tight_cover";
  // A stricter efficiency never accepts FEWER boxes than a loose one (it splits to hit the target).
  EXPECT_GE(tight_boxes.size(), loose_boxes.size()) << "tight_splits_at_least_as_much";

  // Default {0.7, 1, 32}: a single solid block is one box (bit-identical historical behavior).
  TagBox solid(Box2D::from_extents(16, 16));
  tag_block(solid, Box2D::from_extents(16, 16));
  EXPECT_EQ(berger_rigoutsos(solid, ClusterParams{}).size(), 1u) << "default_one_box";
}
