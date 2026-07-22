// Contrat du type PatchRange (revue, point 5 : role promu en type). Empreinte GROSSIERE
// [I0..I1]x[J0..J1] d'un patch fin sous ratio 2. Le patch doit etre exactement le raffinement
// d'une boite parente entiere; les boites decalees/demi-cellule sont refusees avant tout kernel.
// Verifie la conversion box() et la robustesse aux origines non nulles/negatives. C'est
// l'empreinte partagee par average_down,
// la couverture et l'init des registres de reflux ; son integration AMR est couverte par
// les tests de reflux (np=1/2/4 bit-identiques), ici on fige la mecanique d'index.

#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // pops::PatchRange
#include <pops/mesh/index/box2d.hpp>

using namespace pops;

TEST(test_patch_range, aligned_fine_patch_footprint) {
  // patch fin aligne grossier : cellules grossieres [2..5]^2 raffinees -> [4..11]^2.
  PatchRange pr(Box2D{{4, 4}, {11, 11}});
  EXPECT_TRUE(pr.I0 == 2 && pr.I1 == 5 && pr.J0 == 2 && pr.J1 == 5) << "empreinte_alignee";
  EXPECT_EQ(pr.box(), (Box2D{{2, 2}, {5, 5}})) << "box";
}

TEST(test_patch_range, nonzero_origin) {
  // origine non nulle : fin [6..9]x[10..13] -> grossier [3..4]x[5..6].
  PatchRange pr2(Box2D{{6, 10}, {9, 13}});
  EXPECT_TRUE(pr2.I0 == 3 && pr2.I1 == 4 && pr2.J0 == 5 && pr2.J1 == 6) << "origine_non_nulle";
}

TEST(test_patch_range, aligned_high_bound_is_the_parent_high_bound) {
  PatchRange pr3(Box2D{{0, 0}, {3, 3}});  // fin 4x4 -> grossier [0..1]^2
  EXPECT_TRUE(pr3.I1 == 1 && pr3.J1 == 1) << "borne_haute_parent";
}

TEST(test_patch_range, negative_origin_uses_floor_aligned_parent) {
  PatchRange negative(Box2D{{-8, -6}, {-1, 1}});
  EXPECT_EQ(negative.box(), (Box2D{{-4, -3}, {-1, 0}}));
}

TEST(test_patch_range, rejects_partial_parent_cells_and_empty_boxes) {
  EXPECT_THROW((void)PatchRange(Box2D{{1, 0}, {4, 3}}), std::invalid_argument);
  EXPECT_THROW((void)PatchRange(Box2D{{0, 0}, {4, 3}}), std::invalid_argument);
  EXPECT_THROW((void)PatchRange(Box2D{}), std::invalid_argument);
}
