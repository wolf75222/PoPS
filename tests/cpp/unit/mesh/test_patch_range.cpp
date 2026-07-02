// Contrat du type PatchRange (revue, point 5 : role promu en type). Empreinte GROSSIERE
// [I0..I1]x[J0..J1] d'un patch fin sous ratio 2 : I0 = lo/2, I1 = (hi-1)/2. Verifie la
// formule exacte (qui DIFFERE de Box2D::coarsen sur la borne haute), la conversion box(),
// et la robustesse aux origines non nulles. C'est l'empreinte partagee par average_down,
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

TEST(test_patch_range, high_bound_uses_hi_minus_one_over_two) {
  // borne haute (hi-1)/2 et NON floor(hi/2) : pour hi pair les deux differeraient. Un patch
  // aligne a hi impair ; on verifie tout de meme que la formule historique est preservee.
  PatchRange pr3(Box2D{{0, 0}, {3, 3}});  // fin 4x4 -> grossier [0..1]^2
  EXPECT_TRUE(pr3.I1 == 1 && pr3.J1 == 1) << "borne_haute_hi_moins_1_sur_2";
}
