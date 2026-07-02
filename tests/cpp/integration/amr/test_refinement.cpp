// Operateurs de transfert AMR : average_down (moyenne conservative fin->grossier),
// interpolate (injection grossier->fin), identite average_down(interpolate(.)) et
// conservation de la somme.

#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/layout/refinement.hpp>

#include <cmath>

using namespace pops;

namespace {
bool close(Real x, Real y) {
  return std::fabs(x - y) < 1e-9;
}
}  // namespace

TEST(test_refinement, Runs) {
  Box2D cdom = Box2D::from_extents(4, 4);  // grossier [0..3]
  Box2D fdom = Box2D::from_extents(8, 8);  // fin [0..7] (refine 2)
  BoxArray cba = BoxArray::from_domain(cdom, 4);
  BoxArray fba = BoxArray::from_domain(fdom, 8);

  // --- average_down d'un champ lineaire fin ---
  {
    MultiFab fine(fba, DistributionMapping(fba.size(), n_ranks()), 1, 0);
    MultiFab coarse(cba, DistributionMapping(cba.size(), n_ranks()), 1, 0);
    Array4 a = fine.fab(0).array();
    for_each_cell(fdom, [a](int i, int j) { a(i, j, 0) = i + 100.0 * j; });
    average_down(fine, coarse, 2);
    EXPECT_TRUE(close(coarse.fab(0)(0, 0, 0), 50.5)) << "avg_00";
    EXPECT_TRUE(close(coarse.fab(0)(1, 1, 0), 252.5)) << "avg_11";
    EXPECT_TRUE(close(coarse.fab(0)(3, 3, 0), 656.5)) << "avg_33";
  }

  // --- interpolate puis average_down = identite ---
  {
    MultiFab coarse(cba, DistributionMapping(cba.size(), n_ranks()), 1, 0);
    MultiFab fine(fba, DistributionMapping(fba.size(), n_ranks()), 1, 0);
    MultiFab coarse2(cba, DistributionMapping(cba.size(), n_ranks()), 1, 0);
    Array4 ac = coarse.fab(0).array();
    for_each_cell(cdom, [ac](int I, int J) { ac(I, J, 0) = I + 100.0 * J; });

    interpolate(coarse, fine, 2);
    EXPECT_TRUE(close(fine.fab(0)(2, 2, 0), 101.0)) << "inj_22";  // gc(1,1)
    EXPECT_TRUE(close(fine.fab(0)(3, 2, 0), 101.0)) << "inj_32";  // meme cellule grossiere
    EXPECT_TRUE(close(fine.fab(0)(4, 6, 0), 302.0)) << "inj_46";  // gc(2,3)

    average_down(fine, coarse2, 2);
    EXPECT_TRUE(close(coarse2.fab(0)(1, 1, 0), 101.0)) << "id_11";
    EXPECT_TRUE(close(coarse2.fab(0)(2, 3, 0), 302.0)) << "id_23";

    // conservation : injection replique chaque grossiere en 4 fines
    Real sc = sum(coarse), sf = sum(fine), sc2 = sum(coarse2);
    EXPECT_TRUE(close(sf, 4.0 * sc)) << "conserv_inject";
    EXPECT_TRUE(close(sc2, sc)) << "conserv_roundtrip";
  }
}
