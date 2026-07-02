// AmrHierarchy : construction du niveau grossier, ajout d'un niveau fin imbrique,
// et interpolation grossier->fin sur la region raffinee.

#include <gtest/gtest.h>

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
                 /*ref_ratio=*/2);

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
  EXPECT_THROW(h.install_level(0, fba, fine_data()), std::out_of_range)
      << "install_level_rejects_level0";
  EXPECT_THROW(h.install_level(h.num_levels() + 1, fba, fine_data()), std::out_of_range)
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
