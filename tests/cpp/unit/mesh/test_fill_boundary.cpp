// fill_boundary : echange de halos intra-niveau.
//   - 4 boxes, non periodique : aretes et coin interieurs remplis depuis les
//     voisins, ghosts hors domaine laisses a 0.
//   - 1 box, periodique : wrapping correct des deux cotes et des coins.
//   - la somme sur les cellules valides ne change pas (on n'ecrit que les ghosts).

#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>

using namespace pops;

namespace {

// champ global continu a travers les frontieres de boxes
double g(int i, int j) {
  return i + 100.0 * j;
}

void fill_valid(MultiFab& mf) {
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a](int i, int j) { a(i, j, 0) = g(i, j); });
  }
}

// recupere le fab local dont le coin bas vaut (lo0, lo1)
const Fab2D& fab_with_lo(const MultiFab& mf, int lo0, int lo1) {
  for (int li = 0; li < mf.local_size(); ++li)
    if (mf.box(li).lo[0] == lo0 && mf.box(li).lo[1] == lo1)
      return mf.fab(li);
  return mf.fab(0);
}

}  // namespace

TEST(test_fill_boundary, four_boxes_nonperiodic) {
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);  // boxes 4x4
  MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, 1);
  fill_valid(mf);
  Real s_before = sum(mf);

  fill_boundary(mf, dom, Periodicity{false, false});

  const Fab2D& b0 = fab_with_lo(mf, 0, 0);           // box [0..3]x[0..3]
  EXPECT_EQ(b0(4, 2, 0), g(4, 2)) << "edge_right";   // depuis le voisin x
  EXPECT_EQ(b0(2, 4, 0), g(2, 4)) << "edge_top";     // depuis le voisin y
  EXPECT_EQ(b0(4, 4, 0), g(4, 4)) << "corner_diag";  // depuis le voisin diagonal
  EXPECT_EQ(b0(-1, 2, 0), 0.0) << "phys_left_zero";  // bord physique : intact
  EXPECT_EQ(b0(2, -1, 0), 0.0) << "phys_bottom_zero";
  EXPECT_EQ(b0(-1, -1, 0), 0.0) << "phys_corner_zero";

  EXPECT_LT(std::fabs(sum(mf) - s_before), 1e-12) << "sum_unchanged";
}

TEST(test_fill_boundary, single_box_periodic_wraps) {
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 8);  // une seule box [0..7]x[0..7]
  ASSERT_EQ(ba.size(), 1) << "single_box";
  MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, 1);
  fill_valid(mf);

  fill_boundary(mf, dom, Periodicity{true, true});

  const Fab2D& f = mf.fab(0);
  EXPECT_EQ(f(-1, 3, 0), g(7, 3)) << "wrap_left";    // i=-1 <- i=7
  EXPECT_EQ(f(8, 3, 0), g(0, 3)) << "wrap_right";    // i=8  <- i=0
  EXPECT_EQ(f(3, -1, 0), g(3, 7)) << "wrap_bottom";  // j=-1 <- j=7
  EXPECT_EQ(f(3, 8, 0), g(3, 0)) << "wrap_top";      // j=8  <- j=0
  EXPECT_EQ(f(-1, -1, 0), g(7, 7)) << "wrap_corner";
  EXPECT_EQ(f(8, 8, 0), g(0, 0)) << "wrap_corner2";
}

TEST(test_fill_boundary, periodic_halo_deeper_than_one_cell_domain) {
  constexpr int ng = 5;
  const Box2D dom = Box2D::from_extents(1, 1);
  const BoxArray ba = BoxArray::from_domain(dom, 1);
  ASSERT_EQ(ba.size(), 1);
  MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, ng);
  mf.fab(0)(0, 0, 0) = 17.25;

  fill_boundary(mf, dom, Periodicity{true, true});

  const Fab2D& f = mf.fab(0);
  const Box2D grown = dom.grow(ng);
  for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
    for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
      EXPECT_EQ(f(i, j, 0), 17.25) << "deep periodic ghost at (" << i << ", " << j << ")";
}

TEST(test_fill_boundary, deep_periodic_wrap_preserves_nonzero_index_origin) {
  constexpr int ng = 4;
  const Box2D dom{{-7, 11}, {-7, 11}};
  const BoxArray ba(std::vector<Box2D>{dom});
  MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, ng);
  mf.fab(0)(dom.lo[0], dom.lo[1], 0) = -3.75;

  fill_boundary(mf, dom, Periodicity{true, true});

  const Box2D grown = dom.grow(ng);
  for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
    for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
      EXPECT_EQ(mf.fab(0)(i, j, 0), -3.75)
          << "deep periodic ghost at nonzero-origin index (" << i << ", " << j << ")";
}
