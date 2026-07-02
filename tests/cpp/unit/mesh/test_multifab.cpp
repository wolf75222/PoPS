// MultiFab : allocation des fabs locaux, iteration, remplissage via dispatch,
// reduction sum sur les cellules valides.

#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>

using namespace pops;

// Pipeline stateful : le meme MultiFab est rempli puis interroge en plusieurs etapes.
TEST(test_multifab, allocate_fill_and_reduce) {
  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);  // 4 boxes
  DistributionMapping dm(ba.size(), n_ranks());
  MultiFab mf(ba, dm, /*ncomp=*/1, /*ngrow=*/1);

  // rang unique : tous les fabs sont locaux
  EXPECT_EQ(mf.local_size(), 4) << "local_size";
  EXPECT_TRUE(mf.ncomp() == 1 && mf.n_grow() == 1) << "meta";

  mf.set_val(2.0);
  EXPECT_LT(std::fabs(sum(mf) - 2.0 * 64), 1e-12) << "sum_constant";

  // remplir chaque cellule valide avec une valeur globale f(i, j) = i + 100 j
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a](int i, int j) { a(i, j, 0) = i + 100.0 * j; });
  }

  // somme sur tout le domaine : sum_{i,j} (i + 100 j), i,j dans [0..7]
  // = 64*(sum i)/... -> sum_i i = 28 (x8 lignes) + 100 * sum_j j (x8 colonnes)
  Real expected = 0;
  for (int j = 0; j < 8; ++j)
    for (int i = 0; i < 8; ++i)
      expected += i + 100.0 * j;
  EXPECT_LT(std::fabs(sum(mf) - expected), 1e-9) << "sum_field";

  // verifier une cellule precise via la box qui la contient
  bool found = false;
  for (int li = 0; li < mf.local_size(); ++li) {
    if (mf.box(li).contains(5, 6)) {
      found = true;
      EXPECT_EQ(mf.fab(li)(5, 6, 0), 5 + 600.0) << "cell_value";
    }
  }
  EXPECT_TRUE(found) << "cell_located";
}
