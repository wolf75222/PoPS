// MultiFab : allocation des fabs locaux, iteration, remplissage via dispatch,
// reduction sum sur les cellules valides.

#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <vector>

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

TEST(test_multifab, set_val_fills_every_local_fab_valid_and_ghost_cells) {
  const BoxArray boxes(std::vector<Box2D>{{{-9, 4}, {-6, 7}}, {{3, -8}, {7, -5}}});
  const DistributionMapping owners(
      std::vector<int>(static_cast<std::size_t>(boxes.size()), my_rank()));
  MultiFab field(boxes, owners, /*ncomp=*/3, /*ngrow=*/2);

  field.set_val(Real(6.5));

  ASSERT_EQ(field.local_size(), boxes.size());
  for (int local = 0; local < field.local_size(); ++local) {
    const Fab2D& fab = field.fab(local);
    const Box2D grown = fab.grown_box();
    for (int component = 0; component < field.ncomp(); ++component)
      for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
        for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
          EXPECT_DOUBLE_EQ(fab(i, j, component), Real(6.5));
  }
}
