// Fab2D + Array4 + for_each_cell : remplissage de l'interieur via le dispatch
// (handle Array4 capture par valeur, comme un kernel Kokkos), ghosts intacts,
// coherence du handle avec operator(), et layout composante-lente.

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>

#include <limits>

using namespace pops;

namespace {

struct NoOpCellKernel {
  POPS_HD void operator()(int, int) const {}
};

}  // namespace

TEST(test_fab2d, fill_interior_leaves_ghosts_untouched) {
  Box2D valid = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Fab2D fab(valid, /*ncomp=*/2, /*ng=*/1);
  EXPECT_TRUE(fab.grown_box().nx() == 6 && fab.grown_box().ny() == 5) << "grown";
  EXPECT_EQ(fab.size(), 6 * 5 * 2) << "alloc_size";
  EXPECT_TRUE(fab(-1, -1, 0) == 0.0 && fab(4, 3, 1) == 0.0) << "zero_init_ghost";

  // remplir l'interieur via le dispatch, handle capture par valeur
  Array4 a = fab.array();
  for_each_cell(valid, [a](int i, int j) {
    a(i, j, 0) = i + 10.0 * j;
    a(i, j, 1) = -(i + 10.0 * j);
  });

  EXPECT_TRUE(fab(0, 0, 0) == 0.0 && fab(3, 2, 0) == 23.0) << "fill_c0";
  EXPECT_EQ(fab(3, 2, 1), -23.0) << "fill_c1";
  EXPECT_TRUE(fab(-1, 0, 0) == 0.0 && fab(4, 2, 0) == 0.0 && fab(0, -1, 0) == 0.0)
      << "ghost_untouched";

  ConstArray4 ca = fab.const_array();
  EXPECT_TRUE(ca(2, 1, 0) == fab(2, 1, 0) && ca(2, 1, 1) == fab(2, 1, 1)) << "array4_matches";

  // composante-lente : le plan c=1 est un bloc contigu apres c=0,
  // de stride nx_tot * ny_tot = 6 * 5 = 30
  EXPECT_EQ(&fab(0, 0, 1) - &fab(0, 0, 0), 30) << "comp_slowest";
}

TEST(test_fab2d, set_val_fills_valid_ghosts_components_and_nonzero_origin) {
  const Box2D valid{{-7, 11}, {-4, 13}};
  Fab2D fab(valid, /*ncomp=*/3, /*ng=*/2);

  fab.set_val(Real(-3.25));

  const Box2D grown = fab.grown_box();
  for (int component = 0; component < fab.ncomp(); ++component)
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        EXPECT_DOUBLE_EQ(fab(i, j, component), Real(-3.25));
}

TEST(test_fab2d, widened_offsets_support_extreme_negative_origins) {
  constexpr int lo = std::numeric_limits<int>::min();
  Fab2D fab(Box2D{{lo, lo}, {lo + 1, lo + 1}}, /*ncomp=*/1, /*ng=*/0);

  fab(lo + 1, lo + 1) = Real(4.5);
  EXPECT_DOUBLE_EQ(fab.const_array()(lo + 1, lo + 1), Real(4.5));
}

TEST(test_fab2d, rejects_noniterable_bounds_and_oversized_allocation_before_launch) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();

  EXPECT_THROW((void)Fab2D(Box2D{{hi, 0}, {hi, 0}}, /*ncomp=*/1, /*ng=*/0), ValidationError);
  EXPECT_THROW((void)Fab2D(Box2D{{lo, 0}, {-1, 0}}, /*ncomp=*/1, /*ng=*/0), ValidationError);
  EXPECT_THROW((void)Fab2D(Box2D{{0, 0}, {hi - 1, hi - 1}}, /*ncomp=*/3, /*ng=*/0),
               ValidationError);

  // The generic iteration seam must make the same decision before Kokkos sees hi + 1.
  EXPECT_THROW(for_each_cell(Box2D{{hi, 0}, {hi, 0}}, NoOpCellKernel{}), std::overflow_error);
}
