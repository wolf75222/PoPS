// Contract of the native L1 reduction pops::reduce_abs_sum (ADC-542): the absolute-sum reduction the
// Norm(L1) diagnostic measure / P.norm1 lowers to. reduce_sum is SIGNED; reduce_abs_sum folds
// magnitudes (|f(i,j,comp)|), collectively over all ranks (all_reduce_sum).
//
// The test uses an INTEGER-valued field: Kokkos::Sum re-associates per tile, but every partial sum of
// integers is itself an integer with no rounding, so the device reduction is BIT-EXACT vs a
// lexicographic host reference (==, not a tolerance) -- the same discipline test_reduce uses for the
// exact cases. Three facets: exact-vs-host (per-component), ghost exclusion (poisoned ghosts ignored),
// and kernel parity (reduce_abs_sum(u) == reduce_sum(|u|) built by an abs copy).

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>

using namespace pops;

namespace {

// Sequential host reference: lexicographic sum of |f(.,.,comp)| over the VALID cells, reading the
// local fabs directly (never through the reducer seam). Integer-valued field => exact.
double host_abs_sum(const MultiFab& mf, int comp) {
  double s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double a = f(i, j, comp);
        s += a < 0 ? -a : a;
      }
  }
  return s;
}

// 256x256 domain in 32x32 boxes (64 fabs), shared by the cases.
BoxArray make_domain_ba(Box2D& dom_out) {
  dom_out = Box2D::from_extents(256, 256);
  return BoxArray::from_domain(dom_out, 32);
}

}  // namespace

TEST(test_reduce_abs_sum, integer_field_matches_host_exactly) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // Signed integer-valued field on two components; magnitudes folded by reduce_abs_sum are exact
  // (integer partial sums have no rounding), so we assert == against the lexicographic host sum.
  MultiFab mf(ba, dm, 2, 0);
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) {
      const Real s0 = ((i + j) & 1) ? Real(1) : Real(-1);
      a(i, j, 0) = s0 * Real(i - j);
      a(i, j, 1) = ((i * 3 - j) & 1) ? Real(-(i + 2 * j)) : Real(i + 2 * j);
    });
  }
  device_fence();
  EXPECT_EQ(reduce_abs_sum(mf, 0), host_abs_sum(mf, 0)) << "abs_sum_comp0_exact";
  EXPECT_EQ(reduce_abs_sum(mf, 1), host_abs_sum(mf, 1)) << "abs_sum_comp1_exact";
  // The magnitudes are strictly positive somewhere, so the L1 sum is > 0 (guards against a no-op).
  EXPECT_GT(reduce_abs_sum(mf, 0), 0.0) << "abs_sum_nontrivial";
}

TEST(test_reduce_abs_sum, excludes_ghost_cells) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // Grow the fabs and poison the ghosts with a huge value; the valid-box reduction must ignore them
  // (the reduction domain is mf.box(li), never the grown fab box -- the same contract as reduce_sum).
  MultiFab mf(ba, dm, 1, 2);
  mf.set_val(1.0e30);  // poisons everything, valid AND ghost
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) { a(i, j, 0) = Real(i - j); });
  }
  device_fence();
  EXPECT_EQ(reduce_abs_sum(mf, 0), host_abs_sum(mf, 0)) << "abs_sum_ghost_excluded";
}

TEST(test_reduce_abs_sum, equals_reduce_sum_of_abs_copy) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // Cross-check the AbsSumKernel against the proven SumKernel: reduce_abs_sum(u) == reduce_sum(v),
  // v = |u| built by an abs copy. Both fold the same non-negative integer values, so the per-tile
  // Kokkos::Sum reassociation is identical -> bit-exact ==.
  MultiFab u(ba, dm, 1, 0);
  MultiFab v(ba, dm, 1, 0);
  for (int li = 0; li < u.local_size(); ++li) {
    Array4 au = u.fab(li).array();
    Array4 av = v.fab(li).array();
    for_each_cell(u.box(li), [au, av] POPS_HD(int i, int j) {
      const Real w = ((i + j) & 1) ? Real(-(i + j)) : Real(i + j);
      au(i, j, 0) = w;
      av(i, j, 0) = w < 0 ? -w : w;
    });
  }
  device_fence();
  EXPECT_EQ(reduce_abs_sum(u, 0), reduce_sum(v, 0)) << "abs_sum_equals_sum_of_abs";
}

TEST(test_reduce_abs_sum, is_idempotent) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  MultiFab mf(ba, dm, 1, 0);
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) {
      a(i, j, 0) = std::sin(0.1 * i) - std::cos(0.07 * j);
    });
  }
  device_fence();
  EXPECT_EQ(reduce_abs_sum(mf, 0), reduce_abs_sum(mf, 0)) << "abs_sum_idempotent";
}
