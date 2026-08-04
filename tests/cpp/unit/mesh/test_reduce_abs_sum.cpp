#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>

#include "nd_multifab_test_utils.hpp"

#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
void expect_ghost_exclusion() {
  const Box<Dim> domain = cube<Dim>(0, 2);
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution = Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
  HostMultiFab<Dim> field(layout, distribution, Index<Dim>{}, 1, uniform_extent<Dim>(2));
  fill_valid(field, Real{1.0e12}, [](const Index<Dim>& index, int) {
    int parity = 0;
    Real magnitude = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      parity += index[axis];
      magnitude += static_cast<Real>((axis + 1) * index[axis]);
    }
    return parity % 2 == 0 ? magnitude : -magnitude;
  });

  Real expected = 0;
  for (std::size_t cell = 0; cell < static_cast<std::size_t>(domain.numPts()); ++cell) {
    const Index<Dim> index = index_from_ordinal(domain, cell);
    int parity = 0;
    Real magnitude = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      parity += index[axis];
      magnitude += static_cast<Real>((axis + 1) * index[axis]);
    }
    const Real value = parity % 2 == 0 ? magnitude : -magnitude;
    expected += value < 0 ? -value : value;
  }
  EXPECT_DOUBLE_EQ(reduce_abs_sum_local(field), expected);
  EXPECT_DOUBLE_EQ(reduce_abs_sum(field), expected);
}

}  // namespace

TEST(test_reduce_abs_sum, valid_cell_contract_holds_in_1d_2d_and_3d) {
  expect_ghost_exclusion<1>();
  expect_ghost_exclusion<2>();
  expect_ghost_exclusion<3>();
}

TEST(test_reduce_abs_sum, active_mask_keeps_inactive_values_out_of_measure) {
  const BoxArray<1> layout(std::vector<Box<1>>{Box<1>{Index<1>{-2}, Index<1>{2}}});
  const auto distribution = Distribution<1>::replicated(layout, one_rank_space<1>());
  HostMultiFab<1> field(layout, distribution, Index<1>{}, 1, Extent<1>{});
  HostMultiFab<1> active(layout, distribution, Index<1>{}, 1, Extent<1>{});
  fill_valid(field, Real{0},
             [](const Index<1>& index, int) { return static_cast<Real>(index[0] * 3); });
  fill_valid(active, Real{0},
             [](const Index<1>& index, int) { return index[0] >= 0 ? Real{1} : Real{0}; });
  const RelativeCellMeasure<1, Kokkos::HostSpace> measure{&active, nullptr};
  EXPECT_DOUBLE_EQ(reduce_abs_sum(field, 0, measure), Real{9});
}
