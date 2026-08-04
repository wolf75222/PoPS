#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>

#include "nd_multifab_test_utils.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
Real arithmetic_value(const Index<Dim>& index, int component) {
  Real value = component + 2;
  for (int axis = 0; axis < Dim; ++axis)
    value += static_cast<Real>((axis + 1) * index[axis]);
  return value;
}

template <int Dim>
void expect_reductions() {
  const Box<Dim> domain = cube<Dim>(-1, 1);
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, axis_sizes<Dim>(2, 3));
  const auto distribution = Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
  HostMultiFab<Dim> field(layout, distribution, Index<Dim>{}, 2, uniform_extent<Dim>(1));
  fill_valid(field, Real{9000}, arithmetic_value<Dim>);

  for (int component = 0; component < field.ncomp(); ++component) {
    Real sum = 0;
    Real absolute_sum = 0;
    Real maximum = -std::numeric_limits<Real>::infinity();
    Real minimum = std::numeric_limits<Real>::infinity();
    Real square_sum = 0;
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(domain.numPts()); ++cell) {
      const Real value = arithmetic_value(index_from_ordinal(domain, cell), component);
      sum += value;
      absolute_sum += value < 0 ? -value : value;
      maximum = std::max(maximum, value);
      minimum = std::min(minimum, value);
      square_sum += value * value;
    }
    EXPECT_DOUBLE_EQ(reduce_sum(field, component), sum);
    EXPECT_DOUBLE_EQ(reduce_abs_sum(field, component), absolute_sum);
    EXPECT_DOUBLE_EQ(reduce_max(field, component), maximum);
    EXPECT_DOUBLE_EQ(reduce_min(field, component), minimum);
    EXPECT_DOUBLE_EQ(norm_inf(field, component),
                     std::max(maximum < 0 ? -maximum : maximum, minimum < 0 ? -minimum : minimum));
    EXPECT_DOUBLE_EQ(dot(field, field, component), square_sum);
  }

  HostMultiFab<Dim> zero = field;
  scale(zero, Real{0});
  EXPECT_DOUBLE_EQ(difference_sum_sq_all(field, zero), dot_all(field, field));

  HostMultiFab<Dim> destination = field;
  lincomb(destination, Real{2}, field, Real{-1}, field);
  EXPECT_DOUBLE_EQ(dot_all(destination, field), dot_all(field, field));
  saxpy(destination, Real{-1}, field);
  EXPECT_DOUBLE_EQ(norm_inf(destination), Real{0});
}

}  // namespace

TEST(test_reduce, arithmetic_and_collectives_are_dimension_generic) {
  expect_reductions<1>();
  expect_reductions<2>();
  expect_reductions<3>();
}

TEST(test_reduce, relative_cell_measure_uses_exact_nd_metric_identity) {
  const Box<2> domain = cube<2>(0, 2);
  const BoxArray<2> layout(std::vector<Box<2>>{domain});
  const auto distribution = Distribution<2>::replicated(layout, one_rank_space<2>());
  HostMultiFab<2> field(layout, distribution, Index<2>{}, 1, Extent<2>{});
  HostMultiFab<2> active(layout, distribution, Index<2>{}, 1, Extent<2>{});
  HostMultiFab<2> inverse(layout, distribution, Index<2>{}, 1, Extent<2>{});
  fill_valid(field, Real{0}, [](const Index<2>& index, int) {
    return static_cast<Real>(1 + index[0] + 3 * index[1]);
  });
  fill_valid(active, Real{0},
             [](const Index<2>& index, int) { return index[0] == 1 ? Real{1} : Real{0}; });
  fill_valid(inverse, Real{0}, [](const Index<2>&, int) { return Real{2}; });

  const RelativeCellMeasure<2, Kokkos::HostSpace> measure{&active, &inverse};
  Real expected = 0;
  for (int y = 0; y <= 2; ++y)
    expected += Real{2} * static_cast<Real>(2 + 3 * y);
  EXPECT_DOUBLE_EQ(reduce_sum(field, 0, measure), expected);
  EXPECT_DOUBLE_EQ(dot(field, field, 0, measure),
                   Real{2} * (Real{2} * Real{2} + Real{5} * Real{5} + Real{8} * Real{8}));
}

TEST(test_reduce, communicator_rank_mismatch_fails_closed) {
  const BoxArray<1> layout(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution = Distribution<1>::replicated(layout, ranks);
  HostMultiFab<1> field(layout, distribution, Index<1>{0}, 1, Extent<1>{});
  field.set_val(Real{3});
  EXPECT_THROW((void)reduce_sum(field), std::logic_error);
}
