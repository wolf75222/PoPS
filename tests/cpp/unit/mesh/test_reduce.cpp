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
  Extent<Dim> max_grid_size{};
  for (int axis = 0; axis < Dim; ++axis)
    max_grid_size[axis] = axis == 0 ? 2 : 3;
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, max_grid_size);
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

template <int Dim>
void expect_masked_local_max() {
  const Box<Dim> domain = cube<Dim>(0, 1);
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution = Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
  HostMultiFab<Dim> field(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});
  HostMultiFab<Dim> active(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});

  fill_valid(field, Real{0},
             [](const Index<Dim>& index, int) { return index[0] == 0 ? Real{4} : Real{99}; });
  fill_valid(active, Real{0},
             [](const Index<Dim>& index, int) { return index[0] == 0 ? Real{1} : Real{0}; });
  const MaskedMaxLocalResult finite = reduce_masked_max_local(field, 0, &active);
  EXPECT_EQ(finite.maximum, Real{4});
  EXPECT_TRUE(finite.has_active);
  EXPECT_FALSE(finite.has_invalid);

  fill_valid(field, Real{0}, [](const Index<Dim>& index, int) {
    return index[0] == 0 ? Real{4} : std::numeric_limits<Real>::quiet_NaN();
  });
  const MaskedMaxLocalResult inactive_nan = reduce_masked_max_local(field, 0, &active);
  EXPECT_EQ(inactive_nan.maximum, Real{4});
  EXPECT_FALSE(inactive_nan.has_invalid);

  for (const Real invalid :
       {std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity(), Real{-1}}) {
    fill_valid(field, Real{0}, [invalid](const Index<Dim>& index, int) {
      return index[0] == 0 ? invalid : Real{0};
    });
    const MaskedMaxLocalResult active_invalid = reduce_masked_max_local(field, 0, &active);
    EXPECT_TRUE(active_invalid.has_active);
    EXPECT_TRUE(active_invalid.has_invalid);
  }

  active.set_val(Real{0});
  fill_valid(field, Real{0},
             [](const Index<Dim>&, int) { return std::numeric_limits<Real>::quiet_NaN(); });
  const MaskedMaxLocalResult empty = reduce_masked_max_local(field, 0, &active);
  EXPECT_EQ(empty.maximum, -std::numeric_limits<Real>::infinity());
  EXPECT_FALSE(empty.has_active);
  EXPECT_FALSE(empty.has_invalid);

  HostMultiFab<Dim> two_component_mask(layout, distribution, Index<Dim>{}, 2, Extent<Dim>{});
  EXPECT_THROW((void)reduce_masked_max_local(field, 0, &two_component_mask), std::invalid_argument);
}

template <int Dim>
void expect_active_local_extrema() {
  const Box<Dim> domain = cube<Dim>(0, 2);
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto distribution = Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
  HostMultiFab<Dim> field(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});
  HostMultiFab<Dim> active(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});

  fill_valid(field, Real{0}, [](const Index<Dim>& index, int) {
    if (index[0] == 0)
      return Real{4};
    return index[0] == 1 ? Real{99} : Real{-99};
  });
  fill_valid(active, Real{0},
             [](const Index<Dim>& index, int) { return index[0] == 0 ? Real{1} : Real{0}; });
  EXPECT_EQ(reduce_active_max_local(field, 0, &active), Real{4});
  EXPECT_EQ(reduce_active_min_local(field, 0, &active), Real{4});
  EXPECT_EQ(reduce_active_norm_inf_local(field, 0, &active), Real{4});

  fill_valid(field, Real{0}, [](const Index<Dim>& index, int) {
    if (index[0] == 0)
      return Real{4};
    return index[0] == 1 ? Real{-3} : Real{99};
  });
  fill_valid(active, Real{0},
             [](const Index<Dim>& index, int) { return index[0] <= 1 ? Real{1} : Real{0}; });
  EXPECT_EQ(reduce_active_max_local(field, 0, &active), Real{4});
  EXPECT_EQ(reduce_active_min_local(field, 0, &active), Real{-3});
  EXPECT_EQ(reduce_active_norm_inf_local(field, 0, &active), Real{4});

  active.set_val(Real{0});
  const Real all_inactive_max = reduce_active_max_local(field, 0, &active);
  const Real all_inactive_min = reduce_active_min_local(field, 0, &active);
  EXPECT_EQ(all_inactive_max, -std::numeric_limits<Real>::infinity());
  EXPECT_EQ(all_inactive_min, std::numeric_limits<Real>::infinity());
  EXPECT_EQ(reduce_active_norm_inf_local(field, 0, &active), Real{0});
}

}  // namespace

TEST(test_reduce, arithmetic_and_collectives_are_dimension_generic) {
  expect_reductions<1>();
  expect_reductions<2>();
  expect_reductions<3>();
}

TEST(test_reduce, masked_local_max_ignores_inactive_values_in_every_dimension) {
  expect_masked_local_max<1>();
  expect_masked_local_max<2>();
  expect_masked_local_max<3>();
}

TEST(test_reduce, active_local_extrema_use_raw_active_domain_identity_in_every_dimension) {
  expect_active_local_extrema<1>();
  expect_active_local_extrema<2>();
  expect_active_local_extrema<3>();
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
