#include <gtest/gtest.h>
#include <pops/runtime/program/composite_integral_kernel.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/execution/for_each.hpp>

namespace {
struct ConstantIntegrand {
  pops::Real value;
  POPS_HD pops::Real operator()(const pops::Index<2>&) const { return value; }
};
}  // namespace

TEST(CompositePhysicalMeasure, VolumeFractionIsMultipliedOnceAndCoveredCellsAreExcluded) {
  using namespace pops;
  const auto box = Box<2>::from_extents(Extent<2>{2, 2});
  Fab<2> active(box, 1), coverage(box, 1), fraction(box, 1);
  active.set_val(Real{1});
  coverage.set_val(Real{1});
  fraction.set_val(Real{0.25});
  using Kernel = runtime::program::detail::CompositeIntegralKernel<2, ConstantIntegrand>;
  Kernel kernel{{Real{2}},
                std::as_const(active).view(),
                std::as_const(coverage).view(),
                std::as_const(fraction).view(),
                Real{0.125},
                true,
                true};
  auto sum = for_each_cell_reduce_finite_sum(box, kernel);
  ASSERT_TRUE(sum.finite());
  EXPECT_DOUBLE_EQ(sum.high + sum.low, 0.25);
  coverage.set_val(Real{0});
  sum = for_each_cell_reduce_finite_sum(box, kernel);
  EXPECT_DOUBLE_EQ(sum.high + sum.low, 0.);
  coverage.set_val(Real{1});
  fraction.set_val(Real{0});
  kernel.integrand.value = std::numeric_limits<Real>::quiet_NaN();
  sum = for_each_cell_reduce_finite_sum(box, kernel);
  EXPECT_TRUE(sum.finite());
  EXPECT_DOUBLE_EQ(sum.high + sum.low, 0.);
  fraction.set_val(Real{-0.01});
  EXPECT_FALSE(for_each_cell_reduce_finite_sum(box, kernel).finite());
}

TEST(CompositePhysicalMeasure, ChangingOrNonfiniteMetricIsDetectedBeforeAffinePublication) {
  using namespace pops;
  const auto box = Box<2>::from_extents(Extent<2>{2, 2});
  Fab<2> initial(box, 1), current(box, 1);
  initial.set_val(Real{0.25});
  current.set_val(Real{0.25});
  runtime::program::detail::FixedMeasureDifferenceKernel<2> kernel{std::as_const(initial).view(),
                                                                   std::as_const(current).view()};
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(box, kernel), 0.);
  current.set_val(Real{0.5});
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(box, kernel), 1.);
  current.set_val(std::numeric_limits<Real>::quiet_NaN());
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(box, kernel), 1.);
}
