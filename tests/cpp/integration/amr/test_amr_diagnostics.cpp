#include <gtest/gtest.h>

#include <pops/coupling/amr/amr_diagnostics.hpp>

#include "../../unit/mesh/nd_multifab_test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

using namespace pops;
using namespace pops::coupling::amr;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
RealVector<Dim> lower_bounds() {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = Real(-axis);
  return result;
}

template <int Dim>
RealVector<Dim> upper_bounds() {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = Real(2 + axis);
  return result;
}

template <int Dim>
void expect_ranked_diagnostics() {
  const Box<Dim> domain = cube<Dim>(-3, 4);
  Extent<Dim> maximum = uniform_extent<Dim>(8);
  maximum[0] = 4;
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, maximum);
  const auto ranks = one_rank_space<Dim>();
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  HostMultiFab<Dim> field(layout, distribution, Index<Dim>{}, Dim + 1, uniform_extent<Dim>(1));
  fill_valid(field, Real{-100}, [](const Index<Dim>&, int component) {
    if (component == 0)
      return Real(2.5);
    constexpr Real values[3] = {Real(3), Real(4), Real(12)};
    return values[component - 1];
  });

  const Geometry<Dim> geometry =
      Geometry<Dim>::from_bounds(domain, lower_bounds<Dim>(), upper_bounds<Dim>());
  Real volume = Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    volume *= upper_bounds<Dim>()[axis] - lower_bounds<Dim>()[axis];
  EXPECT_NEAR(local_integral(field, geometry, 0), Real(2.5) * volume, Real(1e-12));

  constexpr Real magnitudes[3] = {Real(3), Real(5), Real(13)};
  EXPECT_NEAR(local_max_drift_speed(field, Real(2)), magnitudes[Dim - 1] / Real(2), Real(1e-14));
}

}  // namespace

TEST(test_amr_diagnostics, integrals_and_vector_norms_are_exact_ranked_algorithms) {
  expect_ranked_diagnostics<1>();
  expect_ranked_diagnostics<2>();
  expect_ranked_diagnostics<3>();
}

TEST(test_amr_diagnostics, component_and_geometry_mismatches_fail_before_reduction) {
  const Box<2> domain{Index<2>{2, -4}, Index<2>{5, -1}};
  const BoxArray<2> layout(std::vector<Box<2>>{domain});
  const auto ranks = one_rank_space<2>();
  const auto distribution = Distribution<2>::replicated(layout, ranks);
  HostMultiFab<2> field(layout, distribution, Index<2>{}, 2, Extent<2>{});
  const Geometry<2> geometry =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 1});

  EXPECT_THROW((void)local_integral(field, geometry, -1), std::out_of_range);
  EXPECT_THROW((void)local_integral(field, geometry, 2), std::out_of_range);
  EXPECT_THROW((void)local_max_drift_speed(field, Real(1)), std::out_of_range);
  EXPECT_THROW((void)local_scaled_vector_max(field, 0, Real(0)), std::invalid_argument);

  const Geometry<2> wrong_geometry = Geometry<2>::from_bounds(
      Box<2>{Index<2>{3, -4}, Index<2>{5, -1}}, RealVector<2>{0, 0}, RealVector<2>{1, 1});
  EXPECT_THROW((void)local_integral(field, wrong_geometry), std::invalid_argument);
}

TEST(test_amr_diagnostics, non_finite_fields_are_never_hidden_by_max_or_sum_reducers) {
  const Box<3> domain = cube<3>(0, 1);
  const BoxArray<3> layout(std::vector<Box<3>>{domain});
  const auto ranks = one_rank_space<3>();
  const auto distribution = Distribution<3>::replicated(layout, ranks);
  HostMultiFab<3> field(layout, distribution, Index<3>{}, 4, Extent<3>{});
  fill_valid(field, Real(0), [](const Index<3>& index, int component) {
    if (component == 0)
      return index == Index<3>{0, 0, 0} ? std::numeric_limits<Real>::quiet_NaN() : Real(1);
    return component == 1 && index == Index<3>{1, 1, 1} ? std::numeric_limits<Real>::infinity()
                                                        : Real(1);
  });
  const Geometry<3> geometry =
      Geometry<3>::from_bounds(domain, RealVector<3>{0, 0, 0}, RealVector<3>{1, 1, 1});

  EXPECT_THROW((void)local_integral(field, geometry), std::domain_error);
  EXPECT_THROW((void)local_max_drift_speed(field, Real(1)), std::domain_error);
  EXPECT_THROW((void)local_max_drift_speed(field, std::numeric_limits<Real>::infinity()),
               std::invalid_argument);
}
