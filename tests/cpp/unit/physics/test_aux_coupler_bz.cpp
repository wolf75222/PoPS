#include <gtest/gtest.h>

#include <pops/coupling/base/aux_fill.hpp>

#include "../mesh/nd_multifab_test_utils.hpp"

#include <cstddef>
#include <vector>

using namespace pops;
using namespace pops::coupling;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
struct CoordinateSum {
  POPS_HD Real operator()(const RealVector<Dim>& coordinate) const {
    Real result = Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      result += Real(axis + 1) * coordinate[axis];
    return result;
  }
};

template <int Dim>
void expect_auxiliary_fill() {
  const Box<Dim> domain = cube<Dim>(-2, 3);
  Extent<Dim> maximum = uniform_extent<Dim>(6);
  maximum[0] = 3;
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, maximum);
  const auto ranks = one_rank_space<Dim>();
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  HostMultiFab<Dim> auxiliary(layout, distribution, Index<Dim>{}, 3, uniform_extent<Dim>(1));
  auxiliary.set_val(Real(-17));
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = Real(-axis - 1);
    upper[axis] = Real(axis + 2);
  }
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, lower, upper);

  fill_auxiliary_component(auxiliary, geometry, /*component=*/1, CoordinateSum<Dim>{});
  for (const std::size_t global : auxiliary.local_global_indices()) {
    const auto& fab = auxiliary.fab_global(global);
    const std::size_t cells = static_cast<std::size_t>(fab.grown_box().numPts());
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      const Index<Dim> index = index_from_ordinal(fab.grown_box(), ordinal);
      if (fab.box().contains(index))
        EXPECT_DOUBLE_EQ(value_at(auxiliary, global, index, 1),
                         CoordinateSum<Dim>{}(geometry.cell_center(index)));
      else
        EXPECT_EQ(value_at(auxiliary, global, index, 1), Real(-17));
      EXPECT_EQ(value_at(auxiliary, global, index, 0), Real(-17));
      EXPECT_EQ(value_at(auxiliary, global, index, 2), Real(-17));
    }
  }

  fill_auxiliary_component(auxiliary, geometry, /*component=*/2, CoordinateSum<Dim>{},
                           AuxiliaryFillRegion::allocated);
  for (const std::size_t global : auxiliary.local_global_indices()) {
    const auto& fab = auxiliary.fab_global(global);
    const std::size_t cells = static_cast<std::size_t>(fab.grown_box().numPts());
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      const Index<Dim> index = index_from_ordinal(fab.grown_box(), ordinal);
      EXPECT_DOUBLE_EQ(value_at(auxiliary, global, index, 2),
                       CoordinateSum<Dim>{}(geometry.cell_center(index)));
    }
  }
}

}  // namespace

TEST(test_aux_coupler_bz, prepared_auxiliary_providers_materialize_1d_2d_and_3d) {
  expect_auxiliary_fill<1>();
  expect_auxiliary_fill<2>();
  expect_auxiliary_fill<3>();
}

TEST(test_aux_coupler_bz, invalid_component_or_geometry_is_rejected_before_launch) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> layout(std::vector<Box<1>>{domain});
  const auto ranks = one_rank_space<1>();
  const auto distribution = Distribution<1>::replicated(layout, ranks);
  HostMultiFab<1> auxiliary(layout, distribution, Index<1>{}, 1, Extent<1>{});
  const Geometry<1> geometry = Geometry<1>::from_bounds(domain, RealVector<1>{0}, RealVector<1>{1});
  EXPECT_THROW(fill_auxiliary_component(auxiliary, geometry, -1, CoordinateSum<1>{}),
               std::out_of_range);
  EXPECT_THROW(fill_auxiliary_component(auxiliary, geometry, 1, CoordinateSum<1>{}),
               std::out_of_range);

  const Geometry<1> wrong = Geometry<1>::from_bounds(Box<1>{Index<1>{1}, Index<1>{3}},
                                                     RealVector<1>{0}, RealVector<1>{1});
  EXPECT_THROW(fill_auxiliary_component(auxiliary, wrong, 0, CoordinateSum<1>{}),
               std::invalid_argument);
}
