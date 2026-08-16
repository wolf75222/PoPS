// Polar physics is an explicit 2D capability; Cartesian rank selection never manufactures it.

#include <gtest/gtest.h>

#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>

#include <type_traits>

using namespace pops;

namespace {

using PolarFluid = CompositeModel<IsothermalFluxPolar, NoSource, BackgroundDensity>;

static_assert(ExBVelocityPolar::dimension == 2);
static_assert(ExBVelocityPolar::planar_polar_capability);
static_assert(ExBVelocityPolar::n_providers == 2);
static_assert(IsothermalFluxPolar::dimension == 2);
static_assert(IsothermalFluxPolar::planar_polar_capability);
static_assert(PolarFluid::dimension == 2);
// This composite has no ExB transport or force brick: all three selected consumers are provider-free.
static_assert(PolarFluid::n_providers == 0);

}  // namespace

TEST(PolarFluidTransport, ExBUsesTheTwoPhysicalPolarGradientComponents) {
  ProviderValues<2> providers{};
  providers[0] = Real(6);
  providers[1] = Real(-4);
  const ExBVelocityPolar drift{Real(2)};
  const StateVec<1> density{Real(3)};

  EXPECT_EQ(drift.template velocity<0>(providers), Real(2));
  EXPECT_EQ(drift.template velocity<1>(providers), Real(3));
  EXPECT_EQ(drift.template flux<0>(density, providers)[0], Real(6));
  EXPECT_EQ(drift.template flux<1>(density, providers)[0], Real(9));
}

TEST(PolarFluidTransport, IsothermalCurvatureSourceRemainsPlanarAndPointwise) {
  IsothermalFluxPolar fluid{};
  fluid.cs2 = Real(0.5);
  const StateVec<3> state{Real(2), Real(4), Real(6)};
  const auto source = fluid.polar_geom_source(state, Real(4));

  EXPECT_EQ(source[0], Real(0));
  EXPECT_EQ(source[1], Real(19) / Real(4));
  EXPECT_EQ(source[2], Real(-3));
}
