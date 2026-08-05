// The out-of-plane Lorentz brick is shared algebra in an explicitly selected 2D polar basis.

#include <gtest/gtest.h>

#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>

using namespace pops;

namespace {

using PolarPotential = PotentialForceND<2>;
using PolarLorentz = MagneticLorentzForceND<2>;
using PolarSource = CompositeSource<PolarPotential, PolarLorentz>;
using PolarModel = CompositeModel<IsothermalFluxPolar, PolarSource, ChargeDensity>;

static_assert(PolarLorentz::planar_capability);
static_assert(PolarLorentz::n_aux == AuxComponentLayout<2>::b_z + 1);
static_assert(PolarSource::dimension == 2);
static_assert(PolarSource::n_aux == AuxComponentLayout<2>::b_z + 1);
static_assert(PolarModel::dimension == 2);
static_assert(PolarModel::n_aux == AuxComponentLayout<2>::b_z + 1);

}  // namespace

TEST(PolarLorentzSource, FormulaIsExactAndDoesNoWork) {
  const StateVec<3> state{Real(2), Real(3), Real(-4)};
  AuxState<2> auxiliary{};
  auxiliary.B_z = Real(5);
  const auto source = PolarLorentz{Real(0.2)}.apply(state, auxiliary);

  EXPECT_EQ(source[0], Real(0));
  EXPECT_EQ(source[1], Real(-4));
  EXPECT_EQ(source[2], Real(-3));
  const Real radial_velocity = state[1] / state[0];
  const Real azimuthal_velocity = state[2] / state[0];
  EXPECT_EQ(source[1] * radial_velocity + source[2] * azimuthal_velocity, Real(0));
}

TEST(PolarLorentzSource, CompositeAddsEveryExactPolarGradientAndBz) {
  const StateVec<3> state{Real(2), Real(3), Real(-4)};
  AuxState<2> auxiliary{};
  auxiliary.template gradient<0>() = Real(0.5);
  auxiliary.template gradient<1>() = Real(-0.25);
  auxiliary.B_z = Real(5);

  const PolarPotential potential{Real(2)};
  const PolarLorentz lorentz{Real(0.2)};
  const auto combined = PolarSource{potential, lorentz}.apply(state, auxiliary);
  const auto expected = potential.apply(state, auxiliary) + lorentz.apply(state, auxiliary);
  for (int component = 0; component < StateVec<3>::size(); ++component)
    EXPECT_EQ(combined[component], expected[component]);
}
