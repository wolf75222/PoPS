// The out-of-plane Lorentz brick is shared algebra in an explicitly selected 2D polar basis.

#include <gtest/gtest.h>

#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>

using namespace pops;

namespace {

using PolarPotential = PotentialForceND<2>;
using PolarLorentz = MagneticLorentzForceND<2, ProviderSlots<3, 4, 5>>;
using PolarSource = CompositeSource<PolarPotential, PolarLorentz>;
using PolarModel = CompositeModel<IsothermalFluxPolar, PolarSource, ChargeDensity>;

static_assert(PolarLorentz::n_providers == 6);
static_assert(PolarSource::dimension == 2);
static_assert(PolarSource::n_providers == 6);
static_assert(PolarModel::dimension == 2);
static_assert(PolarModel::n_providers == 6);

}  // namespace

TEST(PolarLorentzSource, FormulaIsExactAndDoesNoWork) {
  const StateVec<3> state{Real(2), Real(3), Real(-4)};
  ProviderValues<6> providers{};
  providers[5] = Real(5);
  const auto source = PolarLorentz{Real(0.2)}.apply(state, providers);

  EXPECT_EQ(source[0], Real(0));
  EXPECT_NEAR(source[1], Real(-4), Real(1e-12));
  EXPECT_NEAR(source[2], Real(-3), Real(1e-12));
  const Real radial_velocity = state[1] / state[0];
  const Real azimuthal_velocity = state[2] / state[0];
  EXPECT_NEAR(source[1] * radial_velocity + source[2] * azimuthal_velocity, Real(0), Real(1e-12));
}

TEST(PolarLorentzSource, CompositeAddsEveryExactPolarGradientAndBz) {
  const StateVec<3> state{Real(2), Real(3), Real(-4)};
  ProviderValues<6> providers{};
  providers[0] = Real(0.5);
  providers[1] = Real(-0.25);
  providers[5] = Real(5);

  const PolarPotential potential{Real(2)};
  const PolarLorentz lorentz{Real(0.2)};
  const auto combined = PolarSource{potential, lorentz}.apply(state, providers);
  const auto expected = potential.apply(state, providers) + lorentz.apply(state, providers);
  for (int component = 0; component < StateVec<3>::size(); ++component)
    EXPECT_EQ(combined[component], expected[component]);
}
