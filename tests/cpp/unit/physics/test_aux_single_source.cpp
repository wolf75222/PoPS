// Exact compact provider packs: no physical slot prefix or globally named auxiliary state.

#include <gtest/gtest.h>

#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>

#include <array>
#include <type_traits>

using namespace pops;

namespace {

template <int Dim, int Count>
struct ProviderModel {
  using State = StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = Count;

  POPS_HD State flux(const State&, const auto&, int) const { return {}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State&, const auto&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

template <int Dim>
void check_gradient_force() {
  using Slots = ProviderSlots<2, 0, 1>;
  using Force = PotentialForceND<Dim, Slots>;
  static_assert(Force::n_providers == 3);
  static_assert(Force::gradient_slots::template slot<0>() == 2);
  static_assert(Force::gradient_slots::template slot<1>() == 0);
  static_assert(Force::gradient_slots::template slot<2>() == 1);

  ProviderValues<3> providers{};
  providers[0] = Real(20);
  providers[1] = Real(30);
  providers[2] = Real(10);
  StateVec<Dim + 2> state{};
  state[0] = Real(2);
  for (int axis = 0; axis < Dim; ++axis)
    state[axis + 1] = Real(axis + 3);

  const auto source = Force{Real(0.5)}.apply(state, providers);
  EXPECT_EQ(source[1], Real(-10));
  EXPECT_EQ(source[2], Real(-20));
  EXPECT_EQ(source[3], Real(-30));
}

struct OwnerAlpha {
  static constexpr int dimension = 2;
  static constexpr int n_providers = 1;
};
struct OwnerBeta {
  static constexpr int dimension = 2;
  static constexpr int n_providers = 1;
};

void check_cartesian_exb_slots_are_permutable() {
  using GradientSlots = ProviderSlots<1, 0>;
  using MagneticSlots = ProviderSlots<4, 2, 3>;
  using ExB = CartesianExBDriftND<2, GradientSlots, MagneticSlots>;
  static_assert(ExB::n_providers == 5);
  ProviderValues<5> providers{};
  providers[0] = Real(4);   // gradient along axis 1
  providers[1] = Real(-6);  // gradient along axis 0
  providers[2] = Real(0);   // B_y
  providers[3] = Real(2);   // B_z
  providers[4] = Real(0);   // B_x
  const ExB law{};
  const StateVec<1> state{Real(3)};
  EXPECT_EQ(law.template velocity<0>(providers), Real(-2));
  EXPECT_EQ(law.template velocity<1>(providers), Real(-3));
  EXPECT_EQ(law.template flux<0>(state, providers)[0], Real(-6));
  EXPECT_EQ(law.template flux<1>(state, providers)[0], Real(-9));
}

template <int Dim>
void check_magnetic_factory_dispatch() {
  ModelSpec spec;
  spec.source = "magnetic";
  spec.qom = 0.5;

  bool visited = false;
  detail::dispatch_source_for_dimension<Dim, Dim + 2>(spec, [&](auto source) {
    using Source = std::remove_cvref_t<decltype(source)>;
    if constexpr (std::is_same_v<Source, MagneticLorentzForceND<Dim>>) {
      visited = true;
      ProviderValues<3> providers{};
      providers[0] = Real(2);
      providers[1] = Real(3);
      providers[2] = Real(5);
      StateVec<Dim + 2> state{};
      state[0] = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        state[axis + 1] = Real(axis + 2);

      const auto result = source.apply(state, providers);
      std::array<Real, 3> momentum{};
      for (int axis = 0; axis < Dim; ++axis)
        momentum[static_cast<std::size_t>(axis)] = state[axis + 1];
      const std::array<Real, 3> expected{
          Real(0.5) * (momentum[1] * providers[2] - momentum[2] * providers[1]),
          Real(0.5) * (momentum[2] * providers[0] - momentum[0] * providers[2]),
          Real(0.5) * (momentum[0] * providers[1] - momentum[1] * providers[0])};
      for (int axis = 0; axis < Dim; ++axis)
        EXPECT_EQ(result[axis + 1], expected[static_cast<std::size_t>(axis)]);
      EXPECT_EQ(result[Dim + 1], Real(0));
    }
  });
  EXPECT_TRUE(visited);

  spec.source = "potential_magnetic";
  bool composite_visited = false;
  detail::dispatch_source_for_dimension<Dim, Dim + 2>(spec, [&](auto source) {
    using Source = std::remove_cvref_t<decltype(source)>;
    if constexpr (std::is_same_v<Source, CompositeSource<PotentialForceND<Dim>,
                                                         MagneticLorentzForceND<Dim>>>) {
      composite_visited = true;
      static_assert(Source::dimension == Dim);
      static_assert(Source::n_providers == 3);
    }
  });
  EXPECT_TRUE(composite_visited);
}

}  // namespace

static_assert(ProviderValues<0>::size == 0);
static_assert(ProviderValues<3>::size == 3);
static_assert(std::is_trivially_copyable_v<ProviderValues<0>>);
static_assert(std::is_trivially_copyable_v<ProviderValues<7>>);
static_assert(PhysicalModelFor<ProviderModel<1, 0>, 1>);
static_assert(PhysicalModelFor<ProviderModel<2, 0>, 2>);
static_assert(PhysicalModelFor<ProviderModel<3, 0>, 3>);
static_assert(PhysicalModelFor<ProviderModel<1, 4>, 1>);
static_assert(!PhysicalModelFor<ProviderModel<1, 0>, 2>);
static_assert(!std::is_same_v<BoundFluxProviders<OwnerAlpha>, BoundFluxProviders<OwnerBeta>>);
static_assert(MagneticLorentzForceND<3>::n_providers == 3);

TEST(ProviderValues, EmptyPackIsAFirstClassDeviceCarrier) {
  const ProviderValues<0> providers{};
  EXPECT_EQ(decltype(providers)::size, 0);
}

TEST(ProviderValues, QualifiedConsumersDoNotAliasBySpelling) {
  FluxProviderValues<OwnerAlpha> alpha_values{};
  FluxProviderValues<OwnerBeta> beta_values{};
  alpha_values[0] = Real(2);
  beta_values[0] = Real(7);
  const auto alpha = bind_flux_providers<OwnerAlpha>(alpha_values);
  const auto beta = bind_flux_providers<OwnerBeta>(beta_values);
  EXPECT_EQ(alpha.template provider<0>(), Real(2));
  EXPECT_EQ(beta.template provider<0>(), Real(7));
}

TEST(ProviderValues, GradientSourcesUseExplicitPermutedSlotsInEveryRank) {
  check_gradient_force<3>();
}

TEST(ProviderValues, ExBUsesExplicitPermutedGradientSlots) {
  check_cartesian_exb_slots_are_permutable();
}

TEST(ProviderValues, CartesianExBConsumesEveryVectorComponentInThreeDimensions) {
  using ExB = CartesianExBDriftND<3>;
  ProviderValues<6> providers{};
  providers[0] = Real(4);
  providers[1] = Real(-6);
  providers[2] = Real(10);
  providers[3] = Real(2);
  providers[4] = Real(3);
  providers[5] = Real(5);
  const ExB law{};

  EXPECT_EQ(law.template velocity<0>(providers), Real(60) / Real(38));
  EXPECT_EQ(law.template velocity<1>(providers), Real(0));
  EXPECT_EQ(law.template velocity<2>(providers), Real(-24) / Real(38));
}

TEST(ProviderValues, CartesianExBOneDimensionalProjectionUsesTheSameContraction) {
  using ExB = CartesianExBDriftND<1>;
  ProviderValues<4> providers{};
  providers[0] = Real(7);
  providers[1] = Real(2);
  providers[2] = Real(-3);
  providers[3] = Real(5);
  const ExB law{};

  EXPECT_EQ(law.template velocity<0>(providers), Real(0));
}

TEST(ProviderValues, LorentzUsesItsExplicitMagneticProvider) {
  ProviderValues<3> providers{};
  providers[0] = Real(2);
  providers[1] = Real(3);
  providers[2] = Real(5);
  const StateVec<5> state{Real(1), Real(3), Real(-4), Real(7), Real(11)};
  const auto source = MagneticLorentzForceND<3>{Real(0.5)}.apply(state, providers);
  EXPECT_EQ(source[1], Real(-20.5));
  EXPECT_EQ(source[2], Real(-0.5));
  EXPECT_EQ(source[3], Real(8.5));
  EXPECT_EQ(source[4], Real(0));
}

TEST(ProviderValues, MagneticFactoryDispatchAndRuntimeAreExactRanked) {
  check_magnetic_factory_dispatch<1>();
  check_magnetic_factory_dispatch<2>();
  check_magnetic_factory_dispatch<3>();
}

TEST(ProviderValues, CompositePropagatesItsExactProviderCount) {
  using Model = CompositeModel<IsothermalFluxND<2>, MagneticLorentzForceND<2>, NoElliptic>;
  static_assert(Model::n_providers == 3);
  ProviderValues<Model::n_providers> providers{};
  providers[0] = Real(4);
  providers[1] = Real(-6);
  providers[2] = Real(5);
  const Model model{};
  const StateVec<3> fluid{Real(1), Real(3), Real(-4)};
  EXPECT_EQ(model.source(fluid, providers)[1], Real(-20));
}
