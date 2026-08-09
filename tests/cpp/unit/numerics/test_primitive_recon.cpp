// Variables primitives (rho, u, p) : conversions cons <-> prim sur les briques de
// transport, leur usage dans max_wave_speed, et le concept HasPrimitiveVars satisfait par
// les modeles composes. La reconstruction primitive de l'operateur spatial s'appuie sur ces
// conversions ; on verifie ici leur exactitude (round-trip) et la centralisation du calcul
// des variables primitives.

#include <gtest/gtest.h>

#include <pops/core/model/physical_model.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>

#include <cmath>

using namespace pops;

namespace {
bool close(Real a, Real b, Real tol = 1e-12) {
  const Real d = a - b;
  return (d < 0 ? -d : d) < tol;
}
}  // namespace

TEST(test_primitive_recon, euler_round_trip_and_wave_speed) {
  const auto check = []<int Dim>() {
    EulerND<Dim> model;
    model.gamma = Real(1.4);
    typename EulerND<Dim>::Prim primitive{};
    primitive[0] = Real(1.2);
    for (int axis = 0; axis < Dim; ++axis)
      primitive[EulerND<Dim>::momentum_component(axis)] = Real(0.3) * Real(axis + 1);
    primitive[EulerND<Dim>::energy_component] = Real(1.1);

    const auto conservative = model.to_conservative(primitive);
    const auto restored = model.to_primitive(conservative);
    for (int component = 0; component < EulerND<Dim>::n_vars; ++component)
      EXPECT_TRUE(close(primitive[component], restored[component]));

    const AuxState<Dim> providers{};
    const Real sound_speed =
        std::sqrt(model.gamma * primitive[EulerND<Dim>::energy_component] / primitive[0]);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real velocity = primitive[EulerND<Dim>::momentum_component(axis)];
      EXPECT_TRUE(close(model.max_wave_speed(conservative, providers, axis),
                        std::fabs(velocity) + sound_speed));
    }
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}

TEST(test_primitive_recon, isothermal_round_trip) {
  const auto check = []<int Dim>() {
    IsothermalFluxND<Dim> model;
    model.cs2 = Real(0.5);
    typename IsothermalFluxND<Dim>::Prim primitive{};
    primitive[0] = Real(2);
    for (int axis = 0; axis < Dim; ++axis)
      primitive[IsothermalFluxND<Dim>::momentum_component(axis)] = Real(0.4) * Real(axis + 1);
    const auto conservative = model.to_conservative(primitive);
    const auto restored = model.to_primitive(conservative);
    for (int component = 0; component < IsothermalFluxND<Dim>::n_vars; ++component)
      EXPECT_TRUE(close(primitive[component], restored[component]));
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}

TEST(test_primitive_recon, scalar_exb_conversions_are_identity) {
  ExBVelocity exb;
  StateVec<1> n{};
  n[0] = 0.7;
  EXPECT_TRUE(close(exb.to_primitive(n)[0], 0.7) && close(exb.to_conservative(n)[0], 0.7))
      << "scalaire : prim == cons (identite)";
}

TEST(test_primitive_recon, composed_models_expose_primitive_vars) {
  using Mc = CompositeModel<CompressibleFlux, NoSource, ChargeDensity>;
  using Mi = CompositeModel<IsothermalFlux, NoSource, ChargeDensity>;
  using Ms = CompositeModel<ExBVelocity, NoSource, BackgroundDensity>;
  static_assert(HasPrimitiveVars<Mc>, "compose Euler doit exposer les variables primitives");
  static_assert(HasPrimitiveVars<Mi>, "compose isotherme doit exposer les variables primitives");
  static_assert(HasPrimitiveVars<Ms>, "compose scalaire : conversions identite");
  SUCCEED() << "HasPrimitiveVars : Euler / isotherme / scalaire (static_assert)";
}
