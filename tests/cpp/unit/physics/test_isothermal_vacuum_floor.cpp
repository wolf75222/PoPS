// ADC-77: quasi-vacuum velocity bound on IsothermalFlux (and the inherited IsothermalFluxPolar).
// At ~52% of the diocotron rollup the background is evacuated (rho -> ~1e-9) and the Schur source
// stage writes O(1) momentum onto those cells, so the raw u = m/rho explodes and collapses the CFL.
// The model now computes u = m/max(rho, vacuum_floor) when vacuum_floor > 0, which bounds BOTH the
// CFL wave speed and the advective flux. This test pins:
//   (1) vacuum_floor <= 0 is bit-identical to the raw 1/rho path (feature OFF, and the one-arg
//       aggregate init defaults to OFF);
//   (2) vacuum_floor > 0 with rho < floor bounds the velocity to m/floor and keeps the wave speed
//       and flux finite, while rho itself (and the pressure cs2*rho) stay the RAW conserved value;
//   (3) vacuum_floor > 0 with rho >= floor is inactive (identical to OFF);
//   (4) the polar geometric source is floored too (finite at vacuum, identical above the floor).

#include <gtest/gtest.h>

#include <pops/physics/bricks/hyperbolic.hpp>

#include <cmath>

using namespace pops;

static const Aux kAux{};
static constexpr Real kCs2 = Real(0.5);

// Cellule quasi-vide : rho ~ 1e-9 avec une quantite de mouvement O(1) -> u brut = m/rho ~ 1e8
// (explose). Partagee par les 4 tests ci-dessous.
static StateVec<3> make_uvac() {
  StateVec<3> uvac{};
  uvac[0] = Real(1e-9);
  uvac[1] = Real(0.3);
  uvac[2] = Real(-0.2);
  return uvac;
}

// (1) plancher OFF : bit-identique au chemin brut 1/rho.
TEST(test_isothermal_vacuum_floor, OffIsBitIdenticalToRaw) {
  const StateVec<3> uvac = make_uvac();
  const IsothermalFlux off{kCs2, Real(0)};
  const Real rho = uvac[0];
  const Real vx = uvac[1] / rho, vy = uvac[2] / rho;
  const auto p = off.to_primitive(uvac);
  EXPECT_TRUE(p[0] == rho && p[1] == vx && p[2] == vy) << "off_to_primitive_raw";
  EXPECT_TRUE(off.max_wave_speed(uvac, kAux, 0) == (vx < 0 ? -vx : vx) + std::sqrt(kCs2))
      << "off_mws_raw_x";
  EXPECT_TRUE(off.flux(uvac, kAux, 0)[1] == uvac[1] * vx + kCs2 * rho) << "off_flux_raw_x";
  // L'init agregat a un seul argument doit aussi etre OFF (vacuum_floor par defaut a 0) :
  // bit-identique.
  const IsothermalFlux dflt{kCs2};
  EXPECT_TRUE(dflt.max_wave_speed(uvac, kAux, 0) == off.max_wave_speed(uvac, kAux, 0))
      << "default_is_off";
}

// (2) plancher ON, rho < plancher : la vitesse utilise max(rho, plancher) = plancher ;
// rho/pression restent bruts.
TEST(test_isothermal_vacuum_floor, OnBoundsVelocityBelowFloor) {
  const StateVec<3> uvac = make_uvac();
  const Real floor = Real(1e-3);
  const IsothermalFlux on{kCs2, floor};
  const Real vx_b = uvac[1] / floor, vy_b = uvac[2] / floor;
  const auto p = on.to_primitive(uvac);
  EXPECT_TRUE(p[1] == vx_b && p[2] == vy_b) << "on_velocity_bounded";
  EXPECT_TRUE(p[0] == uvac[0]) << "on_rho_is_raw";
  const Real mws = on.max_wave_speed(uvac, kAux, 0);
  EXPECT_TRUE(std::isfinite(mws) && mws == (vx_b < 0 ? -vx_b : vx_b) + std::sqrt(kCs2))
      << "on_mws_bounded";
  // flux : vitesse advective plafonnee, pression cs2*rho utilise toujours le rho brut.
  EXPECT_TRUE(on.flux(uvac, kAux, 0)[1] == uvac[1] * vx_b + kCs2 * uvac[0])
      << "on_flux_bounded_raw_pressure";
}

// (3) plancher ON mais rho >= plancher : inactif (identique a OFF).
TEST(test_isothermal_vacuum_floor, OnInactiveAboveFloor) {
  const Real floor = Real(1e-3);
  const IsothermalFlux on{kCs2, floor};
  const IsothermalFlux off{kCs2, Real(0)};
  StateVec<3> u{};
  u[0] = Real(2.0);
  u[1] = Real(0.6);
  u[2] = Real(0.4);
  EXPECT_TRUE(on.to_primitive(u)[1] == u[1] / u[0]) << "on_inactive_above_floor";
  EXPECT_TRUE(on.max_wave_speed(u, kAux, 1) == off.max_wave_speed(u, kAux, 1))
      << "on_eq_off_above_floor";
}

// (4) source geometrique polaire : plafonnee au vide (finie), identique au-dessus du plancher.
TEST(test_isothermal_vacuum_floor, PolarGeomSourceFlooredAtVacuum) {
  const StateVec<3> uvac = make_uvac();
  const IsothermalFluxPolar on{IsothermalFlux{kCs2, Real(1e-3)}};
  const StateVec<3> s = on.polar_geom_source(uvac, Real(1.5));
  EXPECT_TRUE(s[0] == Real(0) && std::isfinite(s[1]) && std::isfinite(s[2]))
      << "polar_geom_finite_at_vacuum";
  const IsothermalFluxPolar off{IsothermalFlux{kCs2, Real(0)}};
  StateVec<3> u{};
  u[0] = Real(2.0);
  u[1] = Real(0.6);
  u[2] = Real(0.4);
  const StateVec<3> so = off.polar_geom_source(u, Real(1.5));
  const StateVec<3> sn = on.polar_geom_source(u, Real(1.5));
  EXPECT_TRUE(so[1] == sn[1] && so[2] == sn[2]) << "polar_geom_eq_above_floor";
}
