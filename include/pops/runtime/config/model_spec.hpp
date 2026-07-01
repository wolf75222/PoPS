#pragma once

#include <pops/runtime/numerical_defaults.hpp>

#include <string>

/// @file
/// @brief Flat specification of a model: chosen bricks plus their parameters.
///
/// Describes a PhysicalModel as a composition of generic bricks (transport, source,
/// elliptic right-hand side) and carries their parameters. No named scenario: the
/// application (adc_cases) is the one that names a composition of these bricks.
/// Flat type (POD) to cross the bindings without friction.

namespace pops {

/// Brick composition of a block plus parameters. The fields are only read by the relevant
/// brick (see dispatch_model in model_factory.hpp).
///
/// CONTRACT (ADC-290): a ModelSpec carries NO silent physics default. `transport` and `elliptic`
/// are UNSET by default (empty string) and MUST be chosen explicitly; an unset tag is rejected with
/// a clear message by validate_model_spec (model_factory.hpp) instead of silently selecting a model
/// (the old defaults `transport="compressible"` / `elliptic="charge"` made a default-constructed
/// ModelSpec mean Euler + Poisson-charge by accident). `source` keeps the only default, "none" --
/// the EXPLICIT, neutral "no source" choice, not a physics selection. The numeric parameters keep
/// their defaults: each is read only once its brick has been chosen by a tag, so it can never inject
/// physics on its own. Historical shortcuts live at the Python edge (pops.Model(...)), which always
/// sets the three tags. See docs/adr/ADR-0001-genericity-contracts.md.
struct ModelSpec {
  std::string transport;        ///< REQUIRED (unset): "exb" | "compressible" | "isothermal"
  std::string source = "none";  ///< "none" (default, neutral: no force) | "potential" | "gravity"
      ///< | "magnetic"/"lorentz" | "potential_magnetic"/"potential_lorentz"
  std::string elliptic;  ///< REQUIRED (unset): "charge" | "background" | "gravity"

  double B0 = static_cast<double>(kPhysicalDefaultB0);        ///< ExBVelocity: magnetic field
  double gamma = static_cast<double>(kPhysicalDefaultGamma);  ///< CompressibleFlux: adiabatic index
  double cs2 =
      static_cast<double>(kPhysicalDefaultFluidStateCs2);  ///< IsothermalFlux: sound speed squared
  double vacuum_floor = static_cast<double>(
      kPhysicalDefaultVacuumFloor);  ///< IsothermalFlux: quasi-vacuum density floor
                                     ///< (ADC-77). Set from pops.FluidState(vacuum_floor=...)
  ///< INDEPENDENT of the spatial positivity_floor (deliberately decoupled --
  ///< coupling them shifts the CFL dt of existing positivity_floor runs).
  ///< 0 = off (bit-identical)
  double qom =
      static_cast<double>(kPhysicalDefaultQOverM);  ///< PotentialForce / MagneticLorentzForce: q/m
  double q = static_cast<double>(kPhysicalDefaultChargeQ);       ///< ChargeDensity: charge q
  double alpha = static_cast<double>(kPhysicalDefaultAlpha);     ///< BackgroundDensity coupling
  double n0 = static_cast<double>(kPhysicalDefaultBackgroundN0);  ///< neutralizing background
  double sign = static_cast<double>(kPhysicalDefaultGravitySign);  ///< +1 gravity, -1 electrostatic
  double four_pi_G = static_cast<double>(kPhysicalDefaultFourPiG);  ///< coupling intensity
  double rho0 = static_cast<double>(kPhysicalDefaultGravityRho0);   ///< GravityCoupling background
};

}  // namespace pops
