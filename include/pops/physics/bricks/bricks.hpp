#pragma once

/// @file
/// @brief Umbrella for composable GENERIC physics bricks (compat). Re-exports the bricks by
///        category plus the CompositeModel that assembles them. The core knows NO named scenario:
///        a scenario is a COMPOSITION of bricks, chosen from the application (adc_cases).
///
/// Split by category (to match the target tree physics/{hyperbolic,source,elliptic,...}):
///   - physics/hyperbolic.hpp: CartesianExBDrift, CompressibleFlux (= Euler), IsothermalFlux;
///   - physics/source.hpp:     NoSource, PotentialForce, GravityForce;
///   - physics/elliptic.hpp:   NoElliptic, ChargeDensity, BackgroundDensity, GravityCoupling;
///   - physics/composite.hpp:  CompositeModel<Hyperbolic, Source, Elliptic>.
/// Including this file gives EVERYTHING (as before); including a precise category is now possible.
///
/// This umbrella re-exports only production/generic bricks. Retired validation/reference models
/// are neither kept here nor aliased through another include path.

#include <pops/physics/composition/composite.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
