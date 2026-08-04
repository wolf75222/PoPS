/// @file
/// @brief Canonical compile-time-ranked hyperbolic finite-volume surface.
///
/// Dimension, normal axis, reconstruction variables and Riemann policy are static properties of
/// the prepared operator.  Face storage is one `FaceField<Dim>` rather than parallel x/y fields.
/// Polar, embedded-boundary and mask providers are intentionally not fallback authorities here;
/// each must be requalified against the same ND view/metric contracts before direct composition.

#pragma once

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/numerics/spatial/nd/finite_volume.hpp>
#include <pops/numerics/spatial/nd/reconstruction.hpp>
#include <pops/numerics/spatial/nd/state_schema.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
