/// @file
/// @brief Explicit planar-polar capability specialization of the canonical ND operator.

#pragma once

#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>

#include <utility>

namespace pops::nd {

/// Prepare the intrinsically two-dimensional planar-polar transport capability.
///
/// Polar geometry is a metric specialization, not another state or face-storage authority.  The
/// returned operator therefore reconstructs `FieldView<const Real, 2>` through `Index<2>` and uses
/// the same axis-static `FaceField<2>`/finite-volume pipeline as every other prepared map.
template <class Model, class Reconstruction = NoSlope, class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires ConservationLaw<2, Model>
auto prepare_polar_operator(const Box<2>& domain, Model model, PlanarPolarCoordinateMap map,
                            Reconstruction reconstruction = {}, NumericalFlux numerical_flux = {}) {
  auto metric = prepare_metric_provider(domain, std::move(map));
  return prepare_cartesian_operator<2, Model, decltype(metric), Reconstruction, NumericalFlux,
                                    Variables>(
      std::move(model), std::move(metric), std::move(reconstruction), std::move(numerical_flux));
}

}  // namespace pops::nd
