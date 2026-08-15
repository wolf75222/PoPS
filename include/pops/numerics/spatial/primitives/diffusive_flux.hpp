/// @file
/// @brief Pointwise isotropic Fickian flux for exact-ranked finite-volume faces.

#pragma once

#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <stdexcept>

namespace pops::nd {

/// Host-side validation for the constitutive coefficient captured by a prepared operator.
template <class Model>
void require_valid_diffusivity(const Model& model) {
  if constexpr (DiffusiveModel<Model>) {
    const Real diffusivity = static_cast<Real>(model.diffusivity());
    if (!std::isfinite(diffusivity) || diffusivity < Real(0))
      throw std::invalid_argument(
          "prepared ND Fickian diffusion requires a finite non-negative diffusivity");
  }
}

/// Add `-nu grad(U)` to one positive-axis face-flux density.
///
/// The normal distance is the projection of the adjacent cell-centre displacement onto the
/// metric's oriented face-area vector.  Face measure is deliberately not applied here: the
/// canonical operator integrates the combined hyperbolic and diffusive density exactly once.
template <int Axis, int Dim, class Model, class Metric>
  requires(DiffusiveModel<Model> && PreparedMetricProvider<Dim, Metric>)
POPS_HD bool add_isotropic_fickian_flux_density(const Model& model, const Metric& metric,
                                                const FieldView<const Real, Dim>& state,
                                                const Index<Dim>& left_cell,
                                                const Index<Dim>& right_cell,
                                                typename Model::State& flux_density) {
  static_assert(Axis >= 0 && Axis < Dim, "Fickian face axis is outside the operator rank");

  const Real diffusivity = static_cast<Real>(model.diffusivity());
  if (!Kokkos::isfinite(diffusivity) || diffusivity < Real(0))
    return false;
  if (diffusivity == Real(0))
    return true;

  const auto left_center = metric.cell_center(left_cell);
  const auto right_center = metric.cell_center(right_cell);
  const auto area =
      metric.template oriented_face_area_vector<Axis, MetricFaceSide::Upper>(left_cell);
  Real area_squared = Real(0);
  Real projected_displacement = Real(0);
  for (int physical_axis = 0; physical_axis < Metric::embedding_dimension; ++physical_axis) {
    area_squared += area[physical_axis] * area[physical_axis];
    projected_displacement +=
        (right_center[physical_axis] - left_center[physical_axis]) * area[physical_axis];
  }
  if (!Kokkos::isfinite(area_squared) || !(area_squared > Real(0)) ||
      !Kokkos::isfinite(projected_displacement))
    return false;
  const Real normal_distance = projected_displacement / Kokkos::sqrt(area_squared);
  if (!Kokkos::isfinite(normal_distance) || !(normal_distance > Real(0)))
    return false;

  for (int component = 0; component < Model::n_vars; ++component) {
    const Real left = state(left_cell, component);
    const Real right = state(right_cell, component);
    if (!Kokkos::isfinite(left) || !Kokkos::isfinite(right))
      return false;
    flux_density[component] -= diffusivity * (right - left) / normal_distance;
    if (!Kokkos::isfinite(flux_density[component]))
      return false;
  }
  return true;
}

}  // namespace pops::nd
