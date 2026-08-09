/// @file
/// @brief Local exact-ranked AMR field diagnostics.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace pops::coupling::amr {

namespace diagnostics_detail {

template <int Dim>
struct IntegrateComponent {
  FieldView<const Real, Dim> field{};
  int component = 0;
  Real cell_measure = Real(0);

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return field(index, component) * cell_measure;
  }
};

template <int Dim>
struct ScaledVectorMagnitude {
  FieldView<const Real, Dim> field{};
  int first_component = 0;
  Real inverse_scale = Real(1);

  POPS_HD Real operator()(const Index<Dim>& index) const {
    Real squared = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real value = field(index, first_component + axis);
      squared += value * value;
    }
    const Real magnitude = Kokkos::sqrt(squared) * inverse_scale;
    return magnitude == magnitude && magnitude < std::numeric_limits<Real>::infinity()
               ? magnitude
               : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim, class MemorySpace>
void require_geometry_contains(const MultiFab<Dim, MemorySpace>& field,
                               const Geometry<Dim>& geometry, const char* operation) {
  for (const Box<Dim>& patch : field.layout().boxes())
    if (!geometry.domain().contains(patch))
      throw std::invalid_argument(operation);
}

inline bool finite(Real value) noexcept {
  return value == value && value != std::numeric_limits<Real>::infinity() &&
         value != -std::numeric_limits<Real>::infinity();
}

}  // namespace diagnostics_detail

/// Local integral of one component over valid cells.  MPI ownership stays explicit: callers choose
/// the reduction lane and operation after this function returns one scalar per rank.
template <int Dim, class MemorySpace>
Real local_integral(const MultiFab<Dim, MemorySpace>& field, const Geometry<Dim>& geometry,
                    int component = 0) {
  if (component < 0 || component >= field.ncomp())
    throw std::out_of_range("AMR integral component lies outside the field channel");
  diagnostics_detail::require_geometry_contains(
      field, geometry, "AMR integral geometry does not contain the exact field layout");
  Real measure = Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    measure *= geometry.spacing(axis);
  if (!diagnostics_detail::finite(measure) || !(measure > Real(0)))
    throw std::invalid_argument("AMR integral requires finite positive cell measure");

  Real result = Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result += for_each_cell_reduce_sum(
        field.box(local),
        diagnostics_detail::IntegrateComponent<Dim>{field.fab(local).view(), component, measure});
  if (!diagnostics_detail::finite(result))
    throw std::domain_error("AMR integral produced a non-finite local result");
  return result;
}

/// Local maximum magnitude of exactly Dim consecutive components, divided by a positive scale.
/// This covers vector diagnostics without embedding a two-component Euclidean norm in the API.
template <int Dim, class MemorySpace>
Real local_scaled_vector_max(const MultiFab<Dim, MemorySpace>& field, int first_component,
                             Real scale = Real(1)) {
  if (first_component < 0 || first_component > field.ncomp() ||
      Dim > field.ncomp() - first_component)
    throw std::out_of_range(
        "AMR vector diagnostic requires Dim consecutive components inside the field channel");
  if (!diagnostics_detail::finite(scale) || !(scale > Real(0)))
    throw std::invalid_argument("AMR vector diagnostic scale must be finite and positive");

  const Real inverse_scale = Real(1) / scale;
  Real result = Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result = std::max(result, for_each_cell_reduce_max(
                                  field.box(local),
                                  diagnostics_detail::ScaledVectorMagnitude<Dim>{
                                      field.fab(local).view(), first_component, inverse_scale}));
  if (!diagnostics_detail::finite(result))
    throw std::domain_error("AMR vector diagnostic observed a non-finite component");
  return result;
}

/// Conventional E x B drift diagnostic for auxiliary channels `[phi, grad(phi)...]`.
template <int Dim, class MemorySpace>
Real local_max_drift_speed(const MultiFab<Dim, MemorySpace>& auxiliary, Real magnetic_scale) {
  return local_scaled_vector_max(auxiliary, /*first_component=*/1, magnetic_scale);
}

}  // namespace pops::coupling::amr
