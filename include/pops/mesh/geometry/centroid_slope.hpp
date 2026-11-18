/// @file
/// @brief The centred slope of the static geometry arrays (N17 DN38–DN39).
/// Solve delta X g = delta e by partial pivoting. Row a of delta X is c(K + 1_a) - c(K - 1_a).
/// Across a seam use c +/- T_a; at a wall use 2 m_f - c_K so ghost value 2 V - u gives V at m_f.
/// Zero couplings leave the diagonal quotient and its sign unchanged (DN39).

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/static_geometry.hpp>
#include <pops/mesh/index/real_vector.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

namespace pops {

enum class CentroidSlopeStatus { ok, singular, nonfinite };

/// The centroid of the neighbour of @p cell across its face of @p axis on @p side (+1 or -1).
template <int Dim>
POPS_HD RealVector<Dim> neighbour_centroid(const StaticGeometryView<Dim>& view,
                                           const Index<Dim>& cell, int axis, int side) {
  Index<Dim> neighbour = cell;
  neighbour[axis] += side;
  RealVector<Dim> result{};
  if (view.contains_cell(neighbour)) {
    for (int component = 0; component < Dim; ++component)
      result[component] = view.centroid_at(component, neighbour);
    return result;
  }
  if (view.periodic[axis]) {
    neighbour[axis] = side > 0 ? view.domain.lo[axis] : view.domain.hi[axis];
    for (int component = 0; component < Dim; ++component) {
      const Real centroid = view.centroid_at(component, neighbour);
      const Real translation = view.seam_translation[axis][component];
      result[component] = side > 0 ? centroid + translation : centroid - translation;
    }
    return result;
  }
  Index<Dim> face = cell;
  if (side > 0)
    ++face[axis];
  for (int component = 0; component < Dim; ++component)
    result[component] =
        Real(2) * view.face_centroid_at(axis, component, face) - view.centroid_at(component, cell);
  return result;
}

/// Set @p slope only when the solve succeeds for the actual centroid differences of @p cell.
template <int Dim>
POPS_HD CentroidSlopeStatus centroid_slope(const StaticGeometryView<Dim>& view,
                                           const Index<Dim>& cell,
                                           const RealVector<Dim>& differences,
                                           RealVector<Dim>& slope) {
  using Status = CentroidSlopeStatus;
  Real rows[Dim][Dim + 1];
  for (int axis = 0; axis < Dim; ++axis) {
    const RealVector<Dim> upper = neighbour_centroid(view, cell, axis, +1);
    const RealVector<Dim> lower = neighbour_centroid(view, cell, axis, -1);
    for (int component = 0; component < Dim; ++component)
      rows[axis][component] = upper[component] - lower[component];
    rows[axis][Dim] = differences[axis];
    for (Real value : rows[axis])
      if (!Kokkos::isfinite(value))
        return Status::nonfinite;
  }
  for (int column = 0; column < Dim; ++column) {
    int pivot = column;
    for (int row = column + 1; row < Dim; ++row)
      if (Kokkos::abs(rows[row][column]) > Kokkos::abs(rows[pivot][column]))
        pivot = row;
    if (rows[pivot][column] == Real(0))
      return Status::singular;
    if (pivot != column)
      for (int entry = column; entry <= Dim; ++entry) {
        const Real held = rows[column][entry];
        rows[column][entry] = rows[pivot][entry];
        rows[pivot][entry] = held;
      }
    for (int row = column + 1; row < Dim; ++row) {
      if (rows[row][column] == Real(0))
        continue;
      const Real multiplier = rows[row][column] / rows[column][column];
      for (int entry = column + 1; entry <= Dim; ++entry) {
        rows[row][entry] -= multiplier * rows[column][entry];
        if (!Kokkos::isfinite(rows[row][entry]))
          return Status::nonfinite;
      }
    }
  }
  RealVector<Dim> solution{};
  for (int row = Dim - 1; row >= 0; --row) {
    Real sum = rows[row][Dim];
    for (int entry = row + 1; entry < Dim; ++entry) {
      if (rows[row][entry] == Real(0))
        continue;
      sum -= rows[row][entry] * solution[entry];
      if (!Kokkos::isfinite(sum))
        return Status::nonfinite;
    }
    const Real value = sum / rows[row][row];
    if (!Kokkos::isfinite(value))
      return Status::nonfinite;
    solution[row] = value;
  }
  slope = solution;
  return Status::ok;
}

}  // namespace pops
