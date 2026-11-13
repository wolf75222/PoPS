/// @file
/// @brief The centred slope of the static geometry arrays (N17 L2b-2, design note N17, DN38, DN39).
///
/// The slope g_K of a cell K solves delta X g = delta e. Row a of delta X is the difference
/// c(K + 1_a) - c(K - 1_a) of the centroids of the two neighbours of K along the logical axis a;
/// delta e_a = e(K + 1_a) - e(K - 1_a) is the difference of their values, which the caller reads
/// from its own values and ghosts. Read at the neighbour centroids, an affine function
/// e = a + b . x gives delta e = delta X b, so g = b to rounding in every cell. The geometric part
/// is written here once, callable from a generated kernel: the Program operation of centred
/// differences calls it (ProgramContext), and the generated cell read of N15 will call it too.
///
/// A neighbour inside the domain is its stored centroid. Across the seam of a periodic axis it is
/// the centroid of the cell at the other end, translated by the seam: c + T_a across the upper
/// seam, c - T_a across the lower one. Across a physical end it is the centroid of K reflected
/// through the centroid of its end face, 2 m_f - c_K: the line through c_K and its image passes
/// through m_f halfway, so a ghost value 2 V - u states the value V at the face (DN38).
///
/// The solve is Gaussian elimination with partial pivoting on Dim <= 3 rows (DN39). On node
/// positions delta X is diagonal, its entries off the axis exact zeros: the centroids of a box
/// share their coordinates off the axis, T_a lies along a, and the centroid of an end face shares
/// them with its cell. Every multiplier is then zero and g_a = delta e_a / delta X_aa, bitwise.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/static_geometry.hpp>
#include <pops/mesh/index/real_vector.hpp>

namespace pops {

namespace centroid_slope_detail {

POPS_HD inline Real magnitude(Real value) {
  return value < Real(0) ? -value : value;
}

}  // namespace centroid_slope_detail

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

/// The slope of @p cell from the differences of the values of its neighbours,
/// @p differences[a] = e(K + 1_a) - e(K - 1_a): the solution g of delta X g = delta e.
template <int Dim>
POPS_HD RealVector<Dim> centroid_slope(const StaticGeometryView<Dim>& view, const Index<Dim>& cell,
                                       const RealVector<Dim>& differences) {
  using centroid_slope_detail::magnitude;
  Real rows[Dim][Dim + 1];
  for (int axis = 0; axis < Dim; ++axis) {
    const RealVector<Dim> upper = neighbour_centroid(view, cell, axis, +1);
    const RealVector<Dim> lower = neighbour_centroid(view, cell, axis, -1);
    for (int component = 0; component < Dim; ++component)
      rows[axis][component] = upper[component] - lower[component];
    rows[axis][Dim] = differences[axis];
  }
  for (int column = 0; column < Dim; ++column) {
    int pivot = column;
    for (int row = column + 1; row < Dim; ++row)
      if (magnitude(rows[row][column]) > magnitude(rows[pivot][column]))
        pivot = row;
    if (pivot != column)
      for (int entry = column; entry <= Dim; ++entry) {
        const Real held = rows[column][entry];
        rows[column][entry] = rows[pivot][entry];
        rows[pivot][entry] = held;
      }
    for (int row = column + 1; row < Dim; ++row) {
      const Real multiplier = rows[row][column] / rows[column][column];
      for (int entry = column; entry <= Dim; ++entry)
        rows[row][entry] -= multiplier * rows[column][entry];
    }
  }
  RealVector<Dim> slope{};
  for (int row = Dim - 1; row >= 0; --row) {
    Real sum = rows[row][Dim];
    for (int entry = row + 1; entry < Dim; ++entry)
      sum -= rows[row][entry] * slope[entry];
    slope[row] = sum / rows[row][row];
  }
  return slope;
}

}  // namespace pops
