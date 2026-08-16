/// @file
/// @brief Compile-time-ranked level-set cut geometry shared by Cartesian EB consumers.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/runtime/numerical_defaults.hpp>

namespace pops::nd {

/// Normalized centre-to-neighbour crossings for one active Cartesian cell.
///
/// The same axis loop and storage shape serve dimensions 1, 2, and 3.  Geometry-specific
/// descriptors only provide signed level-set samples; they do not own another mesh or stencil.
template <int Dim>
struct CutCellFractions {
  static_assert(Dim >= 1 && Dim <= 3);
  static constexpr int dimension = Dim;

  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  Real volume_fraction = Real(0);
};

/// Linear centre-to-neighbour crossing, normalized by the axis spacing.
POPS_HD inline Real cut_cell_crossing_fraction(Real center, Real neighbor,
                                               Real theta_min = kEbCutFractionFloor) {
  if (neighbor < Real(0))
    return Real(1);
  Real theta = center / (center - neighbor);
  if (theta < theta_min)
    theta = theta_min;
  if (theta > Real(1))
    theta = Real(1);
  return theta;
}

/// Materialize all directional crossings and the retained-volume approximation from sampled phi.
template <int Dim>
POPS_HD CutCellFractions<Dim> cut_cell_fractions_from_samples(
    Real center, const RealVector<Dim>& lower_samples, const RealVector<Dim>& upper_samples,
    Real theta_min = kEbCutFractionFloor) {
  CutCellFractions<Dim> result;
  result.volume_fraction = Real(1);
  for (int axis = 0; axis < Dim; ++axis) {
    result.lower[axis] = cut_cell_crossing_fraction(center, lower_samples[axis], theta_min);
    result.upper[axis] = cut_cell_crossing_fraction(center, upper_samples[axis], theta_min);
    result.volume_fraction *= Real(0.5) * (result.lower[axis] + result.upper[axis]);
  }
  return result;
}

/// Exact-rank Shortley-Weller coefficients associated with the same sampled cut geometry.
template <int Dim>
struct ShortleyWellerStencil {
  static_assert(Dim >= 1 && Dim <= 3);
  static constexpr int dimension = Dim;

  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  Real diagonal = Real(0);
};

template <int Dim>
POPS_HD ShortleyWellerStencil<Dim> shortley_weller_stencil(const CutCellFractions<Dim>& fractions,
                                                           const RealVector<Dim>& spacing) {
  ShortleyWellerStencil<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    const Real lower_distance = fractions.lower[axis] * spacing[axis];
    const Real upper_distance = fractions.upper[axis] * spacing[axis];
    const Real span = lower_distance + upper_distance;
    result.lower[axis] = Real(2) / (lower_distance * span);
    result.upper[axis] = Real(2) / (upper_distance * span);
    result.diagonal += Real(2) / (lower_distance * upper_distance);
  }
  return result;
}

}  // namespace pops::nd
