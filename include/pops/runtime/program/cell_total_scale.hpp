#pragma once

/// The decode of a cell total and the encode of its rate (G03 lot R-b, ROM-21).
///
/// A total U_K = ∫_K u holds c = 1/μ(K) in contract @2. Its lowering decodes ū = U/μ(K) into a
/// stage field, runs the rate of the average on ū through today's operators (ghosts and boundary
/// laws filled on ū), and encodes the rate of U as μ(K) times the rate of ū. On a static
/// configuration this is the identity dU_K/dt = μ(K) dū_K/dt. The measure is the static geometry's
/// cell measure on a mapped layout, the product of the spacings otherwise.
///
/// Transitional native route, registered in plan/routes_transitoires.md: it goes when N10 moves
/// the face pass into generated regions and the scale becomes the Riesz `map` row of N16 §5.1.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/static_geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>

namespace pops::runtime::program {

template <int Dim>
struct CellTotalScaleKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  StaticGeometryView<Dim> arrays{};
  bool mapped = false;
  Real uniform_measure = Real(1);
  bool divide = false;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& cell) const {
    const Real measure = mapped ? arrays.measure(cell) : uniform_measure;
    const Real value = source(cell, component);
    destination(cell, component) = divide ? value / measure : value * measure;
  }
};

}  // namespace pops::runtime::program
