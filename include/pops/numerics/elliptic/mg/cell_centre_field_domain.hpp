/// @file
/// @brief Field-only cell-centre conductor authority; never a transport mask.
#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <stdexcept>

namespace pops::elliptic::mg {

/// Exact strict-interior ball sampled independently at every geometric-MG level.
/// The allocated layout remains rectangular; inactive potential rows are fixed to +0.
struct CellCentreBallDomain {
  int dimension = 0;
  std::array<Real, 3> centre{};
  Real radius = Real(0);

  bool operator==(const CellCentreBallDomain&) const = default;

  template <int Dim>
  void validate() const {
    if (dimension != Dim || !std::isfinite(static_cast<double>(radius)) || radius <= Real(0))
      throw std::invalid_argument("cell-centre ball requires its exact rank and positive radius");
    for (int axis = 0; axis < 3; ++axis)
      if (!std::isfinite(static_cast<double>(centre[axis])) ||
          (axis >= Dim && centre[axis] != Real(0)))
        throw std::invalid_argument("cell-centre ball has invalid or foreign coordinate binding");
  }

  template <int Dim>
  POPS_HD bool contains(const Geometry<Dim>& geometry, const Index<Dim>& cell) const {
    const Real x = geometry.cell_coordinate(0, cell[0]) - centre[0];
    if constexpr (Dim == 1)
      return Kokkos::abs(x) < radius;
    else {
      const Real y = geometry.cell_coordinate(1, cell[1]) - centre[1];
      if constexpr (Dim == 2)
        return Kokkos::hypot(x, y) < radius;
      else {
        const Real z = geometry.cell_coordinate(2, cell[2]) - centre[2];
        return Kokkos::hypot(Kokkos::hypot(x, y), z) < radius;
      }
    }
  }
};

template <int Dim>
struct IntersectCellCentreBallKernel {
  FieldView<Real, Dim> active;
  Geometry<Dim> geometry;
  CellCentreBallDomain domain;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    if (!domain.contains(geometry, cell))
      active(cell, 0) = Real(0);
  }
};

template <int Dim>
struct ProjectCellCentreBallKernel {
  FieldView<Real, Dim> phi;
  Geometry<Dim> geometry;
  CellCentreBallDomain domain;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    if (!domain.contains(geometry, cell))
      phi(cell, 0) = Real(0);
  }
};

}  // namespace pops::elliptic::mg
