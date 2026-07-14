#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cstddef>
#include <utility>

namespace pops::flat_grid {

/// Explicit flat-array adapter used only by external component packages.
struct LocalGrid {
  Box2D dom;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;
  bool periodic;
  MultiFab aux;
};

inline LocalGrid make_grid(int n, double dx, double dy, bool periodic,
                           const double* aux_input, int naux) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, dx * n, 0.0, dy * n};
  BoxArray boxes = BoxArray::from_domain(dom, n);
  DistributionMapping distribution(boxes.size(), 1);
  BCRec bc;
  if (!periodic)
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Foextrap;
  MultiFab aux(boxes, distribution, naux, 1);
  aux.set_val(0.0);
  if (aux_input != nullptr) {
    Array4 values = aux.fab(0).array();
    const std::size_t cells = static_cast<std::size_t>(n) * n;
    for (int component = 0; component < naux; ++component)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          values(i, j, component) =
              aux_input[static_cast<std::size_t>(component) * cells +
                        static_cast<std::size_t>(j) * n + i];
    if (periodic)
      fill_boundary(aux, dom, Periodicity{true, true});
    else
      fill_ghosts(aux, dom, bc);
    const std::size_t tail = static_cast<std::size_t>(naux) * cells;
    for (int component = 0; component < naux; ++component) {
      const std::size_t offset = tail + static_cast<std::size_t>(2) * component;
      const int type = static_cast<int>(aux_input[offset]);
      if (type == static_cast<int>(BCType::Foextrap) ||
          type == static_cast<int>(BCType::Dirichlet))
        fill_physical_bc(
            aux, dom,
            aux_halo_override(
                bc, AuxHaloPolicy{static_cast<BCType>(type),
                                  static_cast<Real>(aux_input[offset + 1])}),
            component);
    }
  }
  return LocalGrid{dom, geom, boxes, distribution, bc, periodic, std::move(aux)};
}

inline void fill_interior(MultiFab& field, const double* input, int n, int ncomp) {
  Array4 values = field.fab(0).array();
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  for (int component = 0; component < ncomp; ++component)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        values(i, j, component) =
            input[static_cast<std::size_t>(component) * cells +
                  static_cast<std::size_t>(j) * n + i];
}

inline void extract(const MultiFab& field, double* output, int n, int ncomp) {
  device_fence();
  ConstArray4 values = field.fab(0).const_array();
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  for (int component = 0; component < ncomp; ++component)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        output[static_cast<std::size_t>(component) * cells +
               static_cast<std::size_t>(j) * n + i] = values(i, j, component);
}

}  // namespace pops::flat_grid
