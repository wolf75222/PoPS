#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pops::elliptic::nd {

enum class PhysicalFieldBoundary : unsigned char { periodic, homogeneous_neumann };

/// The scalar Program closure is exactly the cell-centred zero conormal flux closure
/// when the physical relation says homogeneous Neumann. This adapter authenticates
/// that relation against topology, rather than inferring physical data from transport BCs.
template <int Dim>
inline void require_field_boundary(
    const runtime::program::PreparedScalarBoundarySession<Dim>& boundary,
    const std::array<PhysicalFieldBoundary, 2 * Dim>& physical) {
  for (int axis = 0; axis < Dim; ++axis) {
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      const bool periodic = boundary.topology().is_periodic(face);
      const auto law = physical[static_cast<std::size_t>(face.ordinal())];
      if ((law == PhysicalFieldBoundary::periodic) != periodic)
        throw std::invalid_argument("field physical boundary and prepared topology disagree");
    }
  }
}

/// Allocation-free conservative variable-coefficient apply over one physical tuple.
/// The coefficient and boundary session are immutable, prepared solve inputs. Every
/// diagonal diffusion coefficient is sampled by harmonic face averaging; reaction is
/// the complete small joint matrix, including off-diagonal coupling.
template <int Dim, int Components>
inline void apply_general_field(
    MultiFab<Dim>& output, MultiFab<Dim>& input, const MultiFab<Dim>& coefficients,
    const runtime::program::PreparedScalarBoundarySession<Dim>& boundary,
    const std::array<Real, Components * Components>& reaction,
    const std::array<PhysicalFieldBoundary, 2 * Dim>& physical) {
  static_assert(Components == 1 || Components == 2,
                "general field realization supports scalar or joint two-field tuples");
  long invalid = 0;
  try {
    require_field_boundary(boundary, physical);
    if (input.ncomp() != Components || output.ncomp() != Components ||
        coefficients.ncomp() != Components || input.layout() != output.layout() ||
        input.layout() != coefficients.layout() || input.distribution() != output.distribution() ||
        input.distribution() != coefficients.distribution() ||
        input.local_rank() != output.local_rank() ||
        input.local_rank() != coefficients.local_rank())
      invalid = 1;
    for (int axis = 0; axis < Dim; ++axis)
      if (input.ghosts()[axis] < 1 || coefficients.ghosts()[axis] < 1)
        invalid = 1;
    for (const Real value : reaction)
      if (!std::isfinite(value))
        invalid = 1;
  } catch (...) {
    invalid = 1;
  }
  if (all_reduce_max(invalid, boundary.lane()) != 0)
    throw std::invalid_argument(
        "general field apply storage, halo, reaction or physical boundary rejected on a "
        "communicator rank");
  boundary.fill(input);
  const Geometry<Dim> geometry = boundary.geometry();
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const auto result = output.fab(local).view();
    const auto value = std::as_const(input).fab(local).view();
    const auto coefficient = coefficients.fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int component = 0; component < Components; ++component) {
        Real image = Real(0);
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> lower = cell, upper = cell;
          --lower[axis];
          ++upper[axis];
          const Real center = coefficient(cell, component);
          const Real low = harmonic_tensor_face_average(coefficient(lower, component), center);
          const Real high = harmonic_tensor_face_average(center, coefficient(upper, component));
          const Real spacing = geometry.spacing(axis);
          image -= (high * (value(upper, component) - value(cell, component)) -
                    low * (value(cell, component) - value(lower, component))) /
                   (spacing * spacing);
        }
        for (int other = 0; other < Components; ++other)
          image += reaction[component * Components + other] * value(cell, other);
        result(cell, component) = image;
      }
    });
  }
}

/// Validate and prepare coefficient halos once per frozen data version, outside CG.
template <int Dim>
inline void prepare_general_field_coefficients(
    MultiFab<Dim>& coefficients,
    const runtime::program::PreparedScalarBoundarySession<Dim>& boundary) {
  Real invalid = Real(0);
  const int components = coefficients.ncomp();
  for (std::size_t local = 0; local < coefficients.local_size(); ++local) {
    const auto values = std::as_const(coefficients).fab(local).view();
    invalid +=
        for_each_cell_reduce_sum(coefficients.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          Real result = Real(0);
          for (int component = 0; component < components; ++component) {
            const Real value = values(cell, component);
            if (!(value > Real(0)) || value > std::numeric_limits<Real>::max())
              result += Real(1);
          }
          return result;
        });
  }
  if (all_reduce_max(invalid, boundary.lane()) != Real(0))
    throw std::invalid_argument("field diffusion coefficient must be finite and strictly positive");
  boundary.fill(coefficients);
}

}  // namespace pops::elliptic::nd
