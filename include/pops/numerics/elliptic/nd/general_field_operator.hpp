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
/// diagonal-only coefficient uses harmonic face averaging. A complete component matrix
/// uses arithmetic face averaging, preserving SPD even for signed cross terms.
/// Reaction is the complete component matrix supplied by the authored equation.
template <int Dim, int Components, int CoefficientComponents = Components>
inline void apply_general_field(
    MultiFab<Dim>& output, MultiFab<Dim>& input, const MultiFab<Dim>& coefficients,
    const runtime::program::PreparedScalarBoundarySession<Dim>& boundary,
    const std::array<Real, Components * Components>& reaction,
    const std::array<PhysicalFieldBoundary, 2 * Dim>& physical) {
  static_assert(Components > 0);
  static_assert(CoefficientComponents == Components ||
                CoefficientComponents == Components * Components);
  long invalid = 0;
  try {
    require_field_boundary(boundary, physical);
    if (input.ncomp() != Components || output.ncomp() != Components ||
        coefficients.ncomp() != CoefficientComponents || input.layout() != output.layout() ||
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
        for (int other = 0; other < Components; ++other) {
          if constexpr (CoefficientComponents == Components) {
            if (other != component)
              continue;
          }
          const int slot = CoefficientComponents == Components ? component :
                           component * Components + other;
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> lower = cell, upper = cell;
            --lower[axis];
            ++upper[axis];
            const Real center = coefficient(cell, slot);
            Real low, high;
            if constexpr (CoefficientComponents == Components) {
              low = harmonic_tensor_face_average(coefficient(lower, slot), center);
              high = harmonic_tensor_face_average(center, coefficient(upper, slot));
            } else {
              // Convex matrix averaging preserves symmetry and positive definiteness;
              // entrywise harmonic averaging does not preserve either for cross terms.
              low = Real(0.5) * coefficient(lower, slot) + Real(0.5) * center;
              high = Real(0.5) * center + Real(0.5) * coefficient(upper, slot);
            }
            const Real spacing = geometry.spacing(axis);
            image -= (high * (value(upper, other) - value(cell, other)) -
                      low * (value(cell, other) - value(lower, other))) / (spacing * spacing);
          }
        }
        for (int other = 0; other < Components; ++other)
          image += reaction[component * Components + other] * value(cell, other);
        result(cell, component) = image;
      }
    });
  }
}

/// Validate and prepare coefficient halos once per frozen data version, outside CG.
template <int Dim, int Components = 0, int CoefficientComponents = Components>
inline void prepare_general_field_coefficients(
    MultiFab<Dim>& coefficients,
    const runtime::program::PreparedScalarBoundarySession<Dim>& boundary) {
  Real invalid = Real(0);
  const int components = coefficients.ncomp();
  if constexpr (Components > 0) {
    if (all_reduce_max(components != CoefficientComponents ? 1L : 0L, boundary.lane()) != 0)
      throw std::invalid_argument("field coefficient matrix has the wrong component count");
  }
  for (std::size_t local = 0; local < coefficients.local_size(); ++local) {
    const auto values = std::as_const(coefficients).fab(local).view();
    invalid +=
        for_each_cell_reduce_sum(coefficients.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          Real result = Real(0);
          if constexpr (Components > 1 && CoefficientComponents == Components * Components) {
            std::array<Real, Components * Components> matrix{};
            for (int i = 0; i < Components; ++i)
              for (int j = 0; j < Components; ++j) {
                const Real value = values(cell, i * Components + j);
                if (!std::isfinite(value) || value != values(cell, j * Components + i))
                  return Real(1);
                matrix[i * Components + j] = value;
              }
            // A strict local LDL factorization certifies the selected CG principal part.
            for (int k = 0; k < Components; ++k) {
              const Real pivot = matrix[k * Components + k];
              if (!(pivot > Real(0)) || !std::isfinite(pivot))
                return Real(1);
              for (int i = k + 1; i < Components; ++i)
                for (int j = k + 1; j < Components; ++j)
                  matrix[i * Components + j] -=
                      matrix[i * Components + k] * (matrix[k * Components + j] / pivot);
            }
          } else {
            for (int component = 0; component < components; ++component) {
              const Real value = values(cell, component);
              if (!(value > Real(0)) || value > std::numeric_limits<Real>::max())
                result += Real(1);
            }
          }
          return result;
        });
  }
  if (all_reduce_max(invalid, boundary.lane()) != Real(0))
    throw std::invalid_argument("field diffusion matrix must be finite, symmetric and strictly positive definite");
  boundary.fill(coefficients);
}

}  // namespace pops::elliptic::nd
