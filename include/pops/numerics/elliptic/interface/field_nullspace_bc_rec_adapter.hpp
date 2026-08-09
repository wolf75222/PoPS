#pragma once

/// @file
/// @brief Exact-ranked physical-boundary adapter for the field-nullspace fact protocol.

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace detail {

inline FieldBoundaryNullspaceBehavior physical_boundary_nullspace_behavior(
    const PhysicalBoundaryFace& boundary) noexcept {
  switch (boundary.kind) {
    case PhysicalBoundaryKind::constant_extrapolation:
    case PhysicalBoundaryKind::neumann:
      return FieldBoundaryNullspaceBehavior::PreservesConstantMode;
    case PhysicalBoundaryKind::robin:
      return boundary.alpha == Real(0) ? FieldBoundaryNullspaceBehavior::PreservesConstantMode
                                       : FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
    case PhysicalBoundaryKind::dirichlet:
      return FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
    case PhysicalBoundaryKind::external:
      return FieldBoundaryNullspaceBehavior::Opaque;
  }
  return FieldBoundaryNullspaceBehavior::Opaque;
}

}  // namespace detail

/// Translate one complete exact-ranked Cartesian boundary at the preparation boundary. Periodic
/// faces are topology facts and preserve the constant mode; every physical face is interpreted from
/// its affine law. Generic providers consume only the resulting canonical identity/behavior list.
template <int Dim>
inline FieldNullspaceOperatorFacts field_nullspace_operator_facts_from_physical_boundary(
    const PhysicalBoundaryConditions<Dim>& boundary, bool has_reaction,
    bool internal_constraint = false) {
  std::vector<FieldBoundaryNullspaceFact> facts;
  facts.reserve(PhysicalBoundaryConditions<Dim>::face_count);
  constexpr std::array<BoundarySide, 2> sides{BoundarySide::lower, BoundarySide::upper};
  for (int axis = 0; axis < Dim; ++axis) {
    for (const BoundarySide side : sides) {
      const Face<Dim> face{axis, side};
      const FieldBoundaryNullspaceBehavior behavior =
          boundary.topology().is_periodic(face)
              ? FieldBoundaryNullspaceBehavior::PreservesConstantMode
              : detail::physical_boundary_nullspace_behavior(boundary.at(face));
      facts.push_back(FieldBoundaryNullspaceFact{
          "axis:" + std::to_string(axis) + (side == BoundarySide::lower ? ":lower" : ":upper"),
          behavior});
    }
  }
  return make_field_nullspace_operator_facts(
      "pops.mesh.boundary.physical-conditions.cartesian-" + std::to_string(Dim) + "d@1",
      std::move(facts), has_reaction, internal_constraint);
}

}  // namespace pops
