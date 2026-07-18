#pragma once

/// @file
/// @brief Builtin Cartesian BCRec adapter for the generic field-nullspace fact protocol.

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>

#include <string>
#include <vector>

namespace pops {
namespace detail {

inline FieldBoundaryNullspaceBehavior bc_rec_nullspace_behavior(BCType type, Real alpha) noexcept {
  switch (type) {
    case BCType::Periodic:
    case BCType::Foextrap:
      return FieldBoundaryNullspaceBehavior::PreservesConstantMode;
    case BCType::Robin:
      return alpha == Real(0) ? FieldBoundaryNullspaceBehavior::PreservesConstantMode
                              : FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
    case BCType::Dirichlet:
      return FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
    case BCType::External:
      return FieldBoundaryNullspaceBehavior::Opaque;
  }
  return FieldBoundaryNullspaceBehavior::Opaque;
}

}  // namespace detail

/// Translate the builtin four-face Cartesian record at the preparation boundary.  Generic
/// providers consume only the canonical boundary-id/behavior sequence and never depend on BCRec,
/// axis names, face count, or topology-specific branching.
inline FieldNullspaceOperatorFacts field_nullspace_operator_facts_from_bc_rec(
    const BCRec& boundary, bool has_reaction, bool internal_constraint = false) {
  return make_field_nullspace_operator_facts(
      "pops.mesh.boundary.bc-rec.cartesian-2d@1",
      {{"axis:0:lower", detail::bc_rec_nullspace_behavior(boundary.xlo, boundary.xlo_alpha)},
       {"axis:0:upper", detail::bc_rec_nullspace_behavior(boundary.xhi, boundary.xhi_alpha)},
       {"axis:1:lower", detail::bc_rec_nullspace_behavior(boundary.ylo, boundary.ylo_alpha)},
       {"axis:1:upper", detail::bc_rec_nullspace_behavior(boundary.yhi, boundary.yhi_alpha)}},
      has_reaction, internal_constraint);
}

}  // namespace pops
