#pragma once

#include <pops/runtime/context/aux_layout.hpp>  // AuxLayout (non-owning manifest reference)

#include <stdexcept>
#include <string>
#include <string_view>

/// @file
/// @brief Typed carrier for ONE field-solve result: which qualified provider produced it, for which
///        owner and stage, and the manifest (::pops::AuxLayout) describing its outputs. LIGHT
///        host-only header. FieldContext replaces the "read the magic aux component by
///        convention" contract at the compile/bind seam with a validity token: a context solved
///        for stage k of block b cannot be silently consumed as stage k' of block b'.
///
/// FieldContext is a DESCRIPTOR, not a container: it holds no aux storage and does not change
/// any numerics. The aux values still live in the shared ::pops::Aux channel; this only records
/// the provenance so a downstream RHS reads the RIGHT solve (ADC-588).

namespace pops {

/// Owner-qualified provenance + validity token for a field solve.
///
/// @c stage_id == -1 means the live context; @c stage_id >= 0 tags a specific RK stage so a
/// per-stage solve is not read out of order. Provider and owner identities are canonical strings,
/// never registry indices or a reserved default-field sentinel. @c layout is NON-OWNING: it points
/// at the installed FieldSolvePlan's manifest, whose lifetime outlives the context.
struct FieldContext {
  std::string provider_identity;       ///< complete authenticated FieldOperator/provider-pack id
  std::string owner_identity;          ///< qualified block/partition owner id
  int stage_id = -1;                   ///< -1 = live; >= 0 = a specific stage
  const AuxLayout* layout = nullptr;   ///< non-owning manifest for this solve plan's outputs

  /// True when this context was produced by exactly the requested qualified provider, owner and
  /// stage. There is deliberately no wildcard/default-provider match.
  bool matches(std::string_view req_provider, std::string_view req_owner, int req_stage) const {
    return provider_identity == req_provider && owner_identity == req_owner && stage_id == req_stage;
  }

  /// Resolve an output handle to its real aux component, deferring to the manifest. Throws a
  /// named error (via AuxLayout::component_of) when the layout is missing or the handle is
  /// unknown, so a mistyped output fails loud instead of reading component 0 by accident.
  int component_of(std::string_view handle) const {
    if (layout == nullptr) {
      throw std::logic_error("FieldContext: no AuxLayout bound (provider '" + provider_identity +
                             "', owner '" + owner_identity + "'); cannot resolve output '" +
                             std::string(handle) + "'");
    }
    return layout->component_of(handle);
  }
};

}  // namespace pops
