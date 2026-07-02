#pragma once

#include <pops/runtime/context/aux_layout.hpp>  // AuxLayout (non-owning manifest reference)

#include <stdexcept>
#include <string>
#include <string_view>

/// @file
/// @brief Typed carrier for ONE field-solve result: which field problem produced it, for which
///        block and stage, and the manifest (::pops::AuxLayout) describing its outputs. LIGHT
///        host-only header. FieldContext replaces the "read the magic aux component by
///        convention" contract at the compile/bind seam with a validity token: a context solved
///        for stage k of block b cannot be silently consumed as stage k' of block b'.
///
/// FieldContext is a DESCRIPTOR, not a container: it holds no aux storage and does not change
/// any numerics. The aux values still live in the shared ::pops::Aux channel; this only records
/// the provenance so a downstream RHS reads the RIGHT solve (ADC-588).

namespace pops {

/// Provenance + validity token for a field solve.
///
/// @c stage_id == -1 means the live/default context (the single per-step solve that fills the
/// shared phi/grad channel); @c stage_id >= 0 tags a specific RK stage so a per-stage solve is
/// not read out of order. @c layout is NON-OWNING: it points at the field problem's manifest
/// (owned by the registry / system), whose lifetime outlives the context.
struct FieldContext {
  int field_problem_id = -1;          ///< index into the FieldProblemRegistry (-1 = default "phi")
  int block_index = 0;                ///< owning block (multi-block systems)
  int stage_id = -1;                  ///< -1 = live/default; >= 0 = a specific stage
  const AuxLayout* layout = nullptr;  ///< non-owning manifest for this problem's outputs

  /// True when this context was produced by exactly the requested (problem, block, stage). The
  /// bind seam checks this so a stage-k context cannot be mistaken for stage-k' or another
  /// block. A negative @p req_field matches any problem (the default single-field case).
  bool matches(int req_field, int req_block, int req_stage) const {
    return (req_field < 0 || field_problem_id == req_field) && block_index == req_block &&
           stage_id == req_stage;
  }

  /// Resolve an output handle to its real aux component, deferring to the manifest. Throws a
  /// named error (via AuxLayout::component_of) when the layout is missing or the handle is
  /// unknown, so a mistyped output fails loud instead of reading component 0 by accident.
  int component_of(std::string_view handle) const {
    if (layout == nullptr) {
      throw std::logic_error("FieldContext: no AuxLayout bound (field problem " +
                             std::to_string(field_problem_id) + "); cannot resolve output '" +
                             std::string(handle) + "'");
    }
    return layout->component_of(handle);
  }
};

}  // namespace pops
