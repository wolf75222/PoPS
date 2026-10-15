#pragma once

#include <pops/runtime/program/amr_program_checkpoint.hpp>

namespace pops::test_support {

// Genuine AND7 framing for the no-history/no-interface attempt-cursor fixture.
// Weights are two int64 words and face publications have no affine suffix.
// Relabelling a current image as AND7 would instead exercise malformed framing.
template <int Dim>
std::vector<std::uint8_t> legacy_and7_attempt_image(
    const runtime::program::AmrProgramAcceptedState<Dim>& state) {
  namespace wire = runtime::program::checkpoint_detail;
  if (!state.histories.empty() || !state.history_slots.empty() ||
      !state.pending_history_remaps.empty() || !state.accepted_interface_flux.empty() ||
      !state.continuation_levels.empty())
    throw std::invalid_argument("legacy attempt fixture requires its declared empty stores");
  wire::Writer out;
  out.raw(wire::kLegacyMagic7);
  out.i32(Dim);
  out.string(state.spatial_contract);
  out.u64(state.topology_epoch);
  out.u64(state.materialization_generation);
  out.size(state.level_clocks.size());
  for (const auto& clock : state.level_clocks)
    wire::write_clock(out, clock);
  out.size(state.logical_clock_ticks.size());
  for (const auto& [identity, tick] : state.logical_clock_ticks) {
    out.string(identity);
    out.i64(tick);
  }
  out.size(0);  // histories
  out.size(0);  // history slots
  out.size(0);  // pending remaps
  out.bytes(state.history_flux_payload);
  wire::write_temporal_partition(out, state.temporal_partition);
  out.bytes(state.tagging_hysteresis_state);
  out.string(state.flux_budget_contract);
  out.string(state.coupling_contract);
  for (const auto& fragments : state.accepted_face_flux) {
    out.size(fragments.size());
    for (const auto& fragment : fragments) {
      if (fragment.measure.affine_publication)
        throw std::invalid_argument("AND7 cannot encode affine interval publications");
      out.string(fragment.key.owner);
      out.string(fragment.key.state);
      out.i32(fragment.key.levels.coarse);
      out.i32(fragment.key.levels.fine);
      out.u64(static_cast<std::uint64_t>(fragment.key.centering));
      out.i32(fragment.key.axis);
      wire::write_index(out, fragment.key.face);
      wire::write_index(out, fragment.key.coarse_face);
      wire::write_clock(out, fragment.key.clock);
      out.string(fragment.key.temporal_family);
      out.string(fragment.key.stage);
      out.u64(fragment.key.attempt);
      out.u64(static_cast<std::uint64_t>(fragment.key.role));
      out.u64(static_cast<std::uint64_t>(fragment.key.contribution));
      out.i64(fragment.measure.stage_weight.numerator_int64());
      out.i64(fragment.measure.stage_weight.denominator_int64());
      wire::write_rational(out, fragment.measure.substep_begin);
      wire::write_rational(out, fragment.measure.substep_end);
      out.real(fragment.measure.substep_duration);
      out.real(fragment.measure.face_measure);
      out.size(fragment.payload.size());
      for (const auto component : fragment.payload)
        out.real(static_cast<double>(component));
    }
  }
  out.size(0);  // interface fragments
  out.size(state.synchronization_events.size());
  for (const auto& event : state.synchronization_events) {
    out.i32(event.parent_level);
    out.i32(event.child_level);
    out.i32(event.runtime_block);
    out.string(event.phase);
    wire::write_clock(out, event.clock);
  }
  out.u64(state.face_evidence_provenance ? 1U : 0U);
  if (state.face_evidence_provenance) {
    const auto& origin = *state.face_evidence_provenance;
    out.string(origin.spatial_contract);
    out.u64(origin.topology_epoch);
    out.u64(origin.materialization_generation);
    out.size(origin.level_count);
  }
  return std::move(out).take();
}

}  // namespace pops::test_support
