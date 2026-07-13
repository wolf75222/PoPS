#pragma once

#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonReport (IMEX per-block report)
#include <pops/runtime/numerical_defaults.hpp>  // EffectiveBlockOptions

#include <map>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief The per-block / per-stage DIAGNOSTICS and inspection registry of a System (ADC-578).
///
/// Extracted from three inline `std::map`s that lived on `System::Impl`. It groups the metadata a
/// runtime report reads back: the effective numerical/physical block options captured at
/// configuration time and the OPT-IN Newton (IMEX) per-block reports. None of these are read by
/// SystemStepper -> MockImpl-invisible.
///
/// OWNERSHIP CONTRACT
///  - block_options: FROZEN AT BIND. Populated only by structural block installation, refused once
///    bound, and read-only afterwards (effective_options_report).
///  - newton_reports: the map ENTRIES are frozen at bind (allocated by add_block for a block that
///    opted into diagnostics or a fail policy); the report CONTENTS are MUTABLE DURING RUN (the
///    block IMEX advance closures write into them by raw pointer each step). The shared_ptr gives a
///    STABLE address even when the map reallocates at a later add_block.
///  - NOT checkpointed: inspection metadata is re-derived by replaying the composition.
///
/// KEY TYPING: keyed by the user-chosen BLOCK / STAGE NAME (no ADC-584 route id exists for a
/// user-named block or stage) -> the string key is kept by contract.

namespace pops {
namespace runtime {
namespace system {

/// Data-only diagnostics/inspection registry, keyed by user block/stage name.
struct SystemDiagnosticsRegistry {
  /// Effective numerical/physical block options captured when the block/stage is added. The closures
  /// are opaque, so inspection stores the user-facing route decisions here.
  std::map<std::string, EffectiveBlockOptions> block_options;
  /// OPT-IN IMEX Newton reports, in shared_ptr for a STABLE address (the block AdvanceImex* closures
  /// write into it by raw pointer). Absent (missing key) for a block without newton_diagnostics ->
  /// newton_report raises a clear error rather than returning a silently empty report.
  std::map<std::string, std::shared_ptr<NewtonReport>> newton_reports;

  /// Effective block options of @p name, or nullptr if the block was never registered.
  EffectiveBlockOptions* block_options_ptr(const std::string& name) {
    auto it = block_options.find(name);
    return it == block_options.end() ? nullptr : &it->second;
  }
  const EffectiveBlockOptions* block_options_ptr(const std::string& name) const {
    auto it = block_options.find(name);
    return it == block_options.end() ? nullptr : &it->second;
  }

  /// The Newton report of @p name, or nullptr if the block did not enable diagnostics.
  const NewtonReport* newton_report_ptr(const std::string& name) const {
    auto it = newton_reports.find(name);
    return it == newton_reports.end() ? nullptr : it->second.get();
  }

  /// Structured report (ADC-578 acceptance): the effective options of every registered block, in
  /// name order. The caller overlays the live block state (ncomp / ghosts / vars) on each row.
  std::vector<EffectiveBlockOptions> options_report() const {
    std::vector<EffectiveBlockOptions> out;
    out.reserve(block_options.size());
    for (const auto& kv : block_options)
      out.push_back(kv.second);
    return out;
  }
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
