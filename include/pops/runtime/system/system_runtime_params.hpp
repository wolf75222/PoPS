#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief The per-AOT-block RUNTIME parameter registry of a System (ADC-578).
///
/// Extracted from the inline `std::map` that lived on `System::Impl`. It owns exactly one thing: the
/// name -> shared vector of current runtime parameter values for each AOT-compiled block that
/// declared `dsl.Param(..., kind='runtime')`. The vector is SHARED (shared_ptr) with the compiled
/// block closures: overwriting its contents (set_block_params) changes the block behavior at the next
/// step WITHOUT recompiling the .so.
///
/// OWNERSHIP CONTRACT
///  - The map ENTRIES (which blocks have runtime params) are FROZEN AT BIND: only add_compiled_block /
///    add_native_block register a slot, and those are structural setters refused once bound.
///  - The vector CONTENTS are MUTABLE DURING RUN: set_block_params rewrites them on a bound
///    simulation (an allowlisted runtime mutation, not a structural change).
///  - NOT checkpointed here: runtime params are re-declared by replaying the composition before a
///    restart; the checkpoint state is block state + ProgramRuntimeState.
///
/// KEY TYPING: the key is the user-chosen BLOCK NAME (not an ADC-584 route id): there is no stable
/// integer route id for a user-named block, so the string key is kept by contract.

namespace pops {
namespace runtime {
namespace system {

/// Data-only registry of the shared runtime-parameter vectors, keyed by block name.
struct SystemRuntimeParamsRegistry {
  /// name -> shared vector of current runtime param values (shared with the compiled closures).
  /// Absent for a block without runtime params or for the non-AOT paths (native / dynamic).
  std::map<std::string, std::shared_ptr<std::vector<double>>> block_params;

  /// Whether block @p name carries a runtime-parameter vector.
  bool has(const std::string& name) const {
    return block_params.find(name) != block_params.end();
  }

  /// Structured report (ADC-578 acceptance): block name -> number of runtime params, in name order
  /// (std::map iterates sorted). Lets a runtime report enumerate the tunable params of each block.
  std::vector<std::pair<std::string, std::size_t>> params_report() const {
    std::vector<std::pair<std::string, std::size_t>> out;
    out.reserve(block_params.size());
    for (const auto& kv : block_params)
      out.emplace_back(kv.first, kv.second ? kv.second->size() : std::size_t(0));
    return out;
  }
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
