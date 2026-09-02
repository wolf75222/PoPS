#ifndef POPS_RUNTIME_PROGRAM_MODULE_METADATA_HPP
#define POPS_RUNTIME_PROGRAM_MODULE_METADATA_HPP

// Candidate-table module metadata (Spec 2 / ADC-442). The v5 Program candidate carries typed,
// owner-qualified module records that the host deep-copies before preparation. This header validates
// those records once at install; the step body never performs a metadata lookup.
//
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/cell_temporal_partition.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace runtime {
namespace program {

/// Integer id of an operator within a module: its registration index. The generated .so addresses
/// operators by this id; the name/kind/signature strings are metadata only (debug, introspection,
/// validation), never a hot-path lookup.
using OperatorId = std::uint32_t;

/// Integer id of a state or field space within a module.
using SpaceId = std::uint32_t;

/// One operator's metadata carried by the prepared candidate tables.
struct OperatorMetadata {
  OperatorId id = 0;
  std::string owner;  ///< canonical model owner
  std::string name;
  std::string kind;          ///< one of the Spec-2 operator kinds (local_rate, field_operator, ...)
  std::string signature;     ///< human-readable typed signature
  std::string requirements;  ///< JSON, e.g. {"kind":"local_source","aux":["grad_x","grad_y"]}
};

/// The candidate-table module descriptor.
struct ModuleMetadata {
  std::vector<OperatorMetadata> operators;
  std::vector<std::string> state_spaces;
  std::vector<std::string> state_space_owners;
  std::vector<std::string> field_spaces;
  std::vector<std::string> field_space_owners;

  /// The exact owner-qualified operator, or nullptr if none.
  const OperatorMetadata* find(const std::string& owner, const std::string& name) const {
    for (const auto& op : operators) {
      if (op.owner == owner && op.name == name) {
        return &op;
      }
    }
    return nullptr;
  }

  /// Unqualified lookup succeeds only when the name is globally unique.
  const OperatorMetadata* find(const std::string& name) const {
    const OperatorMetadata* result = nullptr;
    for (const auto& op : operators) {
      if (op.name == name) {
        if (result != nullptr)
          return nullptr;
        result = &op;
      }
    }
    return result;
  }
};

/// v5 candidate-table adapter.  Program installation no longer performs per-field symbol lookup;
/// these conversions consume the host-owned, deep-copied tables assembled before preparation.
inline ModuleMetadata read_module_metadata(const ProgramInstallationTables& tables) {
  ModuleMetadata meta;
  meta.operators.reserve(tables.module_operators.size());
  for (std::size_t i = 0; i != tables.module_operators.size(); ++i) {
    const auto& row = tables.module_operators[i];
    if (row.identity.empty() || row.owner.empty() || row.kind.empty() ||
        row.requirements.size() < 2 || row.requirements.front() != '{' ||
        row.requirements.back() != '}')
      throw std::runtime_error("prepared Program module operator metadata is malformed");
    meta.operators.push_back({static_cast<OperatorId>(i), row.owner, row.identity, row.kind,
                              row.signature, row.requirements});
  }
  const auto copy_spaces = [](const std::vector<ProgramInstallationTables::Module>& rows,
                              std::vector<std::string>& names, std::vector<std::string>& owners) {
    names.reserve(rows.size());
    owners.reserve(rows.size());
    std::set<std::pair<std::string, std::string>> seen;
    for (const auto& row : rows) {
      if (row.identity.empty() || row.owner.empty() ||
          !seen.emplace(row.owner, row.identity).second)
        throw std::runtime_error("prepared Program module space metadata is malformed");
      names.push_back(row.identity);
      owners.push_back(row.owner);
    }
  };
  copy_spaces(tables.module_state_spaces, meta.state_spaces, meta.state_space_owners);
  copy_spaces(tables.module_field_spaces, meta.field_spaces, meta.field_space_owners);
  return meta;
}

using ProgramOperatorAuthority = std::array<std::uint64_t, 4>;

using ProgramHistoryReplayAuthority = std::pair<std::string, int>;

inline std::vector<ProgramOperatorAuthority> read_program_operator_authorities(
    const ProgramInstallationTables& tables) {
  std::vector<ProgramOperatorAuthority> result;
  result.reserve(tables.operator_authorities.size());
  for (const auto& row : tables.operator_authorities)
    result.push_back({row.words[0], row.words[1], row.words[2], row.words[3]});
  return result;
}

inline std::vector<ProgramHistoryReplayAuthority> read_program_history_replay_authorities(
    const ProgramInstallationTables& tables) {
  std::vector<ProgramHistoryReplayAuthority> result;
  result.reserve(tables.history_authorities.size());
  for (const auto& row : tables.history_authorities)
    result.push_back({row.identity, static_cast<int>(row.depth)});
  return result;
}

/// Frozen checkpoint shape exported by an AMR Program before its install-time prelude allocates a
/// level-qualified ring.  A negative component count means "the exact bound Program state width";
/// the AMR loader resolves it once through the authenticated ProgramBlockMap before publication.
struct ProgramCheckpointHistoryMetadata {
  std::string name;
  int program_owner = -1;
  std::string state_identity;
  std::string space_identity;
  std::string clock_identity;
  std::string interpolation_identity;
  int depth = 0;
  int components = -1;

  friend bool operator==(const ProgramCheckpointHistoryMetadata&,
                         const ProgramCheckpointHistoryMetadata&) = default;
};

struct ProgramCheckpointMetadata {
  std::vector<ProgramCheckpointHistoryMetadata> histories;
  std::vector<std::string> logical_clock_identities;
  std::string temporal_provider_identity;
  std::size_t temporal_cell_capacity = 0;
  std::size_t temporal_cells_per_topology_cell = 0;

  friend bool operator==(const ProgramCheckpointMetadata&,
                         const ProgramCheckpointMetadata&) = default;
};

inline ProgramCheckpointMetadata read_program_checkpoint_metadata(
    const ProgramInstallationTables& tables) {
  ProgramCheckpointMetadata metadata;
  std::set<std::string> clocks;
  std::set<std::string> identities;
  for (const auto& row : tables.checkpoint_shape) {
    if (row.identity.empty() || row.owner.empty() || row.space.empty() || row.clock.empty() ||
        row.transfer.empty() || row.block < 0 || row.retained_images < 2 || row.components == 0 ||
        row.components < -1 || !identities.emplace(row.identity).second)
      throw std::runtime_error("prepared Program checkpoint metadata is malformed");
    metadata.histories.push_back({row.identity, row.block, row.owner, row.space, row.clock,
                                  row.transfer, static_cast<int>(row.retained_images),
                                  row.components});
    clocks.emplace(row.clock);
  }
  // POPSAND5 serializes history descriptors in canonical identity order.  The ABI table is a
  // set-valued declaration and may be emitted in authoring order, so normalize it once at the
  // metadata boundary before it becomes the frozen capacity/shape authority.  Without this sort,
  // equivalent Programs could prepare identical rings yet fail installation solely because the
  // generated declaration order differed from the map-backed accepted runtime order.
  std::sort(metadata.histories.begin(), metadata.histories.end(),
            [](const ProgramCheckpointHistoryMetadata& left,
               const ProgramCheckpointHistoryMetadata& right) { return left.name < right.name; });
  metadata.logical_clock_identities.assign(clocks.begin(), clocks.end());
  // An empty checkpoint table still carries the global temporal partition.  Cell-local lowering
  // will replace this with its explicit provider/capacity record in the candidate resource plan.
  metadata.temporal_provider_identity = kGlobalTemporalPartitionProvider;
  metadata.temporal_cell_capacity = 0;
  metadata.temporal_cells_per_topology_cell = 0;
  return metadata;
}

/// Collect the quoted tokens of a JSON string array keyed by @p key inside the operator's flat
/// ``requirements`` JSON, e.g. key ``"aux"`` over {"kind":"local_source","aux":["grad_x","B_z"]} ->
/// {"grad_x","B_z"}. A dependency-free scan: the core has no JSON library on the install path and the
/// shape is a flat, closed vocabulary (the codegen emits ``"kind"`` plus a handful of requirement
/// arrays/scalars). It locates @p key, the following ``[``, and collects the quoted tokens up to the
/// closing ``]``. Returns empty when the key is absent or is not an array. Shared by required_aux /
/// required_block (Spec criterion 24).
inline std::vector<std::string> required_string_list(const std::string& requirements_json,
                                                     const std::string& key) {
  std::vector<std::string> out;
  const std::size_t k = requirements_json.find(key);
  if (k == std::string::npos) {
    return out;
  }
  const std::size_t lb = requirements_json.find('[', k + key.size());
  if (lb == std::string::npos) {
    return out;
  }
  const std::size_t rb = requirements_json.find(']', lb);
  if (rb == std::string::npos) {
    return out;
  }
  std::size_t p = lb + 1;
  while (p < rb) {
    const std::size_t q1 = requirements_json.find('"', p);
    if (q1 == std::string::npos || q1 >= rb) {
      break;
    }
    const std::size_t q2 = requirements_json.find('"', q1 + 1);
    if (q2 == std::string::npos || q2 > rb) {
      break;
    }
    out.push_back(requirements_json.substr(q1 + 1, q2 - q1 - 1));
    p = q2 + 1;
  }
  return out;
}

/// Read a single quoted JSON string value keyed by @p key inside the operator's flat ``requirements``
/// JSON, e.g. key ``"solver"`` over {"kind":"field_operator","solver":"geometric_mg"} ->
/// "geometric_mg". Returns "" when the key is absent. Dependency-free, same closed-vocabulary scan as
/// required_string_list; used for the scalar requirement kinds (solver, capability, schedule) of
/// Spec criterion 24.
inline std::string requirement_string(const std::string& requirements_json,
                                      const std::string& key) {
  auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  // @p key is the quoted JSON key (e.g. "\"solver\""). Match it as a genuine KEY, not as an array
  // element or a value substring: the first non-space char before it must be '{' or ',', and the
  // first non-space char after it must be ':'. (Without this, an aux field literally named "solver"
  // -- {"aux":["solver"],...} -- or any value equal to the key would yield a bogus requirement and
  // wrongly reject a valid install.) Scan all occurrences until one is a real key.
  std::size_t k = requirements_json.find(key);
  while (k != std::string::npos) {
    std::size_t before = k;
    while (before > 0 && is_space(requirements_json[before - 1])) {
      --before;
    }
    const bool key_start =
        before == 0 || requirements_json[before - 1] == '{' || requirements_json[before - 1] == ',';
    std::size_t after = k + key.size();
    while (after < requirements_json.size() && is_space(requirements_json[after])) {
      ++after;
    }
    if (key_start && after < requirements_json.size() && requirements_json[after] == ':') {
      const std::size_t q1 = requirements_json.find('"', after + 1);
      if (q1 == std::string::npos) {
        return std::string();
      }
      const std::size_t q2 = requirements_json.find('"', q1 + 1);
      if (q2 == std::string::npos) {
        return std::string();
      }
      return requirements_json.substr(q1 + 1, q2 - q1 - 1);
    }
    k = requirements_json.find(key, k + 1);
  }
  return std::string();
}

/// Aux-field names an operator requires (the ``"aux"`` array). Used by install-time requirement
/// validation (Spec criterion 24, ADC-446); kept as a named wrapper for call-site clarity.
inline std::vector<std::string> required_aux(const std::string& requirements_json) {
  return required_string_list(requirements_json, "\"aux\"");
}

/// Block-instance names an operator requires (the ``"block"`` array), e.g. a ``collisions`` operator
/// reading another species: {"kind":"local_source","block":["ions"]} -> {"ions"}. Install-time
/// validation rejects a simulation that did not instantiate one of them (Spec criterion 24).
inline std::vector<std::string> required_blocks(const std::string& requirements_json) {
  return required_string_list(requirements_json, "\"block\"");
}

/// Solver name a field operator requires (the scalar ``"solver"`` value), e.g.
/// {"kind":"field_operator","solver":"geometric_mg"} -> "geometric_mg". Empty when the operator has
/// no solver requirement. Install-time validation rejects a simulation whose configured field solver
/// does not match (Spec criterion 24).
inline std::string required_solver(const std::string& requirements_json) {
  return requirement_string(requirements_json, "\"solver\"");
}

}  // namespace program
}  // namespace runtime
}  // namespace pops

#endif  // POPS_RUNTIME_PROGRAM_MODULE_METADATA_HPP
