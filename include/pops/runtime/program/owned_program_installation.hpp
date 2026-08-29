#pragma once

/// @file
/// @brief Host ownership for one accepted native Program image and its DSO state.

#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/program/program_persistent_value_store.hpp>

#include <algorithm>
#include <optional>
#include <cctype>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pops::runtime::program {

namespace detail {

inline std::string resource_json_string(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (byte < 0x20 || byte > 0x7f)
          throw std::invalid_argument("Program resource manifest is not canonical ASCII JSON");
        result.push_back(static_cast<char>(byte));
    }
  }
  result.push_back('"');
  return result;
}

inline std::string resource_json_u64(std::uint64_t value) { return std::to_string(value); }

/// Replace one already-authenticated JSON member without introducing a JSON parser into the ABI
/// header.  Candidate manifests are canonical compact JSON produced by codegen; requiring one
/// exact old value makes a forged/ambiguous manifest fail closed instead of silently rewriting an
/// unrelated nested field.
inline void replace_resource_manifest_member(std::string& manifest, std::string_view member,
                                             std::string_view old_value,
                                             std::string_view new_value) {
  const std::string needle = std::string(member) + std::string(old_value);
  const std::size_t first = manifest.find(needle);
  if (first == std::string::npos || manifest.find(needle, first + 1) != std::string::npos)
    throw std::invalid_argument("Program persistent resource manifest has an ambiguous member");
  manifest.replace(first + member.size(), old_value.size(), new_value);
}

inline bool replace_resource_manifest_member_if_present(std::string& manifest,
                                                        std::string_view member,
                                                        std::string_view old_value,
                                                        std::string_view new_value) {
  const std::string needle = std::string(member) + std::string(old_value);
  const std::size_t first = manifest.find(needle);
  if (first == std::string::npos)
    return false;
  if (manifest.find(needle, first + 1) != std::string::npos)
    throw std::invalid_argument("Program persistent resource manifest has an ambiguous member");
  manifest.replace(first + member.size(), old_value.size(), new_value);
  return true;
}

inline std::size_t canonical_component_name_count(std::string_view value) {
  if (value == "[]") return 0;
  if (value.size() < 4 || value.front() != '[' || value.back() != ']')
    throw std::invalid_argument("Program resource component_names is not a canonical JSON array");
  std::size_t cursor = 1, count = 0;
  while (cursor + 1 < value.size()) {
    if (value[cursor++] != '"' || cursor >= value.size() || value[cursor] == '"')
      throw std::invalid_argument("Program resource component_names has an empty or malformed name");
    bool closed = false;
    while (cursor < value.size()) {
      const unsigned char byte = static_cast<unsigned char>(value[cursor++]);
      if (byte == '"') { closed = true; break; }
      if (byte == '\\') {
        if (cursor == value.size())
          throw std::invalid_argument("Program resource component_names has a truncated escape");
        ++cursor;
      } else if (byte < 0x20 || byte > 0x7f) {
        throw std::invalid_argument("Program resource component_names is not canonical ASCII JSON");
      }
    }
    if (!closed) throw std::invalid_argument("Program resource component_names is unterminated");
    ++count;
    if (cursor == value.size() - 1) break;
    if (value[cursor++] != ',' || cursor == value.size() - 1)
      throw std::invalid_argument("Program resource component_names is not compact JSON");
  }
  if (cursor != value.size() - 1)
    throw std::invalid_argument("Program resource component_names has trailing JSON bytes");
  return count;
}

inline std::size_t canonical_shape_extent_count(std::string_view value) {
  if (value == "[]") return 0;
  if (value.size() < 3 || value.front() != '[' || value.back() != ']')
    throw std::invalid_argument("Program resource shape is not a canonical JSON array");
  std::size_t cursor = 1, count = 0;
  while (cursor + 1 < value.size()) {
    const std::size_t begin = cursor;
    if (value[cursor] == '0')
      throw std::invalid_argument("Program resource shape contains a non-positive extent");
    while (cursor < value.size() - 1 && value[cursor] >= '0' && value[cursor] <= '9')
      ++cursor;
    if (cursor == begin)
      throw std::invalid_argument("Program resource shape contains a non-integer extent");
    ++count;
    if (cursor == value.size() - 1) break;
    if (value[cursor++] != ',' || cursor == value.size() - 1)
      throw std::invalid_argument("Program resource shape is not compact JSON");
  }
  if (cursor != value.size() - 1)
    throw std::invalid_argument("Program resource shape has trailing JSON bytes");
  return count;
}

}  // namespace detail

struct ProgramInstallationMetadata final {
  static constexpr std::size_t kMaximumViewBytes = 1024 * 1024;
  static constexpr std::size_t kMaximumAggregateBytes = 4 * 1024 * 1024;
  std::string artifact_identity;
  std::string abi_key;
  std::string route_manifest;
  std::string boundary_manifest;
  std::string persistent_resource_manifest;
  std::string checkpoint_identity;
  std::string program_name;

  [[nodiscard]] static std::string materialize_view(ProgramAbiView view, std::size_t& aggregate,
                                                     std::string_view table_name) {
    if (view.data == nullptr || view.size == 0 ||
        view.size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        view.size > kMaximumViewBytes || view.size > kMaximumAggregateBytes - aggregate)
      throw std::length_error("Program " + std::string(table_name) +
                              " view exceeds its sealed host limit");
    const auto size = static_cast<std::size_t>(view.size);
    for (std::size_t index = 0; index != size; ++index) {
      if (view.data[index] == '\0')
        throw std::invalid_argument("Program " + std::string(table_name) +
                                    " view contains an embedded NUL");
    }
    aggregate += size;
    return std::string(view.data, size);
  }

  [[nodiscard]] static ProgramInstallationMetadata materialize(const ProgramCandidateDescriptor& candidate,
                                                                std::size_t& aggregate) {
    const auto copy = [&](ProgramAbiView view, std::string_view name) {
      return materialize_view(view, aggregate, name);
    };
    return {copy(candidate.artifact_identity, "artifact identity"), copy(candidate.abi_key, "ABI key"),
            copy(candidate.route_manifest, "route manifest"),
            copy(candidate.boundary_manifest, "boundary manifest"),
            copy(candidate.persistent_resource_manifest, "persistent resource manifest"),
            copy(candidate.checkpoint_identity, "checkpoint identity"),
            copy(candidate.program_name, "program name")};
  }
};

struct ProgramInstallationTables;
[[nodiscard]] inline ProgramResourcePlan make_program_resource_plan(
    const ProgramInstallationTables& tables, std::uint64_t ceiling);

/// Every string below is allocated by the host.  This is deliberately not a mirror of the POD ABI:
/// a `PreparedProgramInstallation` remains fully inspectable after candidate state destruction and
/// before `dlclose`, without retaining a pointer into the Program image.
struct ProgramInstallationTables final {
  struct Block final { std::string name; };
  struct Parameter final { std::int32_t block = -1; std::int32_t index = -1; double default_value = 0.0; std::string name; };
  struct Authority final { std::uint64_t words[4]{}; };
  struct HistoryAuthority final { std::string identity; std::uint32_t depth = 0; };
  struct Checkpoint final {
    std::string identity, owner, space, clock, transfer;
    std::int32_t block = -1, components = -1;
    std::uint64_t retained_images = 0;
  };
  struct FluxBudget final {
    std::uint64_t rhs_basis_bound = 0, coefficient_term_bound = 0;
    std::uint64_t interface_application_bound = 0, interface_identity_character_bound = 0;
  };
  struct ResourcePlan final {
    std::uint32_t slot = 0, flags = 0;
    std::uint64_t value_id = 0, occurrence_path_id = 0;
    std::int32_t level = -1;
    std::uint32_t components = 0, ghosts = 0;
    /// A null footprint is intentional for a runtime-sized declaration.  It is not an unknown
    /// value that may be projected to a dummy allocation: only the host materializer may replace
    /// it with exact totals after all prepare_* callbacks have supplied their layouts.
    std::optional<std::uint64_t> bytes, maximum_bytes;
    std::optional<std::uint64_t> cells, itemsize;
    ProgramResourcePlanType resource_type = ProgramResourcePlanType::exact;
    std::string schema, plan_digest, identity, occurrence_path, owner, space, clock;
    std::string lifetime, centering, off_policy, communication, transfer_provider;
    std::string restart_provider, component_names, shape;

    [[nodiscard]] bool runtime_sized() const noexcept {
      return resource_type == ProgramResourcePlanType::runtime_sized ||
             (flags & kProgramResourceRuntimeSized) != 0;
    }
  };

  /// Exact host evidence for one prepared allocation family.  ``cells`` is the complete allocated
  /// cell count represented by the prototype (including any ghost storage); it is never inferred
  /// from a logical one-cell model.  ``bytes``/``maximum_bytes`` are optional assertions from a
  /// backend layout and are checked against the overflow-safe product when present.
  struct ResourceLayout final {
    std::uint64_t cells = 0;
    std::uint64_t itemsize = 0;
    std::uint32_t components = 0;
    std::uint32_t ghosts = 0;
    std::optional<std::uint64_t> bytes;
    std::optional<std::uint64_t> maximum_bytes;
    /// Optional prepared allocation shape.  It is evidence, not a source of the byte product;
    /// ``cells`` remains the complete post-ghost allocation extent.
    std::vector<std::uint64_t> shape;
  };

  /// Runtime preparation family.  A generated solve may use the same numeric subslot for
  /// state- and scalar-scratch families, so ``kind`` is part of the authenticated identity rather
  /// than an incidental label.  ``generic`` is reserved for backend-specific persistent families.
  enum class ResourcePrototypeKind : std::uint8_t {
    // Keep the first three values aligned with ProgramExecutionServices::ScratchKind so a host can
    // carry the prepare_* kind code without an unchecked translation.
    rhs = 0,
    state = 1,
    scalar = 2,
    persistent_schedule = 3,
    generic = 4,
  };

  /// One exact prototype observed for a resource slot/subslot during preparation.  Subslots are
  /// intentionally first-class: transient rhs/state/scalar families and persistent schedule
  /// values all contribute to the final slot totals.
  struct ResourcePrototype final {
    std::uint32_t slot = 0;
    std::int32_t subslot = 0;
    ResourceLayout layout{};
    ResourcePrototypeKind kind = ResourcePrototypeKind::generic;
  };

  [[nodiscard]] static std::string resource_prototype_kind_name(
      ResourcePrototypeKind kind) {
    switch (kind) {
      case ResourcePrototypeKind::generic: return "generic";
      case ResourcePrototypeKind::rhs: return "rhs";
      case ResourcePrototypeKind::state: return "state";
      case ResourcePrototypeKind::scalar: return "scalar";
      case ResourcePrototypeKind::persistent_schedule: return "persistent_schedule";
    }
    throw std::invalid_argument("Program resource prototype has an unknown kind");
  }

  /// Merge rank-local preparation observations into one deterministic collective layout set.
  /// Missing rows and an explicit all-zero layout represent a rank with no local fab and are
  /// ignored; for present rows the largest exact allocated-cell footprint and family ceiling are
  /// retained.  This keeps the published plan/digest identical on every rank without summing
  /// global cells as if they were one rank's memory.
  [[nodiscard]] static std::vector<ResourcePrototype> merge_resource_prototypes(
      std::span<const std::vector<ResourcePrototype>> rank_prototypes) {
    std::vector<ResourcePrototype> merged;
    const auto checked_product = [](std::uint64_t left, std::uint64_t right) -> std::uint64_t {
      if (left == 0 || right == 0)
        throw std::invalid_argument("rank-local Program resource layout has a zero extent");
      if (left > std::numeric_limits<std::uint64_t>::max() / right)
        throw std::overflow_error("rank-local Program resource layout overflows uint64");
      return left * right;
    };
    const auto exact_bytes = [&](const ResourcePrototype& prototype) -> std::uint64_t {
      const auto& layout = prototype.layout;
      for (const auto extent : layout.shape)
        if (extent == 0)
          throw std::invalid_argument("rank-local Program resource layout shape has a zero extent");
      // A rank may own no local fab while still reporting the family item size, component count or
      // ghost depth.  Zero cells plus zero/absent byte assertions is the only no-local marker; it is
      // ignored before the collective max reduction, while a nonzero byte claim remains invalid.
      const bool no_local_fab = layout.cells == 0 &&
                                (!layout.bytes || *layout.bytes == 0) &&
                                (!layout.maximum_bytes || *layout.maximum_bytes == 0);
      if (no_local_fab)
        return 0;
      const auto bytes = checked_product(
          checked_product(layout.components, layout.itemsize), layout.cells);
      if (layout.bytes && *layout.bytes != bytes)
        throw std::invalid_argument(
            "rank-local Program resource layout byte assertion disagrees with its product");
      return bytes;
    };
    for (const auto& rank : rank_prototypes) {
      std::vector<std::tuple<ResourcePrototypeKind, std::uint32_t, std::int32_t>> local_keys;
      for (const auto& prototype : rank) {
        (void)resource_prototype_kind_name(prototype.kind);
        const auto key = std::tuple{prototype.kind, prototype.slot, prototype.subslot};
        if (std::find(local_keys.begin(), local_keys.end(), key) != local_keys.end())
          throw std::invalid_argument(
              "duplicate rank-local Program resource prototype kind/slot/subslot");
        local_keys.push_back(key);
        const auto bytes = exact_bytes(prototype);
        if (bytes == 0)
          continue;
        const auto local_maximum = prototype.layout.maximum_bytes.value_or(bytes);
        if (local_maximum < bytes)
          throw std::invalid_argument(
              "rank-local Program resource layout maximum is below its bytes");
        const auto found = std::find_if(
            merged.begin(), merged.end(), [&](const ResourcePrototype& candidate) {
              return candidate.kind == prototype.kind && candidate.slot == prototype.slot &&
                     candidate.subslot == prototype.subslot;
            });
        if (found == merged.end()) {
          ResourcePrototype normalized = prototype;
          normalized.layout.bytes = bytes;
          normalized.layout.maximum_bytes = local_maximum;
          merged.push_back(std::move(normalized));
          continue;
        }
        if (found->layout.itemsize != prototype.layout.itemsize ||
            found->layout.components != prototype.layout.components ||
            found->layout.ghosts != prototype.layout.ghosts)
          throw std::invalid_argument(
              "rank-local Program resource layouts disagree on family metadata");
        if (!found->layout.shape.empty() && !prototype.layout.shape.empty() &&
            found->layout.shape != prototype.layout.shape)
          throw std::invalid_argument(
              "rank-local Program resource layouts disagree on shape constraints");
        if (found->layout.shape.empty())
          found->layout.shape = prototype.layout.shape;
        const auto maximum_cells = std::max(found->layout.cells, prototype.layout.cells);
        const auto maximum_bytes = checked_product(
            checked_product(found->layout.components, found->layout.itemsize), maximum_cells);
        found->layout.cells = maximum_cells;
        found->layout.bytes = maximum_bytes;
        found->layout.maximum_bytes = std::max(
            found->layout.maximum_bytes.value_or(maximum_bytes), local_maximum);
      }
    }
    std::sort(merged.begin(), merged.end(), [](const ResourcePrototype& left,
                                               const ResourcePrototype& right) {
      const auto left_kind = resource_prototype_kind_name(left.kind);
      const auto right_kind = resource_prototype_kind_name(right.kind);
      return std::tie(left_kind, left.slot, left.subslot) <
             std::tie(right_kind, right.slot, right.subslot);
    });
    return merged;
  }

  [[nodiscard]] static std::vector<ResourcePrototype> merge_resource_prototypes(
      const std::vector<std::vector<ResourcePrototype>>& rank_prototypes) {
    return merge_resource_prototypes(std::span<const std::vector<ResourcePrototype>>(rank_prototypes));
  }
  struct Route final { std::string identity, kind; std::uint64_t capability_bits = 0; };
  struct Module final { std::string identity, kind, signature, requirements, owner; };

  std::vector<Block> blocks;
  std::vector<Parameter> parameters;
  std::vector<Authority> operator_authorities;
  std::vector<HistoryAuthority> history_authorities;
  std::vector<Checkpoint> checkpoint_shape;
  std::vector<FluxBudget> flux_budgets;
  // Materialization updates this copied host table only after the complete prototype set has
  // passed validation.  The vector is mutable because the public materializer is intentionally a
  // transactional const view over an installation owner; no DSO memory is ever modified.
  mutable std::vector<ResourcePlan> resource_plan;
  std::vector<Route> boundary_routes;
  std::vector<Route> provider_routes;
  std::vector<Module> module_operators;
  std::vector<Module> module_state_spaces;
  std::vector<Module> module_field_spaces;

  /// Canonical host-owned list of exact layouts used to seal a runtime-sized plan.  It is kept in
  /// the copied tables so publication/checkpoint receipts retain evidence beyond the aggregate
  /// row bytes.  The field is mutable only because materialization is a const, transactional view
  /// operation; it changes exclusively after a successful checked materialization.
  mutable std::string materialized_layout_manifest;

  [[nodiscard]] static ProgramInstallationTables materialize(const ProgramCandidateDescriptor& candidate,
                                                              std::size_t& aggregate) {
    ProgramInstallationTables result;
    const auto copy = [&](ProgramAbiView view, std::string_view name) {
      return ProgramInstallationMetadata::materialize_view(view, aggregate, name);
    };
    const auto table_count = [](const ProgramAbiTable& table, std::string_view name) -> std::size_t {
      if (table.count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("Program " + std::string(name) + " table count overflows host size");
      return static_cast<std::size_t>(table.count);
    };
    const auto require_unique = [](const auto& rows, const auto& key, std::string_view name) {
      std::unordered_set<std::string> seen;
      for (const auto& row : rows) {
        const auto value = key(row);
        if (!seen.emplace(value).second)
          throw std::invalid_argument("Program " + std::string(name) + " table contains duplicate identity");
      }
    };
    const auto raw_blocks = static_cast<const ProgramBlockRecord*>(candidate.blocks.data);
    const auto block_count = table_count(candidate.blocks, "block");
    result.blocks.reserve(block_count);
    for (std::size_t i = 0; i != block_count; ++i)
      result.blocks.push_back({copy(raw_blocks[i].name, "block name")});
    require_unique(result.blocks, [](const Block& row) -> const std::string& { return row.name; }, "block");

    const auto raw_parameters = static_cast<const ProgramParameterRecord*>(candidate.parameters.data);
    result.parameters.reserve(table_count(candidate.parameters, "parameter"));
    std::unordered_set<std::string> parameter_keys;
    for (std::size_t i = 0, n = table_count(candidate.parameters, "parameter"); i != n; ++i) {
      const auto& raw = raw_parameters[i];
      if (raw.block < 0 || raw.index < 0 || static_cast<std::size_t>(raw.block) >= result.blocks.size())
        throw std::invalid_argument("Program parameter table has an out-of-range block or index");
      auto name = copy(raw.name, "parameter name");
      const auto key = std::to_string(raw.block) + ':' + std::to_string(raw.index) + ':' + name;
      if (!parameter_keys.emplace(key).second)
        throw std::invalid_argument("Program parameter table contains a duplicate entry");
      result.parameters.push_back({raw.block, raw.index, raw.default_value, std::move(name)});
    }

    const auto raw_authorities = static_cast<const ProgramAuthorityRecord*>(candidate.operator_authorities.data);
    result.operator_authorities.reserve(table_count(candidate.operator_authorities, "operator authority"));
    for (std::size_t i = 0, n = table_count(candidate.operator_authorities, "operator authority"); i != n; ++i) {
      if (raw_authorities[i].words[0] == 0 && raw_authorities[i].words[1] == 0 &&
          raw_authorities[i].words[2] == 0 && raw_authorities[i].words[3] == 0)
        throw std::invalid_argument("Program operator authority cannot be all zero");
      result.operator_authorities.push_back({{raw_authorities[i].words[0], raw_authorities[i].words[1], raw_authorities[i].words[2], raw_authorities[i].words[3]}});
    }

    const auto raw_history = static_cast<const ProgramHistoryAuthorityRecord*>(candidate.history_authorities.data);
    result.history_authorities.reserve(table_count(candidate.history_authorities, "history authority"));
    for (std::size_t i = 0, n = table_count(candidate.history_authorities, "history authority"); i != n; ++i) {
      if (raw_history[i].depth == 0 || raw_history[i].reserved != 0)
        throw std::invalid_argument("Program history authority has invalid depth or reserved bits");
      result.history_authorities.push_back({copy(raw_history[i].identity, "history authority identity"), raw_history[i].depth});
    }
    require_unique(result.history_authorities, [](const HistoryAuthority& row) -> const std::string& { return row.identity; }, "history authority");

    const auto raw_checkpoints = static_cast<const ProgramCheckpointRecord*>(candidate.checkpoint_shape.data);
    result.checkpoint_shape.reserve(table_count(candidate.checkpoint_shape, "checkpoint shape"));
    for (std::size_t i = 0, n = table_count(candidate.checkpoint_shape, "checkpoint shape"); i != n; ++i) {
      const auto& raw = raw_checkpoints[i];
      if (raw.block < -1 || (raw.block >= 0 && static_cast<std::size_t>(raw.block) >= result.blocks.size()) || raw.retained_images == 0)
        throw std::invalid_argument("Program checkpoint shape has an out-of-range block or zero retention");
      result.checkpoint_shape.push_back({copy(raw.identity, "checkpoint identity"), copy(raw.owner, "checkpoint owner"), copy(raw.space, "checkpoint space"), copy(raw.clock, "checkpoint clock"), copy(raw.transfer, "checkpoint transfer"), raw.block, raw.components, raw.retained_images});
    }
    require_unique(result.checkpoint_shape, [](const Checkpoint& row) -> const std::string& { return row.identity; }, "checkpoint shape");

    const auto raw_flux = static_cast<const ProgramFluxBudgetRecord*>(candidate.flux_budgets.data);
    result.flux_budgets.reserve(table_count(candidate.flux_budgets, "flux budget"));
    for (std::size_t i = 0, n = table_count(candidate.flux_budgets, "flux budget"); i != n; ++i)
      result.flux_budgets.push_back({raw_flux[i].rhs_basis_bound, raw_flux[i].coefficient_term_bound,
                                     raw_flux[i].interface_application_bound, raw_flux[i].interface_identity_character_bound});

    const auto raw_resources = static_cast<const ProgramResourcePlanRecord*>(candidate.resource_plan.data);
    result.resource_plan.reserve(table_count(candidate.resource_plan, "resource plan"));
    std::uint64_t resource_bytes = 0;
    bool has_runtime_sized_resource = false;
    std::string resource_schema;
    std::string resource_digest;
    for (std::size_t i = 0, n = table_count(candidate.resource_plan, "resource plan"); i != n; ++i) {
      const auto& raw = raw_resources[i];
      const bool runtime_flag = (raw.flags & kProgramResourceRuntimeSized) != 0;
      const bool runtime_type = raw.resource_type == ProgramResourcePlanType::runtime_sized;
      if (runtime_flag != runtime_type ||
          (raw.resource_type != ProgramResourcePlanType::exact && !runtime_type))
        throw std::invalid_argument("Program resource plan has a mismatched runtime sizing flag/type");
      if (runtime_type) {
        // A symbolic row is a declaration only.  In particular, zero is not an exact byte count:
        // all four dimensions remain the explicit ABI sentinel until host materialization.
        if (raw.bytes != kProgramResourcePlanUnknownExtent ||
            raw.maximum_bytes != kProgramResourcePlanUnknownExtent ||
            raw.cells != kProgramResourcePlanUnknownExtent ||
            raw.itemsize != kProgramResourcePlanUnknownExtent ||
            (raw.flags & (kProgramResourceHasCells | kProgramResourceHasItemsize)) != 0)
          throw std::invalid_argument(
              "runtime-sized Program resource plan row claims exact extent or bytes");
        has_runtime_sized_resource = true;
      } else if (raw.bytes == 0 || raw.maximum_bytes < raw.bytes ||
                 raw.bytes == kProgramResourcePlanUnknownExtent ||
                 raw.maximum_bytes == kProgramResourcePlanUnknownExtent) {
        throw std::invalid_argument("Program resource plan has an invalid exact byte bound");
      }
      if (raw.struct_size != sizeof(ProgramResourcePlanRecord) ||
          raw.schema_version != kProgramResourcePlanSchemaVersion || raw.reserved != 0 ||
          raw.slot != i || raw.level < -1 || raw.components == 0 ||
          (raw.flags & ~kKnownProgramResourcePlanFlags) != 0 ||
          ((raw.flags & kProgramResourceHasCells) == 0 &&
           raw.cells != kProgramResourcePlanUnknownExtent) ||
          ((raw.flags & kProgramResourceHasItemsize) == 0 &&
           raw.itemsize != kProgramResourcePlanUnknownExtent) ||
          ((raw.flags & kProgramResourceHasCells) != 0 &&
           (raw.cells == 0 || raw.cells == kProgramResourcePlanUnknownExtent)) ||
          ((raw.flags & kProgramResourceHasItemsize) != 0 &&
           (raw.itemsize == 0 || raw.itemsize == kProgramResourcePlanUnknownExtent)))
        throw std::invalid_argument("Program resource plan has an invalid lossless v1 row");
      if (!runtime_type) {
        if (raw.maximum_bytes > std::numeric_limits<std::uint64_t>::max() - resource_bytes)
          throw std::overflow_error("Program resource plan byte bound overflows uint64");
        resource_bytes += raw.maximum_bytes;
      }
      const auto schema = copy(raw.schema, "resource schema");
      const auto digest = copy(raw.plan_digest, "resource plan digest");
      if (schema != "program-resource-plan:v1" || digest.size() != 64 ||
          !std::all_of(digest.begin(), digest.end(), [](unsigned char value) {
            return std::isdigit(value) || (value >= 'a' && value <= 'f');
          }))
        throw std::invalid_argument("Program resource plan has an unauthenticated schema or digest");
      if ((!resource_schema.empty() && resource_schema != schema) ||
          (!resource_digest.empty() && resource_digest != digest))
        throw std::invalid_argument("Program resource plan rows disagree on their authenticated digest");
      resource_schema = schema;
      resource_digest = digest;
      ResourcePlan copied{};
      copied.slot = raw.slot;
      copied.flags = raw.flags;
      copied.value_id = raw.value_id;
      copied.occurrence_path_id = raw.occurrence_path_id;
      copied.level = raw.level;
      copied.components = raw.components;
      copied.ghosts = raw.ghosts;
      if (!runtime_type) {
        copied.bytes = raw.bytes;
        copied.maximum_bytes = raw.maximum_bytes;
      }
      if ((raw.flags & kProgramResourceHasCells) != 0)
        copied.cells = raw.cells;
      if ((raw.flags & kProgramResourceHasItemsize) != 0)
        copied.itemsize = raw.itemsize;
      copied.resource_type = raw.resource_type;
      copied.schema = std::move(schema);
      copied.plan_digest = std::move(digest);
      copied.identity = copy(raw.identity, "resource identity");
      copied.occurrence_path = copy(raw.occurrence_path, "resource occurrence path");
      copied.owner = copy(raw.owner, "resource owner");
      copied.space = copy(raw.space, "resource space");
      copied.clock = copy(raw.clock, "resource clock");
      copied.lifetime = copy(raw.lifetime, "resource lifetime");
      copied.centering = copy(raw.centering, "resource centering");
      copied.off_policy = copy(raw.off_policy, "resource off policy");
      copied.communication = copy(raw.communication, "resource communication");
      copied.transfer_provider = copy(raw.transfer_provider, "resource transfer provider");
      copied.restart_provider = copy(raw.restart_provider, "resource restart provider");
      copied.component_names = copy(raw.component_names, "resource component names");
      copied.shape = copy(raw.shape, "resource shape");
      result.resource_plan.push_back(std::move(copied));
    }
    if (!has_runtime_sized_resource && candidate.maximum_bytes != kProgramResourcePlanUnknownExtent &&
        resource_bytes > candidate.maximum_bytes)
      throw std::invalid_argument("Program resource plan exceeds the candidate memory ceiling");
    require_unique(result.resource_plan, [](const ResourcePlan& row) -> const std::string& {
      return row.identity;
    }, "resource plan");

    const auto materialize_routes = [&](const ProgramAbiTable& table, std::vector<Route>& destination, std::string_view name) {
      const auto raw = static_cast<const ProgramRouteRecord*>(table.data);
      destination.reserve(table_count(table, name));
      for (std::size_t i = 0, n = table_count(table, name); i != n; ++i) {
        if ((raw[i].capability_bits & ~kKnownProgramCapabilityBits) != 0)
          throw std::invalid_argument("Program route table contains unknown capability bits");
        destination.push_back({copy(raw[i].identity, "route identity"), copy(raw[i].kind, "route kind"), raw[i].capability_bits});
      }
      require_unique(destination, [](const Route& row) -> const std::string& { return row.identity; }, name);
    };
    materialize_routes(candidate.boundary_routes, result.boundary_routes, "boundary route");
    materialize_routes(candidate.provider_routes, result.provider_routes, "provider route");
    const auto materialize_modules = [&](const ProgramAbiTable& table, std::vector<Module>& destination,
                                         std::string_view name) {
      const auto raw = static_cast<const ProgramModuleRecord*>(table.data);
      destination.reserve(table_count(table, name));
      for (std::size_t i = 0, n = table_count(table, name); i != n; ++i)
        destination.push_back({copy(raw[i].identity, "module identity"), copy(raw[i].kind, "module kind"),
                               copy(raw[i].signature, "module signature"), copy(raw[i].requirements, "module requirements"),
                               copy(raw[i].owner, "module owner")});
      require_unique(destination, [](const Module& row) -> const std::string& { return row.identity; }, name);
    };
    materialize_modules(candidate.module_operators, result.module_operators, "module operator");
    materialize_modules(candidate.module_state_spaces, result.module_state_spaces, "module state space");
    materialize_modules(candidate.module_field_spaces, result.module_field_spaces, "module field space");
    return result;
  }

  /// Reconstruct the exact Python lowering payload (keys sorted by ``json.dumps(sort_keys=True)``)
  /// and authenticate it independently of the DSO-provided digest.  Resource strings are required
  /// to be generated ASCII JSON; this is intentional because the emitter uses ``ensure_ascii``.
  /// A null top-level ceiling is the symbolic v5 form.  It is only valid while at least one row is
  /// runtime-sized; a final host materialization always supplies an exact aggregate ceiling.
  [[nodiscard]] std::string canonical_resource_digest_payload(
      std::optional<std::uint64_t> maximum_bytes) const {
    std::string result{"{\"entries\":["};
    bool has_symbolic_plan_row = false;
    for (std::size_t index = 0; index != resource_plan.size(); ++index) {
      const auto& row = resource_plan[index];
      const std::size_t component_names = detail::canonical_component_name_count(row.component_names);
      (void)detail::canonical_shape_extent_count(row.shape);
      if (component_names != 0 && component_names != row.components)
        throw std::invalid_argument("Program resource component_names disagrees with components");
      const bool runtime_flag = (row.flags & kProgramResourceRuntimeSized) != 0;
      const bool runtime_type = row.resource_type == ProgramResourcePlanType::runtime_sized;
      if ((row.flags & ~kKnownProgramResourcePlanFlags) != 0 ||
          (row.resource_type != ProgramResourcePlanType::exact && !runtime_type) ||
          runtime_flag != runtime_type)
        throw std::invalid_argument("Program resource row has a mismatched runtime sizing flag/type");
      const bool runtime = runtime_type;
      has_symbolic_plan_row = has_symbolic_plan_row || runtime;
      if (runtime != !row.bytes.has_value() || runtime != !row.maximum_bytes.has_value() ||
          (runtime && (row.cells || row.itemsize)) ||
          (!runtime && (!row.bytes || !row.maximum_bytes)))
        throw std::invalid_argument("Program resource row has an inconsistent runtime sizing declaration");
      if (index != 0) result.push_back(',');
      result += "{\"bytes\":" +
                (row.bytes ? detail::resource_json_u64(*row.bytes) : std::string{"null"});
      result += ",\"cells\":" +
                (row.cells ? detail::resource_json_u64(*row.cells) : std::string{"null"});
      result += ",\"centering\":" + detail::resource_json_string(row.centering);
      result += ",\"communicates\":" + std::string{row.flags & kProgramResourceCommunicates ? "true" : "false"};
      result += ",\"communication\":" + detail::resource_json_string(row.communication);
      result += ",\"component_names\":" + row.component_names;
      result += ",\"components\":" + detail::resource_json_u64(row.components);
      result += ",\"ghosts\":" + detail::resource_json_u64(row.ghosts);
      result += ",\"itemsize\":" +
                (row.itemsize ? detail::resource_json_u64(*row.itemsize) : std::string{"null"});
      result += ",\"key\":{\"clock\":" + detail::resource_json_string(row.clock);
      result += ",\"level\":" +
                (row.level < 0 ? std::string{"null"} : std::to_string(row.level));
      result += ",\"occurrence_path\":" + detail::resource_json_string(row.occurrence_path);
      result += ",\"occurrence_path_id\":" + detail::resource_json_u64(row.occurrence_path_id);
      result += ",\"owner\":" + detail::resource_json_string(row.owner);
      result += ",\"space\":" + detail::resource_json_string(row.space);
      result += ",\"value_id\":" + detail::resource_json_u64(row.value_id) + "}";
      result += ",\"lifetime\":" + detail::resource_json_string(row.lifetime);
      result += ",\"maximum_bytes\":" +
                (row.maximum_bytes ? detail::resource_json_u64(*row.maximum_bytes)
                                    : std::string{"null"});
      result += ",\"off_policy\":" + detail::resource_json_string(row.off_policy);
      result += ",\"resource_type\":" + detail::resource_json_string(
          runtime ? "runtime_sized" : "exact");
      result += ",\"restart_provider\":" + detail::resource_json_string(row.restart_provider);
      result += ",\"restart_required\":" + std::string{row.flags & kProgramResourceRestartRequired ? "true" : "false"};
      result += ",\"runtime_sized\":" + std::string{runtime ? "true" : "false"};
      result += ",\"shape\":" + row.shape;
      result += ",\"slot\":" + detail::resource_json_u64(row.slot);
      result += ",\"transfer_provider\":" + detail::resource_json_string(row.transfer_provider);
      result += "}";
    }
    result += "],\"maximum_bytes\":";
    std::uint64_t row_total = 0;
    bool has_symbolic_row = false;
    for (const auto& row : resource_plan) {
      if (!row.maximum_bytes) {
        has_symbolic_row = true;
        continue;
      }
      if (*row.maximum_bytes > std::numeric_limits<std::uint64_t>::max() - row_total)
        throw std::overflow_error("Program resource manifest memory ceiling overflows uint64");
      row_total += *row.maximum_bytes;
    }
    if (has_symbolic_row != !maximum_bytes)
      throw std::invalid_argument("Program resource manifest has an inconsistent symbolic ceiling");
    if (maximum_bytes && row_total > *maximum_bytes)
      throw std::invalid_argument("Program resource manifest exceeds its declared memory ceiling");
    result += maximum_bytes ? detail::resource_json_u64(*maximum_bytes) : std::string{"null"};
    if (!has_symbolic_plan_row && !materialized_layout_manifest.empty())
      result += ",\"prepared_layouts\":" + materialized_layout_manifest;
    result += ",\"schema\":\"program-resource-plan:v1\",\"schema_version\":1}";
    return result;
  }

  [[nodiscard]] std::string canonical_resource_digest_payload(std::uint64_t maximum_bytes) const {
    return canonical_resource_digest_payload(std::optional<std::uint64_t>{maximum_bytes});
  }

  [[nodiscard]] std::string canonical_resource_manifest(std::uint64_t maximum_bytes,
                                                        std::string_view digest) const {
    const std::string payload = canonical_resource_digest_payload(maximum_bytes);
    return "{\"digest\":" + detail::resource_json_string(digest) +
           ",\"entries\":" + payload.substr(std::string{"{\"entries\":"}.size(),
                                                        payload.size() - std::string{"{\"entries\":"}.size() - 1) + "}";
  }

  [[nodiscard]] std::string canonical_resource_manifest(std::optional<std::uint64_t> maximum_bytes,
                                                        std::string_view digest) const {
    const std::string payload = canonical_resource_digest_payload(maximum_bytes);
    const std::string prefix{"{\"entries\":"};
    return "{\"digest\":" + detail::resource_json_string(digest) +
           ",\"entries\":" + payload.substr(prefix.size(), payload.size() - prefix.size() - 1) + "}";
  }

  void validate_resource_authority(const ProgramInstallationMetadata& metadata,
                                   std::uint64_t candidate_ceiling) const {
    const bool has_symbolic_row = std::any_of(
        resource_plan.begin(), resource_plan.end(),
        [](const ResourcePlan& row) { return row.runtime_sized(); });
    const std::optional<std::uint64_t> exact_ceiling =
        candidate_ceiling == kProgramResourcePlanUnknownExtent
            ? std::nullopt
            : std::optional<std::uint64_t>{candidate_ceiling};
    if (resource_plan.empty() && !exact_ceiling)
      throw std::invalid_argument("empty Program resource plan cannot use a symbolic ceiling");
    if (has_symbolic_row != !exact_ceiling)
      throw std::invalid_argument("Program resource plan and candidate ceiling disagree on symbolic sizing");
    if (resource_plan.empty() && candidate_ceiling != 0)
      throw std::invalid_argument("Program has a resource ceiling without a resource plan");
    const std::string digest_payload = canonical_resource_digest_payload(exact_ceiling);
    const auto bytes = std::vector<std::uint8_t>(digest_payload.begin(), digest_payload.end());
    const std::string computed = identity::sha256_hex(bytes);
    if (!resource_plan.empty() && computed != resource_plan.front().plan_digest)
      throw std::invalid_argument("Program resource plan digest does not authenticate its rows");
    const std::string resource_manifest = canonical_resource_manifest(exact_ceiling, computed);
    const auto exact_count = [](std::string_view haystack, std::string_view needle) {
      std::size_t count = 0, offset = 0;
      while ((offset = haystack.find(needle, offset)) != std::string_view::npos) {
        ++count;
        offset += needle.size();
      }
      return count;
    };
    const std::string plan_needle = "\"resource_plan\":" + resource_manifest;
    const std::string digest_needle = "\"resource_plan_digest\":" +
                                      detail::resource_json_string(computed);
    if (exact_count(metadata.persistent_resource_manifest, plan_needle) != 1 ||
        exact_count(metadata.persistent_resource_manifest, digest_needle) != 1)
      throw std::invalid_argument("Program persistent resource manifest differs from the authenticated rows");
    std::uint64_t total = 0;
    for (const auto& row : resource_plan) {
      if ((row.flags & kProgramResourceRestartRequired) != 0 && row.restart_provider == "none")
        throw std::invalid_argument("Program restart-required resource has no restart provider");
      if (row.transfer_provider == "qualified_regrid_provider")
        throw std::invalid_argument("Program qualified regrid resource has no provider identity");
      if (!row.maximum_bytes)
        continue;
      if (*row.maximum_bytes > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("Program resource plan memory ceiling overflows uint64");
      total += *row.maximum_bytes;
    }
    if (exact_ceiling && total > *exact_ceiling)
      throw std::invalid_argument("Program resource plan exceeds the declared memory ceiling");
  }

  [[nodiscard]] bool has_runtime_sized_resources() const noexcept {
    return std::any_of(resource_plan.begin(), resource_plan.end(),
                       [](const ResourcePlan& row) { return row.runtime_sized(); });
  }

  /// Read-only declaration accessors intentionally remain available before sealing.  They expose
  /// dense declaration slots only; no value-id-to-slot lookup is created until an exact plan is
  /// materialized and bound.
  [[nodiscard]] std::size_t resource_slot_count() const noexcept {
    return resource_plan.size();
  }

  [[nodiscard]] const std::vector<ResourcePlan>& resource_declarations() const noexcept {
    return resource_plan;
  }

  [[nodiscard]] const ResourcePlan& resource_declaration(std::size_t slot) const {
    if (slot >= resource_plan.size())
      throw std::out_of_range("Program resource declaration slot is outside the copied tables");
    return resource_plan[slot];
  }

  [[nodiscard]] std::string_view prepared_layout_manifest() const noexcept {
    return materialized_layout_manifest;
  }

  /// Materialize the symbolic v5 rows after candidate preparation.
  ///
  /// Every entry in ``prototypes`` identifies one exact allocation family by ``slot`` and
  /// ``subslot``.  The complete ``cells`` extent is multiplied by the exact item size and
  /// component count with checked arithmetic; optional byte/bound assertions are compared with
  /// that product.  All subslots are summed into their owning slot, so transient scratch families
  /// and persistent-schedule values cannot disappear behind a logical one-cell fallback.  The
  /// returned plan carries a newly computed host digest and is never the symbolic DSO digest.
  [[nodiscard]] ProgramResourcePlan materialize_resource_plan(
      std::span<const ResourcePrototype> prototypes,
      std::optional<std::uint64_t> candidate_ceiling = std::nullopt) const {
    if (candidate_ceiling && *candidate_ceiling == kProgramResourcePlanUnknownExtent)
      candidate_ceiling.reset();
    for (const auto& row : resource_plan) {
      const bool runtime_flag = (row.flags & kProgramResourceRuntimeSized) != 0;
      const bool runtime_type = row.resource_type == ProgramResourcePlanType::runtime_sized;
      if ((row.flags & ~kKnownProgramResourcePlanFlags) != 0 ||
          (row.resource_type != ProgramResourcePlanType::exact && !runtime_type) ||
          runtime_flag != runtime_type)
        throw std::invalid_argument("Program resource row has a mismatched runtime sizing flag/type");
    }
    if (!has_runtime_sized_resources()) {
      if (!prototypes.empty())
        throw std::invalid_argument(
            "static Program resource plan received runtime layout prototypes");
      std::uint64_t total = 0;
      for (const auto& row : resource_plan) {
        if (!row.maximum_bytes)
          throw std::invalid_argument("static Program resource plan has an unresolved byte bound");
        if (*row.maximum_bytes > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error("Program resource plan maximum overflows uint64");
        total += *row.maximum_bytes;
      }
      return make_program_resource_plan(*this, candidate_ceiling.value_or(total));
    }

    struct Accumulator final {
      std::uint64_t bytes = 0;
      std::uint64_t maximum_bytes = 0;
      std::size_t prototype_count = 0;
      std::optional<std::uint64_t> cells;
      std::optional<std::uint64_t> itemsize;
    };
    struct PreparedPrototype final {
      std::uint32_t slot = 0;
      std::int32_t subslot = 0;
      ResourcePrototypeKind kind = ResourcePrototypeKind::generic;
      std::uint64_t cells = 0;
      std::uint64_t itemsize = 0;
      std::uint32_t components = 0;
      std::uint32_t ghosts = 0;
      std::uint64_t bytes = 0;
      std::uint64_t maximum_bytes = 0;
      std::vector<std::uint64_t> shape;
    };
    std::vector<Accumulator> accumulators(resource_plan.size());
    std::vector<std::tuple<ResourcePrototypeKind, std::uint32_t, std::int32_t>> seen;
    seen.reserve(prototypes.size());
    std::vector<PreparedPrototype> prepared_prototypes;
    prepared_prototypes.reserve(prototypes.size());

    const auto checked_add = [](std::uint64_t left, std::uint64_t right,
                                std::string_view what) -> std::uint64_t {
      if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("Program resource " + std::string(what) +
                                 " overflows uint64");
      return left + right;
    };
    const auto checked_product = [](std::uint64_t left, std::uint64_t right,
                                    std::string_view what) -> std::uint64_t {
      if (left == 0 || right == 0)
        throw std::invalid_argument("Program resource " + std::string(what) +
                                    " must be positive");
      if (left > std::numeric_limits<std::uint64_t>::max() / right)
        throw std::overflow_error("Program resource " + std::string(what) +
                                  " overflows uint64");
      return left * right;
    };

    for (const auto& prototype : prototypes) {
      if (prototype.slot >= resource_plan.size() || prototype.subslot < 0)
        throw std::invalid_argument(
            "Program resource prototype targets an absent slot or negative subslot");
      const auto& row = resource_plan[prototype.slot];
      if (!row.runtime_sized())
        throw std::invalid_argument(
            "Program resource prototype targets an exact, non-runtime-sized row");
      (void)resource_prototype_kind_name(prototype.kind);
      const auto key = std::tuple{prototype.kind, prototype.slot, prototype.subslot};
      if (std::find(seen.begin(), seen.end(), key) != seen.end())
        throw std::invalid_argument("duplicate Program resource prototype kind/slot/subslot");
      seen.push_back(key);

      const auto& layout = prototype.layout;
      if (layout.components == 0)
        throw std::invalid_argument("Program resource prototype component count is zero");
      for (const auto extent : layout.shape)
        if (extent == 0)
          throw std::invalid_argument("Program resource prototype shape has a zero extent");
      const auto component_bytes = checked_product(layout.components, layout.itemsize, "byte size");
      const auto bytes = checked_product(component_bytes, layout.cells, "byte size");
      if (layout.bytes && *layout.bytes != bytes)
        throw std::invalid_argument("Program resource prototype byte size disagrees with its layout");
      const auto maximum = layout.maximum_bytes.value_or(bytes);
      if (maximum < bytes)
        throw std::invalid_argument("Program resource prototype maximum is below its bytes");

      auto& accumulator = accumulators[prototype.slot];
      accumulator.bytes = checked_add(accumulator.bytes, bytes, "byte total");
      accumulator.maximum_bytes =
          checked_add(accumulator.maximum_bytes, maximum, "maximum byte total");
      ++accumulator.prototype_count;
      prepared_prototypes.push_back({prototype.slot, prototype.subslot, prototype.kind,
                                     layout.cells, layout.itemsize, layout.components,
                                     layout.ghosts, bytes, maximum, layout.shape});
      if (accumulator.prototype_count == 1) {
        accumulator.cells = layout.cells;
        accumulator.itemsize = layout.itemsize;
      } else {
        // There is no single truthful cells/itemsize value for an aggregate with multiple
        // subslots.  Leave both absent in the final lossless row; bytes/max remain exact totals.
        accumulator.cells.reset();
        accumulator.itemsize.reset();
      }
    }

    ProgramInstallationTables resolved = *this;
    std::uint64_t total = 0;
    for (std::size_t slot = 0; slot != resource_plan.size(); ++slot) {
      auto& row = resolved.resource_plan[slot];
      if (row.runtime_sized()) {
        const auto& accumulator = accumulators[slot];
        if (accumulator.prototype_count == 0)
          throw std::invalid_argument(
              "unresolved runtime-sized Program resource slot " + std::to_string(slot));
        row.bytes = accumulator.bytes;
        row.maximum_bytes = accumulator.maximum_bytes;
        row.cells = accumulator.cells;
        row.itemsize = accumulator.itemsize;
        row.flags &= ~kProgramResourceRuntimeSized;
        row.flags &= ~(kProgramResourceHasCells | kProgramResourceHasItemsize);
        if (row.cells) row.flags |= kProgramResourceHasCells;
        if (row.itemsize) row.flags |= kProgramResourceHasItemsize;
        row.resource_type = ProgramResourcePlanType::exact;
      }
      if (!row.bytes || !row.maximum_bytes)
        throw std::invalid_argument("Program resource plan remains unresolved after materialization");
      total = checked_add(total, *row.maximum_bytes, "aggregate maximum");
    }
    if (candidate_ceiling && total > *candidate_ceiling)
      throw std::invalid_argument("materialized Program resource plan exceeds the host budget");

    std::sort(prepared_prototypes.begin(), prepared_prototypes.end(),
              [](const PreparedPrototype& left, const PreparedPrototype& right) {
                const auto left_kind = resource_prototype_kind_name(left.kind);
                const auto right_kind = resource_prototype_kind_name(right.kind);
                return std::tie(left_kind, left.slot, left.subslot) <
                       std::tie(right_kind, right.slot, right.subslot);
              });
    std::string layout_manifest{"["};
    for (std::size_t index = 0; index != prepared_prototypes.size(); ++index) {
      if (index != 0) layout_manifest.push_back(',');
      const auto& prototype = prepared_prototypes[index];
      layout_manifest += "{\"bytes\":" + detail::resource_json_u64(prototype.bytes);
      layout_manifest += ",\"cells\":" + detail::resource_json_u64(prototype.cells);
      layout_manifest += ",\"components\":" + detail::resource_json_u64(prototype.components);
      layout_manifest += ",\"ghosts\":" + detail::resource_json_u64(prototype.ghosts);
      layout_manifest += ",\"itemsize\":" + detail::resource_json_u64(prototype.itemsize);
      layout_manifest += ",\"kind\":" +
                         detail::resource_json_string(resource_prototype_kind_name(prototype.kind));
      layout_manifest += ",\"maximum_bytes\":" +
                         detail::resource_json_u64(prototype.maximum_bytes);
      layout_manifest += ",\"shape\":[";
      for (std::size_t extent = 0; extent != prototype.shape.size(); ++extent) {
        if (extent != 0) layout_manifest.push_back(',');
        layout_manifest += detail::resource_json_u64(prototype.shape[extent]);
      }
      layout_manifest += "]";
      layout_manifest += ",\"slot\":" + detail::resource_json_u64(prototype.slot);
      layout_manifest += ",\"subslot\":" + std::to_string(prototype.subslot) + "}";
    }
    layout_manifest += "]";

    // A materialized plan has an exact aggregate ceiling equal to the sum of all per-slot
    // maxima.  Recompute the payload after replacing every symbolic row and then authenticate
    // that host-owned payload; the DSO's symbolic digest is intentionally not reused.
    resolved.materialized_layout_manifest = std::move(layout_manifest);
    const std::string payload = resolved.canonical_resource_digest_payload(total);
    const std::string digest = identity::sha256_hex(
        std::vector<std::uint8_t>(payload.begin(), payload.end()));
    for (auto& row : resolved.resource_plan)
      row.plan_digest = digest;
    auto result = make_program_resource_plan(resolved, total);
    // Publish the exact rows and the canonical prepared-layout evidence together with the returned
    // plan.  A later PreparedArtifactPublication copies these tables, so leaving the source rows
    // symbolic would create two contradictory authorities in one receipt.
    resource_plan = resolved.resource_plan;
    materialized_layout_manifest = resolved.materialized_layout_manifest;
    return result;
  }

  [[nodiscard]] ProgramResourcePlan materialize_resource_plan(
      const std::vector<ResourcePrototype>& prototypes,
      std::optional<std::uint64_t> candidate_ceiling = std::nullopt) const {
    return materialize_resource_plan(std::span<const ResourcePrototype>(prototypes),
                                     candidate_ceiling);
  }
};

/// Translate the copied v5 POD rows into the bind-sealed runtime inventory.  All string to compact
/// id work is deliberately here, before candidate preparation; warm generated calls use the row's
/// dense ``slot`` only.  The full strings remain in ``ProgramInstallationTables`` for diagnostics.
[[nodiscard]] inline ProgramResourcePlan make_program_resource_plan(
    const ProgramInstallationTables& tables, std::uint64_t ceiling) {
  if (tables.has_runtime_sized_resources())
    throw std::invalid_argument(
        "runtime-sized Program resource plan must be host-materialized before binding");
  if (tables.resource_plan.empty())
    if (ceiling != 0)
      throw std::invalid_argument("Program has a resource ceiling without a resource plan");
  if (tables.resource_plan.empty()) {
    const std::string payload = tables.canonical_resource_digest_payload(0);
    return ProgramResourcePlan({}, 0, "program-resource-plan:v1",
                               identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(),
                                                                             payload.end())));
  }
  const auto compact_identity = [](const std::vector<std::string>& values, const std::string& value) {
    const auto found = std::find(values.begin(), values.end(), value);
    if (found == values.end())
      throw std::logic_error("Program resource identity table was not preinterned");
    return static_cast<std::uint32_t>(std::distance(values.begin(), found));
  };
  std::vector<std::string> owners, spaces, clocks;
  owners.reserve(tables.resource_plan.size()); spaces.reserve(tables.resource_plan.size());
  clocks.reserve(tables.resource_plan.size());
  const auto intern = [](std::vector<std::string>& identities, const std::string& identity) {
    if (std::find(identities.begin(), identities.end(), identity) == identities.end())
      identities.push_back(identity);
  };
  for (const auto& row : tables.resource_plan) {
    intern(owners, row.owner); intern(spaces, row.space); intern(clocks, row.clock);
  }
  std::vector<ProgramResourcePlanEntry> entries;
  entries.reserve(tables.resource_plan.size());
  const auto lifetime = [](const std::string& value) {
    if (value == "transient") return ProgramValueLifetime::transient;
    if (value == "persistent" || value == "persistent_schedule")
      return ProgramValueLifetime::persistent_schedule;
    throw std::invalid_argument("Program resource plan has an unknown lifetime");
  };
  const auto centering = [](const std::string& value) {
    if (value == "cell") return ProgramValueCentering::cell;
    if (value == "face") return ProgramValueCentering::face;
    if (value == "node") return ProgramValueCentering::node;
    throw std::invalid_argument("Program resource plan has an unknown centering");
  };
  const auto off_policy = [](const std::string& value) {
    if (value == "none") return ProgramScheduleOffPolicy::none;
    if (value == "hold") return ProgramScheduleOffPolicy::hold;
    if (value == "accumulate_dt") return ProgramScheduleOffPolicy::accumulate_dt;
    if (value == "zero") return ProgramScheduleOffPolicy::zero;
    if (value == "error") return ProgramScheduleOffPolicy::error;
    throw std::invalid_argument("Program resource plan has an unknown off-schedule policy");
  };
  const auto transfer = [](const std::string& value) {
    if (value == "none") return ProgramSpatialTransferPolicy::refuse;
    if (value == "redistribute_exact")
      return ProgramSpatialTransferPolicy::redistribute_exact;
    // Every other non-empty identity is an explicit qualified provider. Its registry resolution
    // is performed by the detached prepared image in B2; it is never silently coerced to exact
    // redistribution here.
    return ProgramSpatialTransferPolicy::qualified_regrid_provider;
  };
  for (const auto& row : tables.resource_plan) {
    ProgramResourcePlanEntry entry;
    entry.slot = row.slot;
    entry.identity = row.identity;
    entry.key = {row.value_id, row.occurrence_path_id, compact_identity(owners, row.owner),
                 compact_identity(spaces, row.space), compact_identity(clocks, row.clock), row.level};
    entry.occurrence_path = row.occurrence_path;
    entry.owner_identity = row.owner; entry.space_identity = row.space; entry.clock_identity = row.clock;
    entry.lifetime = lifetime(row.lifetime); entry.centering = centering(row.centering);
    entry.off_policy = off_policy(row.off_policy); entry.spatial_transfer = transfer(row.transfer_provider);
    if (!row.bytes || !row.maximum_bytes)
      throw std::invalid_argument("Program resource plan has an unresolved exact byte footprint");
    entry.components = row.components; entry.ghosts = row.ghosts; entry.bytes = *row.bytes;
    entry.maximum_bytes = *row.maximum_bytes;
    entry.communicates = (row.flags & kProgramResourceCommunicates) != 0;
    entry.restart_required = (row.flags & kProgramResourceRestartRequired) != 0;
    entry.communication = row.communication;
    entry.transfer_identity = row.transfer_provider; entry.restart_identity = row.restart_provider;
    entry.component_names = row.component_names; entry.shape = row.shape;
    entry.cells = row.cells; entry.itemsize = row.itemsize;
    entries.push_back(std::move(entry));
  }
  return ProgramResourcePlan(std::move(entries), ceiling, tables.resource_plan.front().schema,
                             tables.resource_plan.front().plan_digest);
}

/// Move-only host shell.  Candidate state is always destroyed before the library handle is closed;
/// this ordering is also used for a refused replacement and runtime teardown.
class OwnedProgramInstallation final {
 public:
  OwnedProgramInstallation() = default;
  OwnedProgramInstallation(pops::dynlib::UniqueHandle image, ProgramCandidateDescriptor candidate,
                           ProgramInstallationMetadata metadata,
                           ProgramInstallationTables tables = {}) noexcept
      : image_(std::move(image)), candidate_(candidate), metadata_(std::move(metadata)),
        tables_(std::move(tables)) {}
  OwnedProgramInstallation(const OwnedProgramInstallation&) = delete;
  OwnedProgramInstallation& operator=(const OwnedProgramInstallation&) = delete;
  OwnedProgramInstallation(OwnedProgramInstallation&& other) noexcept
      : image_(std::move(other.image_)),
        candidate_(std::exchange(other.candidate_, {})),
        metadata_(std::move(other.metadata_)), tables_(std::move(other.tables_)),
        preparation_image_(std::move(other.preparation_image_)),
        prepared_(std::exchange(other.prepared_, false)) {}
  OwnedProgramInstallation& operator=(OwnedProgramInstallation&& other) noexcept {
    if (this != &other) {
      reset();
      if (other.image_) {
        image_.emplace(std::move(*other.image_));
        other.image_.reset();
      }
      candidate_ = std::exchange(other.candidate_, {});
      metadata_ = std::move(other.metadata_);
      tables_ = std::move(other.tables_);
      preparation_image_ = std::move(other.preparation_image_);
      prepared_ = std::exchange(other.prepared_, false);
    }
    return *this;
  }
  ~OwnedProgramInstallation() { reset(); }

  [[nodiscard]] explicit operator bool() const noexcept { return candidate_.step != nullptr; }
  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] const ProgramCandidateDescriptor& candidate() const noexcept { return candidate_; }
  [[nodiscard]] const ProgramInstallationMetadata& metadata() const noexcept { return metadata_; }
  [[nodiscard]] const ProgramInstallationTables& tables() const noexcept { return tables_; }
  [[nodiscard]] bool resource_plan_requires_materialization() const noexcept {
    return tables_.has_runtime_sized_resources();
  }

  /// Replace the symbolic resource object embedded in the copied temporal metadata with the
  /// host-authenticated materialized object.  ``previous_resource_manifest`` is captured before
  /// table materialization because the source rows (and their digest) are updated atomically by
  /// ``ProgramInstallationTables::materialize_resource_plan``.  Other temporal metadata members
  /// remain byte-for-byte untouched; only resource_plan, resource_plan_digest and, when present,
  /// resource_plan_maximum_bytes are replaced.
  void publish_materialized_resource_manifest(std::string previous_resource_manifest,
                                              std::string previous_digest,
                                              const ProgramResourcePlan& final_plan) {
    if (!tables_.has_runtime_sized_resources() && tables_.resource_plan.empty())
      throw std::logic_error("materialized resource publication has no resource rows");
    if (final_plan.entries().empty())
      throw std::invalid_argument("materialized resource publication has an empty plan");
    const std::string final_manifest =
        tables_.canonical_resource_manifest(final_plan.maximum_bytes(), final_plan.digest());
    ProgramInstallationMetadata final_metadata = metadata_;
    detail::replace_resource_manifest_member(final_metadata.persistent_resource_manifest,
                                             "\"resource_plan\":", previous_resource_manifest,
                                             final_manifest);
    detail::replace_resource_manifest_member(
        final_metadata.persistent_resource_manifest, "\"resource_plan_digest\":",
        detail::resource_json_string(previous_digest),
        detail::resource_json_string(final_plan.digest()));
    const std::string previous_ceiling =
        candidate_.maximum_bytes == kProgramResourcePlanUnknownExtent
            ? std::string{"null"}
            : detail::resource_json_u64(candidate_.maximum_bytes);
    (void)detail::replace_resource_manifest_member_if_present(
        final_metadata.persistent_resource_manifest, "\"resource_plan_maximum_bytes\":",
        previous_ceiling, detail::resource_json_u64(final_plan.maximum_bytes()));
    // Revalidate the final pair before changing metadata_.  This proves that the copied tables,
    // prepared-layout receipt, exact digest and metadata object all describe one authority.
    tables_.validate_resource_authority(final_metadata, final_plan.maximum_bytes());
    metadata_ = std::move(final_metadata);
  }

  /// Retain the exact typed preparation image until candidate destruction.  Generated step
  /// closures own execution services through that image, so releasing it before the DSO context is
  /// destroyed would turn a successful replacement into a dangling callback.
  void set_preparation_image(std::shared_ptr<ProgramPreparationImage> image) {
    if (prepared_)
      throw std::logic_error("Program preparation image must be installed before prepare");
    if (!image)
      throw std::invalid_argument("Program preparation image cannot be null");
    preparation_image_ = std::move(image);
  }

  [[nodiscard]] const std::shared_ptr<ProgramPreparationImage>& preparation_image() const noexcept {
    return preparation_image_;
  }

  void prepare(const ProgramHostDescriptor& host) {
    if (!valid_program_host_descriptor(host))
      throw std::invalid_argument("Program preparation received an invalid v5 host descriptor");
    if (prepared_)
      throw std::logic_error("Program installation candidate was already prepared");
    if (candidate_.prepare == nullptr)
      throw std::logic_error("Program installation has no preparation callback");
    if (!preparation_image_ || host.preparation.image != preparation_image_.get())
      throw std::invalid_argument(
          "Program preparation requires its retained typed host image");
    const auto& typed_image = require_program_execution_preparation_image(
        host.preparation, host.native_dimension, host.runtime_kind);
    if (&typed_image != preparation_image_.get())
      throw std::invalid_argument("Program preparation image ownership differs from the host view");
    ProgramInstallDiagnostic diagnostic{};
    const bool accepted = pops::dynlib::invoke_with_host_exception(
        [&] { return candidate_.prepare(candidate_.context, &host, &diagnostic); },
        "Program candidate prepare");
    if (!accepted) {
      std::size_t size = 0;
      while (size != sizeof(diagnostic.message) && diagnostic.message[size] != '\0')
        ++size;
      throw std::runtime_error("Program preparation refused: " + std::string(diagnostic.message, size));
    }
    prepared_ = true;
  }

  /// Invoke the DSO-owned candidate only while this owner still retains its image.  Foreign
  /// exception objects are copied into host-owned diagnostics before the DSO frame unwinds.
  void invoke_step(double dt) const {
    require_prepared_();
    if (candidate_.step == nullptr)
      throw std::logic_error("Program installation has no invocable step candidate");
    pops::dynlib::invoke_with_host_exception(
        [&] { candidate_.step(candidate_.context, dt); }, "Program candidate step");
  }

  /// Return no value when the artifact did not declare a dt callback.  As with `invoke_step`, the
  /// DSO remains owned throughout the callback and any foreign exception is rethrown by the host.
  [[nodiscard]] std::optional<double> invoke_dt_bound(double cfl) const {
    require_prepared_();
    if (candidate_.dt_bound == nullptr)
      return std::nullopt;
    return pops::dynlib::invoke_with_host_exception(
        [&] { return candidate_.dt_bound(candidate_.context, cfl); }, "Program candidate dt bound");
  }

  void invoke_hierarchy_refresh() const {
    invoke_lifecycle_hook(candidate_.hierarchy_refresh, "Program candidate hierarchy refresh");
  }

  void invoke_history_remap_accepted(const void* remap_descriptor) const {
    require_prepared_();
    if (candidate_.history_remap_accepted == nullptr)
      throw std::logic_error("Program installation has no accepted history-remap candidate");
    pops::dynlib::invoke_with_host_exception(
        [&] { candidate_.history_remap_accepted(candidate_.context, remap_descriptor); },
        "Program candidate history remap");
  }

  void invoke_restart_regrid_preflight() const {
    invoke_lifecycle_hook(candidate_.restart_regrid_preflight,
                          "Program candidate restart regrid preflight");
  }

  void invoke_restart_regrid() const {
    invoke_lifecycle_hook(candidate_.restart_regrid, "Program candidate restart regrid");
  }

  void invoke_restart_resync() const {
    invoke_lifecycle_hook(candidate_.restart_resync, "Program candidate restart resync");
  }

  [[nodiscard]] std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>
  invoke_accepted_snapshot() const;

  void reset() noexcept {
    if (candidate_.destroy && candidate_.context)
      candidate_.destroy(candidate_.context);
    candidate_ = {};
    metadata_ = {};
    tables_ = {};
    preparation_image_.reset();
    prepared_ = false;
    // UniqueHandle closes only after candidate destruction above.
    image_.reset();
  }

 private:
  void invoke_lifecycle_hook(ProgramCandidateDescriptor::RestartHookFn hook,
                             std::string_view operation) const {
    require_prepared_();
    if (hook == nullptr)
      throw std::logic_error("Program installation has no invocable lifecycle candidate");
    pops::dynlib::invoke_with_host_exception([&] { hook(candidate_.context); }, operation);
  }
  void require_prepared_() const {
    if (!prepared_)
      throw std::logic_error("Program installation candidate has not been prepared");
  }

  std::optional<pops::dynlib::UniqueHandle> image_;
  ProgramCandidateDescriptor candidate_{};
  ProgramInstallationMetadata metadata_{};
  ProgramInstallationTables tables_{};
  std::shared_ptr<ProgramPreparationImage> preparation_image_;
  bool prepared_ = false;
};

/// Fully host-owned installation handle prepared before runtime publication.  The handle retains
/// one owner for the candidate, host-owned metadata/tables, preparation image, and DSO.  The owner
/// still destroys foreign candidate state before dlclose.
class PreparedProgramInstallation final {
 public:
  struct PublicationPayload final {
    OwnedProgramInstallation owner;
    ProgramResourcePlan resource_plan;
    ProgramPersistentValueStore persistent_values;
    std::uint64_t generation = 0;
  };

  /// Construct only from a candidate whose preparation callback has already completed.
  ///
  /// The prepared handle is deliberately a thin, immutable view over its owner.  Metadata and
  /// tables stay in the owner so there is one host-owned authority for the artifact description;
  /// moving them into a second image would leave the retained owner deceptively empty.
  explicit PreparedProgramInstallation(OwnedProgramInstallation owner)
      : owner_(std::move(owner)) {
    if (!owner_.prepared())
      throw std::invalid_argument(
          "Prepared program installation requires an owner prepared by its candidate");
    if (!owner_.preparation_image() || owner_.preparation_image()->generation() == 0)
      throw std::invalid_argument(
          "Prepared program installation requires a generated host preparation image");
    // Fully static and empty plans retain the historical eager path.  A symbolic row is only an
    // ABI declaration: its candidate has not yet exposed the exact prototype/layout set, so no
    // bind-sealed plan (and therefore no publication payload) may be fabricated here.
    if (!owner_.resource_plan_requires_materialization()) {
      resource_plan_ = make_program_resource_plan(owner_.tables(), owner_.candidate().maximum_bytes);
      persistent_values_.bind(resource_plan_);
      resource_plan_sealed_ = true;
    }
  }
  PreparedProgramInstallation(const PreparedProgramInstallation&) = delete;
  PreparedProgramInstallation& operator=(const PreparedProgramInstallation&) = delete;
  PreparedProgramInstallation(PreparedProgramInstallation&&) noexcept = default;
  PreparedProgramInstallation& operator=(PreparedProgramInstallation&&) noexcept = default;

  [[nodiscard]] bool prepared() const noexcept { return owner_.prepared(); }
  /// Inspect the retained owner without exposing a mutable permanent installation seam.
  [[nodiscard]] const OwnedProgramInstallation& owner() const noexcept { return owner_; }
  /// Consume this rvalue handle and transfer its complete owner to the runtime publication path.
  /// Calling this is the explicit hand-off point; the prepared handle must not remain the owner of
  /// a candidate that has become reachable from ProgramRuntimeState.
  [[nodiscard]] OwnedProgramInstallation release_owner() && noexcept { return std::move(owner_); }

  /// Transfer the complete bind-sealed installation image in one operation.  All allocations and
  /// resource validation have completed before this payload can reach a runtime publication.
  /// Seal a runtime-sized plan using the exact host layouts observed after all prepare_* calls.
  /// The operation is transactional: all validation, digesting and allocation happen in local
  /// temporaries before the final host-owned plan/store are exchanged.
  void seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype> prototypes) {
    if (resource_plan_sealed_)
      throw std::logic_error("Program resource plan was already sealed");
    if (!owner_.resource_plan_requires_materialization())
      throw std::logic_error("static Program resource plan does not require runtime materialization");
    const auto& symbolic_tables = owner_.tables();
    if (symbolic_tables.resource_plan.empty())
      throw std::logic_error("runtime-sized Program resource plan has no symbolic rows");
    const std::string previous_digest = symbolic_tables.resource_plan.front().plan_digest;
    const std::string previous_resource_manifest =
        symbolic_tables.canonical_resource_manifest(std::nullopt, previous_digest);
    const std::optional<std::uint64_t> ceiling =
        owner_.candidate().maximum_bytes == kProgramResourcePlanUnknownExtent
            ? std::nullopt
            : std::optional<std::uint64_t>{owner_.candidate().maximum_bytes};
    ProgramResourcePlan materialized = owner_.tables().materialize_resource_plan(prototypes, ceiling);
    owner_.publish_materialized_resource_manifest(previous_resource_manifest, previous_digest,
                                                   materialized);
    ProgramPersistentValueStore persistent_values;
    persistent_values.bind(materialized);
    resource_plan_ = std::move(materialized);
    persistent_values_ = std::move(persistent_values);
    resource_plan_sealed_ = true;
  }

  void seal_resource_plan(
      const std::vector<ProgramInstallationTables::ResourcePrototype>& prototypes) {
    seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype>(prototypes));
  }

  [[nodiscard]] bool resource_plan_sealed() const noexcept { return resource_plan_sealed_; }

  [[nodiscard]] PublicationPayload release_publication_payload() && {
    if (!resource_plan_sealed_)
      throw std::logic_error(
          "runtime-sized Program resource plan must be materialized before publication");
    const std::uint64_t sealed_generation = generation();
    return PublicationPayload{std::move(owner_), std::move(resource_plan_),
                              std::move(persistent_values_), sealed_generation};
  }

  [[nodiscard]] const ProgramInstallationMetadata& metadata() const noexcept {
    return owner_.metadata();
  }
  [[nodiscard]] const ProgramInstallationTables& tables() const noexcept { return owner_.tables(); }
  [[nodiscard]] const std::vector<ProgramInstallationTables::ResourcePlan>& resource_plan() const noexcept {
    return owner_.tables().resource_plan;
  }
  [[nodiscard]] std::size_t resource_slot_count() const noexcept {
    return owner_.tables().resource_slot_count();
  }
  [[nodiscard]] const std::vector<ProgramInstallationTables::ResourcePlan>& resource_declarations()
      const noexcept {
    return owner_.tables().resource_declarations();
  }
  [[nodiscard]] const ProgramResourcePlan& sealed_resource_plan() const {
    if (!resource_plan_sealed_)
      throw std::logic_error("Program resource plan has not been materialized");
    return resource_plan_;
  }
  [[nodiscard]] const ProgramPersistentValueStore& persistent_values() const {
    if (!resource_plan_sealed_)
      throw std::logic_error("Program persistent values require a materialized resource plan");
    return persistent_values_;
  }
  [[nodiscard]] std::uint64_t resource_ceiling() const {
    if (!resource_plan_sealed_)
      throw std::logic_error("Program resource ceiling requires a materialized plan");
    return resource_plan_.maximum_bytes();
  }
  /// Alias matching the ABI field name; both accessors read the host-materialized exact ceiling.
  [[nodiscard]] std::uint64_t maximum_bytes() const { return resource_ceiling(); }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    const auto& image = owner_.preparation_image();
    return image ? image->generation() : 0;
  }
  [[nodiscard]] const std::vector<ProgramInstallationTables::Block>& blocks() const noexcept {
    return owner_.tables().blocks;
  }
  [[nodiscard]] const std::vector<ProgramInstallationTables::Parameter>& parameters() const noexcept {
    return owner_.tables().parameters;
  }
  [[nodiscard]] const std::vector<ProgramInstallationTables::Authority>& operator_authorities() const noexcept {
    return owner_.tables().operator_authorities;
  }

 private:
  OwnedProgramInstallation owner_;
  ProgramResourcePlan resource_plan_;
  ProgramPersistentValueStore persistent_values_;
  bool resource_plan_sealed_ = false;
};

}  // namespace pops::runtime::program
