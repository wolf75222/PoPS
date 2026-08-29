#pragma once

// Small, real ABI-v5 fixture helpers used by native integration tests.  The generated image is
// intentionally boring: it owns a ProgramExecutionServices context, publishes one ordinary step,
// and carries an exact block table plus an authenticated resource plan (empty unless requested).
// Tests use this helper when they need an installed artifact but are proving a different authority
// (for example, that a selective history replay is refused without a history-authority table).

#include <pops/runtime/program/owned_program_installation.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::test::program_v5 {

inline std::string cxx_string_literal(std::string_view value) {
  std::string result{"\""};
  for (const unsigned char character : value) {
    switch (character) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (character < 0x20) {
          std::ostringstream escaped;
          escaped << "\\x" << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<unsigned int>(character);
          result += escaped.str();
        } else {
          result.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  result.push_back('"');
  return result;
}

/// One install-time resource family used by the generic callback fixture.  Rows are deliberately
/// symbolic: the detached preparation callback observes exact layouts through prepare_* and the
/// host materializes the authenticated v5 plan before publication.  `slot` must be dense and
/// `program_block` is the Program block (not a runtime block number).
struct CallbackProgramResource final {
  enum class Kind : std::uint8_t { rhs, state, scalar, cache };

  Kind kind = Kind::state;
  std::size_t slot = 0;
  int subslot = 0;
  int program_block = 0;
  int level = -1;
  std::uint32_t components = 1;
  std::uint32_t ghosts = 0;
  std::uint64_t value_id = 0;
  std::uint64_t occurrence_path_id = 0;
  std::string identity;
  std::string occurrence_path;
  std::string owner;
  std::string space = "cell";
  std::string clock;
};

/// One generated field route prepared by the callback MODULE before the first temporal dispatch.
/// The route slot is an already-declared dense resource-plan slot; this deliberately reuses the
/// canonical Program preparation table rather than inventing a callback-only field authority.
struct CallbackProgramFieldRoute final {
  std::uint32_t slot = 0;
  std::string provider;
  std::vector<int> program_blocks;
};

struct CallbackProgramHistory final {
  std::string name;
  int depth = 0;  // maximum lag; the runtime ring length is depth + 1
  int components = 0;
  int program_block = -1;
  std::string state_identity;
  std::string space;
  std::string clock;
  std::string interpolation;
};

struct CallbackProgramClockRelation final {
  std::string parent;
  std::string child;
  int count = 0;
};

struct CallbackProgramTransactionAuthorities final {
  std::vector<std::string> diagnostics;
  std::vector<std::string> balance_routes;
  std::vector<std::string> step_projections;
};

inline std::string callback_resource_kind_name(CallbackProgramResource::Kind kind) {
  switch (kind) {
    case CallbackProgramResource::Kind::rhs:
      return "rhs";
    case CallbackProgramResource::Kind::state:
      return "state";
    case CallbackProgramResource::Kind::scalar:
      return "scalar";
    case CallbackProgramResource::Kind::cache:
      return "persistent_schedule";
  }
  throw std::invalid_argument("ABI-v5 callback fixture resource has an unknown kind");
}

inline std::string callback_component_names(std::uint32_t components) {
  if (components == 0)
    throw std::invalid_argument("ABI-v5 callback fixture resource has no components");
  std::string names{"["};
  for (std::uint32_t component = 0; component < components; ++component) {
    if (component != 0)
      names.push_back(',');
    names += "\"component" + std::to_string(component) + "\"";
  }
  names.push_back(']');
  return names;
}

inline pops::runtime::program::ProgramInstallationTables callback_resource_tables(
    const std::vector<CallbackProgramResource>& resources, std::string_view default_clock) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  tables.resource_plan.reserve(resources.size());
  for (std::size_t index = 0; index < resources.size(); ++index) {
    const CallbackProgramResource& resource = resources[index];
    if (resource.slot != index)
      throw std::invalid_argument("ABI-v5 callback fixture resource slots must be dense");
    if (resource.subslot < 0 || resource.program_block < 0 || resource.level < -1 ||
        resource.components == 0)
      throw std::invalid_argument("ABI-v5 callback fixture resource has an invalid shape");
    if (resource.kind == CallbackProgramResource::Kind::cache && resource.subslot != 0)
      throw std::invalid_argument("ABI-v5 callback fixture cache resource requires subslot zero");

    const std::string identity = resource.identity.empty()
                                     ? "pops.test.callback/resource/" + std::to_string(index)
                                     : resource.identity;
    const std::string occurrence_path =
        resource.occurrence_path.empty() ? identity : resource.occurrence_path;
    const std::string owner = resource.owner.empty() ? identity : resource.owner;
    const std::string clock = resource.clock.empty()
                                  ? (default_clock.empty() ? "macro" : std::string(default_clock))
                                  : resource.clock;
    const bool persistent = resource.kind == CallbackProgramResource::Kind::cache;
    ProgramInstallationTables::ResourcePlan row;
    row.slot = static_cast<std::uint32_t>(index);
    row.flags = kProgramResourceRuntimeSized |
                (persistent ? kProgramResourcePersistentSchedule : std::uint32_t{0});
    row.value_id = resource.value_id == 0 ? 0x1000u + index : resource.value_id;
    row.occurrence_path_id =
        resource.occurrence_path_id == 0 ? 0x2000u + index : resource.occurrence_path_id;
    row.level = resource.level;
    row.components = resource.components;
    row.ghosts = resource.ghosts;
    row.resource_type = ProgramResourcePlanType::runtime_sized;
    row.schema = "program-resource-plan:v1";
    row.identity = identity;
    row.occurrence_path = occurrence_path;
    row.owner = owner;
    row.space = resource.space;
    row.clock = clock;
    row.lifetime = persistent ? "persistent_schedule" : "transient";
    row.centering = "cell";
    row.off_policy = persistent ? "hold" : "none";
    row.communication = "none";
    row.transfer_provider = "none";
    row.restart_provider = "none";
    row.component_names = callback_component_names(resource.components);
    row.shape = "[]";
    tables.resource_plan.push_back(std::move(row));
  }
  return tables;
}

inline std::pair<std::string, std::string> callback_resource_manifest(
    const pops::runtime::program::ProgramInstallationTables& tables) {
  const std::optional<std::uint64_t> ceiling =
      tables.resource_plan.empty() ? std::optional<std::uint64_t>{0} : std::nullopt;
  const std::string payload = tables.canonical_resource_digest_payload(ceiling);
  const std::string digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  const std::string manifest =
      "{\"resource_plan\":" + tables.canonical_resource_manifest(ceiling, digest) +
      ",\"resource_plan_digest\":\"" + digest + "\"}";
  return {digest, manifest};
}

inline std::string callback_empty_resource_manifest_literal() {
  pops::runtime::program::ProgramInstallationTables tables;
  const std::string payload = tables.canonical_resource_digest_payload(std::uint64_t{0});
  const std::string digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  return "R\"JSON({\"resource_plan\":" + tables.canonical_resource_manifest(0, digest) +
         ",\"resource_plan_digest\":\"" + digest + "\"})JSON\"";
}

/// Return a complete ABI-v5 MODULE source whose ordinary step is dispatched to a host-exported
/// callback.  The callback receives the opaque `ProgramExecutionServices<POPS_NATIVE_DIM>`
/// context created from the detached preparation image; it never receives a System/AmrSystem
/// facade or a loader handle.  `runtime_kind` is the literal `uniform` or `amr`.  AMR artifacts
/// include the complete lifecycle hook set required by the public AmrSystem installer, while the
/// hooks intentionally perform no work and the typed accepted snapshot is empty.
///
/// The caller owns the callback registry and must export `callback_symbol` from the test
/// executable with the signature `void(std::uint64_t, void*, double)`.  This keeps the fixture
/// useful for unrelated Uniform and AMR tests without adding a second installation authority.
inline std::string callback_program_source(
    std::uint64_t callback_identifier, std::string_view identity, std::string_view clock,
    const std::vector<std::string>& blocks, const std::vector<CallbackProgramResource>& resources,
    std::string_view callback_symbol = "pops_test_program_callback",
    std::string_view runtime_kind = "uniform",
    const std::vector<CallbackProgramFieldRoute>& field_routes = {},
    const CallbackProgramTransactionAuthorities& transaction_authorities = {},
    const std::vector<CallbackProgramHistory>& histories = {},
    const std::vector<CallbackProgramClockRelation>& clock_relations = {},
    const std::optional<std::vector<pops::runtime::program::ProgramFluxBudgetRecord>>&
        flux_budgets = std::nullopt) {
  if (identity.empty())
    throw std::invalid_argument("ABI-v5 callback fixture identity must not be empty");
  if (runtime_kind != "uniform" && runtime_kind != "amr")
    throw std::invalid_argument("ABI-v5 callback fixture runtime kind must be uniform or amr");
  if (callback_symbol.empty() ||
      !(std::isalpha(static_cast<unsigned char>(callback_symbol.front())) ||
        callback_symbol.front() == '_'))
    throw std::invalid_argument("ABI-v5 callback fixture callback symbol is not an identifier");
  for (const char character : callback_symbol) {
    if (!(std::isalnum(static_cast<unsigned char>(character)) || character == '_'))
      throw std::invalid_argument("ABI-v5 callback fixture callback symbol is not an identifier");
  }
  for (const CallbackProgramFieldRoute& route : field_routes) {
    if (route.slot >= resources.size() || route.provider.empty() || route.program_blocks.empty())
      throw std::invalid_argument(
          "ABI-v5 callback fixture field route is outside the resource plan");
    for (const int block : route.program_blocks) {
      if (block < 0 || static_cast<std::size_t>(block) >= blocks.size())
        throw std::invalid_argument("ABI-v5 callback fixture field route has an invalid block");
    }
  }
  std::vector<std::string> history_names;
  history_names.reserve(histories.size());
  for (const CallbackProgramHistory& history : histories) {
    if (history.name.empty() || history.depth < 1 || history.components < 1 ||
        history.program_block < 0 || history.state_identity.empty() || history.space.empty() ||
        history.clock.empty() || history.interpolation.empty())
      throw std::invalid_argument("ABI-v5 callback fixture history declaration is invalid");
    if (static_cast<std::size_t>(history.program_block) >= blocks.size())
      throw std::invalid_argument(
          "ABI-v5 callback fixture history owner is outside its block table");
    if (std::find(history_names.begin(), history_names.end(), history.name) != history_names.end())
      throw std::invalid_argument(
          "ABI-v5 callback fixture history declaration has duplicate names");
    history_names.push_back(history.name);
  }
  for (const CallbackProgramClockRelation& relation : clock_relations) {
    if (relation.parent.empty() || relation.child.empty() || relation.parent == relation.child ||
        relation.count < 1)
      throw std::invalid_argument("ABI-v5 callback fixture clock relation is invalid");
  }

  const auto resource_tables = callback_resource_tables(resources, clock);
  const auto [resource_digest, resource_manifest] = callback_resource_manifest(resource_tables);

  std::vector<pops::runtime::program::ProgramFluxBudgetRecord> default_flux_budgets;
  const std::vector<pops::runtime::program::ProgramFluxBudgetRecord>* flux_budget_rows = nullptr;
  if (flux_budgets.has_value()) {
    if (!flux_budgets->empty() && flux_budgets->size() != blocks.size())
      throw std::invalid_argument(
          "ABI-v5 callback fixture flux budget table must contain exactly one row per Program "
          "block");
    flux_budget_rows = &*flux_budgets;
  } else if (runtime_kind == "amr" && !blocks.empty()) {
    default_flux_budgets.reserve(blocks.size());
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      if (blocks.size() == 1)
        default_flux_budgets.push_back({1, 1, 0, 0});
      else
        default_flux_budgets.push_back({2, 1, 1, 4096});
    }
    flux_budget_rows = &default_flux_budgets;
  }

  // clang-format off
  std::ostringstream source;
  (void)resource_digest;
  source << R"CPP(
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" void )CPP"
         << callback_symbol << R"CPP((std::uint64_t, void*, double);

namespace {
struct ProgramCandidateState final {
  std::shared_ptr<pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>>
      context;
};

void candidate_error(pops::runtime::program::ProgramInstallDiagnostic* diagnostic,
                     const char* message) noexcept {
  if (diagnostic == nullptr)
    return;
  diagnostic->code = pops::runtime::program::ProgramInstallErrorCode::artifact_rejected;
  std::size_t index = 0;
  while (index + 1 < sizeof(diagnostic->message) && message[index] != '\0') {
    diagnostic->message[index] = message[index];
    ++index;
  }
  diagnostic->message[index] = '\0';
}

template <class Function>
void candidate_prepare_stage(const char* stage, Function&& function) {
  try {
    std::forward<Function>(function)();
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string(stage) + ": " + error.what());
  }
}

void candidate_step(void* opaque, double dt) {
  auto& state = *static_cast<ProgramCandidateState*>(opaque);
  )CPP";
  source << "  " << callback_symbol << "(" << callback_identifier
         << ", state.context.get(), dt);\n}\n\n";
  source << R"CPP(
void candidate_destroy(void* opaque) noexcept { delete static_cast<ProgramCandidateState*>(opaque); }

bool candidate_prepare(void* opaque,
                       const pops::runtime::program::ProgramHostDescriptor* host,
                       pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  if (opaque == nullptr || host == nullptr || host->preparation.image == nullptr) {
    candidate_error(diagnostic, "ABI-v5 callback fixture received an invalid host image");
    return false;
  }
  auto& state = *static_cast<ProgramCandidateState*>(opaque);
  if (state.context) {
    candidate_error(diagnostic, "ABI-v5 callback fixture preparation was entered twice");
    return false;
  }
  try {
    state.context = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(
        host->preparation);
)CPP";
  for (const auto& diagnostic : transaction_authorities.diagnostics)
    source << "    state.context->declare_diagnostic(" << cxx_string_literal(diagnostic)
           << ");\n";
  for (const auto& route : transaction_authorities.balance_routes)
    source << "    state.context->declare_balance_route(" << cxx_string_literal(route)
           << ");\n";
  for (const auto& projection : transaction_authorities.step_projections)
    source << "    state.context->declare_step_projection(" << cxx_string_literal(projection)
           << ");\n";
  if (!clock.empty()) {
    source << "    candidate_prepare_stage(\"primary clock\", [&] { "
              "state.context->configure_primary_clock("
           << cxx_string_literal(clock) << "); });\n";
  }
  for (const CallbackProgramClockRelation& relation : clock_relations) {
    source << "    candidate_prepare_stage(\"clock relation\", [&] { "
              "state.context->declare_clock_relation("
           << cxx_string_literal(relation.parent) << ", " << cxx_string_literal(relation.child)
           << ", " << relation.count << "); });\n";
  }
  for (const CallbackProgramHistory& history : histories) {
    source << "    candidate_prepare_stage(\"history declaration\", [&] { "
              "state.context->register_history("
           << cxx_string_literal(history.name) << ", " << history.depth << ", "
           << history.components << ", " << history.program_block << ", "
           << cxx_string_literal(history.state_identity) << ", "
           << cxx_string_literal(history.space) << ", " << cxx_string_literal(history.clock)
           << ", " << cxx_string_literal(history.interpolation) << "); });\n";
  }
  for (const auto& resource : resources) {
    switch (resource.kind) {
      case CallbackProgramResource::Kind::rhs:
        source << "    state.context->prepare_rhs_scratch(" << resource.slot << ", "
               << resource.subslot << ", " << resource.program_block << ");\n";
        break;
      case CallbackProgramResource::Kind::state:
        source << "    state.context->prepare_state_scratch(" << resource.slot << ", "
               << resource.subslot << ", " << resource.program_block << ");\n";
        break;
      case CallbackProgramResource::Kind::scalar:
        source << "    state.context->prepare_scalar_scratch(" << resource.slot << ", "
               << resource.subslot << ", " << resource.program_block << ", "
               << resource.components << ", " << resource.ghosts << ");\n";
        break;
      case CallbackProgramResource::Kind::cache:
        source << "    state.context->prepare_cache_slot(" << resource.slot << ", "
               << resource.program_block << ");\n";
        break;
    }
  }
  for (const CallbackProgramFieldRoute& route : field_routes) {
    source << "    state.context->prepare_generated_field_route(" << route.slot << ", "
           << cxx_string_literal(route.provider) << ", {";
    for (std::size_t index = 0; index < route.program_blocks.size(); ++index) {
      if (index != 0)
        source << ", ";
      source << route.program_blocks[index];
    }
    source << "});\n";
  }
  source << R"CPP(
    return true;
  } catch (const std::exception& error) {
    candidate_error(diagnostic, error.what());
    return false;
  } catch (...) {
    candidate_error(diagnostic, "ABI-v5 callback fixture preparation failed");
    return false;
  }
}
)CPP";
  if (runtime_kind == "amr") {
    source << R"CPP(
void candidate_hierarchy_refresh(void*) {}
void candidate_history_remap(void*, const void*) {}
void candidate_restart_preflight(void*) {}
void candidate_restart_regrid(void*) {}
void candidate_restart_resync(void*) {}
pops::runtime::program::AcceptedProgramExecutionServicesSnapshot*
candidate_create_snapshot(void* opaque) {
  auto& state = *static_cast<ProgramCandidateState*>(opaque);
  if (!state.context)
    throw std::logic_error("ABI-v5 callback fixture creates its AMR snapshot before prepare");
  auto snapshot = state.context->create_accepted_context_snapshot();
  if (!snapshot)
    throw std::logic_error("ABI-v5 callback fixture received an empty AMR accepted snapshot");
  return snapshot.release();
}
)CPP";
  }
  source << R"CPP(
}  // namespace
)CPP";

  if (blocks.empty()) {
    source << "namespace { constexpr pops::runtime::program::ProgramBlockRecord* "
               "kProgramBlocks = nullptr; }\n";
  } else {
    source << "namespace { constexpr pops::runtime::program::ProgramBlockRecord kProgramBlocks[] = {";
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      if (index != 0)
        source << ", ";
      source << "{{" << cxx_string_literal(blocks[index]) << ", " << blocks[index].size()
             << "}}";
    }
    source << "}; }\n";
  }

  if (!histories.empty()) {
    source << "namespace {\n";
    for (std::size_t index = 0; index < histories.size(); ++index) {
      const auto& history = histories[index];
      source << "static constexpr char kProgramHistoryName" << index << "[] = "
             << cxx_string_literal(history.name) << ";\n";
      source << "static constexpr char kProgramHistoryState" << index << "[] = "
             << cxx_string_literal(history.state_identity) << ";\n";
      source << "static constexpr char kProgramHistorySpace" << index << "[] = "
             << cxx_string_literal(history.space) << ";\n";
      source << "static constexpr char kProgramHistoryClock" << index << "[] = "
             << cxx_string_literal(history.clock) << ";\n";
      source << "static constexpr char kProgramHistoryInterpolation" << index << "[] = "
             << cxx_string_literal(history.interpolation) << ";\n";
    }
    source << "static constexpr pops::runtime::program::ProgramHistoryAuthorityRecord "
               "kProgramHistoryAuthorities[] = {";
    for (std::size_t index = 0; index < histories.size(); ++index) {
      if (index != 0)
        source << ", ";
      const auto retained_images = static_cast<std::uint64_t>(histories[index].depth) + 1;
      source << "{{kProgramHistoryName" << index << ", sizeof(kProgramHistoryName" << index
             << ") - 1}, " << retained_images << "ULL, 0}";
    }
    source << "};\n";
    source << "static constexpr pops::runtime::program::ProgramCheckpointRecord "
               "kProgramCheckpointShape[] = {";
    for (std::size_t index = 0; index < histories.size(); ++index) {
      if (index != 0)
        source << ", ";
      const auto& history = histories[index];
      const auto retained_images = static_cast<std::uint64_t>(history.depth) + 1;
      source << "{{kProgramHistoryName" << index << ", sizeof(kProgramHistoryName" << index
             << ") - 1}, {kProgramHistoryState" << index << ", sizeof(kProgramHistoryState"
             << index << ") - 1}, {kProgramHistorySpace" << index
             << ", sizeof(kProgramHistorySpace" << index << ") - 1}, {kProgramHistoryClock"
             << index << ", sizeof(kProgramHistoryClock" << index
             << ") - 1}, {kProgramHistoryInterpolation" << index
             << ", sizeof(kProgramHistoryInterpolation" << index << ") - 1}, "
             << history.program_block << ", " << history.components << ", " << retained_images
             << "ULL}";
    }
    source << "};\n}\n";
  }

  // When no explicit table is supplied, AMR installation authenticates one default
  // flux-expression budget row per Program block.  An engaged empty vector deliberately omits
  // the table, while an engaged non-empty vector is emitted exactly as supplied.
  if (flux_budget_rows != nullptr && !flux_budget_rows->empty()) {
    source << "namespace { constexpr pops::runtime::program::ProgramFluxBudgetRecord "
               "kProgramFluxBudgets[] = {";
    for (std::size_t index = 0; index < flux_budget_rows->size(); ++index) {
      if (index != 0)
        source << ", ";
      const auto& row = flux_budget_rows->at(index);
      source << "{" << row.rhs_basis_bound << "ULL, " << row.coefficient_term_bound << "ULL, "
             << row.interface_application_bound << "ULL, "
             << row.interface_identity_character_bound << "ULL}";
    }
    source << "}; }\n";
  }

  if (!resources.empty()) {
    source << "namespace {\n";
    source << "static constexpr char kResourceSchema[] = \"program-resource-plan:v1\";\n";
    source << "static constexpr char kResourceDigest[] = "
           << cxx_string_literal(resource_digest) << ";\n";
    for (std::size_t index = 0; index < resources.size(); ++index) {
      const auto& row = resource_tables.resource_plan[index];
      source << "static constexpr char kResourceIdentity" << index << "[] = "
             << cxx_string_literal(row.identity) << ";\n";
      source << "static constexpr char kResourceOccurrence" << index << "[] = "
             << cxx_string_literal(row.occurrence_path) << ";\n";
      source << "static constexpr char kResourceOwner" << index << "[] = "
             << cxx_string_literal(row.owner) << ";\n";
      source << "static constexpr char kResourceSpace" << index << "[] = "
             << cxx_string_literal(row.space) << ";\n";
      source << "static constexpr char kResourceClock" << index << "[] = "
             << cxx_string_literal(row.clock) << ";\n";
      source << "static constexpr char kResourceLifetime" << index << "[] = "
             << cxx_string_literal(row.lifetime) << ";\n";
      source << "static constexpr char kResourceOffPolicy" << index << "[] = "
             << cxx_string_literal(row.off_policy) << ";\n";
      source << "static constexpr char kResourceCommunication" << index << "[] = \"none\";\n";
      source << "static constexpr char kResourceComponents" << index << "[] = "
             << cxx_string_literal(row.component_names) << ";\n";
      source << "static constexpr char kResourceShape" << index << "[] = \"[]\";\n";
      source << "constexpr pops::runtime::program::ProgramResourcePlanRecord makeResource"
             << index << "() {\n"
             << "  using namespace pops::runtime::program;\n"
             << "  ProgramResourcePlanRecord row{};\n"
             << "  row.slot = " << row.slot << ";\n"
             << "  row.flags = kProgramResourceRuntimeSized"
             << (row.flags & pops::runtime::program::kProgramResourcePersistentSchedule
                     ? " | kProgramResourcePersistentSchedule"
                     : "")
             << ";\n"
             << "  row.value_id = " << row.value_id << "ULL;\n"
             << "  row.occurrence_path_id = " << row.occurrence_path_id << "ULL;\n"
             << "  row.level = " << row.level << ";\n"
             << "  row.components = " << row.components << ";\n"
             << "  row.ghosts = " << row.ghosts << ";\n"
             << "  row.bytes = kProgramResourcePlanUnknownExtent;\n"
             << "  row.maximum_bytes = kProgramResourcePlanUnknownExtent;\n"
             << "  row.cells = kProgramResourcePlanUnknownExtent;\n"
             << "  row.itemsize = kProgramResourcePlanUnknownExtent;\n"
             << "  row.schema = {kResourceSchema, sizeof(kResourceSchema) - 1};\n"
             << "  row.plan_digest = {kResourceDigest, sizeof(kResourceDigest) - 1};\n"
             << "  row.identity = {kResourceIdentity" << index
             << ", sizeof(kResourceIdentity" << index << ") - 1};\n"
             << "  row.occurrence_path = {kResourceOccurrence" << index
             << ", sizeof(kResourceOccurrence" << index << ") - 1};\n"
             << "  row.owner = {kResourceOwner" << index
             << ", sizeof(kResourceOwner" << index << ") - 1};\n"
             << "  row.space = {kResourceSpace" << index
             << ", sizeof(kResourceSpace" << index << ") - 1};\n"
             << "  row.clock = {kResourceClock" << index
             << ", sizeof(kResourceClock" << index << ") - 1};\n"
             << "  row.lifetime = {kResourceLifetime" << index
             << ", sizeof(kResourceLifetime" << index << ") - 1};\n"
             << "  row.centering = {\"cell\", 4};\n"
             << "  row.off_policy = {kResourceOffPolicy" << index
             << ", sizeof(kResourceOffPolicy" << index << ") - 1};\n"
             << "  row.communication = {kResourceCommunication" << index
             << ", sizeof(kResourceCommunication" << index << ") - 1};\n"
             << "  row.transfer_provider = {\"none\", 4};\n"
             << "  row.restart_provider = {\"none\", 4};\n"
             << "  row.component_names = {kResourceComponents" << index
             << ", sizeof(kResourceComponents" << index << ") - 1};\n"
             << "  row.shape = {kResourceShape" << index
             << ", sizeof(kResourceShape" << index << ") - 1};\n"
             << "  row.resource_type = ProgramResourcePlanType::runtime_sized;\n"
             << "  return row;\n}\n";
    }
    source << "constexpr pops::runtime::program::ProgramResourcePlanRecord kProgramResources[] = {";
    for (std::size_t index = 0; index < resources.size(); ++index) {
      if (index != 0)
        source << ", ";
      source << "makeResource" << index << "()";
    }
    source << "};\n}\n";
  }

  source << R"CPP(

extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor* host,
    pops::runtime::program::ProgramCandidateDescriptor* candidate,
    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  using namespace pops::runtime::program;
  if (host == nullptr || candidate == nullptr || !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::)CPP"
         << runtime_kind << R"CPP( || host->services.state_store == nullptr) {
    candidate_error(diagnostic, "ABI-v5 callback fixture received an invalid host descriptor");
    return false;
  }
  *candidate = {};
  try {
    auto state = std::make_unique<ProgramCandidateState>();
    ProgramCandidateDescriptor descriptor{};
    descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));
    descriptor.abi_version = kProgramInstallAbiVersion;
    descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
    descriptor.runtime_kind = ProgramRuntimeKind::)CPP"
         << runtime_kind << R"CPP(;
    descriptor.provided_capability_bits = host->capability_bits;
    descriptor.required_capability_bits = 0;
    descriptor.required_service_bits = kKnownProgramServiceBits;
    static constexpr char kIdentity[] = )CPP"
         << cxx_string_literal(identity) << R"CPP(;
    static constexpr char kAbiKey[] = POPS_ABI_KEY_LITERAL;
    static constexpr char kResourceManifest[] = )CPP"
         << cxx_string_literal(resource_manifest)
         << R"CPP(;
    descriptor.program_name = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.artifact_identity = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.abi_key = {kAbiKey, sizeof(kAbiKey) - 1};
    descriptor.route_manifest = {pops::kRouteRegistrySignature,
                                 static_cast<std::uint64_t>(
                                     std::char_traits<char>::length(pops::kRouteRegistrySignature))};
    descriptor.boundary_manifest = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.persistent_resource_manifest = {kResourceManifest, sizeof(kResourceManifest) - 1};
    descriptor.checkpoint_identity = {kIdentity, sizeof(kIdentity) - 1};
)CPP";
  if (blocks.empty()) {
    source << "    descriptor.blocks = {nullptr, 0, 0};\n";
  } else {
    source << "    descriptor.blocks = {kProgramBlocks, sizeof(kProgramBlocks) / "
               "sizeof(kProgramBlocks[0]), sizeof(ProgramBlockRecord)};\n";
  }
  if (!resources.empty())
    source << "    descriptor.resource_plan = {kProgramResources, " << resources.size()
           << ", sizeof(ProgramResourcePlanRecord)};\n";
  if (!histories.empty()) {
    source << "    descriptor.history_authorities = {kProgramHistoryAuthorities, "
               "sizeof(kProgramHistoryAuthorities) / sizeof(kProgramHistoryAuthorities[0]), "
               "sizeof(ProgramHistoryAuthorityRecord)};\n";
    source << "    descriptor.checkpoint_shape = {kProgramCheckpointShape, "
               "sizeof(kProgramCheckpointShape) / sizeof(kProgramCheckpointShape[0]), "
               "sizeof(ProgramCheckpointRecord)};\n";
  }
  if (flux_budget_rows != nullptr && !flux_budget_rows->empty())
    source << "    descriptor.flux_budgets = {kProgramFluxBudgets, sizeof(kProgramFluxBudgets) / "
               "sizeof(kProgramFluxBudgets[0]), sizeof(ProgramFluxBudgetRecord)};\n";
  source << R"CPP(
    descriptor.maximum_bytes = )CPP"
         << (resources.empty() ? "0" : "kProgramResourcePlanUnknownExtent") << R"CPP(;
    descriptor.context = state.release();
    descriptor.prepare = &candidate_prepare;
    descriptor.step = &candidate_step;
    descriptor.destroy = &candidate_destroy;
)CPP";
  if (runtime_kind == "amr") {
    source << R"CPP(
    descriptor.hierarchy_refresh = &candidate_hierarchy_refresh;
    descriptor.history_remap_accepted = &candidate_history_remap;
    descriptor.restart_regrid_preflight = &candidate_restart_preflight;
    descriptor.restart_regrid = &candidate_restart_regrid;
    descriptor.restart_resync = &candidate_restart_resync;
    descriptor.create_accepted_snapshot = &candidate_create_snapshot;
)CPP";
  }
  source << R"CPP(
    *candidate = descriptor;
    return true;
  } catch (const std::exception& error) {
    candidate_error(diagnostic, error.what());
    return false;
  } catch (...) {
    candidate_error(diagnostic, "ABI-v5 callback fixture candidate construction failed");
    return false;
  }
}
)CPP";
  // clang-format on
  return source.str();
}

inline std::string callback_program_source(
    std::uint64_t callback_identifier, std::string_view identity, std::string_view clock,
    const std::vector<std::string>& blocks,
    std::string_view callback_symbol = "pops_test_program_callback",
    std::string_view runtime_kind = "uniform") {
  return callback_program_source(callback_identifier, identity, clock, blocks,
                                 std::vector<CallbackProgramResource>{}, callback_symbol,
                                 runtime_kind);
}

/// Return a complete source file for a uniform ABI-v5 artifact with an ordinary deterministic
/// step.  `blocks` is the exact ordered Program block table; `owner` is a Program block index, not
/// a runtime value-id.  The descriptor deliberately exports no history-authority rows.
inline std::string ramp_program_source(std::string_view history, int depth, double rate, int owner,
                                       const std::vector<std::string>& blocks) {
  if (history.empty())
    throw std::invalid_argument("ABI-v5 fixture history identity must not be empty");
  if (depth < 2)
    throw std::invalid_argument("ABI-v5 fixture history depth must be at least two");
  if (blocks.empty())
    throw std::invalid_argument("ABI-v5 fixture requires at least one Program block");
  if (owner < 0 || owner >= static_cast<int>(blocks.size()))
    throw std::invalid_argument("ABI-v5 fixture history owner is outside its block table");

  // clang-format off
  std::ostringstream source;
  source << R"CPP(
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/core/foundation/types.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace {
struct ProgramCandidateState final {
  std::shared_ptr<pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>>
      context;
  std::function<void(double)> step;
};

void candidate_destroy(void* opaque) noexcept { delete static_cast<ProgramCandidateState*>(opaque); }

void candidate_error(pops::runtime::program::ProgramInstallDiagnostic* diagnostic,
                    const char* message) noexcept {
  if (diagnostic == nullptr)
    return;
  diagnostic->code = pops::runtime::program::ProgramInstallErrorCode::artifact_rejected;
  std::size_t index = 0;
  while (index + 1 < sizeof(diagnostic->message) && message[index] != '\0') {
    diagnostic->message[index] = message[index];
    ++index;
  }
  diagnostic->message[index] = '\0';
}

void candidate_step(void* opaque, double dt) {
  static_cast<ProgramCandidateState*>(opaque)->step(dt);
}

bool candidate_prepare(void* opaque,
                       const pops::runtime::program::ProgramHostDescriptor* host,
                       pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  if (opaque == nullptr || host == nullptr || !host->preparation.image) {
    candidate_error(diagnostic, "ABI-v5 fixture received an invalid preparation image");
    return false;
  }
  auto* state = static_cast<ProgramCandidateState*>(opaque);
  if (state->context || state->step) {
    candidate_error(diagnostic, "ABI-v5 fixture preparation was entered twice");
    return false;
  }
  try {
    state->context = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(
        host->preparation);
    auto context = state->context;
    context->declare_diagnostic("test.program.v5.ramp.executed");
    context->register_history(%HISTORY%, %DEPTH% - 1, -1, %OWNER%, %STATE_ID%, "test.space",
                              "test.clock", "test.exact");
    context->configure_primary_clock("test.clock");
    state->step = [context](double dt) {
      context->begin_step(dt);
      context->set_stage_time(0, 1);
      context->record_scalar("test.program.v5.ramp.executed", pops::Real(1));
      auto& value = context->state(%OWNER%);
      context->store_history(%HISTORY%, value);
      pops::MultiFab<pops::kNativeDimension> bump = value;
      bump.set_val(static_cast<pops::Real>(%RATE%) * static_cast<pops::Real>(dt));
      pops::saxpy(value, pops::Real(1), bump);
      context->rotate_histories();
    };
    return true;
  } catch (const std::exception& error) {
    candidate_error(diagnostic, error.what());
    return false;
  } catch (...) {
    candidate_error(diagnostic, "ABI-v5 fixture preparation failed");
    return false;
  }
}
}  // namespace
)CPP";

  std::string block_table =
      "namespace { constexpr pops::runtime::program::ProgramBlockRecord kProgramBlocks[] = {";
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    if (index != 0)
      block_table += ", ";
    block_table += "{{" + cxx_string_literal(blocks[index]) + ", " +
                   std::to_string(blocks[index].size()) + "}}";
  }
  block_table += "}; }\n";
  source << block_table;

  std::ostringstream rate_text;
  rate_text << std::setprecision(17) << rate;
  const std::string history_literal = cxx_string_literal(history);
  const std::string state_identity =
      cxx_string_literal("test.state." + std::to_string(owner));
  const std::string fixture_identity = cxx_string_literal("test-program-v5-ramp");
  const std::string empty_manifest = callback_empty_resource_manifest_literal();

  std::string body = R"CPP(
extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor* host,
    pops::runtime::program::ProgramCandidateDescriptor* candidate,
    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  using namespace pops::runtime::program;
  if (host == nullptr || candidate == nullptr || !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::uniform || host->services.state_store == nullptr) {
    candidate_error(diagnostic, "ABI-v5 fixture received an invalid host descriptor");
    return false;
  }
  *candidate = {};
  try {
    auto state = std::make_unique<ProgramCandidateState>();
    ProgramCandidateDescriptor descriptor{};
    descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));
    descriptor.abi_version = kProgramInstallAbiVersion;
    descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
    descriptor.runtime_kind = ProgramRuntimeKind::uniform;
    descriptor.provided_capability_bits = host->capability_bits;
    descriptor.required_capability_bits = 0;
    descriptor.required_service_bits = kProgramServiceState | kProgramServiceHistory |
                                       kProgramServiceClock | kProgramServiceTransaction;
    static constexpr char kIdentity[] = %IDENTITY%;
    static constexpr char kAbiKey[] = POPS_ABI_KEY_LITERAL;
    static constexpr char kResourceManifest[] = %MANIFEST%;
    descriptor.program_name = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.artifact_identity = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.abi_key = {kAbiKey, sizeof(kAbiKey) - 1};
    descriptor.route_manifest = {pops::kRouteRegistrySignature,
                                 static_cast<std::uint64_t>(
                                     std::char_traits<char>::length(pops::kRouteRegistrySignature))};
    descriptor.boundary_manifest = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.persistent_resource_manifest = {kResourceManifest,
                                                sizeof(kResourceManifest) - 1};
    descriptor.checkpoint_identity = {%HISTORY%, sizeof(%HISTORY%) - 1};
    descriptor.blocks = {kProgramBlocks, %BLOCK_COUNT%, sizeof(ProgramBlockRecord)};
    descriptor.maximum_bytes = 0;
    descriptor.context = state.get();
    descriptor.prepare = &candidate_prepare;
    descriptor.step = &candidate_step;
    descriptor.destroy = &candidate_destroy;
    *candidate = descriptor;
    (void)state.release();
    return true;
  } catch (const std::exception& error) {
    candidate_error(diagnostic, error.what());
    return false;
  } catch (...) {
    candidate_error(diagnostic, "ABI-v5 fixture candidate construction failed");
    return false;
  }
}
)CPP";

  auto replace_all = [&body](std::string_view needle, std::string_view replacement) {
    std::size_t offset = 0;
    while ((offset = body.find(needle, offset)) != std::string::npos) {
      body.replace(offset, needle.size(), replacement);
      offset += replacement.size();
    }
  };
  replace_all("%OWNER%", std::to_string(owner));
  replace_all("%HISTORY%", history_literal);
  replace_all("%DEPTH%", std::to_string(depth));
  replace_all("%STATE_ID%", state_identity);
  replace_all("%RATE%", rate_text.str());
  replace_all("%IDENTITY%", fixture_identity);
  replace_all("%MANIFEST%", empty_manifest);
  replace_all("%BLOCK_COUNT%", std::to_string(blocks.size()));
  source << body;
  // clang-format on
  return source.str();
}

/// Return a complete ABI-v5 source for authority-only tests.  The candidate still goes through the
/// real v5 install/prepare/publish path, but its step does not depend on a numerical model.  The
/// modes are deliberately finite test fixtures rather than a second runtime API:
/// `reject_first_attempt` and `reject_first_dt` inject one DSO-owned failure, `wait_for_release`
/// provides a file-backed coordination point, and the two `probe_*` modes call test-only exported
/// probes from the host executable while the candidate visibility writer is held.
inline std::string authority_program_source(std::string_view mode, std::string_view identity,
                                            const std::vector<std::string>& blocks,
                                            std::string_view marker_path = {},
                                            std::string_view release_path = {}) {
  if (identity.empty())
    throw std::invalid_argument("ABI-v5 authority fixture identity must not be empty");
  if (blocks.empty())
    throw std::invalid_argument("ABI-v5 authority fixture requires an explicit block table");
  constexpr std::string_view kRejectFirstAttempt = "reject_first_attempt";
  constexpr std::string_view kRejectFirstDt = "reject_first_dt";
  constexpr std::string_view kWaitForRelease = "wait_for_release";
  constexpr std::string_view kProbePublic = "probe_public_readers";
  constexpr std::string_view kProbeProvisional = "probe_provisional";
  constexpr std::string_view kNoop = "noop";
  if (mode != kRejectFirstAttempt && mode != kRejectFirstDt && mode != kWaitForRelease &&
      mode != kProbePublic && mode != kProbeProvisional && mode != kNoop)
    throw std::invalid_argument("ABI-v5 authority fixture has an unknown step mode");
  if (mode == kWaitForRelease && (marker_path.empty() || release_path.empty()))
    throw std::invalid_argument("ABI-v5 wait fixture requires marker and release paths");

  // clang-format off
  std::ostringstream source;
  source << R"CPP(
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/core/foundation/types.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

extern "C" void pops_test_v5_public_reader_probe() noexcept;
extern "C" void pops_test_v5_provisional_probe() noexcept;

namespace {
struct ProgramCandidateState final {
  std::shared_ptr<pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>>
      context;
  std::function<void(double)> step;
  std::uint64_t calls = 0;
};

void candidate_destroy(void* opaque) noexcept { delete static_cast<ProgramCandidateState*>(opaque); }

void candidate_error(pops::runtime::program::ProgramInstallDiagnostic* diagnostic,
                    const char* message) noexcept {
  if (diagnostic == nullptr)
    return;
  diagnostic->code = pops::runtime::program::ProgramInstallErrorCode::artifact_rejected;
  std::size_t index = 0;
  while (index + 1 < sizeof(diagnostic->message) && message[index] != '\0') {
    diagnostic->message[index] = message[index];
    ++index;
  }
  diagnostic->message[index] = '\0';
}

void candidate_step(void* opaque, double dt) {
  static_cast<ProgramCandidateState*>(opaque)->step(dt);
}

bool candidate_prepare(void* opaque,
                       const pops::runtime::program::ProgramHostDescriptor* host,
                       pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  if (opaque == nullptr || host == nullptr || !host->preparation.image) {
    candidate_error(diagnostic, "ABI-v5 authority fixture received an invalid preparation image");
    return false;
  }
  auto* state = static_cast<ProgramCandidateState*>(opaque);
  if (state->context || state->step) {
    candidate_error(diagnostic, "ABI-v5 authority fixture preparation was entered twice");
    return false;
  }
  try {
    state->context = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(
        host->preparation);
    auto context = state->context;
    context->declare_diagnostic("test.program.v5.authority.calls");
    context->declare_diagnostic("test.program.v5.authority.last_dt");
    if (%PROBE_PROVISIONAL%)
      context->declare_diagnostic("candidate");
    context->configure_primary_clock("test.authority.clock");
    state->step = [state, context](double dt) {
      context->begin_step(dt);
      context->set_stage_time(0, 1);
      ++state->calls;
      if (%REJECT_FIRST_ATTEMPT% && state->calls == 1)
        throw std::runtime_error("injected v5 authority attempt failure");
      if (%REJECT_FIRST_DT% && state->calls == 1)
        throw pops::runtime::program::StepAttemptRejected(
            pops::SolveStatus::kIterationLimit, "authority", "injected v5 authority dt failure");
      context->record_scalar("test.program.v5.authority.calls",
                             static_cast<pops::Real>(state->calls));
      context->record_scalar("test.program.v5.authority.last_dt", static_cast<pops::Real>(dt));
      if (%PROBE_PUBLIC%)
        pops_test_v5_public_reader_probe();
      if (%PROBE_PROVISIONAL%) {
        context->record_scalar("candidate", pops::Real(7));
        pops_test_v5_provisional_probe();
      }
      if (%WAIT_FOR_RELEASE%) {
        std::ofstream marker(%MARKER%);
        marker << "started\n";
        marker.close();
        for (;;) {
          std::ifstream release(%RELEASE%);
          if (release.good())
            break;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    };
    return true;
  } catch (const std::exception& error) {
    candidate_error(diagnostic, error.what());
    return false;
  } catch (...) {
    candidate_error(diagnostic, "ABI-v5 authority fixture preparation failed");
    return false;
  }
}
}  // namespace
)CPP";

  std::string block_table =
      "namespace { constexpr pops::runtime::program::ProgramBlockRecord kProgramBlocks[] = {";
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    if (index != 0)
      block_table += ", ";
    block_table += "{{" + cxx_string_literal(blocks[index]) + ", " +
                   std::to_string(blocks[index].size()) + "}}";
  }
  block_table += "}; }\n";
  source << block_table;

  const std::string identity_literal = cxx_string_literal(identity);
  const std::string marker_literal = cxx_string_literal(marker_path);
  const std::string release_literal = cxx_string_literal(release_path);
  const std::string empty_manifest = callback_empty_resource_manifest_literal();

  std::string body = R"CPP(
extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor* host,
    pops::runtime::program::ProgramCandidateDescriptor* candidate,
    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  using namespace pops::runtime::program;
  if (host == nullptr || candidate == nullptr || !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::uniform || host->services.state_store == nullptr) {
    candidate_error(diagnostic, "ABI-v5 authority fixture received an invalid host descriptor");
    return false;
  }
  *candidate = {};
  try {
    auto state = std::make_unique<ProgramCandidateState>();
    ProgramCandidateDescriptor descriptor{};
    descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));
    descriptor.abi_version = kProgramInstallAbiVersion;
    descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
    descriptor.runtime_kind = ProgramRuntimeKind::uniform;
    descriptor.provided_capability_bits = host->capability_bits;
    descriptor.required_capability_bits = 0;
    descriptor.required_service_bits = kProgramServiceState | kProgramServiceClock |
                                       kProgramServiceTransaction;
    static constexpr char kIdentity[] = %IDENTITY%;
    static constexpr char kAbiKey[] = POPS_ABI_KEY_LITERAL;
    static constexpr char kResourceManifest[] = %MANIFEST%;
    descriptor.program_name = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.artifact_identity = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.abi_key = {kAbiKey, sizeof(kAbiKey) - 1};
    descriptor.route_manifest = {pops::kRouteRegistrySignature,
                                 static_cast<std::uint64_t>(
                                     std::char_traits<char>::length(pops::kRouteRegistrySignature))};
    descriptor.boundary_manifest = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.persistent_resource_manifest = {kResourceManifest,
                                                sizeof(kResourceManifest) - 1};
    descriptor.checkpoint_identity = {kIdentity, sizeof(kIdentity) - 1};
    descriptor.blocks = {kProgramBlocks, %BLOCK_COUNT%, sizeof(ProgramBlockRecord)};
    descriptor.maximum_bytes = 0;
    descriptor.context = state.get();
    descriptor.prepare = &candidate_prepare;
    descriptor.step = &candidate_step;
    descriptor.destroy = &candidate_destroy;
    *candidate = descriptor;
    (void)state.release();
    return true;
  } catch (const std::exception& error) {
    candidate_error(diagnostic, error.what());
    return false;
  } catch (...) {
    candidate_error(diagnostic, "ABI-v5 authority fixture candidate construction failed");
    return false;
  }
}
)CPP";
  source << body;
  std::string result = source.str();
  auto replace_all = [&result](std::string_view needle, std::string_view replacement) {
    std::size_t offset = 0;
    while ((offset = result.find(needle, offset)) != std::string::npos) {
      result.replace(offset, needle.size(), replacement);
      offset += replacement.size();
    }
  };
  replace_all("%REJECT_FIRST_ATTEMPT%", mode == kRejectFirstAttempt ? "true" : "false");
  replace_all("%REJECT_FIRST_DT%", mode == kRejectFirstDt ? "true" : "false");
  replace_all("%PROBE_PUBLIC%", mode == kProbePublic ? "true" : "false");
  replace_all("%PROBE_PROVISIONAL%", mode == kProbeProvisional ? "true" : "false");
  replace_all("%WAIT_FOR_RELEASE%", mode == kWaitForRelease ? "true" : "false");
  replace_all("%MARKER%", marker_literal);
  replace_all("%RELEASE%", release_literal);
  replace_all("%IDENTITY%", identity_literal);
  replace_all("%MANIFEST%", empty_manifest);
  replace_all("%BLOCK_COUNT%", std::to_string(blocks.size()));
  // clang-format on
  return result;
}

}  // namespace pops::test::program_v5
