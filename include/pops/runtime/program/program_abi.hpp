#pragma once

/// @file
/// @brief Versioned, ownership-explicit native Program installation ABI.
///
/// This header is deliberately POD-only at the DSO boundary.  A generated Program contributes
/// descriptors plus function/context pointers; the host keeps ownership of the loaded image and
/// destroys candidate state before it ever closes that image.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace pops::runtime::program {

template <int Dim>
class ProgramExecutionServices;
class AcceptedProgramExecutionServicesSnapshot;
struct ProgramInstallDiagnostic;
struct ProgramHostDescriptor;

inline constexpr std::uint32_t kProgramInstallAbiVersion = 5;

enum class ProgramRuntimeKind : std::uint32_t { uniform = 1, amr = 2 };
enum class ProgramExecutionLane : std::uint32_t { host = 1, device = 2, distributed = 3 };
enum class ProgramInstallErrorCode : std::uint32_t {
  none = 0,
  invalid_host_descriptor = 1,
  unsupported_runtime = 2,
  invalid_candidate = 3,
  resource_limit = 4,
  artifact_rejected = 5,
};

enum ProgramCapability : std::uint64_t {
  kProgramCapabilityNone = 0,
  kProgramCapabilityHierarchy = std::uint64_t{1} << 0,
  kProgramCapabilitySchedules = std::uint64_t{1} << 1,
  kProgramCapabilityCellTemporal = std::uint64_t{1} << 2,
  kProgramCapabilityPersistentValues = std::uint64_t{1} << 3,
  kProgramCapabilityTransactions = std::uint64_t{1} << 4,
};

enum ProgramRequiredService : std::uint64_t {
  kProgramServiceState = std::uint64_t{1} << 0,
  kProgramServiceFields = std::uint64_t{1} << 1,
  kProgramServiceSpatial = std::uint64_t{1} << 2,
  kProgramServiceHierarchy = std::uint64_t{1} << 3,
  kProgramServiceHistory = std::uint64_t{1} << 4,
  kProgramServiceClock = std::uint64_t{1} << 5,
  kProgramServiceReduction = std::uint64_t{1} << 6,
  kProgramServiceTransaction = std::uint64_t{1} << 7,
  kProgramServicePersistentValues = std::uint64_t{1} << 8,
};

inline constexpr std::uint64_t kKnownProgramCapabilityBits =
    kProgramCapabilityHierarchy | kProgramCapabilitySchedules | kProgramCapabilityCellTemporal |
    kProgramCapabilityPersistentValues | kProgramCapabilityTransactions;
inline constexpr std::uint64_t kKnownProgramServiceBits =
    kProgramServiceState | kProgramServiceFields | kProgramServiceSpatial |
    kProgramServiceHierarchy | kProgramServiceHistory | kProgramServiceClock |
    kProgramServiceReduction | kProgramServiceTransaction | kProgramServicePersistentValues;

struct ProgramAbiView final {
  const char* data = nullptr;
  std::uint64_t size = 0;
};

/// Fixed-layout DSO view.  The host copies every element before running candidate preparation;
/// a table never transfers allocation or ownership across the ABI.
struct ProgramAbiTable final {
  const void* data = nullptr;
  std::uint64_t count = 0;
  std::uint64_t element_size = 0;
};

struct ProgramBlockRecord final {
  ProgramAbiView name{};
};
struct ProgramParameterRecord final {
  std::int32_t block = -1;
  std::int32_t index = -1;
  double default_value = 0.0;
  ProgramAbiView name{};
};
struct ProgramAuthorityRecord final {
  std::uint64_t words[4]{};
};
/// Selective-history authority.  `identity` is the stable declared history key; depth is the
/// number of accepted images retained by the candidate (and is never inferred from a live ring).
struct ProgramHistoryAuthorityRecord final {
  ProgramAbiView identity{};
  std::uint32_t depth = 0;
  std::uint32_t reserved = 0;
};

/// One immutable checkpoint component.  Every identity is explicit so a regrid/rank-change
/// implementation can reject an incomplete candidate before it mutates the live hierarchy.
struct ProgramCheckpointRecord final {
  ProgramAbiView identity{};
  ProgramAbiView owner{};
  ProgramAbiView space{};
  ProgramAbiView clock{};
  ProgramAbiView transfer{};
  std::int32_t block = -1;
  std::int32_t components = -1;
  std::uint64_t retained_images = 0;
};

/// Frozen per-block interface bounds.  Zero is an authenticated bound, not an absent entry.
struct ProgramFluxBudgetRecord final {
  std::uint64_t rhs_basis_bound = 0;
  std::uint64_t coefficient_term_bound = 0;
  std::uint64_t interface_application_bound = 0;
  std::uint64_t interface_identity_character_bound = 0;
};

/// Persistent/transient allocation promised by the compiled Program before a single candidate
/// callback runs.  This is deliberately a lossless projection of
/// ``ProgramResourcePlan.abi_data()``: compact ids make the bound store cheap, while the complete
/// path/owner/space/clock and policy strings let the host reject a collision or a lossy lowering
/// before it gives the candidate any service callback.  Every view is copied by the host.
///
/// Runtime-sized rows use ``kProgramResourcePlanUnknownExtent`` for ``bytes``, ``maximum_bytes``,
/// ``cells`` and ``itemsize``; none of those fields is an exact claim until the host seals the plan.
/// Exact rows use the same sentinel only for an omitted cells/itemsize JSON field.  ``shape`` and
/// ``component_names`` are canonical JSON arrays, including ``[]`` when empty; this avoids
/// delimiter-dependent ABI encodings.
inline constexpr std::uint32_t kProgramResourcePlanSchemaVersion = 1;
inline constexpr std::uint64_t kProgramResourcePlanUnknownExtent =
    std::numeric_limits<std::uint64_t>::max();
enum ProgramResourcePlanFlags : std::uint32_t {
  kProgramResourcePersistentSchedule = std::uint32_t{1} << 0,
  kProgramResourceCommunicates = std::uint32_t{1} << 1,
  kProgramResourceRestartRequired = std::uint32_t{1} << 2,
  kProgramResourceHasCells = std::uint32_t{1} << 3,
  kProgramResourceHasItemsize = std::uint32_t{1} << 4,
  /// The row is an install-time declaration.  It must not claim any extent or byte count;
  /// the host seals it only after every prepare_* callback has supplied its exact layout.
  kProgramResourceRuntimeSized = std::uint32_t{1} << 5,
};
inline constexpr std::uint32_t kKnownProgramResourcePlanFlags =
    kProgramResourcePersistentSchedule | kProgramResourceCommunicates |
    kProgramResourceRestartRequired | kProgramResourceHasCells | kProgramResourceHasItemsize |
    kProgramResourceRuntimeSized;

/// Explicit v5 resource sizing tag.  ``runtime_sized`` is a declaration only: its extent and
/// footprint fields are sentinels until a host materializer seals the final plan.
enum class ProgramResourcePlanType : std::uint32_t {
  exact = 0,
  runtime_sized = 1,
};

struct ProgramResourcePlanRecord final {
  std::uint32_t struct_size = sizeof(ProgramResourcePlanRecord);
  std::uint32_t schema_version = kProgramResourcePlanSchemaVersion;
  std::uint32_t slot = 0;
  std::uint32_t flags = 0;
  std::uint64_t value_id = 0;
  std::uint64_t occurrence_path_id = 0;
  std::int32_t level = -1;
  std::uint32_t components = 0;
  std::uint32_t ghosts = 0;
  std::uint32_t reserved = 0;
  std::uint64_t bytes = 0;
  std::uint64_t maximum_bytes = 0;
  std::uint64_t cells = kProgramResourcePlanUnknownExtent;
  std::uint64_t itemsize = kProgramResourcePlanUnknownExtent;
  ProgramAbiView schema{};
  ProgramAbiView plan_digest{};
  ProgramAbiView identity{};
  ProgramAbiView occurrence_path{};
  ProgramAbiView owner{};
  ProgramAbiView space{};
  ProgramAbiView clock{};
  ProgramAbiView lifetime{};
  ProgramAbiView centering{};
  ProgramAbiView off_policy{};
  ProgramAbiView communication{};
  ProgramAbiView transfer_provider{};
  ProgramAbiView restart_provider{};
  ProgramAbiView component_names{};
  ProgramAbiView shape{};
  /// Must agree with ``kProgramResourceRuntimeSized``. Generated symbolic rows set both the flag
  /// and this explicit type; there is no alternate spelling or compatibility alias in ABI v5.
  ProgramResourcePlanType resource_type = ProgramResourcePlanType::exact;
};

/// Declared field-boundary/provider route.  Routes are metadata, never a host callback or a
/// dynamic handle: their callable realization stays inside the prepared candidate state.
struct ProgramRouteRecord final {
  ProgramAbiView identity{};
  ProgramAbiView kind{};
  std::uint64_t capability_bits = 0;
};

/// Owner-qualified compiled module record.  This replaces the retired accessor family, including
/// the structural operator requirements used during installation validation.
struct ProgramModuleRecord final {
  ProgramAbiView identity{};
  ProgramAbiView kind{};
  ProgramAbiView signature{};
  ProgramAbiView requirements{};
  ProgramAbiView owner{};
};

/// Host-only staging callback invoked exclusively by candidate preparation.  A false return leaves
/// the candidate unprepared and the enclosing facade transaction unchanged.
using ProgramStageFn = bool (*)(void* host_state, const ProgramAbiTable* table,
                                ProgramInstallDiagnostic* diagnostic) noexcept;

/// Non-owning references to host-prepared services.  Their concrete types are intentionally not an
/// ABI concern: their lifetime is the installation transaction and then the owning System/AmrSystem.
struct ProgramExecutionServicesRef final {
  void* state_store = nullptr;
  void* field_store = nullptr;
  void* spatial_executor = nullptr;
  void* hierarchy_executor = nullptr;
  void* history_store = nullptr;
  void* clock_service = nullptr;
  void* reduction_service = nullptr;
  void* transaction_service = nullptr;
  void* persistent_value_store = nullptr;
};

/// Host-owned preparation image.  The opaque image is intentionally distinct from the eventual
/// facade: generated candidates may use only this sealed service set while `prepare` runs.
/// Publication replaces the image with the accepted runtime owner in one noexcept move.
struct ProgramPreparationHostRef final {
  std::uint32_t struct_size = sizeof(ProgramPreparationHostRef);
  std::uint32_t abi_version = kProgramInstallAbiVersion;
  void* image = nullptr;
  ProgramExecutionServicesRef services{};
};

/// Host descriptor passed to the sole `pops_install_program` entry point.
struct ProgramHostDescriptor final {
  std::uint32_t struct_size = sizeof(ProgramHostDescriptor);
  std::uint32_t abi_version = kProgramInstallAbiVersion;
  std::uint32_t native_dimension = 0;
  ProgramRuntimeKind runtime_kind = ProgramRuntimeKind::uniform;
  ProgramExecutionLane execution_lane = ProgramExecutionLane::host;
  std::uint64_t capability_bits = kProgramCapabilityNone;
  ProgramPreparationHostRef preparation{};
  ProgramExecutionServicesRef services{};
};

/// DSO candidate returned by `pops_install_program`.  No field may own a loader handle.
struct ProgramCandidateDescriptor final {
  using StepFn = void (*)(void* context, double dt);
  using DtBoundFn = double (*)(void* context, double cfl);
  using HierarchyRefreshFn = void (*)(void* context);
  using HistoryRemapAcceptedFn = void (*)(void* context, const void* remap_descriptor);
  using RestartHookFn = void (*)(void* context);
  using AcceptedSnapshotCreateFn = AcceptedProgramExecutionServicesSnapshot* (*)(void* context);
  using PrepareFn = bool (*)(void* context, const ProgramHostDescriptor* host,
                             ProgramInstallDiagnostic* diagnostic) noexcept;
  using DestroyFn = void (*)(void* context) noexcept;

  std::uint32_t struct_size = sizeof(ProgramCandidateDescriptor);
  std::uint32_t abi_version = kProgramInstallAbiVersion;
  std::uint32_t native_dimension = 0;
  ProgramRuntimeKind runtime_kind = ProgramRuntimeKind::uniform;
  std::uint64_t provided_capability_bits = kProgramCapabilityNone;
  std::uint64_t required_capability_bits = kProgramCapabilityNone;
  std::uint64_t required_service_bits = 0;
  ProgramAbiView program_name{};
  ProgramAbiView artifact_identity{};
  ProgramAbiView abi_key{};
  ProgramAbiView route_manifest{};
  ProgramAbiView boundary_manifest{};
  ProgramAbiView persistent_resource_manifest{};
  ProgramAbiView checkpoint_identity{};
  ProgramAbiTable blocks{};
  ProgramAbiTable parameters{};
  ProgramAbiTable operator_authorities{};
  ProgramAbiTable history_authorities{};
  ProgramAbiTable checkpoint_shape{};
  ProgramAbiTable flux_budgets{};
  ProgramAbiTable resource_plan{};
  ProgramAbiTable boundary_routes{};
  ProgramAbiTable provider_routes{};
  ProgramAbiTable module_operators{};
  ProgramAbiTable module_state_spaces{};
  ProgramAbiTable module_field_spaces{};
  std::uint64_t maximum_bytes = 0;
  void* context = nullptr;
  PrepareFn prepare = nullptr;
  StepFn step = nullptr;
  DtBoundFn dt_bound = nullptr;
  HierarchyRefreshFn hierarchy_refresh = nullptr;
  HistoryRemapAcceptedFn history_remap_accepted = nullptr;
  RestartHookFn restart_regrid_preflight = nullptr;
  RestartHookFn restart_regrid = nullptr;
  RestartHookFn restart_resync = nullptr;
  AcceptedSnapshotCreateFn create_accepted_snapshot = nullptr;
  DestroyFn destroy = nullptr;
};

/// Fixed-size, host-owned failure record.  The DSO must never propagate a C++ exception across the
/// native Program ABI; it writes this diagnostic and returns false instead.
struct ProgramInstallDiagnostic final {
  ProgramInstallErrorCode code = ProgramInstallErrorCode::none;
  char message[192]{};
};

using ProgramInstallFn = bool (*)(const ProgramHostDescriptor*, ProgramCandidateDescriptor*,
                                  ProgramInstallDiagnostic*) noexcept;

[[nodiscard]] inline bool valid_program_host_descriptor(
    const ProgramHostDescriptor& value) noexcept {
  const auto same_services = [](const ProgramExecutionServicesRef& left,
                                const ProgramExecutionServicesRef& right) noexcept {
    return left.state_store == right.state_store && left.field_store == right.field_store &&
           left.spatial_executor == right.spatial_executor &&
           left.hierarchy_executor == right.hierarchy_executor &&
           left.history_store == right.history_store && left.clock_service == right.clock_service &&
           left.reduction_service == right.reduction_service &&
           left.transaction_service == right.transaction_service &&
           left.persistent_value_store == right.persistent_value_store;
  };
  return value.struct_size == sizeof(ProgramHostDescriptor) &&
         value.abi_version == kProgramInstallAbiVersion && value.native_dimension >= 1 &&
         value.native_dimension <= 3 &&
         (value.runtime_kind == ProgramRuntimeKind::uniform ||
          value.runtime_kind == ProgramRuntimeKind::amr) &&
         (value.execution_lane == ProgramExecutionLane::host ||
          value.execution_lane == ProgramExecutionLane::device ||
          value.execution_lane == ProgramExecutionLane::distributed) &&
         value.services.state_store && value.services.field_store &&
         value.services.spatial_executor && value.services.history_store &&
         value.services.clock_service && value.services.reduction_service &&
         value.services.transaction_service && value.services.persistent_value_store &&
         value.preparation.struct_size == sizeof(ProgramPreparationHostRef) &&
         value.preparation.abi_version == kProgramInstallAbiVersion &&
         value.preparation.image != nullptr &&
         same_services(value.services, value.preparation.services) &&
         (value.runtime_kind == ProgramRuntimeKind::uniform || value.services.hierarchy_executor) &&
         (value.capability_bits & ~kKnownProgramCapabilityBits) == 0;
}

[[nodiscard]] inline bool valid_program_candidate_descriptor(
    const ProgramCandidateDescriptor& value) noexcept {
  const auto valid_view = [](ProgramAbiView view) noexcept {
    return view.data != nullptr && view.size != 0;
  };
  const bool any_lifecycle_hook = value.hierarchy_refresh || value.history_remap_accepted ||
                                  value.restart_regrid_preflight || value.restart_regrid ||
                                  value.restart_resync || value.create_accepted_snapshot;
  const bool complete_lifecycle_hook_set =
      value.hierarchy_refresh && value.history_remap_accepted && value.restart_regrid_preflight &&
      value.restart_regrid && value.restart_resync && value.create_accepted_snapshot;
  const bool requires_amr_lifecycle =
      value.runtime_kind == ProgramRuntimeKind::amr &&
      ((value.provided_capability_bits | value.required_capability_bits) &
       (kProgramCapabilityHierarchy | kProgramCapabilityTransactions)) != 0;
  const auto valid_table = [](ProgramAbiTable table, std::size_t element_size) noexcept {
    constexpr std::uint64_t kMaximumElements = 1u << 20;
    constexpr std::uint64_t kMaximumTableBytes = 64u * 1024u * 1024u;
    if (table.count == 0)
      return table.data == nullptr && table.element_size == 0;
    if (table.data == nullptr || table.element_size != element_size ||
        table.count > kMaximumElements)
      return false;
    return table.count <= kMaximumTableBytes / element_size;
  };
  return value.struct_size == sizeof(ProgramCandidateDescriptor) &&
         value.abi_version == kProgramInstallAbiVersion && valid_view(value.program_name) &&
         valid_view(value.artifact_identity) && valid_view(value.abi_key) &&
         value.native_dimension >= 1 && value.native_dimension <= 3 &&
         (value.runtime_kind == ProgramRuntimeKind::uniform ||
          value.runtime_kind == ProgramRuntimeKind::amr) &&
         valid_view(value.route_manifest) && valid_view(value.boundary_manifest) &&
         valid_view(value.persistent_resource_manifest) && valid_view(value.checkpoint_identity) &&
         valid_table(value.blocks, sizeof(ProgramBlockRecord)) &&
         valid_table(value.parameters, sizeof(ProgramParameterRecord)) &&
         valid_table(value.operator_authorities, sizeof(ProgramAuthorityRecord)) &&
         valid_table(value.history_authorities, sizeof(ProgramHistoryAuthorityRecord)) &&
         valid_table(value.checkpoint_shape, sizeof(ProgramCheckpointRecord)) &&
         valid_table(value.flux_budgets, sizeof(ProgramFluxBudgetRecord)) &&
         valid_table(value.resource_plan, sizeof(ProgramResourcePlanRecord)) &&
         valid_table(value.boundary_routes, sizeof(ProgramRouteRecord)) &&
         valid_table(value.provider_routes, sizeof(ProgramRouteRecord)) &&
         valid_table(value.module_operators, sizeof(ProgramModuleRecord)) &&
         valid_table(value.module_state_spaces, sizeof(ProgramModuleRecord)) &&
         valid_table(value.module_field_spaces, sizeof(ProgramModuleRecord)) && value.prepare &&
         value.step &&
         ((value.context != nullptr && value.destroy != nullptr) ||
          (value.context == nullptr && value.destroy == nullptr)) &&
         (value.runtime_kind == ProgramRuntimeKind::uniform
              ? !any_lifecycle_hook
              : (requires_amr_lifecycle ? complete_lifecycle_hook_set : !any_lifecycle_hook)) &&
         (value.provided_capability_bits & ~kKnownProgramCapabilityBits) == 0 &&
         (value.required_capability_bits & ~kKnownProgramCapabilityBits) == 0 &&
         (value.required_service_bits & ~kKnownProgramServiceBits) == 0;
}

static_assert(std::is_standard_layout_v<ProgramAbiView> &&
              std::is_trivially_copyable_v<ProgramAbiView>);
static_assert(std::is_standard_layout_v<ProgramAbiTable> &&
              std::is_trivially_copyable_v<ProgramAbiTable>);
static_assert(std::is_standard_layout_v<ProgramBlockRecord> &&
              std::is_trivially_copyable_v<ProgramBlockRecord>);
static_assert(std::is_standard_layout_v<ProgramParameterRecord> &&
              std::is_trivially_copyable_v<ProgramParameterRecord>);
static_assert(std::is_standard_layout_v<ProgramAuthorityRecord> &&
              std::is_trivially_copyable_v<ProgramAuthorityRecord>);
static_assert(std::is_standard_layout_v<ProgramHistoryAuthorityRecord> &&
              std::is_trivially_copyable_v<ProgramHistoryAuthorityRecord>);
static_assert(std::is_standard_layout_v<ProgramCheckpointRecord> &&
              std::is_trivially_copyable_v<ProgramCheckpointRecord>);
static_assert(std::is_standard_layout_v<ProgramFluxBudgetRecord> &&
              std::is_trivially_copyable_v<ProgramFluxBudgetRecord>);
static_assert(std::is_standard_layout_v<ProgramResourcePlanRecord> &&
              std::is_trivially_copyable_v<ProgramResourcePlanRecord>);
static_assert(std::is_standard_layout_v<ProgramRouteRecord> &&
              std::is_trivially_copyable_v<ProgramRouteRecord>);
static_assert(std::is_standard_layout_v<ProgramModuleRecord> &&
              std::is_trivially_copyable_v<ProgramModuleRecord>);
static_assert(std::is_standard_layout_v<ProgramExecutionServicesRef> &&
              std::is_trivially_copyable_v<ProgramExecutionServicesRef>);
static_assert(std::is_standard_layout_v<ProgramPreparationHostRef> &&
              std::is_trivially_copyable_v<ProgramPreparationHostRef>);
static_assert(std::is_standard_layout_v<ProgramHostDescriptor> &&
              std::is_trivially_copyable_v<ProgramHostDescriptor>);
static_assert(std::is_standard_layout_v<ProgramCandidateDescriptor> &&
              std::is_trivially_copyable_v<ProgramCandidateDescriptor>);
static_assert(std::is_standard_layout_v<ProgramInstallDiagnostic> &&
              std::is_trivially_copyable_v<ProgramInstallDiagnostic>);

}  // namespace pops::runtime::program
