// Compiled time-program LOADER path (epic ADC-399 / ADC-401 Phase 2c-i): System::install_program
// dlopens a generated problem.so and installs its compiled time Program across the ABI boundary.
//
// We compile AT RUNTIME a stub problem.so -- the role the codegen (Phase 2c-ii) will fill -- that
// exports the sole v5 pops_install_program candidate entry; the installer selects the System
// provider and installs the
// SAME Forward-Euler closure as the in-process test_program_runtime.
// We then sim.install_program(so) + sim.step(dt) and check bit-parity against a reference Forward-Euler
// step computed from the same primitives (solve_fields + eval_rhs + U + dt*R). This validates the
// dlopen + ABI-key guard + globally visible host seams with a locally scoped package, end to end.
//
// The runtime package is compiled with the exact compiler/Kokkos contract injected by CMake. A
// missing compiler or a package compilation failure is a test failure: this proof never self-skips.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/cache_manager.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/system.hpp>
#include <pops/core/identity/sha256.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int kTestDimension = kNativeDimension;
using NativeSystem = System<kTestDimension>;
using NativeSystemConfig = SystemConfig<kTestDimension>;
using NativeField = MultiFab<kTestDimension>;
using GasLaw = nd::IdealGasEuler<kTestDimension>;
using GasModel = CompositeModel<GasLaw, NoSource, NoElliptic>;
using CacheSlot = runtime::program::ProgramCacheSlot;
constexpr double kGamma = 1.4;
constexpr int kGasComponents = GasModel::n_vars;

template <int Dim>
void install_runtime_authority(System<Dim>& system, std::string_view identity) {
  auto lane =
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(identity));
  system.install_prepared_boundary_execution_lane(std::move(lane));
}

std::size_t cell_count(int n) {
  std::size_t count = 1;
  for (int axis = 0; axis < kTestDimension; ++axis)
    count *= static_cast<std::size_t>(n);
  return count;
}

NativeSystemConfig native_config(int n) {
  NativeSystemConfig config;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    config.shape[axis] = n;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  config.boxes = {Box<kTestDimension>::from_extents(config.shape)};
  return config;
}

void fill_ic(std::vector<double>& U, int n) {
  const std::size_t nn = cell_count(n);
  const double pi = 3.14159265358979323846;
  for (std::size_t cell = 0; cell < nn; ++cell) {
    std::size_t remaining = cell;
    double mode = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis) {
      const int index = static_cast<int>(remaining % static_cast<std::size_t>(n));
      remaining /= static_cast<std::size_t>(n);
      const double coordinate = (static_cast<double>(index) + 0.5) / n;
      mode *= std::cos(2 * pi * coordinate);
    }
    const double pressure = 3.0 + 0.5 * mode;
    U[cell] = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis)
      U[static_cast<std::size_t>(axis + 1) * nn + cell] = 0.0;
    U[static_cast<std::size_t>(kTestDimension + 1) * nn + cell] = pressure / (kGamma - 1.0);
  }
}

void add_gas(NativeSystem& system) {
  GasModel model{};
  model.hyp = GasLaw::prepare(static_cast<Real>(kGamma));
  system.install_block_state_route("gas", "test:program-loader/gas/state");
  add_compiled_model(system, "gas", std::move(model), "minmod", "rusanov", "conservative",
                     "explicit", kGamma);
  system.set_poisson("charge_density", "cartesian_cg");
}

runtime::system::AuxiliaryComponentKey install_field_output(NativeSystem& system,
                                                            const std::string& owner,
                                                            const std::string& field) {
  using namespace runtime::system;
  AuxiliaryStorageShape<kTestDimension> shape;
  for (int axis = 0; axis < kTestDimension; ++axis)
    shape.halo[axis] = 1;
  AuxiliaryComponentKey key{owner, "field", field, "potential"};
  AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kTestDimension>{
      "test.field-output/" + owner + "/" + field,
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
}

std::vector<std::string> read_program_markers(const std::string& path) {
  std::ifstream input(path);
  std::vector<std::string> result;
  for (std::string line; std::getline(input, line);)
    result.push_back(std::move(line));
  return result;
}

std::size_t marker_index(const std::vector<std::string>& markers, const std::string& needle) {
  for (std::size_t index = 0; index < markers.size(); ++index)
    if (markers[index] == needle)
      return index;
  return markers.size();
}

std::size_t marker_count(const std::vector<std::string>& markers, const std::string& needle) {
  std::size_t count = 0;
  for (const auto& marker : markers)
    if (marker == needle)
      ++count;
  return count;
}

std::pair<std::string, std::string> canonical_cache_manifest() {
  using namespace runtime::program;
  ProgramInstallationTables tables;
  ProgramInstallationTables::ResourcePlan row;
  row.slot = 0;
  row.flags = kProgramResourcePersistentSchedule;
  row.value_id = 0x100;
  row.occurrence_path_id = 0x200;
  row.level = -1;
  row.components = 1;
  row.ghosts = 0;
  row.bytes = 8;
  row.maximum_bytes = 8;
  row.resource_type = ProgramResourcePlanType::exact;
  row.schema = "program-resource-plan:v1";
  row.identity = "test.program-loader/cache";
  row.occurrence_path = row.identity;
  row.owner = "test.program-loader";
  row.space = "cell";
  row.clock = "clock.macro";
  row.lifetime = "persistent_schedule";
  row.centering = "cell";
  row.off_policy = "hold";
  row.communication = "none";
  row.transfer_provider = "none";
  row.restart_provider = "none";
  row.component_names = "[\"state\"]";
  row.shape = "[]";
  tables.resource_plan.push_back(std::move(row));
  const std::string payload = tables.canonical_resource_digest_payload(std::uint64_t{8});
  const std::string digest =
      identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  tables.resource_plan.front().plan_digest = digest;
  const std::string manifest =
      "{\"resource_plan\":" + tables.canonical_resource_manifest(8, digest) +
      ",\"resource_plan_digest\":\"" + digest + "\"}";
  return {digest, manifest};
}

std::pair<std::string, std::string> canonical_empty_resource_manifest() {
  using namespace runtime::program;
  ProgramInstallationTables tables;
  const std::string payload = tables.canonical_resource_digest_payload(std::uint64_t{0});
  const std::string digest =
      identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  const std::string manifest =
      "{\"resource_plan\":" + tables.canonical_resource_manifest(0, digest) +
      ",\"resource_plan_digest\":\"" + digest + "\"}";
  return {digest, manifest};
}

// The generated problem.so: a Forward-Euler Program installed via ProgramExecutionServices. This is exactly the
// source the Phase 2c-ii codegen will emit (here hand-written for an autonomous C++ test). The ABI key
// is the preprocessor LITERAL (not the inline abi_key_string(), which would be interposed via RTLD).
std::string loader_source(bool include_block_identities = true, bool install_step = true,
                          bool incomplete_dt_bound = false,
                          const std::string& dynamic_boundary_slot = {},
                          bool register_history = false, bool boundary_install_throws = false,
                          const std::string& marker_path = {}, const std::string& marker_tag = {},
                          bool include_cache_plan = false) {
  const auto [cache_digest, cache_manifest] = canonical_cache_manifest();
  const auto empty_manifest = canonical_empty_resource_manifest().second;
  // clang-format off
  std::string source = R"CPP(
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/core/foundation/types.hpp>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
)CPP";
  if (!marker_path.empty()) {
    source += "namespace {\n"
              "static constexpr char kProgramMarkerPath[] = \"" + marker_path + "\";\n"
              "static constexpr char kProgramMarkerTag[] = \"" + marker_tag + "\";\n"
              "void program_marker(const char* event) {\n"
              "  std::ofstream marker(kProgramMarkerPath, std::ios::app);\n"
              "  marker << kProgramMarkerTag << ':' << event << '\\n';\n"
              "}\n"
              "__attribute__((destructor)) void program_image_unloaded() { program_marker(\"dlclose\"); }\n"
              "}  // namespace\n";
  } else {
    source += "namespace { void program_marker(const char*) {} }  // namespace\n";
  }
  source += R"CPP(
namespace {
struct ProgramCandidateState final {
  std::shared_ptr<pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>> ctx_owner;
  std::function<void(double)> step;
  bool register_history = false;
  bool boundary_install_throws = false;
  bool prepared = false;

  ~ProgramCandidateState() {
    // The host must release the callback closure before its shared execution-services owner.  The
    // markers are intentionally emitted from the DSO so the parent process can prove this order
    // across the actual destroy-callback/dlclose boundary.
    if (step) {
      step = {};
      program_marker("closure_destroy");
    }
    if (ctx_owner) {
      ctx_owner.reset();
      program_marker("owner_destroy");
    }
  }
};
void program_candidate_step(void* opaque, double dt) {
  static_cast<ProgramCandidateState*>(opaque)->step(dt);
}
void program_candidate_destroy(void* opaque) noexcept {
  delete static_cast<ProgramCandidateState*>(opaque);
  // These events occur after the opaque candidate state is gone but while the DSO is still loaded.
  // They distinguish the callback context and callback itself from the host's subsequent dlclose.
  program_marker("ctx_destroy");
  program_marker("candidate_destroy");
}
void program_install_error(pops::runtime::program::ProgramInstallDiagnostic*, const char*) noexcept;
bool program_candidate_prepare(void* opaque, const pops::runtime::program::ProgramHostDescriptor* host,
                               pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  if (!opaque || !host || !host->preparation.image) {
    if (diagnostic) diagnostic->code = pops::runtime::program::ProgramInstallErrorCode::invalid_host_descriptor;
    return false;
  }
  auto* state = static_cast<ProgramCandidateState*>(opaque);
  if (state->prepared || state->ctx_owner || state->step) {
    if (diagnostic) diagnostic->code = pops::runtime::program::ProgramInstallErrorCode::artifact_rejected;
    return false;
  }
  try {
    state->ctx_owner = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(
        host->preparation);
    auto& ctx = *state->ctx_owner;
    ctx.declare_diagnostic("kept-diagnostic");
    ctx.declare_diagnostic("artifact-A-diagnostic");
    if (state->register_history)
      ctx.register_history("artifact.history", 1, 1, 0, "test:artifact/state",
                           "test:artifact/space", "clock.macro", "test:artifact/interp");
    ctx.configure_primary_clock("clock.macro");
    state->step = [ctx_owner = state->ctx_owner](double dt) {
      auto& ctx = *ctx_owner;
      ctx.begin_step(dt);
      ctx.set_stage_time(0, 1);
      auto field_outcome = ctx.solve_fields();
      (void)field_outcome.consume(pops::SolveConsumption::kAccept);
      for (int b = 0; b < ctx.n_blocks(); ++b) {
        pops::MultiFab<pops::kNativeDimension>& U = ctx.state(b);
        pops::MultiFab<pops::kNativeDimension> R = ctx.rhs_scratch_like(U);
        ctx.rhs_into(b, U, R, 0);
        ctx.axpy(U, static_cast<pops::Real>(dt), R);
      }
    };
    // The fault is deliberately injected only after both DSO-owned objects exist.  A refused
    // replacement must therefore release its callback closure, then its execution-services owner,
    // then its opaque candidate, all before dlclose.
    if (state->boundary_install_throws)
      throw std::runtime_error("injected boundary publication failure");
    state->prepared = true;
    return true;
  } catch (const std::exception& error) {
    program_install_error(diagnostic, error.what());
    return false;
  } catch (...) {
    program_install_error(diagnostic, "test Program preparation failed with a non-standard exception");
    return false;
  }
}
void program_install_error(pops::runtime::program::ProgramInstallDiagnostic* diagnostic,
                           const char* message) noexcept {
  if (!diagnostic)
    return;
  diagnostic->code = pops::runtime::program::ProgramInstallErrorCode::artifact_rejected;
  std::size_t index = 0;
  while (index + 1 < sizeof(diagnostic->message) && message[index] != '\0') {
    diagnostic->message[index] = message[index];
    ++index;
  }
  diagnostic->message[index] = '\0';
}
}  // namespace
)CPP";
  if (include_block_identities) {
    source += R"CPP(
namespace { constexpr pops::runtime::program::ProgramBlockRecord kProgramBlocks[] = {{{"gas", 3}}}; }
namespace {
// v5 authenticates a zero flux envelope for every declared Program block, even when this
// uniform loader fixture emits no static basis or face-flux rows.
constexpr pops::runtime::program::ProgramFluxBudgetRecord kProgramFluxBudgets[] = {
    {0, 0, 0, 0}};
}
)CPP";
  }
  if (include_cache_plan) {
    source += "static constexpr char kProgramCacheDigest[] = \"" + cache_digest + "\";\n";
    source += R"CPP(
namespace {
static constexpr char kProgramCacheSchema[] = "program-resource-plan:v1";
// The host inserts the digest computed from ProgramInstallationTables' canonical payload.
static constexpr char kProgramCacheIdentity[] = "test.program-loader/cache";
static constexpr char kProgramCacheOwner[] = "test.program-loader";
static constexpr char kProgramCacheSpace[] = "cell";
static constexpr char kProgramCacheClock[] = "clock.macro";
static constexpr char kProgramCacheLifetime[] = "persistent_schedule";
static constexpr char kProgramCacheCentering[] = "cell";
static constexpr char kProgramCacheOffPolicy[] = "hold";
static constexpr char kProgramCacheCommunication[] = "none";
static constexpr char kProgramCacheProvider[] = "none";
static constexpr char kProgramCacheComponents[] = "[\"state\"]";
static constexpr char kProgramCacheShape[] = "[]";
constexpr pops::runtime::program::ProgramResourcePlanRecord make_program_cache_resource() {
  using namespace pops::runtime::program;
  ProgramResourcePlanRecord row{};
  row.slot = 0;
  row.flags = kProgramResourcePersistentSchedule;
  row.value_id = 0x100;
  row.occurrence_path_id = 0x200;
  row.level = -1;
  row.components = 1;
  row.ghosts = 0;
  row.bytes = 8;
  row.maximum_bytes = 8;
  row.cells = kProgramResourcePlanUnknownExtent;
  row.itemsize = kProgramResourcePlanUnknownExtent;
  row.schema = {kProgramCacheSchema, sizeof(kProgramCacheSchema) - 1};
  row.plan_digest = {kProgramCacheDigest, sizeof(kProgramCacheDigest) - 1};
  row.identity = {kProgramCacheIdentity, sizeof(kProgramCacheIdentity) - 1};
  row.occurrence_path = {kProgramCacheIdentity, sizeof(kProgramCacheIdentity) - 1};
  row.owner = {kProgramCacheOwner, sizeof(kProgramCacheOwner) - 1};
  row.space = {kProgramCacheSpace, sizeof(kProgramCacheSpace) - 1};
  row.clock = {kProgramCacheClock, sizeof(kProgramCacheClock) - 1};
  row.lifetime = {kProgramCacheLifetime, sizeof(kProgramCacheLifetime) - 1};
  row.centering = {kProgramCacheCentering, sizeof(kProgramCacheCentering) - 1};
  row.off_policy = {kProgramCacheOffPolicy, sizeof(kProgramCacheOffPolicy) - 1};
  row.communication = {kProgramCacheCommunication, sizeof(kProgramCacheCommunication) - 1};
  row.transfer_provider = {kProgramCacheProvider, sizeof(kProgramCacheProvider) - 1};
  row.restart_provider = {kProgramCacheProvider, sizeof(kProgramCacheProvider) - 1};
  row.component_names = {kProgramCacheComponents, sizeof(kProgramCacheComponents) - 1};
  row.shape = {kProgramCacheShape, sizeof(kProgramCacheShape) - 1};
  return row;
}
constexpr pops::runtime::program::ProgramResourcePlanRecord kProgramResources[] = {
    make_program_cache_resource()};
}  // namespace
)CPP";
    source += "static constexpr char persistent_resource_manifest[] = R\"JSON(" +
              cache_manifest + ")JSON\";\n";
  } else {
    source += "static constexpr char persistent_resource_manifest[] = R\"JSON(" +
              empty_manifest + ")JSON\";\n";
  }
  (void)incomplete_dt_bound;  // v5 has no auxiliary dt symbol: nullptr means no authored bound.
  source += R"CPP(
extern "C" pops::runtime::program::ProgramInstallAbiProbe
pops_program_install_abi_probe_v5() noexcept {
  return pops::runtime::program::make_program_install_abi_probe();
}

)CPP";
  if (install_step) {
    source += R"CPP(
extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor* host,
    pops::runtime::program::ProgramCandidateDescriptor* candidate,
    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  using namespace pops::runtime::program;
  if (!host || !candidate || !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::uniform || !host->services.state_store) {
    program_install_error(diagnostic, "invalid test Program host");
    return false;
  }
  *candidate = {};
  try {
  auto state = std::make_unique<ProgramCandidateState>();
)CPP";
    if (register_history) {
      source += R"CPP(
  state->register_history = true;
)CPP";
    }
    if (boundary_install_throws) {
      source += R"CPP(
  state->boundary_install_throws = true;
)CPP";
    }
    source += R"CPP(
  ProgramCandidateDescriptor descriptor{};
  descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));
  descriptor.abi_version = kProgramInstallAbiVersion;
  descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  descriptor.runtime_kind = ProgramRuntimeKind::uniform;
  descriptor.provided_capability_bits = host->capability_bits;
  descriptor.required_capability_bits = 0;
  descriptor.required_service_bits = kProgramServiceState;
  static constexpr char identity[] = "test-program-loader";
  static constexpr char abi_key[] = POPS_ABI_KEY_LITERAL;
  descriptor.program_name = {identity, sizeof(identity) - 1};
  descriptor.artifact_identity = {identity, sizeof(identity) - 1};
  descriptor.abi_key = {abi_key, sizeof(abi_key) - 1};
  descriptor.route_manifest = {pops::kRouteRegistrySignature,
      static_cast<std::uint64_t>(std::char_traits<char>::length(pops::kRouteRegistrySignature))};
  descriptor.boundary_manifest = {identity, sizeof(identity) - 1};
  descriptor.persistent_resource_manifest = {persistent_resource_manifest,
                                             sizeof(persistent_resource_manifest) - 1};
  descriptor.checkpoint_identity = {identity, sizeof(identity) - 1};
)CPP";
  if (include_block_identities) {
    source += R"CPP(
  descriptor.blocks = {kProgramBlocks, 1, sizeof(ProgramBlockRecord)};
  descriptor.flux_budgets = {kProgramFluxBudgets, 1, sizeof(ProgramFluxBudgetRecord)};
)CPP";
    }
    if (include_cache_plan)
      source += R"CPP(
  descriptor.resource_plan = {kProgramResources, 1, sizeof(ProgramResourcePlanRecord)};
)CPP";
    // The generated value rows authenticate only cache storage.  The host adds its detached,
    // exact hot workspace at seal time, so this pre-seal ceiling must remain symbolic.
    source += R"CPP(
  descriptor.maximum_bytes = kProgramResourcePlanUnknownExtent;
)CPP";
    source += R"CPP(
  descriptor.context = state.get();
  descriptor.prepare = &program_candidate_prepare;
  descriptor.step = &program_candidate_step;
  descriptor.destroy = &program_candidate_destroy;
  *candidate = descriptor;
  (void)state.release();
  return true;
  } catch (...) {
    program_install_error(diagnostic, "test Program candidate construction failed");
    return false;
  }
}
)CPP";
  } else {
    source += R"CPP(
extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor* host,
    pops::runtime::program::ProgramCandidateDescriptor* candidate,
    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  using namespace pops::runtime::program;
  if (!host || !candidate || !valid_program_host_descriptor(*host) || !host->services.state_store) {
    program_install_error(diagnostic, "invalid test Program host");
    return false;
  }
  *candidate = {};
  try {
  // Entry is intentionally descriptor-only.  A refusing artifact must not materialize a provider
  // or mutate host registries before its prepare callback has been accepted.
  program_install_error(diagnostic, "test Program deliberately omitted its step candidate");
  return false;
  } catch (...) {
    program_install_error(diagnostic, "test Program candidate construction failed");
    return false;
  }
}
)CPP";
  }
  // Boundary route metadata is part of the candidate descriptor.  The synthetic loader fixture
  // deliberately carries no second boundary-install entry point.
  (void)dynamic_boundary_slot;
  // clang-format on
  return source;
}

// This is an actual historical Program entry shape.  It intentionally has no v5 probe: the test
// below proves the loader rejects it before this body can receive a v5 host descriptor.
std::string legacy_uniform_source(const std::string& marker_path) {
  return std::string("#include <pops/runtime/system.hpp>\n"
                     "#include <cstdlib>\n"
                     "#include <fstream>\n"
                     "\n"
                     "extern \"C\" void pops_install_program("
                     "pops::System<pops::kNativeDimension>*) {\n"
                     "  std::ofstream marker(\"") +
         marker_path + "\", std::ios::app);\n"
                       "  marker << \"legacy-install-invoked\\n\";\n"
                       "  std::abort();\n"
                       "}\n";
}

}  // namespace

static int pops_run_test_program_loader(int argc, char** argv) {
  (void)argc;
  const bool marker_child = std::getenv("POPS_PROGRAM_LOADER_MARKER_CHILD") != nullptr;
  if (!marker_child) {
    const std::string marker_path = std::string(POPS_TEST_TMPDIR) + "/program_loader_markers_" +
                                    std::to_string(static_cast<long>(std::clock())) + ".log";
    ::setenv("POPS_PROGRAM_LOADER_MARKER_CHILD", "1", 1);
    ::setenv("POPS_PROGRAM_LOADER_MARKER_PATH", marker_path.c_str(), 1);
    const std::string test_directory = POPS_TEST_TMPDIR;
    const std::size_t separator = test_directory.find_last_of('/');
    const std::string executable = test_directory.substr(0, separator) + "/bin/test_program_loader";
    const std::string command =
        std::string("\"") + executable + "\" --gtest_filter=test_program_loader.Runs";
    const int child_status = std::system(command.c_str());
    ::unsetenv("POPS_PROGRAM_LOADER_MARKER_CHILD");
    ::unsetenv("POPS_PROGRAM_LOADER_MARKER_PATH");
    if (child_status != 0) {
      std::printf("FAIL program-loader lifecycle child exited with status %d\n", child_status);
      return 1;
    }
    const auto markers = read_program_markers(marker_path);
    const auto lifecycle_ok = [&](const std::string& tag, bool prepared) {
      const auto count = [&](const char* event) {
        return marker_count(markers, tag + ":" + event);
      };
      const std::size_t close = marker_index(markers, tag + ":dlclose");
      if (count("ctx_destroy") != 1 || count("candidate_destroy") != 1 || count("dlclose") != 1 ||
          close == markers.size())
        return false;
      if (prepared && (count("closure_destroy") != 1 || count("owner_destroy") != 1))
        return false;
      if (!prepared && (count("closure_destroy") != 0 || count("owner_destroy") != 0))
        return false;
      const std::size_t ctx = marker_index(markers, tag + ":ctx_destroy");
      const std::size_t candidate = marker_index(markers, tag + ":candidate_destroy");
      if (ctx >= candidate || candidate >= close)
        return false;
      if (prepared) {
        const std::size_t closure = marker_index(markers, tag + ":closure_destroy");
        const std::size_t owner = marker_index(markers, tag + ":owner_destroy");
        if (closure >= owner || owner >= ctx)
          return false;
      }
      // No DSO callback, destructor, or trace event may execute after the handle is closed.
      const std::string prefix = tag + ":";
      for (std::size_t index = close + 1; index < markers.size(); ++index)
        if (markers[index].compare(0, prefix.size(), prefix) == 0)
          return false;
      return true;
    };
    if (!lifecycle_ok("first-failure", false) || !lifecycle_ok("failed-replacement", true) ||
        !lifecycle_ok("accepted-a", true) || !lifecycle_ok("replacement-b", true)) {
      std::printf("FAIL Program DSO lifecycle markers did not prove the v5 ownership order\n");
      return 1;
    }
    return 0;
  }

  const int n = 16;
  const double dt = 1e-3;
  const std::size_t nn = cell_count(n);
  std::vector<double> U0(static_cast<std::size_t>(kGasComponents) * nn);
  fill_ic(U0, n);

  const NativeSystemConfig cfg = native_config(n);

  // Reference: one Forward-Euler step via the existing primitives, combined on the host.
  NativeSystem ref(cfg);
  install_runtime_authority(ref, "test.program-loader/runtime-reference@1");
  add_gas(ref);
  ref.set_state("gas", U0);
  (void)pops::consume_solve_outcome(ref.solve_fields());
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(static_cast<std::size_t>(kGasComponents) * nn);
  for (std::size_t k = 0; k < Uref.size(); ++k)
    Uref[k] = U0[k] + dt * R0[k];

  // Compile the stub problem.so and load it via System::install_program.
  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/program_loader_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  const std::string legacy_src = tmp + "_missing_block_identities.cpp";
  const std::string legacy_so = tmp + "_missing_block_identities.so";
  const std::string legacy_signature_src = tmp + "_legacy_signature.cpp";
  const std::string legacy_signature_so = tmp + "_legacy_signature.so";
  const std::string legacy_signature_marker = tmp + "_legacy_signature.markers";
  const std::string no_op_src = tmp + "_no_op_installer.cpp";
  const std::string no_op_so = tmp + "_no_op_installer.so";
  const std::string incomplete_dt_src = tmp + "_incomplete_dt.cpp";
  const std::string incomplete_dt_so = tmp + "_incomplete_dt.so";
  const std::string dynamic_boundary_src = tmp + "_dynamic_boundary.cpp";
  const std::string dynamic_boundary_so = tmp + "_dynamic_boundary.so";
  const std::string failing_boundary_src = tmp + "_failing_boundary.cpp";
  const std::string failing_boundary_so = tmp + "_failing_boundary.so";
  const std::string lifecycle_replacement_src = tmp + "_lifecycle_replacement.cpp";
  const std::string lifecycle_replacement_so = tmp + "_lifecycle_replacement.so";
  const char* configured_marker_path = std::getenv("POPS_PROGRAM_LOADER_MARKER_PATH");
  const std::string marker_path =
      configured_marker_path ? configured_marker_path : tmp + ".markers";
  {
    std::ofstream f(src);
    f << loader_source(true, true, false, {}, false, false, {}, {}, true);
  }
  {
    std::ofstream f(legacy_src);
    f << loader_source(false, true, false, {}, false, false, marker_path, "first-failure");
  }
  {
    std::ofstream f(legacy_signature_src);
    f << legacy_uniform_source(legacy_signature_marker);
  }
  {
    std::ofstream f(no_op_src);
    f << loader_source(true, false);
  }
  {
    std::ofstream f(incomplete_dt_src);
    f << loader_source(true, true, true);
  }
  {
    std::ofstream f(dynamic_boundary_src);
    f << loader_source(true, true, false, "program-boundary-field", true, false, marker_path,
                       "accepted-a", true);
  }
  {
    std::ofstream f(failing_boundary_src);
    f << loader_source(true, true, false, "program-boundary-field", true, true, marker_path,
                       "failed-replacement", true);
  }
  {
    std::ofstream f(lifecycle_replacement_src);
    f << loader_source(true, true, false, {}, false, false, marker_path, "replacement-b", true);
  }
  const auto contains_legacy_install_symbol = [](const std::string& source) {
    return source.find("install_program_step") != std::string::npos ||
           source.find("install_unverified_step") != std::string::npos ||
           source.find("install_program_amr") != std::string::npos ||
           source.find("install_program_cell") != std::string::npos ||
           source.find("install_cell_temporal") != std::string::npos;
  };
  {
    std::ifstream f(src);
    const std::string source((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (contains_legacy_install_symbol(source)) {
      std::printf("FAIL v5 DSO source contains a removed install symbol\n");
      return 1;
    }
  }
  const auto package = pops::test::native_dso::compile_shared(src, so);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader", package);
    return 1;
  }
  const auto legacy_package = pops::test::native_dso::compile_shared(legacy_src, legacy_so);
  if (!legacy_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader legacy package",
                                                   legacy_package);
    return 1;
  }
  const auto legacy_signature_package =
      pops::test::native_dso::compile_shared(legacy_signature_src, legacy_signature_so);
  if (!legacy_signature_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader legacy-signature package",
                                                   legacy_signature_package);
    return 1;
  }
  const auto no_op_package = pops::test::native_dso::compile_shared(no_op_src, no_op_so);
  if (!no_op_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader no-op package",
                                                   no_op_package);
    return 1;
  }
  const auto incomplete_dt_package =
      pops::test::native_dso::compile_shared(incomplete_dt_src, incomplete_dt_so);
  if (!incomplete_dt_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader incomplete-dt package",
                                                   incomplete_dt_package);
    return 1;
  }
  const auto dynamic_boundary_package =
      pops::test::native_dso::compile_shared(dynamic_boundary_src, dynamic_boundary_so);
  if (!dynamic_boundary_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader dynamic-boundary package",
                                                   dynamic_boundary_package);
    return 1;
  }
  const auto failing_boundary_package =
      pops::test::native_dso::compile_shared(failing_boundary_src, failing_boundary_so);
  if (!failing_boundary_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader failing-boundary package",
                                                   failing_boundary_package);
    return 1;
  }
  const auto lifecycle_replacement_package =
      pops::test::native_dso::compile_shared(lifecycle_replacement_src, lifecycle_replacement_so);
  if (!lifecycle_replacement_package.ok) {
    pops::test::native_dso::report_compile_failure(
        "test_program_loader lifecycle-replacement package", lifecycle_replacement_package);
    return 1;
  }

  int fails = 0;
  // The legacy Uniform ABI used the same installation symbol but took System*.  The v5 loader
  // must refuse its missing non-colliding probe before resolving or invoking that symbol.
  NativeSystem legacy_signature(cfg);
  install_runtime_authority(legacy_signature, "test.program-loader/runtime-legacy-signature@1");
  add_gas(legacy_signature);
  try {
    legacy_signature.install_program(legacy_signature_so);
    std::printf("FAIL legacy void(System*) Program entry was accepted\n");
    ++fails;
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find("refusing unprobed pops_install_program") ==
            std::string::npos ||
        std::string(error.what()).find("pops_program_install_abi_probe_v5") == std::string::npos) {
      std::printf("FAIL legacy void(System*) Program diagnostic: %s\n", error.what());
      ++fails;
    }
  }
  if (!read_program_markers(legacy_signature_marker).empty()) {
    std::printf("FAIL legacy void(System*) Program entry was invoked before refusal\n");
    ++fails;
  }

  // An artifact with an empty block table is state-free authority.  It must never install onto a
  // stateful System by add-order: the old positional fallback could silently bind the right
  // equations to the wrong instances.
  NativeSystem missing_identity(cfg);
  install_runtime_authority(missing_identity, "test.program-loader/runtime-missing-identity@1");
  add_gas(missing_identity);
  try {
    missing_identity.install_program(legacy_so);
    std::printf("FAIL Program without a block identity table installed positionally\n");
    ++fails;
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    if (message.find("state-free Program has an empty block identity table") == std::string::npos ||
        message.find("positional Program-to-System binding") == std::string::npos) {
      std::printf("FAIL missing block identity table diagnostic: %s\n", message.c_str());
      ++fails;
    }
  }
  // A v5 installer that runs a prelude but refuses to return a step candidate must not inherit its
  // candidate block map/history or replace an already usable Program. The transaction restores the
  // exact image.
  NativeSystem no_op(cfg);
  install_runtime_authority(no_op, "test.program-loader/runtime-no-op@1");
  add_gas(no_op);
  no_op.set_state("gas", U0);
  // The transaction image is bind-sealed before the first Program step.  Materialize the default
  // field once here so the refused replacement and the subsequent proof step preserve the complete
  // accepted composition, including field ownership.
  no_op.set_potential(std::vector<double>(nn, 0.0));
  // Seed the rollback witness with the same accepted ABI-v5 artifact that the simulation uses;
  // the refusing replacement must preserve this public installation verbatim.
  try {
    no_op.install_program(so);
  } catch (const std::exception& e) {
    std::printf("FAIL baseline v5 loader artifact was refused: %s\n", e.what());
    ++fails;
  }
  const auto program_blocks_before_no_op = no_op.program_block_map();
  no_op.restore_program_cache(0, 1, 0, 0, 0.0, "kept-cache",
                              std::vector<double>(cell_count(n), 0.0));
  no_op.record_program_diagnostic("kept-diagnostic", Real(2.5));
  const auto histories_before_no_op = no_op.history_names();
  try {
    no_op.install_program(no_op_so);
    std::printf("FAIL no-op artifact installer was accepted\\n");
    ++fails;
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find("deliberately omitted its step candidate") ==
        std::string::npos) {
      std::printf("FAIL no-op installer diagnostic: %s\\n", e.what());
      ++fails;
    }
  }
  if (no_op.program_block_map() != program_blocks_before_no_op) {
    std::printf("FAIL no-op installer changed the accepted block map\\n");
    ++fails;
  }
  if (no_op.history_names() != histories_before_no_op) {
    std::printf("FAIL no-op installer leaked its candidate history ring\\n");
    ++fails;
  }
  if (no_op.program_cache_slots() != std::vector<CacheSlot>{CacheSlot{0}} ||
      !no_op.program_cache_valid(0) || no_op.program_cache_name(0) != "kept-cache" ||
      no_op.program_diagnostics() != std::map<std::string, Real>{{"kept-diagnostic", Real(2.5)}}) {
    std::printf("FAIL prelude-only installer did not restore cache/diagnostics\\n");
    ++fails;
  }
  try {
    no_op.mark_bound();
    no_op.step(dt);
  } catch (const std::exception& e) {
    std::printf("FAIL no-op installer did not restore the prior Program: %s\\n", e.what());
    ++fails;
  }

  // A v5 candidate without descriptor.dt_bound has no authored bound.  The loader must not probe
  // any auxiliary symbol or synthesize a fallback.
  NativeSystem incomplete_dt(cfg);
  install_runtime_authority(incomplete_dt, "test.program-loader/runtime-incomplete-dt@1");
  add_gas(incomplete_dt);
  incomplete_dt.install_program(so);
  try {
    incomplete_dt.install_program(incomplete_dt_so);
  } catch (const std::exception& e) {
    std::printf("FAIL v5 artifact without a dt callback was refused: %s\\n", e.what());
    ++fails;
  }

  // Program-owned field-boundary kernels are an artifact overlay, not durable System authoring.
  // Replacing artifact A (dynamic boundary export) with artifact B (no export) must therefore
  // restore the static baseline while retaining the exact configured backend route.
  {
    NativeSystem replacement(cfg);
    install_runtime_authority(replacement, "test.program-loader/runtime-replacement@1");
    add_gas(replacement);
    constexpr const char* slot = "program-boundary-field";
    constexpr const char* backend = "program-boundary-cartesian-cg";
    replacement.register_configured_field_solver_provider(
        "cartesian_cg", backend,
        PreparedProviderOptions{
            "pops.system.cartesian-cg-options@1",
            {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}});
    replacement.set_field_solver_plan(
        slot, "test:program-boundary-plan", "test:program-boundary-provider", "test:gas", "gas",
        "program-boundary-potential", {"test:gas/program-boundary-rhs"}, {"gas"},
        {"program-boundary-potential"}, {1.0}, backend);
    replacement.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                             "test:periodic-cartesian",
                                             "test:periodic-cartesian:v1");
    const std::vector<std::string> periodic_faces(static_cast<std::size_t>(2 * kTestDimension),
                                                  "periodic");
    const std::vector<double> zero_faces(static_cast<std::size_t>(2 * kTestDimension), 0.0);
    replacement.set_field_boundary_plan(slot, periodic_faces, zero_faces, zero_faces, zero_faces);
    const auto field_output =
        install_field_output(replacement, "test.program-boundary", "program-boundary-potential");
    replacement.register_elliptic_field("gas", "program-boundary-potential", {field_output}, 1);
    replacement.set_block_elliptic_field("gas", "program-boundary-potential",
                                         [](const NativeField&, NativeField&) {});
    replacement.set_state("gas", U0);

    try {
      replacement.install_program(dynamic_boundary_so);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dynamic-boundary install diagnostic: %s\n", e.what());
      throw;
    }
    if (replacement.history_names() != std::vector<std::string>{"artifact.history"}) {
      std::printf("FAIL artifact A did not materialize its qualified history\\n");
      ++fails;
    }
    const NativeField replacement_state = [&] {
      const auto state_view = replacement.block_state(0);
      return NativeField(*state_view.get());
    }();
    replacement.restore_program_cache(0, 1, 0, 0, 0.0, "artifact-A-cache",
                                      std::vector<double>(cell_count(n), 0.0));
    replacement.record_program_diagnostic("artifact-A-diagnostic", Real(1));
    try {
      const SolveReport report =
          consume_solve_outcome(replacement.solve_fields_from_state(slot, 0, replacement_state));
      if (!report.solved()) {
        std::printf("FAIL dynamic-boundary field solve returned %s\\n", report.status_name());
        ++fails;
      }
    } catch (const std::exception& e) {
      std::printf("FAIL dynamic-boundary artifact was not executable: %s\\n", e.what());
      ++fails;
    }
    // A later artifact that stages a valid overlay and then throws must publish neither its
    // boundary nor its Program/history/cache image. Artifact A remains the accepted owner and its
    // function pointers remain backed by the still-live accepted DSO.
    try {
      replacement.install_program(failing_boundary_so);
      std::printf("FAIL partially failing boundary artifact was accepted\\n");
      ++fails;
    } catch (const std::runtime_error& e) {
      if (std::string(e.what()).find("injected boundary publication failure") ==
          std::string::npos) {
        std::printf("FAIL partial boundary rollback diagnostic: %s\\n", e.what());
        ++fails;
      }
    }
    if (replacement.history_names() != std::vector<std::string>{"artifact.history"} ||
        replacement.program_cache_slots() != std::vector<CacheSlot>{CacheSlot{0}} ||
        !replacement.program_cache_valid(0) ||
        replacement.program_cache_name(0) != "artifact-A-cache" ||
        replacement.program_diagnostics() !=
            std::map<std::string, Real>{{"artifact-A-diagnostic", Real(1)}}) {
      std::printf("FAIL partial boundary artifact mutated accepted Program state\\n");
      ++fails;
    }
    try {
      const SolveReport report =
          consume_solve_outcome(replacement.solve_fields_from_state(slot, 0, replacement_state));
      if (!report.solved()) {
        std::printf("FAIL accepted boundary after rollback returned %s\\n", report.status_name());
        ++fails;
      }
    } catch (const std::exception& e) {
      std::printf("FAIL accepted boundary was lost after rollback: %s\\n", e.what());
      ++fails;
    }

    replacement.install_program(lifecycle_replacement_so);
    if (!replacement.history_names().empty() ||
        replacement.program_cache_slots() != std::vector<CacheSlot>{CacheSlot{0}} ||
        replacement.program_cache_valid(0) || !replacement.program_diagnostics().empty()) {
      std::printf("FAIL artifact B retained Program-owned state from artifact A\\n");
      ++fails;
    }
    try {
      const SolveReport report =
          consume_solve_outcome(replacement.solve_fields_from_state(slot, 0, replacement_state));
      if (!report.solved()) {
        std::printf("FAIL static replacement field solve returned %s\\n", report.status_name());
        ++fails;
      }
    } catch (const std::exception& e) {
      std::printf("FAIL static artifact inherited A's dynamic boundary: %s\\n", e.what());
      ++fails;
    }
  }
  NativeSystem sim(cfg);
  install_runtime_authority(sim, "test.program-loader/runtime-simulation@1");
  add_gas(sim);
  sim.set_potential(std::vector<double>(nn, 0.0));
  sim.set_state("gas", U0);
  sim.install_program(so);  // dlopen + ABI check + pops_install_program(this)
  const int step0 = sim.macro_step();
  sim.mark_bound();
  sim.step(dt);  // The exact-ranked System facade dispatches to the installed Program.
  const std::vector<double> Up = sim.get_state("gas");

  double err = 0, change = 0;
  for (std::size_t k = 0; k < Up.size(); ++k) {
    err = std::fmax(err, std::fabs(Up[k] - Uref[k]));
    change = std::fmax(change, std::fabs(Up[k] - U0[k]));
  }
  if (!(err < 1e-12)) {
    std::printf("FAIL parity: max|Up - Uref| = %.3e\n", err);
    ++fails;
  }
  if (sim.macro_step() != step0 + 1) {
    std::printf("FAIL macro_step not advanced (%d -> %d)\n", step0, sim.macro_step());
    ++fails;
  }
  if (!(change > 1e-9)) {
    std::printf("FAIL loaded program did not change the state (change = %.3e)\n", change);
    ++fails;
  }

  if (fails == 0)
    std::printf(
        "OK test_program_loader (problem.so Forward Euler via install_program == reference; "
        "max|d| = %.2e, change = %.2e)\n",
        err, change);
  return fails ? 1 : 0;
}

TEST(test_program_loader, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_program_loader, "test_program_loader"), 0);
}
