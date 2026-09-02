/// @file
/// @brief Synthetic source-built Program loader and AMR transaction artifact.
///
/// Its DSO body is deliberately authored in this C++ fixture to exercise loader ABI/hash/budget
/// authentication and hierarchy retry/rollback/publication.  It is not evidence for a user-authored
/// Python Program, its tableau, or temporal composition semantics.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "component_abi_test_helpers.hpp"
#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <string_view>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

std::atomic<bool> g_heap_measurement_enabled{false};
std::atomic<std::uint64_t> g_measured_heap_allocations{0};
void note_measured_heap_allocation() noexcept {
  if (g_heap_measurement_enabled.load(std::memory_order_relaxed))
    g_measured_heap_allocations.fetch_add(1, std::memory_order_relaxed);
}

void* measured_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_allocate_nothrow(std::size_t size) noexcept {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate_nothrow(std::size_t size, std::size_t alignment) noexcept {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

class HeapAllocationWindow final {
 public:
  HeapAllocationWindow() : before_(g_measured_heap_allocations.load(std::memory_order_relaxed)) {
    g_heap_measurement_enabled.store(true, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t close() noexcept {
    g_heap_measurement_enabled.store(false, std::memory_order_relaxed);
    return g_measured_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_ = 0;
};

}  // namespace

void* operator new(std::size_t size) {
  return measured_allocate(size);
}
void* operator new[](std::size_t size) {
  return measured_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

namespace {

constexpr int Dim = pops::kNativeDimension;
constexpr std::uint32_t kSyntheticLoaderRetryReason = 0x534C5452u;
constexpr const char* kBlock = "tracer";
constexpr const char* kStateRoute = "tests.synthetic-loader/state/tracer";
constexpr const char* kProviderConsumer = "tests.synthetic-loader/providers/tracer";
constexpr const char* kSyntheticLoaderProgramHash =
    "tests.synthetic-loader/program/loader-transaction-v1";

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

std::size_t marker_index_with_prefix(const std::vector<std::string>& markers,
                                     const std::string& prefix) {
  for (std::size_t index = 0; index < markers.size(); ++index)
    if (markers[index].starts_with(prefix))
      return index;
  return markers.size();
}

std::optional<std::string> marker_with_prefix(const std::vector<std::string>& markers,
                                              const std::string& prefix) {
  const auto found = std::find_if(markers.begin(), markers.end(), [&](const std::string& marker) {
    return marker.starts_with(prefix);
  });
  if (found == markers.end())
    return std::nullopt;
  return *found;
}

std::optional<std::string> marker_attribute(const std::string& marker, const std::string& key) {
  const std::string prefix = key + "=";
  const std::size_t begin = marker.find(prefix);
  if (begin == std::string::npos)
    return std::nullopt;
  const std::size_t value_begin = begin + prefix.size();
  const std::size_t end = marker.find(';', value_begin);
  const std::size_t value_size = end == std::string::npos ? std::string::npos : end - value_begin;
  return marker.substr(value_begin, value_size);
}

std::shared_ptr<const pops::component::PreparedExecutionContextV1> prepared_execution() {
  const PopsExecutionContextV1 execution = pops::component::test_support::host_execution_context();
  return std::make_shared<const pops::component::PreparedExecutionContextV1>(
      execution.execution_identity, execution.context_version, execution.memory_space,
      execution.backend_identity, execution.device_identity, execution.scalar_type,
      execution.storage_precision, execution.compute_precision, execution.accumulation_precision,
      execution.reduction_precision, execution.stream_handle, execution.stream_identity,
      execution.communicator_f_handle, execution.communicator_datatype_f_handle,
      execution.communicator_identity, execution.communicator_datatype_identity);
}

std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(shape[axis]);
  return count;
}

pops::AmrSystemConfig<Dim> config() {
  pops::AmrSystemConfig<Dim> result;
  const int width = Dim == 3 ? 12 : 24;
  result.level_count = 2;
  result.transition_ratios.resize(1);
  result.transition_buffers.resize(1);
  result.transition_lookaheads.resize(1);
  result.regrid_every = 0;
  result.explicit_bootstrap = true;
  result.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = width;
    result.lower[axis] = pops::Real(0);
    result.upper[axis] = pops::Real(1);
    result.periodicity[axis] = true;
    result.coarse_max_grid[axis] = width / 2;
    result.transition_ratios[0][axis] = 2;
    result.transition_buffers[0][axis] = 1;
    result.transition_lookaheads[0][axis] = 1;
  }
  return result;
}

std::vector<double> initial_state(const pops::Extent<Dim>& shape) {
  const std::size_t cells = cell_count(shape);
  std::vector<double> result(cells, 1.0);
  const double pi = std::acos(-1.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t quotient = cell;
    double wave = 0.08;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(shape[axis]));
      quotient /= static_cast<std::size_t>(shape[axis]);
      const double x = (static_cast<double>(coordinate) + 0.5) / shape[axis];
      const double envelope = std::sin(pi * x);
      wave *= envelope * envelope;
    }
    result[cell] += wave;
  }
  return result;
}

std::string loader_source(const std::string& marker_path = {}, const std::string& marker_tag = {},
                          bool malformed_candidate = false, bool prepare_throws = false,
                          bool observe_preparation_authority = false,
                          bool mutate_detached_preparation_state = false) {
  // clang-format off
  std::string result = R"CPP(
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system/native_package_capability.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#error "synthetic Program loader requires the shared runtime exception ABI"
#endif
namespace {
static constexpr char kProgramMarkerPath[] = "@@MARKER_PATH@@";
static constexpr char kProgramMarkerTag[] = "@@MARKER_TAG@@";
void program_marker(const char* event) {
  if (kProgramMarkerPath[0] == '\0')
    return;
  std::ofstream marker(kProgramMarkerPath, std::ios::app);
  marker << kProgramMarkerTag << ':' << event << '\n';
}
#if defined(__APPLE__) || defined(__linux__)
__attribute__((destructor)) void program_image_unloaded() { program_marker("dlclose"); }
#endif
}  // namespace
namespace pops_generated {
template <int Dim>
struct RelaxingAdvection {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  Law law{};
  pops::Real decay = pops::Real(0);
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.synthetic-loader.relaxing-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis) contract.scalar(law.velocity()[axis]);
    contract.scalar(decay);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis> POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis> POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State& state, const pops::ProviderValues<0>&) const {
    const pops::Real departure = state[0] - pops::Real(1);
    return State{-decay * (departure + departure * departure * departure)};
  }
  POPS_HD void source_jacobian(const State& state, const pops::ProviderValues<0>&,
                               pops::Real (&jacobian)[1][1]) const {
    const pops::Real departure = state[0] - pops::Real(1);
    jacobian[0][0] = -decay * (pops::Real(1) + pops::Real(3) * departure * departure);
  }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
using Model = RelaxingAdvection<pops::kNativeDimension>;
}
extern "C" const char* pops_native_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" int pops_native_system_package_abi_version() {
  return pops::runtime::system::kNativeSystemPackageAbiVersion;
}
extern "C" const char* pops_compiled_model_identity() {
  return "2222222222222222222222222222222222222222222222222222222222222222";
}
extern "C" const char* pops_compiled_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" int pops_compiled_nparams() { return 0; }
extern "C" const char* pops_compiled_param_names() { return ""; }
extern "C" void pops_register_provider_routes_amr(
    pops::AmrSystem<pops::kNativeDimension>* system) {
  if (system == nullptr)
    throw std::invalid_argument("AMR provider route installer received null exact runtime");
}
extern "C" void pops_install_native_amr(void* sys, const char* name, const char* limiter,
                                        const char* riemann, const char* recon, const char* time,
                                        double gamma, int substeps, int stride, const double*, int,
                                        double pos_floor, double weno_epsilon,
                                        bool wave_speed_cache, int newton_max_iters,
                                        double newton_rel_tol, double newton_abs_tol,
                                        double newton_fd_eps, double newton_damping,
                                        int newton_diagnostics) {
  pops::RealVector<pops::kNativeDimension> velocity{};
  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
    velocity[axis] = pops::Real(0.2) / pops::Real(axis + 1);
  pops_generated::Model model{
      pops::nd::ScalarAdvection<pops::kNativeDimension>::prepare(velocity), pops::Real(80)};
  auto* system = static_cast<pops::AmrSystem<pops::kNativeDimension>*>(sys);
  const pops::NewtonOptions newton = pops::newton_options_from_abi(
      newton_max_iters, newton_rel_tol, newton_abs_tol, newton_fd_eps, newton_damping);
  pops::PreparedNativeAmrPackage<pops::kNativeDimension> package;
  package.block = pops::prepare_compiled_amr_system_block<pops::kNativeDimension>(
      name, std::move(model), limiter, riemann, recon, time, gamma, substeps, stride, pos_floor,
      weno_epsilon, wave_speed_cache, "tests.synthetic-loader/providers/tracer", newton,
      newton_diagnostics != 0);
  system->install_prepared_native_amr_package(std::move(package));
}

namespace {
using ExecutionServices =
    pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>;

struct ProgramCandidateState final {
  std::shared_ptr<ExecutionServices> ctx_owner;
  std::function<void(double)> step;
  std::function<void(const pops::runtime::program::ProgramPreparationHostRef&)> prepare;
  std::array<pops::runtime::program::ProgramResourcePlanRecord, 11> resources{};
  std::string resource_digest;
  std::string resource_manifest;
};

constexpr std::array<std::uint64_t, 11> kScratchValueIds = {
    UINT64_C(1000), UINT64_C(3000), UINT64_C(3001), UINT64_C(3002),
    UINT64_C(3003), UINT64_C(3004), UINT64_C(3005), UINT64_C(3006),
    UINT64_C(3007), UINT64_C(3008), UINT64_C(3009)};
constexpr std::array<const char*, 11> kScratchIdentities = {
    "tests.synthetic-loader/scratch/state/1000",
    "tests.synthetic-loader/scratch/rhs/3000",
    "tests.synthetic-loader/scratch/rhs/3001",
    "tests.synthetic-loader/scratch/rhs/3002",
    "tests.synthetic-loader/scratch/rhs/3003",
    "tests.synthetic-loader/scratch/rhs/3004",
    "tests.synthetic-loader/scratch/rhs/3005",
    "tests.synthetic-loader/scratch/rhs/3006",
    "tests.synthetic-loader/scratch/rhs/3007",
    "tests.synthetic-loader/scratch/rhs/3008",
    "tests.synthetic-loader/scratch/rhs/3009"};

pops::runtime::program::ProgramAbiView program_view(const char* value) {
  return {value, static_cast<std::uint64_t>(std::strlen(value))};
}

void prepare_candidate_resource_plan(ProgramCandidateState& state) {
  using namespace pops::runtime::program;
  static constexpr char kSchema[] = "program-resource-plan:v1";
  static constexpr char kOwner[] = "tracer";
  static constexpr char kSpace[] = "cell";
  static constexpr char kClock[] = "clock.macro";
  static constexpr char kLifetime[] = "transient";
  static constexpr char kCentering[] = "cell";
  static constexpr char kOffPolicy[] = "none";
  static constexpr char kCommunication[] = "none";
  static constexpr char kTransfer[] = "none";
  static constexpr char kRestart[] = "none";
  static constexpr char kComponents[] = "[\"u\"]";
  static constexpr char kShape[] = "[]";

  ProgramInstallationTables tables;
  tables.resource_plan.reserve(kScratchValueIds.size());
  for (std::size_t slot = 0; slot < kScratchValueIds.size(); ++slot) {
    ProgramInstallationTables::ResourcePlan row;
    row.slot = static_cast<std::uint32_t>(slot);
    row.flags = kProgramResourceRuntimeSized;
    row.value_id = kScratchValueIds[slot];
    row.occurrence_path_id = UINT64_C(0x2000) + slot;
    row.level = -1;
    row.components = 1;
    row.ghosts = pops::Minmod::n_ghost;
    row.resource_type = ProgramResourcePlanType::runtime_sized;
    row.schema = kSchema;
    row.identity = kScratchIdentities[slot];
    row.occurrence_path = kScratchIdentities[slot];
    row.owner = kOwner;
    row.space = kSpace;
    row.clock = kClock;
    row.lifetime = kLifetime;
    row.centering = kCentering;
    row.off_policy = kOffPolicy;
    row.communication = kCommunication;
    row.transfer_provider = kTransfer;
    row.restart_provider = kRestart;
    row.component_names = kComponents;
    row.shape = kShape;
    tables.resource_plan.push_back(std::move(row));
  }
  const std::string payload = tables.canonical_resource_digest_payload(std::nullopt);
  state.resource_digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  for (auto& row : tables.resource_plan)
    row.plan_digest = state.resource_digest;
  state.resource_manifest =
      "{\"resource_plan\":" +
      tables.canonical_resource_manifest(std::nullopt, state.resource_digest) +
      ",\"resource_plan_digest\":\"" + state.resource_digest + "\"}";

  for (std::size_t slot = 0; slot < state.resources.size(); ++slot) {
    ProgramResourcePlanRecord& row = state.resources[slot];
    row = {};
    row.slot = static_cast<std::uint32_t>(slot);
    row.flags = kProgramResourceRuntimeSized;
    row.value_id = kScratchValueIds[slot];
    row.occurrence_path_id = UINT64_C(0x2000) + slot;
    row.level = -1;
    row.components = 1;
    row.ghosts = pops::Minmod::n_ghost;
    row.bytes = kProgramResourcePlanUnknownExtent;
    row.maximum_bytes = kProgramResourcePlanUnknownExtent;
    row.cells = kProgramResourcePlanUnknownExtent;
    row.itemsize = kProgramResourcePlanUnknownExtent;
    row.schema = program_view(kSchema);
    row.plan_digest = {state.resource_digest.data(),
                       static_cast<std::uint64_t>(state.resource_digest.size())};
    row.identity = program_view(kScratchIdentities[slot]);
    row.occurrence_path = program_view(kScratchIdentities[slot]);
    row.owner = program_view(kOwner);
    row.space = program_view(kSpace);
    row.clock = program_view(kClock);
    row.lifetime = program_view(kLifetime);
    row.centering = program_view(kCentering);
    row.off_policy = program_view(kOffPolicy);
    row.communication = program_view(kCommunication);
    row.transfer_provider = program_view(kTransfer);
    row.restart_provider = program_view(kRestart);
    row.component_names = program_view(kComponents);
    row.shape = program_view(kShape);
    row.resource_type = ProgramResourcePlanType::runtime_sized;
  }
}

struct ProgramStepRejectSentinel final : pops::runtime::program::ProgramStepRejectSignal {
  using ProgramStepRejectSignal::ProgramStepRejectSignal;
};
struct ProgramStepRejectPublishFailure final {};

void program_candidate_step(void* opaque, double dt) {
  auto* state = static_cast<ProgramCandidateState*>(opaque);
  try {
    state->step(dt);
  } catch (const pops::runtime::program::ProgramStepRejectSignal& signal) {
    if (!state->ctx_owner->adopt_step_attempt_rejection(signal.record))
      throw ProgramStepRejectPublishFailure{};
  }
}
void program_candidate_hierarchy_refresh(void* opaque) {
  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->refresh_accepted_hierarchy();
}
void program_candidate_history_remap(void* opaque, const void* descriptor) {
  if (descriptor == nullptr)
    throw std::invalid_argument("synthetic AMR candidate received null history remap");
  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->accept_history_remap(
      *static_cast<const pops::runtime::program::AmrProgramHistoryRemapDescriptor*>(descriptor));
}
void program_candidate_restart_preflight(void* opaque) {
  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->preflight_restart_regrid();
}
void program_candidate_restart_regrid(void* opaque) {
  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->restart_regrid();
}
void program_candidate_restart_resync(void* opaque) {
  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->resync_after_restart();
}
pops::runtime::program::AcceptedProgramExecutionServicesSnapshot*
program_candidate_accepted_snapshot(void* opaque) {
  return static_cast<ProgramCandidateState*>(opaque)->ctx_owner
      ->create_accepted_context_snapshot().release();
}
void program_candidate_destroy(void* opaque) noexcept {
  program_marker("destroy");
  delete static_cast<ProgramCandidateState*>(opaque);
}
void observe_preparation_authority(
    const pops::runtime::program::ProgramPreparationHostRef& preparation) {
  using namespace pops::runtime::program;
  // The marker path is an opaque host-owned test sink.  The DSO receives neither an AMR facade
  // nor a callback to one: every observed value below comes from the tagged preparation image and
  // its detached ProgramExecutionServices provider.
  const auto& image = require_program_execution_preparation_image(
      preparation, static_cast<std::uint32_t>(pops::kNativeDimension), ProgramRuntimeKind::amr);
  auto context_owner =
      make_program_execution_provider<pops::kNativeDimension>(preparation);
  auto& context = *context_owner;
  const auto topology = context.program_resource_topology();
  const auto& lane = context.prepared_execution_lane();
  int prototype_components = -1;
  int prototype_ghosts = -1;
  std::size_t prototype_local_fabs = 0;
  context.with_program_resource_level(0, [&] {
    const auto& prototype = context.state(0);
    prototype_components = prototype.ncomp();
    prototype_ghosts = prototype.ghosts()[0];
    prototype_local_fabs = prototype.local_size();
#if @@MUTATE_DETACHED_PREPARATION_STATE@@
    // This mutation is deliberately made through B's detached preparation provider.  The host
    // must discard it with the refused candidate; A's published POPSAND5 image remains immutable.
    auto& detached = context.state(0);
    detached.set_val(pops::Real(7));
#endif
  });
  const std::string marker =
      "prepare-authority:generation=" + std::to_string(image.generation()) +
      ";levels=" + std::to_string(topology.levels) + ";epoch=" +
      std::to_string(topology.epoch) + ";topology-generation=" +
      std::to_string(topology.generation) + ";blocks=" + std::to_string(context.n_blocks()) +
      ";lane=" + std::string(lane.identity()) + ";lane-size=" + std::to_string(lane.size()) +
      ";prototype-ncomp=" + std::to_string(prototype_components) +
      ";prototype-ghosts=" + std::to_string(prototype_ghosts) +
      ";prototype-local-fabs=" + std::to_string(prototype_local_fabs);
  program_marker(marker.c_str());
#if @@MUTATE_DETACHED_PREPARATION_STATE@@
  program_marker("prepare-authority:mutated-detached-prototype");
#endif
}
void program_install_error(pops::runtime::program::ProgramInstallDiagnostic* diagnostic,
                           const char* message) noexcept;
bool program_candidate_prepare(void* opaque,
                               const pops::runtime::program::ProgramHostDescriptor* host,
                               pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  if (opaque == nullptr || host == nullptr || diagnostic == nullptr ||
      !pops::runtime::program::valid_program_host_descriptor(*host))
    return false;
  auto* state = static_cast<ProgramCandidateState*>(opaque);
  if (state->ctx_owner || !state->prepare)
    return false;
  try {
#if @@OBSERVE_PREPARATION_AUTHORITY@@
    observe_preparation_authority(host->preparation);
#endif
#if @@PREPARE_THROWS@@
    throw std::runtime_error("injected AMR candidate preparation failure");
#endif
    state->prepare(host->preparation);
    state->prepare = {};
    return true;
  } catch (const std::exception& error) {
    program_install_error(diagnostic, error.what());
    return false;
  } catch (...) {
    program_install_error(diagnostic, "unknown AMR candidate preparation failure");
    return false;
  }
}
void program_install_error(pops::runtime::program::ProgramInstallDiagnostic* diagnostic,
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
constexpr std::uint64_t kCandidateCapabilities =
    pops::runtime::program::kProgramCapabilityHierarchy |
    pops::runtime::program::kProgramCapabilitySchedules |
    pops::runtime::program::kProgramCapabilityCellTemporal |
    pops::runtime::program::kProgramCapabilityPersistentValues |
    pops::runtime::program::kProgramCapabilityTransactions;
constexpr std::uint64_t kCandidateServices =
    pops::runtime::program::kProgramServiceState |
    pops::runtime::program::kProgramServiceFields |
    pops::runtime::program::kProgramServiceSpatial |
    pops::runtime::program::kProgramServiceHierarchy |
    pops::runtime::program::kProgramServiceHistory |
    pops::runtime::program::kProgramServiceClock |
    pops::runtime::program::kProgramServiceReduction |
    pops::runtime::program::kProgramServiceTransaction |
    pops::runtime::program::kProgramServicePersistentValues;
constexpr char kCandidateIdentity[] = "tests.synthetic-loader.v5";
constexpr char kCandidateArtifact[] = "@@CANDIDATE_ARTIFACT@@";
constexpr char kCandidateAbiKey[] = POPS_ABI_KEY_LITERAL;
constexpr const char* kCandidateRouteManifest = pops::kRouteRegistrySignature;
constexpr char kCandidateBoundaryManifest[] = "tests.synthetic-loader.boundary.v5";
constexpr char kCandidateCheckpointIdentity[] = "tests.synthetic-loader.checkpoint.v5";
constexpr char kCandidateBlockName[] = "tracer";
constexpr pops::runtime::program::ProgramBlockRecord kCandidateBlocks[] = {
    {{kCandidateBlockName, sizeof(kCandidateBlockName) - 1}}};
constexpr pops::runtime::program::ProgramFluxBudgetRecord kCandidateFluxBudgets[] = {
    {UINT64_C(10), UINT64_C(1), UINT64_C(0), UINT64_C(0)}};
constexpr char kCandidateFaceFluxOwner[] = "tracer";
constexpr char kCandidateFaceFluxClock[] = "clock.macro";
constexpr std::uint32_t kCandidatePreparedDefaultFluxProvider = 1;

template <std::size_t Size>
constexpr pops::runtime::program::ProgramAbiView candidate_view(const char (&value)[Size]) {
  return {value, static_cast<std::uint64_t>(Size - 1)};
}

constexpr pops::runtime::program::ProgramFluxBasisOccurrenceRecord candidate_flux_basis(
    std::uint32_t basis_slot, std::uint32_t expression_slot,
    pops::runtime::program::ProgramAbiView identity,
    pops::runtime::program::ProgramAbiView occurrence_path) {
  using namespace pops::runtime::program;
  return {sizeof(ProgramFluxBasisOccurrenceRecord),
          kProgramFluxBasisOccurrenceSchemaVersion,
          basis_slot,
          expression_slot,
          0,
          -1,
          static_cast<std::int32_t>(3000 + basis_slot),
          kCandidatePreparedDefaultFluxProvider,
          0,
          1,
          identity,
          occurrence_path,
          candidate_view(kCandidateFaceFluxOwner),
          candidate_view(kCandidateFaceFluxClock)};
}

constexpr pops::runtime::program::ProgramFaceFluxStageRecord candidate_flux_term(
    std::uint32_t slot, std::uint32_t basis_slot, std::uint32_t expression_slot,
    std::int64_t denominator, pops::runtime::program::ProgramAbiView identity,
    pops::runtime::program::ProgramAbiView occurrence_path) {
  using namespace pops::runtime::program;
  return {sizeof(ProgramFaceFluxStageRecord),
          kProgramFaceFluxStageSchemaVersion,
          slot,
          basis_slot,
          expression_slot,
          1,
          1,
          denominator,
          identity,
          occurrence_path,
          candidate_view(kCandidateFaceFluxOwner),
          candidate_view(kCandidateFaceFluxClock)};
}

constexpr pops::runtime::program::ProgramFluxBasisOccurrenceRecord kCandidateFluxBases[] = {
    candidate_flux_basis(0, 1, candidate_view("synthetic.default-flux.basis.0"),
                        candidate_view("synthetic.default-flux.loop.basis.0")),
    candidate_flux_basis(1, 2, candidate_view("synthetic.default-flux.basis.1"),
                        candidate_view("synthetic.default-flux.loop.basis.1")),
    candidate_flux_basis(2, 3, candidate_view("synthetic.default-flux.basis.2"),
                        candidate_view("synthetic.default-flux.loop.basis.2")),
    candidate_flux_basis(3, 4, candidate_view("synthetic.default-flux.basis.3"),
                        candidate_view("synthetic.default-flux.loop.basis.3")),
    candidate_flux_basis(4, 5, candidate_view("synthetic.default-flux.basis.4"),
                        candidate_view("synthetic.default-flux.loop.basis.4")),
    candidate_flux_basis(5, 6, candidate_view("synthetic.default-flux.basis.5"),
                        candidate_view("synthetic.default-flux.loop.basis.5")),
    candidate_flux_basis(6, 7, candidate_view("synthetic.default-flux.basis.6"),
                        candidate_view("synthetic.default-flux.loop.basis.6")),
    candidate_flux_basis(7, 8, candidate_view("synthetic.default-flux.basis.7"),
                        candidate_view("synthetic.default-flux.loop.basis.7")),
    candidate_flux_basis(8, 9, candidate_view("synthetic.default-flux.basis.8"),
                        candidate_view("synthetic.default-flux.loop.basis.8")),
    candidate_flux_basis(9, 10, candidate_view("synthetic.default-flux.basis.9"),
                        candidate_view("synthetic.default-flux.loop.basis.9"))};
constexpr pops::runtime::program::ProgramFaceFluxStageRecord kCandidateFluxTerms[] = {
    candidate_flux_term(0, 0, 0, 2, candidate_view("synthetic.default-flux.stage.0"),
                       candidate_view("synthetic.default-flux.loop.basis.0")),
    candidate_flux_term(1, 1, 0, 4, candidate_view("synthetic.default-flux.stage.1"),
                       candidate_view("synthetic.default-flux.loop.basis.1")),
    candidate_flux_term(2, 2, 0, 8, candidate_view("synthetic.default-flux.stage.2"),
                       candidate_view("synthetic.default-flux.loop.basis.2")),
    candidate_flux_term(3, 3, 0, 16, candidate_view("synthetic.default-flux.stage.3"),
                       candidate_view("synthetic.default-flux.loop.basis.3")),
    candidate_flux_term(4, 4, 0, 32, candidate_view("synthetic.default-flux.stage.4"),
                       candidate_view("synthetic.default-flux.loop.basis.4")),
    candidate_flux_term(5, 5, 0, 64, candidate_view("synthetic.default-flux.stage.5"),
                       candidate_view("synthetic.default-flux.loop.basis.5")),
    candidate_flux_term(6, 6, 0, 128, candidate_view("synthetic.default-flux.stage.6"),
                       candidate_view("synthetic.default-flux.loop.basis.6")),
    candidate_flux_term(7, 7, 0, 256, candidate_view("synthetic.default-flux.stage.7"),
                       candidate_view("synthetic.default-flux.loop.basis.7")),
    candidate_flux_term(8, 8, 0, 512, candidate_view("synthetic.default-flux.stage.8"),
                       candidate_view("synthetic.default-flux.loop.basis.8")),
    candidate_flux_term(9, 9, 0, 512, candidate_view("synthetic.default-flux.stage.9"),
                       candidate_view("synthetic.default-flux.loop.basis.9"))};
constexpr pops::runtime::program::ProgramAbiTable kCandidateBlockTable{
    kCandidateBlocks, 1, sizeof(kCandidateBlocks[0])};
constexpr pops::runtime::program::ProgramAbiTable kCandidateFluxTable{
    kCandidateFluxBudgets, 1, sizeof(kCandidateFluxBudgets[0])};
constexpr pops::runtime::program::ProgramAbiTable kCandidateFluxBasisTable{
    kCandidateFluxBases, 10, sizeof(kCandidateFluxBases[0])};
constexpr pops::runtime::program::ProgramAbiTable kCandidateFaceFluxStageTable{
    kCandidateFluxTerms, 10, sizeof(kCandidateFluxTerms[0])};
}  // namespace

extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor* host,
    pops::runtime::program::ProgramCandidateDescriptor* candidate,
    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {
  using namespace pops::runtime::program;
  if (!host || !candidate || !diagnostic || !valid_program_host_descriptor(*host) ||
      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||
      host->runtime_kind != ProgramRuntimeKind::amr ||
      host->execution_lane != ProgramExecutionLane::host || !host->services.state_store) {
    program_install_error(diagnostic, "invalid synthetic AMR Program host");
    return false;
  }
  *candidate = {};
  try {
  auto state = std::make_unique<ProgramCandidateState>();
  prepare_candidate_resource_plan(*state);
  state->prepare = [state_ptr = state.get()](
                       const pops::runtime::program::ProgramPreparationHostRef& preparation) {
  state_ptr->ctx_owner = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(
      preparation);
  auto& context = *state_ptr->ctx_owner;
  // The semantic value identities remain 1000 and 3000..3009 above, while the bind-sealed
  // runtime indices below are the compact dense ProgramResourcePlan slots 0..10.
  context.prepare_state_scratch(0, 0, 0);
  for (std::size_t slot = 1; slot <= 10; ++slot)
    context.prepare_rhs_scratch(slot, 0, 0);
  auto inject_retry = std::make_shared<bool>(true);
  context.configure_primary_clock("clock.macro");
  state_ptr->step = [context_owner = state_ptr->ctx_owner, inject_retry](double macro_dt) {
    auto& context = *context_owner;
    context.advance_hierarchy(macro_dt, [context_owner, inject_retry](double level_dt) {
          auto& context = *context_owner;
          context.set_stage_time(0, 1);
          auto& accepted = context.state(0);
          auto& candidate = context.scratch_state(0, 0, accepted);
          auto& explicit_rate = context.rhs_scratch(1, 0, accepted);
          // Exercise the grouped static-table route before the injected rejection.  The request
          // materializes basis 0 and its authenticated face term into the resident carrier; the
          // rejected attempt below must restore that carrier before the retry reuses it.
          context.rhs_group(3010, {{0, &accepted, &explicit_rate, 3000, 1}});
          if (*inject_retry) {
            *inject_retry = false;
            pops::runtime::program::ProgramStepRejectRecord record{};
            if (!context.publish_step_attempt_rejection(
                    pops::SolveStatus::kIterationLimit,
                    pops::runtime::program::StepAttemptDisposition::kRetry,
                    UINT32_C(0x534C5452), "rhs-group-static-flux",
                    "injected-after-rhs-group-resident-write", record))
              throw ProgramStepRejectPublishFailure{};
            throw ProgramStepRejectSentinel{record};
          }
          context.lincomb(candidate, pops::Real(1), accepted, pops::Real(0), accepted);
          // Materialize ten independent, authenticated default-flux bases. The dyadic weights
          // sum exactly to one, so this decimal-boundary capacity witness preserves the fixture's
          // physical update while forcing identities 1 through 10 into the live expression.
          for (int basis = 0; basis < 10; ++basis) {
            auto& rate = basis == 0 ? explicit_rate
                                    : context.rhs_scratch(static_cast<std::size_t>(basis + 1), 0,
                                                          accepted);
            if (basis != 0)
              context.neg_div_flux_default_into(0, accepted, rate, 3000 + basis);
            const int exponent = basis == 9 ? 9 : basis + 1;
            context.axpy(candidate, pops::Real(level_dt / static_cast<double>(1 << exponent)), rate);
          }
          pops::SolveOutcome implicit = context.solve_source_default(
              0, candidate, pops::Real(level_dt), pops::NewtonOptions{});
          const pops::SolveReport solved = implicit.consume(pops::SolveConsumption::kAccept);
          if (!solved.solved())
            throw std::runtime_error("source-built AMR implicit source did not converge");
          context.commit_many({{&accepted, &candidate}});
        });
  };
  };
  ProgramCandidateDescriptor descriptor{};
  descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));
  descriptor.abi_version = kProgramInstallAbiVersion;
  descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  descriptor.runtime_kind = ProgramRuntimeKind::amr;
  descriptor.provided_capability_bits = kCandidateCapabilities;
  descriptor.required_capability_bits = kCandidateCapabilities;
  descriptor.required_service_bits = kCandidateServices;
#if @@MALFORMED_CANDIDATE@@
  descriptor.required_service_bits |= UINT64_C(1) << 63;
#endif
  descriptor.program_name = {kCandidateIdentity, sizeof(kCandidateIdentity) - 1};
  descriptor.artifact_identity = {kCandidateArtifact, sizeof(kCandidateArtifact) - 1};
  descriptor.abi_key = {kCandidateAbiKey, sizeof(kCandidateAbiKey) - 1};
  descriptor.route_manifest = {kCandidateRouteManifest,
                               static_cast<std::uint64_t>(std::strlen(kCandidateRouteManifest))};
  descriptor.boundary_manifest = {kCandidateBoundaryManifest, sizeof(kCandidateBoundaryManifest) - 1};
  descriptor.persistent_resource_manifest = {
      state->resource_manifest.data(), static_cast<std::uint64_t>(state->resource_manifest.size())};
  descriptor.checkpoint_identity = {kCandidateCheckpointIdentity,
                                    sizeof(kCandidateCheckpointIdentity) - 1};
  descriptor.blocks = kCandidateBlockTable;
  descriptor.flux_budgets = kCandidateFluxTable;
  descriptor.flux_basis_occurrences = kCandidateFluxBasisTable;
  descriptor.face_flux_stages = kCandidateFaceFluxStageTable;
  descriptor.resource_plan = {state->resources.data(), state->resources.size(),
                              sizeof(ProgramResourcePlanRecord)};
  descriptor.maximum_bytes = kProgramResourcePlanUnknownExtent;
  descriptor.context = state.get();
  descriptor.prepare = &program_candidate_prepare;
  descriptor.step = &program_candidate_step;
  descriptor.hierarchy_refresh = &program_candidate_hierarchy_refresh;
  descriptor.history_remap_accepted = &program_candidate_history_remap;
  descriptor.restart_regrid_preflight = &program_candidate_restart_preflight;
  descriptor.restart_regrid = &program_candidate_restart_regrid;
  descriptor.restart_resync = &program_candidate_restart_resync;
  descriptor.create_accepted_snapshot = &program_candidate_accepted_snapshot;
  descriptor.destroy = &program_candidate_destroy;
  // The malformed fixture deliberately returns its descriptor to the host so the shared preparer
  // owns (and must destroy) the rejected candidate before closing the image.
  if (!valid_program_candidate_descriptor(descriptor) &&
      (descriptor.required_service_bits & (UINT64_C(1) << 63)) == 0) {
    program_install_error(diagnostic, "invalid synthetic AMR Program candidate");
    return false;
  }
  *candidate = descriptor;
  (void)state.release();
  return true;
  } catch (...) {
    program_install_error(diagnostic, "synthetic AMR Program candidate construction failed");
    return false;
  }
}
static_assert(std::is_same_v<decltype(&pops_install_program),
                             pops::runtime::program::ProgramInstallFn>);
)CPP";
  const auto replace = [&](const std::string& from, const std::string& to) {
    std::size_t position = result.find(from);
    if (position == std::string::npos)
      throw std::logic_error("synthetic AMR loader source placeholder is missing");
    do {
      result.replace(position, from.size(), to);
      position = result.find(from, position + to.size());
    } while (position != std::string::npos);
  };
  replace("@@MARKER_PATH@@", marker_path);
  replace("@@MARKER_TAG@@", marker_tag);
  replace("@@MALFORMED_CANDIDATE@@", malformed_candidate ? "1" : "0");
  replace("@@PREPARE_THROWS@@", prepare_throws ? "1" : "0");
  replace("@@OBSERVE_PREPARATION_AUTHORITY@@", observe_preparation_authority ? "1" : "0");
  replace("@@MUTATE_DETACHED_PREPARATION_STATE@@",
          mutate_detached_preparation_state ? "1" : "0");
  const std::string artifact_identity = marker_tag.empty()
                                            ? kSyntheticLoaderProgramHash
                                            : std::string(kSyntheticLoaderProgramHash) + "/" + marker_tag;
  replace("@@CANDIDATE_ARTIFACT@@", artifact_identity);
  return result;
  // clang-format on
}

void configure_unbound_system(pops::AmrSystem<Dim>& system, const std::string& shared_object,
                              const std::vector<double>& state) {
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively("test.synthetic-loader.package"));
  auto execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
      prepared_execution()->for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
  system.install_block_state_route(kBlock, kStateRoute);
  const pops::dynlib::AuthenticatedNativeFile authenticated(shared_object);
  system.add_native_block(
      kBlock, shared_object, "2222222222222222222222222222222222222222222222222222222222222222",
      authenticated.binary_identity(), "minmod", "rusanov", "conservative", "explicit", 1.4, 1);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.bind_bootstrap_subject(kStateRoute, kBlock, "bound_level_zero");
  system.stage_bootstrap_array(kStateRoute, kBlock, "cell", "cell", 1, system.spatial_shape(),
                               state);
  pops::Extent<Dim> transfer_ghosts{};
  pops::Extent<Dim> refinement_ratio{};
  for (int axis = 0; axis < Dim; ++axis) {
    transfer_ghosts[axis] = 1;
    refinement_ratio[axis] = 2;
  }
  system.register_bootstrap_transfer_route(
      "tests.synthetic-loader/bootstrap/prolongation", {kStateRoute},
      "tests.synthetic-loader/bootstrap/provider", "cell", "cell", "conservative", "dense",
      "prolongation", "conservative_linear", 2, transfer_ghosts, refinement_ratio);
  pops::test::install_prepared_threshold_union(system, {{kBlock, "u", 1.03}},
                                               "tests.synthetic-loader.tagging@1");
}

void bind_refined_system(pops::AmrSystem<Dim>& system) {
  // Match the public bind lifecycle: explicit bootstrap remains inside the assembling transaction,
  // and the final bound transition rechecks the Program checkpoint-capacity seal afterward.
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(kStateRoute, "initialize_level_zero",
                                            "bound_level_zero", 0);
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::runtime_error("synthetic loader fixture did not create its refined level");
  }
  (void)system.materialize_bootstrap_action(kStateRoute, "prolong_from_parent",
                                            "conservative_linear", 1);
  system.commit_bootstrap_level();
  system.mark_bound();
}

void build_refined_system(pops::AmrSystem<Dim>& system, const std::string& shared_object,
                          const std::vector<double>& state) {
  configure_unbound_system(system, shared_object, state);
  // The v5 loader authenticates the copied candidate tables before it prepares the artifact and
  // materializes the hierarchy.
  system.install_program(shared_object);
  bind_refined_system(system);
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::fabs(left[index] - right[index]));
  return result;
}

bool byte_exact_equal(const std::vector<double>& left, const std::vector<double>& right) {
  return left.size() == right.size() &&
         (left.empty() ||
          std::memcmp(left.data(), right.data(), left.size() * sizeof(double)) == 0);
}

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

double max_departure_from_equilibrium(const std::vector<double>& values) {
  double result = 0.0;
  for (const double value : values)
    result = std::max(result, std::fabs(value - 1.0));
  return result;
}

std::vector<std::size_t> interior_level_indices(pops::AmrSystem<Dim>& system, int level) {
  const pops::Box<Dim> domain = system.prepared_amr_level_geometry(level).domain();
  auto state_view = system.prepared_amr_block_state(0, level);
  if (!state_view)
    throw std::logic_error("AMR accepted block view is invalid");
  const auto& boxes = state_view->layout().boxes();
  std::vector<std::size_t> indices;
  for (const pops::Box<Dim>& patch : boxes) {
    // The fixture authenticates loader publication and rollback, not monotonic transport through
    // the coarse/fine halo band.  Sample only cells beyond that interface influence.
    const pops::Box<Dim> interior = patch.grow(-4);
    if (interior.empty())
      continue;
    indices.reserve(indices.size() + static_cast<std::size_t>(interior.numPts()));
    for (std::int64_t ordinal = 0; ordinal < interior.numPts(); ++ordinal) {
      std::int64_t remaining = ordinal;
      pops::Index<Dim> cell{};
      for (int axis = 0; axis < Dim; ++axis) {
        cell[axis] = interior.lo[axis] + remaining % interior.length(axis);
        remaining /= interior.length(axis);
      }
      std::size_t linear = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        linear += static_cast<std::size_t>(cell[axis] - domain.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(domain.length(axis));
      }
      indices.push_back(linear);
    }
  }
  return indices;
}

std::vector<double> select_indices(const std::vector<double>& values,
                                   const std::vector<std::size_t>& indices) {
  std::vector<double> selected;
  selected.reserve(indices.size());
  for (const std::size_t index : indices)
    selected.push_back(values.at(index));
  return selected;
}

}  // namespace

void verify_v5_candidate_ownership();

TEST(test_amr_synthetic_program_loader_transaction,
     SourceBuiltArtifactLoadsBudgetRollsBackAndRetries) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_synthetic_loader_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source_path = stem + ".cpp";
  const std::string shared_object = stem + ".so";
  {
    std::ofstream source(source_path);
    source << loader_source();
  }
  const auto package = pops::test::native_dso::compile_shared(
      source_path, shared_object, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_amr_synthetic_program_loader_transaction",
                                                   package);
    FAIL() << "synthetic source-built AMR loader transaction artifact did not compile";
  }

  const auto system_config = config();
  const std::vector<double> initial = initial_state(system_config.shape);
  constexpr double dt = 3.0e-2;

  pops::AmrSystem<Dim> continuous(system_config);
  build_refined_system(continuous, shared_object, initial);
  ASSERT_EQ(continuous.n_levels(), 2);
  ASSERT_GT(continuous.n_patches(), 0);
  EXPECT_EQ(continuous.installed_program_hash(), kSyntheticLoaderProgramHash);
  const auto budget = continuous.prepared_amr_program_flux_expression_budget();
  EXPECT_EQ(budget.program_hash, kSyntheticLoaderProgramHash);
  ASSERT_EQ(budget.blocks.size(), 1u);
  ASSERT_EQ(budget.program_block_map.canonical_indices.size(), 1u);
  EXPECT_EQ(budget.program_block_map.canonical_indices[0], 0u);
  EXPECT_EQ(budget.blocks[0].rhs_basis_bound, 10u);
  EXPECT_EQ(budget.blocks[0].coefficient_term_bound, 1u);

  const std::vector<double> coarse_before = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_before = continuous.block_level_state_global(kBlock, 1);
  const std::vector<std::size_t> fine_interior_indices = interior_level_indices(continuous, 1);
  ASSERT_FALSE(fine_interior_indices.empty());
  const std::vector<double> fine_interior_before =
      select_indices(fine_before, fine_interior_indices);
  EXPECT_GT(max_departure_from_equilibrium(fine_interior_before), 0.0);
  EXPECT_LT(max_departure_from_equilibrium(fine_interior_before), 0.25);

  // This is the sealed A image before `rhs_group` materializes basis 0 in the rejected
  // candidate.  It is deliberately captured before the heap window: checkpoint inspection is
  // cold/read-only and must not be attributed to the bind-primed hot route below.
  const std::vector<std::uint8_t> accepted_before_fault = continuous.program_accepted_state();

  bool retry_rejected = false;
  bool retry_fields_reported = false;
  const pops::AllocationEventStats allocation_before_fault = pops::allocation_event_stats();
  HeapAllocationWindow fault_heap;
  try {
    continuous.step(dt);
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    retry_rejected = true;
    retry_fields_reported =
        rejected.status() == pops::SolveStatus::kIterationLimit &&
        rejected.disposition() == pops::runtime::program::StepAttemptDisposition::kRetry &&
        rejected.reason_code() == UINT32_C(0x534C5452) &&
        rejected.phase() == "rhs-group-static-flux" &&
        rejected.detail() == "injected-after-rhs-group-resident-write";
  }
  const std::uint64_t fault_heap_allocations = fault_heap.close();
  EXPECT_TRUE(retry_rejected) << "the injected rhs_group retry was not surfaced";
  EXPECT_TRUE(retry_fields_reported);
  // Fault construction and cross-DSO exception propagation are deliberately outside the
  // successful accepted-step allocation contract.  Keep this number in the native result so a
  // regression remains visible, but qualify the retry/repeat windows below independently: they
  // are the paths which must reuse the bind-sealed POPSAND5 staging image without allocation.
  ::testing::Test::RecordProperty("fault_heap_allocations", std::to_string(fault_heap_allocations));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_fault);
  EXPECT_EQ(continuous.macro_step(), 0);
  EXPECT_DOUBLE_EQ(continuous.time(), 0.0);
  EXPECT_TRUE(byte_exact_equal(continuous.block_level_state_global(kBlock, 0), coarse_before));
  EXPECT_TRUE(byte_exact_equal(continuous.block_level_state_global(kBlock, 1), fine_before));
  EXPECT_EQ(continuous.program_accepted_state(), accepted_before_fault)
      << "the rejected rhs_group attempt leaked its resident flux/ledger image into A";

  const pops::AllocationEventStats allocation_before_retry = pops::allocation_event_stats();
  HeapAllocationWindow retry_heap;
  continuous.step(dt);
  const std::uint64_t retry_heap_allocations = retry_heap.close();
  EXPECT_EQ(retry_heap_allocations, 0u);
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_retry);
  const std::vector<double> coarse_first = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_first = continuous.block_level_state_global(kBlock, 1);
  const std::vector<double> fine_interior_first = select_indices(fine_first, fine_interior_indices);
  ASSERT_TRUE(all_finite(coarse_first));
  ASSERT_TRUE(all_finite(fine_first));
  EXPECT_GT(max_difference(coarse_first, coarse_before), 0.0);
  EXPECT_GT(max_difference(fine_first, fine_before), 0.0);
  EXPECT_LT(max_departure_from_equilibrium(coarse_first),
            max_departure_from_equilibrium(coarse_before));
  EXPECT_LT(max_departure_from_equilibrium(fine_interior_first),
            max_departure_from_equilibrium(fine_interior_before));
  EXPECT_EQ(continuous.macro_step(), 1);
  EXPECT_DOUBLE_EQ(continuous.time(), dt);
  const std::vector<std::uint8_t> accepted_bytes = continuous.program_accepted_state();
  const auto accepted =
      pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(accepted_bytes);
  EXPECT_TRUE(std::any_of(accepted.accepted_face_flux.begin(), accepted.accepted_face_flux.end(),
                          [](const auto& fragments) { return !fragments.empty(); }));
  std::set<std::string> materialized_stages;
  for (const auto& fragments : accepted.accepted_face_flux)
    for (const auto& fragment : fragments)
      materialized_stages.insert(fragment.key.stage);
  EXPECT_EQ(materialized_stages.size(), 10u);
  EXPECT_TRUE(std::any_of(materialized_stages.begin(), materialized_stages.end(),
                          [](const std::string& stage) {
                            return stage.find("/basis/9/expression/0/") != std::string::npos;
                          }));
  EXPECT_TRUE(std::any_of(materialized_stages.begin(), materialized_stages.end(),
                          [](const std::string& stage) {
                            return stage.find("/basis/0/expression/0/") != std::string::npos;
                          }))
      << "the retry did not publish the static basis produced by rhs_group";
  EXPECT_LE(accepted_bytes.size(), continuous.checkpoint_program_state_capacity().first);

  const pops::AllocationEventStats allocation_before_repeat = pops::allocation_event_stats();
  HeapAllocationWindow repeat_heap;
  continuous.step(dt);
  const std::uint64_t repeat_heap_allocations = repeat_heap.close();
  EXPECT_EQ(repeat_heap_allocations, 0u);
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_repeat);
  const pops::AllocationEventStats allocation_before_second_repeat = pops::allocation_event_stats();
  HeapAllocationWindow second_repeat_heap;
  continuous.step(dt);
  const std::uint64_t second_repeat_heap_allocations = second_repeat_heap.close();
  EXPECT_EQ(second_repeat_heap_allocations, 0u);
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_second_repeat);
  const std::vector<double> coarse_accepted = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_accepted = continuous.block_level_state_global(kBlock, 1);
  const std::vector<double> fine_interior_accepted =
      select_indices(fine_accepted, fine_interior_indices);
  ASSERT_TRUE(all_finite(coarse_accepted));
  ASSERT_TRUE(all_finite(fine_accepted));
  EXPECT_LT(max_departure_from_equilibrium(coarse_accepted),
            max_departure_from_equilibrium(coarse_first));
  EXPECT_LT(max_departure_from_equilibrium(fine_interior_accepted),
            max_departure_from_equilibrium(fine_interior_first));
  EXPECT_EQ(continuous.macro_step(), 3);
  EXPECT_DOUBLE_EQ(continuous.time(), 3.0 * dt);

  // The accepted POPSAND5 image now has a nonempty logical face-flux vector while its
  // bind-sealed resident slots remain the larger capacity envelope.  A forward regrid must
  // detach and cold-prime that exact image without treating the logical fragment count as the
  // slot-pool shape.  This is deliberately after the warm repeat windows: regrid preparation is
  // cold, whereas the assertion protects the copy/reprime contract used to construct B.
  const std::uint64_t topology_before_forward_copy = continuous.checkpoint_topology_epoch();
  ASSERT_NO_THROW(continuous.execute_prepared_tagging(0));
  ASSERT_TRUE(continuous.regrid_from_prepared_tagging(0));
  EXPECT_GT(continuous.checkpoint_topology_epoch(), topology_before_forward_copy);
  const auto regridded_checkpoint = continuous.program_accepted_state();
  EXPECT_FALSE(regridded_checkpoint.empty());
  EXPECT_LE(regridded_checkpoint.size(), continuous.checkpoint_program_state_capacity().first);
  verify_v5_candidate_ownership();
}

void verify_v5_candidate_ownership() {
  const bool marker_child = std::getenv("POPS_AMR_PROGRAM_MARKER_CHILD") != nullptr;
  if (!marker_child) {
    const std::string marker_path = std::string(POPS_TEST_TMPDIR) + "/amr_program_v5_markers_" +
                                    std::to_string(static_cast<long>(std::clock())) + ".log";
    ::setenv("POPS_AMR_PROGRAM_MARKER_CHILD", "1", 1);
    ::setenv("POPS_AMR_PROGRAM_MARKER_PATH", marker_path.c_str(), 1);
    const std::string test_directory = POPS_TEST_TMPDIR;
    const std::size_t separator = test_directory.find_last_of('/');
    const std::string executable =
        test_directory.substr(0, separator) + "/bin/test_amr_synthetic_program_loader_transaction";
    const std::string command = std::string("\"") + executable +
                                "\" --gtest_filter=test_amr_synthetic_program_loader_transaction."
                                "SourceBuiltArtifactLoadsBudgetRollsBackAndRetries";
    const int child_status = std::system(command.c_str());
    ::unsetenv("POPS_AMR_PROGRAM_MARKER_CHILD");
    ::unsetenv("POPS_AMR_PROGRAM_MARKER_PATH");
    EXPECT_EQ(child_status, 0) << "AMR Program lifecycle child failed";
    const auto markers = read_program_markers(marker_path);
    const auto require_destroy_before_close = [&](const std::string& tag) {
      const std::size_t destroy = marker_index(markers, tag + ":destroy");
      const std::size_t close = marker_index(markers, tag + ":dlclose");
      EXPECT_NE(destroy, markers.size()) << tag;
      EXPECT_NE(close, markers.size()) << tag;
      EXPECT_LT(destroy, close) << tag;
    };
    require_destroy_before_close("first-failure");
    require_destroy_before_close("first-prepare-failure");
    require_destroy_before_close("failed-replacement");
    require_destroy_before_close("accepted-a");
    require_destroy_before_close("replacement-b");
    return;
  }

  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_program_v5_lifecycle_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  const char* configured_marker_path = std::getenv("POPS_AMR_PROGRAM_MARKER_PATH");
  const std::string marker_path =
      configured_marker_path ? configured_marker_path : stem + ".markers";
  const auto compile_fixture = [&](const std::string& tag, bool malformed, bool prepare_throws,
                                   bool observe_authority = false,
                                   bool mutate_detached_state = false) {
    const std::string source_path = stem + "_" + tag + ".cpp";
    const std::string shared_object = stem + "_" + tag + ".so";
    {
      std::ofstream source(source_path);
      source << loader_source(marker_path, tag, malformed, prepare_throws, observe_authority,
                              mutate_detached_state);
    }
    const auto package = pops::test::native_dso::compile_shared(
        source_path, shared_object, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
    if (!package.ok)
      pops::test::native_dso::report_compile_failure(
          "test_amr_synthetic_program_loader_transaction lifecycle", package);
    EXPECT_TRUE(package.ok);
    return shared_object;
  };

  const std::string failed_first = compile_fixture("first-failure", true, false);
  const std::string failed_first_prepare = compile_fixture("first-prepare-failure", false, true);
  const std::string accepted_a = compile_fixture("accepted-a", false, false);
  const std::string failed_replacement =
      compile_fixture("failed-replacement", false, true, true, true);
  const std::string replacement_b = compile_fixture("replacement-b", false, false, true);

  const auto system_config = config();
  const std::vector<double> initial = initial_state(system_config.shape);
  {
    pops::AmrSystem<Dim> system(system_config);
    configure_unbound_system(system, accepted_a, initial);
    const auto cold_field_manifest = system.field_provider_checkpoint_manifest();
    const std::string cold_auxiliary_contract = system.auxiliary_registry_contract();

    EXPECT_THROW(system.install_program(failed_first), std::runtime_error);
    // Inspection failure is rejected before the installer can materialize a live AMR graph.  In
    // particular, a failed first candidate has no topology publication to clean up later.
    EXPECT_EQ(system.checkpoint_topology_epoch(), 0u);
    EXPECT_TRUE(system.installed_program_hash().empty());
    EXPECT_EQ(system.field_provider_checkpoint_manifest(), cold_field_manifest);
    EXPECT_EQ(system.auxiliary_registry_contract(), cold_auxiliary_contract);
    // A structurally valid candidate reaches the DSO prelude, but its cold engine/carrier/graph
    // remains private until consensus.  A preparation fault therefore leaves no topology behind.
    EXPECT_THROW(system.install_program(failed_first_prepare), std::runtime_error);
    EXPECT_EQ(system.checkpoint_topology_epoch(), 0u);
    EXPECT_TRUE(system.installed_program_hash().empty());
    EXPECT_EQ(system.field_provider_checkpoint_manifest(), cold_field_manifest);
    // The candidate registry may be sealed privately for graph construction, but the accepted
    // assembling declaration bytes cannot change across a rejected Program prelude.
    EXPECT_EQ(system.auxiliary_registry_contract(), cold_auxiliary_contract);
    system.install_program(accepted_a);
    const std::string accepted_hash = system.installed_program_hash();
    const std::vector<int> accepted_map = system.program_block_map();
    const auto accepted_budget = system.prepared_amr_program_flux_expression_budget();
    const std::string accepted_budget_contract = accepted_budget.exact_contract;
    // The complete POPSAND5 image and its configured-depth capacity pair are published before
    // owner-last. Re-validating the public capacity is therefore read-only, and a refused B
    // candidate cannot clobber either half of A's bootstrap authority.
    const std::vector<std::uint8_t> accepted_bootstrap = system.program_accepted_state();
    const auto accepted_capacity = system.checkpoint_program_state_capacity();
    EXPECT_FALSE(accepted_bootstrap.empty());
    EXPECT_LE(accepted_bootstrap.size(), accepted_capacity.first);
    EXPECT_EQ(system.program_accepted_state(), accepted_bootstrap);
    EXPECT_EQ(system.checkpoint_program_state_capacity(), accepted_capacity);
    EXPECT_THROW(system.install_program(failed_replacement), std::runtime_error);
    // A DSO prepare fault has no live rollback phase: the prior Program authority, block map and
    // flux ledger remain byte-for-byte the accepted A image until B is collectively published.
    EXPECT_EQ(system.installed_program_hash(), accepted_hash);
    EXPECT_EQ(system.program_block_map(), accepted_map);
    EXPECT_EQ(system.prepared_amr_program_flux_expression_budget().exact_contract,
              accepted_budget_contract);
    EXPECT_EQ(system.program_accepted_state(), accepted_bootstrap);
    EXPECT_EQ(system.checkpoint_program_state_capacity(), accepted_capacity);
    // B is permitted to inspect only the opaque, tagged preparation image.  It observes the
    // detached topology, lane and exact prototype, then mutates that detached prototype before
    // throwing.  None of that state is permitted to reach already accepted A.
    const auto markers_after_refusal = read_program_markers(marker_path);
    const auto refused_authority =
        marker_with_prefix(markers_after_refusal, "failed-replacement:prepare-authority:");
    ASSERT_TRUE(refused_authority.has_value());
    const auto refused_generation = marker_attribute(*refused_authority, "generation");
    const auto refused_levels = marker_attribute(*refused_authority, "levels");
    const auto refused_epoch = marker_attribute(*refused_authority, "epoch");
    const auto refused_topology_generation =
        marker_attribute(*refused_authority, "topology-generation");
    const auto refused_blocks = marker_attribute(*refused_authority, "blocks");
    const auto refused_lane = marker_attribute(*refused_authority, "lane");
    const auto refused_lane_size = marker_attribute(*refused_authority, "lane-size");
    const auto refused_components = marker_attribute(*refused_authority, "prototype-ncomp");
    const auto refused_ghosts = marker_attribute(*refused_authority, "prototype-ghosts");
    const auto refused_fabs = marker_attribute(*refused_authority, "prototype-local-fabs");
    ASSERT_TRUE(refused_generation.has_value());
    ASSERT_TRUE(refused_levels.has_value());
    ASSERT_TRUE(refused_epoch.has_value());
    ASSERT_TRUE(refused_topology_generation.has_value());
    ASSERT_TRUE(refused_blocks.has_value());
    ASSERT_TRUE(refused_lane.has_value());
    ASSERT_TRUE(refused_lane_size.has_value());
    ASSERT_TRUE(refused_components.has_value());
    ASSERT_TRUE(refused_ghosts.has_value());
    ASSERT_TRUE(refused_fabs.has_value());
    EXPECT_GT(std::stoull(*refused_generation), 0u);
    EXPECT_EQ(*refused_levels, "1");
    EXPECT_EQ(*refused_blocks, "1");
    EXPECT_FALSE(refused_lane->empty());
    EXPECT_EQ(*refused_lane_size, "1");
    EXPECT_EQ(*refused_components, "1");
    EXPECT_EQ(*refused_ghosts, std::to_string(pops::Minmod::n_ghost));
    EXPECT_GT(std::stoull(*refused_fabs), 0u);
    EXPECT_NE(marker_index(markers_after_refusal,
                           "failed-replacement:prepare-authority:mutated-detached-prototype"),
              markers_after_refusal.size());
    EXPECT_LT(
        marker_index_with_prefix(markers_after_refusal, "failed-replacement:prepare-authority:"),
        marker_index(markers_after_refusal, "failed-replacement:destroy"));
    // The old accepted owner must survive the rejected replacement.  A successful candidate then
    // atomically replaces it while the runtime remains in its assembling/bootstrap window.  Its
    // distinct artifact identity proves that the replacement crossed the sole owner-last publish
    // boundary, rather than being an alias of A's accepted DSO.
    system.install_program(replacement_b);
    EXPECT_EQ(system.installed_program_hash(),
              std::string(kSyntheticLoaderProgramHash) + "/replacement-b");
    EXPECT_NE(system.installed_program_hash(), accepted_hash);
    const auto markers_after_accept = read_program_markers(marker_path);
    const auto accepted_authority =
        marker_with_prefix(markers_after_accept, "replacement-b:prepare-authority:");
    ASSERT_TRUE(accepted_authority.has_value());
    const auto accepted_generation = marker_attribute(*accepted_authority, "generation");
    ASSERT_TRUE(accepted_generation.has_value());
    EXPECT_GT(std::stoull(*accepted_generation), 0u);
    EXPECT_EQ(marker_attribute(*accepted_authority, "levels"), refused_levels);
    EXPECT_EQ(marker_attribute(*accepted_authority, "epoch"), refused_epoch);
    EXPECT_EQ(marker_attribute(*accepted_authority, "topology-generation"),
              refused_topology_generation);
    EXPECT_EQ(marker_attribute(*accepted_authority, "blocks"), refused_blocks);
    EXPECT_EQ(marker_attribute(*accepted_authority, "lane"), refused_lane);
    EXPECT_EQ(marker_attribute(*accepted_authority, "lane-size"), refused_lane_size);
    EXPECT_EQ(marker_attribute(*accepted_authority, "prototype-ncomp"), refused_components);
    EXPECT_EQ(marker_attribute(*accepted_authority, "prototype-ghosts"), refused_ghosts);
    EXPECT_EQ(marker_attribute(*accepted_authority, "prototype-local-fabs"), refused_fabs);
    // Candidate B observes its detached image before A can be retired.  This ordering, combined
    // with the bit-identical POPSAND5 checks above, forbids any pre-publish leakage from B to A.
    EXPECT_LT(marker_index_with_prefix(markers_after_accept, "replacement-b:prepare-authority:"),
              marker_index(markers_after_accept, "accepted-a:destroy"));
    bind_refined_system(system);
  }
}
