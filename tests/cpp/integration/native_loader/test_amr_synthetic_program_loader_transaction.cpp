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
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

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
                          bool malformed_candidate = false, bool prepare_throws = false) {
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
#include <pops/runtime/program/step_transaction.hpp>
#include <cstdint>
#include <functional>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
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
};

void program_candidate_step(void* opaque, double dt) {
  static_cast<ProgramCandidateState*>(opaque)->step(dt);
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
#if @@PREPARE_THROWS@@
    throw std::runtime_error("injected AMR candidate preparation failure");
#endif
    state->prepare(host->preparation);
    state->prepare = {};
    return true;
  } catch (...) {
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
constexpr char kCandidateArtifact[] = "tests.synthetic-loader/program/loader-transaction-v1";
constexpr char kCandidateAbiKey[] = POPS_ABI_KEY_LITERAL;
constexpr const char* kCandidateRouteManifest = pops::kRouteRegistrySignature;
constexpr char kCandidateBoundaryManifest[] = "tests.synthetic-loader.boundary.v5";
constexpr char kCandidateResourceManifest[] =
    R"JSON({"resource_plan":{"digest":"4ca46764b074a0c691ab69f5853aad7492d5a0ed2bb899f8ceb1ed94e3f477df","entries":[],"maximum_bytes":0,"schema":"program-resource-plan:v1","schema_version":1},"resource_plan_digest":"4ca46764b074a0c691ab69f5853aad7492d5a0ed2bb899f8ceb1ed94e3f477df"})JSON";
constexpr char kCandidateCheckpointIdentity[] = "tests.synthetic-loader.checkpoint.v5";
constexpr char kCandidateBlockName[] = "tracer";
constexpr pops::runtime::program::ProgramBlockRecord kCandidateBlocks[] = {
    {{kCandidateBlockName, sizeof(kCandidateBlockName) - 1}}};
constexpr pops::runtime::program::ProgramFluxBudgetRecord kCandidateFluxBudgets[] = {
    {UINT64_C(10), UINT64_C(1), UINT64_C(0), UINT64_C(0)}};
constexpr pops::runtime::program::ProgramAbiTable kCandidateBlockTable{
    kCandidateBlocks, 1, sizeof(kCandidateBlocks[0])};
constexpr pops::runtime::program::ProgramAbiTable kCandidateFluxTable{
    kCandidateFluxBudgets, 1, sizeof(kCandidateFluxBudgets[0])};
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
  state->prepare = [state_ptr = state.get()](
                       const pops::runtime::program::ProgramPreparationHostRef& preparation) {
  state_ptr->ctx_owner = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(
      preparation);
  auto& context = *state_ptr->ctx_owner;
  auto inject_retry = std::make_shared<bool>(true);
  context.configure_primary_clock("clock.macro");
  state_ptr->step = [context_owner = state_ptr->ctx_owner, inject_retry](double macro_dt) {
    auto& context = *context_owner;
    context.advance_hierarchy(macro_dt, [context_owner, inject_retry](double level_dt) {
          auto& context = *context_owner;
          context.set_stage_time(0, 1);
          auto& accepted = context.state(0);
          auto& candidate = context.scratch_state(1000, 0, accepted);
          auto& explicit_rate = context.rhs_scratch(2000, 0, accepted);
          context.neg_div_flux_default_into(0, accepted, explicit_rate, 3000);
          context.lincomb(candidate, pops::Real(1), accepted, pops::Real(0), accepted);
          // Materialize ten independent, authenticated default-flux bases. The dyadic weights
          // sum exactly to one, so this decimal-boundary capacity witness preserves the fixture's
          // physical update while forcing identities 1 through 10 into the live expression.
          for (int basis = 0; basis < 10; ++basis) {
            auto& rate = basis == 0 ? explicit_rate
                                    : context.rhs_scratch(2000 + basis, 0, accepted);
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
          if (*inject_retry) {
            *inject_retry = false;
            throw pops::runtime::program::StepAttemptRejected(
                pops::SolveStatus::kIterationLimit,
                pops::runtime::program::StepAttemptDisposition::kRetry, UINT32_C(0x534C5452),
                "implicit-source", "injected-synthetic-loader-transaction-retry");
          }
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
  descriptor.persistent_resource_manifest = {kCandidateResourceManifest,
                                             sizeof(kCandidateResourceManifest) - 1};
  descriptor.checkpoint_identity = {kCandidateCheckpointIdentity,
                                    sizeof(kCandidateCheckpointIdentity) - 1};
  descriptor.blocks = kCandidateBlockTable;
  descriptor.flux_budgets = kCandidateFluxTable;
  // The host/DSO carrier belongs to ProgramCandidateDescriptor::context; it is not a scientific
  // Program buffer.  Authenticate the canonical empty resource plan instead of inventing a row.
  descriptor.maximum_bytes = 0;
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
    const std::size_t position = result.find(from);
    if (position == std::string::npos)
      throw std::logic_error("synthetic AMR loader source placeholder is missing");
    result.replace(position, from.size(), to);
  };
  replace("@@MARKER_PATH@@", marker_path);
  replace("@@MARKER_TAG@@", marker_tag);
  replace("@@MALFORMED_CANDIDATE@@", malformed_candidate ? "1" : "0");
  replace("@@PREPARE_THROWS@@", prepare_throws ? "1" : "0");
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

  try {
    continuous.step(dt);
    FAIL() << "the injected implicit-source retry was not surfaced";
  } catch (const std::runtime_error& rejected) {
    // v5 callbacks translate DSO exception objects while the image is still resident.  The host
    // retains the failure's durable semantic bytes rather than allowing a foreign exception to
    // escape into the facade.
    const std::string message = rejected.what();
    EXPECT_NE(message.find("implicit-source"), std::string::npos);
    EXPECT_NE(message.find(std::to_string(kSyntheticLoaderRetryReason)), std::string::npos);
  }
  EXPECT_EQ(continuous.macro_step(), 0);
  EXPECT_DOUBLE_EQ(continuous.time(), 0.0);
  EXPECT_TRUE(byte_exact_equal(continuous.block_level_state_global(kBlock, 0), coarse_before));
  EXPECT_TRUE(byte_exact_equal(continuous.block_level_state_global(kBlock, 1), fine_before));

  continuous.step(dt);
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
  EXPECT_TRUE(std::any_of(
      materialized_stages.begin(), materialized_stages.end(),
      [](const std::string& stage) { return stage.find("/basis/10/") != std::string::npos; }));
  EXPECT_LE(accepted_bytes.size(), continuous.checkpoint_program_state_capacity().first);

  continuous.step(dt);
  continuous.step(dt);
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
  const auto compile_fixture = [&](const std::string& tag, bool malformed, bool prepare_throws) {
    const std::string source_path = stem + "_" + tag + ".cpp";
    const std::string shared_object = stem + "_" + tag + ".so";
    {
      std::ofstream source(source_path);
      source << loader_source(marker_path, tag, malformed, prepare_throws);
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
  const std::string failed_replacement = compile_fixture("failed-replacement", false, true);
  const std::string replacement_b = compile_fixture("replacement-b", false, false);

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
    EXPECT_THROW(system.install_program(failed_replacement), std::runtime_error);
    // A DSO prepare fault has no live rollback phase: the prior Program authority, block map and
    // flux ledger remain byte-for-byte the accepted A image until B is collectively published.
    EXPECT_EQ(system.installed_program_hash(), accepted_hash);
    EXPECT_EQ(system.program_block_map(), accepted_map);
    EXPECT_EQ(system.prepared_amr_program_flux_expression_budget().exact_contract,
              accepted_budget_contract);
    // The old accepted owner must survive the rejected replacement.  A successful candidate then
    // atomically replaces it while the runtime remains in its assembling/bootstrap window.
    system.install_program(replacement_b);
    bind_refined_system(system);
  }
}
