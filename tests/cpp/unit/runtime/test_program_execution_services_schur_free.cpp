// Exact-ranked ProgramExecutionServices compile-fire plus grid-free cadence transaction proofs.

#include <gtest/gtest.h>

#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/program_persistent_value_checkpoint.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>

#include <concepts>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <functional>
#include <vector>

template <int Dim>
concept ExactRankedFieldProgramRoutes = requires(
    pops::runtime::program::ProgramExecutionServices<Dim>& context, pops::MultiFab<Dim>& stage,
    const pops::runtime::multiblock::BoundaryEvaluationPoint& point,
    const pops::CompiledFieldBoundaryKernel<Dim>& boundary_kernel,
    const pops::FieldLogicalTimePoint& logical_point,
    const std::vector<const pops::MultiFab<Dim>*>& stages,
    std::initializer_list<
        typename pops::runtime::program::ProgramExecutionServices<Dim>::FieldStageOverride>
        overrides,
    const std::string& identity, const std::vector<double>& parameters) {
  { context.solve_fields() } -> std::same_as<pops::SolveOutcome>;
  { context.solve_fields_from_state(0, stage) } -> std::same_as<pops::SolveOutcome>;
  {
    context.solve_fields_from_state_at(point, identity, 0, stage)
  } -> std::same_as<pops::SolveOutcome>;
  { context.solve_fields_from_blocks(stages) } -> std::same_as<pops::SolveOutcome>;
  { context.solve_fields_from_blocks_at(point, 0, overrides) } -> std::same_as<pops::SolveOutcome>;
  context.set_field_boundary_kernel(identity, boundary_kernel);
  context.set_field_logical_timepoint(identity, logical_point);
  context.set_field_boundary_parameters(identity, parameters);
};

static_assert(std::is_class_v<pops::runtime::program::ProgramExecutionServices<1>>);
static_assert(std::is_class_v<pops::runtime::program::ProgramExecutionServices<2>>);
static_assert(std::is_class_v<pops::runtime::program::ProgramExecutionServices<3>>);
static_assert(
    !std::is_trivially_constructible_v<pops::runtime::program::ProgramExecutionServices<2>>);
static_assert(ExactRankedFieldProgramRoutes<1>);
static_assert(ExactRankedFieldProgramRoutes<2>);
static_assert(ExactRankedFieldProgramRoutes<3>);

namespace {

class TestPreparationImage final : public pops::runtime::program::ProgramPreparationImage {
 public:
  TestPreparationImage(std::uint32_t dimension, pops::runtime::program::ProgramRuntimeKind kind,
                       pops::runtime::program::ProgramExecutionServicesRef services,
                       std::uint64_t generation = 1)
      : ProgramPreparationImage(dimension, kind, services, generation) {
    bind_image_services(services);
  }
};

}  // namespace

namespace {

const std::string& exact_empty_resource_manifest(std::uint64_t maximum_bytes = 0) {
  using namespace pops::runtime::program;
  static const std::string zero_manifest = [] {
    ProgramInstallationTables tables;
    const std::string payload = tables.canonical_resource_digest_payload(0);
    const std::string digest =
        pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
    return "{\"resource_plan\":" + tables.canonical_resource_manifest(0, digest) +
           ",\"resource_plan_digest\":\"" + digest + "\"}";
  }();
  if (maximum_bytes == 0)
    return zero_manifest;
  static const std::string one_byte_manifest = [] {
    ProgramInstallationTables tables;
    const std::string payload = tables.canonical_resource_digest_payload(1);
    const std::string digest =
        pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
    return "{\"resource_plan\":" + tables.canonical_resource_manifest(1, digest) +
           ",\"resource_plan_digest\":\"" + digest + "\"}";
  }();
  if (maximum_bytes == 1)
    return one_byte_manifest;
  throw std::invalid_argument("synthetic fixture only materializes exact empty ceilings 0 or 1");
}

struct CadenceSyntheticCandidate final {
  std::function<void(double)> on_step;
};

class CadencePreparationImage final : public pops::runtime::program::ProgramPreparationImage {
 public:
  CadencePreparationImage(std::uint32_t dimension, std::uint64_t generation,
                          pops::runtime::program::ProgramExecutionServicesRef services)
      : ProgramPreparationImage(dimension, pops::runtime::program::ProgramRuntimeKind::uniform,
                                services, generation) {
    bind_image_services(services);
  }
};

void cadence_synthetic_step(void* opaque, double dt) {
  auto& candidate = *static_cast<CadenceSyntheticCandidate*>(opaque);
  if (candidate.on_step)
    candidate.on_step(dt);
}

bool cadence_synthetic_prepare(void*, const pops::runtime::program::ProgramHostDescriptor*,
                               pops::runtime::program::ProgramInstallDiagnostic*) noexcept {
  return true;
}

void cadence_synthetic_destroy(void* opaque) noexcept {
  // The fixture borrows the stack-owned candidate so the test can observe dispatches directly.
  // PreparedProgramInstallation owns the descriptor/DSO lifetime, not this callback context.
  (void)opaque;
}

template <int Dim>
pops::runtime::program::PreparedProgramInstallation prepared_cadence_artifact(
    CadenceSyntheticCandidate& candidate, std::uint64_t generation) {
  using namespace pops::runtime::program;
  static constexpr char metadata[] = "cadence-synthetic";
  ProgramCandidateDescriptor descriptor{};
  descriptor.struct_size = sizeof(ProgramCandidateDescriptor);
  descriptor.abi_version = kProgramInstallAbiVersion;
  descriptor.native_dimension = Dim;
  descriptor.runtime_kind = ProgramRuntimeKind::uniform;
  descriptor.provided_capability_bits = kKnownProgramCapabilityBits;
  descriptor.program_name = {metadata, sizeof(metadata) - 1};
  descriptor.artifact_identity = {metadata, sizeof(metadata) - 1};
  descriptor.abi_key = {metadata, sizeof(metadata) - 1};
  descriptor.route_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.boundary_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.persistent_resource_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.checkpoint_identity = {metadata, sizeof(metadata) - 1};
  descriptor.context = &candidate;
  descriptor.prepare = &cadence_synthetic_prepare;
  descriptor.step = &cadence_synthetic_step;
  descriptor.destroy = &cadence_synthetic_destroy;
  const std::string& resource_manifest = exact_empty_resource_manifest();
  descriptor.persistent_resource_manifest = {resource_manifest.data(), resource_manifest.size()};
  ProgramExecutionServicesRef services{&candidate, &candidate, &candidate, &candidate, &candidate,
                                       &candidate, &candidate, &candidate, &candidate};
  ProgramHostDescriptor host{};
  host.native_dimension = Dim;
  host.runtime_kind = ProgramRuntimeKind::uniform;
  host.capability_bits = kKnownProgramCapabilityBits;
  host.services = services;
  auto image = std::make_shared<CadencePreparationImage>(Dim, generation, services);
  bind_program_preparation_image(host, image);
  OwnedProgramInstallation owner(
      pops::dynlib::UniqueHandle{nullptr}, descriptor,
      ProgramInstallationMetadata{"cadence-synthetic", "abi", "route", "boundary",
                                  exact_empty_resource_manifest(), "checkpoint",
                                  "cadence-synthetic"});
  owner.set_preparation_image(image);
  owner.prepare(host);
  PreparedProgramInstallation artifact(std::move(owner));
  artifact.seal_resource_plan(std::vector<ProgramInstallationTables::ResourcePrototype>{});
  return artifact;
}

}  // namespace

TEST(ProgramExecutionServicesSchurFree, ExactRankedHeaderRequiresTaggedPreparationImage) {
  pops::runtime::program::ProgramPreparationHostRef unbound{};
  EXPECT_THROW((void)pops::runtime::program::make_program_execution_provider<2>(unbound),
               std::invalid_argument);
}

TEST(ProgramPreparationImage, BindsOneStableTypedImageWithoutFacadeRecovery) {
  using namespace pops::runtime::program;
  int token = 0;
  ProgramExecutionServicesRef services{};
  services.state_store = &token;
  services.field_store = &token;
  services.spatial_executor = &token;
  services.hierarchy_executor = &token;
  services.history_store = &token;
  services.clock_service = &token;
  services.reduction_service = &token;
  services.transaction_service = &token;
  services.persistent_value_store = &token;
  auto image = std::make_shared<TestPreparationImage>(2, ProgramRuntimeKind::uniform, services);
  ProgramHostDescriptor host{};
  host.native_dimension = 2;
  host.runtime_kind = ProgramRuntimeKind::uniform;
  host.capability_bits = kKnownProgramCapabilityBits;
  host.services = services;

  bind_program_preparation_image(host, image);
  const auto& bound =
      require_program_preparation_image(host.preparation, 2, ProgramRuntimeKind::uniform);
  EXPECT_EQ(&bound, image.get());
  EXPECT_EQ(host.preparation.image, image.get());
  EXPECT_TRUE(valid_program_host_descriptor(host));

  ProgramPreparationHostRef forged = host.preparation;
  int other_service = 0;
  forged.services.clock_service = &other_service;
  EXPECT_THROW((void)require_program_preparation_image(forged, 2, ProgramRuntimeKind::uniform),
               std::invalid_argument);

  EXPECT_THROW(bind_program_preparation_image(host, std::make_shared<TestPreparationImage>(
                                                        3, ProgramRuntimeKind::uniform, services)),
               std::invalid_argument);
}

TEST(ProgramRuntimeStateCadence, SharedDispatcherOwnsHoldSubstepAndCursorCommit) {
  pops::runtime::program::ProgramRuntimeState<2> state;
  struct Dispatch {
    double start = 0.0;
    double dt = 0.0;
    int macro_step = -1;
  };
  std::vector<Dispatch> dispatches;
  double physical_time = 2.0;
  int macro_step = 4;
  CadenceSyntheticCandidate candidate;
  candidate.on_step = [&](double dt) { dispatches.push_back({physical_time, dt, macro_step}); };
  state.install_prepared_artifact(
      prepared_cadence_artifact<2>(candidate, state.step_install_generation_ + 1));
  state.set_cadence(/*substeps=*/2, /*stride=*/2, "Fixture");

  state.dispatch_cadence_step(physical_time, macro_step, 0.1, "Fixture");
  EXPECT_TRUE(dispatches.empty());
  EXPECT_DOUBLE_EQ(physical_time, 2.1);
  EXPECT_EQ(macro_step, 5);

  state.dispatch_cadence_step(physical_time, macro_step, 0.3, "Fixture");
  ASSERT_EQ(dispatches.size(), 2);
  EXPECT_DOUBLE_EQ(dispatches[0].start, 2.0);
  EXPECT_EQ(dispatches[0].macro_step, 4);
  EXPECT_DOUBLE_EQ(dispatches[1].start, dispatches[0].start + dispatches[0].dt);
  EXPECT_EQ(macro_step, 6);
}

TEST(ProgramRuntimeStateCadence, DispatchFailureRestoresCursorAndReentrancyLease) {
  pops::runtime::program::ProgramRuntimeState<3> state;
  double physical_time = 1.0;
  int macro_step = 0;
  int calls = 0;
  bool fail_second_substep = true;
  CadenceSyntheticCandidate candidate;
  candidate.on_step = [&](double) {
    ++calls;
    if (fail_second_substep && calls == 2)
      throw std::runtime_error("injected cadence substep failure");
  };
  state.install_prepared_artifact(
      prepared_cadence_artifact<3>(candidate, state.step_install_generation_ + 1));
  state.set_cadence(/*substeps=*/2, /*stride=*/1, "Fixture");

  EXPECT_THROW(state.dispatch_cadence_step(physical_time, macro_step, 0.4, "Fixture"),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(physical_time, 1.0);
  EXPECT_EQ(macro_step, 0);
  EXPECT_FALSE(state.cadence_dispatch_active_);

  calls = 0;
  fail_second_substep = false;
  EXPECT_NO_THROW(state.dispatch_cadence_step(physical_time, macro_step, 0.4, "Fixture"));
  EXPECT_EQ(calls, 2);
  EXPECT_DOUBLE_EQ(physical_time, 1.4);
  EXPECT_EQ(macro_step, 1);
}

TEST(ProgramRuntimeStateCadence, MacroStepOverflowFailsBeforeProgramDispatch) {
  pops::runtime::program::ProgramRuntimeState<1> state;
  double physical_time = 0.0;
  int macro_step = std::numeric_limits<int>::max();
  int calls = 0;
  CadenceSyntheticCandidate candidate;
  candidate.on_step = [&](double) { ++calls; };
  state.install_prepared_artifact(
      prepared_cadence_artifact<1>(candidate, state.step_install_generation_ + 1));

  EXPECT_THROW(state.dispatch_cadence_step(physical_time, macro_step, 0.1, "Fixture"),
               std::overflow_error);
  EXPECT_EQ(calls, 0);
  EXPECT_DOUBLE_EQ(physical_time, 0.0);
  EXPECT_EQ(macro_step, std::numeric_limits<int>::max());
}

namespace {

struct SyntheticProgramCandidate {
  int step_calls = 0;
  int dt_bound_calls = 0;
  int hierarchy_refresh_calls = 0;
  int history_remap_calls = 0;
  int restart_preflight_calls = 0;
  int restart_regrid_calls = 0;
  int restart_resync_calls = 0;
  int snapshot_create_calls = 0;
  int snapshot_destroy_calls = 0;
  int prepare_calls = 0;
  int destroy_calls = 0;
  double last_dt = 0.0;
  std::function<void(double)> on_step;
};

int stateless_step_calls = 0;
int stateless_dt_bound_calls = 0;

void stateless_program_step(void*, double) {
  ++stateless_step_calls;
}
bool stateless_program_prepare(void*, const pops::runtime::program::ProgramHostDescriptor*,
                               pops::runtime::program::ProgramInstallDiagnostic*) noexcept {
  return true;
}
double stateless_program_dt_bound(void*, double cfl) {
  ++stateless_dt_bound_calls;
  return 0.25 * cfl;
}

void synthetic_program_step(void* opaque, double dt) {
  auto& candidate = *static_cast<SyntheticProgramCandidate*>(opaque);
  ++candidate.step_calls;
  candidate.last_dt = dt;
  if (candidate.on_step)
    candidate.on_step(dt);
}

bool synthetic_program_prepare(void* opaque, const pops::runtime::program::ProgramHostDescriptor*,
                               pops::runtime::program::ProgramInstallDiagnostic*) noexcept {
  ++static_cast<SyntheticProgramCandidate*>(opaque)->prepare_calls;
  return true;
}

double synthetic_program_dt_bound(void* opaque, double cfl) {
  auto& candidate = *static_cast<SyntheticProgramCandidate*>(opaque);
  ++candidate.dt_bound_calls;
  return 0.5 * cfl;
}

void synthetic_program_destroy(void* opaque) noexcept {
  ++static_cast<SyntheticProgramCandidate*>(opaque)->destroy_calls;
}

class SyntheticAcceptedSnapshot final
    : public pops::runtime::program::AcceptedProgramExecutionServicesSnapshot {
 public:
  explicit SyntheticAcceptedSnapshot(SyntheticProgramCandidate& candidate)
      : candidate_(&candidate) {}
  ~SyntheticAcceptedSnapshot() override { ++candidate_->snapshot_destroy_calls; }

  std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot>
  prepare_restore() const override {
    return std::make_unique<SyntheticAcceptedSnapshot>(*candidate_);
  }
  void publish_restore() noexcept override {}

 private:
  SyntheticProgramCandidate* candidate_ = nullptr;
};

void synthetic_hierarchy_refresh(void* opaque) {
  ++static_cast<SyntheticProgramCandidate*>(opaque)->hierarchy_refresh_calls;
}

void synthetic_history_remap(void* opaque, const void* descriptor) {
  if (descriptor == nullptr)
    throw std::invalid_argument("synthetic history remap requires a descriptor");
  ++static_cast<SyntheticProgramCandidate*>(opaque)->history_remap_calls;
}

void synthetic_restart_preflight(void* opaque) {
  ++static_cast<SyntheticProgramCandidate*>(opaque)->restart_preflight_calls;
}

void synthetic_restart_regrid(void* opaque) {
  ++static_cast<SyntheticProgramCandidate*>(opaque)->restart_regrid_calls;
}

void synthetic_restart_resync(void* opaque) {
  ++static_cast<SyntheticProgramCandidate*>(opaque)->restart_resync_calls;
}

pops::runtime::program::AcceptedProgramExecutionServicesSnapshot* synthetic_snapshot_create(
    void* opaque) {
  auto& candidate = *static_cast<SyntheticProgramCandidate*>(opaque);
  ++candidate.snapshot_create_calls;
  return new SyntheticAcceptedSnapshot(candidate);
}

pops::runtime::program::ProgramCandidateDescriptor synthetic_descriptor(
    SyntheticProgramCandidate& candidate, bool with_dt_bound = true, bool with_lifecycle = false) {
  using namespace pops::runtime::program;
  static constexpr char metadata[] = "synthetic";
  ProgramCandidateDescriptor descriptor{};
  descriptor.prepare = &stateless_program_prepare;
  descriptor.struct_size = sizeof(ProgramCandidateDescriptor);
  descriptor.abi_version = kProgramInstallAbiVersion;
  descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  descriptor.runtime_kind = with_lifecycle ? ProgramRuntimeKind::amr : ProgramRuntimeKind::uniform;
  descriptor.provided_capability_bits =
      with_lifecycle ? kProgramCapabilityHierarchy | kProgramCapabilityTransactions : 0;
  descriptor.program_name = {metadata, sizeof(metadata) - 1};
  descriptor.artifact_identity = {metadata, sizeof(metadata) - 1};
  descriptor.abi_key = {metadata, sizeof(metadata) - 1};
  descriptor.route_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.boundary_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.persistent_resource_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.checkpoint_identity = {metadata, sizeof(metadata) - 1};
  descriptor.maximum_bytes = 0;
  descriptor.context = &candidate;
  descriptor.prepare = &synthetic_program_prepare;
  descriptor.step = &synthetic_program_step;
  descriptor.dt_bound = with_dt_bound ? &synthetic_program_dt_bound : nullptr;
  descriptor.destroy = &synthetic_program_destroy;
  const std::string& resource_manifest = exact_empty_resource_manifest();
  descriptor.persistent_resource_manifest = {resource_manifest.data(), resource_manifest.size()};
  if (with_lifecycle) {
    descriptor.hierarchy_refresh = &synthetic_hierarchy_refresh;
    descriptor.history_remap_accepted = &synthetic_history_remap;
    descriptor.restart_regrid_preflight = &synthetic_restart_preflight;
    descriptor.restart_regrid = &synthetic_restart_regrid;
    descriptor.restart_resync = &synthetic_restart_resync;
    descriptor.create_accepted_snapshot = &synthetic_snapshot_create;
  }
  return descriptor;
}

pops::runtime::program::OwnedProgramInstallation make_synthetic_owner(
    SyntheticProgramCandidate& candidate, bool with_dt_bound = true, bool with_lifecycle = false,
    std::uint64_t generation = 1) {
  using namespace pops::runtime::program;
  OwnedProgramInstallation artifact(
      pops::dynlib::UniqueHandle{nullptr},
      synthetic_descriptor(candidate, with_dt_bound, with_lifecycle),
      ProgramInstallationMetadata{"artifact", "abi", "route", "boundary",
                                  exact_empty_resource_manifest(), "checkpoint"});
  ProgramHostDescriptor host{};
  host.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  host.runtime_kind = with_lifecycle ? ProgramRuntimeKind::amr : ProgramRuntimeKind::uniform;
  host.services = {&candidate, &candidate, &candidate, &candidate, &candidate,
                   &candidate, &candidate, &candidate, &candidate};
  auto preparation_image =
      std::make_shared<TestPreparationImage>(static_cast<std::uint32_t>(pops::kNativeDimension),
                                             host.runtime_kind, host.services, generation);
  bind_program_preparation_image(host, preparation_image);
  artifact.set_preparation_image(preparation_image);
  artifact.prepare(host);
  return artifact;
}

pops::runtime::program::PreparedProgramInstallation prepared_synthetic_artifact(
    SyntheticProgramCandidate& candidate, std::uint64_t generation, bool with_dt_bound = true,
    bool with_lifecycle = false) {
  using namespace pops::runtime::program;
  PreparedProgramInstallation artifact(
      make_synthetic_owner(candidate, with_dt_bound, with_lifecycle, generation));
  // The synthetic descriptor declares no runtime resources, but ABI-v5 publication still needs
  // the host to seal that exact empty resource plan before it becomes an artifact receipt.
  artifact.seal_resource_plan(std::vector<ProgramInstallationTables::ResourcePrototype>{});
  return artifact;
}

pops::runtime::program::OwnedProgramInstallation make_synthetic_owner_with_unplanned_ceiling(
    SyntheticProgramCandidate& candidate, std::uint64_t generation = 1) {
  using namespace pops::runtime::program;
  auto descriptor = synthetic_descriptor(candidate);
  descriptor.maximum_bytes = 1;
  OwnedProgramInstallation artifact(
      pops::dynlib::UniqueHandle{nullptr}, descriptor,
      ProgramInstallationMetadata{"artifact", "abi", "route", "boundary",
                                  exact_empty_resource_manifest(1), "checkpoint", "artifact"});
  ProgramHostDescriptor host{};
  host.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  host.services = {&candidate, &candidate, &candidate, &candidate, &candidate,
                   &candidate, &candidate, &candidate, &candidate};
  auto preparation_image = std::make_shared<TestPreparationImage>(
      static_cast<std::uint32_t>(pops::kNativeDimension), ProgramRuntimeKind::uniform,
      host.services, generation);
  bind_program_preparation_image(host, preparation_image);
  artifact.set_preparation_image(preparation_image);
  artifact.prepare(host);
  return artifact;
}

}  // namespace

TEST(OwnedProgramInstallation, DispatchesSyntheticCallbacksWhileOwnerIsResident) {
  SyntheticProgramCandidate candidate;
  auto artifact = make_synthetic_owner(candidate);

  artifact.invoke_step(0.25);
  const auto bound = artifact.invoke_dt_bound(0.8);

  EXPECT_EQ(candidate.step_calls, 1);
  EXPECT_DOUBLE_EQ(candidate.last_dt, 0.25);
  ASSERT_TRUE(bound);
  EXPECT_DOUBLE_EQ(*bound, 0.4);
  EXPECT_EQ(candidate.dt_bound_calls, 1);
  artifact.reset();
  EXPECT_EQ(candidate.destroy_calls, 1);
}

TEST(OwnedProgramInstallation, RefusesDispatchBeforeAndPreparationTwice) {
  using namespace pops::runtime::program;
  SyntheticProgramCandidate candidate;
  OwnedProgramInstallation artifact(pops::dynlib::UniqueHandle{nullptr},
                                    synthetic_descriptor(candidate), ProgramInstallationMetadata{});
  EXPECT_THROW(artifact.invoke_step(0.25), std::logic_error);
  ProgramHostDescriptor host{};
  host.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  host.services = {&candidate, &candidate, &candidate, &candidate, &candidate,
                   &candidate, &candidate, &candidate, &candidate};
  EXPECT_THROW(artifact.prepare(host), std::invalid_argument);
  EXPECT_EQ(candidate.prepare_calls, 0);
  auto preparation_image =
      std::make_shared<TestPreparationImage>(static_cast<std::uint32_t>(pops::kNativeDimension),
                                             ProgramRuntimeKind::uniform, host.services);
  bind_program_preparation_image(host, preparation_image);
  artifact.set_preparation_image(preparation_image);
  artifact.prepare(host);
  EXPECT_EQ(candidate.prepare_calls, 1);
  EXPECT_THROW(artifact.prepare(host), std::logic_error);
}

TEST(ProgramCandidateDescriptor, EnforcesExactAmrLifecycleHookSet) {
  using namespace pops::runtime::program;
  SyntheticProgramCandidate candidate;
  auto uniform = synthetic_descriptor(candidate);
  EXPECT_TRUE(valid_program_candidate_descriptor(uniform));
  uniform.hierarchy_refresh = &synthetic_hierarchy_refresh;
  EXPECT_FALSE(valid_program_candidate_descriptor(uniform));

  auto amr = synthetic_descriptor(candidate, true, true);
  EXPECT_TRUE(valid_program_candidate_descriptor(amr));
  amr.restart_resync = nullptr;
  EXPECT_FALSE(valid_program_candidate_descriptor(amr));
}

TEST(OwnedProgramInstallation, InvokesStatelessNullContextCandidates) {
  using namespace pops::runtime::program;
  stateless_step_calls = 0;
  stateless_dt_bound_calls = 0;
  ProgramCandidateDescriptor descriptor{};
  descriptor.prepare = &stateless_program_prepare;
  descriptor.step = &stateless_program_step;
  descriptor.dt_bound = &stateless_program_dt_bound;
  OwnedProgramInstallation artifact(pops::dynlib::UniqueHandle{nullptr}, descriptor,
                                    ProgramInstallationMetadata{});
  ProgramHostDescriptor host{};
  host.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);
  int service = 0;
  host.services = {&service, &service, &service, &service, &service,
                   &service, &service, &service, &service};
  auto preparation_image =
      std::make_shared<TestPreparationImage>(static_cast<std::uint32_t>(pops::kNativeDimension),
                                             ProgramRuntimeKind::uniform, host.services);
  bind_program_preparation_image(host, preparation_image);
  artifact.set_preparation_image(preparation_image);
  artifact.prepare(host);
  artifact.invoke_step(0.5);
  ASSERT_TRUE(artifact.invoke_dt_bound(0.8));
  EXPECT_EQ(stateless_step_calls, 1);
  EXPECT_EQ(stateless_dt_bound_calls, 1);
}

TEST(ProgramInstallationTables, MaterializesEveryDsoViewBeforePreparation) {
  using namespace pops::runtime::program;
  SyntheticProgramCandidate candidate;
  auto descriptor = synthetic_descriptor(candidate);
  char block_name[] = "plasma";
  char parameter_name[] = "gamma";
  char history_name[] = "history:plasma";
  char checkpoint_name[] = "checkpoint:plasma";
  char owner_name[] = "plasma";
  char space_name[] = "state";
  char clock_name[] = "clock.macro";
  char transfer_name[] = "conservative";
  char route_name[] = "route:primary";
  char route_kind[] = "provider";
  char resource_schema[] = "program-resource-plan:v1";
  char resource_digest[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  char resource_path[] = "root/0";
  char resource_lifetime[] = "transient";
  char resource_centering[] = "cell";
  char resource_communication[] = "none";
  char resource_transfer[] = "redistribute_exact";
  char resource_off_policy[] = "none";
  char resource_components[] = "[]";
  char resource_shape[] = "[]";
  ProgramBlockRecord blocks[] = {{{block_name, sizeof(block_name) - 1}}};
  ProgramParameterRecord parameters[] = {
      {0, 0, 1.25, {parameter_name, sizeof(parameter_name) - 1}}};
  ProgramAuthorityRecord authorities[] = {{{1, 2, 3, 4}}};
  ProgramHistoryAuthorityRecord histories[] = {{{history_name, sizeof(history_name) - 1}, 2, 0}};
  ProgramCheckpointRecord checkpoints[] = {{{checkpoint_name, sizeof(checkpoint_name) - 1},
                                            {owner_name, sizeof(owner_name) - 1},
                                            {space_name, sizeof(space_name) - 1},
                                            {clock_name, sizeof(clock_name) - 1},
                                            {transfer_name, sizeof(transfer_name) - 1},
                                            0,
                                            1,
                                            2}};
  // This DSO-view fixture has no flux-basis or stage rows, so its final per-block active budget
  // is exactly empty.  Non-zero values would claim unmaterialized active basis authority.
  ProgramFluxBudgetRecord flux[] = {{0, 0, 0, 0}};
  ProgramResourcePlanRecord resources[1]{};
  resources[0].slot = 0;
  resources[0].value_id = 7;
  resources[0].occurrence_path_id = 11;
  resources[0].components = 1;
  resources[0].ghosts = 2;
  resources[0].bytes = 64;
  resources[0].maximum_bytes = 64;
  resources[0].schema = {resource_schema, sizeof(resource_schema) - 1};
  resources[0].plan_digest = {resource_digest, sizeof(resource_digest) - 1};
  resources[0].identity = {checkpoint_name, sizeof(checkpoint_name) - 1};
  resources[0].occurrence_path = {resource_path, sizeof(resource_path) - 1};
  resources[0].owner = {owner_name, sizeof(owner_name) - 1};
  resources[0].space = {space_name, sizeof(space_name) - 1};
  resources[0].clock = {clock_name, sizeof(clock_name) - 1};
  resources[0].lifetime = {resource_lifetime, sizeof(resource_lifetime) - 1};
  resources[0].centering = {resource_centering, sizeof(resource_centering) - 1};
  resources[0].off_policy = {resource_off_policy, sizeof(resource_off_policy) - 1};
  resources[0].communication = {resource_communication, sizeof(resource_communication) - 1};
  resources[0].transfer_provider = {resource_transfer, sizeof(resource_transfer) - 1};
  resources[0].restart_provider = {resource_off_policy, sizeof(resource_off_policy) - 1};
  resources[0].component_names = {resource_components, sizeof(resource_components) - 1};
  resources[0].shape = {resource_shape, sizeof(resource_shape) - 1};
  ProgramRouteRecord routes[] = {
      {{route_name, sizeof(route_name) - 1}, {route_kind, sizeof(route_kind) - 1}, 0}};
  descriptor.blocks = {blocks, 1, sizeof(ProgramBlockRecord)};
  descriptor.parameters = {parameters, 1, sizeof(ProgramParameterRecord)};
  descriptor.operator_authorities = {authorities, 1, sizeof(ProgramAuthorityRecord)};
  descriptor.history_authorities = {histories, 1, sizeof(ProgramHistoryAuthorityRecord)};
  descriptor.checkpoint_shape = {checkpoints, 1, sizeof(ProgramCheckpointRecord)};
  descriptor.flux_budgets = {flux, 1, sizeof(ProgramFluxBudgetRecord)};
  descriptor.resource_plan = {resources, 1, sizeof(ProgramResourcePlanRecord)};
  descriptor.maximum_bytes = 64;
  descriptor.boundary_routes = {routes, 1, sizeof(ProgramRouteRecord)};
  descriptor.provider_routes = {routes, 1, sizeof(ProgramRouteRecord)};
  ASSERT_TRUE(valid_program_candidate_descriptor(descriptor));
  std::size_t bytes = 0;
  auto metadata = ProgramInstallationMetadata::materialize(descriptor, bytes);
  auto tables = ProgramInstallationTables::materialize(descriptor, bytes);
  block_name[0] = 'X';
  route_name[0] = 'X';
  EXPECT_EQ(tables.blocks.at(0).name, "plasma");
  EXPECT_EQ(tables.parameters.at(0).name, "gamma");
  EXPECT_EQ(tables.history_authorities.at(0).identity, "history:plasma");
  EXPECT_EQ(tables.checkpoint_shape.at(0).identity, "checkpoint:plasma");
  EXPECT_EQ(tables.resource_plan.at(0).occurrence_path, "root/0");
  EXPECT_EQ(tables.resource_plan.at(0).owner, "plasma");
  EXPECT_EQ(tables.resource_plan.at(0).plan_digest,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
  EXPECT_EQ(tables.boundary_routes.at(0).identity, "route:primary");
  EXPECT_EQ(metadata.artifact_identity, "synthetic");

  // The host recomputes the digest over the canonical rows and requires the exact embedded
  // resource-plan object; a merely well-formed 64-hex digest is not authority.
  const std::string resource_payload = tables.canonical_resource_digest_payload(64);
  tables.resource_plan.at(0).plan_digest = pops::identity::sha256_hex(
      std::vector<std::uint8_t>(resource_payload.begin(), resource_payload.end()));
  metadata.persistent_resource_manifest =
      "{\"resource_plan\":" +
      tables.canonical_resource_manifest(64, tables.resource_plan.at(0).plan_digest) +
      ",\"resource_plan_digest\":\"" + tables.resource_plan.at(0).plan_digest + "\"}";
  EXPECT_NO_THROW(tables.validate_resource_authority(metadata, 64));
  const auto resource_plan = make_program_resource_plan(tables, 64);
  ASSERT_EQ(resource_plan.entries().size(), 1U);
  EXPECT_EQ(resource_plan.entries().front().identity, "checkpoint:plasma");
  EXPECT_EQ(resource_plan.entries().front().communication, "none");
  metadata.persistent_resource_manifest = "{\"resource_plan\":{}}";
  EXPECT_THROW(tables.validate_resource_authority(metadata, 64), std::invalid_argument);
  const std::string forged_text =
      "\"resource_plan\":" +
      tables.canonical_resource_manifest(64, tables.resource_plan.at(0).plan_digest) +
      ",\"resource_plan_digest\":\"" + tables.resource_plan.at(0).plan_digest + "\"";
  metadata.persistent_resource_manifest =
      "{\"note\":" + pops::runtime::program::detail::resource_json_string(forged_text) + "}";
  EXPECT_THROW(tables.validate_resource_authority(metadata, 64), std::invalid_argument);
  auto malformed_components = tables;
  malformed_components.resource_plan.at(0).component_names = "[\"rho\",]";
  EXPECT_THROW((void)malformed_components.canonical_resource_digest_payload(64),
               std::invalid_argument);
  auto malformed_shape = tables;
  malformed_shape.resource_plan.at(0).shape = "[0]";
  EXPECT_THROW((void)malformed_shape.canonical_resource_digest_payload(64), std::invalid_argument);

  descriptor.maximum_bytes = 63;
  bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
  descriptor.maximum_bytes = 64;
  resources[0].flags = 32;
  bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
  resources[0].flags = 0;
  resources[0].plan_digest = {resource_off_policy, sizeof(resource_off_policy) - 1};
  bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
}

TEST(ProgramInstallationTables, FluxRowsKeepQualifiedResourceOwnersSeparateFromDisplayBlocks) {
  using namespace pops::runtime::program;
  SyntheticProgramCandidate candidate;
  auto descriptor = synthetic_descriptor(candidate, true, true);
  constexpr std::string_view display_block = "tracer";
  constexpr std::string_view qualified_owner = "pops.handle.v1::case:fixture::block::tracer";
  constexpr std::string_view forged_owner = "pops.handle.v1::case:fixture::block::forged";
  constexpr std::string_view clock = "clock.macro";
  constexpr std::string_view schema = "program-resource-plan:v1";
  constexpr std::string_view digest =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  const auto view = [](std::string_view value) {
    return ProgramAbiView{value.data(), static_cast<std::uint64_t>(value.size())};
  };
  const ProgramBlockRecord blocks[] = {{view(display_block)}};
  const ProgramFluxBudgetRecord budgets[] = {{1, 1, 0, 0}};
  ProgramResourcePlanRecord resources[2]{};
  const auto resource = [&](std::size_t slot, std::uint64_t value_id, std::string_view identity,
                            std::string_view occurrence_path) {
    auto& row = resources[slot];
    row.slot = static_cast<std::uint32_t>(slot);
    row.value_id = value_id;
    row.occurrence_path_id = 100 + slot;
    row.level = -1;
    row.components = 1;
    row.bytes = 8;
    row.maximum_bytes = 8;
    row.schema = view(schema);
    row.plan_digest = view(digest);
    row.identity = view(identity);
    row.occurrence_path = view(occurrence_path);
    row.owner = view(qualified_owner);
    row.space = view("cell");
    row.clock = view(clock);
    row.lifetime = view("transient");
    row.centering = view("cell");
    row.off_policy = view("none");
    row.communication = view("none");
    row.transfer_provider = view("none");
    row.restart_provider = view("none");
    row.component_names = view("[]");
    row.shape = view("[]");
  };
  resource(0, 7, "resource:rhs", "root/rhs");
  resource(1, 8, "resource:commit", "root/commit");

  ProgramFluxBasisOccurrenceRecord basis{};
  basis.basis_slot = 0;
  basis.expression_slot = 0;
  basis.block = 0;
  basis.level = -1;
  basis.rhs_identity = 7;
  basis.provider = 1;
  basis.stage_numerator = 0;
  basis.stage_denominator = 1;
  basis.identity = view("basis:rhs");
  basis.occurrence_path = view("root/rhs:basis");
  basis.owner = view(qualified_owner);
  basis.clock = view(clock);
  ProgramFaceFluxStageRecord term{};
  term.slot = 0;
  term.basis_slot = 0;
  term.expression_slot = 1;
  term.dt_power = 1;
  term.coefficient_numerator = 1;
  term.coefficient_denominator = 1;
  term.identity = view("term:commit");
  term.occurrence_path = view("root/rhs:basis/final");
  term.owner = view(qualified_owner);
  term.clock = view(clock);

  descriptor.blocks = {blocks, 1, sizeof(ProgramBlockRecord)};
  descriptor.flux_budgets = {budgets, 1, sizeof(ProgramFluxBudgetRecord)};
  descriptor.resource_plan = {resources, 2, sizeof(ProgramResourcePlanRecord)};
  descriptor.flux_basis_occurrences = {&basis, 1, sizeof(ProgramFluxBasisOccurrenceRecord)};
  descriptor.face_flux_stages = {&term, 1, sizeof(ProgramFaceFluxStageRecord)};
  descriptor.maximum_bytes = 16;
  ASSERT_TRUE(valid_program_candidate_descriptor(descriptor));

  std::size_t bytes = 0;
  EXPECT_NO_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes));

  const ProgramFluxBudgetRecord forged_budgets[] = {{0, 1, 0, 0}};
  descriptor.flux_budgets = {forged_budgets, 1, sizeof(ProgramFluxBudgetRecord)};
  bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
  descriptor.flux_budgets = {budgets, 1, sizeof(ProgramFluxBudgetRecord)};

  basis.owner = view(forged_owner);
  bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
}

TEST(ProgramInstallationTables, RejectsMalformedAbiTableShapesAndContents) {
  using namespace pops::runtime::program;
  SyntheticProgramCandidate candidate;
  auto descriptor = synthetic_descriptor(candidate);
  ProgramAbiTable malformed{reinterpret_cast<const void*>(1), 1, sizeof(ProgramBlockRecord) - 1};
  descriptor.blocks = malformed;
  EXPECT_FALSE(valid_program_candidate_descriptor(descriptor));
  descriptor = synthetic_descriptor(candidate);
  descriptor.blocks = {reinterpret_cast<const void*>(1), (1u << 20) + 1,
                       sizeof(ProgramBlockRecord)};
  EXPECT_FALSE(valid_program_candidate_descriptor(descriptor));
  char nul_name[] = {'a', '\0', 'b'};
  ProgramBlockRecord nul_blocks[] = {{{nul_name, sizeof(nul_name)}}};
  descriptor = synthetic_descriptor(candidate);
  descriptor.blocks = {nul_blocks, 1, sizeof(ProgramBlockRecord)};
  std::size_t bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
}

TEST(ProgramInstallationTables, RejectsDuplicateAndOutOfRangeRecords) {
  using namespace pops::runtime::program;
  SyntheticProgramCandidate candidate;
  auto descriptor = synthetic_descriptor(candidate);
  static constexpr char name[] = "plasma";
  ProgramBlockRecord duplicate_blocks[] = {{{name, sizeof(name) - 1}}, {{name, sizeof(name) - 1}}};
  descriptor.blocks = {duplicate_blocks, 2, sizeof(ProgramBlockRecord)};
  std::size_t bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
  descriptor = synthetic_descriptor(candidate);
  ProgramBlockRecord blocks[] = {{{name, sizeof(name) - 1}}};
  ProgramParameterRecord parameter[] = {{1, 0, 0.0, {name, sizeof(name) - 1}}};
  descriptor.blocks = {blocks, 1, sizeof(ProgramBlockRecord)};
  descriptor.parameters = {parameter, 1, sizeof(ProgramParameterRecord)};
  bytes = 0;
  EXPECT_THROW((void)ProgramInstallationTables::materialize(descriptor, bytes),
               std::invalid_argument);
}

TEST(ProgramRuntimeStateArtifactOwner, PreparedArtifactDispatchesStepAndDtBound) {
  using State = pops::runtime::program::ProgramRuntimeState<2>;
  SyntheticProgramCandidate candidate;
  State state;
  const auto generation = state.step_install_generation_;
  state.operator_authorities_ = {{{1, 2, 3, 4}}};
  state.history_replay_authorities_ = {{"old-history", 2}};
  state.installed_hash_ = "old-artifact";
  state.artifact_backed_ = true;

  state.install_prepared_artifact(
      prepared_synthetic_artifact(candidate, state.step_install_generation_ + 1));
  state.step_(0.125);

  EXPECT_EQ(state.step_install_generation_, generation + 1);
  EXPECT_EQ(candidate.step_calls, 1);
  EXPECT_DOUBLE_EQ(candidate.last_dt, 0.125);
  ASSERT_TRUE(state.dt_bound_);
  EXPECT_DOUBLE_EQ(state.dt_bound_(0.6), 0.3);
  EXPECT_EQ(candidate.dt_bound_calls, 1);
  EXPECT_FALSE(state.artifact_backed_);
  EXPECT_TRUE(state.operator_authorities_.empty());
  EXPECT_TRUE(state.history_replay_authorities_.empty());
  EXPECT_TRUE(state.installed_hash_.empty());
  ASSERT_TRUE(state.artifact_publication_receipt());
  EXPECT_EQ(state.artifact_publication_receipt()->metadata.artifact_identity, "artifact");
  EXPECT_TRUE(state.artifact_publication_receipt()->resource_plan.entries().empty());
}

TEST(ProgramRuntimeStateArtifactOwner,
     FailedPublicationPreparationLeavesAcceptedArtifactUntouched) {
  using State = pops::runtime::program::ProgramRuntimeState<2>;
  SyntheticProgramCandidate accepted_candidate;
  SyntheticProgramCandidate refused_candidate;
  State state;
  state.install_prepared_artifact(
      prepared_synthetic_artifact(accepted_candidate, state.step_install_generation_ + 1));
  const auto generation = state.step_install_generation_;
  const auto receipt = state.artifact_publication_receipt()->metadata.artifact_identity;

  pops::runtime::program::PreparedProgramInstallation refused(
      make_synthetic_owner_with_unplanned_ceiling(refused_candidate,
                                                  state.step_install_generation_ + 1));
  const std::vector<pops::runtime::program::ProgramInstallationTables::ResourcePrototype>
      over_budget{
          {0,
           0,
           {2, 1, 1, 0, 2, 2},
           pops::runtime::program::ProgramInstallationTables::ResourcePrototypeKind::hot_snapshot}};
  EXPECT_THROW(refused.seal_resource_plan(over_budget), std::invalid_argument);
  EXPECT_EQ(state.step_install_generation_, generation);
  ASSERT_TRUE(state.artifact_publication_receipt());
  EXPECT_EQ(state.artifact_publication_receipt()->metadata.artifact_identity, receipt);
  state.step_(0.25);
  EXPECT_EQ(accepted_candidate.step_calls, 1);
  EXPECT_EQ(refused_candidate.step_calls, 0);
}

TEST(ProgramRuntimeStateArtifactOwner, PreparedAmrArtifactDispatchesEveryLifecycleHook) {
  using State = pops::runtime::program::ProgramRuntimeState<2>;
  using Remap = pops::runtime::program::AmrProgramHistoryRemapDescriptor;
  SyntheticProgramCandidate candidate;
  State state;
  state.install_prepared_artifact(
      prepared_synthetic_artifact(candidate, state.step_install_generation_ + 1, true, true));
  state.artifact_backed_ = true;

  Remap descriptor;
  descriptor.parent_level = 0;
  descriptor.child_level = 1;
  descriptor.prior_topology_epoch = 3;
  descriptor.prior_materialization_generation = 7;
  descriptor.published_topology_epoch = 4;
  descriptor.published_materialization_generation = 8;
  descriptor.accepted_macro_step = 2;
  descriptor.operation_identity = "synthetic-remap";

  state.refresh_hierarchy_state("Fixture");
  state.accept_history_remap(descriptor, "Fixture");
  state.preflight_regrid_on_restart("Fixture");
  state.regrid_on_restart("Fixture");
  state.resync_after_restart("Fixture");
  auto snapshot = state.capture_accepted_context_snapshot("Fixture");

  EXPECT_EQ(candidate.hierarchy_refresh_calls, 1);
  EXPECT_EQ(candidate.history_remap_calls, 1);
  EXPECT_EQ(candidate.restart_preflight_calls, 1);
  EXPECT_EQ(candidate.restart_regrid_calls, 1);
  EXPECT_EQ(candidate.restart_resync_calls, 1);
  EXPECT_EQ(candidate.snapshot_create_calls, 1);
  ASSERT_TRUE(snapshot);
  snapshot.reset();
  EXPECT_EQ(candidate.snapshot_destroy_calls, 1);
}

TEST(ProgramRuntimeStateArtifactOwner,
     PreparedPublicationRejectsStaleGenerationWithoutTouchingAcceptedOwner) {
  using State = pops::runtime::program::ProgramRuntimeState<2>;
  SyntheticProgramCandidate old_candidate;
  SyntheticProgramCandidate replacement_candidate;
  State state;
  auto old_artifact =
      prepared_synthetic_artifact(old_candidate, state.step_install_generation_ + 1, true, true);
  state.install_prepared_artifact(std::move(old_artifact));
  const auto accepted_generation = state.step_install_generation_;

  auto stale = prepared_synthetic_artifact(replacement_candidate, accepted_generation, true, true);
  EXPECT_THROW(
      (void)State::PreparedArtifactPublication::prepare(std::move(stale), accepted_generation + 1),
      std::invalid_argument);
  state.refresh_hierarchy_state("Fixture");

  EXPECT_EQ(old_candidate.hierarchy_refresh_calls, 1);
  EXPECT_EQ(replacement_candidate.hierarchy_refresh_calls, 0);
  EXPECT_EQ(old_candidate.destroy_calls, 0);
  EXPECT_EQ(state.step_install_generation_, accepted_generation);
}

TEST(ProgramRuntimeStateArtifactOwner,
     AcceptedSnapshotRetainsArtifactUntilItsForeignDestructorRuns) {
  using State = pops::runtime::program::ProgramRuntimeState<2>;
  SyntheticProgramCandidate old_candidate;
  SyntheticProgramCandidate replacement_candidate;
  State state;
  auto old_artifact =
      prepared_synthetic_artifact(old_candidate, state.step_install_generation_ + 1, true, true);
  state.install_prepared_artifact(std::move(old_artifact));
  state.artifact_backed_ = true;
  auto snapshot = state.capture_accepted_context_snapshot("Fixture");

  state.install_prepared_artifact(
      prepared_synthetic_artifact(replacement_candidate, state.step_install_generation_ + 1));
  EXPECT_EQ(old_candidate.destroy_calls, 0);
  snapshot.reset();

  EXPECT_EQ(old_candidate.snapshot_destroy_calls, 1);
  EXPECT_EQ(old_candidate.destroy_calls, 1);
}

TEST(ProgramRuntimeStateArtifactOwner, SuccessfulReplacementReleasesOldOwnerAfterCommit) {
  using State = pops::runtime::program::ProgramRuntimeState<3>;
  SyntheticProgramCandidate old_candidate;
  SyntheticProgramCandidate replacement_candidate;
  State state;
  auto old_artifact =
      prepared_synthetic_artifact(old_candidate, state.step_install_generation_ + 1);
  state.install_prepared_artifact(std::move(old_artifact));

  {
    auto replacement = prepared_synthetic_artifact(replacement_candidate,
                                                   state.step_install_generation_ + 1, false);
    auto publication = State::PreparedArtifactPublication::prepare(
        std::move(replacement), state.step_install_generation_ + 1);
    EXPECT_EQ(old_candidate.destroy_calls, 0);
    state.publish_prepared_artifact(std::move(publication));
    // The noexcept exchange keeps the retired owner in the caller's publication image until all
    // of its retired closures are destroyed at this scope boundary.
    EXPECT_EQ(old_candidate.destroy_calls, 0);
  }
  EXPECT_EQ(old_candidate.destroy_calls, 1);
  state.step_(0.2);
  EXPECT_EQ(replacement_candidate.step_calls, 1);
  EXPECT_FALSE(state.dt_bound_);
}

TEST(ProgramRuntimeStateArtifactOwner, TeardownDestroysCandidateBeforeReleasingItsOwner) {
  SyntheticProgramCandidate candidate;
  {
    pops::runtime::program::ProgramRuntimeState<2> state;
    auto artifact = prepared_synthetic_artifact(candidate, state.step_install_generation_ + 1);
    state.install_prepared_artifact(std::move(artifact));
    EXPECT_EQ(candidate.destroy_calls, 0);
  }
  EXPECT_EQ(candidate.destroy_calls, 1);
}

TEST(ProgramPersistentValueStore, BindIsAtomicAndInvalidSlotsRetainTheirTemporalWindow) {
  using namespace pops::runtime::program;
  ProgramResourcePlanEntry row;
  row.slot = 0;
  row.key = {17, 23, 0, 0, 0, -1};
  row.identity = "program-resource:v1:17";
  row.occurrence_path = "root/0";
  row.owner_identity = "block:plasma";
  row.space_identity = "state";
  row.clock_identity = "clock.macro";
  row.communication = "none";
  row.lifetime = ProgramValueLifetime::persistent_schedule;
  row.centering = ProgramValueCentering::cell;
  row.off_policy = ProgramScheduleOffPolicy::accumulate_dt;
  row.components = 1;
  row.bytes = 8;
  row.maximum_bytes = 16;
  row.transfer_identity = "none";
  row.restart_identity = "none";
  row.component_names = "[]";
  row.shape = "[]";
  ProgramResourcePlan plan({row}, 16, "program-resource-plan:v1",
                           "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");

  ProgramPersistentValueStore store;
  store.bind(plan);
  ASSERT_TRUE(store.bound());
  ASSERT_EQ(store.size(), 1u);
  auto& metadata = store.metadata(0);
  metadata.valid = false;
  metadata.cold = true;
  metadata.accumulated_dt = 0.125;
  metadata.accepted_coordinate = 9;
  metadata.cursor = 4;
  metadata.topology_epoch = 5;
  metadata.layout_generation = 6;
  const auto accepted = store.snapshot();

  metadata.accumulated_dt = 99.0;
  store.restore(accepted);
  EXPECT_FALSE(store.metadata(0).valid);
  EXPECT_TRUE(store.metadata(0).cold);
  EXPECT_DOUBLE_EQ(store.metadata(0).accumulated_dt, 0.125);
  EXPECT_EQ(store.metadata(0).accepted_coordinate, 9u);
  EXPECT_EQ(store.metadata(0).cursor, 4u);
  EXPECT_EQ(store.metadata(0).topology_epoch, 5u);
  EXPECT_EQ(store.metadata(0).layout_generation, 6u);

  auto malformed = row;
  malformed.slot = 1;
  EXPECT_THROW(
      (void)ProgramResourcePlan({malformed}, 16, "program-resource-plan:v1",
                                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
      std::invalid_argument);
  // The rejected preparation did not clobber the accepted image.
  EXPECT_TRUE(store.bound());
  EXPECT_DOUBLE_EQ(store.metadata(0).accumulated_dt, 0.125);
}

TEST(ProgramPersistentValueCheckpoint, LosslessRoundTripAndNoAllocationAfterPreparedSwap) {
  using namespace pops::runtime::program;
  ProgramResourcePlanEntry row;
  row.slot = 0;
  row.key = {71, 91, 2, 3, 4, 5};
  row.identity = "program-resource:v1:71";
  row.occurrence_path = "root/branch/loop/0";
  row.owner_identity = "block:plasma";
  row.space_identity = "state";
  row.clock_identity = "clock.macro";
  row.communication = "mpi";
  row.lifetime = ProgramValueLifetime::persistent_schedule;
  row.centering = ProgramValueCentering::face;
  row.off_policy = ProgramScheduleOffPolicy::accumulate_dt;
  row.spatial_transfer = ProgramSpatialTransferPolicy::redistribute_exact;
  row.components = 2;
  row.ghosts = 1;
  row.bytes = 8;
  row.maximum_bytes = 16;
  row.communicates = true;
  row.restart_required = true;
  row.transfer_identity = "redistribute_exact:v1";
  row.restart_identity = "restart:v1";
  row.component_names = "[\"u\",\"v\"]";
  row.shape = "[4,1]";
  row.cells = 4;
  row.itemsize = 8;
  const std::string digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  const ProgramResourcePlan plan({row}, 16, "program-resource-plan:v1", digest);

  ProgramPersistentValueStore source;
  source.bind(plan);
  for (std::size_t index = 0; index != source.value(0).size(); ++index)
    source.value(0)[index] = static_cast<std::byte>(index + 1);
  auto& metadata = source.metadata(0);
  metadata.accepted_coordinate = 17;
  metadata.cursor = 19;
  metadata.accumulated_dt = 0.375;
  metadata.topology_epoch = 23;
  metadata.layout_generation = 29;
  metadata.valid = true;
  metadata.cold = false;

  const auto image = capture_program_persistent_value_checkpoint(source);
  EXPECT_EQ(image.slot_count, 1u);
  ASSERT_EQ(image.rows, plan.entries());
  ASSERT_EQ(image.value_bytes, std::vector<std::uint64_t>{8});
  ASSERT_EQ(image.offsets, std::vector<std::uint64_t>({0, 16}));
  const auto encoded = serialize_program_persistent_value_checkpoint(image);
  const auto decoded = deserialize_program_persistent_value_checkpoint(encoded);
  EXPECT_EQ(decoded, image);

  auto prepared = prepare_program_persistent_value_restore(decoded, plan);
  const auto* const prepared_storage = prepared.value(0).data();
  source.metadata(0).accumulated_dt = 99.0;
  source.value(0)[0] = std::byte{0};
  publish_program_persistent_value_restore(source, std::move(prepared));
  EXPECT_EQ(source.value(0).data(), prepared_storage);
  EXPECT_EQ(source.value(0)[0], std::byte{1});
  EXPECT_DOUBLE_EQ(source.metadata(0).accumulated_dt, 0.375);
  EXPECT_EQ(source.metadata(0).topology_epoch, 23u);
  EXPECT_EQ(source.snapshot().plan_entries, plan.entries());

  auto tampered = encoded;
  tampered[10] ^= 1U;
  EXPECT_THROW((void)deserialize_program_persistent_value_checkpoint(tampered),
               std::invalid_argument);
  auto malformed = image;
  malformed.offsets[1] = 8;
  EXPECT_THROW((void)serialize_program_persistent_value_checkpoint(malformed),
               std::invalid_argument);
}

TEST(ProgramPersistentValueCheckpoint, QualifiedRegridRequiresProviderBeforePublication) {
  using namespace pops::runtime::program;
  ProgramResourcePlanEntry row;
  row.slot = 0;
  row.key = {3, 5, 0, 0, 0, -1};
  row.identity = "program-resource:v1:3";
  row.occurrence_path = "root/0";
  row.owner_identity = "block";
  row.space_identity = "state";
  row.clock_identity = "macro";
  row.communication = "none";
  row.lifetime = ProgramValueLifetime::persistent_schedule;
  row.centering = ProgramValueCentering::cell;
  row.off_policy = ProgramScheduleOffPolicy::hold;
  row.spatial_transfer = ProgramSpatialTransferPolicy::qualified_regrid_provider;
  row.components = 1;
  row.ghosts = 0;
  row.bytes = 8;
  row.maximum_bytes = 8;
  row.transfer_identity = "regrid:provider:v1";
  row.restart_identity = "restart:v1";
  row.component_names = "[\"u\"]";
  row.shape = "[1]";
  const ProgramResourcePlan plan(
      {row}, 8, "program-resource-plan:v1",
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789");
  ProgramPersistentValueStore store;
  store.bind(plan);
  const auto image = capture_program_persistent_value_checkpoint(store);
  EXPECT_THROW((void)prepare_program_persistent_value_regrid(image, plan), std::invalid_argument);
  EXPECT_TRUE(store.bound());
  EXPECT_EQ(store.size(), 1u);
}

TEST(ProgramPersistentValueCheckpoint, RejectsCollisionUnknownBytesOverflowAndDigestMismatch) {
  using namespace pops::runtime::program;
  const std::string digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  auto valid_row = [](std::uint32_t slot, std::uint64_t value_id, std::uint64_t path_id,
                      std::string path) {
    ProgramResourcePlanEntry row;
    row.slot = slot;
    row.key = {value_id, path_id, 0, 0, 0, -1};
    row.identity = "program-resource:v1:" + std::to_string(value_id);
    row.occurrence_path = std::move(path);
    row.owner_identity = "block";
    row.space_identity = "state";
    row.clock_identity = "macro";
    row.communication = "none";
    row.lifetime = ProgramValueLifetime::persistent_schedule;
    row.centering = ProgramValueCentering::cell;
    row.off_policy = ProgramScheduleOffPolicy::hold;
    row.spatial_transfer = ProgramSpatialTransferPolicy::redistribute_exact;
    row.components = 1;
    row.bytes = 8;
    row.maximum_bytes = 8;
    row.transfer_identity = "exact:v1";
    row.restart_identity = "restart:v1";
    row.component_names = "[\"u\"]";
    row.shape = "[1]";
    return row;
  };
  const auto row = valid_row(0, 1, 2, "root/0");
  const ProgramResourcePlan plan({row}, 8, "program-resource-plan:v1", digest);
  ProgramPersistentValueStore store;
  store.bind(plan);
  const auto image = capture_program_persistent_value_checkpoint(store);

  auto unknown_bytes = image;
  unknown_bytes.rows[0].bytes = 0;
  EXPECT_THROW(validate_program_persistent_value_checkpoint(unknown_bytes), std::invalid_argument);
  const ProgramResourcePlan different_plan(
      {row}, 8, "program-resource-plan:v1",
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789");
  EXPECT_THROW((void)prepare_program_persistent_value_restore(image, different_plan),
               std::invalid_argument);

  auto duplicate = valid_row(1, 1, 2, "root/0");
  EXPECT_THROW((void)ProgramResourcePlan({row, duplicate}, 16, "program-resource-plan:v1", digest),
               std::invalid_argument);
  auto collision = valid_row(1, 9, 2, "root/other");
  EXPECT_THROW((void)ProgramResourcePlan({row, collision}, 16, "program-resource-plan:v1", digest),
               std::invalid_argument);
  auto overflowing = valid_row(1, 9, 7, "root/1");
  overflowing.bytes = 1;
  overflowing.maximum_bytes = std::numeric_limits<std::uint64_t>::max();
  EXPECT_THROW((void)ProgramResourcePlan({overflowing, overflowing},
                                         std::numeric_limits<std::uint64_t>::max(),
                                         "program-resource-plan:v1", digest),
               std::overflow_error);
}

TEST(ProgramInstallationTables, EmptyPlanStillAuthenticatesItsManifestAndDigest) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  const std::string payload = tables.canonical_resource_digest_payload(0);
  const std::string digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  ProgramInstallationMetadata metadata;
  metadata.persistent_resource_manifest =
      "{\"resource_plan\":" + tables.canonical_resource_manifest(0, digest) +
      ",\"resource_plan_digest\":\"" + digest + "\"}";
  EXPECT_NO_THROW(tables.validate_resource_authority(metadata, 0));
  auto plan = make_program_resource_plan(tables, 0);
  EXPECT_TRUE(plan.entries().empty());
  EXPECT_EQ(plan.digest(), digest);
  metadata.persistent_resource_manifest = "{}";
  EXPECT_THROW(tables.validate_resource_authority(metadata, 0), std::invalid_argument);
}
