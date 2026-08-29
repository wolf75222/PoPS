#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/system.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

constexpr int kDim = pops::kNativeDimension;

void ensure_kokkos() {
#if defined(POPS_HAS_KOKKOS)
  static std::optional<Kokkos::ScopeGuard> guard;
  if (!Kokkos::is_initialized())
    guard.emplace();
#endif
}

template <class Config>
Config config_with_shape(int cells) {
  Config config;
  for (int axis = 0; axis < kDim; ++axis) {
    config.shape[axis] = cells;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = false;
  }
  return config;
}

void expect_same_services(const pops::runtime::program::ProgramHostDescriptor& first,
                          const pops::runtime::program::ProgramHostDescriptor& second) {
  const auto& left = first.services;
  const auto& right = second.services;
  EXPECT_EQ(left.state_store, right.state_store);
  EXPECT_EQ(left.field_store, right.field_store);
  EXPECT_EQ(left.spatial_executor, right.spatial_executor);
  EXPECT_EQ(left.hierarchy_executor, right.hierarchy_executor);
  EXPECT_EQ(left.history_store, right.history_store);
  EXPECT_EQ(left.clock_service, right.clock_service);
  EXPECT_EQ(left.reduction_service, right.reduction_service);
  EXPECT_EQ(left.transaction_service, right.transaction_service);
  EXPECT_EQ(left.persistent_value_store, right.persistent_value_store);
}

class TestPreparationImage final : public pops::runtime::program::ProgramPreparationImage {
 public:
  TestPreparationImage(std::uint32_t dimension, pops::runtime::program::ProgramRuntimeKind kind,
                       pops::runtime::program::ProgramExecutionServicesRef services,
                       std::uint64_t generation = 1)
      : ProgramPreparationImage(dimension, kind, services, generation) {
    void* const adapter = static_cast<void*>(this);
    bind_image_services(
        {adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter, adapter});
  }
};

struct PreparedInstallationCandidate final {
  int prepare_calls = 0;
  int destroy_calls = 0;
  int service_token = 0;
};

void prepared_installation_step(void*, double) {}

bool prepared_installation_prepare(void* opaque,
                                   const pops::runtime::program::ProgramHostDescriptor*,
                                   pops::runtime::program::ProgramInstallDiagnostic*) noexcept {
  ++static_cast<PreparedInstallationCandidate*>(opaque)->prepare_calls;
  return true;
}

void prepared_installation_destroy(void* opaque) noexcept {
  ++static_cast<PreparedInstallationCandidate*>(opaque)->destroy_calls;
}

pops::runtime::program::ProgramHostDescriptor prepared_installation_host(int& service_token) {
  using namespace pops::runtime::program;
  ProgramHostDescriptor host{};
  host.native_dimension = 2;
  host.runtime_kind = ProgramRuntimeKind::uniform;
  host.capability_bits = kKnownProgramCapabilityBits;
  host.services = {&service_token, &service_token, &service_token, &service_token, &service_token,
                   &service_token, &service_token, &service_token, &service_token};
  return host;
}

pops::runtime::program::ProgramInstallationTables::ResourcePlan symbolic_resource_row(
    std::uint32_t slot, std::uint64_t value_id, std::string identity) {
  using namespace pops::runtime::program;
  ProgramInstallationTables::ResourcePlan resource;
  resource.slot = slot;
  resource.flags = kProgramResourceRuntimeSized;
  resource.value_id = value_id;
  resource.occurrence_path_id = value_id + 100;
  resource.level = -1;
  resource.components = 2;
  resource.ghosts = 1;
  resource.resource_type = ProgramResourcePlanType::runtime_sized;
  resource.schema = "program-resource-plan:v1";
  resource.plan_digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  resource.identity = std::move(identity);
  resource.occurrence_path = "root/" + std::to_string(slot);
  resource.owner = "block";
  resource.space = "cell";
  resource.clock = "macro";
  resource.lifetime = "transient";
  resource.centering = "cell";
  resource.off_policy = "none";
  resource.communication = "halo_exchange";
  resource.transfer_provider = "none";
  resource.restart_provider = "none";
  resource.component_names = "[]";
  resource.shape = "[]";
  return resource;
}

void authenticate_symbolic_resource_metadata(
    pops::runtime::program::ProgramInstallationTables& tables,
    pops::runtime::program::ProgramInstallationMetadata& metadata) {
  using namespace pops::runtime::program;
  const std::string payload = tables.canonical_resource_digest_payload(std::nullopt);
  const std::string digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  for (auto& row : tables.resource_plan)
    row.plan_digest = digest;
  metadata.persistent_resource_manifest =
      "{\"resource_plan\":" + tables.canonical_resource_manifest(std::nullopt, digest) +
      ",\"resource_plan_digest\":\"" + digest + "\",\"resource_plan_maximum_bytes\":null}";
}

pops::runtime::program::ProgramCandidateDescriptor prepared_installation_descriptor(
    PreparedInstallationCandidate& candidate) {
  using namespace pops::runtime::program;
  ProgramCandidateDescriptor descriptor{};
  descriptor.native_dimension = 2;
  descriptor.runtime_kind = ProgramRuntimeKind::uniform;
  descriptor.provided_capability_bits = kKnownProgramCapabilityBits;
  descriptor.maximum_bytes = 64;
  descriptor.context = &candidate;
  descriptor.prepare = &prepared_installation_prepare;
  descriptor.step = &prepared_installation_step;
  descriptor.destroy = &prepared_installation_destroy;
  return descriptor;
}

pops::runtime::program::PreparedProgramInstallation prepared_installation(
    PreparedInstallationCandidate& candidate,
    std::shared_ptr<pops::runtime::program::ProgramPreparationImage>& image) {
  using namespace pops::runtime::program;
  auto host = prepared_installation_host(candidate.service_token);
  image = std::make_shared<TestPreparationImage>(2, ProgramRuntimeKind::uniform, host.services);
  bind_program_preparation_image(host, image);
  EXPECT_NE(host.services.state_store, &candidate.service_token);
  EXPECT_EQ(host.services.state_store, host.preparation.services.state_store);

  ProgramInstallationTables tables;
  ProgramInstallationTables::ResourcePlan resource;
  resource.slot = 0;
  resource.value_id = 7;
  resource.occurrence_path_id = 11;
  resource.level = -1;
  resource.components = 1;
  resource.bytes = 8;
  resource.maximum_bytes = 64;
  resource.schema = "program-resource-plan:v1";
  resource.plan_digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  resource.identity = "resource";
  resource.occurrence_path = "root/0";
  resource.owner = "block";
  resource.space = "cell";
  resource.clock = "macro";
  resource.lifetime = "persistent_schedule";
  resource.centering = "cell";
  resource.off_policy = "hold";
  resource.communication = "none";
  resource.transfer_provider = "redistribute_exact";
  resource.restart_provider = "none";
  resource.component_names = "[\"value\"]";
  resource.shape = "[1]";
  tables.resource_plan.push_back(std::move(resource));

  OwnedProgramInstallation owner(pops::dynlib::UniqueHandle{nullptr},
                                 prepared_installation_descriptor(candidate),
                                 ProgramInstallationMetadata{"artifact", "abi", "route", "boundary",
                                                             "persistent", "checkpoint", "program"},
                                 std::move(tables));
  owner.set_preparation_image(image);
  owner.prepare(host);
  return PreparedProgramInstallation(std::move(owner));
}

TEST(ProgramHostDescriptor, UniformRequiresTaggedPreparationImageBeforeProviderMaterialization) {
  ensure_kokkos();
  pops::System<kDim> system(config_with_shape<pops::SystemConfig<kDim>>(2));
  system.install_prepared_boundary_execution_lane(std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::world("test.program-host-descriptor/uniform@1")));

  const auto first = system.program_host_descriptor();
  const auto second = system.program_host_descriptor();
  EXPECT_FALSE(pops::runtime::program::valid_program_host_descriptor(first));
  EXPECT_EQ(first.runtime_kind, pops::runtime::program::ProgramRuntimeKind::uniform);
  EXPECT_EQ(first.execution_lane, pops::runtime::program::ProgramExecutionLane::host);
  EXPECT_EQ(first.capability_bits, pops::runtime::program::kProgramCapabilitySchedules |
                                       pops::runtime::program::kProgramCapabilityPersistentValues |
                                       pops::runtime::program::kProgramCapabilityTransactions);
  EXPECT_NE(first.services.hierarchy_executor, nullptr);
  expect_same_services(first, second);

  auto tagged = first;
  const auto image = pops::runtime::program::make_program_preparation_image<kDim>(&system, 1);
  pops::runtime::program::bind_program_preparation_image(tagged, image);
  EXPECT_TRUE(pops::runtime::program::valid_program_host_descriptor(tagged));
  EXPECT_NE(tagged.services.state_store, first.services.state_store);
}

TEST(ProgramHostDescriptor, AmrInspectionImageIsTaggedAndNeverProvidesExecutionServices) {
  ensure_kokkos();
  pops::AmrSystem<kDim> system(config_with_shape<pops::AmrSystemConfig<kDim>>(2));
  EXPECT_FALSE(system.uses_runtime_engine());

  const auto first = system.program_host_descriptor();
  const auto second = system.program_host_descriptor();
  EXPECT_FALSE(pops::runtime::program::valid_program_host_descriptor(first));
  EXPECT_EQ(first.runtime_kind, pops::runtime::program::ProgramRuntimeKind::amr);
  EXPECT_EQ(first.execution_lane, pops::runtime::program::ProgramExecutionLane::host);
  EXPECT_EQ(first.capability_bits, pops::runtime::program::kProgramCapabilityHierarchy |
                                       pops::runtime::program::kProgramCapabilitySchedules |
                                       pops::runtime::program::kProgramCapabilityPersistentValues |
                                       pops::runtime::program::kProgramCapabilityTransactions |
                                       pops::runtime::program::kProgramCapabilityCellTemporal);
  expect_same_services(first, second);
  EXPECT_FALSE(system.uses_runtime_engine());

  auto tagged = first;
  const auto image = pops::runtime::program::make_program_inspection_image<kDim>(
      pops::runtime::program::ProgramRuntimeKind::amr, 1);
  pops::runtime::program::bind_program_preparation_image(tagged, image);
  EXPECT_TRUE(pops::runtime::program::valid_program_host_descriptor(tagged));
  EXPECT_NE(tagged.services.state_store, first.services.state_store);
  EXPECT_THROW(
      (void)pops::runtime::program::make_program_execution_provider<kDim>(tagged.preparation),
      std::logic_error);
}

TEST(PreparedProgramInstallation, RetainsOneImmutableDescriptionAndPreparationWitness) {
  using namespace pops::runtime::program;
  using Prepared = PreparedProgramInstallation;
  static_assert(
      std::is_const_v<std::remove_reference_t<decltype(std::declval<Prepared&>().owner())>>);

  PreparedInstallationCandidate candidate;
  std::shared_ptr<ProgramPreparationImage> image;
  auto prepared = prepared_installation(candidate, image);

  EXPECT_TRUE(prepared.prepared());
  EXPECT_TRUE(prepared.owner().prepared());
  EXPECT_EQ(candidate.prepare_calls, 1);
  EXPECT_EQ(&prepared.metadata(), &prepared.owner().metadata());
  EXPECT_EQ(&prepared.tables(), &prepared.owner().tables());
  EXPECT_EQ(prepared.metadata().artifact_identity, "artifact");
  ASSERT_EQ(prepared.tables().resource_plan.size(), 1u);
  EXPECT_EQ(&prepared.resource_plan(), &prepared.tables().resource_plan);
  EXPECT_EQ(prepared.resource_plan().front().identity, "resource");
  EXPECT_EQ(prepared.resource_ceiling(), 64u);
  EXPECT_EQ(prepared.maximum_bytes(), 64u);
  EXPECT_EQ(prepared.generation(), image->generation());
  ASSERT_EQ(prepared.sealed_resource_plan().entries().size(), 1u);
  EXPECT_EQ(prepared.sealed_resource_plan().entries().front().slot, 0u);
  EXPECT_TRUE(prepared.persistent_values().bound());
  EXPECT_EQ(candidate.destroy_calls, 0);
}

TEST(ProgramResourceMaterializer, AggregatesExactBytesAcrossAllSubslots) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  tables.resource_plan.push_back(symbolic_resource_row(0, 7, "runtime"));
  std::vector<ProgramInstallationTables::ResourcePrototype> prototypes{
      {0,
       0,
       {10, 8, 2, 1, std::nullopt, 200},
       ProgramInstallationTables::ResourcePrototypeKind::state},
      {0,
       0,
       {1, 8, 11, 1, std::nullopt, 96},
       ProgramInstallationTables::ResourcePrototypeKind::scalar},
  };

  const auto plan = tables.materialize_resource_plan(
      std::span<const ProgramInstallationTables::ResourcePrototype>(prototypes));
  ASSERT_EQ(plan.entries().size(), 1u);
  EXPECT_EQ(plan.entries().front().bytes, 248u);
  EXPECT_EQ(plan.entries().front().maximum_bytes, 296u);
  EXPECT_EQ(plan.maximum_bytes(), 296u);
  EXPECT_FALSE(plan.entries().front().cells.has_value());
  EXPECT_FALSE(plan.entries().front().itemsize.has_value());
  EXPECT_EQ(plan.digest().size(), 64u);
}

TEST(ProgramResourceMaterializer, RefusesUnresolvedDuplicateOverflowAndBudgetViolations) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  tables.resource_plan.push_back(symbolic_resource_row(0, 7, "runtime"));
  EXPECT_THROW(tables.materialize_resource_plan(
                   std::span<const ProgramInstallationTables::ResourcePrototype>{}),
               std::invalid_argument);

  const ProgramInstallationTables::ResourcePrototype exact{
      0,
      0,
      {2, 8, 2, 1, std::nullopt, 32},
      ProgramInstallationTables::ResourcePrototypeKind::state};
  EXPECT_THROW(tables.materialize_resource_plan(
                   std::vector<ProgramInstallationTables::ResourcePrototype>{exact, exact}),
               std::invalid_argument);

  const ProgramInstallationTables::ResourcePrototype overflowing{
      0,
      0,
      {std::numeric_limits<std::uint64_t>::max(), 2, 2, 1, std::nullopt, std::nullopt},
      ProgramInstallationTables::ResourcePrototypeKind::state};
  EXPECT_THROW(tables.materialize_resource_plan(
                   std::vector<ProgramInstallationTables::ResourcePrototype>{overflowing}),
               std::overflow_error);

  EXPECT_THROW(tables.materialize_resource_plan(
                   std::vector<ProgramInstallationTables::ResourcePrototype>{exact}, 31),
               std::invalid_argument);
}

TEST(ProgramResourceMaterializer, MergesRankLocalLayoutsByFamilyMaximum) {
  using namespace pops::runtime::program;
  const auto state = ProgramInstallationTables::ResourcePrototypeKind::state;
  const auto scalar = ProgramInstallationTables::ResourcePrototypeKind::scalar;
  const std::vector<ProgramInstallationTables::ResourcePrototype> rank_zero{
      {0, 0, {10, 8, 2, 1, std::nullopt, 160, {10, 1}}, state},
      {0, 0, {2, 8, 11, 1, std::nullopt, 176, {1, 2}}, scalar},
  };
  const std::vector<ProgramInstallationTables::ResourcePrototype> rank_one{
      // A rank with no local fab contributes no row for state, while its scalar family is larger.
      {0, 0, {0, 0, 0, 0, std::uint64_t{0}, std::uint64_t{0}, {10, 1}}, state},
      {0, 0, {4, 8, 11, 1, std::nullopt, 352, {1, 2}}, scalar},
  };
  const std::vector<std::vector<ProgramInstallationTables::ResourcePrototype>> ranks{rank_zero,
                                                                                     rank_one};
  const auto merged = ProgramInstallationTables::merge_resource_prototypes(ranks);
  ASSERT_EQ(merged.size(), 2u);
  EXPECT_EQ(merged[0].kind, scalar);
  EXPECT_EQ(merged[0].layout.cells, 4u);
  EXPECT_EQ(merged[0].layout.bytes, 352u);
  EXPECT_EQ(merged[0].layout.maximum_bytes, 352u);
  EXPECT_EQ(merged[1].kind, state);
  EXPECT_EQ(merged[1].layout.cells, 10u);
  EXPECT_EQ(merged[1].layout.bytes, 160u);
  EXPECT_EQ(merged[1].layout.shape, (std::vector<std::uint64_t>{10, 1}));

  ProgramInstallationTables tables;
  tables.resource_plan.push_back(symbolic_resource_row(0, 7, "runtime"));
  const auto plan = tables.materialize_resource_plan(merged);
  EXPECT_EQ(plan.entries().front().bytes, 512u);
  EXPECT_EQ(plan.entries().front().maximum_bytes, 512u);
}

TEST(ProgramResourceMaterializer, NeverProjectsSymbolicRowToTheFormerEightByteFallback) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  tables.resource_plan.push_back(symbolic_resource_row(0, 7, "runtime"));
  EXPECT_THROW(make_program_resource_plan(tables, kProgramResourcePlanUnknownExtent),
               std::invalid_argument);
  EXPECT_THROW(tables.materialize_resource_plan(
                   std::span<const ProgramInstallationTables::ResourcePrototype>{}),
               std::invalid_argument);
}

TEST(PreparedProgramInstallation, SymbolicPlanMustBeExplicitlySealedAfterPreparation) {
  using namespace pops::runtime::program;
  PreparedInstallationCandidate candidate;
  auto host = prepared_installation_host(candidate.service_token);
  auto image =
      std::make_shared<TestPreparationImage>(2, ProgramRuntimeKind::uniform, host.services);
  bind_program_preparation_image(host, image);

  ProgramInstallationTables tables;
  tables.resource_plan.push_back(symbolic_resource_row(0, 7, "runtime"));
  ProgramInstallationMetadata metadata{"artifact", "abi",        "route",  "boundary",
                                       "",         "checkpoint", "program"};
  authenticate_symbolic_resource_metadata(tables, metadata);
  auto descriptor = prepared_installation_descriptor(candidate);
  descriptor.maximum_bytes = kProgramResourcePlanUnknownExtent;
  OwnedProgramInstallation owner(pops::dynlib::UniqueHandle{nullptr}, descriptor,
                                 std::move(metadata), std::move(tables));
  owner.set_preparation_image(image);
  owner.prepare(host);
  PreparedProgramInstallation prepared(std::move(owner));
  EXPECT_FALSE(prepared.resource_plan_sealed());
  EXPECT_EQ(prepared.resource_slot_count(), 1u);
  EXPECT_TRUE(prepared.resource_declarations().front().runtime_sized());
  EXPECT_THROW(std::move(prepared).release_publication_payload(), std::logic_error);
  const std::vector<ProgramInstallationTables::ResourcePrototype> prototypes{
      {0,
       0,
       {5, 8, 2, 1, std::nullopt, 80},
       ProgramInstallationTables::ResourcePrototypeKind::state}};
  prepared.seal_resource_plan(prototypes);
  EXPECT_TRUE(prepared.resource_plan_sealed());
  EXPECT_EQ(prepared.sealed_resource_plan().entries().front().bytes, 80u);
  EXPECT_EQ(prepared.resource_plan().front().bytes, std::optional<std::uint64_t>{80});
  EXPECT_EQ(prepared.resource_plan().front().maximum_bytes, std::optional<std::uint64_t>{80});
  EXPECT_EQ(prepared.resource_plan().front().resource_type, ProgramResourcePlanType::exact);
  EXPECT_EQ(prepared.resource_plan().front().flags & kProgramResourceRuntimeSized, 0u);
  EXPECT_NE(prepared.metadata().persistent_resource_manifest.find("\"maximum_bytes\":80"),
            std::string::npos);
  EXPECT_NE(prepared.metadata().persistent_resource_manifest.find("\"prepared_layouts\":"),
            std::string::npos);
  EXPECT_EQ(prepared.resource_ceiling(), 80u);
  EXPECT_TRUE(prepared.persistent_values().bound());
}

TEST(PreparedProgramInstallation, RefusesAnUnversionedPreparationImage) {
  using namespace pops::runtime::program;
  PreparedInstallationCandidate candidate;
  auto host = prepared_installation_host(candidate.service_token);
  EXPECT_THROW((void)std::make_shared<TestPreparationImage>(2, ProgramRuntimeKind::uniform,
                                                            host.services, 0),
               std::invalid_argument);
}

TEST(PreparedProgramInstallation, RefusesAnOwnerThatWasNotPrepared) {
  using namespace pops::runtime::program;
  PreparedInstallationCandidate candidate;
  const auto descriptor = prepared_installation_descriptor(candidate);
  OwnedProgramInstallation owner(
      pops::dynlib::UniqueHandle{nullptr}, descriptor,
      ProgramInstallationMetadata{"artifact", "abi", "route", "boundary", "persistent",
                                  "checkpoint", "program"});

  EXPECT_THROW((void)PreparedProgramInstallation(std::move(owner)), std::invalid_argument);
  EXPECT_EQ(candidate.destroy_calls, 1);
}

TEST(PreparedProgramInstallation, ExplicitOwnerTransferPreservesPreparedStateUntilDestruction) {
  PreparedInstallationCandidate candidate;
  std::shared_ptr<pops::runtime::program::ProgramPreparationImage> image;
  auto prepared = prepared_installation(candidate, image);

  auto owner = std::move(prepared).release_owner();
  EXPECT_TRUE(owner.prepared());
  EXPECT_EQ(owner.metadata().artifact_identity, "artifact");
  ASSERT_EQ(owner.tables().resource_plan.size(), 1u);
  EXPECT_EQ(owner.tables().resource_plan.front().identity, "resource");
  EXPECT_EQ(candidate.destroy_calls, 0);

  owner.reset();
  EXPECT_EQ(candidate.destroy_calls, 1);
}

}  // namespace
