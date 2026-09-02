#include <gtest/gtest.h>

#include "program_v5_fixture.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/system.hpp>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
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

std::atomic<bool> g_seal_allocation_fault_enabled{false};
std::atomic<std::uint64_t> g_seal_allocation_fault_after{0};
std::atomic<std::uint64_t> g_seal_allocation_attempts{0};

void* seal_test_allocate(std::size_t size) {
  if (g_seal_allocation_fault_enabled.load(std::memory_order_relaxed) &&
      g_seal_allocation_attempts.fetch_add(1, std::memory_order_relaxed) + 1 ==
          g_seal_allocation_fault_after.load(std::memory_order_relaxed))
    throw std::bad_alloc();
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  return pointer;
}

class SealAllocationFault final {
 public:
  explicit SealAllocationFault(std::uint64_t fail_after) {
    g_seal_allocation_attempts.store(0, std::memory_order_relaxed);
    g_seal_allocation_fault_after.store(fail_after, std::memory_order_relaxed);
    g_seal_allocation_fault_enabled.store(true, std::memory_order_relaxed);
  }
  SealAllocationFault(const SealAllocationFault&) = delete;
  SealAllocationFault& operator=(const SealAllocationFault&) = delete;
  ~SealAllocationFault() { close(); }

  void close() noexcept { g_seal_allocation_fault_enabled.store(false, std::memory_order_relaxed); }
};

}  // namespace

void* operator new(std::size_t size) {
  return seal_test_allocate(size);
}
void* operator new[](std::size_t size) {
  return seal_test_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return seal_test_allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return ::operator new(size, std::nothrow);
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
  void* pointer = nullptr;
  if (posix_memalign(&pointer, static_cast<std::size_t>(alignment), size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  return pointer;
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return ::operator new(size, alignment, std::nothrow);
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
  const std::string payload = tables.canonical_resource_digest_payload(std::uint64_t{64});
  const std::string digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  tables.resource_plan.front().plan_digest = digest;
  const std::string manifest = tables.canonical_resource_manifest(std::uint64_t{64}, digest);

  OwnedProgramInstallation owner(
      pops::dynlib::UniqueHandle{nullptr}, prepared_installation_descriptor(candidate),
      ProgramInstallationMetadata{"artifact", "abi", "route", "boundary",
                                  "{\"resource_plan\":" + manifest +
                                      ",\"resource_plan_digest\":\"" + digest +
                                      "\",\"resource_plan_maximum_bytes\":64}",
                                  "checkpoint", "program"},
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
  EXPECT_FALSE(prepared.resource_plan_sealed());
  EXPECT_THROW((void)prepared.resource_ceiling(), std::logic_error);
  prepared.seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype>{});
  EXPECT_EQ(prepared.resource_ceiling(), 64u);
  EXPECT_EQ(prepared.maximum_bytes(), 64u);
  EXPECT_EQ(prepared.generation(), image->generation());
  ASSERT_EQ(prepared.sealed_resource_plan().entries().size(), 1u);
  EXPECT_EQ(prepared.sealed_resource_plan().entries().front().slot, 0u);
  EXPECT_TRUE(prepared.persistent_values().bound());
  EXPECT_EQ(candidate.destroy_calls, 0);
}

TEST(PreparedProgramInstallation,
     FaultedSealKeepsTheSymbolicAuthorityRetryableAndConsumesTheSealExactlyOnce) {
  using namespace pops::runtime::program;
  bool observed_bad_alloc = false;
  bool completed = false;

  // Run every allocation ordinal through the whole seal path.  This includes the persistent-value
  // bind allocation after materialization; each failed image must leave the live owner symbolic and
  // a retry must still publish the same exact authority.
  for (std::uint64_t fail_after = 1; fail_after != 128 && !completed; ++fail_after) {
    PreparedInstallationCandidate candidate;
    std::shared_ptr<ProgramPreparationImage> image;
    auto prepared = prepared_installation(candidate, image);
    const std::string manifest_before = prepared.metadata().persistent_resource_manifest;
    const auto type_before = prepared.resource_plan().front().resource_type;
    const auto flags_before = prepared.resource_plan().front().flags;
    bool faulted = false;
    {
      SealAllocationFault fault(fail_after);
      try {
        prepared.seal_resource_plan(
            std::span<const ProgramInstallationTables::ResourcePrototype>{});
      } catch (const std::bad_alloc&) {
        faulted = true;
      }
      fault.close();
    }

    if (faulted) {
      observed_bad_alloc = true;
      EXPECT_FALSE(prepared.resource_plan_sealed());
      EXPECT_EQ(prepared.metadata().persistent_resource_manifest, manifest_before);
      EXPECT_EQ(prepared.resource_plan().front().resource_type, type_before);
      EXPECT_EQ(prepared.resource_plan().front().flags, flags_before);
      EXPECT_TRUE(prepared.tables().prepared_layout_manifest().empty());

      prepared.seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype>{});
      EXPECT_TRUE(prepared.resource_plan_sealed());
      EXPECT_TRUE(prepared.persistent_values().bound());
      EXPECT_THROW(prepared.seal_resource_plan(
                       std::span<const ProgramInstallationTables::ResourcePrototype>{}),
                   std::logic_error);
    } else {
      completed = true;
      EXPECT_TRUE(prepared.resource_plan_sealed());
    }
  }

  EXPECT_TRUE(observed_bad_alloc);
  EXPECT_TRUE(completed);
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

TEST(ProgramResourceMaterializer, SealsHostOnlyFamiliesForAnExactEmptyValuePlan) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  const std::vector<ProgramInstallationTables::ResourcePrototype> prototypes{
      {0, 0, {96, 1, 1, 0, 96, 96}, ProgramInstallationTables::ResourcePrototypeKind::hot_snapshot},
      {0, 1, {32, 1, 1, 0, 32, 32}, ProgramInstallationTables::ResourcePrototypeKind::reduction},
      {0,
       2,
       {64, 1, 1, 0, 64, 64},
       ProgramInstallationTables::ResourcePrototypeKind::prepared_coupling},
  };
  const auto plan = tables.materialize_resource_plan(prototypes);
  EXPECT_TRUE(plan.entries().empty());
  EXPECT_EQ(plan.maximum_bytes(), 192u);
  EXPECT_NE(tables.prepared_layout_manifest().find("\"hot_snapshot\""), std::string::npos);
  EXPECT_NE(tables.prepared_layout_manifest().find("\"reduction\""), std::string::npos);
  EXPECT_NE(tables.prepared_layout_manifest().find("\"prepared_coupling\""), std::string::npos);
  EXPECT_THROW(tables.materialize_resource_plan(prototypes, 191), std::invalid_argument);
}

TEST(ProgramResourcePlan, BindsZeroPersistentSlotsForAnExactHostOnlyCeiling) {
  using namespace pops::runtime::program;
  ProgramInstallationTables tables;
  const std::string payload = tables.canonical_resource_digest_payload(std::uint64_t{128});
  const std::string digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  const ProgramResourcePlan plan({}, 128, "program-resource-plan:v1", digest);
  EXPECT_TRUE(plan.entries().empty());
  EXPECT_EQ(plan.slot_count(), 0u);
  EXPECT_EQ(plan.maximum_bytes(), 128u);

  ProgramPersistentValueStore store;
  store.bind(plan);
  EXPECT_TRUE(store.bound());
  EXPECT_EQ(store.size(), 0u);
  EXPECT_EQ(store.maximum_bytes(), 128u);
}

TEST(PreparedProgramInstallation, SealsAnEmptyValuePlanWithHostCapacityExactly) {
  using namespace pops::runtime::program;
  const std::vector<ProgramInstallationTables::ResourcePrototype> prototypes{
      {0, 0, {96, 1, 1, 0, 96, 96}, ProgramInstallationTables::ResourcePrototypeKind::hot_snapshot},
      {0, 1, {32, 1, 1, 0, 32, 32}, ProgramInstallationTables::ResourcePrototypeKind::reduction},
  };
  const auto make_prepared = [&](std::uint64_t maximum_bytes,
                                 std::optional<std::uint64_t> manifest_ceiling) {
    auto candidate = std::make_shared<PreparedInstallationCandidate>();
    auto host = prepared_installation_host(candidate->service_token);
    auto image =
        std::make_shared<TestPreparationImage>(2, ProgramRuntimeKind::uniform, host.services);
    bind_program_preparation_image(host, image);
    ProgramInstallationTables tables;
    const std::string payload = tables.canonical_resource_digest_payload(manifest_ceiling);
    const std::string digest =
        pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
    ProgramInstallationMetadata metadata{
        "artifact",
        "abi",
        "route",
        "boundary",
        "{\"resource_plan\":" + tables.canonical_resource_manifest(manifest_ceiling, digest) +
            ",\"resource_plan_digest\":\"" + digest + "\",\"resource_plan_maximum_bytes\":" +
            (manifest_ceiling ? std::to_string(*manifest_ceiling) : std::string{"null"}) + "}",
        "checkpoint",
        "program"};
    auto descriptor = prepared_installation_descriptor(*candidate);
    descriptor.maximum_bytes = maximum_bytes;
    OwnedProgramInstallation owner(pops::dynlib::UniqueHandle{nullptr}, descriptor,
                                   std::move(metadata), std::move(tables));
    owner.set_preparation_image(image);
    owner.prepare(host);
    // The candidate context must outlive the descriptor in this small direct-construction test.
    return std::pair{PreparedProgramInstallation(std::move(owner)), std::move(candidate)};
  };

  auto symbolic_prepared = make_prepared(kProgramResourcePlanUnknownExtent, std::uint64_t{0});
  auto candidate = std::move(symbolic_prepared.second);
  auto prepared = std::move(symbolic_prepared.first);
  prepared.seal_resource_plan(prototypes);
  EXPECT_TRUE(prepared.resource_plan_sealed());
  EXPECT_TRUE(prepared.sealed_resource_plan().entries().empty());
  EXPECT_EQ(prepared.sealed_resource_plan().maximum_bytes(), 128u);
  EXPECT_NE(prepared.metadata().persistent_resource_manifest.find("\"maximum_bytes\":128"),
            std::string::npos);
  EXPECT_NE(prepared.metadata().persistent_resource_manifest.find("\"prepared_layouts\":"),
            std::string::npos);

  auto exact_zero_prepared = make_prepared(0, std::uint64_t{0});
  auto zero_candidate = std::move(exact_zero_prepared.second);
  auto zero_ceiling = std::move(exact_zero_prepared.first);
  const std::string before = zero_ceiling.metadata().persistent_resource_manifest;
  EXPECT_THROW(zero_ceiling.seal_resource_plan(prototypes), std::invalid_argument);
  EXPECT_FALSE(zero_ceiling.resource_plan_sealed());
  EXPECT_EQ(zero_ceiling.metadata().persistent_resource_manifest, before);

  auto state_free_prepared = make_prepared(0, std::uint64_t{0});
  auto state_free_candidate = std::move(state_free_prepared.second);
  auto state_free = std::move(state_free_prepared.first);
  state_free.seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype>{});
  EXPECT_TRUE(state_free.resource_plan_sealed());
  EXPECT_TRUE(state_free.sealed_resource_plan().entries().empty());
  EXPECT_EQ(state_free.sealed_resource_plan().maximum_bytes(), 0u);
}

TEST(PreparedProgramInstallation, ExactValueRowsRemainSymbolicUntilHostCarriersAreSealed) {
  using namespace pops::runtime::program;
  PreparedInstallationCandidate candidate;
  auto host = prepared_installation_host(candidate.service_token);
  auto image =
      std::make_shared<TestPreparationImage>(2, ProgramRuntimeKind::uniform, host.services);
  bind_program_preparation_image(host, image);
  ProgramInstallationTables tables;
  ProgramInstallationTables::ResourcePlan row;
  row.slot = 0;
  row.value_id = 7;
  row.occurrence_path_id = 11;
  row.level = -1;
  row.components = 1;
  row.bytes = 64;
  row.maximum_bytes = 64;
  row.schema = "program-resource-plan:v1";
  row.identity = "resource";
  row.occurrence_path = "root/0";
  row.owner = "block";
  row.space = "cell";
  row.clock = "macro";
  row.lifetime = "persistent_schedule";
  row.centering = "cell";
  row.off_policy = "hold";
  row.communication = "none";
  row.transfer_provider = "redistribute_exact";
  row.restart_provider = "none";
  row.component_names = "[\"value\"]";
  row.shape = "[8]";
  tables.resource_plan.push_back(std::move(row));
  const std::string payload = tables.canonical_resource_digest_payload(std::uint64_t{64});
  tables.resource_plan.front().plan_digest =
      pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
  const std::string manifest = tables.canonical_resource_manifest(
      std::uint64_t{64}, tables.resource_plan.front().plan_digest);
  ProgramInstallationMetadata metadata{
      "artifact",
      "abi",
      "route",
      "boundary",
      "{\"resource_plan\":" + manifest + ",\"resource_plan_digest\":\"" +
          tables.resource_plan.front().plan_digest + "\",\"resource_plan_maximum_bytes\":64}",
      "checkpoint",
      "program"};
  auto descriptor = prepared_installation_descriptor(candidate);
  descriptor.maximum_bytes = kProgramResourcePlanUnknownExtent;
  OwnedProgramInstallation owner(pops::dynlib::UniqueHandle{nullptr}, descriptor,
                                 std::move(metadata), std::move(tables));
  owner.set_preparation_image(image);
  owner.prepare(host);
  PreparedProgramInstallation prepared(std::move(owner));
  const std::vector<ProgramInstallationTables::ResourcePrototype> prototypes{
      {0, 0, {32, 1, 1, 0, 32, 32}, ProgramInstallationTables::ResourcePrototypeKind::hot_snapshot},
  };
  prepared.seal_resource_plan(prototypes);
  ASSERT_EQ(prepared.sealed_resource_plan().entries().size(), 1u);
  EXPECT_EQ(prepared.sealed_resource_plan().entries().front().maximum_bytes, 64u);
  EXPECT_EQ(prepared.sealed_resource_plan().maximum_bytes(), 96u);
  EXPECT_NE(prepared.metadata().persistent_resource_manifest.find("\"maximum_bytes\":96"),
            std::string::npos);
  EXPECT_NE(
      prepared.metadata().persistent_resource_manifest.find("\"resource_plan_maximum_bytes\":96"),
      std::string::npos);
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

TEST(PreparedProgramInstallation, SealedPayloadConsumesOwnerAndRefusesSecondOrMovedFromRelease) {
  using namespace pops::runtime::program;
  PreparedInstallationCandidate candidate;
  std::shared_ptr<ProgramPreparationImage> image;
  auto prepared = prepared_installation(candidate, image);
  prepared.seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype>{});

  auto moved = std::move(prepared);
  EXPECT_THROW(std::move(prepared).release_publication_payload(), std::logic_error);

  auto payload = std::move(moved).release_publication_payload();
  EXPECT_THROW(std::move(moved).release_publication_payload(), std::logic_error);
  EXPECT_TRUE(payload.owner.prepared());
  EXPECT_EQ(payload.generation, image->generation());
  EXPECT_TRUE(payload.persistent_values.bound());
  EXPECT_EQ(payload.resource_plan.maximum_bytes(), 64u);
  EXPECT_EQ(payload.owner.metadata().artifact_identity, "artifact");
  ASSERT_EQ(payload.owner.tables().resource_plan.size(), 1u);
  EXPECT_EQ(payload.owner.tables().resource_plan.front().identity, "resource");
  EXPECT_EQ(candidate.destroy_calls, 0);

  payload.owner.reset();
  EXPECT_EQ(candidate.destroy_calls, 1);
}

TEST(ProgramV5Fixture, RefusesMalformedFluxBasisAndFinalTermRows) {
  using pops::test::program_v5::CallbackProgramFaceFluxStage;
  using pops::test::program_v5::CallbackProgramFluxBasisOccurrence;
  using pops::test::program_v5::callback_program_source;
  using pops::runtime::program::ProgramFluxBudgetRecord;

  const auto source = [&](std::vector<CallbackProgramFluxBasisOccurrence> bases,
                          std::vector<CallbackProgramFaceFluxStage> terms) {
    return callback_program_source(17, "fixture-flux", "clock.macro", {"tracer"}, {},
                                   "pops_test_program_callback", "amr", {}, {}, {}, {},
                                   std::nullopt, std::nullopt, bases, terms);
  };
  EXPECT_NE(source({}, {}).find("kProgramFluxBudgets[] = {{0ULL, 0ULL, 0ULL, 0ULL}}"),
            std::string::npos);
  CallbackProgramFluxBasisOccurrence basis;
  basis.basis_slot = 0;
  basis.expression_slot = 1;
  basis.block = 0;
  basis.level = 0;
  basis.rhs_identity = 3000;
  basis.provider = pops::test::program_v5::kPreparedDefaultFluxProvider;
  basis.identity = "fixture-flux/basis/0";
  basis.occurrence_path = "fixture-flux/loop/basis/0";
  basis.owner = "tracer";
  basis.clock = "clock.macro";

  auto malformed_basis = basis;
  malformed_basis.block = -1;
  EXPECT_THROW(source({malformed_basis}, {}), std::invalid_argument);

  auto duplicate_basis = basis;
  duplicate_basis.identity = "fixture-flux/basis/1";
  duplicate_basis.occurrence_path = "fixture-flux/loop/basis/1";
  EXPECT_THROW(source({basis, duplicate_basis}, {}), std::invalid_argument);

  CallbackProgramFaceFluxStage malformed_term;
  malformed_term.slot = 0;
  malformed_term.basis_slot = 0;
  malformed_term.expression_slot = 0;
  malformed_term.dt_power = 2;
  malformed_term.coefficient_numerator = 1;
  malformed_term.coefficient_denominator = 2;
  malformed_term.identity = "fixture-flux/stage/0";
  malformed_term.occurrence_path = "fixture-flux/loop/basis/0";
  malformed_term.owner = "tracer";
  malformed_term.clock = "clock.macro";
  EXPECT_THROW(source({basis}, {malformed_term}), std::invalid_argument);

  const auto source_with_budget = [&](std::vector<CallbackProgramFluxBasisOccurrence> bases,
                                      std::vector<CallbackProgramFaceFluxStage> terms,
                                      std::vector<ProgramFluxBudgetRecord> budgets) {
    return callback_program_source(17, "fixture-flux", "clock.macro", {"tracer"}, {},
                                   "pops_test_program_callback", "amr", {}, {}, {}, {}, budgets,
                                   std::nullopt, bases, terms);
  };
  auto canceled_basis = basis;
  canceled_basis.basis_slot = 1;
  canceled_basis.expression_slot = 2;
  canceled_basis.identity = "fixture-flux/basis/1";
  canceled_basis.occurrence_path = "fixture-flux/loop/basis/1";
  CallbackProgramFaceFluxStage live_term;
  live_term.slot = 0;
  live_term.basis_slot = 0;
  live_term.expression_slot = 0;
  live_term.coefficient_numerator = 1;
  live_term.coefficient_denominator = 2;
  live_term.identity = "fixture-flux/stage/0";
  live_term.occurrence_path = "fixture-flux/loop/basis/0";
  live_term.owner = "tracer";
  live_term.clock = "clock.macro";
  EXPECT_NO_THROW(source_with_budget({basis, canceled_basis}, {live_term}, {{1, 1, 0, 0}}));

  auto second_live_term = live_term;
  second_live_term.slot = 1;
  second_live_term.basis_slot = 1;
  second_live_term.identity = "fixture-flux/stage/1";
  second_live_term.occurrence_path = "fixture-flux/loop/basis/1";
  EXPECT_THROW(
      source_with_budget({basis, canceled_basis}, {live_term, second_live_term}, {{1, 1, 0, 0}}),
      std::invalid_argument);
}

}  // namespace
