#include <pops/runtime/amr/amr_layout_transfer.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include "native_dso_compiler.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int Dim = pops::kNativeDimension;
using Field = pops::MultiFab<Dim>;
using Endpoint = pops::AmrTransferEndpoint<Dim>;
using Spec = pops::AmrPhysicalTransferSpec<Dim>;
using Transfer = pops::PreparedAmrLayoutTransfer<Dim>;

struct Runtime {
  Runtime() {
    int argc = 0;
    char** argv = nullptr;
    pops::comm_init(&argc, &argv);
    Kokkos::initialize();
  }
  ~Runtime() {
    Kokkos::finalize();
    pops::comm_finalize();
  }
};

void runtime() {
  static Runtime runtime;
}

pops::Extent<Dim> shape(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

pops::Geometry<Dim> geometry(const pops::Extent<Dim>& shape) {
  pops::RealVector<Dim> lower{}, upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 1;
  return pops::Geometry<Dim>::from_bounds(pops::Box<Dim>::from_extents(shape), lower, upper);
}

std::shared_ptr<pops::component::LoadedComponent> provider(const pops::ExecutionLane& lane,
                                                           int failure = 0, bool pullback = false,
                                                           bool legacy = false) {
  const auto base = std::filesystem::path(POPS_TEST_TMPDIR) /
                    ("amr_physical_transport_provider_" + std::to_string(failure) + "_" +
                     std::to_string(pullback) + "_" + std::to_string(legacy));
#if defined(__APPLE__)
  const auto library = base.string() + ".dylib";
#else
  const auto library = base.string() + ".so";
#endif
  const auto source = base.string() + ".cpp";
  long failed = 0;
  if (lane.rank() == 0) {
    std::ofstream output(source);
    output << "#define TEST_INTEGRAL_FAILURE " << failure << "\n"
           << "#define TEST_TRANSFER_LEGACY " << legacy << "\n"
           << "#define TEST_PHYSICAL_IDENTITY \""
           << (pullback ? "test::physical-pullback-contract"
                        : "test::declared-piecewise-constant-base-bin-measure")
           << "\"\n";
    output << R"CPP(
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/physical_support_transfer.hpp>
#include <cstring>
      namespace {
      unsigned long long integral_calls = 0;
      int apply(void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* status) {
        // Fixture-only observation of the actual DSO's integral entry point invocation count.
        if (request && request->destination.data)
          *static_cast<double*>(request->destination.data) = static_cast<double>(integral_calls);
        *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
        return 0;
      }
      int apply_integral(void*, const PopsTransferIntegralRequestV2* request,
                         PopsComponentStatusV1* status) {
        ++integral_calls;
        if (!request || !request->physical_contract_identity ||
            std::strcmp(request->physical_contract_identity, TEST_PHYSICAL_IDENTITY) != 0) {
          *status = {sizeof(PopsComponentStatusV1), 72, POPS_COMPONENT_ABORT_RUN_V1,
                     "immutable physical identity differs from provider"};
          return 72;
        }
        if (TEST_INTEGRAL_FAILURE == 92) {
          *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_RETRY_STEP_V1,
                     "injected non-continuing action"};
          return 0;
        }
        if (TEST_INTEGRAL_FAILURE == 93)
          return 0;  // Deliberately leaves status unwritten.
        if (TEST_INTEGRAL_FAILURE) {
          *static_cast<double*>(request->destination.data) = -444.0;
          *status = {sizeof(PopsComponentStatusV1), TEST_INTEGRAL_FAILURE,
                     POPS_COMPONENT_ABORT_RUN_V1, "injected integral failure"};
          return TEST_INTEGRAL_FAILURE;
        }
        pops::component::PhysicalSupportIntegral integral{};
        integral.dimension = request->dimension;
        integral.weights = request->axis_weights;
        integral.weight_count = request->weight_count;
        for (int axis = 0; axis < request->dimension; ++axis)
          integral.weight_offsets[axis] = request->weight_offsets[axis];
        return pops::component::apply_physical_support_integral(integral, request->source,
                                                                request->destination, status);
      }
#if TEST_TRANSFER_LEGACY
      const PopsTransferApiV1 table{{sizeof(PopsTransferApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
                                     POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, nullptr, nullptr},
                                    &apply};
      constexpr auto version = 1;
#else
      const PopsTransferApiV2 table{{sizeof(PopsTransferApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,
                                     POPS_NATIVE_INTERFACE_TRANSFER_V2, 2, nullptr, nullptr},
                                    &apply,
                                    &apply_integral};
      constexpr auto version = 2;
#endif
      const PopsComponentInterfaceEntryV1 entry{POPS_NATIVE_INTERFACE_TRANSFER_V2, version,
                                                sizeof(table), &table};
      const PopsComponentApiV1 api{sizeof(PopsComponentApiV1),
                                   POPS_COMPONENT_PROTOCOL_ABI_V1,
                                   POPS_ABI_KEY_LITERAL,
                                   POPS_COMPONENT_CATALOG_SHA256_V1,
                                   "test::amr-physical-provider",
                                   "test::amr-physical-semantic",
                                   "test::amr-physical-manifest",
                                   1,
                                   &entry};
      }  // namespace
      extern "C" const PopsComponentApiV1* pops_component_interface_v1() {
        return &api;
      }
    )CPP";
    output.close();
    const auto result = pops::test::native_dso::compile_shared(source, library);
    failed = result.ok ? 0 : 1;
    if (!result.ok)
      pops::test::native_dso::report_compile_failure("test_amr_layout_transfer", result);
  }
  if (pops::all_reduce_max(failed, lane))
    throw std::runtime_error("AMR fixture provider compilation failed");
  pops::barrier(lane);
  pops::component::ExpectedNativeComponent expected{
      "test::amr-physical-provider",
      "test::amr-physical-semantic",
      "test::amr-physical-manifest",
      POPS_COMPONENT_CATALOG_SHA256_V1,
      POPS_ABI_KEY_LITERAL,
      pops::dynlib::AuthenticatedNativeFile(library).binary_identity(),
      {{POPS_NATIVE_INTERFACE_TRANSFER_V2, legacy ? 1u : 2u,
        legacy ? sizeof(PopsTransferApiV1) : sizeof(PopsTransferApiV2)}}};
  return std::make_shared<pops::component::LoadedComponent>(
      pops::component::LoadedComponent::load(library, expected));
}

std::uint64_t provider_integral_calls(const pops::component::LoadedComponent& provider,
                                      const pops::ExecutionLane& lane) {
  double local = 0;
  PopsTransferRequestV1 observation{};
  observation.destination.data = &local;
  PopsComponentStatusV1 status{};
  const auto& table = provider.table<PopsTransferApiV2>(POPS_NATIVE_INTERFACE_TRANSFER_V2, 2);
  if (table.apply(nullptr, &observation, &status) != 0)
    throw std::runtime_error("fixture cannot observe integral invocation count");
  return static_cast<std::uint64_t>(pops::all_reduce_sum(static_cast<long>(local), lane));
}

pops::SystemLayoutTransferExecution execution(const pops::ExecutionLane& lane) {
  pops::SystemLayoutTransferExecution value{};
  value.context_version = 1;
  value.execution_identity = "test::amr-physical-execution";
  value.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  value.backend_identity = "test::cpu";
  value.device_identity = "test::cpu:0";
  value.scalar_type = POPS_SCALAR_FLOAT64_V1;
  value.storage_precision = value.compute_precision = value.accumulation_precision =
      value.reduction_precision = POPS_PRECISION_FLOAT64_V1;
  value.stream_identity = "test::host-synchronous";
#ifdef POPS_HAS_MPI
  value.communicator_identity = std::string(lane.identity());
  value.communicator_datatype_identity = "MPI_DOUBLE";
  value.communicator_f_handle = MPI_Comm_c2f(lane.native_handle());
  value.communicator_datatype_f_handle = MPI_Type_c2f(MPI_DOUBLE);
#else
  value.communicator_identity = "serial";
  value.communicator_datatype_identity = "none";
#endif
  return value;
}

struct Hierarchy {
  std::vector<Field> state, coverage;
  std::vector<pops::Geometry<Dim>> geometries;
  std::string identity;
  std::uint64_t generation = 1;

  Endpoint endpoint() const {
    Endpoint endpoint;
    endpoint.layout_identity = identity;
    endpoint.state_identity = identity + "::state";
    endpoint.hierarchy_identity = identity + "::hierarchy";
    endpoint.hierarchy_generation = generation;
    endpoint.periodicity.fill(true);
    for (std::size_t level = 0; level < state.size(); ++level)
      endpoint.levels.push_back(
          {static_cast<int>(level), geometries[level], &state[level], &coverage[level]});
    return endpoint;
  }
};

void add_level(Hierarchy& hierarchy, pops::Extent<Dim> domain, pops::Box<Dim> patch, int owner,
               const pops::ExecutionLane& lane) {
  auto ranks = shape(1);
  ranks[0] = lane.size();
  pops::mesh::RankSpace<Dim> rank_space({}, ranks);
  pops::mesh::BoxArray<Dim> layout({patch});
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(
      layout, rank_space, {rank_space.coordinate(static_cast<std::size_t>(owner))});
  hierarchy.state.emplace_back(layout, distribution, rank_space.coordinate(lane.rank()), 2,
                               pops::Extent<Dim>{});
  hierarchy.coverage.emplace_back(layout, distribution, rank_space.coordinate(lane.rank()), 1,
                                  pops::Extent<Dim>{});
  hierarchy.geometries.push_back(geometry(domain));
}

void fill(Hierarchy& hierarchy, bool high, double refined_value = 3) {
  for (std::size_t level = 0; level < hierarchy.state.size(); ++level)
    for (std::size_t local = 0; local < hierarchy.state[level].local_size(); ++local) {
      auto& fab = hierarchy.state[level].fab(local);
      auto values = fab.create_host_mirror();
      auto& mask_fab = hierarchy.coverage[level].fab(local);
      auto mask = mask_fab.create_host_mirror();
      const auto count = static_cast<std::size_t>(fab.box().numPts());
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
        const bool covered = level == 0 && ordinal == 0 && hierarchy.state.size() > 1;
        mask(ordinal) = covered ? 0 : 1;
        for (std::size_t component = 0; component < 2; ++component)
          values(component * count + ordinal) =
              high ? ((covered ? 1000 : (level ? refined_value : 1)) + 5 * component) : 7;
      }
      fab.copy_from_host(values);
      mask_fab.copy_from_host(mask);
    }
}

Hierarchy high_hierarchy(const pops::ExecutionLane& lane) {
  Hierarchy high;
  high.identity = "test::high-support";
  auto base_shape = shape(1);
  base_shape[0] = 2;
  if constexpr (Dim > 1)
    base_shape[1] = 2;
  add_level(high, base_shape, pops::Box<Dim>::from_extents(base_shape), 0, lane);
  auto fine_shape = base_shape;
  fine_shape[0] *= 2;
  if constexpr (Dim > 1)
    fine_shape[1] *= 2;
  add_level(high, fine_shape, pops::Box<Dim>::from_extents(base_shape), 1 % lane.size(), lane);
  fill(high, true);
  return high;
}

Hierarchy low_hierarchy(const pops::ExecutionLane& lane) {
  Hierarchy low;
  low.identity = "test::low-support";
  auto base_shape = shape(1);
  if constexpr (Dim > 1)
    base_shape[Dim - 1] = 2;
  add_level(low, base_shape, pops::Box<Dim>::from_extents(base_shape), 1 % lane.size(), lane);
  if constexpr (Dim > 1) {
    auto fine_shape = base_shape;
    fine_shape[Dim - 1] *= 2;
    add_level(low, fine_shape, pops::Box<Dim>::from_extents(base_shape), 0, lane);
  }
  fill(low, false);
  return low;
}

Spec specification(const Hierarchy& high, const Hierarchy& low) {
  Spec spec;
  auto& map = spec.authentication;
  map.mapping_identity = "test::physical-moment";
  map.provider_identity = "test::physical-provider";
  map.provider_component_identity = "test::amr-physical-provider";
  map.provider_manifest_identity = "test::amr-physical-manifest";
  map.source_layout_identity = high.identity;
  map.target_layout_identity = low.identity;
  map.source_block = "high";
  map.target_block = "low";
  map.source_representation = map.target_representation = "pops://representations/cell-average@1";
  map.synchronization_identity = "pops://synchronization/after-source-step@1";
  map.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  map.physical_contract = true;
  map.physical_source_active[0] = 1;
  const int reduced_axis = Dim == 1 ? 0 : 1;
  if constexpr (Dim > 1) {
    map.physical_source_active[1] = map.physical_target_active[Dim - 1] = 1;
    map.physical_source_to_target[0] = Dim - 1;
  }
  spec.physical_contract_identity = "test::declared-piecewise-constant-base-bin-measure";
  spec.base_bin_weights[reduced_axis] = {-1, 3};
  spec.base_bin_lower[reduced_axis] = 0;
  spec.base_bin_upper[reduced_axis] = 1;
  spec.budget = {1000, 10000, 10000, 100000, 16 * 1024 * 1024};
  return spec;
}

std::vector<Field*> pointers(std::vector<Field>& fields) {
  std::vector<Field*> result;
  for (auto& field : fields)
    result.push_back(&field);
  return result;
}

void check_low(const Hierarchy& low, const std::vector<Field>& candidate, double fine_value) {
  for (std::size_t level = 0; level < candidate.size(); ++level)
    for (std::size_t local = 0; local < candidate[level].local_size(); ++local) {
      const auto& fab = candidate[level].fab(local);
      auto values = fab.create_host_mirror();
      fab.copy_to_host(values);
      const auto count = static_cast<std::size_t>(fab.box().numPts());
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
        for (std::size_t component = 0; component < 2; ++component) {
          const bool covered = Dim > 1 && level == 0 && ordinal == 0;
          const double expected =
              covered ? 7 : ((Dim == 1 || level > 0) ? fine_value : 2) + 10 * component;
          EXPECT_DOUBLE_EQ(values(component * count + ordinal), expected);
        }
    }
  // Detached publication never changes the accepted target state.
  for (const auto& field : low.state)
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const auto& fab = field.fab(local);
      auto values = fab.create_host_mirror();
      fab.copy_to_host(values);
      for (std::size_t ordinal = 0; ordinal < values.size(); ++ordinal)
        EXPECT_DOUBLE_EQ(values(ordinal), 7);
    }
}

TEST(AmrLayoutTransfer, ActiveCompositeMeasureStageRebindingRetryAndRestartFence) {
  runtime();
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test::amr-layout-transfer");
  auto component = provider(lane);
  auto high = high_hierarchy(lane), low = low_hierarchy(lane);
  const auto spec = specification(high, low);
  auto transfer =
      Transfer::prepare(high.endpoint(), low.endpoint(), spec, component, execution(lane), lane);
  EXPECT_GT(transfer->canonical_jobs(), 0u);
  EXPECT_GT(transfer->transported_elements(), 0u);
  EXPECT_LE(transfer->prepared_bytes(), spec.budget.prepared_bytes);
  auto candidates = low.state;
  transfer->begin_transaction(1);
  const auto expected = transfer->expected_receipt_contract(high.endpoint(), low.endpoint());
  EXPECT_FALSE(expected.transfer.applied);
  EXPECT_EQ(expected.source_active_elements, Dim == 1 ? 6u : 14u);
  EXPECT_EQ(expected.destination_active_elements, Dim == 1 ? 2u : 6u);
  const auto calls_before = provider_integral_calls(*component, lane);
  transfer->capture(high.endpoint(), 1, 1);
  const auto receipt = transfer->apply(low.endpoint(), pointers(candidates), 1, 1);
  EXPECT_EQ(provider_integral_calls(*component, lane) - calls_before, receipt.canonical_jobs);
  EXPECT_EQ(receipt.source_active_elements, Dim == 1 ? 6u : 14u);
  EXPECT_EQ(receipt.destination_active_elements, Dim == 1 ? 2u : 6u);
  EXPECT_EQ(receipt.physical_contract_identity, spec.physical_contract_identity);
  EXPECT_EQ(receipt.source_active_elements, expected.source_active_elements);
  EXPECT_EQ(receipt.destination_active_elements, expected.destination_active_elements);
  EXPECT_EQ(receipt.source_hierarchy_identity, expected.source_hierarchy_identity);
  EXPECT_EQ(receipt.target_hierarchy_generation, expected.target_hierarchy_generation);
  EXPECT_EQ(receipt.target_stage_identity, expected.target_stage_identity);
  EXPECT_EQ(receipt.transported_elements, expected.transported_elements);
  check_low(low, candidates, 0);

  transfer->reject_attempt(1, 1);
  candidates = low.state;
  // New addresses are intentional: the prepared transport must borrow this stage, not retain U.
  auto staged = high;
  fill(staged, true, 4);
  auto stage = staged.endpoint();
  stage.stage_identity = "test::qualified-stage::corrector";
  stage.stage_generation = 19;
  transfer->capture(stage, 1, 2);
  const auto retry = transfer->apply(low.endpoint(), pointers(candidates), 1, 2);
  EXPECT_EQ(retry.source_stage_identity, stage.stage_identity);
  EXPECT_EQ(retry.source_stage_generation, 19u);
  check_low(low, candidates, -1);
  transfer->rollback_transaction(1);

  auto restarted = staged;
  ++restarted.generation;
  transfer->begin_transaction(2);
  EXPECT_THROW(transfer->capture(restarted.endpoint(), 2, 1), std::exception);
  transfer->rollback_transaction(2);
  auto restored = Transfer::prepare(restarted.endpoint(), low.endpoint(), spec, component,
                                    execution(lane), lane);
  candidates = low.state;
  restored->begin_transaction(1);
  restored->capture(restarted.endpoint(), 1, 1);
  (void)restored->apply(low.endpoint(), pointers(candidates), 1, 1);
  check_low(low, candidates, -1);
  restored->finalize_transaction(1);
}

TEST(AmrLayoutTransfer, CoverageForgeryAndBudgetsFailBeforeCandidatePublication) {
  runtime();
  auto lane =
      pops::ExecutionLane::duplicate_world_collectively("test::amr-layout-transfer-refusal");
  auto component = provider(lane);
  auto high = high_hierarchy(lane), low = low_hierarchy(lane);
  auto spec = specification(high, low);
  auto too_small = spec;
  too_small.budget.canonical_jobs = 0;
  EXPECT_THROW(Transfer::prepare(high.endpoint(), low.endpoint(), too_small, component,
                                 execution(lane), lane),
               std::exception);
  auto transfer =
      Transfer::prepare(high.endpoint(), low.endpoint(), spec, component, execution(lane), lane);
  for (std::size_t local = 0; local < high.coverage[0].local_size(); ++local)
    high.coverage[0].fab(local).set_val(1);
  transfer->begin_transaction(1);
  EXPECT_THROW(transfer->capture(high.endpoint(), 1, 1), std::exception);
  auto candidates = low.state;
  EXPECT_THROW(transfer->apply(low.endpoint(), pointers(candidates), 1, 1), std::exception);
  transfer->rollback_transaction(1);
}

TEST(AmrLayoutTransfer, CompositePullbackUsesIndependentLevelsAndRepeatedSourceCarriers) {
  runtime();
  auto lane =
      pops::ExecutionLane::duplicate_world_collectively("test::amr-layout-transfer-pullback");
  auto component = provider(lane);
  auto high = high_hierarchy(lane), low = low_hierarchy(lane);
  auto spec = specification(high, low);
  auto reduction =
      Transfer::prepare(high.endpoint(), low.endpoint(), spec, component, execution(lane), lane);
  auto lower_values = low.state;
  reduction->begin_transaction(1);
  reduction->capture(high.endpoint(), 1, 1);
  (void)reduction->apply(low.endpoint(), pointers(lower_values), 1, 1);
  reduction->finalize_transaction(1);
  low.state = lower_values;
  auto& map = spec.authentication;
  map.mapping_identity = "test::physical-pullback";
  map.operation = POPS_TRANSFER_OPERATION_PHYSICAL_PULLBACK_V1;
  std::swap(map.source_layout_identity, map.target_layout_identity);
  std::swap(map.source_block, map.target_block);
  std::swap(map.physical_source_active, map.physical_target_active);
  map.physical_source_to_target.fill(-1);
  if constexpr (Dim > 1)
    map.physical_source_to_target[Dim - 1] = 0;
  for (auto& weights : spec.base_bin_weights)
    weights.clear();
  spec.physical_contract_identity = "test::physical-pullback-contract";
  auto pullback_component = provider(lane, 0, true);
  auto broadcast = Transfer::prepare(low.endpoint(), high.endpoint(), spec, pullback_component,
                                     execution(lane), lane);
  auto candidate = high.state;
  broadcast->begin_transaction(1);
  broadcast->capture(low.endpoint(), 1, 1);
  const auto receipt = broadcast->apply(high.endpoint(), pointers(candidate), 1, 1);
  EXPECT_EQ(receipt.source_active_elements, Dim == 1 ? 2u : 6u);
  EXPECT_EQ(receipt.destination_active_elements, Dim == 1 ? 6u : 14u);
  EXPECT_GE(receipt.transported_elements, receipt.source_active_elements);
  for (std::size_t level = 0; level < candidate.size(); ++level)
    for (std::size_t local = 0; local < candidate[level].local_size(); ++local) {
      const auto& fab = candidate[level].fab(local);
      auto values = fab.create_host_mirror();
      fab.copy_to_host(values);
      const auto count = static_cast<std::size_t>(fab.box().numPts());
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
        for (std::size_t component = 0; component < 2; ++component) {
          const bool covered = level == 0 && ordinal == 0;
          const double expected =
              covered ? 1000 + 5 * component
                      : ((Dim == 1 || level > 0) ? 0 : 2 * (ordinal % 2)) + 10 * component;
          EXPECT_DOUBLE_EQ(values(component * count + ordinal), expected);
        }
    }
  broadcast->rollback_transaction(1);
}

TEST(AmrLayoutTransfer, ProviderFailureRollsBackDetachedCandidateAndLegacyProviderIsRefused) {
  runtime();
  auto lane =
      pops::ExecutionLane::duplicate_world_collectively("test::amr-transfer-provider-failure");
  EXPECT_THROW(provider(lane, 0, false, true), std::runtime_error);
  auto high = high_hierarchy(lane), low = low_hierarchy(lane);
  const auto spec = specification(high, low);
  auto candidates = low.state;
  for (int failure : {91, 92, 93}) {
    auto component = provider(lane, failure);
    auto transfer =
        Transfer::prepare(high.endpoint(), low.endpoint(), spec, component, execution(lane), lane);
    transfer->begin_transaction(1);
    transfer->capture(high.endpoint(), 1, 1);
    const auto before = provider_integral_calls(*component, lane);
    EXPECT_THROW(transfer->apply(low.endpoint(), pointers(candidates), 1, 1), std::runtime_error);
    EXPECT_GT(provider_integral_calls(*component, lane), before);
    for (const auto& field : candidates)
      for (std::size_t local = 0; local < field.local_size(); ++local) {
        const auto& fab = field.fab(local);
        auto values = fab.create_host_mirror();
        fab.copy_to_host(values);
        for (std::size_t value = 0; value < fab.size(); ++value)
          EXPECT_DOUBLE_EQ(values(value), 7);
      }
    transfer->reject_attempt(1, 1);
    transfer->rollback_transaction(1);
  }
  // Negotiation is valid, but the executable rejects a physical identity it did not compile.
  auto valid = provider(lane);
  auto forged = spec;
  forged.physical_contract_identity = "test::forged-physical-contract";
  auto wrong_identity =
      Transfer::prepare(high.endpoint(), low.endpoint(), forged, valid, execution(lane), lane);
  wrong_identity->begin_transaction(1);
  wrong_identity->capture(high.endpoint(), 1, 1);
  EXPECT_THROW(wrong_identity->apply(low.endpoint(), pointers(candidates), 1, 1),
               std::runtime_error);
  wrong_identity->rollback_transaction(1);
}

}  // namespace
