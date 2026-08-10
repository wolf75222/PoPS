#include <gtest/gtest.h>

#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include "component_abi_test_helpers.hpp"
#include "native_dso_compiler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace abi = pops::component::test_support;

enum class FluxTableFixture { Exact, HeaderOnly, ForgedEntrySize, WrongAbi };

constexpr const char* kComponentId = "pops://test/final-flux@1.0.0";
constexpr const char* kSemanticIdentity = "semantic-final-flux";
constexpr const char* kManifestIdentity = "manifest-final-flux";

std::string component_source() {
  return R"CPP(
#include <pops/runtime/config/generated_component_abi.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

    namespace {
    int prepare_count = 0;
    int destroy_count = 0;
    int tag_call_count = 0;
    int partial_tag_output = 0;
    const void* last_tag_state_data = nullptr;
    std::int64_t last_tag_logical_tick = -1;
    double last_tag_physical_time = -1.0;

    int evaluate(void*, const PopsNumericalFluxRequestV1* request, PopsNumericalFluxResultV1* result) {
      const auto* left = static_cast<const double*>(request->left.data);
      const auto* right = static_cast<const double*>(request->right.data);
      auto* output = static_cast<double*>(result->normal_flux.data);
      const auto points = request->left.extents[0] * request->left.extents[1];
      for (std::size_t point = 0; point < points; ++point) {
        for (std::size_t component = 0; component < request->left.component_count; ++component) {
          const auto index = point * static_cast<std::size_t>(request->left.axis_strides[1]) +
                             component * static_cast<std::size_t>(request->left.component_stride);
          output[index] = 0.25 * left[index] + 0.75 * right[index];
        }
        result->stability_bounds[point] = 3.0;
        result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
      }
      result->status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    int apply_transfer(void*, const PopsTransferRequestV1*, PopsComponentStatusV1* status) {
      *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    int apply_ghost(void* state, const PopsGhostBoundaryRequestV1* request,
                    PopsComponentStatusV1* status) {
      if (state == nullptr || request == nullptr || request->ghosts.data == nullptr) {
        *status = {sizeof(PopsComponentStatusV1), 41, POPS_COMPONENT_ABORT_RUN_V1,
                   "ghost session state is missing"};
        return 41;
      }
      auto* values = static_cast<double*>(request->ghosts.data);
      const auto points = request->ghosts.extents[0] * request->ghosts.extents[1];
      for (std::size_t point = 0; point < points; ++point)
        for (std::size_t component = 0; component < request->ghosts.component_count; ++component) {
          const auto index = point * static_cast<std::size_t>(request->ghosts.axis_strides[0]) +
                             component * static_cast<std::size_t>(request->ghosts.component_stride);
          values[index] = static_cast<double>(*static_cast<int*>(state));
        }
      *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    int transform_boundary_flux(void* state, const PopsBoundaryFluxRequestV1* request,
                                PopsBoundaryFluxResultV1* result) {
      if (result == nullptr)
        return 42;
      if (state == nullptr || request == nullptr || request->region.kind != POPS_BOUNDARY_FACE_V1 ||
          request->region.axis_count != 1 || request->region.axes == nullptr ||
          request->region.sides == nullptr || request->outward_normals.data == nullptr ||
          request->face_measures == nullptr || result->outward_normal_flux.data == nullptr) {
        result->status = {sizeof(PopsComponentStatusV1), 42, POPS_COMPONENT_ABORT_RUN_V1,
                          "boundary flux contract is incomplete"};
        return 42;
      }
      const auto* base = static_cast<const double*>(request->base_outward_normal_flux.data);
      const auto* normals = static_cast<const double*>(request->outward_normals.data);
      auto* output = static_cast<double*>(result->outward_normal_flux.data);
      const auto points = request->base_outward_normal_flux.extents[0] *
                          request->base_outward_normal_flux.extents[1];
      const auto axis = static_cast<std::size_t>(request->region.axes[0]);
      const double side = static_cast<double>(request->region.sides[0]);
      for (std::size_t point = 0; point < points; ++point) {
        const auto normal_offset =
            point * static_cast<std::size_t>(request->outward_normals.axis_strides[0]) +
            axis * static_cast<std::size_t>(request->outward_normals.component_stride);
        if (normals[normal_offset] != side || request->face_measures[point] <= 0.0) {
          result->status = {sizeof(PopsComponentStatusV1), 43, POPS_COMPONENT_ABORT_RUN_V1,
                            "boundary flux orientation is inconsistent"};
          return 43;
        }
        for (std::size_t component = 0;
             component < request->base_outward_normal_flux.component_count; ++component) {
          const auto index =
              point * static_cast<std::size_t>(request->base_outward_normal_flux.axis_strides[0]) +
              component *
                  static_cast<std::size_t>(request->base_outward_normal_flux.component_stride);
          output[index] = base[index] + 10.0;
        }
        result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
      }
      result->status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    int tag_batch(void*, const PopsTaggerRequestV2* request, PopsComponentStatusV1* status) {
      ++tag_call_count;
      last_tag_state_data = request->states[0].values.data;
      last_tag_logical_tick = request->logical_time.tick;
      last_tag_physical_time = request->logical_time.physical_time;
      if ((request->execution_mode != POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2 &&
           request->execution_mode != POPS_TAGGER_EXECUTION_HOST_V2) ||
          request->collective_scope != POPS_TAGGER_COLLECTIVE_NONE_V2 ||
          std::string(request->execution.communicator_identity) !=
              POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1 ||
          request->execution.communicator_f_handle != 0 ||
          request->execution.communicator_datatype_f_handle != 0 ||
          std::string(request->execution.communicator_datatype_identity) != "none" ||
          request->states[0].values.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
          request->refine_candidates.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
          request->refine_candidates.ownership != POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1) {
        *status = {sizeof(PopsComponentStatusV1), 20, POPS_COMPONENT_ABORT_RUN_V1,
                   "tagger executor contract mismatch"};
        return 20;
      }
      if (request->program.non_finite_policy != POPS_TAGGING_NON_FINITE_REJECT_V1) {
        *status = {sizeof(PopsComponentStatusV1), 21, POPS_COMPONENT_ABORT_RUN_V1,
                   "unsupported non-finite policy"};
        return 21;
      }
      const std::size_t points = request->refine_candidates.size;
      std::fill_n(request->refine_candidates.data, points, std::uint8_t{0});
      if (!partial_tag_output) {
        std::fill_n(request->coarsen_candidates.data, points, std::uint8_t{0});
        std::fill_n(request->refine_equalities.data, points, std::uint8_t{0});
        std::fill_n(request->coarsen_equalities.data, points, std::uint8_t{0});
      }
      const auto evaluate = [&](const int32_t* opcodes, const int32_t* arguments,
                                std::size_t instruction_count, PopsTaggerMaskViewV2 candidates,
                                PopsTaggerMaskViewV2 equalities) -> bool {
        const auto& reference = request->states[0].values;
        const std::size_t nx =
            reference.extents[0] - reference.ghost_lower[0] - reference.ghost_upper[0];
        for (std::size_t point = 0; point < points; ++point) {
          bool matches[128]{}, equality[128]{};
          std::size_t depth = 0;
          const std::size_t i = point % nx, j = point / nx;
          for (std::size_t instruction = 0; instruction < instruction_count; ++instruction) {
            const int32_t opcode = opcodes[instruction];
            const int32_t argument = arguments[instruction];
            if (opcode >= 1 && opcode <= 5) {
              const auto& leaf = request->program.leaves[argument];
              const auto& view = request->states[leaf.state_index].values;
              const auto* values = static_cast<const double*>(view.data);
              const auto read = [&](std::ptrdiff_t x, std::ptrdiff_t y) {
                const auto offset =
                    (x + static_cast<std::ptrdiff_t>(view.ghost_lower[0])) * view.axis_strides[0] +
                    (y + static_cast<std::ptrdiff_t>(view.ghost_lower[1])) * view.axis_strides[1] +
                    static_cast<std::ptrdiff_t>(leaf.component) * view.component_stride;
                return values[offset];
              };
              double sample = read(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j));
              if (opcode == 3)
                sample = sample < 0.0 ? -sample : sample;
              if (opcode == 4 || opcode == 5) {
                const auto& stencil = request->program.stencils[leaf.stencil_index];
                double squared_norm = 0.0;
                for (std::size_t axis_index = 0; axis_index < stencil.axis_count; ++axis_index) {
                  const auto& axis = stencil.axes[axis_index];
                  double derivative = 0.0;
                  for (std::size_t term = 0; term < axis.term_count; ++term) {
                    const auto x =
                        static_cast<std::ptrdiff_t>(i) + (axis.axis == 0 ? axis.offsets[term] : 0);
                    const auto y =
                        static_cast<std::ptrdiff_t>(j) + (axis.axis == 1 ? axis.offsets[term] : 0);
                    derivative += axis.coefficients[term] * read(x, y);
                  }
                  derivative /= request->cell_size[axis.axis];
                  squared_norm += derivative * derivative;
                }
                sample = std::sqrt(squared_norm);
              }
              if (!std::isfinite(sample)) {
                *status = {sizeof(PopsComponentStatusV1), 22, POPS_COMPONENT_ABORT_RUN_V1,
                           "non-finite AMR indicator sample"};
                return false;
              }
              const bool greater = opcode == 1 || opcode == 3 || opcode == 4;
              matches[depth] = greater ? sample > leaf.threshold : sample < leaf.threshold;
              equality[depth] = sample == leaf.threshold;
              ++depth;
            } else if (opcode == 18) {
              if (!equality[depth - 1])
                matches[depth - 1] = !matches[depth - 1];
            } else {
              const std::size_t begin = depth - static_cast<std::size_t>(argument);
              bool any_true = false, any_false = false, any_unknown = false;
              for (std::size_t child = begin; child < depth; ++child) {
                any_unknown = any_unknown || equality[child];
                any_true = any_true || (matches[child] && !equality[child]);
                any_false = any_false || (!matches[child] && !equality[child]);
              }
              depth = begin + 1;
              matches[begin] = opcode == 16 ? any_true : !any_false && !any_unknown;
              equality[begin] = opcode == 16 ? !any_true && any_unknown : !any_false && any_unknown;
            }
          }
          if (instruction_count != 0) {
            candidates.data[point] = matches[0] ? 1u : 0u;
            equalities.data[point] = equality[0] ? 1u : 0u;
          }
        }
        return true;
      };
      if (!evaluate(request->program.refine_opcodes, request->program.refine_arguments,
                    request->program.refine_instruction_count, request->refine_candidates,
                    request->refine_equalities))
        return 22;
      if (request->program.coarsen_instruction_count != 0) {
        if (!evaluate(request->program.coarsen_opcodes, request->program.coarsen_arguments,
                      request->program.coarsen_instruction_count, request->coarsen_candidates,
                      request->coarsen_equalities))
          return 22;
      }
      *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    int cluster(void*, const PopsClusteringRequestV1* request, PopsComponentStatusV1* status) {
      if (request->dimension != 2 || request->box_capacity < 1)
        return 2;
      const std::size_t nx = static_cast<std::size_t>(request->extents[0]);
      std::int64_t lo_x = request->extents[0], lo_y = request->extents[1];
      std::int64_t hi_x = -1, hi_y = -1;
      for (std::size_t point = 0; point < request->tags.size; ++point) {
        if (request->tags.data[point] == 0)
          continue;
        const auto i = static_cast<std::int64_t>(point % nx);
        const auto j = static_cast<std::int64_t>(point / nx);
        lo_x = std::min(lo_x, i);
        lo_y = std::min(lo_y, j);
        hi_x = std::max(hi_x, i);
        hi_y = std::max(hi_y, j);
      }
      if (hi_x < 0) {
        *request->box_count = 0;
      } else {
        request->boxes[0] = lo_x;
        request->boxes[1] = lo_y;
        request->boxes[2] = hi_x;
        request->boxes[3] = hi_y;
        *request->box_count = 1;
      }
      *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    int prepare(const PopsComponentPrepareRequestV1*, void** state, PopsComponentStatusV1* status) {
      *state = new int(++prepare_count);
      *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }
    void destroy(void* state) {
      ++destroy_count;
      delete static_cast<int*>(state);
    }

#if defined(POPS_TEST_HEADER_ONLY_FLUX_TABLE)
    const PopsComponentTableHeaderV1 flux{sizeof(PopsComponentTableHeaderV1),
                                          POPS_COMPONENT_PROTOCOL_ABI_V1,
                                          POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1,
                                          1,
                                          &prepare,
                                          &destroy};
#else
    const PopsNumericalFluxApiV1 flux{
        {sizeof(PopsNumericalFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
         POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, &prepare, &destroy},
        &evaluate};
#endif
    const PopsTransferApiV1 transfer{{sizeof(PopsTransferApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
                                      POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, &prepare, &destroy},
                                     &apply_transfer};
    const PopsGhostBoundaryApiV1 ghost{
        {sizeof(PopsGhostBoundaryApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
         POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1, 1, &prepare, &destroy},
        &apply_ghost};
    const PopsBoundaryFluxApiV1 boundary_flux{
        {sizeof(PopsBoundaryFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
         POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1, 1, &prepare, &destroy},
        &transform_boundary_flux};
    const PopsTaggerApiV2 tagger{{sizeof(PopsTaggerApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,
                                  POPS_NATIVE_INTERFACE_TAGGER_V2, 2, &prepare, &destroy},
                                 &tag_batch};
    const PopsClusteringApiV1 clustering{
        {sizeof(PopsClusteringApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
         POPS_NATIVE_INTERFACE_CLUSTERING_V1, 1, &prepare, &destroy},
        &cluster};
    const PopsComponentInterfaceEntryV1 interfaces[]{
        {POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
#if defined(POPS_TEST_FORGED_FLUX_ENTRY_SIZE)
         sizeof(PopsNumericalFluxApiV1), &flux},
#else
         sizeof(flux), &flux},
#endif
        {POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1), &transfer},
        {POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1, 1, sizeof(PopsGhostBoundaryApiV1), &ghost},
        {POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1, 1, sizeof(PopsBoundaryFluxApiV1), &boundary_flux},
        {POPS_NATIVE_INTERFACE_TAGGER_V2, 2, sizeof(PopsTaggerApiV2), &tagger},
        {POPS_NATIVE_INTERFACE_CLUSTERING_V1, 1, sizeof(PopsClusteringApiV1), &clustering}};
    const PopsComponentApiV1 component{
        sizeof(PopsComponentApiV1),
        POPS_COMPONENT_PROTOCOL_ABI_V1,
#if defined(POPS_TEST_WRONG_COMPONENT_ABI)
        "compiler=MPICH;std=202002;headers=wrong;kokkos=1;stdlib=libc++;mpi=1;mpi_abi=wrong",
#else
        POPS_ABI_KEY_LITERAL,
#endif
        POPS_COMPONENT_CATALOG_SHA256_V1,
        "pops://test/final-flux@1.0.0",
        "semantic-final-flux",
        "manifest-final-flux",
        6,
        interfaces};
    }  // namespace

    extern "C" const PopsComponentApiV1* pops_component_interface_v1() {
      return &component;
    }
    extern "C" int pops_test_prepare_count() {
      return prepare_count;
    }
    extern "C" int pops_test_destroy_count() {
      return destroy_count;
    }
    extern "C" int pops_test_tag_call_count() {
      return tag_call_count;
    }
    extern "C" const void* pops_test_last_tag_state_data() {
      return last_tag_state_data;
    }
    extern "C" std::int64_t pops_test_last_tag_logical_tick() {
      return last_tag_logical_tick;
    }
    extern "C" double pops_test_last_tag_physical_time() {
      return last_tag_physical_time;
    }
    extern "C" void pops_test_set_partial_tag_output(int value) {
      partial_tag_output = value;
    }
  )CPP";
}

std::filesystem::path compile_component(FluxTableFixture fixture = FluxTableFixture::Exact) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto base =
      std::filesystem::path(POPS_TEST_TMPDIR) / ("final_component_abi_" + std::to_string(stamp));
  const auto source = base.string() + ".cpp";
#if defined(__APPLE__)
  const auto library = base.string() + ".dylib";
#else
  const auto library = base.string() + ".so";
#endif
  {
    std::ofstream stream(source);
    stream << component_source();
  }
  std::string fixture_flags;
  if (fixture == FluxTableFixture::HeaderOnly || fixture == FluxTableFixture::ForgedEntrySize)
    fixture_flags += " -DPOPS_TEST_HEADER_ONLY_FLUX_TABLE";
  if (fixture == FluxTableFixture::ForgedEntrySize)
    fixture_flags += " -DPOPS_TEST_FORGED_FLUX_ENTRY_SIZE";
  if (fixture == FluxTableFixture::WrongAbi)
    fixture_flags += " -DPOPS_TEST_WRONG_COMPONENT_ABI";
  const auto package = pops::test::native_dso::compile_shared(source, library, fixture_flags);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_amr_native_loader", package);
    throw std::runtime_error("failed to compile exact component ABI fixture; log: " +
                             package.log_path);
  }
  std::filesystem::remove(source);
  return library;
}

pops::component::ExpectedNativeComponent expected() {
  return {kComponentId,
          kSemanticIdentity,
          kManifestIdentity,
          POPS_COMPONENT_CATALOG_SHA256_V1,
          POPS_ABI_KEY_LITERAL,
          {{POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, sizeof(PopsNumericalFluxApiV1)},
           {POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1)},
           {POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1, 1, sizeof(PopsGhostBoundaryApiV1)},
           {POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1, 1, sizeof(PopsBoundaryFluxApiV1)},
           {POPS_NATIVE_INTERFACE_TAGGER_V2, 2, sizeof(PopsTaggerApiV2)},
           {POPS_NATIVE_INTERFACE_CLUSTERING_V1, 1, sizeof(PopsClusteringApiV1)}}};
}

std::shared_ptr<const pops::component::PreparedExecutionContextV1> prepared_execution() {
  const PopsExecutionContextV1 execution = abi::host_execution_context();
  return std::make_shared<const pops::component::PreparedExecutionContextV1>(
      execution.execution_identity, execution.context_version, execution.memory_space,
      execution.backend_identity, execution.device_identity, execution.scalar_type,
      execution.storage_precision, execution.compute_precision, execution.accumulation_precision,
      execution.reduction_precision, execution.stream_handle, execution.stream_identity,
      execution.communicator_f_handle, execution.communicator_datatype_f_handle,
      execution.communicator_identity, execution.communicator_datatype_identity);
}

TEST(test_amr_native_loader, LoadsAuthenticatesAndExecutesExactFinalTable) {
  const auto library = compile_component();
  {
    auto loaded = pops::component::LoadedComponent::load(library.string(), expected());
    const auto& table =
        loaded.table<PopsNumericalFluxApiV1>(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1);
    const std::array<double, 4> left{1.0, 3.0, 5.0, 7.0};
    const std::array<double, 4> right{2.0, 4.0, 6.0, 8.0};
    const std::array<double, 4> normals{1.0, 0.0, 1.0, 0.0};
    std::array<double, 4> flux{};
    std::array<double, 2> stability{};
    std::array<PopsComponentActionV1, 2> actions{};
    const auto execution = abi::host_execution_context();
    const PopsNumericalFluxRequestV1 request{sizeof(PopsNumericalFluxRequestV1),
                                             abi::const_field_view(left.data(), 1, 2, 2),
                                             abi::const_field_view(right.data(), 1, 2, 2),
                                             abi::const_field_view(normals.data(), 1, 2, 2),
                                             nullptr,
                                             abi::logical_time(),
                                             execution};
    PopsNumericalFluxResultV1 result{sizeof(PopsNumericalFluxResultV1),
                                     abi::field_view(flux.data(), 1, 2, 2),
                                     stability.data(),
                                     actions.data(),
                                     {}};
    void* state = loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(pops::component::evaluate_faces(table, state, request, result), 0);
    EXPECT_EQ(flux, (std::array<double, 4>{1.75, 3.75, 5.75, 7.75}));
    EXPECT_EQ(stability, (std::array<double, 2>{3.0, 3.0}));
    auto mismatched_context = execution;
    mismatched_context.execution_identity = "test::other-execution-context";
    EXPECT_THROW(
        (void)loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, mismatched_context),
        std::invalid_argument);
  }
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, CachesPreparedResourcesPerExactTargetAndPinsExecutionContext) {
  const auto library = compile_component();
  const auto inspection = pops::dynlib::open(library.string());
  ASSERT_TRUE(pops::dynlib::valid(inspection));
  using CounterFn = int (*)();
  const auto prepare_count =
      reinterpret_cast<CounterFn>(pops::dynlib::sym(inspection, "pops_test_prepare_count"));
  const auto destroy_count =
      reinterpret_cast<CounterFn>(pops::dynlib::sym(inspection, "pops_test_destroy_count"));
  ASSERT_NE(prepare_count, nullptr);
  ASSERT_NE(destroy_count, nullptr);
  {
    auto loaded = pops::component::LoadedComponent::load(library.string(), expected());
    const auto execution = abi::host_execution_context();
    auto anonymous_execution = execution;
    anonymous_execution.execution_identity = "";
    EXPECT_THROW(
        (void)loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, anonymous_execution,
                                    R"({"scheme":"shared"})", R"({"identity":"target-a"})"),
        std::invalid_argument);
    EXPECT_EQ(prepare_count(), 0);
    auto incomplete_execution = execution;
    incomplete_execution.backend_identity = nullptr;
    EXPECT_THROW((void)loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
                                             incomplete_execution, R"({"scheme":"shared"})",
                                             R"({"identity":"target-a"})"),
                 std::invalid_argument);
    EXPECT_EQ(prepare_count(), 0);
    void* first = loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
                                        R"({"scheme":"shared"})", R"({"identity":"target-a"})");
    EXPECT_EQ(prepare_count(), 1);
    EXPECT_EQ(loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
                                    R"({"scheme":"shared"})", R"({"identity":"target-a"})"),
              first);
    EXPECT_EQ(prepare_count(), 1);
    void* second = loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
                                         R"({"scheme":"shared"})", R"({"identity":"target-b"})");
    EXPECT_NE(second, first);
    EXPECT_EQ(prepare_count(), 2);

    auto mismatched_context = execution;
    mismatched_context.execution_identity = "test::other-execution-context";
    EXPECT_THROW(
        (void)loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, mismatched_context,
                                    R"({"scheme":"shared"})", R"({"identity":"target-c"})"),
        std::invalid_argument);
    EXPECT_EQ(prepare_count(), 2);
    EXPECT_EQ(destroy_count(), 0);
  }
  EXPECT_EQ(destroy_count(), 2);
  pops::dynlib::close(inspection);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, FreshSessionStatesAreIndependentMoveOnlyRaiiOwners) {
  static_assert(!std::is_copy_constructible_v<pops::component::LoadedComponent::PreparedState>);
  static_assert(
      std::is_nothrow_move_constructible_v<pops::component::LoadedComponent::PreparedState>);

  const auto library = compile_component();
  const auto inspection = pops::dynlib::open(library.string());
  ASSERT_TRUE(pops::dynlib::valid(inspection));
  using CounterFn = int (*)();
  const auto prepare_count =
      reinterpret_cast<CounterFn>(pops::dynlib::sym(inspection, "pops_test_prepare_count"));
  const auto destroy_count =
      reinterpret_cast<CounterFn>(pops::dynlib::sym(inspection, "pops_test_destroy_count"));
  ASSERT_NE(prepare_count, nullptr);
  ASSERT_NE(destroy_count, nullptr);
  {
    auto loaded = pops::component::LoadedComponent::load(library.string(), expected());
    const auto execution = abi::host_execution_context();
    {
      auto first =
          loaded.prepare_fresh_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
                                     R"({"scheme":"session"})", R"({"identity":"same-target"})");
      auto second =
          loaded.prepare_fresh_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
                                     R"({"scheme":"session"})", R"({"identity":"same-target"})");
      EXPECT_NE(first.get(), second.get());
      EXPECT_EQ(prepare_count(), 2);
      EXPECT_EQ(destroy_count(), 0);

      auto moved = std::move(first);
      EXPECT_EQ(first.get(), nullptr);
      EXPECT_NE(moved.get(), nullptr);
    }
    EXPECT_EQ(destroy_count(), 2);
  }
  pops::dynlib::close(inspection);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesIdentityInterfaceAndTableSizeMismatches) {
  const auto library = compile_component();
  auto forged = expected();
  forged.semantic_identity = "forged-semantic";
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), forged),
               std::runtime_error);

  auto undeclared_export = expected();
  undeclared_export.interfaces.pop_back();
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), undeclared_export),
               std::runtime_error);

  auto duplicate_expectation = expected();
  duplicate_expectation.interfaces.push_back(duplicate_expectation.interfaces.front());
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), duplicate_expectation),
               std::runtime_error);

  auto missing = expected();
  missing.interfaces = {{POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1)}};
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), missing),
               std::runtime_error);

  auto truncated = expected();
  truncated.interfaces[0].minimum_table_size = sizeof(PopsNumericalFluxApiV1) + 1;
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), truncated),
               std::runtime_error);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesHonestlyReportedHeaderOnlyInterfaceTable) {
  const auto library = compile_component(FluxTableFixture::HeaderOnly);
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), expected()),
               std::runtime_error);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesHeaderOnlyTableWithForgedFullEntrySize) {
  const auto library = compile_component(FluxTableFixture::ForgedEntrySize);
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), expected()),
               std::runtime_error);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesComponentBuiltForAnotherNativeAbi) {
  const auto library = compile_component(FluxTableFixture::WrongAbi);
  EXPECT_THROW(pops::component::LoadedComponent::load(library.string(), expected()),
               std::runtime_error);
  std::filesystem::remove(library);
}

}  // namespace
