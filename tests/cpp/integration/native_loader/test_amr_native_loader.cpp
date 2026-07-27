#include <gtest/gtest.h>

#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/mesh/boundary/prepared_boundary_plan.hpp>
#include <pops/runtime/amr/prepared_component_providers.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include "component_abi_test_helpers.hpp"
#include "native_dso_compiler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
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
        5,
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

TEST(test_amr_native_loader, PreparedAmrProvidersExecuteExactTablesAndProvenance) {
  const auto library = compile_component();
  const auto inspection = pops::dynlib::open(library.string());
  ASSERT_TRUE(pops::dynlib::valid(inspection));
  using CounterFn = int (*)();
  using SetIntFn = void (*)(int);
  using PointerFn = const void* (*)();
  using TickFn = std::int64_t (*)();
  using PhysicalTimeFn = double (*)();
  const auto tag_call_count =
      reinterpret_cast<CounterFn>(pops::dynlib::sym(inspection, "pops_test_tag_call_count"));
  const auto set_partial_tag_output =
      reinterpret_cast<SetIntFn>(pops::dynlib::sym(inspection, "pops_test_set_partial_tag_output"));
  const auto last_tag_state_data =
      reinterpret_cast<PointerFn>(pops::dynlib::sym(inspection, "pops_test_last_tag_state_data"));
  const auto last_tag_logical_tick =
      reinterpret_cast<TickFn>(pops::dynlib::sym(inspection, "pops_test_last_tag_logical_tick"));
  const auto last_tag_physical_time = reinterpret_cast<PhysicalTimeFn>(
      pops::dynlib::sym(inspection, "pops_test_last_tag_physical_time"));
  ASSERT_NE(tag_call_count, nullptr);
  ASSERT_NE(set_partial_tag_output, nullptr);
  ASSERT_NE(last_tag_state_data, nullptr);
  ASSERT_NE(last_tag_logical_tick, nullptr);
  ASSERT_NE(last_tag_physical_time, nullptr);
  {
    auto component = std::make_shared<pops::component::LoadedComponent>(
        pops::component::LoadedComponent::load(library.string(), expected()));
    const auto execution = prepared_execution();
    pops::runtime::amr::PreparedTaggerSpec tagger_spec{
        "test::tagger-provider",
        kComponentId,
        kManifestIdentity,
        "case::layout",
        "case::clock",
        {1, 2, 3, 4, 5},
        {16, 17, 18},
        {POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1},
        POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1,
        128,
        POPS_TAGGING_NON_FINITE_REJECT_V1,
        POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2,
        POPS_TAGGER_COLLECTIVE_NONE_V2,
        {POPS_MEMORY_SPACE_HOST_V1},
        2,
        execution};
    pops::runtime::amr::PreparedClusteringSpec clustering_spec{
        "test::clustering-provider", kComponentId, kManifestIdentity, "case::layout", 1, execution};
    pops::runtime::amr::PreparedTaggerComponent tagger(std::move(tagger_spec), component);
    pops::runtime::amr::PreparedClusteringComponent clustering(std::move(clustering_spec),
                                                               component);
    const pops::runtime::amr::PreparedTaggingProgram program{
        {},
        {{0, 0, 1, 0.5, POPS_TAGGING_NO_STENCIL_V1}},
        {1},
        {0},
        {},
        {},
        0,
        0,
        0,
        POPS_TAGGING_NON_FINITE_REJECT_V1,
        "case::clock",
        "case::bound-tagging-program",
        true};

    const pops::Box2D domain{{0, 0}, {3, 2}};
    const std::vector<pops::Box2D> patches{pops::Box2D{{0, 0}, {1, 2}},
                                           pops::Box2D{{2, 0}, {3, 2}}};
    pops::MultiFab state(pops::BoxArray(patches), pops::DistributionMapping(2, pops::n_ranks()), 1,
                         1);
    state.set_val(pops::Real(0));
    for (int local = 0; local < state.local_size(); ++local) {
      auto values = state.fab(local).array();
      if (state.box(local).contains(1, 0))
        values(1, 0, 0) = pops::Real(2);
      if (state.box(local).contains(2, 1))
        values(2, 1, 0) = pops::Real(3);
    }
    const int calls_before = tag_call_count();
    auto candidates = tagger.tag({{"case::tracer::U", &state}}, program, domain, 0, 7, 0.25, 0.25,
                                 1.0 / 3.0, false, false, false);
    pops::TagBox& tags = candidates.refine;
    EXPECT_EQ(tag_call_count() - calls_before, state.local_size());
    ASSERT_GT(state.local_size(), 0);
    EXPECT_EQ(last_tag_state_data(), state.fab(state.local_size() - 1).data())
        << "native-backend Tagger must borrow the resident field instead of staging HostSpace";
    ASSERT_EQ(tags.count(), 2);
    EXPECT_TRUE(tags.tagged(1, 0));
    EXPECT_TRUE(tags.tagged(2, 1));
    ASSERT_EQ(tagger.provider_identity(), "test::tagger-provider");

    pops::runtime::amr::PreparedTaggerSpec host_tagger_spec{
        "test::host-tagger-provider",
        kComponentId,
        kManifestIdentity,
        "case::layout",
        "case::clock",
        {1, 2, 3, 4, 5},
        {16, 17, 18},
        {POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1},
        POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1,
        128,
        POPS_TAGGING_NON_FINITE_REJECT_V1,
        POPS_TAGGER_EXECUTION_HOST_V2,
        POPS_TAGGER_COLLECTIVE_NONE_V2,
        {POPS_MEMORY_SPACE_HOST_V1},
        2,
        execution};
    pops::runtime::amr::PreparedTaggerComponent host_tagger(std::move(host_tagger_spec), component);
    const auto host_candidates = host_tagger.tag({{"case::tracer::U", &state}}, program, domain, 0,
                                                 7, 0.25, 0.25, 1.0 / 3.0, false, false, false);
    EXPECT_EQ(host_candidates.refine.count(), candidates.refine.count());
    EXPECT_NE(last_tag_state_data(), state.fab(state.local_size() - 1).data())
        << "only an explicitly host-scoped Tagger may receive a staged field image";

    set_partial_tag_output(1);
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, program, domain, 0, 7, 0.25, 0.25,
                                  1.0 / 3.0, false, false, false),
                 std::runtime_error);
    set_partial_tag_output(0);

    const auto set_state = [&state](int i, int j, pops::Real value) {
      for (int local = 0; local < state.local_size(); ++local)
        if (state.box(local).contains(i, j))
          state.fab(local).array()(i, j, 0) = value;
    };
    set_state(0, 0, std::numeric_limits<pops::Real>::quiet_NaN());
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, program, domain, 0, 7, 0.25, 0.25,
                                  1.0 / 3.0, false, false, false),
                 std::runtime_error);
    set_state(0, 0, pops::Real(0));

    auto exact_gradient = program;
    using TagProgram = pops::runtime::amr::PreparedTaggingProgram;
    exact_gradient.stencils = {
        TagProgram::Stencil{"case::forward-gradient",
                            POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1,
                            "l2",
                            "inverse_cell_size",
                            "ghost_extension",
                            2,
                            {TagProgram::AxisStencil{0, 1, 1, 0, 1, {0, 1}, {-1.0, 1.0}},
                             TagProgram::AxisStencil{1, 1, 1, 0, 1, {0, 1}, {-1.0, 1.0}}}}};
    exact_gradient.leaves[0].opcode = POPS_TAGGING_GRADIENT_ABOVE_V1;
    exact_gradient.leaves[0].threshold = 6.0;
    exact_gradient.leaves[0].stencil_index = 0;
    exact_gradient.refine_ops = {POPS_TAGGING_GRADIENT_ABOVE_V1};
    exact_gradient.provider_identity = "case::bound-tagging-program-forward-gradient";
    const auto gradient_candidates =
        tagger.tag({{"case::tracer::U", &state}}, exact_gradient, domain, 0, 7, 0.25, 0.25,
                   1.0 / 3.0, false, false, false);
    EXPECT_TRUE(gradient_candidates.refine.tagged(0, 0));
    EXPECT_GT(gradient_candidates.refine.count(), 0);

    set_state(0, 0, std::numeric_limits<pops::Real>::quiet_NaN());
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, exact_gradient, domain, 0, 7, 0.25,
                                  0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);
    set_state(0, 0, pops::Real(0));

    set_state(0, 0, -std::numeric_limits<pops::Real>::max());
    set_state(1, 0, std::numeric_limits<pops::Real>::max());
    const int calls_before_derived_overflow = tag_call_count();
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, exact_gradient, domain, 0, 7, 0.25,
                                  0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);
    EXPECT_GT(tag_call_count(), calls_before_derived_overflow);
    set_state(0, 0, pops::Real(0));
    set_state(1, 0, pops::Real(2));

    auto false_order = exact_gradient;
    false_order.stencils[0].axes[0].formal_order = 2;
    false_order.provider_identity = "case::forged-gradient-order";
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, false_order, domain, 0, 7, 0.25,
                                  0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);

    auto insufficient_halo = exact_gradient;
    insufficient_halo.stencils[0].axes[0] =
        TagProgram::AxisStencil{0, 1, 1, 0, 2, {0, 2}, {-0.5, 0.5}};
    insufficient_halo.provider_identity = "case::gradient-halo-too-thin";
    const int calls_before_halo_rejection = tag_call_count();
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, insufficient_halo, domain, 0, 7,
                                  0.25, 0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);
    EXPECT_EQ(tag_call_count(), calls_before_halo_rejection);

    auto minimum_offset = exact_gradient;
    minimum_offset.stencils[0].axes[0] = TagProgram::AxisStencil{
        0, 1, 1, 0, 0, {std::numeric_limits<std::int32_t>::min(), 0}, {-1.0, 1.0}};
    minimum_offset.provider_identity = "case::gradient-minimum-offset";
    const int calls_before_minimum_offset = tag_call_count();
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, minimum_offset, domain, 0, 7, 0.25,
                                  0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);
    EXPECT_EQ(tag_call_count(), calls_before_minimum_offset);

    auto not_equality = program;
    not_equality.leaves[0].threshold = 0.0;
    not_equality.refine_ops = {POPS_TAGGING_ABOVE_V1, POPS_TAGGING_NOT_V1};
    not_equality.refine_args = {0, 1};
    not_equality.provider_identity = "case::bound-tagging-program-not-equality";
    const auto not_candidates = tagger.tag({{"case::tracer::U", &state}}, not_equality, domain, 0,
                                           7, 0.25, 0.25, 1.0 / 3.0, false, false, false);
    EXPECT_EQ(not_candidates.refine.count(), 0);
    EXPECT_EQ(not_candidates.refine_equalities.count(), 10);
    set_state(0, 0, std::numeric_limits<pops::Real>::quiet_NaN());
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, not_equality, domain, 0, 7, 0.25,
                                  0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);
    set_state(0, 0, pops::Real(0));

    auto any_equality = program;
    any_equality.leaves[0].threshold = 0.0;
    any_equality.leaves.push_back({0, 0, POPS_TAGGING_BELOW_V1, -1.0, POPS_TAGGING_NO_STENCIL_V1});
    any_equality.refine_ops = {POPS_TAGGING_ABOVE_V1, POPS_TAGGING_BELOW_V1,
                               POPS_TAGGING_ANY_OF_V1};
    any_equality.refine_args = {0, 1, 2};
    any_equality.provider_identity = "case::bound-tagging-program-any-equality";
    const auto any_candidates = tagger.tag({{"case::tracer::U", &state}}, any_equality, domain, 0,
                                           7, 0.25, 0.25, 1.0 / 3.0, false, false, false);
    EXPECT_EQ(any_candidates.refine.count(), 2);
    EXPECT_EQ(any_candidates.refine_equalities.count(), 10);

    auto all_equality = program;
    all_equality.leaves[0].threshold = 0.0;
    all_equality.leaves.push_back({0, 0, POPS_TAGGING_BELOW_V1, 1.0, POPS_TAGGING_NO_STENCIL_V1});
    all_equality.refine_ops = {POPS_TAGGING_ABOVE_V1, POPS_TAGGING_BELOW_V1,
                               POPS_TAGGING_ALL_OF_V1};
    all_equality.refine_args = {0, 1, 2};
    all_equality.provider_identity = "case::bound-tagging-program-all-equality";
    const auto all_candidates = tagger.tag({{"case::tracer::U", &state}}, all_equality, domain, 0,
                                           7, 0.25, 0.25, 1.0 / 3.0, false, false, false);
    EXPECT_EQ(all_candidates.refine.count(), 0);
    EXPECT_EQ(all_candidates.refine_equalities.count(), 10);

    auto changed_threshold = program;
    changed_threshold.leaves[0].threshold = 2.5;
    changed_threshold.provider_identity = "case::bound-tagging-program-threshold-2.5";
    const auto changed = tagger.tag({{"case::tracer::U", &state}}, changed_threshold, domain, 0, 7,
                                    0.25, 0.25, 1.0 / 3.0, false, false, false);
    EXPECT_EQ(changed.refine.count(), 1);
    EXPECT_TRUE(changed.refine.tagged(2, 1));

    auto refine_and_coarsen = program;
    refine_and_coarsen.leaves.push_back({0, 0, 2, 1.0, POPS_TAGGING_NO_STENCIL_V1});
    refine_and_coarsen.coarsen_ops = {2};
    refine_and_coarsen.coarsen_args = {1};
    refine_and_coarsen.provider_identity = "case::bound-tagging-program-with-coarsen";
    const auto dual = tagger.tag({{"case::tracer::U", &state}}, refine_and_coarsen, domain, 0, 7,
                                 0.25, 0.25, 1.0 / 3.0, false, false, false);
    EXPECT_EQ(dual.refine.count(), 2);
    EXPECT_EQ(dual.coarsen.count(), 10);

    // A direct/restarted Program context must publish its current evaluation coordinate at the
    // tagger boundary. The facade clock is changed only after the lazy runtime build, so observing
    // either build-time zero here would prove that stale runtime metadata leaked into regrid.
    pops::AmrSystemConfig config;
    config.n = 4;
    config.ny = 4;
    config.L = 1.0;
    config.Ly = 1.0;
    config.level_count = 2;
    config.regrid_every = 1;
    config.periodicity = {true, true};
    pops::AmrSystem system(config);
    system.set_temporal_relations({2}, {1}, {"integral_only"});
    pops::ModelSpec model;
    model.transport = "compressible";
    model.source = "none";
    model.elliptic = "background";
    model.gamma = 1.4;
    model.alpha = 0.0;
    model.n0 = 0.0;
    system.add_block("gas", model, "minmod", "rusanov", "conservative", "explicit", 1);
    const std::size_t cells = static_cast<std::size_t>(config.n) * config.ny;
    std::vector<double> conservative(4u * cells, 0.0);
    std::fill_n(conservative.data(), cells, 1.0);
    std::fill_n(conservative.data() + 3u * cells, cells, 2.5);
    system.set_conservative_state("gas", conservative);
    system.set_refinement(0.5);
    pops::runtime::amr::PreparedTaggerSpec context_tagger_spec{
        "test::direct-context-tagger-provider",
        kComponentId,
        kManifestIdentity,
        "case::direct-context-layout",
        "amr::refinement",
        {1, 2, 3, 4, 5},
        {16, 17, 18},
        {POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1},
        POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1,
        128,
        POPS_TAGGING_NON_FINITE_REJECT_V1,
        POPS_TAGGER_EXECUTION_HOST_V2,
        POPS_TAGGER_COLLECTIVE_NONE_V2,
        {POPS_MEMORY_SPACE_HOST_V1},
        2,
        execution};
    system.install_amr_tagger_component(std::move(context_tagger_spec), component);
    ASSERT_TRUE(system.uses_runtime_engine());
    ASSERT_NE(system.engine(), nullptr);
    system.set_clock(2.75, 99);
    pops::runtime::program::AmrProgramContext direct_context(system.engine(), &system);
    const int direct_calls_before = tag_call_count();
    direct_context.regrid_if_due(1);
    EXPECT_GT(tag_call_count(), direct_calls_before);
    EXPECT_EQ(last_tag_logical_tick(), 1);
    EXPECT_DOUBLE_EQ(last_tag_physical_time(), 2.75);

    // GLOBAL Program substeps share one public AMR macro-step. They must cover distinct physical
    // intervals while the head-of-step regrid/tagger runs exactly once, not once per closure call.
    system.set_clock(3.0, 1);
    system.set_program_block_map({0});
    system.set_program_cadence(/*substeps=*/2, /*stride=*/1);
    pops::runtime::program::AmrProgramContext cadence_context(system.engine(), &system);
    cadence_context.configure_primary_clock("clock.macro");
    std::vector<double> cadence_times;
    std::vector<int> cadence_macro_steps;
    std::vector<int> cadence_tag_counts;
    cadence_context.install([&](double step) {
      cadence_times.push_back(static_cast<double>(cadence_context.physical_time()));
      cadence_macro_steps.push_back(cadence_context.macro_step());
      cadence_context.advance_hierarchy(step, [&](double level_step) {
        cadence_context.set_stage_time(0, 1);
        pops::MultiFab& state = cadence_context.state(0);
        pops::MultiFab& rate = cadence_context.rhs_scratch(101, 0, state);
        cadence_context.rhs_into(0, state, rate, 101);
        cadence_context.axpy(state, static_cast<pops::Real>(level_step), rate);
      });
      cadence_tag_counts.push_back(tag_call_count());
    });
    const int cadence_calls_before = tag_call_count();
    system.step(0.2);
    ASSERT_EQ(cadence_times.size(), 2u);
    ASSERT_EQ(cadence_macro_steps.size(), 2u);
    ASSERT_EQ(cadence_tag_counts.size(), 2u);
    EXPECT_NEAR(cadence_times[0], 3.0, 1.0e-14);
    EXPECT_NEAR(cadence_times[1], 3.1, 1.0e-14);
    EXPECT_EQ(cadence_macro_steps[0], 1);
    EXPECT_EQ(cadence_macro_steps[1], 1);
    EXPECT_GT(cadence_tag_counts[0], cadence_calls_before);
    EXPECT_EQ(cadence_tag_counts[1], cadence_tag_counts[0]);
    EXPECT_EQ(last_tag_logical_tick(), 1);
    EXPECT_DOUBLE_EQ(last_tag_physical_time(), 3.0);
    EXPECT_NEAR(system.time(), 3.2, 1.0e-14);
    EXPECT_EQ(system.macro_step(), 2);

    const auto configure_cadenced_system = [&](pops::AmrSystem& target,
                                               const std::string& provider_identity,
                                               const std::string& layout_identity) {
      target.set_temporal_relations({2}, {1}, {"integral_only"});
      target.add_block("gas", model, "minmod", "rusanov", "conservative", "explicit", 1);
      target.set_conservative_state("gas", conservative);
      target.set_refinement(0.5);
      pops::runtime::amr::PreparedTaggerSpec spec{
          provider_identity,
          kComponentId,
          kManifestIdentity,
          layout_identity,
          "amr::refinement",
          {1, 2, 3, 4, 5},
          {16, 17, 18},
          {POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1},
          POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1,
          128,
          POPS_TAGGING_NON_FINITE_REJECT_V1,
          POPS_TAGGER_EXECUTION_HOST_V2,
          POPS_TAGGER_COLLECTIVE_NONE_V2,
          {POPS_MEMORY_SPACE_HOST_V1},
          2,
          execution};
      target.install_amr_tagger_component(std::move(spec), component);
      ASSERT_TRUE(target.uses_runtime_engine());
      ASSERT_NE(target.engine(), nullptr);
      target.set_program_block_map({0});
    };

    // A stride-two Program closes windows at facade ticks 1, 3, 5, ... but regrid cadence belongs
    // to the accepted Program clock at the START of each window.  The second window must therefore
    // tag at tick 2 / t=0.2.  Passing the facade cursor (the old behavior) yields 1, 3, 5 and this
    // regrid_every=2 fixture never calls the tagger.
    pops::AmrSystemConfig stride_config = config;
    stride_config.regrid_every = 2;
    pops::AmrSystem stride_system(stride_config);
    configure_cadenced_system(stride_system, "test::stride-window-tagger-provider",
                              "case::stride-window-layout");
    stride_system.set_clock(0.0, 0);
    stride_system.set_program_cadence(/*substeps=*/1, /*stride=*/2);
    pops::runtime::program::AmrProgramContext stride_context(stride_system.engine(),
                                                             &stride_system);
    stride_context.configure_primary_clock("clock.macro");
    std::vector<double> stride_context_times;
    std::vector<int> stride_context_macro_steps;
    stride_context.install([&](double step) {
      stride_context_macro_steps.push_back(stride_context.macro_step());
      stride_context.advance_hierarchy(step, [&](double level_step) {
        if (stride_context.level() == 0)
          stride_context_times.push_back(static_cast<double>(stride_context.physical_time()));
        stride_context.set_stage_time(0, 1);
        pops::MultiFab& state_value = stride_context.state(0);
        pops::MultiFab& rate = stride_context.rhs_scratch(201, 0, state_value);
        stride_context.rhs_into(0, state_value, rate, 201);
        stride_context.axpy(state_value, static_cast<pops::Real>(level_step), rate);
      });
    });
    const int stride_calls_before = tag_call_count();
    stride_system.step(0.05);  // held
    stride_system.step(0.15);  // first variable-dt window starts at tick 0
    EXPECT_EQ(tag_call_count(), stride_calls_before);
    stride_system.step(0.07);  // held
    stride_system.step(0.13);  // second window starts at accepted tick 2: regrid is due
    ASSERT_EQ(stride_context_times.size(), 2u);
    ASSERT_EQ(stride_context_macro_steps.size(), 2u);
    EXPECT_NEAR(stride_context_times[0], 0.0, 1.0e-14);
    EXPECT_NEAR(stride_context_times[1], 0.2, 1.0e-14);
    EXPECT_EQ(stride_context_macro_steps[0], 0);
    EXPECT_EQ(stride_context_macro_steps[1], 2);
    EXPECT_GT(tag_call_count(), stride_calls_before);
    EXPECT_EQ(last_tag_logical_tick(), 2);
    EXPECT_NEAR(last_tag_physical_time(), 0.2, 1.0e-14);
    EXPECT_NEAR(stride_system.time(), 0.4, 1.0e-14);
    EXPECT_EQ(stride_system.macro_step(), 4);

    // The first GLOBAL substep commits an AMR-context image before the second starts.  If substep
    // two fails, the facade transaction rolls back farther than that inner image.  A retry must
    // import the original accepted context, repeat the head-of-window regrid/tagger, and traverse
    // exactly the same physical coordinates instead of retaining the substep-one marker/clock.
    pops::AmrSystemConfig retry_config = config;
    retry_config.regrid_every = 2;
    pops::AmrSystem retry_system(retry_config);
    configure_cadenced_system(retry_system, "test::substep-retry-tagger-provider",
                              "case::substep-retry-layout");
    retry_system.set_clock(1.25, 2);
    retry_system.set_program_cadence(/*substeps=*/2, /*stride=*/1);
    pops::runtime::program::AmrProgramContext retry_context(retry_system.engine(), &retry_system);
    retry_context.configure_primary_clock("clock.macro");
    bool reject_second_substep = true;
    int program_invocation = 0;
    std::vector<double> retry_context_times;
    std::vector<int> retry_context_macro_steps;
    retry_context.install([&](double step) {
      ++program_invocation;
      retry_context_macro_steps.push_back(retry_context.macro_step());
      retry_context.advance_hierarchy(step, [&](double level_step) {
        if (retry_context.level() == 0)
          retry_context_times.push_back(static_cast<double>(retry_context.physical_time()));
        retry_context.set_stage_time(0, 1);
        pops::MultiFab& state_value = retry_context.state(0);
        pops::MultiFab& rate = retry_context.rhs_scratch(301, 0, state_value);
        retry_context.rhs_into(0, state_value, rate, 301);
        retry_context.axpy(state_value, static_cast<pops::Real>(level_step), rate);
        if (reject_second_substep && program_invocation == 2 && retry_context.level() == 0)
          throw std::runtime_error("intentional second AMR Program substep failure");
      });
    });

    const double retry_time_before = retry_system.time();
    const int retry_macro_before = retry_system.macro_step();
    const int retry_levels_before = retry_system.engine()->nlev();
    const int retry_regrids_before = retry_system.checkpoint_regrid_count();
    const std::uint64_t retry_topology_before = retry_system.checkpoint_topology_epoch();
    const std::vector<double> retry_state_before = retry_system.engine()->block_level_state(0, 0);
    const std::vector<std::uint8_t> retry_program_state_before =
        retry_system.program_accepted_state();
    const std::uint64_t retry_program_revision_before =
        retry_system.program_accepted_state_revision();
    const int retry_calls_before = tag_call_count();

    EXPECT_THROW(retry_system.step(0.2), std::runtime_error);
    ASSERT_EQ(retry_context_times.size(), 2u);
    ASSERT_EQ(retry_context_macro_steps.size(), 2u);
    EXPECT_NEAR(retry_context_times[0], 1.25, 1.0e-14);
    EXPECT_NEAR(retry_context_times[1], 1.35, 1.0e-14);
    EXPECT_EQ(retry_context_macro_steps[0], retry_macro_before);
    EXPECT_EQ(retry_context_macro_steps[1], retry_macro_before);
    EXPECT_GT(tag_call_count(), retry_calls_before);
    EXPECT_EQ(last_tag_logical_tick(), retry_macro_before);
    EXPECT_NEAR(last_tag_physical_time(), retry_time_before, 1.0e-14);
    EXPECT_DOUBLE_EQ(retry_system.time(), retry_time_before);
    EXPECT_EQ(retry_system.macro_step(), retry_macro_before);
    EXPECT_EQ(retry_system.engine()->nlev(), retry_levels_before);
    EXPECT_EQ(retry_system.checkpoint_regrid_count(), retry_regrids_before);
    EXPECT_EQ(retry_system.checkpoint_topology_epoch(), retry_topology_before);
    EXPECT_EQ(retry_system.engine()->block_level_state(0, 0), retry_state_before);
    EXPECT_EQ(retry_system.program_accepted_state(), retry_program_state_before);
    EXPECT_EQ(retry_system.program_accepted_state_revision(), retry_program_revision_before);
    EXPECT_NEAR(static_cast<double>(retry_context.physical_time()), retry_time_before, 1.0e-14);

    reject_second_substep = false;
    const int retry_calls_after_failure = tag_call_count();
    retry_system.step(0.2);
    ASSERT_EQ(retry_context_times.size(), 4u);
    ASSERT_EQ(retry_context_macro_steps.size(), 4u);
    EXPECT_NEAR(retry_context_times[2], retry_context_times[0], 1.0e-14);
    EXPECT_NEAR(retry_context_times[3], retry_context_times[1], 1.0e-14);
    EXPECT_EQ(retry_context_macro_steps[2], retry_macro_before);
    EXPECT_EQ(retry_context_macro_steps[3], retry_macro_before);
    EXPECT_GT(tag_call_count(), retry_calls_after_failure)
        << "rollback must not retain the failed attempt's automatic-regrid marker";
    EXPECT_EQ(last_tag_logical_tick(), retry_macro_before);
    EXPECT_NEAR(last_tag_physical_time(), retry_time_before, 1.0e-14);
    EXPECT_NEAR(retry_system.time(), retry_time_before + 0.2, 1.0e-14);
    EXPECT_EQ(retry_system.macro_step(), retry_macro_before + 1);
    EXPECT_GT(retry_system.checkpoint_regrid_count(), retry_regrids_before);
    EXPECT_GT(retry_system.checkpoint_topology_epoch(), retry_topology_before);

    auto unsupported_hysteresis = program;
    unsupported_hysteresis.min_cycles = 1;
    unsupported_hysteresis.provider_identity = "case::bound-tagging-program-hysteresis-1";
    const int calls_before_hysteresis = tag_call_count();
    EXPECT_THROW((void)tagger.tag({{"case::tracer::U", &state}}, unsupported_hysteresis, domain, 0,
                                  7, 0.25, 0.25, 1.0 / 3.0, false, false, false),
                 std::runtime_error);
    EXPECT_EQ(tag_call_count(), calls_before_hysteresis);

    const std::vector<pops::Box2D> boxes = clustering.cluster(tags);
    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_EQ(boxes[0].lo[0], 1);
    EXPECT_EQ(boxes[0].lo[1], 0);
    EXPECT_EQ(boxes[0].hi[0], 2);
    EXPECT_EQ(boxes[0].hi[1], 1);
    ASSERT_EQ(clustering.provider_identity(), "test::clustering-provider");
  }
  pops::dynlib::close(inspection);
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

TEST(test_amr_native_loader, BoundaryPlanSessionsOwnFreshLaneQualifiedComponentStates) {
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
    auto component = std::make_shared<pops::component::LoadedComponent>(
        pops::component::LoadedComponent::load(library.string(), expected()));
    pops::PreparedBoundaryComponentSpec spec;
    spec.target_identity = "case::boundary::ghost-target";
    spec.component_id = kComponentId;
    spec.manifest_identity = kManifestIdentity;
    spec.interface_version = 1;
    spec.producer_identity = "case::boundary::ghost-producer";
    spec.state_identity = "case::state::u";
    spec.ghost_identity = "case::boundary::xlo";
    spec.layout_identity = "case::layout::cells";
    spec.region.kind = POPS_BOUNDARY_FACE_V1;
    spec.region.dimension = 2;
    spec.region.codimension = 1;
    spec.region.axes = {0};
    spec.region.sides = {-1};
    spec.region.identity = "case::boundary::xlo";
    spec.parameters_json = R"({"coefficient":1.0})";
    spec.target_json = R"({"identity":"case::boundary::ghost-target"})";
    spec.execution = prepared_execution();

    pops::BCRec bc;
    bc.xlo = pops::BCType::Foextrap;
    bc.xhi = pops::BCType::Foextrap;
    bc.ylo = pops::BCType::Foextrap;
    bc.yhi = pops::BCType::Foextrap;
    pops::PreparedBoundaryPlan plan("case::boundary::plan", 1, {bc}, {}, spec.state_identity);
    plan.install_ghost_component(std::move(spec), component);

    const auto lane =
        pops::ExecutionLane::duplicate_world_collectively("case::boundary::component-session");
    {
      auto first = plan.make_session(lane);
      auto second = plan.make_session(lane);
      EXPECT_EQ(prepare_count(), 2);
      EXPECT_EQ(destroy_count(), 0);

      const pops::Box2D domain = pops::Box2D::from_extents(3, 3);
      const pops::BoxArray boxes = pops::BoxArray::from_domain(domain, domain.nx());
      pops::MultiFab state(boxes, pops::DistributionMapping(boxes.size(), pops::n_ranks()), 1, 1);
      state.set_val(pops::Real(9));
      const pops::Geometry geometry(domain, pops::Real(0), pops::Real(1), pops::Real(0),
                                    pops::Real(1));
      const pops::runtime::multiblock::BoundaryEvaluationPoint point{
          "clock.boundary-session", 0, 0, 0, 0, pops::amr::Rational(0, 1), 0.1, 0.0};
      pops::detail::BoundaryFieldRegistry fields;
      fields.configure_states(plan.required_state_identities());
      fields.begin_binding();
      fields.bind_state(plan.state_identity(), state);
      first.prepare_ghost_executor(state, fields, geometry);
      second.prepare_ghost_executor(state, fields, geometry);

      first.fill_same_level_and_physical(state, fields, geometry, point);
      if (state.local_size() != 0)
        EXPECT_EQ(state.fab(0)(-1, 1, 0), pops::Real(1));
      second.fill_same_level_and_physical(state, fields, geometry, point);
      if (state.local_size() != 0)
        EXPECT_EQ(state.fab(0)(-1, 1, 0), pops::Real(2));
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
