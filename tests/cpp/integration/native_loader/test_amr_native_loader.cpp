#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

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
constexpr int kDim = pops::kNativeDimension;

template <std::size_t Dim>
std::size_t extent_product(const std::array<std::size_t, Dim>& extents) {
  std::size_t result = 1;
  for (const std::size_t extent : extents)
    result *= extent;
  return result;
}

template <int Dim>
PopsConstFieldViewV1 const_field_view(const double* data,
                                      const std::array<std::size_t, Dim>& extents,
                                      std::size_t components) {
  PopsConstFieldViewV1 result{};
  result.struct_size = sizeof(PopsConstFieldViewV1);
  result.data = data;
  result.dimension = Dim;
  std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(components);
  for (int axis = 0; axis < Dim; ++axis) {
    result.extents[axis] = extents[axis];
    result.axis_strides[axis] = stride;
    stride *= static_cast<std::ptrdiff_t>(extents[axis]);
  }
  result.component_count = components;
  result.component_stride = 1;
  result.centering = POPS_FIELD_CENTERING_CELL_V1;
  result.scalar_type = POPS_SCALAR_FLOAT64_V1;
  result.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  result.layout_identity = "test::layout";
  result.patch_identity = "test::patch";
  result.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  return result;
}

template <int Dim>
PopsFieldViewV1 field_view(double* data, const std::array<std::size_t, Dim>& extents,
                           std::size_t components) {
  PopsFieldViewV1 result{};
  result.struct_size = sizeof(PopsFieldViewV1);
  result.data = data;
  result.dimension = Dim;
  std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(components);
  for (int axis = 0; axis < Dim; ++axis) {
    result.extents[axis] = extents[axis];
    result.axis_strides[axis] = stride;
    stride *= static_cast<std::ptrdiff_t>(extents[axis]);
  }
  result.component_count = components;
  result.component_stride = 1;
  result.centering = POPS_FIELD_CENTERING_CELL_V1;
  result.scalar_type = POPS_SCALAR_FLOAT64_V1;
  result.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  result.layout_identity = "test::layout";
  result.patch_identity = "test::patch";
  result.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  return result;
}

enum class FluxTableFixture { Exact, HeaderOnly, ForgedEntrySize, WrongAbi };

constexpr const char* kComponentId = "pops://test/final-flux@1.0.0";
constexpr const char* kSemanticIdentity = "semantic-final-flux";
constexpr const char* kManifestIdentity = "manifest-final-flux";

std::string component_source() {
  return R"CPP(
#include <pops/runtime/config/generated_component_abi.hpp>
#include <algorithm>
#include <array>
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

    template <class View>
    std::size_t point_count(const View& view) {
      std::size_t result = 1;
      for (std::int32_t axis = 0; axis < view.dimension; ++axis)
        result *= view.extents[axis];
      return result;
    }

    template <class View>
    std::ptrdiff_t field_offset(const View& view, std::size_t point, std::size_t component) {
      std::ptrdiff_t result = static_cast<std::ptrdiff_t>(component) * view.component_stride;
      for (std::int32_t axis = 0; axis < view.dimension; ++axis) {
        const std::size_t coordinate = point % view.extents[axis];
        point /= view.extents[axis];
        result += static_cast<std::ptrdiff_t>(coordinate) * view.axis_strides[axis];
      }
      return result;
    }

    int evaluate(void*, const PopsNumericalFluxRequestV1* request, PopsNumericalFluxResultV1* result) {
      const auto* left = static_cast<const double*>(request->left.data);
      const auto* right = static_cast<const double*>(request->right.data);
      const auto* normals = static_cast<const double*>(request->normals.data);
      auto* output = static_cast<double*>(result->normal_flux.data);
      const auto points = point_count(request->left);
      for (std::size_t point = 0; point < points; ++point) {
        double squared_normal = 0.0;
        for (std::int32_t axis = 0; axis < request->normals.dimension; ++axis) {
          const double normal = normals[field_offset(request->normals, point, axis)];
          squared_normal += normal * normal;
        }
        if (!std::isfinite(squared_normal) || squared_normal == 0.0) {
          result->status = {sizeof(PopsComponentStatusV1), 40, POPS_COMPONENT_ABORT_RUN_V1,
                            "numerical flux normal is invalid"};
          return 40;
        }
        for (std::size_t component = 0; component < request->left.component_count; ++component) {
          const auto left_index = field_offset(request->left, point, component);
          const auto right_index = field_offset(request->right, point, component);
          const auto output_index = field_offset(result->normal_flux, point, component);
          output[output_index] = 0.25 * left[left_index] + 0.75 * right[right_index];
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
      const auto points = point_count(request->ghosts);
      for (std::size_t point = 0; point < points; ++point)
        for (std::size_t component = 0; component < request->ghosts.component_count; ++component) {
          const auto index = field_offset(request->ghosts, point, component);
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
      const auto points = point_count(request->base_outward_normal_flux);
      const auto axis = static_cast<std::size_t>(request->region.axes[0]);
      const double side = static_cast<double>(request->region.sides[0]);
      for (std::size_t point = 0; point < points; ++point) {
        const auto normal_offset = field_offset(request->outward_normals, point, axis);
        if (normals[normal_offset] != side || request->face_measures[point] <= 0.0) {
          result->status = {sizeof(PopsComponentStatusV1), 43, POPS_COMPONENT_ABORT_RUN_V1,
                            "boundary flux orientation is inconsistent"};
          return 43;
        }
        for (std::size_t component = 0;
             component < request->base_outward_normal_flux.component_count; ++component) {
          const auto base_index = field_offset(request->base_outward_normal_flux, point, component);
          const auto output_index = field_offset(result->outward_normal_flux, point, component);
          output[output_index] = base[base_index] + 10.0;
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
        std::array<std::size_t, 3> interior_extents{1, 1, 1};
        std::size_t interior_points = 1;
        for (std::int32_t axis = 0; axis < reference.dimension; ++axis) {
          interior_extents[axis] =
              reference.extents[axis] - reference.ghost_lower[axis] - reference.ghost_upper[axis];
          interior_points *= interior_extents[axis];
        }
        if (interior_points != points) {
          *status = {sizeof(PopsComponentStatusV1), 22, POPS_COMPONENT_ABORT_RUN_V1,
                     "AMR tag mask shape differs from the exact-ranked state"};
          return false;
        }
        for (std::size_t point = 0; point < points; ++point) {
          bool matches[128]{}, equality[128]{};
          std::size_t depth = 0;
          std::array<std::size_t, 3> coordinates{};
          std::size_t remaining = point;
          for (std::int32_t axis = 0; axis < reference.dimension; ++axis) {
            coordinates[axis] = remaining % interior_extents[axis];
            remaining /= interior_extents[axis];
          }
          for (std::size_t instruction = 0; instruction < instruction_count; ++instruction) {
            const int32_t opcode = opcodes[instruction];
            const int32_t argument = arguments[instruction];
            if (opcode >= 1 && opcode <= 5) {
              const auto& leaf = request->program.leaves[argument];
              const auto& view = request->states[leaf.state_index].values;
              const auto* values = static_cast<const double*>(view.data);
              const auto read = [&](const std::array<std::ptrdiff_t, 3>& sample_coordinates) {
                auto offset = static_cast<std::ptrdiff_t>(leaf.component) * view.component_stride;
                for (std::int32_t axis = 0; axis < view.dimension; ++axis)
                  offset += (sample_coordinates[axis] +
                             static_cast<std::ptrdiff_t>(view.ghost_lower[axis])) *
                            view.axis_strides[axis];
                return values[offset];
              };
              std::array<std::ptrdiff_t, 3> sample_coordinates{};
              for (std::int32_t axis = 0; axis < view.dimension; ++axis)
                sample_coordinates[axis] = static_cast<std::ptrdiff_t>(coordinates[axis]);
              double sample = read(sample_coordinates);
              if (opcode == 3)
                sample = sample < 0.0 ? -sample : sample;
              if (opcode == 4 || opcode == 5) {
                const auto& stencil = request->program.stencils[leaf.stencil_index];
                double squared_norm = 0.0;
                for (std::size_t axis_index = 0; axis_index < stencil.axis_count; ++axis_index) {
                  const auto& axis = stencil.axes[axis_index];
                  double derivative = 0.0;
                  for (std::size_t term = 0; term < axis.term_count; ++term) {
                    auto stencil_coordinates = sample_coordinates;
                    stencil_coordinates[axis.axis] += axis.offsets[term];
                    derivative += axis.coefficients[term] * read(stencil_coordinates);
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
      if (request->dimension < 1 || request->dimension > 3 || request->box_capacity < 1)
        return 2;
      std::array<std::int64_t, 3> lo{0, 0, 0};
      std::array<std::int64_t, 3> hi{-1, -1, -1};
      for (std::int32_t axis = 0; axis < request->dimension; ++axis)
        lo[axis] = request->extents[axis];
      for (std::size_t point = 0; point < request->tags.size; ++point) {
        if (request->tags.data[point] == 0)
          continue;
        std::size_t remaining = point;
        for (std::int32_t axis = 0; axis < request->dimension; ++axis) {
          const auto coordinate = static_cast<std::int64_t>(
              remaining % static_cast<std::size_t>(request->extents[axis]));
          remaining /= static_cast<std::size_t>(request->extents[axis]);
          lo[axis] = std::min(lo[axis], coordinate);
          hi[axis] = std::max(hi[axis], coordinate);
        }
      }
      if (hi[0] < 0) {
        *request->box_count = 0;
      } else {
        for (std::int32_t axis = 0; axis < request->dimension; ++axis) {
          request->boxes[axis] = lo[axis];
          request->boxes[request->dimension + axis] = hi[axis];
        }
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
    std::array<std::size_t, kDim> extents{};
    extents.fill(2);
    const std::size_t points = extent_product(extents);
    constexpr std::size_t components = 2;
    std::vector<double> left(points * components);
    std::vector<double> right(points * components);
    std::vector<double> normals(points * static_cast<std::size_t>(kDim), 0.0);
    for (std::size_t index = 0; index < left.size(); ++index) {
      left[index] = static_cast<double>(2 * index + 1);
      right[index] = static_cast<double>(2 * index + 2);
    }
    for (std::size_t point = 0; point < points; ++point)
      normals[point * static_cast<std::size_t>(kDim)] = 1.0;
    std::vector<double> flux(points * components, 0.0);
    std::vector<double> stability(points, 0.0);
    std::vector<PopsComponentActionV1> actions(points);
    const auto execution = abi::host_execution_context();
    const PopsNumericalFluxRequestV1 request{
        sizeof(PopsNumericalFluxRequestV1),
        const_field_view<kDim>(left.data(), extents, components),
        const_field_view<kDim>(right.data(), extents, components),
        const_field_view<kDim>(normals.data(), extents, static_cast<std::size_t>(kDim)),
        nullptr,
        abi::logical_time(),
        execution};
    PopsNumericalFluxResultV1 result{sizeof(PopsNumericalFluxResultV1),
                                     field_view<kDim>(flux.data(), extents, components),
                                     stability.data(),
                                     actions.data(),
                                     {}};
    void* state = loaded.prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(pops::component::evaluate_faces<kDim>(table, state, request, result), 0);
    std::vector<double> expected_flux(left.size());
    for (std::size_t index = 0; index < expected_flux.size(); ++index)
      expected_flux[index] = 0.25 * left[index] + 0.75 * right[index];
    EXPECT_EQ(flux, expected_flux);
    EXPECT_EQ(stability, std::vector<double>(points, 3.0));
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
