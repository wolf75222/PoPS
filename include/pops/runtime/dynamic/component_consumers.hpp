#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pops::component {

inline void require_operation(bool present, const char* name) {
  if (!present)
    throw std::runtime_error(std::string("native component table misses operation ") + name);
}

inline bool component_text(const char* value) {
  return value != nullptr && *value != '\0';
}

inline constexpr std::int32_t kUnwrittenComponentStatusCode =
    std::numeric_limits<std::int32_t>::min();

inline PopsComponentStatusV1 unwritten_component_status() {
  return {sizeof(PopsComponentStatusV1), kUnwrittenComponentStatusCode,
          POPS_COMPONENT_ABORT_RUN_V1, "native component did not write its status"};
}

inline bool component_action_is_known(PopsComponentActionV1 action) {
  return action == POPS_COMPONENT_CONTINUE_V1 ||
         action == POPS_COMPONENT_RETRY_STEP_V1 ||
         action == POPS_COMPONENT_REJECT_STEP_V1 ||
         action == POPS_COMPONENT_ABORT_RUN_V1;
}

inline bool component_status_is_well_formed(const PopsComponentStatusV1& status) {
  return status.struct_size >= sizeof(PopsComponentStatusV1) &&
         status.code != kUnwrittenComponentStatusCode &&
         component_action_is_known(status.action);
}

inline bool solve_status_is_known(PopsSolveStatusV2 status) {
  switch (status) {
    case POPS_SOLVE_SOLVED_V2:
    case POPS_SOLVE_SINGULAR_V2:
    case POPS_SOLVE_BREAKDOWN_V2:
    case POPS_SOLVE_ITERATION_LIMIT_V2:
    case POPS_SOLVE_INVALID_EVALUATION_V2:
    case POPS_SOLVE_CAPABILITY_FAILURE_V2:
    case POPS_SOLVE_INVALID_INPUT_V2:
    case POPS_SOLVE_INCOMPATIBLE_RHS_V2:
      return true;
  }
  return false;
}

inline bool solve_action_is_known(PopsSolveActionV2 action) {
  return action == POPS_SOLVE_ACTION_NONE_V2 ||
         action == POPS_SOLVE_ACTION_FAIL_RUN_V2 ||
         action == POPS_SOLVE_ACTION_REJECT_ATTEMPT_V2;
}

inline bool valid_precision(PopsPrecisionV1 value) {
  return value == POPS_PRECISION_FLOAT16_V1 ||
         value == POPS_PRECISION_BFLOAT16_V1 ||
         value == POPS_PRECISION_FLOAT32_V1 ||
         value == POPS_PRECISION_FLOAT64_V1;
}

inline void validate_execution_context(const PopsExecutionContextV1& context) {
  if (context.struct_size < sizeof(PopsExecutionContextV1) ||
      context.context_version != 1 ||
      !component_text(context.execution_identity) ||
      (context.memory_space != POPS_MEMORY_SPACE_HOST_V1 &&
       context.memory_space != POPS_MEMORY_SPACE_DEVICE_V1 &&
       context.memory_space != POPS_MEMORY_SPACE_MANAGED_V1) ||
      !component_text(context.backend_identity) ||
      !component_text(context.device_identity) ||
      !component_text(context.stream_identity) ||
      !component_text(context.communicator_identity) ||
      !component_text(context.communicator_datatype_identity) ||
      context.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
      context.storage_precision != POPS_PRECISION_FLOAT64_V1 ||
      !valid_precision(context.compute_precision) ||
      !valid_precision(context.accumulation_precision) ||
      !valid_precision(context.reduction_precision))
    throw std::invalid_argument(
        "native component execution context has invalid size, identities or precision policy");
  const bool serial = std::string(context.communicator_identity) == "serial";
  if (serial) {
    // MPI_Comm_c2f/MPI_Type_c2f may legally return zero for predefined handles.  The explicit
    // identities, not a guessed numeric sentinel, therefore distinguish serial from distributed
    // execution.  Serial retains the canonical all-zero representation.
    if (context.communicator_datatype_f_handle != 0 ||
        context.communicator_f_handle != 0 ||
        std::string(context.communicator_datatype_identity) != "none")
      throw std::invalid_argument(
          "serial component execution context cannot hide MPI handles or identities");
  } else if (std::string(context.communicator_identity) != "MPI_COMM_WORLD" ||
             std::string(context.communicator_datatype_identity) != "MPI_DOUBLE") {
    throw std::invalid_argument(
        "distributed component execution context supports only exact MPI_COMM_WORLD/MPI_DOUBLE");
  }
}

inline void validate_logical_time(const PopsLogicalTimeV1& time) {
  if (time.struct_size < sizeof(PopsLogicalTimeV1) ||
      !component_text(time.clock_identity) || time.tick < 0 || time.level < 0 ||
      time.substep < 0 || time.stage < 0 || time.fraction_numerator < 0 ||
      time.fraction_denominator <= 0 ||
      time.fraction_numerator > time.fraction_denominator ||
      std::gcd(time.fraction_numerator, time.fraction_denominator) != 1 ||
      !std::isfinite(time.dt) || time.dt < 0.0 || !std::isfinite(time.physical_time))
    throw std::invalid_argument("native component logical time is incomplete or non-canonical");
}

template <class View>
inline std::size_t field_point_count(const View& view) {
  std::size_t result = 1;
  for (std::int32_t axis = 0; axis < view.dimension; ++axis) {
    if (view.extents[axis] > std::numeric_limits<std::size_t>::max() / result)
      throw std::invalid_argument("native field view extents overflow");
    result *= view.extents[axis];
  }
  return result;
}

template <class View>
inline std::size_t field_interior_point_count(const View& view) {
  std::size_t result = 1;
  for (std::int32_t axis = 0; axis < view.dimension; ++axis) {
    const std::size_t extent = view.extents[axis];
    const std::size_t lower = view.ghost_lower[axis];
    const std::size_t upper = view.ghost_upper[axis];
    if (lower >= extent || upper >= extent - lower)
      throw std::invalid_argument("native field view interior extents overflow");
    const std::size_t interior = extent - lower - upper;
    if (interior > std::numeric_limits<std::size_t>::max() / result)
      throw std::invalid_argument("native field view interior extents overflow");
    result *= interior;
  }
  return result;
}

template <class View>
inline void validate_field_view(const View& view, const char* where) {
  if (view.struct_size < sizeof(View) || view.data == nullptr || view.dimension < 1 ||
      view.dimension > 3 || view.component_count == 0 || view.component_stride <= 0 ||
      !component_text(view.layout_identity) || !component_text(view.patch_identity) ||
      (view.scalar_type != POPS_SCALAR_FLOAT32_V1 &&
       view.scalar_type != POPS_SCALAR_FLOAT64_V1) ||
      (view.memory_space != POPS_MEMORY_SPACE_HOST_V1 &&
       view.memory_space != POPS_MEMORY_SPACE_DEVICE_V1 &&
       view.memory_space != POPS_MEMORY_SPACE_MANAGED_V1) ||
      (view.ownership != POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1 &&
       view.ownership != POPS_FIELD_OWNERSHIP_COMPONENT_BORROWED_V1 &&
       view.ownership != POPS_FIELD_OWNERSHIP_COMPONENT_OWNED_V1))
    throw std::invalid_argument(std::string(where) + " has an incomplete field descriptor");
  const auto active_axes = (1u << static_cast<unsigned>(view.dimension)) - 1u;
  if ((view.centering == POPS_FIELD_CENTERING_FACE_V1 &&
       (view.centering_axes == 0 ||
        (view.centering_axes & (view.centering_axes - 1u)) != 0)) ||
      (view.centering == POPS_FIELD_CENTERING_EDGE_V1 && view.centering_axes == 0) ||
      ((view.centering == POPS_FIELD_CENTERING_CELL_V1 ||
        view.centering == POPS_FIELD_CENTERING_NODE_V1) && view.centering_axes != 0) ||
      view.centering < POPS_FIELD_CENTERING_CELL_V1 ||
      view.centering > POPS_FIELD_CENTERING_EDGE_V1 ||
      (view.centering_axes & ~active_axes) != 0)
    throw std::invalid_argument(std::string(where) + " has invalid centering axes");
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    if (axis < view.dimension) {
      if (view.extents[axis] == 0 || view.axis_strides[axis] <= 0 ||
          view.ghost_lower[axis] >= view.extents[axis] ||
          view.ghost_upper[axis] >= view.extents[axis] - view.ghost_lower[axis])
        throw std::invalid_argument(std::string(where) +
                                    " has invalid extent, stride or ghost widths");
    } else if (view.extents[axis] != 1 || view.axis_strides[axis] != 0 ||
               view.ghost_lower[axis] != 0 || view.ghost_upper[axis] != 0) {
      throw std::invalid_argument(std::string(where) +
                                  " carries hidden inactive-axis metadata");
    }
  }
  (void)field_point_count(view);
}

template <class View>
inline void validate_backend_field_view(const View& view, const char* where) {
  validate_field_view(view, where);
  if (view.dimension != 2)
    throw std::invalid_argument(std::string(where) +
                                " is not representable by the current 2D backend");
  if (view.scalar_type != POPS_SCALAR_FLOAT64_V1)
    throw std::invalid_argument(std::string(where) +
                                " scalar type differs from the current binary64 backend");
}

template <class Left, class Right>
inline bool same_field_domain(const Left& left, const Right& right) {
  if (left.dimension != right.dimension || left.component_count != right.component_count ||
      std::string(left.layout_identity) != right.layout_identity ||
      std::string(left.patch_identity) != right.patch_identity ||
      left.centering != right.centering || left.centering_axes != right.centering_axes ||
      left.scalar_type != right.scalar_type || left.memory_space != right.memory_space)
    return false;
  for (std::int32_t axis = 0; axis < 3; ++axis)
    if (left.extents[axis] != right.extents[axis] ||
        left.ghost_lower[axis] != right.ghost_lower[axis] ||
        left.ghost_upper[axis] != right.ghost_upper[axis])
      return false;
  return true;
}

template <class Left, class Right>
inline bool same_spatial_domain(const Left& left, const Right& right) {
  if (left.dimension != right.dimension ||
      std::string(left.layout_identity) != right.layout_identity ||
      std::string(left.patch_identity) != right.patch_identity ||
      left.memory_space != right.memory_space)
    return false;
  for (std::int32_t axis = 0; axis < 3; ++axis)
    if (left.extents[axis] != right.extents[axis] ||
        left.ghost_lower[axis] != right.ghost_lower[axis] ||
        left.ghost_upper[axis] != right.ghost_upper[axis])
      return false;
  return true;
}

template <class View>
inline void validate_execution_field(const PopsExecutionContextV1& context,
                                     const View& view, const char* where) {
  validate_backend_field_view(view, where);
  if (view.memory_space != context.memory_space || view.scalar_type != context.scalar_type)
    throw std::invalid_argument(std::string(where) +
                                " disagrees with its execution context");
}

inline bool empty_field_view(const PopsConstFieldViewV1& view) {
  if (view.struct_size != 0 || view.data != nullptr || view.dimension != 0 ||
      view.component_count != 0 || view.component_stride != 0 || view.centering != 0 ||
      view.centering_axes != 0 || view.scalar_type != 0 || view.memory_space != 0 ||
      view.layout_identity != nullptr || view.patch_identity != nullptr ||
      view.ownership != 0)
    return false;
  for (std::int32_t axis = 0; axis < 3; ++axis)
    if (view.extents[axis] != 0 || view.axis_strides[axis] != 0 ||
        view.ghost_lower[axis] != 0 || view.ghost_upper[axis] != 0)
      return false;
  return true;
}

inline void validate_boundary_region(const PopsBoundaryRegionV1& region) {
  if (region.struct_size < sizeof(PopsBoundaryRegionV1) ||
      region.dimension < 1 || region.dimension > 3 || region.codimension < 1 ||
      region.codimension > region.dimension ||
      region.axis_count != static_cast<std::size_t>(region.codimension) ||
      region.axes == nullptr || region.sides == nullptr ||
      !component_text(region.region_identity))
    throw std::invalid_argument("native boundary region is incomplete");
  if ((region.kind == POPS_BOUNDARY_FACE_V1 && region.codimension != 1) ||
      (region.kind == POPS_BOUNDARY_EDGE_V1 && region.codimension != 2) ||
      (region.kind == POPS_BOUNDARY_CORNER_V1 &&
       region.codimension != region.dimension))
    throw std::invalid_argument("native boundary kind and codimension disagree");
  std::unordered_set<std::int32_t> axes;
  for (std::size_t index = 0; index < region.axis_count; ++index) {
    if (region.axes[index] < 0 || region.axes[index] >= region.dimension ||
        !axes.insert(region.axes[index]).second ||
        (region.sides[index] != -1 && region.sides[index] != 1))
      throw std::invalid_argument("native boundary axes/sides are invalid");
  }
}

inline void validate_const_fields(const PopsQualifiedConstFieldV1* rows,
                                  std::size_t count, const char* where) {
  if (count != 0 && rows == nullptr)
    throw std::invalid_argument(std::string(where) + " table is null");
  std::unordered_set<std::string> identities;
  for (std::size_t index = 0; index < count; ++index) {
    const auto& row = rows[index];
    if (row.struct_size < sizeof(PopsQualifiedConstFieldV1) || row.present != 1 ||
        !component_text(row.qualified_id) ||
        !identities.insert(row.qualified_id).second)
      throw std::invalid_argument(std::string(where) +
                                  " entries must be present, qualified and unique");
    validate_backend_field_view(row.values, where);
  }
}

inline void validate_optional_const_field(const PopsQualifiedConstFieldV1& row,
                                          const char* where) {
  if (row.struct_size < sizeof(PopsQualifiedConstFieldV1) || row.present > 1)
    throw std::invalid_argument(std::string(where) + " has invalid size/presence");
  if (row.present == 0) {
    if (row.qualified_id != nullptr || !empty_field_view(row.values))
      throw std::invalid_argument(std::string(where) + " absent value carries hidden data");
    return;
  }
  validate_const_fields(&row, 1, where);
}

inline void validate_scalars(const PopsQualifiedScalarV1* rows,
                             std::size_t count, const char* where) {
  if (count != 0 && rows == nullptr)
    throw std::invalid_argument(std::string(where) + " table is null");
  std::unordered_set<std::string> identities;
  for (std::size_t index = 0; index < count; ++index) {
    if (rows[index].struct_size < sizeof(PopsQualifiedScalarV1) ||
        !component_text(rows[index].qualified_id) ||
        !identities.insert(rows[index].qualified_id).second)
      throw std::invalid_argument(std::string(where) +
                                  " entries must be qualified and unique");
  }
}

inline int evaluate_faces(const PopsNumericalFluxApiV1& api, void* state,
                          const PopsNumericalFluxRequestV1& request,
                          PopsNumericalFluxResultV1& result) {
  require_operation(api.evaluate_faces != nullptr, "evaluate_faces");
  validate_execution_context(request.execution);
  validate_logical_time(request.logical_time);
  validate_execution_field(request.execution, request.left, "numerical flux left");
  validate_execution_field(request.execution, request.right, "numerical flux right");
  validate_execution_field(request.execution, request.normals, "numerical flux normals");
  validate_execution_field(request.execution, result.normal_flux,
                           "numerical flux output");
  if (!same_field_domain(request.left, request.right) ||
      !same_field_domain(request.left, result.normal_flux) ||
      !same_spatial_domain(request.left, request.normals) ||
      request.normals.component_count !=
          static_cast<std::size_t>(request.left.dimension))
    throw std::invalid_argument("numerical flux field descriptors disagree");
  return api.evaluate_faces(state, &request, &result);
}

inline int apply_ghost_boundary(const PopsGhostBoundaryApiV1& api, void* state,
                                const PopsGhostBoundaryRequestV1& request,
                                PopsComponentStatusV1& status) {
  require_operation(api.apply_region_batch != nullptr, "apply_region_batch");
  validate_execution_context(request.execution);
  validate_logical_time(request.logical_time);
  validate_boundary_region(request.region);
  if (!component_text(request.producer_identity) ||
      !component_text(request.state_identity) ||
      !component_text(request.ghost_identity))
    throw std::invalid_argument("ghost boundary requires qualified producer/state/output ids");
  validate_execution_field(request.execution, request.interior, "ghost interior");
  validate_execution_field(request.execution, request.ghosts, "ghost output");
  validate_execution_field(request.execution, request.coordinates, "ghost coordinates");
  if (!same_field_domain(request.interior, request.ghosts) ||
      !same_spatial_domain(request.interior, request.coordinates) ||
      request.coordinates.component_count !=
          static_cast<std::size_t>(request.region.dimension) ||
      request.interior.dimension != request.region.dimension)
    throw std::invalid_argument("ghost boundary field descriptors disagree");
  validate_const_fields(request.dependencies, request.dependency_count,
                        "ghost boundary dependencies");
  for (std::size_t index = 0; index < request.dependency_count; ++index)
    validate_execution_field(request.execution, request.dependencies[index].values,
                             "ghost boundary dependency");
  validate_scalars(request.parameters, request.parameter_count,
                   "ghost boundary parameters");
  return api.apply_region_batch(state, &request, &status);
}

inline int evaluate_field_boundary(const PopsFieldBoundaryClosureApiV1& api, void* state,
                                   const PopsFieldBoundaryRequestV1& request,
                                   PopsComponentStatusV1& status, bool jvp) {
  const auto operation = jvp ? api.jvp : api.residual;
  require_operation(operation != nullptr, jvp ? "jvp" : "residual");
  validate_execution_context(request.execution);
  validate_logical_time(request.logical_time);
  validate_boundary_region(request.region);
  if (!component_text(request.closure_identity) || request.state_count == 0 ||
      (jvp ? request.direction_count == 0 : request.direction_count != 0) ||
      request.output_count == 0 || request.outputs == nullptr ||
      request.coordinates.data == nullptr || request.level < 0)
    throw std::invalid_argument("field boundary requires qualified closure input/output tables");
  validate_execution_field(request.execution, request.coordinates,
                           "field boundary coordinates");
  if (request.coordinates.component_count !=
          static_cast<std::size_t>(request.region.dimension) ||
      request.coordinates.dimension != request.region.dimension)
    throw std::invalid_argument("field boundary coordinate descriptor disagrees with region");
  validate_const_fields(request.states, request.state_count, "field boundary states");
  validate_const_fields(request.directions, request.direction_count,
                        "field boundary directions");
  validate_const_fields(request.fields, request.field_count, "field boundary fields");
  for (std::size_t index = 0; index < request.state_count; ++index)
    validate_execution_field(request.execution, request.states[index].values,
                             "field boundary state");
  for (std::size_t index = 0; index < request.direction_count; ++index)
    validate_execution_field(request.execution, request.directions[index].values,
                             "field boundary direction");
  for (std::size_t index = 0; index < request.field_count; ++index)
    validate_execution_field(request.execution, request.fields[index].values,
                             "field boundary field");
  validate_scalars(request.parameters, request.parameter_count,
                   "field boundary parameters");
  validate_optional_const_field(request.rate, "field boundary rate");
  validate_optional_const_field(request.nonlinear_iterate,
                                "field boundary nonlinear iterate");
  if (request.rate.present)
    validate_execution_field(request.execution, request.rate.values,
                             "field boundary rate");
  if (request.nonlinear_iterate.present)
    validate_execution_field(request.execution, request.nonlinear_iterate.values,
                             "field boundary nonlinear iterate");
  std::unordered_set<std::string> output_ids;
  for (std::size_t index = 0; index < request.output_count; ++index) {
    if (request.outputs[index].struct_size < sizeof(PopsQualifiedFieldV1) ||
        !component_text(request.outputs[index].qualified_id) ||
        !output_ids.insert(request.outputs[index].qualified_id).second)
      throw std::invalid_argument("field boundary outputs must be qualified and writable");
    validate_execution_field(request.execution, request.outputs[index].values,
                             "field boundary output");
  }
  return operation(state, &request, &status);
}

inline int tag_batch(const PopsTaggerApiV1& api, void* state,
                     const PopsTaggerRequestV1& request, PopsComponentStatusV1& status) {
  require_operation(api.tag_batch != nullptr, "tag_batch");
  validate_execution_context(request.execution);
  validate_logical_time(request.logical_time);
  if (request.struct_size < sizeof(PopsTaggerRequestV1) || request.state_count == 0 ||
      request.states == nullptr || request.program.struct_size < sizeof(PopsTaggingProgramV1) ||
      request.program.program_identity == nullptr || request.program.leaf_count == 0 ||
      request.program.leaves == nullptr || request.program.refine_instruction_count == 0 ||
      request.program.refine_opcodes == nullptr || request.program.refine_arguments == nullptr ||
      (request.program.coarsen_instruction_count != 0 &&
       (request.program.coarsen_opcodes == nullptr ||
        request.program.coarsen_arguments == nullptr)) ||
      request.program.minimum_cycles < 0 || request.program.equality_policy < 0 ||
      request.program.equality_policy > 2 || request.program.conflict_policy < 0 ||
      request.program.conflict_policy > 3 ||
      request.program.non_finite_policy != POPS_TAGGING_NON_FINITE_REJECT_V1)
    throw std::invalid_argument("tagger request has no exact graph program");
  std::size_t points = 0;
  for (std::size_t index = 0; index < request.state_count; ++index) {
    const auto& state_view = request.states[index];
    if (state_view.struct_size < sizeof(PopsQualifiedConstFieldV1) ||
        state_view.present != 1 || state_view.qualified_id == nullptr)
      throw std::invalid_argument("tagger state route is incomplete");
    validate_execution_field(request.execution, state_view.values, "tagger state");
    const std::size_t count = field_interior_point_count(state_view.values);
    if (index == 0)
      points = count;
    else if (count != points)
      throw std::invalid_argument("tagger state patches do not share one point shape");
  }
  for (std::size_t index = 0; index < request.program.leaf_count; ++index) {
    const auto& leaf = request.program.leaves[index];
    const bool gradient = leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                          leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
    if (leaf.struct_size < sizeof(PopsTaggingLeafV1) ||
        leaf.state_index >= request.state_count ||
        leaf.component >= request.states[leaf.state_index].values.component_count ||
        !pops_tagging_opcode_is_leaf_v1(leaf.opcode) || !std::isfinite(leaf.threshold) ||
        gradient != (leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1) ||
        (gradient && leaf.stencil_index >= request.program.stencil_count))
      throw std::invalid_argument("tagger graph leaf is invalid");
  }
  if (request.program.stencil_count != 0 && request.program.stencils == nullptr)
    throw std::invalid_argument("tagger graph stencil table is absent");
  for (std::size_t index = 0; index < request.program.stencil_count; ++index) {
    const auto& stencil = request.program.stencils[index];
    if (stencil.struct_size < sizeof(PopsTaggingStencilV1) ||
        !component_text(stencil.stencil_identity) || !component_text(stencil.route) ||
        std::string_view(stencil.route) !=
            POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1 ||
        !component_text(stencil.norm) || std::string_view(stencil.norm) != "l2" ||
        !component_text(stencil.scale) ||
        std::string_view(stencil.scale) != "inverse_cell_size" ||
        !component_text(stencil.boundary_mode) ||
        std::string_view(stencil.boundary_mode) != "ghost_extension" ||
        stencil.dimension != request.states[0].values.dimension ||
        stencil.axis_count != static_cast<std::size_t>(stencil.dimension) ||
        stencil.axes == nullptr)
      throw std::invalid_argument("tagger graph stencil route is invalid");
    for (std::size_t axis_index = 0; axis_index < stencil.axis_count; ++axis_index) {
      const auto& axis = stencil.axes[axis_index];
      if (axis.struct_size < sizeof(PopsTaggingAxisStencilV1) ||
          axis.axis != static_cast<std::int32_t>(axis_index) ||
          axis.derivative_order != 1 || axis.formal_order < 1 || axis.term_count == 0 ||
          static_cast<std::size_t>(axis.formal_order) > axis.term_count ||
          axis.term_count > POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 ||
          axis.offsets == nullptr || axis.coefficients == nullptr)
        throw std::invalid_argument("tagger graph axis stencil is invalid");
      std::vector<std::int32_t> offsets(axis.offsets, axis.offsets + axis.term_count);
      std::sort(offsets.begin(), offsets.end());
      if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end())
        throw std::invalid_argument("tagger graph axis stencil repeats an offset");
      std::size_t ghost_lower = 0, ghost_upper = 0;
      for (std::size_t term = 0; term < axis.term_count; ++term) {
        const auto widened_offset = static_cast<std::int64_t>(axis.offsets[term]);
        if (!std::isfinite(axis.coefficients[term]))
          throw std::invalid_argument(
              "tagger graph axis stencil coefficient is not finite");
        ghost_lower = std::max(
            ghost_lower,
            static_cast<std::size_t>(
                std::max<std::int64_t>(0, -widened_offset)));
        ghost_upper = std::max(
            ghost_upper,
            static_cast<std::size_t>(
                std::max<std::int64_t>(0, widened_offset)));
      }
      if (ghost_lower != axis.ghost_lower || ghost_upper != axis.ghost_upper)
        throw std::invalid_argument("tagger graph axis stencil halo is inconsistent");
      for (std::int32_t power = 0; power <= axis.formal_order; ++power) {
        double moment = 0.0, scale = 0.0;
        for (std::size_t term = 0; term < axis.term_count; ++term) {
          const double value = axis.coefficients[term] *
              std::pow(static_cast<double>(axis.offsets[term]), power);
          moment += value;
          scale += std::abs(value);
        }
        const double expected = power == 1 ? 1.0 : 0.0;
        if (std::abs(moment - expected) >
            1.0e-13 * std::max(1.0, scale))
          throw std::invalid_argument(
              "tagger graph axis stencil falsely declares its formal order");
      }
    }
  }
  for (std::size_t index = 0; index < request.program.leaf_count; ++index) {
    const auto& leaf = request.program.leaves[index];
    if (leaf.stencil_index == POPS_TAGGING_NO_STENCIL_V1)
      continue;
    const auto& stencil = request.program.stencils[leaf.stencil_index];
    const auto& view = request.states[leaf.state_index].values;
    for (std::size_t axis = 0; axis < stencil.axis_count; ++axis)
      if (stencil.axes[axis].ghost_lower > view.ghost_lower[axis] ||
          stencil.axes[axis].ghost_upper > view.ghost_upper[axis])
        throw std::invalid_argument(
            "tagger graph stencil exceeds the supplied state halo");
  }
  for (const PopsByteViewV1* output : {
           &request.refine_candidates, &request.coarsen_candidates,
           &request.refine_equalities, &request.coarsen_equalities})
    if (output->struct_size < sizeof(PopsByteViewV1) || output->data == nullptr ||
        output->size != points)
      throw std::invalid_argument("tagger candidate output does not match its patch shape");
  return api.tag_batch(state, &request, &status);
}

inline int cluster_tags(const PopsClusteringApiV1& api, void* state,
                        const PopsClusteringRequestV1& request,
                        PopsComponentStatusV1& status) {
  require_operation(api.cluster != nullptr, "cluster");
  validate_execution_context(request.execution);
  if (request.struct_size < sizeof(PopsClusteringRequestV1) ||
      request.tags.struct_size < sizeof(PopsConstByteViewV1) ||
      request.tags.data == nullptr || request.extents == nullptr ||
      request.dimension < 1 || request.dimension > 3 || request.boxes == nullptr ||
      request.box_capacity == 0 || request.box_count == nullptr)
    throw std::invalid_argument("clustering request is incomplete");
  std::size_t points = 1;
  for (std::int32_t axis = 0; axis < request.dimension; ++axis) {
    if (request.extents[axis] <= 0 ||
        static_cast<std::uint64_t>(request.extents[axis]) >
            std::numeric_limits<std::size_t>::max() / points)
      throw std::invalid_argument("clustering extents are invalid or overflow");
    points *= static_cast<std::size_t>(request.extents[axis]);
  }
  if (request.tags.size != points ||
      request.box_capacity > std::numeric_limits<std::size_t>::max() /
                                 (2u * static_cast<std::size_t>(request.dimension)))
    throw std::invalid_argument("clustering tag shape or box capacity is invalid");
  *request.box_count = 0;
  const int code = api.cluster(state, &request, &status);
  if (*request.box_count > request.box_capacity)
    throw std::runtime_error(
        "clustering component exceeded the output capacity supplied by PoPS");
  return code;
}

inline int apply_transfer(const PopsTransferApiV1& api, void* state,
                          const PopsTransferRequestV1& request,
                          PopsComponentStatusV1& status) {
  require_operation(api.apply != nullptr, "apply");
  validate_execution_context(request.execution);
  validate_execution_field(request.execution, request.source, "transfer source");
  validate_execution_field(request.execution, request.destination, "transfer destination");
  if (request.dimension != request.source.dimension ||
      request.dimension != request.destination.dimension || request.refinement_ratio == nullptr ||
      request.operation != POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1 ||
      request.source.component_count != request.destination.component_count ||
      request.source.centering != request.destination.centering ||
      request.source.centering_axes != request.destination.centering_axes ||
      request.source.scalar_type != request.destination.scalar_type)
    throw std::invalid_argument("transfer field descriptors disagree with dimension");
  for (std::int32_t axis = 0; axis < request.dimension; ++axis) {
    const auto ratio = request.refinement_ratio[axis];
    const auto source_interior = request.source.extents[axis] -
                                 request.source.ghost_lower[axis] -
                                 request.source.ghost_upper[axis];
    const auto destination_interior = request.destination.extents[axis] -
                                      request.destination.ghost_lower[axis] -
                                      request.destination.ghost_upper[axis];
    if (ratio <= 0 || destination_interior >
                          std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(ratio) ||
        source_interior != destination_interior * static_cast<std::size_t>(ratio))
      throw std::invalid_argument("transfer refinement ratio must be positive");
  }
  return api.apply(state, &request, &status);
}

inline std::string writer_geometry_key(const char* layout, std::int32_t level) {
  return std::string(layout) + "\n" + std::to_string(level);
}

inline std::size_t writer_box_volume(const std::int64_t* lower,
                                     const std::int64_t* upper,
                                     std::int32_t dimension) {
  std::size_t volume = 1;
  for (std::int32_t axis = 0; axis < dimension; ++axis) {
    const auto extent = static_cast<std::size_t>(upper[axis] - lower[axis]);
    if (extent == 0 || extent > std::numeric_limits<std::size_t>::max() / volume)
      throw std::invalid_argument("Writer box volume overflows");
    volume *= extent;
  }
  return volume;
}

inline bool writer_boxes_overlap(const std::int64_t* left_lower,
                                 const std::int64_t* left_upper,
                                 const std::int64_t* right_lower,
                                 const std::int64_t* right_upper,
                                 std::int32_t dimension) {
  for (std::int32_t axis = 0; axis < dimension; ++axis)
    if (left_upper[axis] <= right_lower[axis] || right_upper[axis] <= left_lower[axis])
      return false;
  return true;
}

inline bool writer_box_contains(const std::int64_t* outer_lower,
                                const std::int64_t* outer_upper,
                                const std::int64_t* inner_lower,
                                const std::int64_t* inner_upper,
                                std::int32_t dimension) {
  for (std::int32_t axis = 0; axis < dimension; ++axis)
    if (inner_lower[axis] < outer_lower[axis] || inner_upper[axis] > outer_upper[axis])
      return false;
  return true;
}

inline void validate_writer_request(const PopsWriterRequestV1& request) {
  validate_execution_context(request.execution);
  validate_logical_time(request.logical_time);
  if (request.geometry_count == 0 || request.geometries == nullptr ||
      (request.field_count == 0 && request.diagnostic_count == 0) ||
      !component_text(request.metadata_json) ||
      !component_text(request.selection_identity) ||
      !component_text(request.temporary_path) || !component_text(request.published_path) ||
      !component_text(request.snapshot_identity))
    throw std::invalid_argument("Writer request is incomplete");
  std::unordered_map<std::string, const PopsWriterGeometryV1*> geometries;
  for (std::size_t index = 0; index < request.geometry_count; ++index) {
    const auto& geometry = request.geometries[index];
    if (geometry.struct_size < sizeof(PopsWriterGeometryV1) ||
        !component_text(geometry.layout_identity) || !component_text(geometry.layout_kind) ||
        geometry.level < 0 || geometry.dimension < 1 || geometry.dimension > 3 ||
        geometry.origin == nullptr || geometry.spacing == nullptr ||
        geometry.cell_shape == nullptr || geometry.box_count == 0 ||
        geometry.boxes == nullptr)
      throw std::invalid_argument("Writer geometry is incomplete");
    if (!geometries.emplace(
            writer_geometry_key(geometry.layout_identity, geometry.level), &geometry).second)
      throw std::invalid_argument("Writer geometry identity/level is duplicated");
    validate_execution_field(request.execution, geometry.cell_volumes,
                             "Writer cell volumes");
    if (geometry.cell_volumes.dimension != geometry.dimension ||
        geometry.cell_volumes.component_count != 1 ||
        std::string(geometry.cell_volumes.layout_identity) != geometry.layout_identity ||
        geometry.valid_cells.size != field_point_count(geometry.cell_volumes) ||
        geometry.coverage.size != field_point_count(geometry.cell_volumes) ||
        geometry.valid_cells.data == nullptr || geometry.coverage.data == nullptr)
      throw std::invalid_argument("Writer geometry masks disagree with its descriptor");
    for (std::int32_t axis = 0; axis < geometry.dimension; ++axis) {
      if (!std::isfinite(geometry.origin[axis]) ||
          !std::isfinite(geometry.spacing[axis]) || geometry.spacing[axis] <= 0.0 ||
          geometry.cell_shape[axis] != geometry.cell_volumes.extents[axis])
        throw std::invalid_argument("Writer geometry axes disagree with its field descriptor");
    }
    for (std::size_t box = 0; box < geometry.box_count; ++box) {
      const auto& bounds = geometry.boxes[box];
      if (bounds.struct_size < sizeof(PopsWriterBoxV1) ||
          bounds.dimension != geometry.dimension || bounds.lower == nullptr ||
          bounds.upper == nullptr)
        throw std::invalid_argument("Writer geometry box is incomplete");
      for (std::int32_t axis = 0; axis < geometry.dimension; ++axis)
        if (bounds.lower[axis] < 0 || bounds.upper[axis] <= bounds.lower[axis] ||
            bounds.upper[axis] > static_cast<std::int64_t>(geometry.cell_shape[axis]))
          throw std::invalid_argument("Writer geometry box exceeds its cell shape");
      for (std::size_t prior = 0; prior < box; ++prior)
        if (writer_boxes_overlap(bounds.lower, bounds.upper,
                                 geometry.boxes[prior].lower,
                                 geometry.boxes[prior].upper, geometry.dimension))
          throw std::invalid_argument("Writer geometry boxes overlap");
    }
  }
  std::unordered_set<std::string> qualified_states;
  for (std::size_t index = 0; index < request.field_count; ++index) {
    const auto& field = request.fields[index];
    if (field.struct_size < sizeof(PopsWriterFieldV1) ||
        !component_text(field.field_identity) || !component_text(field.reference_id) ||
        !component_text(field.component_manifest_identity) ||
        !component_text(field.layout_identity) || !component_text(field.state_id) ||
        !component_text(field.centering) || !component_text(field.units) ||
        field.dimension < 1 || field.dimension > 3 || field.global_shape == nullptr ||
        field.piece_count == 0 || field.pieces == nullptr)
      throw std::invalid_argument("Writer field is incomplete");
    const auto geometry_it = geometries.find(
        writer_geometry_key(field.layout_identity, field.level));
    if (geometry_it == geometries.end())
      throw std::invalid_argument("Writer field has no exact geometry authority");
    const auto& geometry = *geometry_it->second;
    if (field.dimension != geometry.dimension)
      throw std::invalid_argument("Writer field and geometry dimensions disagree");
    for (std::int32_t axis = 0; axis < field.dimension; ++axis)
      if (field.global_shape[axis] != geometry.cell_shape[axis])
        throw std::invalid_argument("Writer field global shape differs from geometry");
    if (field.component_name_count == 0 || field.component_names == nullptr)
      throw std::invalid_argument("Writer field has no exact component vocabulary");
    std::unordered_set<std::string> component_names;
    for (std::size_t component = 0; component < field.component_name_count; ++component)
      if (!component_text(field.component_names[component]) ||
          !component_names.insert(field.component_names[component]).second)
        throw std::invalid_argument("Writer component names must be non-empty and unique");
    qualified_states.insert(writer_geometry_key(field.layout_identity, field.level) +
                            "\n" + field.state_id);
    std::size_t piece_volume = 0;
    for (std::size_t piece = 0; piece < field.piece_count; ++piece) {
      validate_execution_field(request.execution, field.pieces[piece].values,
                               "Writer field piece");
      const auto& values = field.pieces[piece].values;
      if (field.pieces[piece].struct_size < sizeof(PopsWriterPieceV1) ||
          field.pieces[piece].dimension != field.dimension ||
          field.pieces[piece].lower == nullptr || field.pieces[piece].upper == nullptr ||
          values.dimension != field.dimension ||
          values.component_count != field.component_name_count ||
          std::string(values.layout_identity) != field.layout_identity)
          throw std::invalid_argument("Writer field piece descriptor disagrees with its bounds");
      const auto expected_centering =
          values.centering == POPS_FIELD_CENTERING_CELL_V1 ? "cell" :
          values.centering == POPS_FIELD_CENTERING_FACE_V1 ? "face" :
          values.centering == POPS_FIELD_CENTERING_NODE_V1 ? "node" : "edge";
      if (field.centering != std::string(expected_centering))
        throw std::invalid_argument("Writer centering text differs from its field descriptor");
      for (std::int32_t axis = 0; axis < field.dimension; ++axis)
        if (field.pieces[piece].lower[axis] < 0 ||
            field.pieces[piece].upper[axis] <= field.pieces[piece].lower[axis] ||
            field.pieces[piece].upper[axis] >
                static_cast<std::int64_t>(field.global_shape[axis]) ||
            values.extents[axis] != static_cast<std::size_t>(
                field.pieces[piece].upper[axis] - field.pieces[piece].lower[axis]))
          throw std::invalid_argument("Writer field piece bounds disagree with its descriptor");
      bool contained = false;
      for (std::size_t box = 0; box < geometry.box_count; ++box)
        contained = contained || writer_box_contains(
            geometry.boxes[box].lower, geometry.boxes[box].upper,
            field.pieces[piece].lower, field.pieces[piece].upper, field.dimension);
      if (!contained)
        throw std::invalid_argument("Writer field piece is outside declared geometry boxes");
      for (std::size_t prior = 0; prior < piece; ++prior)
        if (writer_boxes_overlap(field.pieces[piece].lower, field.pieces[piece].upper,
                                 field.pieces[prior].lower, field.pieces[prior].upper,
                                 field.dimension))
          throw std::invalid_argument("Writer field pieces overlap");
      const auto volume = writer_box_volume(field.pieces[piece].lower,
                                            field.pieces[piece].upper,
                                            field.dimension);
      if (volume > std::numeric_limits<std::size_t>::max() - piece_volume)
        throw std::invalid_argument("Writer field piece volume overflows");
      piece_volume += volume;
    }
    std::size_t geometry_volume = 0;
    for (std::size_t box = 0; box < geometry.box_count; ++box) {
      const auto volume = writer_box_volume(geometry.boxes[box].lower,
                                            geometry.boxes[box].upper,
                                            geometry.dimension);
      if (volume > std::numeric_limits<std::size_t>::max() - geometry_volume)
        throw std::invalid_argument("Writer geometry volume overflows");
      geometry_volume += volume;
    }
    if (piece_volume != geometry_volume)
      throw std::invalid_argument("Writer field pieces do not exactly cover geometry boxes");
  }
  for (std::size_t index = 0; index < request.diagnostic_count; ++index) {
    const auto& diagnostic = request.diagnostics[index];
    if (diagnostic.struct_size < sizeof(PopsWriterDiagnosticV1) ||
        !component_text(diagnostic.diagnostic_identity) ||
        !component_text(diagnostic.reference_id) ||
        !component_text(diagnostic.component_manifest_identity) ||
        !component_text(diagnostic.layout_identity) ||
        !component_text(diagnostic.state_id) || !component_text(diagnostic.reduction) ||
        !component_text(diagnostic.units) || !component_text(diagnostic.terms_json) ||
        !std::isfinite(diagnostic.value))
      throw std::invalid_argument("Writer diagnostic is incomplete");
    if (geometries.find(writer_geometry_key(
            diagnostic.layout_identity, diagnostic.level)) == geometries.end() ||
        qualified_states.find(writer_geometry_key(
            diagnostic.layout_identity, diagnostic.level) + "\n" +
            diagnostic.state_id) == qualified_states.end())
      throw std::invalid_argument(
          "Writer diagnostic does not reference an existing qualified field state");
  }
}

inline int publish_output(const PopsWriterApiV1& api, void* state,
                          const PopsWriterRequestV1& request,
                          PopsWriterReceiptV1& receipt) {
  require_operation(api.verify != nullptr, "verify");
  require_operation(api.publish != nullptr, "publish");
  validate_writer_request(request);
  PopsWriterReceiptV1 verification{
      sizeof(PopsWriterReceiptV1), 0, nullptr,
      {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  const int verified = api.verify(state, &request, &verification);
  if (verification.struct_size < sizeof(PopsWriterReceiptV1) ||
      verification.status.struct_size < sizeof(PopsComponentStatusV1))
    throw std::runtime_error("Writer verify returned an undersized receipt or status");
  if (verified != 0)
    return verified;
  if (verification.status.code != 0 ||
      verification.status.action != POPS_COMPONENT_CONTINUE_V1)
    throw std::runtime_error(
        verification.status.reason == nullptr
            ? "Writer verify rejected publication without a reason"
            : verification.status.reason);
  receipt = {
      sizeof(PopsWriterReceiptV1), 0, nullptr,
      {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return api.publish(state, &request, &receipt);
}

struct PreparedTopologyLabelV2 {
  std::int32_t id = 0;
  std::string label;
  std::string provenance;
};

struct FieldTopologyPatchInputV2 {
  std::size_t metadata_index = 0;
  PopsFieldMaterialRepresentationV1 material_representation =
      POPS_FIELD_MATERIAL_FULL_V1;
  PopsConstByteViewV1 material_coverage{};
  PopsConstFieldViewV1 cut_cell_volume_fraction{};
  PopsConstInt32ViewV1 material_ids{};
};

struct FieldSolverPatchBindingV2 {
  std::size_t metadata_index = 0;
  PopsConstFieldViewV1 rhs{};
  PopsFieldViewV1 solution{};
  PopsConstFieldViewV1 coefficients{};
};

struct OwnedFieldSolverTopologyLabelV2 {
  std::int32_t id = 0;
  std::string label;
  std::string provenance;
};

class PreparedFieldTopologyV2 final {
 public:
  struct OwnedPatchMetadata {
    std::size_t global_patch_index = 0;
    std::int32_t owner_rank = 0;
    std::int32_t level = 0;
    std::int32_t dimension = 0;
    std::array<std::int64_t, 3> lower{};
    std::array<std::int64_t, 3> upper{};
    std::array<double, 3> physical_lower{};
    std::array<double, 3> cell_spacing{};
    PopsFieldCenteringV1 centering = POPS_FIELD_CENTERING_CELL_V1;
    std::uint32_t centering_axes = 0;
    std::string layout_identity;
    std::string patch_identity;

    [[nodiscard]] PopsFieldPatchMetadataV1 view() const {
      PopsFieldPatchMetadataV1 result{
          sizeof(PopsFieldPatchMetadataV1), global_patch_index, owner_rank, level,
          dimension, {}, {}, {}, {}, centering, centering_axes,
          layout_identity.c_str(), patch_identity.c_str()};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        result.lower[axis] = lower[axis];
        result.upper[axis] = upper[axis];
        result.physical_lower[axis] = physical_lower[axis];
        result.cell_spacing[axis] = cell_spacing[axis];
      }
      return result;
    }
  };

  struct LocalPatchState {
    std::size_t metadata_index = 0;
    PopsFieldMaterialRepresentationV1 material_representation =
        POPS_FIELD_MATERIAL_FULL_V1;
    std::vector<std::uint8_t> expected_material_mask;
    std::vector<std::uint8_t> material_mask;
    std::vector<std::int32_t> component_labels;
  };

  struct ImmutableState {
    std::string topology_recipe_identity;
    std::string source_layout_identity;
    std::string materialized_layout_identity;
    std::int32_t dimension = 0;
    std::array<std::int64_t, 3> domain_lower{};
    std::array<std::int64_t, 3> domain_upper{};
    std::uint32_t periodic_axes = 0;
    std::vector<OwnedPatchMetadata> owned_global_patches;
    std::vector<PopsFieldPatchMetadataV1> global_patches;
    PopsFieldGlobalTopologyV1 global_topology{};
    std::vector<LocalPatchState> local_patches;
    std::vector<PreparedTopologyLabelV2> labels;
    std::string provenance;
    std::string topology_digest;
  };

  [[nodiscard]] const std::string& topology_recipe_identity() const {
    return state_->topology_recipe_identity;
  }
  [[nodiscard]] const std::vector<PopsFieldPatchMetadataV1>& global_patches() const {
    return state_->global_patches;
  }
  [[nodiscard]] const PopsFieldGlobalTopologyV1& global_topology() const {
    return state_->global_topology;
  }
  [[nodiscard]] const std::vector<LocalPatchState>& local_patches() const {
    return state_->local_patches;
  }
  [[nodiscard]] const std::vector<PreparedTopologyLabelV2>& labels() const {
    return state_->labels;
  }
  [[nodiscard]] const std::string& provenance() const { return state_->provenance; }
  [[nodiscard]] const std::string& topology_digest() const {
    return state_->topology_digest;
  }
  [[nodiscard]] const std::shared_ptr<const ImmutableState>& shared_state() const {
    return state_;
  }

 private:
  friend PreparedFieldTopologyV2 prepare_field_topology(
      const PopsFieldTopologyApiV2&, void*, const PopsFieldGlobalTopologyV1&,
      const std::vector<FieldTopologyPatchInputV2>&,
      const PopsExecutionContextV1&);
  friend class TopologyBoundFieldSolverRequestV2;
  std::shared_ptr<const ImmutableState> state_;
};

inline std::size_t field_patch_axis_extent(const PopsFieldPatchMetadataV1& patch,
                                           std::int32_t axis) {
  const std::int64_t lower = patch.lower[axis];
  const std::int64_t upper = patch.upper[axis];
  if (upper < lower)
    throw std::invalid_argument("field patch metadata has an invalid extent");
  const std::uint64_t span = static_cast<std::uint64_t>(upper) -
                             static_cast<std::uint64_t>(lower);
  if (span == std::numeric_limits<std::uint64_t>::max() ||
      span + 1u > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("field patch metadata extent is not representable");
  return static_cast<std::size_t>(span + 1u);
}

inline std::size_t field_patch_point_count(const PopsFieldPatchMetadataV1& patch) {
  std::size_t result = 1;
  for (std::int32_t axis = 0; axis < patch.dimension; ++axis) {
    const std::size_t extent = field_patch_axis_extent(patch, axis);
    if (extent > std::numeric_limits<std::size_t>::max() / result)
      throw std::invalid_argument("field patch metadata has an invalid extent");
    result *= extent;
  }
  return result;
}

inline void validate_field_patch_metadata(
    const PopsFieldPatchMetadataV1& patch, std::size_t expected_index,
    const char* expected_layout = nullptr) {
  if (patch.struct_size < sizeof(PopsFieldPatchMetadataV1) ||
      patch.global_patch_index != expected_index || patch.owner_rank < 0 ||
      patch.level < 0 || patch.dimension < 1 || patch.dimension > 3 ||
      !component_text(patch.layout_identity) || !component_text(patch.patch_identity) ||
      (expected_layout != nullptr &&
       std::string_view(patch.layout_identity) != expected_layout))
    throw std::invalid_argument("field patch metadata is incomplete or non-canonical");
  if (patch.centering != POPS_FIELD_CENTERING_CELL_V1 &&
      patch.centering != POPS_FIELD_CENTERING_FACE_V1 &&
      patch.centering != POPS_FIELD_CENTERING_NODE_V1 &&
      patch.centering != POPS_FIELD_CENTERING_EDGE_V1)
    throw std::invalid_argument("field patch metadata has an unknown centering");
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    if (axis < patch.dimension) {
      if (patch.upper[axis] < patch.lower[axis] ||
          !std::isfinite(patch.physical_lower[axis]) ||
          !std::isfinite(patch.cell_spacing[axis]) || patch.cell_spacing[axis] <= 0.0)
        throw std::invalid_argument("field patch metadata has invalid bounds or spacing");
    } else if (patch.lower[axis] != 0 || patch.upper[axis] != 0 ||
               patch.physical_lower[axis] != 0.0 || patch.cell_spacing[axis] != 0.0) {
      throw std::invalid_argument("field patch metadata has non-canonical unused axes");
    }
  }
  (void)field_patch_point_count(patch);
}

inline void validate_field_global_topology(const PopsFieldGlobalTopologyV1& topology) {
  if (topology.struct_size < sizeof(PopsFieldGlobalTopologyV1) ||
      !component_text(topology.topology_recipe_identity) ||
      !component_text(topology.source_layout_identity) ||
      !component_text(topology.materialized_layout_identity) ||
      topology.dimension < 1 || topology.dimension > 3 || topology.patch_count == 0 ||
      topology.patches == nullptr)
    throw std::invalid_argument("field global topology is incomplete");
  const auto active_axes = (1u << static_cast<unsigned>(topology.dimension)) - 1u;
  if ((topology.periodic_axes & ~active_axes) != 0)
    throw std::invalid_argument("field global topology periodic axes are invalid");
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    if (axis < topology.dimension) {
      if (topology.domain_upper[axis] < topology.domain_lower[axis])
        throw std::invalid_argument("field global topology domain bounds are invalid");
    } else if (topology.domain_lower[axis] != 0 || topology.domain_upper[axis] != 0) {
      throw std::invalid_argument("field global topology has hidden unused-axis bounds");
    }
  }
  for (std::size_t index = 0; index < topology.patch_count; ++index) {
    const auto& patch = topology.patches[index];
    validate_field_patch_metadata(patch, index, topology.source_layout_identity);
    if (patch.dimension != topology.dimension)
      throw std::invalid_argument("field patch dimension differs from global topology");
    for (std::int32_t axis = 0; axis < topology.dimension; ++axis)
      if (patch.lower[axis] < topology.domain_lower[axis] ||
          patch.upper[axis] > topology.domain_upper[axis])
        throw std::invalid_argument("field patch lies outside global topology domain");
  }
}

inline void validate_field_view_matches_patch(
    const PopsConstFieldViewV1& view, const PopsFieldPatchMetadataV1& patch,
    const char* what) {
  if (view.dimension != patch.dimension || view.centering != patch.centering ||
      view.centering_axes != patch.centering_axes ||
      !component_text(view.layout_identity) || !component_text(view.patch_identity) ||
      std::string_view(view.layout_identity) != patch.layout_identity ||
      std::string_view(view.patch_identity) != patch.patch_identity)
    throw std::invalid_argument(std::string(what) +
                                " does not match its exact global patch metadata");
  for (std::int32_t axis = 0; axis < patch.dimension; ++axis)
    if (view.extents[axis] - view.ghost_lower[axis] - view.ghost_upper[axis] !=
        field_patch_axis_extent(patch, axis))
      throw std::invalid_argument(std::string(what) +
                                  " interior extent differs from global patch bounds");
}

inline void validate_field_view_matches_patch(
    const PopsFieldViewV1& view, const PopsFieldPatchMetadataV1& patch,
    const char* what) {
  PopsConstFieldViewV1 read_only{
      view.struct_size, view.data, view.dimension, {}, {}, view.component_count,
      view.component_stride, view.centering, view.centering_axes, {}, {},
      view.scalar_type, view.memory_space, view.layout_identity, view.patch_identity,
      view.ownership};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    read_only.extents[axis] = view.extents[axis];
    read_only.axis_strides[axis] = view.axis_strides[axis];
    read_only.ghost_lower[axis] = view.ghost_lower[axis];
    read_only.ghost_upper[axis] = view.ghost_upper[axis];
  }
  validate_field_view_matches_patch(read_only, patch, what);
}

inline void validate_topology_material_input(
    const FieldTopologyPatchInputV2& input,
    const PopsFieldPatchMetadataV1& metadata,
    const PopsExecutionContextV1& execution) {
  const auto points = field_patch_point_count(metadata);
  const bool has_coverage = input.material_coverage.data != nullptr ||
                            input.material_coverage.size != 0;
  const bool has_fraction = !empty_field_view(input.cut_cell_volume_fraction);
  const bool has_ids = input.material_ids.data != nullptr || input.material_ids.size != 0;
  if (has_coverage &&
      (input.material_coverage.struct_size < sizeof(PopsConstByteViewV1) ||
       input.material_coverage.data == nullptr || input.material_coverage.size != points))
    throw std::invalid_argument("field topology coverage does not match patch bounds");
  if (has_ids &&
      (input.material_ids.struct_size < sizeof(PopsConstInt32ViewV1) ||
       input.material_ids.data == nullptr || input.material_ids.size != points))
    throw std::invalid_argument("field topology material ids do not match patch bounds");
  if (has_fraction) {
    validate_execution_field(execution, input.cut_cell_volume_fraction,
                             "field topology cut-cell volume fraction");
    validate_field_view_matches_patch(
        input.cut_cell_volume_fraction, metadata,
        "field topology cut-cell volume fraction");
    if (input.cut_cell_volume_fraction.component_count != 1)
      throw std::invalid_argument("field topology cut-cell volume fraction must be scalar");
  }
  switch (input.material_representation) {
    case POPS_FIELD_MATERIAL_FULL_V1:
      if (has_coverage || has_fraction || has_ids)
        throw std::invalid_argument(
            "full-material topology must not carry competing material arrays");
      break;
    case POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1:
      if (!has_coverage || has_fraction || has_ids)
        throw std::invalid_argument(
            "binary topology requires only exact material coverage");
      break;
    case POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1:
      if (!has_fraction || has_coverage || has_ids)
        throw std::invalid_argument(
            "cut-cell topology requires only volume fractions");
      break;
    case POPS_FIELD_MATERIAL_IDS_V1:
      if (!has_ids || has_coverage || has_fraction)
        throw std::invalid_argument(
            "multi-material topology requires only material ids");
      break;
    case POPS_FIELD_MATERIAL_IDS_WITH_CUT_CELL_FRACTION_V1:
      if (!has_ids || !has_fraction || has_coverage)
        throw std::invalid_argument(
            "cut-cell multi-material topology requires only ids and volume fractions");
      break;
    default:
      throw std::invalid_argument("field topology material representation is unknown");
  }
}

inline std::vector<std::uint8_t> expected_topology_material_mask(
    const FieldTopologyPatchInputV2& input,
    const PopsFieldPatchMetadataV1& metadata) {
  const std::size_t points = field_patch_point_count(metadata);
  std::vector<std::uint8_t> expected(points, 0);
  auto require_binary = [](std::uint8_t value, const char* what) {
    if (value > 1u)
      throw std::invalid_argument(std::string(what) + " must contain only 0 or 1");
    return value;
  };
  auto fractions = [&]() -> const double* {
    if (input.cut_cell_volume_fraction.memory_space != POPS_MEMORY_SPACE_HOST_V1)
      throw std::invalid_argument(
          "cut-cell topology validation currently requires a host-readable fraction field");
    return static_cast<const double*>(input.cut_cell_volume_fraction.data);
  };
  auto fraction_at = [&](const double* values, std::size_t point) {
    const std::size_t nx = field_patch_axis_extent(metadata, 0);
    const std::size_t i = point % nx;
    const std::size_t j = point / nx;
    const auto& view = input.cut_cell_volume_fraction;
    const std::size_t x = i + view.ghost_lower[0];
    const std::size_t y = j + view.ghost_lower[1];
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    const auto sx = static_cast<std::size_t>(view.axis_strides[0]);
    const auto sy = static_cast<std::size_t>(view.axis_strides[1]);
    if (x > maximum / sx || y > maximum / sy || x * sx > maximum - y * sy)
      throw std::invalid_argument("cut-cell topology fraction strides overflow");
    const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(x * sx + y * sy);
    const double value = values[offset];
    if (!std::isfinite(value) || value < 0.0 || value > 1.0)
      throw std::invalid_argument(
          "cut-cell topology fractions must be finite values in [0, 1]");
    return value;
  };

  switch (input.material_representation) {
    case POPS_FIELD_MATERIAL_FULL_V1:
      std::fill(expected.begin(), expected.end(), std::uint8_t{1});
      break;
    case POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1:
      for (std::size_t point = 0; point < points; ++point)
        expected[point] = require_binary(input.material_coverage.data[point],
                                         "binary material coverage");
      break;
    case POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1: {
      const double* values = fractions();
      for (std::size_t point = 0; point < points; ++point)
        expected[point] = fraction_at(values, point) > 0.0 ? 1u : 0u;
      break;
    }
    case POPS_FIELD_MATERIAL_IDS_V1:
      for (std::size_t point = 0; point < points; ++point) {
        const std::int32_t id = input.material_ids.data[point];
        if (id < 0)
          throw std::invalid_argument("material ids must be non-negative");
        expected[point] = id == 0 ? 0u : 1u;
      }
      break;
    case POPS_FIELD_MATERIAL_IDS_WITH_CUT_CELL_FRACTION_V1: {
      const double* values = fractions();
      for (std::size_t point = 0; point < points; ++point) {
        const std::int32_t id = input.material_ids.data[point];
        const double fraction = fraction_at(values, point);
        if (id < 0 || (id == 0 && fraction != 0.0))
          throw std::invalid_argument(
              "material ids and cut-cell fractions are inconsistent");
        expected[point] = id != 0 && fraction > 0.0 ? 1u : 0u;
      }
      break;
    }
    default:
      throw std::invalid_argument("field topology material representation is unknown");
  }
  return expected;
}

inline PreparedFieldTopologyV2 prepare_field_topology(
    const PopsFieldTopologyApiV2& api, void* state,
    const PopsFieldGlobalTopologyV1& global_topology,
    const std::vector<FieldTopologyPatchInputV2>& local_patches,
    const PopsExecutionContextV1& execution) {
  require_operation(api.prepare_topology != nullptr, "prepare_topology");
  validate_execution_context(execution);
  validate_field_global_topology(global_topology);
  auto storage = std::make_shared<PreparedFieldTopologyV2::ImmutableState>();
  storage->topology_recipe_identity = global_topology.topology_recipe_identity;
  storage->source_layout_identity = global_topology.source_layout_identity;
  storage->materialized_layout_identity = global_topology.materialized_layout_identity;
  storage->dimension = global_topology.dimension;
  storage->periodic_axes = global_topology.periodic_axes;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    storage->domain_lower[axis] = global_topology.domain_lower[axis];
    storage->domain_upper[axis] = global_topology.domain_upper[axis];
  }
  storage->owned_global_patches.reserve(global_topology.patch_count);
  std::unordered_set<std::string> patch_identities;
  for (std::size_t index = 0; index < global_topology.patch_count; ++index) {
    const auto& patch = global_topology.patches[index];
    if (!patch_identities.insert(patch.patch_identity).second)
      throw std::invalid_argument("field topology global patch identities must be unique");
    PreparedFieldTopologyV2::OwnedPatchMetadata owned;
    owned.global_patch_index = patch.global_patch_index;
    owned.owner_rank = patch.owner_rank;
    owned.level = patch.level;
    owned.dimension = patch.dimension;
    owned.centering = patch.centering;
    owned.centering_axes = patch.centering_axes;
    owned.layout_identity = patch.layout_identity;
    owned.patch_identity = patch.patch_identity;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      owned.lower[axis] = patch.lower[axis];
      owned.upper[axis] = patch.upper[axis];
      owned.physical_lower[axis] = patch.physical_lower[axis];
      owned.cell_spacing[axis] = patch.cell_spacing[axis];
    }
    storage->owned_global_patches.push_back(std::move(owned));
  }
  storage->global_patches.reserve(storage->owned_global_patches.size());
  for (const auto& patch : storage->owned_global_patches)
    storage->global_patches.push_back(patch.view());
  storage->global_topology = {
      sizeof(PopsFieldGlobalTopologyV1), storage->topology_recipe_identity.c_str(),
      storage->source_layout_identity.c_str(),
      storage->materialized_layout_identity.c_str(), storage->dimension, {}, {},
      storage->periodic_axes, storage->global_patches.size(),
      storage->global_patches.data()};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    storage->global_topology.domain_lower[axis] = storage->domain_lower[axis];
    storage->global_topology.domain_upper[axis] = storage->domain_upper[axis];
  }
  storage->local_patches.reserve(local_patches.size());
  std::vector<PopsFieldTopologyPatchV2> request_patches;
  request_patches.reserve(local_patches.size());
  std::unordered_set<std::size_t> local_indices;
  for (const auto& input : local_patches) {
    if (input.metadata_index >= storage->global_patches.size() ||
        !local_indices.insert(input.metadata_index).second)
      throw std::invalid_argument(
          "field topology local metadata indices must be unique and globally valid");
    const auto& metadata = storage->global_patches[input.metadata_index];
    validate_topology_material_input(input, metadata, execution);
    PreparedFieldTopologyV2::LocalPatchState local;
    local.metadata_index = input.metadata_index;
    local.material_representation = input.material_representation;
    const auto points = field_patch_point_count(metadata);
    local.expected_material_mask = expected_topology_material_mask(input, metadata);
    // Use values outside both output vocabularies. A component must materialize every point; a
    // zero-initialized buffer would silently turn an omitted full-material patch into an empty one.
    local.material_mask.assign(points, std::uint8_t{0xff});
    local.component_labels.assign(points, std::numeric_limits<std::int32_t>::min());
    storage->local_patches.push_back(std::move(local));
    auto& retained = storage->local_patches.back();
    request_patches.push_back({
        sizeof(PopsFieldTopologyPatchV2), input.metadata_index,
        input.material_representation, input.material_coverage,
        input.cut_cell_volume_fraction, input.material_ids,
        {sizeof(PopsByteViewV1), retained.material_mask.data(),
         retained.material_mask.size()},
        {sizeof(PopsInt32ViewV1), retained.component_labels.data(),
         retained.component_labels.size()}});
  }
  const PopsFieldTopologyRequestV2 request{
      sizeof(PopsFieldTopologyRequestV2), storage->global_topology,
      request_patches.size(), request_patches.empty() ? nullptr : request_patches.data(),
      execution};
  PopsFieldTopologyResultV2 result{};
  result.struct_size = sizeof(PopsFieldTopologyResultV2);
  result.status = unwritten_component_status();
  const int code = api.prepare_topology(state, &request, &result);
  if (result.struct_size < sizeof(PopsFieldTopologyResultV2) ||
      !component_status_is_well_formed(result.status) || code != 0 ||
      result.status.code != 0 ||
      result.status.action != POPS_COMPONENT_CONTINUE_V1)
    throw std::runtime_error(
        result.status.reason == nullptr ? "field topology preparation failed"
                                        : result.status.reason);
  if (result.topology_digest == nullptr || *result.topology_digest == '\0' ||
      result.provenance == nullptr || *result.provenance == '\0' ||
      result.labels == nullptr || result.label_count == 0)
    throw std::runtime_error("field topology returned no stable labels or digest");
  storage->topology_digest = result.topology_digest;
  storage->provenance = result.provenance;
  storage->labels.reserve(result.label_count);
  std::unordered_set<std::int32_t> vocabulary;
  for (std::size_t index = 0; index < result.label_count; ++index) {
    const auto& label = result.labels[index];
    if (label.struct_size < sizeof(PopsTopologyLabelV2) || label.id <= 0 ||
        label.label == nullptr || *label.label == '\0' ||
        label.provenance == nullptr || *label.provenance == '\0' ||
        !vocabulary.insert(label.id).second)
      throw std::runtime_error(
          "field topology label vocabulary must be positive, unique and fully attributed");
    storage->labels.push_back(
        {label.id, std::string(label.label), std::string(label.provenance)});
  }
  for (const auto& local : storage->local_patches) {
    for (std::size_t index = 0; index < local.material_mask.size(); ++index) {
      const auto active = local.material_mask[index];
      const auto label = local.component_labels[index];
      if (active == std::uint8_t{0xff} ||
          label == std::numeric_limits<std::int32_t>::min())
        throw std::runtime_error(
            "field topology provider did not write every material mask and label output");
      if ((active == 0 && label != 0) ||
          (active == 1 && vocabulary.find(label) == vocabulary.end()) || active > 1)
        throw std::runtime_error(
            "field topology mask and connected-component labels are inconsistent");
      if (active != local.expected_material_mask[index])
        throw std::runtime_error(
            "field topology material mask contradicts its typed material representation");
    }
  }
  PreparedFieldTopologyV2 prepared;
  prepared.state_ = std::move(storage);
  return prepared;
}

class TopologyBoundFieldSolverRequestV2 final {
 public:
  [[nodiscard]] const PopsFieldSolverRequestV2& request() const { return request_; }

 private:
  friend TopologyBoundFieldSolverRequestV2 bind_field_solver_request(
      const PreparedFieldTopologyV2&,
      const std::vector<FieldSolverPatchBindingV2>&,
      const PopsExecutionContextV1&, const char*, double, double, std::int32_t);
  friend int solve_field(const PopsFieldSolverApiV2&, void*,
                         const TopologyBoundFieldSolverRequestV2&, PopsSolveReportV2&);
  std::shared_ptr<const PreparedFieldTopologyV2::ImmutableState> topology_;
  std::shared_ptr<const std::string> boundary_contract_;
  std::shared_ptr<std::string> topology_provenance_;
  std::shared_ptr<std::string> topology_digest_;
  std::shared_ptr<std::vector<OwnedFieldSolverTopologyLabelV2>> owned_topology_labels_;
  std::shared_ptr<std::vector<PopsFieldSolverTopologyLabelV2>> topology_labels_;
  std::shared_ptr<std::vector<std::vector<std::uint8_t>>> material_masks_;
  std::shared_ptr<std::vector<std::vector<std::int32_t>>> component_labels_;
  std::shared_ptr<std::vector<PopsFieldSolverPatchV2>> patches_;
  PopsFieldSolverRequestV2 request_{};
};

inline TopologyBoundFieldSolverRequestV2 bind_field_solver_request(
    const PreparedFieldTopologyV2& topology,
    const std::vector<FieldSolverPatchBindingV2>& patch_bindings,
    const PopsExecutionContextV1& execution,
    const char* boundary_contract_json = nullptr, double relative_tolerance = 0.0,
    double absolute_tolerance = 0.0, std::int32_t max_iterations = 0) {
  validate_execution_context(execution);
  if (topology.shared_state() == nullptr)
    throw std::invalid_argument("field solver requires a live prepared topology");
  if (topology.topology_recipe_identity().empty() || topology.global_patches().empty() ||
      topology.labels().empty() || topology.provenance().empty() ||
      topology.topology_digest().empty() ||
      patch_bindings.size() != topology.local_patches().size())
    throw std::invalid_argument("field solver requires an exact prepared topology");
  if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0 ||
      !std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
      max_iterations <= 0 || !component_text(boundary_contract_json) ||
      std::string_view(boundary_contract_json).find("\"identity\"") ==
          std::string_view::npos)
    throw std::invalid_argument(
        "field solver requires finite tolerances, iterations and qualified boundary JSON");
  TopologyBoundFieldSolverRequestV2 bound;
  bound.topology_ = topology.shared_state();
  bound.boundary_contract_ = std::make_shared<const std::string>(boundary_contract_json);
  bound.topology_provenance_ = std::make_shared<std::string>(topology.provenance());
  bound.topology_digest_ = std::make_shared<std::string>(topology.topology_digest());
  bound.owned_topology_labels_ =
      std::make_shared<std::vector<OwnedFieldSolverTopologyLabelV2>>();
  bound.owned_topology_labels_->reserve(topology.labels().size());
  for (const auto& label : topology.labels())
    bound.owned_topology_labels_->push_back(
        {label.id, label.label, label.provenance});
  bound.topology_labels_ =
      std::make_shared<std::vector<PopsFieldSolverTopologyLabelV2>>();
  bound.topology_labels_->reserve(bound.owned_topology_labels_->size());
  for (const auto& label : *bound.owned_topology_labels_)
    bound.topology_labels_->push_back({sizeof(PopsFieldSolverTopologyLabelV2), label.id,
                                       label.label.c_str(), label.provenance.c_str()});
  bound.material_masks_ =
      std::make_shared<std::vector<std::vector<std::uint8_t>>>();
  bound.component_labels_ =
      std::make_shared<std::vector<std::vector<std::int32_t>>>();
  bound.material_masks_->reserve(topology.local_patches().size());
  bound.component_labels_->reserve(topology.local_patches().size());
  for (const auto& patch : topology.local_patches()) {
    bound.material_masks_->push_back(patch.material_mask);
    bound.component_labels_->push_back(patch.component_labels);
  }
  std::vector<PopsFieldSolverPatchV2> patches;
  patches.reserve(patch_bindings.size());
  for (std::size_t index = 0; index < patch_bindings.size(); ++index) {
    const auto& binding = patch_bindings[index];
    const auto& prepared = topology.local_patches()[index];
    if (binding.metadata_index != prepared.metadata_index ||
        binding.metadata_index >= topology.global_patches().size())
      throw std::invalid_argument(
          "field solver local patches differ from prepared topology order");
    const auto& metadata = topology.global_patches()[binding.metadata_index];
    validate_execution_field(execution, binding.rhs, "field solver rhs");
    validate_execution_field(execution, binding.solution, "field solver solution");
    validate_field_view_matches_patch(binding.rhs, metadata, "field solver rhs");
    validate_field_view_matches_patch(binding.solution, metadata,
                                      "field solver solution");
    if (!same_field_domain(binding.rhs, binding.solution) ||
        binding.rhs.component_count != 1 || binding.solution.component_count != 1 ||
        field_interior_point_count(binding.rhs) != prepared.material_mask.size())
      throw std::invalid_argument(
          "field solver RHS/solution must be matching scalar prepared patch views");
    if (!empty_field_view(binding.coefficients)) {
      validate_execution_field(execution, binding.coefficients,
                               "field solver coefficients");
      validate_field_view_matches_patch(binding.coefficients, metadata,
                                        "field solver coefficients");
      if (field_interior_point_count(binding.coefficients) !=
          prepared.material_mask.size())
        throw std::invalid_argument(
            "field solver coefficients do not match their prepared patch interior");
    }
    patches.push_back({
        sizeof(PopsFieldSolverPatchV2), binding.metadata_index, binding.rhs,
        binding.solution, binding.coefficients,
        {sizeof(PopsConstByteViewV1), (*bound.material_masks_)[index].data(),
         (*bound.material_masks_)[index].size()},
        {sizeof(PopsConstInt32ViewV1), (*bound.component_labels_)[index].data(),
         (*bound.component_labels_)[index].size()}});
  }
  bound.patches_ =
      std::make_shared<std::vector<PopsFieldSolverPatchV2>>(std::move(patches));
  bound.request_ = {
      sizeof(PopsFieldSolverRequestV2), topology.global_topology(),
      bound.patches_->size(), bound.patches_->empty() ? nullptr : bound.patches_->data(),
      bound.topology_labels_->size(),
      bound.topology_labels_->empty() ? nullptr : bound.topology_labels_->data(),
      bound.topology_provenance_->c_str(), bound.topology_digest_->c_str(),
      bound.boundary_contract_->c_str(),
      relative_tolerance, absolute_tolerance, max_iterations, execution};
  return bound;
}

inline int solve_field(const PopsFieldSolverApiV2& api, void* state,
                       const TopologyBoundFieldSolverRequestV2& bound,
                       PopsSolveReportV2& report) {
  require_operation(api.solve != nullptr, "solve");
  if (bound.topology_ == nullptr)
    throw std::invalid_argument("field solver request has no prepared topology authority");
  const auto& topology = *bound.topology_;
  const auto& request = bound.request_;
  const auto validate_request_authority = [&]() {
    validate_execution_context(request.execution);
    if (bound.boundary_contract_ == nullptr || bound.topology_provenance_ == nullptr ||
        bound.topology_digest_ == nullptr || bound.owned_topology_labels_ == nullptr ||
        bound.topology_labels_ == nullptr || bound.material_masks_ == nullptr ||
        bound.component_labels_ == nullptr || bound.patches_ == nullptr ||
        request.struct_size < sizeof(PopsFieldSolverRequestV2) ||
        request.topology.struct_size < sizeof(PopsFieldGlobalTopologyV1) ||
        request.topology.topology_recipe_identity !=
            topology.topology_recipe_identity.c_str() ||
        request.topology.source_layout_identity != topology.source_layout_identity.c_str() ||
        request.topology.materialized_layout_identity !=
            topology.materialized_layout_identity.c_str() ||
        request.topology.dimension != topology.dimension ||
        request.topology.patch_count != topology.global_patches.size() ||
        request.topology.patches != topology.global_patches.data() ||
        request.topology.periodic_axes != topology.periodic_axes ||
        request.boundary_contract_json != bound.boundary_contract_->c_str())
      throw std::invalid_argument(
          "field solver request is not paired with its prepared global topology");
    for (std::int32_t axis = 0; axis < 3; ++axis) {
      if (request.topology.domain_lower[axis] != topology.domain_lower[axis] ||
          request.topology.domain_upper[axis] != topology.domain_upper[axis])
        throw std::invalid_argument(
            "field solver request substituted its prepared topology domain");
    }
    if (request.local_patch_count != bound.patches_->size() ||
        request.local_patch_count != topology.local_patches.size() ||
        request.local_patches !=
            (bound.patches_->empty() ? nullptr : bound.patches_->data()) ||
        request.topology_label_count != bound.topology_labels_->size() ||
        request.topology_label_count != bound.owned_topology_labels_->size() ||
        request.topology_label_count != topology.labels.size() ||
        request.topology_labels !=
            (bound.topology_labels_->empty() ? nullptr : bound.topology_labels_->data()) ||
        request.topology_provenance != bound.topology_provenance_->c_str() ||
        request.topology_digest != bound.topology_digest_->c_str() ||
        *bound.topology_provenance_ != topology.provenance ||
        *bound.topology_digest_ != topology.topology_digest ||
        bound.material_masks_->size() != topology.local_patches.size() ||
        bound.component_labels_->size() != topology.local_patches.size())
      throw std::invalid_argument(
          "field solver request omitted or substituted its topological evidence");
    for (std::size_t index = 0; index < request.topology_label_count; ++index) {
      const auto& label = request.topology_labels[index];
      const auto& owned = (*bound.owned_topology_labels_)[index];
      const auto& prepared = topology.labels[index];
      if (label.struct_size < sizeof(PopsFieldSolverTopologyLabelV2) ||
          label.id != owned.id || label.label != owned.label.c_str() ||
          label.provenance != owned.provenance.c_str() || owned.id != prepared.id ||
          owned.label != prepared.label || owned.provenance != prepared.provenance)
        throw std::invalid_argument(
            "field solver label vocabulary is not paired with its prepared topology");
    }
    for (std::size_t index = 0; index < request.local_patch_count; ++index) {
      const auto& patch = request.local_patches[index];
      const auto& prepared = topology.local_patches[index];
      const auto& material_mask = (*bound.material_masks_)[index];
      const auto& component_labels = (*bound.component_labels_)[index];
      if (patch.struct_size < sizeof(PopsFieldSolverPatchV2) ||
          patch.metadata_index != prepared.metadata_index ||
          patch.material_mask.struct_size < sizeof(PopsConstByteViewV1) ||
          patch.material_mask.data != material_mask.data() ||
          patch.material_mask.size != material_mask.size() ||
          patch.component_labels.struct_size < sizeof(PopsConstInt32ViewV1) ||
          patch.component_labels.data != component_labels.data() ||
          patch.component_labels.size != component_labels.size() ||
          material_mask != prepared.material_mask ||
          component_labels != prepared.component_labels)
        throw std::invalid_argument(
            "field solver patch is not paired with its prepared mask and labels");
    }
  };
  validate_request_authority();
  report = {};
  report.struct_size = sizeof(PopsSolveReportV2);
  report.status = static_cast<PopsSolveStatusV2>(-1);
  report.action = static_cast<PopsSolveActionV2>(-1);
  report.iterations = -1;
  report.relative_residual = std::numeric_limits<double>::quiet_NaN();
  report.reference_residual_norm = std::numeric_limits<double>::quiet_NaN();
  report.residual_norm = std::numeric_limits<double>::quiet_NaN();
  report.reason = nullptr;
  const int code = api.solve(state, &request, &report);
  if (code != 0)
    throw std::runtime_error(
        "field solver transport failed with code " + std::to_string(code));
  try {
    validate_request_authority();
  } catch (const std::invalid_argument&) {
    throw std::runtime_error(
        "field solver mutated its authenticated topological request");
  }
  // The relative report denominator uses one when R(0) is zero. The scientific mixed criterion
  // always uses the authentic unmodified ||R(0)|| and therefore remains zero when both tolerances
  // and the reference residual are zero.
  const double ratio_denominator =
      report.reference_residual_norm > 0.0 ? report.reference_residual_norm : 1.0;
  const double expected_relative_residual = report.residual_norm / ratio_denominator;
  const double ratio_scale = std::max(
      {1.0, std::abs(report.relative_residual),
       std::abs(expected_relative_residual)});
  const bool ratio_is_coherent =
      std::isfinite(expected_relative_residual) &&
      std::abs(report.relative_residual - expected_relative_residual) <=
          64.0 * std::numeric_limits<double>::epsilon() * ratio_scale;
  const double relative_threshold =
      request.relative_tolerance * report.reference_residual_norm;
  const double convergence_threshold =
      std::max(request.absolute_tolerance, relative_threshold);
  const bool solved = report.status == POPS_SOLVE_SOLVED_V2;
  if (report.struct_size < sizeof(PopsSolveReportV2) ||
      !solve_status_is_known(report.status) || !solve_action_is_known(report.action) ||
      !component_text(report.reason) || report.iterations < 0 ||
      report.iterations > request.max_iterations ||
      !std::isfinite(report.reference_residual_norm) ||
      report.reference_residual_norm < 0.0 ||
      !std::isfinite(report.residual_norm) || report.residual_norm < 0.0 ||
      !std::isfinite(report.relative_residual) || report.relative_residual < 0.0 ||
      !ratio_is_coherent ||
      !std::isfinite(relative_threshold) || !std::isfinite(convergence_threshold) ||
      solved != (report.action == POPS_SOLVE_ACTION_NONE_V2) ||
      (solved && report.residual_norm > convergence_threshold))
    throw std::runtime_error("field solver returned an incoherent convergence report");
  return code;
}

}  // namespace pops::component
