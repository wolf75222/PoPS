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
  if (context.communicator_f_handle == 0) {
    if (context.communicator_datatype_f_handle != 0 ||
        std::string(context.communicator_identity) != "serial" ||
        std::string(context.communicator_datatype_identity) != "none")
      throw std::invalid_argument(
          "serial component execution context cannot hide MPI handles or identities");
  } else if (context.communicator_f_handle < 0 ||
             context.communicator_datatype_f_handle <= 0 ||
             std::string(context.communicator_identity) == "serial" ||
             std::string(context.communicator_datatype_identity) == "none") {
    throw std::invalid_argument(
        "distributed component execution context must declare exact Fortran MPI handles");
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
          view.ghost_lower[axis] + view.ghost_upper[axis] >= view.extents[axis])
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
  validate_execution_field(request.execution, request.state, "tagger state");
  if (request.tags.struct_size < sizeof(PopsByteViewV1) || request.tags.data == nullptr ||
      request.tags.size != field_point_count(request.state))
    throw std::invalid_argument("tagger output does not match its field descriptor");
  return api.tag_batch(state, &request, &status);
}

inline int cluster_tags(const PopsClusteringApiV1& api, void* state,
                        const PopsClusteringRequestV1& request,
                        PopsComponentStatusV1& status) {
  require_operation(api.cluster != nullptr, "cluster");
  validate_execution_context(request.execution);
  return api.cluster(state, &request, &status);
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

inline int deposit_reflux(const PopsRefluxApiV1& api, void* state,
                          const PopsRefluxRequestV1& request,
                          PopsComponentStatusV1& status) {
  require_operation(api.deposit_integrated != nullptr, "deposit_integrated");
  validate_execution_context(request.execution);
  validate_execution_field(request.execution, request.coarse_integrated,
                           "reflux coarse integral");
  validate_execution_field(request.execution, request.fine_integrated,
                           "reflux fine integral");
  validate_execution_field(request.execution, request.flux_register,
                           "reflux register");
  if (!same_field_domain(request.coarse_integrated, request.fine_integrated) ||
      !same_field_domain(request.coarse_integrated, request.flux_register))
    throw std::invalid_argument("reflux field descriptors disagree");
  return api.deposit_integrated(state, &request, &status);
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

struct PreparedTopologyLabelV1 {
  std::int32_t id = 0;
  std::string label;
  std::string provenance;
};

class PreparedFieldTopologyV1 final {
 public:
  struct OwnedFieldDescriptor {
    std::int32_t dimension = 0;
    std::array<std::size_t, 3> extents{};
    std::array<std::ptrdiff_t, 3> axis_strides{};
    std::size_t component_count = 0;
    std::ptrdiff_t component_stride = 0;
    PopsFieldCenteringV1 centering = POPS_FIELD_CENTERING_CELL_V1;
    std::uint32_t centering_axes = 0;
    std::array<std::size_t, 3> ghost_lower{};
    std::array<std::size_t, 3> ghost_upper{};
    PopsScalarTypeV1 scalar_type = POPS_SCALAR_FLOAT64_V1;
    PopsMemorySpaceV1 memory_space = POPS_MEMORY_SPACE_HOST_V1;
    std::string layout_identity;
    std::string patch_identity;
    PopsFieldOwnershipV1 ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;

    [[nodiscard]] PopsConstFieldViewV1 rebind(const void* data) const {
      PopsConstFieldViewV1 view{
          sizeof(PopsConstFieldViewV1), data, dimension, {}, {}, component_count,
          component_stride, centering, centering_axes, {}, {}, scalar_type,
          memory_space, layout_identity.c_str(), patch_identity.c_str(), ownership};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        view.extents[axis] = extents[axis];
        view.axis_strides[axis] = axis_strides[axis];
        view.ghost_lower[axis] = ghost_lower[axis];
        view.ghost_upper[axis] = ghost_upper[axis];
      }
      return view;
    }
  };

  struct ImmutableState {
    OwnedFieldDescriptor geometry_descriptor;
    std::vector<std::uint8_t> material_mask;
    std::vector<std::int32_t> component_labels;
    std::vector<PreparedTopologyLabelV1> labels;
    std::string provenance;
    std::string topology_digest;
  };

  [[nodiscard]] const std::vector<std::uint8_t>& material_mask() const {
    return state_->material_mask;
  }
  [[nodiscard]] const std::vector<std::int32_t>& component_labels() const {
    return state_->component_labels;
  }
  [[nodiscard]] const std::vector<PreparedTopologyLabelV1>& labels() const {
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
  friend PreparedFieldTopologyV1 prepare_field_topology(
      const PopsFieldTopologyApiV1&, void*, const PopsConstFieldViewV1&,
      const PopsExecutionContextV1&);
  friend class TopologyBoundFieldSolverRequestV1;
  std::shared_ptr<const ImmutableState> state_;
};

inline PreparedFieldTopologyV1 prepare_field_topology(
    const PopsFieldTopologyApiV1& api, void* state,
    const PopsConstFieldViewV1& geometry,
    const PopsExecutionContextV1& execution) {
  require_operation(api.prepare_topology != nullptr, "prepare_topology");
  validate_execution_context(execution);
  validate_execution_field(execution, geometry, "field topology geometry");
  if (geometry.component_count != 1)
    throw std::invalid_argument("field topology geometry must be scalar");
  auto storage = std::make_shared<PreparedFieldTopologyV1::ImmutableState>();
  storage->geometry_descriptor.dimension = geometry.dimension;
  storage->geometry_descriptor.component_count = geometry.component_count;
  storage->geometry_descriptor.component_stride = geometry.component_stride;
  storage->geometry_descriptor.centering = geometry.centering;
  storage->geometry_descriptor.centering_axes = geometry.centering_axes;
  storage->geometry_descriptor.scalar_type = geometry.scalar_type;
  storage->geometry_descriptor.memory_space = geometry.memory_space;
  storage->geometry_descriptor.layout_identity = geometry.layout_identity;
  storage->geometry_descriptor.patch_identity = geometry.patch_identity;
  storage->geometry_descriptor.ownership = geometry.ownership;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    storage->geometry_descriptor.extents[axis] = geometry.extents[axis];
    storage->geometry_descriptor.axis_strides[axis] = geometry.axis_strides[axis];
    storage->geometry_descriptor.ghost_lower[axis] = geometry.ghost_lower[axis];
    storage->geometry_descriptor.ghost_upper[axis] = geometry.ghost_upper[axis];
  }
  const auto points = field_point_count(geometry);
  storage->material_mask.resize(points);
  storage->component_labels.resize(points);
  const PopsFieldTopologyRequestV1 request{
      sizeof(PopsFieldTopologyRequestV1), geometry,
      {sizeof(PopsByteViewV1), storage->material_mask.data(),
       storage->material_mask.size()},
      {sizeof(PopsInt32ViewV1), storage->component_labels.data(),
       storage->component_labels.size()},
      execution};
  PopsFieldTopologyResultV1 result{};
  result.struct_size = sizeof(PopsFieldTopologyResultV1);
  result.status = {sizeof(PopsComponentStatusV1), 0,
                   POPS_COMPONENT_CONTINUE_V1, nullptr};
  const int code = api.prepare_topology(state, &request, &result);
  if (result.struct_size < sizeof(PopsFieldTopologyResultV1) ||
      result.status.struct_size < sizeof(PopsComponentStatusV1) || code != 0 ||
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
    if (label.id <= 0 || label.label == nullptr || *label.label == '\0' ||
        label.provenance == nullptr || *label.provenance == '\0' ||
        !vocabulary.insert(label.id).second)
      throw std::runtime_error(
          "field topology label vocabulary must be positive, unique and fully attributed");
    storage->labels.push_back(
        {label.id, std::string(label.label), std::string(label.provenance)});
  }
  for (std::size_t index = 0; index < storage->material_mask.size(); ++index) {
    const auto active = storage->material_mask[index];
    const auto label = storage->component_labels[index];
    if ((active == 0 && label != 0) ||
        (active == 1 && vocabulary.find(label) == vocabulary.end()) || active > 1)
      throw std::runtime_error(
          "field topology mask and connected-component labels are inconsistent");
  }
  PreparedFieldTopologyV1 prepared;
  prepared.state_ = std::move(storage);
  return prepared;
}

class TopologyBoundFieldSolverRequestV1 final {
 public:
  [[nodiscard]] const PopsFieldSolverRequestV1& request() const { return request_; }

 private:
  friend TopologyBoundFieldSolverRequestV1 bind_field_solver_request(
      const PreparedFieldTopologyV1&, const PopsConstFieldViewV1&,
      const PopsFieldViewV1&, const PopsExecutionContextV1&,
      const PopsConstFieldViewV1&, const char*, double, double, std::int32_t);
  friend int solve_field(const PopsFieldSolverApiV1&, void*,
                         const TopologyBoundFieldSolverRequestV1&, PopsSolveReportV1&);
  std::shared_ptr<const PreparedFieldTopologyV1::ImmutableState> topology_;
  std::shared_ptr<const std::string> boundary_contract_;
  PopsFieldSolverRequestV1 request_{};
};

inline TopologyBoundFieldSolverRequestV1 bind_field_solver_request(
    const PreparedFieldTopologyV1& topology, const PopsConstFieldViewV1& rhs,
    const PopsFieldViewV1& solution, const PopsExecutionContextV1& execution,
    const PopsConstFieldViewV1& coefficients = {},
    const char* boundary_contract_json = nullptr, double relative_tolerance = 0.0,
    double absolute_tolerance = 0.0, std::int32_t max_iterations = 0) {
  validate_execution_context(execution);
  if (topology.shared_state() == nullptr)
    throw std::invalid_argument("field solver requires a live prepared topology");
  if (topology.material_mask().empty() || topology.component_labels().empty() ||
      topology.topology_digest().empty())
    throw std::invalid_argument("field solver requires an exact prepared topology");
  validate_execution_field(execution, rhs, "field solver rhs");
  validate_execution_field(execution, solution, "field solver solution");
  const auto geometry_descriptor =
      topology.shared_state()->geometry_descriptor.rebind(rhs.data);
  validate_execution_field(execution, geometry_descriptor,
                           "prepared field topology geometry");
  if (!same_field_domain(rhs, solution) ||
      !same_spatial_domain(rhs, geometry_descriptor) ||
      field_point_count(rhs) != topology.material_mask().size())
    throw std::invalid_argument(
        "field solver fields must exactly match the prepared topology extent");
  if (!empty_field_view(coefficients)) {
    validate_execution_field(execution, coefficients, "field solver coefficients");
    if (!same_spatial_domain(rhs, coefficients))
      throw std::invalid_argument(
          "field solver coefficients do not match the prepared topology domain");
  }
  if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0 ||
      !std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
      max_iterations <= 0 || !component_text(boundary_contract_json) ||
      std::string_view(boundary_contract_json).find("\"identity\"") ==
          std::string_view::npos)
    throw std::invalid_argument(
        "field solver requires finite tolerances, iterations and qualified boundary JSON");
  TopologyBoundFieldSolverRequestV1 bound;
  bound.topology_ = topology.shared_state();
  bound.boundary_contract_ = std::make_shared<const std::string>(boundary_contract_json);
  bound.request_ = {
      sizeof(PopsFieldSolverRequestV1), rhs, solution, coefficients,
      {sizeof(PopsConstByteViewV1), topology.material_mask().data(),
       topology.material_mask().size()},
      {sizeof(PopsConstInt32ViewV1), topology.component_labels().data(),
       topology.component_labels().size()},
      topology.topology_digest().c_str(), bound.boundary_contract_->c_str(), relative_tolerance,
      absolute_tolerance, max_iterations, execution};
  return bound;
}

inline int solve_field(const PopsFieldSolverApiV1& api, void* state,
                       const TopologyBoundFieldSolverRequestV1& bound,
                       PopsSolveReportV1& report) {
  require_operation(api.solve != nullptr, "solve");
  validate_execution_context(bound.request_.execution);
  if (bound.topology_ == nullptr)
    throw std::invalid_argument("field solver request has no prepared topology authority");
  const auto& topology = *bound.topology_;
  const auto& request = bound.request_;
  if (request.material_mask.data != topology.material_mask.data() ||
      request.material_mask.size != topology.material_mask.size() ||
      request.component_labels.data != topology.component_labels.data() ||
      request.component_labels.size != topology.component_labels.size() ||
      request.topology_digest == nullptr ||
      request.topology_digest != topology.topology_digest.c_str() ||
      field_point_count(request.rhs) != topology.material_mask.size() ||
      field_point_count(request.solution) != topology.material_mask.size())
    throw std::invalid_argument(
        "field solver request is not paired with its prepared mask, labels and digest");
  report = {};
  report.struct_size = sizeof(PopsSolveReportV1);
  report.status = {sizeof(PopsComponentStatusV1), 0,
                   POPS_COMPONENT_CONTINUE_V1, nullptr};
  const int code = api.solve(state, &request, &report);
  if (report.struct_size < sizeof(PopsSolveReportV1) ||
      (report.converged != 0 && report.converged != 1) || report.iterations < 0 ||
      report.iterations > request.max_iterations ||
      !std::isfinite(report.initial_residual) || report.initial_residual < 0.0 ||
      !std::isfinite(report.final_residual) || report.final_residual < 0.0 ||
      (report.converged != 0 &&
       (code != 0 || report.status.code != 0 ||
        report.status.action != POPS_COMPONENT_CONTINUE_V1 ||
        report.final_residual > report.initial_residual)) ||
      (report.converged == 0 && report.status.action == POPS_COMPONENT_CONTINUE_V1))
    throw std::runtime_error("field solver returned an incoherent convergence report");
  return code;
}

}  // namespace pops::component
