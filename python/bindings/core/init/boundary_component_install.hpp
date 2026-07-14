#pragma once

#include <pops/mesh/boundary/prepared_boundary_component.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/multiblock/prepared_interface_flux_component.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::python::detail {

namespace py = pybind11;

inline std::shared_ptr<const component::PreparedExecutionContextV1>
make_component_execution_context(const py::dict& row) {
  return std::make_shared<const component::PreparedExecutionContextV1>(
      py::cast<std::string>(row["execution_identity"]),
      py::cast<std::uint32_t>(row["context_version"]),
      static_cast<PopsMemorySpaceV1>(py::cast<std::int32_t>(row["memory_space"])),
      py::cast<std::string>(row["backend_identity"]),
      py::cast<std::string>(row["device_identity"]),
      static_cast<PopsScalarTypeV1>(py::cast<std::int32_t>(row["scalar_type"])),
      static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["storage_precision"])),
      static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["compute_precision"])),
      static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["accumulation_precision"])),
      static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["reduction_precision"])),
      py::cast<std::uint64_t>(row["stream_handle"]),
      py::cast<std::string>(row["stream_identity"]),
      py::cast<std::int64_t>(row["communicator_f_handle"]),
      py::cast<std::int64_t>(row["communicator_datatype_f_handle"]),
      py::cast<std::string>(row["communicator_identity"]),
      py::cast<std::string>(row["communicator_datatype_identity"]));
}

inline runtime::field::PreparedFieldSolverSpec field_solver_spec_from_python(
    std::string provider_slot, const py::dict& topology, const py::dict& solver,
    std::string topology_parameters_json, std::string solver_parameters_json,
    std::string source_layout_identity, std::string topology_recipe_identity,
    std::string boundary_contract_json,
    double relative_tolerance, double absolute_tolerance, std::int32_t max_iterations,
    const py::dict& execution_data) {
  runtime::field::PreparedFieldSolverSpec spec;
  spec.provider_slot = std::move(provider_slot);
  spec.topology_component_id =
      py::cast<std::string>(topology["component_id"]);
  spec.topology_manifest_identity =
      py::cast<std::string>(topology["component_manifest_identity"]);
  spec.topology_interface_version =
      py::cast<std::uint32_t>(topology["interface_version"]);
  spec.topology_parameters_json = std::move(topology_parameters_json);
  spec.solver_component_id = py::cast<std::string>(solver["component_id"]);
  spec.solver_manifest_identity =
      py::cast<std::string>(solver["component_manifest_identity"]);
  spec.solver_interface_version =
      py::cast<std::uint32_t>(solver["interface_version"]);
  spec.solver_parameters_json = std::move(solver_parameters_json);
  spec.source_layout_identity = std::move(source_layout_identity);
  spec.topology_recipe_identity = std::move(topology_recipe_identity);
  spec.boundary_contract_json = std::move(boundary_contract_json);
  spec.relative_tolerance = relative_tolerance;
  spec.absolute_tolerance = absolute_tolerance;
  spec.max_iterations = max_iterations;
  spec.execution = make_component_execution_context(execution_data);
  return spec;
}

inline PopsBoundaryRegionKindV1 boundary_region_kind(const std::string& kind) {
  if (kind == "face") return POPS_BOUNDARY_FACE_V1;
  if (kind == "edge") return POPS_BOUNDARY_EDGE_V1;
  if (kind == "corner") return POPS_BOUNDARY_CORNER_V1;
  throw std::invalid_argument("native boundary component region kind is unknown");
}

inline PreparedBoundaryComponentSpec make_boundary_component_spec(
    std::string target_identity, std::string component_id,
    std::string manifest_identity, std::uint32_t interface_version,
    std::string producer_identity, std::string state_identity, std::string ghost_identity,
    std::string layout_identity,
    std::string region_kind, int dimension, int codimension, std::vector<int> axes,
    std::vector<int> sides, std::string region_identity, std::vector<std::string> states,
    std::vector<std::string> directions, std::vector<std::string> fields,
    std::vector<std::string> parameter_ids, std::vector<double> parameter_values,
    std::vector<std::string> outputs, std::string rate, std::string nonlinear_iterate,
    std::string parameters_json, std::string target_json,
    std::shared_ptr<const component::PreparedExecutionContextV1> execution) {
  PreparedBoundaryComponentSpec spec;
  spec.target_identity = std::move(target_identity);
  spec.component_id = std::move(component_id);
  spec.manifest_identity = std::move(manifest_identity);
  spec.interface_version = interface_version;
  spec.producer_identity = std::move(producer_identity);
  spec.state_identity = std::move(state_identity);
  spec.ghost_identity = std::move(ghost_identity);
  spec.layout_identity = std::move(layout_identity);
  spec.region.kind = boundary_region_kind(region_kind);
  spec.region.dimension = dimension;
  spec.region.codimension = codimension;
  spec.region.axes.assign(axes.begin(), axes.end());
  spec.region.sides.assign(sides.begin(), sides.end());
  spec.region.identity = std::move(region_identity);
  spec.states = std::move(states);
  spec.directions = std::move(directions);
  spec.fields = std::move(fields);
  spec.parameter_ids = std::move(parameter_ids);
  spec.parameter_values = std::move(parameter_values);
  spec.outputs = std::move(outputs);
  spec.rate = std::move(rate);
  spec.nonlinear_iterate = std::move(nonlinear_iterate);
  spec.parameters_json = std::move(parameters_json);
  spec.target_json = std::move(target_json);
  spec.execution = std::move(execution);
  return spec;
}

inline PreparedBoundaryComponentSpec boundary_component_spec_from_python(
    const py::dict& row, const std::string& parameters_json,
    const std::string& target_json, const py::dict& execution_data) {
  const py::dict target = py::cast<py::dict>(row["target"]);
  const py::dict region = py::cast<py::dict>(row["region"]);
  std::vector<std::string> parameter_ids;
  std::vector<double> parameter_values;
  for (const py::handle value : py::cast<py::list>(row["parameters"])) {
    const py::dict parameter = py::cast<py::dict>(value);
    parameter_ids.push_back(py::cast<std::string>(parameter["qualified_id"]));
    parameter_values.push_back(py::cast<double>(parameter["value"]));
  }
  auto optional_identity = [](const py::handle value) {
    return value.is_none() ? std::string() : py::cast<std::string>(value);
  };
  return make_boundary_component_spec(
      py::cast<std::string>(target["qualified_id"]),
      py::cast<std::string>(row["component_id"]),
      py::cast<std::string>(row["component_manifest_identity"]),
      py::cast<std::uint32_t>(row["interface_version"]),
      py::cast<std::string>(row["producer_identity"]),
      py::cast<std::string>(row["state_identity"]),
      py::cast<std::string>(row["ghost_identity"]),
      py::cast<std::string>(region["layout_identity"]),
      py::cast<std::string>(region["kind"]), py::cast<int>(region["dimension"]),
      py::cast<int>(region["codimension"]),
      py::cast<std::vector<int>>(region["axes"]),
      py::cast<std::vector<int>>(region["sides"]),
      py::cast<std::string>(region["region_identity"]),
      py::cast<std::vector<std::string>>(row["states"]),
      py::cast<std::vector<std::string>>(row["directions"]),
      py::cast<std::vector<std::string>>(row["fields"]),
      std::move(parameter_ids), std::move(parameter_values),
      py::cast<std::vector<std::string>>(row["outputs"]),
      optional_identity(row["rate"]), optional_identity(row["nonlinear_iterate"]),
      parameters_json, target_json, make_component_execution_context(execution_data));
}

inline runtime::multiblock::InterfaceAxis interface_axis(int axis) {
  if (axis == 0) return runtime::multiblock::InterfaceAxis::X;
  if (axis == 1) return runtime::multiblock::InterfaceAxis::Y;
  throw std::invalid_argument("native shared interface requires a 2-D axis (0 or 1)");
}

inline runtime::multiblock::InterfaceSide interface_side(const std::string& side) {
  if (side == "lower") return runtime::multiblock::InterfaceSide::Low;
  if (side == "upper") return runtime::multiblock::InterfaceSide::High;
  throw std::invalid_argument("native shared interface side must be lower or upper");
}

inline runtime::multiblock::AxisAlignedInterface interface_route_from_python(
    const py::dict& row, std::size_t left_block, std::size_t right_block, int level) {
  const py::dict handle = py::cast<py::dict>(row["handle"]);
  const py::dict left = py::cast<py::dict>(row["left"]);
  const py::dict right = py::cast<py::dict>(row["right"]);
  const py::dict left_orientation = py::cast<py::dict>(left["orientation"]);
  const py::dict right_orientation = py::cast<py::dict>(right["orientation"]);
  const py::dict permutation = py::cast<py::dict>(row["permutation"]);
  const py::dict mapping = py::cast<py::dict>(row["mapping"]);
  const py::dict mapping_handle = py::cast<py::dict>(mapping["handle"]);
  const std::string tangential = py::cast<std::string>(mapping["tangential_orientation"]);
  runtime::multiblock::TangentialOrientation orientation;
  if (tangential == "aligned")
    orientation = runtime::multiblock::TangentialOrientation::Aligned;
  else if (tangential == "reversed")
    orientation = runtime::multiblock::TangentialOrientation::Reversed;
  else
    throw std::invalid_argument(
        "native shared interface tangential orientation must be aligned or reversed");
  runtime::multiblock::AxisAlignedInterface route;
  route.identity = py::cast<std::string>(handle["qualified_id"]);
  route.left_block = left_block;
  route.right_block = right_block;
  route.level = level;
  route.left_axis = interface_axis(py::cast<int>(left_orientation["axis"]));
  route.right_axis = interface_axis(py::cast<int>(right_orientation["axis"]));
  route.left_side = interface_side(py::cast<std::string>(left_orientation["side"]));
  route.right_side = interface_side(py::cast<std::string>(right_orientation["side"]));
  route.tangential_orientation = orientation;
  route.right_component_for_left =
      py::cast<std::vector<int>>(permutation["right_component_for_left"]);
  route.affine_mapping_identity =
      py::cast<std::string>(mapping_handle["qualified_id"]);
  route.right_normal_translation =
      static_cast<Real>(py::cast<double>(mapping["right_normal_translation"]));
  route.right_tangential_scale =
      static_cast<Real>(py::cast<double>(mapping["right_tangential_scale"]));
  route.right_tangential_offset =
      static_cast<Real>(py::cast<double>(mapping["right_tangential_offset"]));
  return route;
}

inline runtime::multiblock::PreparedInterfaceFluxSpec interface_flux_spec_from_python(
    const py::dict& interface, const py::dict& binding, const std::string& parameters_json,
    const std::string& target_json, const py::dict& execution_data) {
  const py::dict handle = py::cast<py::dict>(interface["handle"]);
  const std::string identity = py::cast<std::string>(handle["qualified_id"]);
  if (py::cast<std::string>(binding["operation"]) != "evaluate_faces")
    throw std::invalid_argument(
        "shared conservative flux requires the typed NumericalFlux evaluate_faces operation");
  runtime::multiblock::PreparedInterfaceFluxSpec spec;
  spec.interface_identity = identity;
  spec.component_id = py::cast<std::string>(binding["component_id"]);
  spec.manifest_identity =
      py::cast<std::string>(binding["component_manifest_identity"]);
  spec.interface_version = py::cast<std::uint32_t>(binding["interface_version"]);
  spec.canonical_layout_identity = identity + "::canonical-face-layout@1";
  spec.parameters_json = parameters_json;
  spec.target_json = target_json;
  spec.execution = make_component_execution_context(execution_data);
  return spec;
}

}  // namespace pops::python::detail
