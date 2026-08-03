#pragma once

#include <pops/core/state/variables.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/boundary/prepared_boundary_plan.hpp>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::detail {

inline const char* prepared_boundary_role_token(VariableRole role) {
  switch (role) {
    case VariableRole::Density:
      return "Density";
    case VariableRole::MomentumX:
      return "MomentumX";
    case VariableRole::MomentumY:
      return "MomentumY";
    case VariableRole::MomentumZ:
      return "MomentumZ";
    case VariableRole::Energy:
      return "Energy";
    case VariableRole::VelocityX:
      return "VelocityX";
    case VariableRole::VelocityY:
      return "VelocityY";
    case VariableRole::VelocityZ:
      return "VelocityZ";
    case VariableRole::Pressure:
      return "Pressure";
    case VariableRole::Temperature:
      return "Temperature";
    case VariableRole::Scalar:
      return "Scalar";
    case VariableRole::Custom:
      return "Custom";
    case VariableRole::AxialX:
      return "AxialX";
    case VariableRole::AxialY:
      return "AxialY";
    case VariableRole::AxialZ:
      return "AxialZ";
  }
  throw std::logic_error("unknown variable role in prepared boundary lowering");
}

inline std::string prepared_boundary_face_type(BCType type) {
  switch (type) {
    case BCType::Periodic:
      return "periodic";
    case BCType::Foextrap:
      return "foextrap";
    case BCType::Dirichlet:
      return "dirichlet";
    case BCType::Robin:
      throw std::invalid_argument("hyperbolic transport has no prepared Robin boundary provider");
    case BCType::External:
      throw std::invalid_argument(
          "hyperbolic transport requires an explicitly installed external boundary provider");
  }
  throw std::logic_error("unknown BCType in prepared boundary lowering");
}

/// Lower the legacy mesh-level BC descriptor exactly once during block materialization. The
/// returned PreparedBoundaryPlan is the only executable transport authority retained by the
/// closure. `close_radial_flux` is the annular default: radial ghosts remain extrapolated while the
/// already evaluated radial numerical flux is closed by the plan's NoFlux law.
inline std::shared_ptr<PreparedBoundaryPlan> prepare_builtin_boundary_plan(
    const std::string& block_name, const std::string& state_identity, int required_depth,
    const VariableSet& variables, const BCRec& descriptor, bool close_radial_flux = false) {
  if (block_name.empty() || required_depth < 1 || variables.size < 1 ||
      static_cast<int>(variables.names.size()) != variables.size ||
      (!variables.roles.empty() && static_cast<int>(variables.roles.size()) != variables.size))
    throw std::invalid_argument("built-in prepared boundary requires one complete block layout");
  validate_periodic_pairs(descriptor);

  const std::array<BCType, 4> types{descriptor.xlo, descriptor.xhi, descriptor.ylo, descriptor.yhi};
  std::vector<std::string> face_types;
  std::vector<std::string> face_identities;
  face_types.reserve(4);
  face_identities.reserve(4);
  for (int face = 0; face < 4; ++face) {
    const bool radial_physical = close_radial_flux && face < 2 && types[face] != BCType::Periodic;
    face_types.push_back(radial_physical ? "no_flux" : prepared_boundary_face_type(types[face]));
    face_identities.push_back("pops://runtime/boundary/" + block_name + "/face/" +
                              std::to_string(face));
  }

  std::vector<std::string> component_roles;
  component_roles.reserve(static_cast<std::size_t>(variables.size));
  for (int component = 0; component < variables.size; ++component) {
    const VariableRole role = variables.roles.empty()
                                  ? VariableRole::Custom
                                  : variables.roles[static_cast<std::size_t>(component)];
    component_roles.emplace_back(prepared_boundary_role_token(role));
  }

  const std::array<double, 4> values{
      static_cast<double>(descriptor.xlo_val), static_cast<double>(descriptor.xhi_val),
      static_cast<double>(descriptor.ylo_val), static_cast<double>(descriptor.yhi_val)};
  std::vector<double> face_values(static_cast<std::size_t>(4 * variables.size), 0.0);
  for (int component = 0; component < variables.size; ++component)
    for (int face = 0; face < 4; ++face)
      if (types[face] == BCType::Dirichlet)
        face_values[static_cast<std::size_t>(4 * component + face)] = values[face];

  auto hyperbolic =
      prepare_hyperbolic_boundary<2>(face_types, face_values, face_identities, component_roles);
  return std::make_shared<PreparedBoundaryPlan>(
      "pops://runtime/boundary/" + block_name + "/builtin@1", required_depth, std::move(hyperbolic),
      std::vector<int>{}, state_identity);
}

}  // namespace pops::detail
