#pragma once

#include <pops/runtime/builders/block/prepared_boundary_defaults.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pops::test_support {

inline std::shared_ptr<PreparedBoundaryPlan> polar_boundary_plan(int ncomp, bool close_radial_flux,
                                                                 int required_depth) {
  BCRec descriptor;
  descriptor.xlo = descriptor.xhi = BCType::Foextrap;
  std::vector<std::string> names;
  names.reserve(static_cast<std::size_t>(ncomp));
  for (int component = 0; component < ncomp; ++component)
    names.push_back("u" + std::to_string(component));
  VariableSet variables{
      VariableKind::Conservative, std::move(names), ncomp,
      std::vector<VariableRole>(static_cast<std::size_t>(ncomp), VariableRole::Scalar)};
  return detail::prepare_builtin_boundary_plan(
      close_radial_flux ? "test-polar-closed" : "test-polar-outflow", {}, required_depth, variables,
      descriptor, close_radial_flux);
}

}  // namespace pops::test_support
