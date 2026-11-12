/// @file
/// @brief Uniform System seam for the static geometry arrays of a mapped layout (N17, C1).
///
/// The arrays are generated once, at bind, from the analytic programs of a coordinate map that
/// reads no clock (data, never a callback) and from the stated rule of a chart. Every rank
/// generates the same replicated arrays; a rank-local failure becomes one collective refusal
/// before any rank publishes them. Blocks and Program operators then read them through the owner
/// that `static_geometry()` returns.

#include <pops/mesh/geometry/static_geometry.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>

#include "system_impl.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

template <int Dim>
void System<Dim>::install_static_geometry(const std::string& identity, const std::string& kind,
                                          const std::vector<std::vector<std::string>>& opcodes,
                                          const std::vector<std::vector<double>>& literals,
                                          const std::vector<double>& rule_nodes,
                                          const std::vector<double>& rule_weights) {
  require_assembling(p_->lifecycle_, "install_static_geometry");
  std::shared_ptr<const StaticGeometry<Dim>> prepared;
  std::exception_ptr local_error;
  try {
    if (p_->static_geometry_)
      throw std::logic_error("System static geometry arrays are installed once");
    if (p_->blocks_.size() != 0 || !p_->pending_native_packages_.empty())
      throw std::logic_error(
          "System static geometry arrays are installed before any block, which captures them");
    if (p_->embedded_boundary_)
      throw std::invalid_argument(
          "System static geometry arrays and an embedded boundary are not composed");
    StaticGeometryKind parsed = StaticGeometryKind::nodes;
    if (kind == "chart")
      parsed = StaticGeometryKind::chart;
    else if (kind != "nodes")
      throw std::invalid_argument("System static geometry kind is 'nodes' or 'chart'");
    if (rule_nodes.size() != rule_weights.size())
      throw std::invalid_argument("System static geometry rule needs one weight per node");
    StaticGeometryRule rule;
    rule.nodes.assign(rule_nodes.begin(), rule_nodes.end());
    rule.weights.assign(rule_weights.begin(), rule_weights.end());
    const auto programs = analytic::compile_component_programs(opcodes, literals);
    prepared =
        StaticGeometry<Dim>::generate(identity, parsed, programs, rule, p_->geom, p_->periodicity);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System static geometry arrays failed collectively");
  }
  p_->static_geometry_ = std::move(prepared);
}

template <int Dim>
std::shared_ptr<const StaticGeometry<Dim>> System<Dim>::static_geometry() const {
  return p_->static_geometry_;
}

template <int Dim>
void System<Dim>::require_static_geometry_operator(int block, const std::string& operation) const {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System generated Program operator block index is out of range");
  if (operation.empty())
    throw std::invalid_argument("System generated Program operator identity cannot be empty");
  if (!p_->static_geometry_)
    throw std::runtime_error("System generated Program operator '" + operation + "' on block " +
                             std::to_string(block) +
                             " reads static geometry arrays, but none are installed");
}

template void System<kNativeDimension>::install_static_geometry(
    const std::string&, const std::string&, const std::vector<std::vector<std::string>>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&,
    const std::vector<double>&);
template std::shared_ptr<const StaticGeometry<kNativeDimension>>
System<kNativeDimension>::static_geometry() const;
template void System<kNativeDimension>::require_static_geometry_operator(int,
                                                                         const std::string&) const;

}  // namespace pops
