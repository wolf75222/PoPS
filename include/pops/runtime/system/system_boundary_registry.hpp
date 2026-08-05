/// @file
/// @brief Transactional exact-ranked boundary/state-route registry shared by System facades.

#pragma once

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::system {

/// Pre-materialization ownership of exact state routes and model-qualified hyperbolic boundaries.
///
/// Lifecycle and collective-consensus checks stay with the owning facade. This value provides the
/// shared strong-exception-safe registry operation used by both uniform and AMR implementations.
/// A failed Python installation transaction clears all three route tables together.
template <int Dim>
class SystemBoundaryRegistry {
  static_assert(Dim >= 1 && Dim <= 3,
                "SystemBoundaryRegistry only supports dimensions 1, 2, and 3");

 public:
  using boundary_type = PreparedHyperbolicBoundary<Dim>;

  struct InstalledBoundary {
    std::string identity;
    int required_depth = 0;
    std::string state_identity;
    std::shared_ptr<const boundary_type> authority;
  };

  void install_state_route(std::string name, std::string state_identity) {
    if (name.empty() || state_identity.empty() || state_routes_.contains(name))
      throw std::runtime_error(
          "System block state route requires unique non-empty block/state identities");
    for (const auto& [_, installed] : state_routes_)
      if (installed == state_identity)
        throw std::runtime_error(
            "System block state route has a duplicate qualified state identity");
    state_routes_.emplace(std::move(name), std::move(state_identity));
  }

  void install_field_storage_route(std::string field_identity, std::string provider_slot) {
    if (field_identity.empty() || provider_slot.empty() ||
        !field_storage_routes_.emplace(std::move(field_identity), std::move(provider_slot)).second)
      throw std::runtime_error(
          "System field storage route requires unique non-empty qualified identities");
  }

  void install_boundary(std::string name, std::string identity, int required_depth,
                        std::string state_identity,
                        std::shared_ptr<const boundary_type> authority) {
    if (name.empty() || identity.empty() || state_identity.empty() || required_depth < 1 ||
        !authority || authority->ncomp() < 1)
      throw std::runtime_error(
          "System hyperbolic boundary requires exact identities, positive depth, and components");
    const auto route = state_routes_.find(name);
    if (route == state_routes_.end() || route->second != state_identity)
      throw std::runtime_error(
          "System hyperbolic boundary state differs from the exact block state route");
    if (boundaries_.contains(name))
      throw std::runtime_error("System hyperbolic boundary has a duplicate block route");
    for (const auto& [_, installed] : boundaries_)
      if (installed.identity == identity || installed.state_identity == state_identity)
        throw std::runtime_error("System hyperbolic boundary has a duplicate qualified identity");

    boundaries_.emplace(std::move(name),
                        InstalledBoundary{std::move(identity), required_depth,
                                          std::move(state_identity), std::move(authority)});
  }

  void install_boundary(std::string name, std::string identity, int required_depth,
                        const std::vector<std::string>& face_types,
                        const std::vector<double>& face_values,
                        const std::vector<std::string>& face_identities,
                        const std::vector<std::string>& component_roles, std::string state_identity,
                        const std::vector<std::string>& face_representations = {},
                        const std::vector<std::string>& face_converter_identities = {},
                        const std::vector<std::vector<std::string>>& face_analytic_opcodes = {},
                        const std::vector<std::vector<double>>& face_analytic_literals = {},
                        const std::vector<std::string>& face_analytic_clocks = {}) {
    auto authority = std::make_shared<boundary_type>(prepare_hyperbolic_boundary<Dim>(
        face_types, face_values, face_identities, component_roles, false, face_representations,
        face_converter_identities, face_analytic_opcodes, face_analytic_literals,
        face_analytic_clocks));
    install_boundary(std::move(name), std::move(identity), required_depth,
                     std::move(state_identity), std::move(authority));
  }

  const std::string& state_route(const std::string& name) const {
    const auto found = state_routes_.find(name);
    if (found == state_routes_.end())
      throw std::runtime_error("System block has no exact state route");
    return found->second;
  }

  const std::string& field_storage_route(const std::string& field_identity) const {
    const auto found = field_storage_routes_.find(field_identity);
    if (found == field_storage_routes_.end())
      throw std::runtime_error("System field has no exact native storage route");
    return found->second;
  }

  const InstalledBoundary* find_boundary(const std::string& name) const noexcept {
    const auto found = boundaries_.find(name);
    return found == boundaries_.end() ? nullptr : &found->second;
  }

  InstalledBoundary& boundary(const std::string& name) {
    const auto found = boundaries_.find(name);
    if (found == boundaries_.end())
      throw std::runtime_error("System block has no installed hyperbolic boundary");
    return found->second;
  }

  /// Complete the only model-dependent preparation step by replacing the immutable authority.
  void convert_fixed_states(
      const std::string& name,
      const std::function<void(const double* input, double* output)>& primitive_to_conservative) {
    InstalledBoundary& installed = boundary(name);
    installed.authority = std::make_shared<boundary_type>(
        installed.authority->with_converted_fixed_states(primitive_to_conservative));
  }

  void discard_transaction() noexcept {
    boundaries_.clear();
    state_routes_.clear();
    field_storage_routes_.clear();
  }

  bool empty() const noexcept {
    return boundaries_.empty() && state_routes_.empty() && field_storage_routes_.empty();
  }

 private:
  std::map<std::string, InstalledBoundary> boundaries_;
  std::map<std::string, std::string> state_routes_;
  std::map<std::string, std::string> field_storage_routes_;
};

}  // namespace pops::runtime::system
