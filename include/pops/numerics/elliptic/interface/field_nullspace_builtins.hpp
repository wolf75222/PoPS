#pragma once

/// @file
/// @brief Builtin nullspace providers installed through the public provider protocol.

#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pops {
namespace detail {

inline bool field_nullspace_request_shape_is_valid(
    const FieldNullspaceProviderRequest& request) noexcept {
  const auto& topology = request.topology;
  if (request.plan_identity.empty() || topology.identity.empty() ||
      topology.exact_layout_contract.empty() || topology.first_level < 0 ||
      topology.field_component < 0 || topology.layouts.empty() ||
      topology.cell_measure.size() != topology.layouts.size())
    return false;
  if ((!topology.coverage.empty() && topology.coverage.size() != topology.layouts.size()) ||
      topology.coverage_contracts.size() != topology.coverage.size())
    return false;
  if (topology.component_label_contracts.size() != topology.component_labels.size() ||
      std::any_of(topology.component_label_contracts.begin(),
                  topology.component_label_contracts.end(),
                  [](const std::string& contract) { return contract.empty(); }) ||
      std::any_of(topology.coverage_contracts.begin(), topology.coverage_contracts.end(),
                  [](const std::string& contract) { return contract.empty(); }))
    return false;
  if (topology.level_distributions.size() != topology.layouts.size())
    return false;
  if (std::any_of(topology.layouts.begin(), topology.layouts.end(),
                  [](const MultiFab* layout) { return layout == nullptr; }) ||
      std::any_of(topology.cell_measure.begin(), topology.cell_measure.end(),
                  [](Real measure) { return !(measure > Real(0)); }))
    return false;
  return true;
}

inline bool field_operator_preserves_constant_mode(
    const FieldNullspaceOperatorFacts& facts) noexcept {
  return !facts.has_reaction && !facts.has_coercive_internal_constraint &&
         std::all_of(facts.boundaries.begin(), facts.boundaries.end(), [](const auto& boundary) {
           return boundary.behavior == FieldBoundaryNullspaceBehavior::PreservesConstantMode;
         });
}

inline bool field_operator_has_opaque_nullspace_facts(
    const FieldNullspaceOperatorFacts& facts) noexcept {
  return std::any_of(facts.boundaries.begin(), facts.boundaries.end(), [](const auto& boundary) {
    return boundary.behavior == FieldBoundaryNullspaceBehavior::Opaque;
  });
}

inline Real operator_topology_gauge_value(const PreparedProviderOptions& options) {
  const auto found = options.values.find("gauge.value");
  if (found == options.values.end() || !std::holds_alternative<double>(found->second))
    throw std::invalid_argument(
        "operator-topology nullspace provider requires a binary64 gauge.value");
  const double value = std::get<double>(found->second);
  if (!std::isfinite(value) ||
      !std::isfinite(static_cast<double>(static_cast<Real>(value))))
    throw std::invalid_argument(
        "operator-topology nullspace provider requires a finite gauge.value");
  return static_cast<Real>(value);
}

inline std::string operator_topology_nullspace_contract(
    const FieldNullspaceProviderRequest& request) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-field-nullspace")
      .scalar(std::uint32_t{2})
      .text("pops.field-nullspace.operator-topology-derived")
      .scalar(std::uint64_t{2})
      .text(request.plan_identity)
      .bytes(request.operator_facts.exact_contract())
      .text(request.topology.identity)
      .bytes(request.topology.exact_layout_contract)
      .scalar(static_cast<std::int64_t>(request.topology.first_level))
      .scalar(static_cast<std::int64_t>(request.topology.field_component))
      .bytes(request.topology.connected_component_contract)
      .scalar(static_cast<std::uint64_t>(request.topology.layouts.size()))
      .sequence(request.topology.cell_measure)
      .sequence(request.topology.level_distributions,
                [](ExactContractBuilder& item,
                   const PreparedVectorDistribution& distribution) {
                  item.bytes(distribution.collective_contract());
                })
      .sequence(request.topology.connected_components,
                [](ExactContractBuilder& item, const FieldConnectedComponent& component) {
                  item.scalar(static_cast<std::int64_t>(component.label))
                      .text(component.identity)
                      .text(component.provenance);
                })
      .sequence(request.topology.component_label_contracts,
                [](ExactContractBuilder& item, const std::string& resource) {
                  item.bytes(resource);
                })
      .sequence(request.topology.coverage_contracts,
                [](ExactContractBuilder& item, const std::string& resource) {
                  item.bytes(resource);
                })
      .bytes(request.options.exact_contract());
  return std::move(contract).release();
}

inline FieldNullspacePlan connected_operator_topology_plan(
    const FieldNullspaceProviderRequest& request, Real gauge_value) {
  const auto& topology = request.topology;
  FieldNullspacePlan result;
  result.identity = request.plan_identity + ":nullspace";
  result.layout_identity = topology.identity;
  FieldNullspaceBasis basis;
  basis.identity = result.identity + ":connected-component:0";
  basis.provenance = "provider:pops.field-nullspace.operator-topology-derived@2";
  basis.recipe_identity = topology.identity + ":constant-component@1";
  basis.field_component = topology.field_component;
  basis.coverage = topology.coverage;
  basis.cell_measure.assign(static_cast<std::size_t>(topology.first_level), Real(0));
  basis.cell_measure.insert(basis.cell_measure.end(), topology.cell_measure.begin(),
                            topology.cell_measure.end());
  result.gauges.push_back(FieldGaugeConstraint{basis.identity, gauge_value});
  result.bases.push_back(std::move(basis));
  validate_field_nullspace_basis(topology.layouts, result, topology.level_distributions,
                                 topology.first_level);
  return result;
}

class OperatorTopologyFieldNullspaceProvider final : public FieldNullspaceProvider {
 public:
  [[nodiscard]] std::string_view identity() const noexcept override {
    return "pops.field-nullspace.operator-topology-derived";
  }
  [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 2; }
  [[nodiscard]] std::string_view collective_contract() const noexcept override {
    return "pops.field-nullspace.operator-topology-derived@2";
  }
  [[nodiscard]] PreparedProviderOptions default_options() const override {
    return {"pops.field-nullspace.operator-topology-derived.options@1",
            {{"gauge.value", 0.0}}};
  }
  [[nodiscard]] bool accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    try {
      return options.schema_identity ==
                 "pops.field-nullspace.operator-topology-derived.options@1" &&
             options.values.size() == 1 &&
             std::isfinite(static_cast<double>(operator_topology_gauge_value(options)));
    } catch (...) {
      return false;
    }
  }
  [[nodiscard]] PreparedProviderSupport supports(
      const FieldNullspaceProviderRequest& request) const noexcept override {
    if (!accepts_options(request.options))
      return PreparedProviderSupport::reject(1, "invalid operator-topology nullspace options");
    if (!request.operator_facts.is_canonical())
      return PreparedProviderSupport::reject(2, "malformed canonical operator facts");
    if (!field_nullspace_request_shape_is_valid(request))
      return PreparedProviderSupport::reject(3, "malformed prepared field topology");
    if (field_operator_has_opaque_nullspace_facts(request.operator_facts))
      return PreparedProviderSupport::reject(4, "operator nullspace facts are opaque");
    if (!field_operator_preserves_constant_mode(request.operator_facts))
      return PreparedProviderSupport::accept();
    const bool connected = !request.topology.connected_component_contract.empty() &&
                           request.topology.component_labels.empty() &&
                           request.topology.component_label_contracts.empty() &&
                           request.topology.connected_components.empty();
    const bool labelled =
        request.topology.component_labels.size() == request.topology.layouts.size() &&
        request.topology.component_label_contracts.size() ==
            request.topology.component_labels.size() &&
        !request.topology.connected_components.empty();
    if (!connected && !labelled)
      return PreparedProviderSupport::reject(
          5, "constant mode requires connectedness proof or explicit component partition");
    return PreparedProviderSupport::accept();
  }
  [[nodiscard]] std::string expected_prepared_contract(
      const FieldNullspaceProviderRequest& request) const override {
    const PreparedProviderSupport support = supports(request);
    if (!support.accepted())
      throw std::invalid_argument(std::string(support.reason));
    return operator_topology_nullspace_contract(request);
  }
  [[nodiscard]] PreparedFieldNullspace prepare(
      const FieldNullspaceProviderRequest& request) const override {
    const std::string contract = expected_prepared_contract(request);
    const Real gauge_value = operator_topology_gauge_value(request.options);
    FieldNullspacePlan plan;
    if (field_operator_preserves_constant_mode(request.operator_facts)) {
      if (!request.topology.connected_component_contract.empty()) {
        plan = connected_operator_topology_plan(request, gauge_value);
      } else {
        plan = labelled_mean_zero_nullspace(
            request.plan_identity + ":nullspace", request.topology.identity,
            request.topology.component_labels,
            request.topology.connected_components, request.topology.coverage,
            request.topology.cell_measure, request.topology.field_component,
            request.topology.level_distributions,
            request.topology.first_level);
        for (FieldGaugeConstraint& gauge : plan.gauges)
          gauge.value = gauge_value;
      }
    }
    return {std::string(identity()), interface_version(), contract, std::move(plan)};
  }
};

}  // namespace detail

inline std::shared_ptr<FieldNullspaceProviderRegistry>
make_default_field_nullspace_provider_registry() {
  auto registry = std::make_shared<FieldNullspaceProviderRegistry>();
  registry->add(std::make_shared<detail::OperatorTopologyFieldNullspaceProvider>());
  return registry;
}

inline FieldNullspaceProviderSelection operator_topology_zero_mean_nullspace() {
  FieldNullspaceProviderSelection selection;
  selection.provider_identity = "pops.field-nullspace.operator-topology-derived";
  selection.options =
      {"pops.field-nullspace.operator-topology-derived.options@1", {{"gauge.value", 0.0}}};
  return selection;
}

}  // namespace pops
