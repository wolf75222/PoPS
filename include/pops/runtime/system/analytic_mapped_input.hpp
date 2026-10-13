#pragma once
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace pops::runtime::system {

enum class AnalyticMappedInputKind : std::uint8_t {
  state_component = 0,
  provider_component = 1,
  provider_derivative = 2
};

/// One exact input to a bind-time mapped analytic state expression.
///
/// State components are local to the target state carrier. Provider components are addressed only
/// by their owner-qualified key and are resolved through a sealed consumer plan. No physical name
/// or global storage component crosses this interface.
struct AnalyticMappedInput {
  AnalyticMappedInputKind kind = AnalyticMappedInputKind::state_component;
  int state_component = -1;
  AuxiliaryComponentKey provider_key;
  int derivative_axis = -1;
  int derivative_edge_order = 0;

  static AnalyticMappedInput state(int component) {
    return {AnalyticMappedInputKind::state_component, component, {}};
  }
  static AnalyticMappedInput provider(AuxiliaryComponentKey key) {
    return {AnalyticMappedInputKind::provider_component, -1, std::move(key)};
  }
  static AnalyticMappedInput derivative(AuxiliaryComponentKey key, int axis, int edge_order) {
    return {AnalyticMappedInputKind::provider_derivative, -1, std::move(key), axis, edge_order};
  }

  void validate() const {
    if (kind == AnalyticMappedInputKind::state_component) {
      if (state_component < 0)
        throw std::invalid_argument("mapped analytic state component must be non-negative");
      return;
    }
    if (kind != AnalyticMappedInputKind::provider_component &&
        kind != AnalyticMappedInputKind::provider_derivative)
      throw std::invalid_argument("mapped analytic input kind is invalid");
    provider_key.validate();
    if (kind == AnalyticMappedInputKind::provider_derivative &&
        (derivative_axis < 0 || derivative_axis > 2 || derivative_edge_order != 2))
      throw std::invalid_argument(
          "mapped analytic derivative requires a ranked axis and edge_order=2");
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    validate();
    contract.scalar(static_cast<std::uint8_t>(kind));
    if (kind == AnalyticMappedInputKind::state_component)
      contract.scalar(state_component);
    else
      provider_key.serialize_exact(contract);
    if (kind == AnalyticMappedInputKind::provider_derivative)
      contract.scalar(derivative_axis).scalar(derivative_edge_order);
  }
};

}  // namespace pops::runtime::system
