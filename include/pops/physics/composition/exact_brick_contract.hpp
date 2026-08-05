#pragma once

#include <pops/core/identity/prepared_provider.hpp>

#include <concepts>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace pops::physics_contract_detail {

template <class Brick>
using ExactBrick = std::remove_cvref_t<Brick>;

/// Host-side semantic protocol required of every brick entering an authenticated generated model.
template <class Brick>
concept ExactPhysicsBrickContract =
    requires(const ExactBrick<Brick>& brick, ExactContractBuilder& contract) {
      {
        ExactBrick<Brick>::provider_identity()
      } noexcept -> std::same_as<PreparedProviderIdentity>;
      { brick.serialize_exact_parameters(contract) } -> std::same_as<void>;
    };

/// Append one recursively framed brick contract without inspecting its C++ representation.
template <ExactPhysicsBrickContract Brick>
inline void append_exact_brick(ExactContractBuilder& contract, std::string_view role,
                               const Brick& brick) {
  const PreparedProviderIdentity identity = ExactBrick<Brick>::provider_identity();
  if (identity.name.empty() || identity.version == 0)
    throw std::invalid_argument("physics brick provider identity is incomplete");

  ExactContractBuilder parameters;
  brick.serialize_exact_parameters(parameters);
  contract.text(role).text(identity.name).scalar(identity.version).bytes(parameters.view());
}

}  // namespace pops::physics_contract_detail
