#pragma once

/// @file
/// @brief Exact communicator-wide consensus for an optional prepared provider.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/comm.hpp>

#include <concepts>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// Requires every rank to hold the same optional prepared-provider contract.
///
/// The concrete provider type and callable signature are deliberately unconstrained. Any immutable
/// provider exposing optional presence and a stable collective contract can use this boundary,
/// independently of its physical meaning, dimensionality, storage, or native consumer. Presence,
/// implementation identity, version, and all exact parameters are already framed by the provider;
/// ExactContractBuilder adds the optional-presence frame before canonical bytes are compared. The
/// collective result is uniform, so a mismatch is rejected before any rank invokes its callable.
template <class Provider>
  requires requires(const Provider& provider) {
    { static_cast<bool>(provider) } -> std::same_as<bool>;
    { provider.collective_contract() } -> std::convertible_to<std::string_view>;
  }
void require_prepared_provider_collective_consensus(const Provider& provider) {
  ExactContractBuilder contract;
  contract.optional_collective_contract(provider);
  const std::vector<std::pair<std::string_view, std::string_view>> identities{
      {"pops.prepared-provider", contract.view()}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(identities))
    throw std::invalid_argument("prepared provider contract differs between communicator ranks");
}

}  // namespace pops
