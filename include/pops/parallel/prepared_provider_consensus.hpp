#pragma once

/// @file
/// @brief Exact communicator-wide consensus for an optional prepared provider.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <concepts>
#include <stdexcept>
#include <string_view>
#include <utility>

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
void require_prepared_provider_collective_consensus(const Provider& provider,
                                                    const ExecutionLane& lane) {
  ExactContractBuilder contract;
  long contract_failure_local = 0;
  try {
    contract.optional_collective_contract(provider);
  } catch (...) {
    contract_failure_local = 1;
  }
  if (all_reduce_max(contract_failure_local, lane) != 0)
    throw std::runtime_error(
        "prepared provider contract materialization failed on at least one communicator rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"pops.prepared-provider", contract.view()}},
                                                lane))
    throw std::invalid_argument("prepared provider contract differs between communicator ranks");
}

template <class Provider>
  requires requires(const Provider& provider) {
    { static_cast<bool>(provider) } -> std::same_as<bool>;
    { provider.collective_contract() } -> std::convertible_to<std::string_view>;
  }
void require_prepared_provider_collective_consensus(const Provider& provider) {
  const ExecutionLane lane = ExecutionLane::world();
  require_prepared_provider_collective_consensus(provider, lane);
}

}  // namespace pops
