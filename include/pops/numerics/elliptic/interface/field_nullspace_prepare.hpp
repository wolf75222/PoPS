#pragma once

/// @file
/// @brief Communicator-safe preparation of a provider-owned field nullspace.

#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/parallel/comm.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// Resolve, authenticate and materialize one field-nullspace provider collectively.
///
/// The algorithm is shared by uniform and AMR runtimes. It deliberately knows neither a
/// nullspace family nor a solver backend: the request carries operator/topology facts, the
/// selection carries opaque provider options, and the provider owns all mathematical decisions.
/// Every potentially rank-local failure is converted to a uniform failure before another provider
/// callback can enter a scientific collective.
inline PreparedFieldNullspace prepare_field_nullspace_collectively(
    const FieldNullspaceProviderRegistry& registry,
    const FieldNullspaceProviderSelection& selection, FieldNullspaceProviderRequest request) {
  request.options = selection.options;
  std::string operator_facts_contract;
  bool operator_facts_failed = false;
  try {
    operator_facts_contract = request.operator_facts.exact_contract();
  } catch (...) {
    operator_facts_failed = true;
  }
  if (all_reduce_max(operator_facts_failed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "field-nullspace operator facts are malformed on at least one rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"field-nullspace-operator-facts", operator_facts_contract}}))
    throw std::runtime_error("field-nullspace operator facts differ across MPI ranks");

  std::shared_ptr<const FieldNullspaceProvider> provider;
  std::string provider_contract;
  PreparedProviderSupport support = PreparedProviderSupport::reject(
      1, "field-nullspace provider resolution failed");
  std::string support_contract;
  std::string expected_contract;
  bool declaration_failed = false;
  try {
    provider = registry.resolve(selection.provider_identity);
    provider_contract = provider->collective_contract();
    support = provider->supports(request);
    support_contract = exact_prepared_provider_support(support);
    declaration_failed = !support.well_formed();
    if (!declaration_failed && support.accepted())
      expected_contract = provider->expected_prepared_contract(request);
  } catch (...) {
    declaration_failed = true;
  }
  if (all_reduce_max(declaration_failed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "field-nullspace provider declaration failed on at least one rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"field-nullspace-provider", provider_contract},
           {"field-nullspace-support", support_contract},
           {"field-nullspace-expected-contract", expected_contract}}))
    throw std::runtime_error(
        "field-nullspace provider declaration differs across MPI ranks");
  if (!support.accepted())
    throw std::runtime_error("field-nullspace provider rejected the exact prepared request: " +
                             std::string(support.reason) + " (provider status " +
                             std::to_string(support.code) + ")");
  if (expected_contract.empty())
    throw std::runtime_error(
        "field-nullspace provider accepted the request without an expected contract");

  PreparedFieldNullspace prepared;
  bool preparation_failed = false;
  try {
    prepared = provider->prepare(request);
  } catch (...) {
    preparation_failed = true;
  }
  if (all_reduce_max(preparation_failed ? 1L : 0L) != 0)
    throw std::runtime_error(
        "field-nullspace provider preparation failed on at least one rank");
  const bool mismatch = prepared.provider_identity != provider->identity() ||
                        prepared.provider_version != provider->interface_version() ||
                        prepared.exact_prepared_contract != expected_contract;
  if (all_reduce_max(mismatch ? 1L : 0L) != 0)
    throw std::runtime_error(
        "field-nullspace provider published an invalid prepared contract");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"field-nullspace-actual-contract", prepared.exact_prepared_contract}}))
    throw std::runtime_error("prepared field nullspace differs across MPI ranks");
  return prepared;
}

}  // namespace pops
