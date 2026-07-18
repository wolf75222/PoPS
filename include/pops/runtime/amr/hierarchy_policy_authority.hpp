#pragma once

/// @file
/// @brief Opaque, versioned hierarchy-policy authority carried by prepared AMR field plans.

#include <pops/core/identity/prepared_provider_options.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops {

/// Exact authority selected by Python field authoring and interpreted only by the chosen native
/// field-solver provider. The AMR core authenticates these bytes but assigns no meaning to a policy
/// identity or to provider-owned options.
struct AmrFieldHierarchyPolicyAuthority {
  std::string policy_id;
  std::uint64_t interface_version = 0;
  PreparedProviderOptions options;

  void validate() const {
    if (policy_id.empty() || interface_version == 0)
      throw std::invalid_argument(
          "AMR field hierarchy policy requires an exact identity and interface version");
    (void)options.exact_contract();
  }

  [[nodiscard]] std::string exact_contract() const {
    validate();
    ExactContractBuilder contract;
    contract.text("pops.amr.field-hierarchy-policy-authority")
        .scalar(std::uint32_t{1})
        .text(policy_id)
        .scalar(interface_version)
        .bytes(options.exact_contract());
    return std::move(contract).release();
  }
};

}  // namespace pops
