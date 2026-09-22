#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>

namespace pops {

/// Neutral source. The auxiliary rank is deduced from the exact pointwise carrier.
struct NoSource {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.source.none", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder&) const {}

  template <class State, class Providers>
  POPS_HD State apply(const State&, const Providers&) const {
    return State{};
  }
};

}  // namespace pops
