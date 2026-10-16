#pragma once

#include <pops/runtime/program/continuation_image.hpp>

namespace pops::runtime::program {

// Per-level live issuance state, intentionally excluded from accepted snapshots.
// Only a committed image owner may supply retained_domains / retained_distributed.
// Temporary invocation exit never rewinds either issuer's ordinal.
template <int Dim>
struct AmrFiniteValueLifetime {
  FiniteDomainRegistry finite;
  FiniteDomainRegistry local_finite{false};
  ContinuationTransportSession<Dim> transport;
  std::shared_ptr<distributed::Registry> distributed;
  FiniteDomainRealmRef retained_realm;
  std::vector<FiniteDomainRef> retained_domains;
  std::vector<FiniteDomainRef> retained_local_domains;
  std::shared_ptr<distributed::Registry> retained_distributed;
  std::vector<std::shared_ptr<distributed::Registry>> protected_distributed;
  std::uint64_t invocation = 0;
  bool active = false;

  void begin() {
    if (active)
      throw std::logic_error("AMR finite-value issuer already has an active invocation");
    if (invocation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR finite-value invocation identity exhausted");
    ++invocation;
    active = true;
  }
  void finish() noexcept {
    finite.revoke_unretained(retained_domains);
    if (!retained_realm && retained_domains.empty())
      finite.revoke_on_restart();
    local_finite.revoke_unretained(retained_local_domains);
    if (retained_local_domains.empty())
      local_finite.revoke_on_restart();
    transport.clear();
    if (distributed != retained_distributed && distributed &&
        std::find(protected_distributed.begin(), protected_distributed.end(), distributed) ==
            protected_distributed.end())
      distributed->revoke();
    distributed = retained_distributed;
    active = false;
  }
  ~AmrFiniteValueLifetime() noexcept { finish(); }
};

}  // namespace pops::runtime::program
