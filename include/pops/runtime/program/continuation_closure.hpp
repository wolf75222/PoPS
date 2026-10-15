#pragma once
#include <set>
#include <pops/runtime/program/continuation_image.hpp>

namespace pops::runtime::program {
// Collective graph closure before rank-local codecs. This retains every rank's shard of
// every reachable immutable Array, including a reference occurring on only one rank.
template <int Dim>
std::shared_ptr<const ContinuationClosure<Dim>> continuation_collective_closure(
    const ContinuationValueRef<Dim>& root, distributed::Registry* registry,
    const ExecutionLane& lane, std::uint64_t capacity, FiniteDomainRealmRef replicated_realm = {},
    std::span<const std::shared_ptr<distributed::Registry>> authorized_registries = {},
    bool registry_local_gc = false) {
  using Ref = ContinuationValueRef<Dim>;
  using A = ContinuationDistributedArrayRef<Dim>;
  using Key = std::array<std::uint64_t, 3>;
  auto key = [](distributed::Identity id) { return Key{id.issuer[0], id.issuer[1], id.ordinal}; };
  auto identity = [](const Key& k) { return distributed::Identity{{k[0], k[1]}, k[2]}; };
  std::map<Key, distributed::DomainRef> domains;
  std::map<Key, A> arrays;
  using Issuer = std::array<std::uint64_t, 2>;
  std::map<Issuer, distributed::Registry*> authorities;
  std::map<Issuer, std::shared_ptr<distributed::Registry>> owners;
  distributed::phase(lane, [&] {
    const auto add = [&](distributed::Registry* authority) {
      if (!authority || &authority->lane() != &lane)
        throw std::logic_error("continuation registry has a different execution lane");
      auto [found, inserted] = authorities.emplace(authority->issuer(), authority);
      if (!inserted && found->second != authority)
        throw std::logic_error("continuation has conflicting distributed issuer authorities");
    };
    if (registry)
      add(registry);
    for (const auto& authority : authorized_registries) {
      add(authority.get());
      owners.emplace(authority->issuer(), authority);
    }
  });
  const auto resolve = [&](distributed::Identity id) -> distributed::Registry& {
    const auto found = authorities.find(id.issuer);
    if (found == authorities.end())
      throw std::logic_error("distributed continuation has no authorized issuer registry");
    return *found->second;
  };
  auto collect = [&](const Ref& value) {
    std::unordered_set<const ContinuationDistributedArray<Dim>*> active_arrays;
    continuation_detail::walk_value<Dim>(
        value, capacity,
        [&](const ContinuationValue<Dim>& node) {
          // This proof comes from immutable Value constructors, never from an
          // Array's declared element type. A malformed native Array mutation
          // therefore still exposes its actual distributed edges and cycles.
          if (node.is_scalar_structure())
            return false;
          if (const auto& d = node.distributed_domain_ref()) {
            // A registry-local GC census treats other issuers as external
            // ownership edges. Full checkpoint/transport closure never uses this mode.
            if (registry_local_gc && registry && d->identity().issuer != registry->issuer())
              return false;
            resolve(d->identity()).authenticate(d);
            domains.emplace(key(d->identity()), d);
          }
          if (const auto& a = node.distributed_array_ref()) {
            if (active_arrays.contains(a.get()))
              throw std::invalid_argument("cyclic continuation Array dependency");
            auto published =
                resolve(a->version).resolve_value(a->version, distributed::array_type_tag<Dim>());
            if (published.get() != a.get())
              throw std::logic_error("continuation contains an unpublished Array version");
            if (a->index_domain && (!registry_local_gc || !registry ||
                                    a->index_domain->identity().issuer == registry->issuer())) {
              resolve(a->index_domain->identity()).authenticate(a->index_domain);
              domains.emplace(key(a->index_domain->identity()), a->index_domain);
            }
            if (!arrays.emplace(key(a->version), a).second)
              return false;
            active_arrays.insert(a.get());
          }
          return true;
        },
        [&](const ContinuationValue<Dim>& node) {
          if (const auto& array = node.distributed_array_ref())
            active_arrays.erase(array.get());
        });
  };
  distributed::phase(lane, [&] { collect(root); });
  if (all_reduce_max(authorities.empty() ? 0L : 1L, lane)) {
    bool complete = false;
    for (std::uint64_t round = 0; round <= capacity / 24; ++round) {
      const auto old_arrays = arrays.size(), old_domains = domains.size();
      std::vector<std::uint8_t> local;
      distributed::phase(lane, [&] {
        continuation_detail::Writer out(capacity);
        out.word(domains.size());
        out.word(arrays.size());
        for (const auto& [id, d] : domains)
          for (auto word : id)
            out.word(word);
        for (const auto& [id, a] : arrays)
          for (auto word : id)
            out.word(word);
        local = std::move(out.bytes);
      });
      for (int owner = 0; owner < lane.size(); ++owner) {
        std::uint64_t length = lane.rank() == owner ? local.size() : 0;
#ifdef POPS_HAS_MPI
        distributed::mpi_status(MPI_Bcast(&length, 1, MPI_UINT64_T, owner, lane.native_handle()));
#endif
        std::vector<std::uint8_t> received;
        distributed::phase(lane, [&] {
          if (length > capacity)
            throw std::length_error("continuation closure metadata exceeds capacity");
          const auto count = distributed::byte_size(length);
          if (lane.rank() == owner)
            received = local;
          else
            received.resize(count);
        });
        distributed::broadcast(received, owner, lane);
        distributed::phase(lane, [&] {
          continuation_detail::Reader in(received);
          const auto nd = in.word(), na = in.word();
          if (nd > (received.size() - in.at) / 24 || na > (received.size() - in.at - nd * 24) / 24)
            throw std::invalid_argument("continuation closure metadata is truncated");
          for (std::uint64_t i = 0; i < nd; ++i) {
            const Key id{in.word(), in.word(), in.word()};
            domains.emplace(id, resolve(identity(id)).resolve_domain(identity(id)));
          }
          for (std::uint64_t i = 0; i < na; ++i) {
            const Key id{in.word(), in.word(), in.word()};
            auto a = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
                resolve(identity(id))
                    .resolve_value(identity(id), distributed::array_type_tag<Dim>()));
            if (!a || key(a->version) != id)
              throw std::logic_error("continuation closure version differs");
            collect(ContinuationValue<Dim>::distributed_array(a));
          }
          if (in.at != received.size())
            throw std::invalid_argument("continuation closure metadata has trailing bytes");
        });
      }
      const long changed = arrays.size() != old_arrays || domains.size() != old_domains;
      if (all_reduce_max(changed, lane) == 0) {
        complete = true;
        break;
      }
    }
    if (!complete)
      throw std::length_error("continuation closure did not fit its version budget");
  }
  std::shared_ptr<ContinuationClosure<Dim>> result;
  std::string descriptor;
  distributed::phase(lane, [&] {
    result = std::make_shared<ContinuationClosure<Dim>>();
    continuation_detail::Writer out(capacity);
    if (replicated_realm && replicated_realm->synchronized())
      result->replicated_realm = replicated_realm;
    out.word(bool(result->replicated_realm));
    if (result->replicated_realm)
      for (auto word : result->replicated_realm->transport_identity())
        out.word(word);
    std::set<Issuer> used_issuers;
    for (const auto& [id, d] : domains)
      used_issuers.insert(d->identity().issuer);
    for (const auto& issuer : used_issuers) {
      const auto found = owners.find(issuer);
      if (found != owners.end())
        result->registries.push_back(found->second);
    }
    out.word(used_issuers.size());
    for (const auto& issuer : used_issuers) {
      const auto& registry = *authorities.at(issuer);
      out.word(issuer[0]);
      out.word(issuer[1]);
      out.word(registry.storage_bytes());
      out.word(registry.storage_versions());
    }
    out.word(domains.size());
    for (const auto& [id, d] : domains) {
      for (auto w : id)
        out.word(w);
      out.word(d->family());
      out.word(d->capacity());
      out.word(d->partition_offsets().size());
      for (auto x : d->partition_offsets())
        out.word(x);
      result->domains.push_back(d);
    }
    out.word(arrays.size());
    for (const auto& [id, a] : arrays) {
      for (auto w : id)
        out.word(w);
      for (auto w : key(a->domain->identity()))
        out.word(w);
      out.text(continuation_transport_type_identity(a->element, true, capacity));
      out.word(bool(a->index_domain));
      if (a->index_domain)
        for (auto w : key(a->index_domain->identity()))
          out.word(w);
      result->arrays.push_back(a);
    }
    descriptor.assign(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
  });
  const std::array<ExactOrderedBytePair, 1> contract{{{"continuation-closure", descriptor}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(contract, lane))
    throw std::invalid_argument(
        "continuation closure identities/types/ownership differ across ranks");
  return result;
}
template <int Dim>
std::shared_ptr<const ContinuationClosure<Dim>> continuation_collective_closure(
    const ContinuationValueRef<Dim>& root, const std::shared_ptr<distributed::Registry>& primary,
    const ExecutionLane& lane, std::uint64_t capacity, FiniteDomainRealmRef replicated_realm = {},
    std::span<const std::shared_ptr<distributed::Registry>> authorized_registries = {}) {
  std::vector<std::shared_ptr<distributed::Registry>> authorities;
  distributed::phase(lane, [&] {
    std::size_t unique_count = primary ? 1 : 0;
    for (std::size_t i = 0; i < authorized_registries.size(); ++i) {
      const auto& authority = authorized_registries[i];
      if (!authority)
        throw std::invalid_argument("continuation registry catalog has an absent authority");
      if (authority != primary &&
          std::find(authorized_registries.begin(), authorized_registries.begin() + i, authority) ==
              authorized_registries.begin() + i) {
        if (unique_count == std::numeric_limits<std::size_t>::max())
          throw std::length_error("continuation registry catalog count overflow");
        ++unique_count;
      }
    }
    if (unique_count > capacity / sizeof(std::shared_ptr<distributed::Registry>))
      throw std::length_error("continuation registry catalog exceeds capacity");
    authorities.reserve(unique_count);
    if (primary)
      authorities.push_back(primary);
    for (const auto& authority : authorized_registries)
      if (std::find(authorities.begin(), authorities.end(), authority) == authorities.end())
        authorities.push_back(authority);
  });
  return continuation_collective_closure<Dim>(root, primary.get(), lane, capacity,
                                              std::move(replicated_realm), authorities);
}
}  // namespace pops::runtime::program
