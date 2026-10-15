// Included after ContinuationTransportSession is complete. Interning never
// accepts an identity match alone: every descriptor and immutable body matches.
template <int Dim>
std::shared_ptr<distributed::Registry> ContinuationCodec<Dim>::DecodeContext::registry(
    const ExecutionLane& lane, std::array<std::uint64_t, 2> issuer, std::uint64_t bytes,
    std::uint64_t versions) {
  const auto found = registries_.find(issuer);
  if (found != registries_.end()) {
    const auto& held = found->second;
    if (&held->lane() != &lane || held->storage_bytes() != bytes ||
        held->storage_versions() != versions)
      throw std::invalid_argument(
          "continuation images disagree on their shared registry authority");
    return held;
  }
  charge_(sizeof(typename decltype(registries_)::value_type));
  auto held = distributed::Registry::restore_local(lane, issuer, bytes, versions);
  registries_.emplace(issuer, held);
  return held;
}

template <int Dim>
distributed::DomainRef ContinuationCodec<Dim>::DecodeContext::domain(
    distributed::Registry& registry, distributed::Identity id, std::int64_t family,
    std::int64_t capacity, std::span<const std::int64_t> offsets) {
  const auto owner = registries_.find(id.issuer);
  if (owner == registries_.end() || owner->second.get() != &registry)
    throw std::invalid_argument("continuation domain has no shared decoded registry");
  const auto key = key_(id);
  const auto found = domains_.find(key);
  if (found != domains_.end()) {
    const auto held = registry.resolve(id, family, capacity, offsets);
    if (held != found->second)
      throw std::invalid_argument("continuation domain changes its exact decoded authority");
    return held;
  }
  charge_(sizeof(typename decltype(domains_)::value_type));
  auto held = registry.restore_domain_local(id, family, capacity, offsets);
  domains_.emplace(key, held);
  return held;
}

template <int Dim>
FiniteDomainRealmRef ContinuationCodec<Dim>::DecodeContext::intern_realm(
    FiniteDomainRealmRef candidate) {
  if (!candidate || !candidate->visible() || !candidate->synchronized())
    throw std::invalid_argument("continuation image has no exact exported finite authority");
  const auto origin = candidate->transport_identity();
  const auto found = realms_.find(origin);
  if (found != realms_.end()) {
    if (!found->second->visible() || found->second->replicated() != candidate->replicated())
      throw std::invalid_argument("continuation images disagree on finite ownership");
    return found->second;
  }
  charge_(sizeof(typename decltype(realms_)::value_type));
  realms_.emplace(origin, candidate);
  return candidate;
}

template <int Dim>
FiniteDomainRef ContinuationCodec<Dim>::DecodeContext::intern_domain(FiniteDomainRef candidate) {
  if (!candidate || !candidate->visible() || candidate->version <= 0)
    throw std::invalid_argument("continuation image has no live finite domain authority");
  const auto origin = candidate->realm()->transport_identity();
  const auto realm = realms_.find(origin);
  if (realm == realms_.end() || realm->second != candidate->realm())
    throw std::invalid_argument("continuation finite domain does not use its interned realm");
  const Key key{origin[0], origin[1], static_cast<std::uint64_t>(candidate->version)};
  const auto found = finite_.find(key);
  if (found != finite_.end()) {
    const auto& held = found->second;
    if (!held->visible() || held->realm() != candidate->realm() || held->size != candidate->size ||
        held->capacity != candidate->capacity || held->family != candidate->family)
      throw std::invalid_argument("continuation images disagree on a finite domain version");
    return held;
  }
  charge_(sizeof(typename decltype(finite_)::value_type));
  finite_.emplace(key, candidate);
  return candidate;
}

template <int Dim>
ContinuationDistributedArrayRef<Dim> ContinuationCodec<Dim>::DecodeContext::array(
    distributed::Registry& registry, ContinuationDistributedArrayRef<Dim> candidate,
    std::uint64_t charge, std::uint64_t capacity,
    std::span<const std::shared_ptr<distributed::Registry>> authorized) {
  if (!candidate || !candidate->domain)
    throw std::invalid_argument("continuation image contains an absent Array");
  const auto owner = registries_.find(candidate->version.issuer);
  if (owner == registries_.end() || owner->second.get() != &registry)
    throw std::invalid_argument("continuation Array has no shared decoded registry");
  registry.authenticate(candidate->domain);
  const auto key = key_(candidate->version);
  const auto found = arrays_.find(key);
  if (found == arrays_.end()) {
    charge_(sizeof(typename decltype(arrays_)::value_type));
    const auto [entry, inserted] = arrays_.emplace(key, ArrayEntry{candidate, charge});
    (void)inserted;
    try {
      registry.restore_value_local(candidate->version, candidate,
                                   distributed::array_type_tag<Dim>(), charge);
    } catch (...) {
      arrays_.erase(entry);
      throw;
    }
    return candidate;
  }
  const auto& held = found->second.value;
  if (found->second.charge != charge || held->domain != candidate->domain ||
      held->index_domain != candidate->index_domain || held->element != candidate->element ||
      held->values.size() != candidate->values.size() ||
      registry.resolve_value(candidate->version, distributed::array_type_tag<Dim>()).get() !=
          held.get())
    throw std::invalid_argument("continuation images disagree on an immutable Array version");

  ContinuationTransportSession<Dim> session;
  const auto bind = [&](distributed::Registry& source) {
    const auto known = registries_.find(source.issuer());
    if (known == registries_.end() || known->second.get() != &source)
      throw std::invalid_argument("continuation Array body has a foreign decoded issuer");
    session.bind_distributed_registry(
        source, [pointer = &source](distributed::Identity version, distributed::DomainRef domain,
                                    const ContinuationType& element) {
          pointer->authenticate(domain);
          auto value = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
              pointer->resolve_value(version, distributed::array_type_tag<Dim>()));
          if (!value || value->domain != domain || value->element != element ||
              value->version.issuer != version.issuer || value->version.ordinal != version.ordinal)
            throw std::invalid_argument("continuation Array body changed a dependency version");
          return value;
        });
  };
  bind(registry);
  for (const auto& source : authorized) {
    if (!source)
      throw std::invalid_argument("continuation Array body has an absent decoded issuer");
    bind(*source);
  }
  for (std::size_t i = 0; i < held->values.size(); ++i)
    if (session.encode(held->element, held->values[i], capacity) !=
        session.encode(candidate->element, candidate->values[i], capacity))
      throw std::invalid_argument("continuation images change the body of one Array version");
  return held;
}
