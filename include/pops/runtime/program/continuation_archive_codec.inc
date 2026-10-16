// Included in ContinuationCodec's public section. An archive is inert metadata:
// distributed identities below possess no Registry, Domain or execution lane.
// Only the later collective migration plan may mint target native capabilities.
struct ArchiveRegistry {
  std::array<std::uint64_t, 2> issuer{};
  std::uint64_t bytes = 0, versions = 0;
  bool operator==(const ArchiveRegistry&) const = default;
};
struct ArchiveDomain {
  distributed::Identity identity{};
  std::int64_t family = 0, capacity = 0;
  std::vector<std::int64_t> offsets;
};
struct ArchiveNode {
  ContinuationKind kind{};
  Ref leaf;  // genuinely native scalar, finite capability, or Field; never distributed
  FiniteDomainRef finite;  // only finite Array containers
  std::vector<std::size_t> children;
  std::uint64_t arm = 0, slot = 0;
  std::int64_t gid = 0;
};
struct ArchiveArray {
  distributed::Identity identity{};
  std::size_t domain = 0;
  std::optional<std::size_t> index_domain;
  ContinuationType element;
  std::uint64_t charge = 0;
  std::vector<std::size_t> values;
};
struct ArchiveImage {
  std::uint64_t publication = 0;
  std::int64_t max_domain_version = 0;
  int source_rank = 0, source_ranks = 0;
  FiniteDomainRealmRef realm;
  std::vector<FiniteDomainRealmRef> realms;
  std::vector<FiniteDomainRef> finite;
  std::vector<ArchiveRegistry> registries;
  std::optional<std::size_t> primary_registry;
  std::vector<ArchiveDomain> domains;
  std::vector<ArchiveArray> arrays;
  std::vector<ArchiveNode> nodes;  // postorder: destruction/traversal has no recursive C++ shape
  std::size_t root = 0;
};

ArchiveImage decode_archive(std::span<const std::uint8_t> bytes, int source_rank, int source_ranks,
                            const std::function<void(std::uint64_t)>& reserve) const {
  if (!reserve || source_ranks < 1 || source_rank < 0 || source_rank >= source_ranks ||
      bytes.size() > capacity_)
    throw std::invalid_argument("continuation archive rank table or budget is invalid");
  const auto charge = [&](std::uint64_t count, std::uint64_t width) {
    if (width && count > std::numeric_limits<std::uint64_t>::max() / width)
      throw std::length_error("continuation archive workspace overflow");
    reserve(count * width);
  };
  continuation_detail::Reader in(bytes);
  const auto wire = in.text();
  if (wire == "POPSCONT1") {
    auto decoded = decode(bytes);
    ArchiveImage out;
    out.publication = decoded.publication;
    out.realm = decoded.realm;
    out.realms = decoded.realms;
    out.max_domain_version = decoded.max_domain_version;
    out.source_rank = source_rank;
    out.source_ranks = source_ranks;
    std::map<const Value*, std::size_t> ids;
    continuation_detail::walk_value<Dim>(
        decoded.root, capacity_, [](const Value&) { return true; },
        [&](const Value& value) {
          charge(1, sizeof(ArchiveNode) + sizeof(typename decltype(ids)::value_type));
          ArchiveNode node;
          node.kind = value.kind();
          if (value.kind() == ContinuationKind::Array)
            node.finite = value.domain_ref();
          if (value.kind() == ContinuationKind::Variant)
            node.arm = value.arm();
          charge(value.members().size(), sizeof(std::size_t));
          for (const auto& child : value.members())
            node.children.push_back(ids.at(child.get()));
          if (value.kind() != ContinuationKind::Array &&
              value.kind() != ContinuationKind::Variant &&
              value.kind() != ContinuationKind::Product) {
            // Keep the actual owning leaf from the decoded graph, without copying native fields.
            if (&value == decoded.root.get())
              node.leaf = decoded.root;
            else {
              // A leaf is found through its existing shared ownership while traversing parents below.
              // Use the decoder's table after this postorder pass to bind those references.
            }
          }
          ids[&value] = out.nodes.size();
          out.nodes.push_back(std::move(node));
        });
    continuation_detail::walk_value<Dim>(
        decoded.root, capacity_,
        [&](const Value& value) {
          for (const auto& child : value.members()) {
            auto& node = out.nodes.at(ids.at(child.get()));
            if (child->kind() != ContinuationKind::Array &&
                child->kind() != ContinuationKind::Variant &&
                child->kind() != ContinuationKind::Product)
              node.leaf = child;
          }
          return true;
        },
        [](const Value&) {});
    out.root = ids.at(decoded.root.get());
    return out;
  }
  if (wire != "POPSCONT2" && wire != "POPSCONT3" && wire != "POPSCONT4" && wire != "POPSCONT5" &&
      wire != "POPSCONT6")
    throw std::invalid_argument("inert distributed archive requires an authority-bearing codec");
  bool expressions_wire = wire == "POPSCONT4";
  const bool traits_wire = wire == "POPSCONT6";
  const bool support_wire = wire == "POPSCONT5" || traits_wire;
  const bool multi = wire != "POPSCONT2";
  if (in.text() != identity_ || in.word() != Dim)
    throw std::invalid_argument("continuation archive changes installed schema or dimension");
  ArchiveImage out;
  out.publication = in.word();
  if (support_wire) {
    const auto tag = in.word();
    if (tag > 1)
      throw std::invalid_argument("invalid CONT5 archive field expression table tag");
    expressions_wire = tag != 0;
  }
  bool support_seen = false, traits_seen = false;
  out.source_rank = source_rank;
  out.source_ranks = source_ranks;
  std::vector<FieldExchangeExpressions::Ref> expressions;
  std::vector<bool> expressions_used;
  if (expressions_wire) {
    const auto encoded = in.blob();
    const auto limits = field_expression_limits_(capacity_);
    reserve(FieldExchangeExpressions::decoded_storage_bound(encoded, limits));
    expressions = FieldExchangeExpressions::decode_roots(encoded, limits);
    if (expressions.empty() || std::find(expressions.begin(), expressions.end(),
                                         FieldExchangeExpressions::Ref{}) != expressions.end())
      throw std::invalid_argument("CONT4 archive has no retained field expression");
    charge(expressions.size(), sizeof(bool));
    expressions_used.resize(expressions.size());
  }
  const auto realms = in.word(), primary_realm = multi ? in.word() : 0;
  if (realms > (bytes.size() - in.at) / 24 || primary_realm > realms)
    throw std::invalid_argument("continuation archive has an invalid finite realm table");
  charge(realms, sizeof(FiniteDomainRealmRef) + sizeof(FiniteDomainRealm));
  out.realms.reserve(static_cast<std::size_t>(realms));
  for (std::uint64_t i = 0; i < realms; ++i) {
    const std::array<std::uint64_t, 2> origin{in.word(), in.word()};
    const auto replicated = in.word();
    if (replicated > 1)
      throw std::invalid_argument("continuation archive finite ownership tag is invalid");
    for (const auto& prior : out.realms)
      if (prior->unexported_identity() == origin)
        throw std::invalid_argument("continuation archive repeats a finite realm");
    auto realm = FiniteDomainRealmRef(new FiniteDomainRealm(origin, bool(replicated)));
    if (decode_context_)
      realm = decode_context_->intern_realm(std::move(realm));
    if (replicated && (!multi || primary_realm == i + 1)) {
      if (out.realm)
        throw std::invalid_argument("continuation archive has multiple finite allocators");
      out.realm = realm;
    }
    if (multi && primary_realm == i + 1 && !replicated)
      throw std::invalid_argument("continuation archive finite allocator is not replicated");
    out.realms.push_back(std::move(realm));
  }
  const auto finite = in.word();
  if (finite > (bytes.size() - in.at) / 40)
    throw std::invalid_argument("continuation archive finite domain table is truncated");
  charge(finite, sizeof(FiniteDomainRef) + sizeof(FiniteDomain));
  out.finite.reserve(static_cast<std::size_t>(finite));
  for (std::uint64_t i = 0; i < finite; ++i) {
    const auto realm = in.word();
    const auto size = signed_word_(in), cap = signed_word_(in), family = signed_word_(in),
               version = signed_word_(in);
    if (realm >= realms || size < 0 || cap < size || family < 0 || version <= 0)
      throw std::invalid_argument("continuation archive finite descriptor is invalid");
    for (const auto& prior : out.finite)
      if (prior->realm() == out.realms[realm] && prior->version == version)
        throw std::invalid_argument("continuation archive duplicates a finite version");
    auto domain = FiniteDomainRef(new FiniteDomain(size, cap, family, version, out.realms[realm]));
    if (decode_context_)
      domain = decode_context_->intern_domain(std::move(domain));
    if (domain->realm() == out.realm)
      out.max_domain_version = std::max(out.max_domain_version, version);
    out.finite.push_back(std::move(domain));
  }
  const auto read_registry = [&] {
    ArchiveRegistry registry{{in.word(), in.word()}, in.word(), in.word()};
    if ((!registry.issuer[0] && !registry.issuer[1]) ||
        !distributed::valid_budget(registry.bytes) ||
        !distributed::valid_budget(registry.versions) ||
        (!out.registries.empty() && out.registries.back().issuer >= registry.issuer))
      throw std::invalid_argument("continuation archive registry table is invalid");
    charge(1, sizeof(ArchiveRegistry));
    out.registries.push_back(registry);
  };
  if (multi) {
    const auto count = in.word(), primary = in.word();
    if (count > (bytes.size() - in.at) / 32 || primary > count)
      throw std::invalid_argument("continuation archive registry table is truncated");
    if (primary)
      out.primary_registry = static_cast<std::size_t>(primary - 1);
    for (std::uint64_t i = 0; i < count; ++i)
      read_registry();
  }
  const auto domains = in.word();
  if (domains > (bytes.size() - in.at) / 40)
    throw std::invalid_argument("continuation archive distributed domain table is truncated");
  if (domains && !multi) {
    read_registry();
    out.primary_registry = 0;
  }
  charge(domains, sizeof(ArchiveDomain));
  out.domains.reserve(static_cast<std::size_t>(domains));
  for (std::uint64_t i = 0; i < domains; ++i) {
    const auto issuer = multi ? in.word() : 0, ordinal = in.word();
    const auto family = signed_word_(in), cap = signed_word_(in);
    const auto count = in.word();
    if (issuer >= out.registries.size() || ordinal == 0 || ordinal >= INT64_MAX || family < 0 ||
        cap < 0 || count != std::uint64_t(source_ranks) + 1 || count > (bytes.size() - in.at) / 8)
      throw std::invalid_argument("continuation archive distributed descriptor is invalid");
    ArchiveDomain domain{{out.registries[issuer].issuer, ordinal}, family, cap, {}};
    charge(count, sizeof(std::int64_t));
    domain.offsets.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t k = 0; k < count; ++k) {
      const auto offset = signed_word_(in);
      if (offset < 0 || offset > cap || (k == 0 && offset != 0) ||
          (k && offset < domain.offsets.back()))
        throw std::invalid_argument("continuation archive source GID ownership is invalid");
      domain.offsets.push_back(offset);
    }
    if (!out.domains.empty() &&
        std::tie(out.domains.back().identity.issuer, out.domains.back().identity.ordinal) >=
            std::tie(domain.identity.issuer, domain.identity.ordinal))
      throw std::invalid_argument("continuation archive domain table is not canonical");
    out.domains.push_back(std::move(domain));
  }
  std::map<std::string, const ContinuationType*> array_types;
  continuation_detail::StructuralBudget work(capacity_);
  continuation_detail::walk_structure(
      &type_, [](const ContinuationType& t) { return t.children.size(); },
      [](const ContinuationType& t, std::size_t i) { return &t.children[i]; },
      [&](const ContinuationType& t) {
        if (t.kind == ContinuationKind::DistributedArray) {
          auto id = continuation_transport_type_identity(t, true, capacity_);
          charge(id.size(), 1);
          charge(1, sizeof(typename decltype(array_types)::value_type));
          array_types.emplace(std::move(id), &t);
        }
        return true;
      },
      [](const ContinuationType&) {}, work);
  struct Frame {
    const ContinuationType* type;
    ArchiveNode node;
    std::size_t count;
  };
  const auto read_value = [&](const ContinuationType& type) {
    std::vector<Frame> stack;
    const auto push = [&](const ContinuationType& t) {
      charge(1, sizeof(Frame) + sizeof(ArchiveNode));
      Frame frame{&t, {}, 0};
      auto& node = frame.node;
      node.kind = t.kind;
      if (t.kind == ContinuationKind::Field || t.kind == ContinuationKind::Boolean ||
          t.kind == ContinuationKind::Int64 || t.kind == ContinuationKind::Float64 ||
          t.kind == ContinuationKind::Domain || t.kind == ContinuationKind::Index) {
        auto representation = t.kind == ContinuationKind::Field && traits_wire
                                  ? decode_field_traits_(t, in, reserve)
                                  : FieldRepresentationObligation::Ref{};
        auto support = t.kind == ContinuationKind::Field && support_wire
                           ? decode_field_support_(in, reserve, traits_wire)
                           : ContinuationFieldSupport::Ref{};
        support_seen = support_seen || bool(support);
        traits_seen = traits_seen || bool(representation) || !t.field_provider.empty();
        node.leaf = decode_(t, out.finite, in, std::move(support), std::move(representation));
        if (t.kind == ContinuationKind::Field && expressions_wire) {
          const auto slot = in.word();
          if (slot > expressions.size())
            throw std::invalid_argument("archive Field expression alias is absent");
          if (slot) {
            expressions_used[slot - 1] = true;
            charge(1, sizeof(Value));
            node.leaf = Value::with_field_exchange_expression(node.leaf, expressions[slot - 1]);
          }
        }
      } else {
        if (in.word() != static_cast<unsigned>(t.kind))
          throw std::invalid_argument("continuation archive value changes its installed kind");
        if (t.kind == ContinuationKind::DistributedArray) {
          node.slot = in.word();
          if (node.slot >= out.arrays.size())
            throw std::invalid_argument(
                "continuation archive Array dependency is cyclic or forward");
          const auto& array = out.arrays[node.slot];
          const auto& domain = out.domains[array.domain];
          if (domain.family != t.family || domain.capacity != t.capacity ||
              array.element != t.children.at(0))
            throw std::invalid_argument("continuation archive Array alias changes its type");
        } else if (continuation_distributed_kind(t.kind)) {
          node.slot = in.word();
          if (node.slot >= out.domains.size())
            throw std::invalid_argument("continuation archive domain alias is absent");
          const auto& domain = out.domains[node.slot];
          if (domain.family != t.family || domain.capacity != t.capacity)
            throw std::invalid_argument("continuation archive domain alias changes its type");
          if (t.kind == ContinuationKind::DistributedIndex) {
            node.gid = signed_word_(in);
            if (node.gid < 0 || node.gid >= domain.offsets.back())
              throw std::invalid_argument("continuation archive distributed Index is invalid");
          }
        } else {
          if (t.kind == ContinuationKind::Variant) {
            node.arm = in.word();
            if (node.arm >= t.children.size())
              throw std::invalid_argument("continuation archive Variant tag is invalid");
          }
          if (t.kind == ContinuationKind::Array) {
            const auto slot = in.word();
            if (slot >= out.finite.size())
              throw std::invalid_argument("continuation archive finite Array domain is absent");
            node.finite = out.finite[slot];
            continuation_detail::domain(node.finite, t);
          }
          const auto count = in.word();
          const auto expected = t.kind == ContinuationKind::Product   ? t.children.size()
                                : t.kind == ContinuationKind::Variant ? 1
                                : node.finite ? std::uint64_t(node.finite->size)
                                              : UINT64_MAX;
          if (count != expected || count > (bytes.size() - in.at) / 8)
            throw std::invalid_argument("continuation archive container arity is invalid");
          frame.count = static_cast<std::size_t>(count);
          charge(count, sizeof(std::size_t));
          node.children.reserve(frame.count);
        }
      }
      stack.push_back(std::move(frame));
    };
    push(type);
    while (true) {
      auto& frame = stack.back();
      if (frame.node.children.size() < frame.count) {
        const auto selected = frame.type->kind == ContinuationKind::Product
                                  ? frame.node.children.size()
                              : frame.type->kind == ContinuationKind::Variant ? frame.node.arm
                                                                              : 0;
        push(frame.type->children[selected]);
        continue;
      }
      const auto result = out.nodes.size();
      out.nodes.push_back(std::move(frame.node));
      validate_archive_bindings(out, result, *frame.type, reserve);
      stack.pop_back();
      if (stack.empty())
        return result;
      stack.back().node.children.push_back(result);
    }
  };
  const auto arrays = in.word();
  if (arrays > (bytes.size() - in.at) / 40 || (arrays && out.registries.empty()))
    throw std::invalid_argument("continuation archive Array table is truncated or authority-free");
  charge(arrays, sizeof(ArchiveArray));
  out.arrays.reserve(static_cast<std::size_t>(arrays));
  for (std::uint64_t i = 0; i < arrays; ++i) {
    const auto ordinal = in.word();
    const auto type_id = in.text();
    const auto domain = in.word(), witness = in.word(), retained = in.word(), count = in.word();
    const auto found = array_types.find(type_id);
    if (found == array_types.end() || domain >= out.domains.size() || ordinal == 0 ||
        ordinal >= INT64_MAX || witness > out.domains.size() || retained < 32)
      throw std::invalid_argument("continuation archive Array descriptor is invalid");
    const auto& t = *found->second;
    const auto& d = out.domains[domain];
    if (d.family != t.family || d.capacity != t.capacity ||
        count != std::uint64_t(d.offsets[source_rank + 1] - d.offsets[source_rank]) ||
        count > (bytes.size() - in.at) / 8)
      throw std::invalid_argument("continuation archive Array shard/type is invalid");
    for (const auto& prior : out.arrays)
      if (prior.identity.issuer == d.identity.issuer && prior.identity.ordinal == ordinal)
        throw std::invalid_argument("continuation archive repeats an immutable Array version");
    ArchiveArray array;
    array.identity = {d.identity.issuer, ordinal};
    array.domain = static_cast<std::size_t>(domain);
    continuation_detail::StructuralBudget copy_budget(capacity_);
    continuation_detail::walk_structure(
        &t.children[0], [](const ContinuationType& v) { return v.children.size(); },
        [](const ContinuationType& v, std::size_t n) { return &v.children[n]; },
        [&](const ContinuationType& v) {
          charge(1, sizeof(ContinuationType));
          charge(v.names.size(), sizeof(std::string));
          for (const auto& name : v.names)
            charge(name.size(), 1);
          charge(v.scientific_contract.size(), 1);
          charge(v.field_provider.size(), 1);
          charge(v.domain_bindings.size(), sizeof(ContinuationDomainBinding));
          for (const auto& binding : v.domain_bindings)
            charge(binding.domain_path.size() + binding.value_path.size(), sizeof(std::size_t));
          return true;
        },
        [](const ContinuationType&) {}, copy_budget);
    array.element = t.children[0];
    array.charge = retained;
    if (witness)
      array.index_domain = static_cast<std::size_t>(witness - 1);
    charge(count, sizeof(std::size_t));
    array.values.reserve(static_cast<std::size_t>(count));
    const auto start = in.at;
    for (std::uint64_t j = 0; j < count; ++j)
      array.values.push_back(read_value(array.element));
    if (in.at - start > retained - 32)
      throw std::invalid_argument("continuation archive understates Array storage");
    if (array.index_domain) {
      const auto& target = out.domains[*array.index_domain];
      if (array.element.kind != ContinuationKind::DistributedIndex ||
          target.family != array.element.family || target.capacity != array.element.capacity)
        throw std::invalid_argument("continuation archive Index witness changes schema");
      for (auto item : array.values)
        if (out.nodes[item].slot != *array.index_domain)
          throw std::invalid_argument("continuation archive Array mixes Index witnesses");
    }
    out.arrays.push_back(std::move(array));
  }
  out.root = read_value(type_);
  if (traits_wire && !traits_seen)
    throw std::invalid_argument("CONT6 archive has no provider or representation obligation");
  if (!traits_wire && support_wire && !support_seen)
    throw std::invalid_argument("CONT5 archive has no retained Field producer support");
  if (in.at != bytes.size())
    throw std::invalid_argument("continuation archive has trailing bytes");
  if (std::find(expressions_used.begin(), expressions_used.end(), false) != expressions_used.end())
    throw std::invalid_argument("continuation archive retains an unreferenced field expression");
  validate_archive_tables(out, multi, reserve);
  validate_archive_charges(out, reserve);
  return out;
}

// The archive walker only follows finite structural children. Immutable Array
// references remain references; the migration catalogue visits their dependency
// graph explicitly, once per version, before constructing any target authority.
template <class Visit>
static void walk_archive(const ArchiveImage& image, std::size_t root, Visit&& visit,
                         const std::function<void(std::uint64_t)>& reserve) {
  std::vector<std::size_t> pending;
  reserve(sizeof(std::size_t));
  pending.push_back(root);
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    const auto& node = image.nodes.at(id);
    visit(node);
    if (node.children.size() > UINT64_MAX / sizeof(std::size_t))
      throw std::length_error("continuation archive traversal size overflow");
    reserve(node.children.size() * sizeof(std::size_t));
    for (auto i = node.children.rbegin(); i != node.children.rend(); ++i) {
      if (*i >= id)
        throw std::invalid_argument("continuation archive structural edge is not postorder");
      pending.push_back(*i);
    }
  }
}

std::uint64_t archive_transport_size(
    const ArchiveImage& image, std::size_t root, const ContinuationType& type,
    const std::function<void(std::uint64_t)>& reserve,
    const std::function<std::span<const std::int64_t>(distributed::Identity)>& placements = {})
    const {
  // Count the exact existing POPSTRNS2 protocol. This is storage validation,
  // with no numerical execution and no possession of distributed capabilities.
  struct Counter {
    std::uint64_t count = 0, capacity;
    void add(std::uint64_t n) {
      if (n > capacity - count)
        throw std::length_error("continuation archive transport exceeds its installed capacity");
      count += n;
    }
    void word(std::uint64_t) { add(8); }
    void text(const std::string& text) {
      word(text.size());
      add(text.size());
    }
  } out{0, capacity_};
  std::vector<FiniteDomainRef> domains;
  walk_archive(
      image, root,
      [&](const ArchiveNode& node) {
        const auto& domain = node.leaf ? node.leaf->domain_ref() : node.finite;
        if (domain && std::find(domains.begin(), domains.end(), domain) == domains.end()) {
          reserve(sizeof(FiniteDomainRef));
          domains.push_back(domain);
        }
      },
      reserve);
  const auto identity = continuation_transport_type_identity(type, true, capacity_);
  reserve(identity.size());
  continuation_detail::transport_header(out, identity, Dim, domains);
  walk_archive(
      image, root,
      [&](const ArchiveNode& node) {
        if (continuation_distributed_kind(node.kind)) {
          const auto& domain = image.domains.at(node.kind == ContinuationKind::DistributedArray
                                                    ? image.arrays.at(node.slot).domain
                                                    : node.slot);
          const auto version = node.kind == ContinuationKind::DistributedArray
                                   ? image.arrays.at(node.slot).identity
                                   : distributed::Identity{};
          continuation_detail::transport_distributed_node(
              out, node.kind, domain.identity, domain.family, domain.capacity,
              placements ? placements(domain.identity)
                         : std::span<const std::int64_t>(domain.offsets),
              node.gid, version);
        } else {
          const auto word = node.leaf ? node.leaf->word_ : node.arm;
          (void)continuation_detail::transport_finite_node(out, node.kind, word, 0,
                                                           node.children.size());
        }
      },
      reserve);
  return out.count;
}

void validate_archive_charges(const ArchiveImage& image,
                              const std::function<void(std::uint64_t)>& reserve) const {
  std::map<std::array<std::uint64_t, 2>, std::pair<std::uint64_t, std::uint64_t>> usage;
  for (const auto& registry : image.registries) {
    reserve(sizeof(typename decltype(usage)::value_type));
    usage.emplace(registry.issuer, std::pair<std::uint64_t, std::uint64_t>{0, 0});
  }
  for (const auto& array : image.arrays) {
    const auto& domain = image.domains.at(array.domain);
    const auto type_bytes = continuation_transport_type_identity(array.element, true, capacity_);
    reserve(type_bytes.size());
    if (domain.offsets.size() > (UINT64_MAX - 256) / 8)
      throw std::length_error("continuation archive Array charge overflows");
    auto charge = 256 + 8 * std::uint64_t(domain.offsets.size());
    const auto add = [&](std::uint64_t bytes) {
      if (bytes > UINT64_MAX - charge)
        throw std::length_error("continuation archive Array charge overflows");
      charge += bytes;
    };
    add(type_bytes.size());
    for (const auto item : array.values) {
      add(32);
      add(archive_transport_size(image, item, array.element, reserve));
    }
    if (charge != array.charge)
      throw std::invalid_argument(
          "continuation archive retained charge differs from native policy");
    auto& used = usage.at(array.identity.issuer);
    if (charge > UINT64_MAX - used.first || used.second == UINT64_MAX)
      throw std::length_error("continuation archive registry accounting overflows");
    used.first += charge;
    ++used.second;
  }
  for (const auto& registry : image.registries) {
    const auto& used = usage.at(registry.issuer);
    if (used.first > registry.bytes || used.second > registry.versions)
      throw std::length_error("continuation archive exceeds its source registry capacities");
  }
}

std::string archive_collective_identity(const ArchiveImage& image, std::uint64_t output_capacity,
                                        const std::function<void(std::uint64_t)>& reserve) const {
  continuation_detail::Writer out(std::min(capacity_, output_capacity));
  out.text(identity_);
  out.word(image.publication);
  walk_archive(
      image, image.root,
      [&](const ArchiveNode& node) {
        if (node.leaf) {
          (void)collective_node_(*node.leaf, out);
          return;
        }
        out.word(static_cast<unsigned>(node.kind));
        if (node.finite) {
          const auto& domain = node.finite;
          out.word(domain->family);
          out.word(domain->capacity);
          out.word(domain->size);
          out.word(domain->version);
          out.word(domain->realm()->replicated());
          const bool named = !domain->realm()->replicated() || domain->realm()->synchronized();
          out.word(named);
          if (named) {
            const auto origin = domain->realm()->transport_identity();
            out.word(origin[0]);
            out.word(origin[1]);
          }
        }
        if (continuation_distributed_kind(node.kind)) {
          const auto& domain = image.domains.at(node.kind == ContinuationKind::DistributedArray
                                                    ? image.arrays.at(node.slot).domain
                                                    : node.slot);
          out.word(domain.identity.issuer[0]);
          out.word(domain.identity.issuer[1]);
          out.word(domain.identity.ordinal);
          out.word(domain.family);
          out.word(domain.capacity);
          out.word(domain.offsets.size());
          for (const auto offset : domain.offsets)
            out.word(offset);
          if (node.kind == ContinuationKind::DistributedIndex)
            out.word(node.gid);
          if (node.kind == ContinuationKind::DistributedArray) {
            const auto& array = image.arrays.at(node.slot);
            out.word(array.identity.issuer[0]);
            out.word(array.identity.issuer[1]);
            out.word(array.identity.ordinal);
            out.word(bool(array.index_domain));
            if (array.index_domain) {
              const auto& target = image.domains.at(*array.index_domain).identity;
              out.word(target.issuer[0]);
              out.word(target.issuer[1]);
              out.word(target.ordinal);
            }
          }
          return;
        }
        if (node.kind == ContinuationKind::Variant)
          out.word(node.arm);
        out.word(node.children.size());
      },
      reserve);
  reserve(out.bytes.size());
  return std::string(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
}

static void validate_archive_bindings(const ArchiveImage& image, std::size_t root,
                                      const ContinuationType& type,
                                      const std::function<void(std::uint64_t)>& reserve) {
  const auto finite = [&](const ArchiveNode& node) -> FiniteDomainRef {
    return node.leaf ? node.leaf->domain_ref() : node.finite;
  };
  const auto distributed = [&](const ArchiveNode& node) -> const ArchiveDomain* {
    if (!continuation_distributed_kind(node.kind))
      return nullptr;
    return &image.domains.at(node.kind == ContinuationKind::DistributedArray
                                 ? image.arrays.at(node.slot).domain
                                 : node.slot);
  };
  for (const auto& binding : type.domain_bindings) {
    auto binder = root;
    for (auto at : binding.domain_path)
      binder = image.nodes.at(binder).children.at(at);
    const auto& bound = image.nodes.at(binder);
    const auto target_finite = finite(bound);
    const auto* target_distributed = distributed(bound);
    if (!target_finite && !target_distributed)
      throw std::invalid_argument("continuation archive Bundle has an absent domain binder");
    std::vector<std::pair<std::size_t, std::size_t>> pending;
    reserve(sizeof(typename decltype(pending)::value_type));
    pending.emplace_back(root, 0);
    while (!pending.empty()) {
      const auto [id, at] = pending.back();
      pending.pop_back();
      const auto& node = image.nodes.at(id);
      if (at == binding.value_path.size()) {
        if (finite(node) != target_finite || distributed(node) != target_distributed)
          throw std::invalid_argument("continuation archive Bundle changes exact domain alias");
        continue;
      }
      const auto push = [&](std::size_t child) {
        reserve(sizeof(typename decltype(pending)::value_type));
        pending.emplace_back(child, at + 1);
      };
      const auto item = binding.value_path[at];
      if (continuation_array_kind(node.kind) && item == continuation_each) {
        if (node.kind == ContinuationKind::DistributedArray) {
          const auto& array = image.arrays.at(node.slot);
          if (at + 1 == binding.value_path.size() && array.index_domain &&
              &image.domains.at(*array.index_domain) != target_distributed)
            throw std::invalid_argument("continuation archive Bundle changes empty Index witness");
          for (const auto child : array.values)
            push(child);
        } else {
          for (const auto child : node.children)
            push(child);
        }
      } else if (node.kind == ContinuationKind::Variant) {
        if (node.arm == item)
          push(node.children.at(0));
      } else
        push(node.children.at(item));
    }
  }
}

static void validate_archive_tables(const ArchiveImage& image, bool multi,
                                    const std::function<void(std::uint64_t)>& reserve) {
  std::vector<FiniteDomainRef> finite;
  std::vector<FiniteDomainRealmRef> realms;
  std::vector<std::size_t> arrays;
  std::set<std::size_t> visited;
  struct Task {
    unsigned kind;
    std::size_t index;
  };
  std::vector<Task> pending;
  const auto push = [&](unsigned kind, std::size_t index) {
    reserve(sizeof(Task));
    pending.push_back({kind, index});
  };
  const auto process = [&] {
    while (!pending.empty()) {
      const auto task = pending.back();
      pending.pop_back();
      if (task.kind == 2) {
        reserve(sizeof(std::size_t));
        arrays.push_back(task.index);
        continue;
      }
      if (task.kind == 1) {
        reserve(sizeof(std::size_t));
        if (!visited.insert(task.index).second)
          continue;
        push(2, task.index);
        const auto& values = image.arrays.at(task.index).values;
        for (auto i = values.rbegin(); i != values.rend(); ++i)
          push(0, *i);
        continue;
      }
      const auto& node = image.nodes.at(task.index);
      const auto& domain = node.leaf ? node.leaf->domain_ref() : node.finite;
      if (domain && std::find(finite.begin(), finite.end(), domain) == finite.end()) {
        reserve(sizeof(FiniteDomainRef));
        finite.push_back(domain);
        if (std::find(realms.begin(), realms.end(), domain->realm()) == realms.end()) {
          reserve(sizeof(FiniteDomainRealmRef));
          realms.push_back(domain->realm());
        }
      }
      if (node.kind == ContinuationKind::DistributedArray)
        push(1, node.slot);
      else
        for (auto i = node.children.rbegin(); i != node.children.rend(); ++i)
          push(0, *i);
    }
  };
  push(0, image.root);
  process();
  std::vector<std::size_t> sorted;
  reserve(image.arrays.size() * sizeof(std::size_t));
  for (std::size_t i = 0; i < image.arrays.size(); ++i)
    sorted.push_back(i);
  std::sort(sorted.begin(), sorted.end(), [&](std::size_t a, std::size_t b) {
    const auto& x = image.arrays[a].identity;
    const auto& y = image.arrays[b].identity;
    return std::tie(x.issuer, x.ordinal) < std::tie(y.issuer, y.ordinal);
  });
  for (const auto array : sorted) {
    push(1, array);
    process();
  }
  if (image.realm && std::find(realms.begin(), realms.end(), image.realm) == realms.end())
    realms.push_back(image.realm);
  if (finite != image.finite || realms != image.realms)
    throw std::invalid_argument(
        "continuation archive finite alias tables are not canonical/reachable");
  const auto& expected = multi ? arrays : sorted;
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (expected[i] != i)
      throw std::invalid_argument("continuation archive Array table is not canonical");
}
