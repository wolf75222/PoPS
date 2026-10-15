// Included in ContinuationCodec<Dim>. Structural checkpoint of multiple finite issuers
// and immutable distributed Array DAGs; construction remains entirely rank-local.
struct ExtendedTable {
  std::vector<FiniteDomainRef> finite;
  std::vector<FiniteDomainRealmRef> realms;
  std::vector<std::shared_ptr<distributed::Registry>> registries;
  std::vector<distributed::DomainRef> distributed;
  std::vector<ContinuationDistributedArrayRef<Dim>> arrays;  // dependencies before users
  std::vector<FieldExchangeExpressions::Ref> field_expressions;
  mutable std::vector<bool> field_expression_used;
  bool field_expression_wire = false;
};
static FieldExchangeExpressions::Limits field_expression_limits_(std::uint64_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max())
    throw std::length_error("field expression budget exceeds native address space");
  const auto limit = static_cast<std::size_t>(bytes);
  return {limit, limit, limit, limit};
}
bool requires_extended_(const Ref& root) const {
  FiniteDomainRealmRef first;
  bool needed = false;
  continuation_detail::walk_value<Dim>(
      root, capacity_,
      [&](const Value& value) {
        if (continuation_distributed_kind(value.kind()))
          needed = true;
        if (value.field_exchange_expression())
          needed = true;
        if (const auto& domain = value.domain_ref()) {
          // CONT1 has no realm transport-identity table.  Once a replicated realm has been
          // synchronized, its identity is an accepted authority and must survive the image.
          if (!domain->realm()->replicated() || domain->realm()->synchronized())
            needed = true;
          if (first && first != domain->realm())
            needed = true;
          else
            first = domain->realm();
        }
        return true;
      },
      [](const Value&) {});
  return needed;
}
template <class T>
static std::size_t table_slot_(const std::vector<T>& values, const T& value) {
  const auto found = std::find(values.begin(), values.end(), value);
  if (found == values.end())
    throw std::invalid_argument("continuation alias table is incomplete");
  return static_cast<std::size_t>(found - values.begin());
}
static ContinuationType array_type_(const ContinuationDistributedArrayRef<Dim>& array) {
  ContinuationType type;
  type.kind = ContinuationKind::DistributedArray;
  type.family = array->domain->family();
  type.capacity = array->domain->capacity();
  type.children.push_back(array->element);
  return type;
}
void collect_extended_(const Ref& value, ExtendedTable& table) const {
  continuation_detail::walk_value<Dim>(
      value, capacity_,
      [&](const Value& node) {
        if (const auto& expression = node.field_exchange_expression();
            expression && std::find(table.field_expressions.begin(), table.field_expressions.end(),
                                    expression) == table.field_expressions.end())
          table.field_expressions.push_back(expression);
        if (const auto& domain = node.domain_ref();
            domain &&
            std::find(table.finite.begin(), table.finite.end(), domain) == table.finite.end()) {
          table.finite.push_back(domain);
          if (std::find(table.realms.begin(), table.realms.end(), domain->realm()) ==
              table.realms.end())
            table.realms.push_back(domain->realm());
        }
        if (const auto& domain = node.distributed_domain_ref();
            domain && std::find(table.distributed.begin(), table.distributed.end(), domain) ==
                          table.distributed.end())
          table.distributed.push_back(domain);
        const auto& array = node.distributed_array_ref();
        if (array && array->index_domain &&
            std::find(table.distributed.begin(), table.distributed.end(), array->index_domain) ==
                table.distributed.end())
          table.distributed.push_back(array->index_domain);
        return !array ||
               std::find(table.arrays.begin(), table.arrays.end(), array) == table.arrays.end();
      },
      [&](const Value& node) {
        const auto& array = node.distributed_array_ref();
        if (array &&
            std::find(table.arrays.begin(), table.arrays.end(), array) == table.arrays.end())
          table.arrays.push_back(array);  // immutable dependencies precede their users
      });
}
std::vector<const distributed::Registry*> distributed_authorities_() const {
  std::map<std::array<std::uint64_t, 2>, const distributed::Registry*> by_issuer;
  const auto add = [&](const distributed::Registry* authority) {
    if (!authority)
      throw std::logic_error("continuation has an absent distributed issuer authority");
    if (!by_issuer.empty() && &by_issuer.begin()->second->lane() != &authority->lane())
      throw std::logic_error("continuation registries belong to different execution lanes");
    const auto [found, inserted] = by_issuer.emplace(authority->issuer(), authority);
    if (!inserted && found->second != authority)
      throw std::logic_error("continuation has conflicting distributed issuer authorities");
  };
  if (registry_)
    add(registry_);
  if (closure_)
    for (const auto& authority : closure_->registries)
      add(authority.get());
  std::vector<const distributed::Registry*> result;
  for (const auto& [issuer, authority] : by_issuer)
    result.push_back(authority);
  return result;
}
std::vector<std::uint8_t> encode_extended_(const Ref& root, std::uint64_t publication) const {
  ExtendedTable table;
  collect_extended_(root, table);
  const bool distributed_image =
      !table.distributed.empty() ||
      (closure_ && (!closure_->domains.empty() || !closure_->arrays.empty()));
  const auto registries = distributed_authorities_();
  // CONT3 records allocator identity separately from retained authorities.
  // A parent may borrow one child issuer without adopting it as its allocator.
  bool multi_issuer = registries.size() > 1 || (!registry_ && !registries.empty());
  const auto authority = [&](distributed::Identity id) -> const distributed::Registry& {
    for (const auto* registry : registries)
      if (registry->issuer() == id.issuer)
        return *registry;
    throw std::logic_error("distributed checkpoint lacks its authorized issuer registry");
  };
  if (distributed_image && registries.empty())
    throw std::logic_error("distributed checkpoint lacks its installed registry authority");
  if (distributed_image) {
    if (!closure_)
      throw std::logic_error(
          "distributed checkpoint requires a complete collective dependency closure");
    for (const auto& array : closure_->arrays)
      collect_extended_(Value::distributed_array(array), table);
    // The shared closure is in native publication order; dependencies necessarily precede
    // their immutable users. All ranks retain the same version table and their own shards.
    table.distributed = closure_->domains;
    if (multi_issuer) {
      // Local shards may contain cross-issuer edges. Their alias table must put
      // dependencies before users; closure consensus itself remains ID-sorted.
      auto sorted = table.arrays;
      const auto less = [](const auto& a, const auto& b) {
        return std::tie(a->version.issuer, a->version.ordinal) <
               std::tie(b->version.issuer, b->version.ordinal);
      };
      std::sort(sorted.begin(), sorted.end(), less);
      if (sorted != closure_->arrays)
        throw std::logic_error("distributed checkpoint closure is incomplete");
    } else {
      table.arrays = closure_->arrays;
    }
  }
  if (closure_ && closure_->replicated_realm &&
      std::find(table.realms.begin(), table.realms.end(), closure_->replicated_realm) ==
          table.realms.end())
    table.realms.push_back(closure_->replicated_realm);
  // The allocator is distinct from retained child authorities. Each exact
  // domain keeps its original realm, including replicated child domains.
  FiniteDomainRealmRef replicated = closure_ ? closure_->replicated_realm : FiniteDomainRealmRef{};
  std::size_t replicated_count = 0;
  for (std::size_t i = 0; i < table.realms.size(); ++i) {
    const auto& realm = table.realms[i];
    if (!realm || !realm->visible())
      throw std::invalid_argument("continuation contains a revoked finite authority");
    if (realm->replicated()) {
      ++replicated_count;
      if (!replicated)
        replicated = realm;
    }
    for (std::size_t j = 0; j < i; ++j)
      if (table.realms[j]->unexported_identity() == realm->unexported_identity())
        throw std::invalid_argument("duplicate finite realm origin");
  }
  for (std::size_t i = 0; i < table.finite.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (table.finite[i]->realm() == table.finite[j]->realm() &&
          table.finite[i]->version == table.finite[j]->version)
        throw std::invalid_argument("duplicate finite domain version");
  table.field_expression_wire = !table.field_expressions.empty();
  multi_issuer = multi_issuer || replicated_count > 1 || table.field_expression_wire;

  continuation_detail::Writer out(output_capacity_);

  out.text(table.field_expression_wire ? "POPSCONT4" : multi_issuer ? "POPSCONT3" : "POPSCONT2");
  out.text(identity_);
  out.word(Dim);
  out.word(publication);
  if (table.field_expression_wire) {
    if (out.bytes.size() + 8 > output_capacity_)
      throw std::length_error("field expression table exceeds continuation output capacity");
    const auto expression_bytes = FieldExchangeExpressions::encode_roots(
        table.field_expressions, field_expression_limits_(output_capacity_ - out.bytes.size() - 8));
    out.blob(expression_bytes);
  }
  out.word(table.realms.size());
  if (multi_issuer)
    out.word(replicated ? table_slot_(table.realms, replicated) + 1 : 0);
  for (const auto& realm : table.realms) {
    const auto origin = realm->transport_identity();
    out.word(origin[0]);
    out.word(origin[1]);
    out.word(realm->replicated());
  }
  out.word(table.finite.size());
  for (const auto& domain : table.finite) {
    out.word(table_slot_(table.realms, domain->realm()));
    out.word(domain->size);
    out.word(domain->capacity);
    out.word(domain->family);
    out.word(domain->version);
  }
  if (multi_issuer) {
    out.word(registries.size());
    out.word(registry_ ? table_slot_(registries, registry_) + 1 : 0);
    for (const auto* registry : registries) {
      out.word(registry->issuer()[0]);
      out.word(registry->issuer()[1]);
      out.word(registry->storage_bytes());
      out.word(registry->storage_versions());
    }
  }
  out.word(table.distributed.size());
  if (!table.distributed.empty()) {
    if (!multi_issuer) {
      const auto* registry = registries.front();
      out.word(registry->issuer()[0]);
      out.word(registry->issuer()[1]);
      out.word(registry->storage_bytes());
      out.word(registry->storage_versions());
    }
    for (const auto& domain : table.distributed) {
      const auto id = domain->identity();
      const auto& registry = authority(id);
      registry.authenticate(domain);
      if (multi_issuer)
        out.word(table_slot_(registries, &registry));
      out.word(id.ordinal);
      out.word(domain->family());
      out.word(domain->capacity());
      out.word(domain->partition_offsets().size());
      for (auto offset : domain->partition_offsets())
        out.word(offset);
    }
  }
  out.word(table.arrays.size());
  for (const auto& array : table.arrays) {
    const auto& registry = authority(array->version);
    if (array->version.issuer != array->domain->identity().issuer)
      throw std::logic_error("checkpoint Array changes its domain issuer");
    const auto held = registry.resolve_value(array->version, distributed::array_type_tag<Dim>());
    if (held.get() != array.get())
      throw std::logic_error("checkpoint Array is not its published immutable version");
    out.word(array->version.ordinal);
    out.text(continuation_transport_type_identity(array_type_(array), true, capacity_));
    out.word(table_slot_(table.distributed, array->domain));
    out.word(array->index_domain ? table_slot_(table.distributed, array->index_domain) + 1 : 0);
    out.word(registry.value_charge(array->version, distributed::array_type_tag<Dim>()));
    out.word(array->values.size());
    for (const auto& item : array->values)
      encode_extended_value_(item, table, out);
  }
  encode_extended_value_(root, table, out);
  return std::move(out.bytes);
}
static void encode_extended_value_(const Ref& value, const ExtendedTable& table,
                                   continuation_detail::Writer& out) {
  continuation_detail::walk_value<Dim>(
      value, out.capacity(),
      [&](const Value& node) {
        const auto* value = &node;
        const auto kind = value->kind();
        if (kind == ContinuationKind::Field || kind == ContinuationKind::Boolean ||
            kind == ContinuationKind::Int64 || kind == ContinuationKind::Float64 ||
            kind == ContinuationKind::Domain || kind == ContinuationKind::Index) {
          encode_(value, table.finite, out);
          if (kind == ContinuationKind::Field && table.field_expression_wire)
            out.word(
                value->field_exchange_expression()
                    ? table_slot_(table.field_expressions, value->field_exchange_expression()) + 1
                    : 0);
          return false;
        }
        out.word(static_cast<unsigned>(kind));
        if (kind == ContinuationKind::DistributedDomain ||
            kind == ContinuationKind::DistributedIndex) {
          out.word(table_slot_(table.distributed, value->distributed_domain_ref()));
          if (kind == ContinuationKind::DistributedIndex)
            out.word(std::bit_cast<std::uint64_t>(value->distributed_index_offset()));
          return false;
        }
        if (kind == ContinuationKind::DistributedArray) {
          out.word(table_slot_(table.arrays, value->distributed_array_ref()));
          return false;
        }
        if (kind == ContinuationKind::Variant)
          out.word(value->arm());
        if (kind == ContinuationKind::Array)
          out.word(table_slot_(table.finite, value->domain_ref()));
        out.word(value->members().size());
        return true;
      },
      [](const Value&) {});
}
Decoded decode_extended_(continuation_detail::Reader& in, bool multi_issuer = false,
                         bool field_expression_wire = false) const {
  if (in.text() != identity_ || in.word() != Dim)
    throw std::invalid_argument("extended continuation schema or dimension differs");
  Decoded out;
  out.publication = in.word();
  ExtendedTable table;
  table.field_expression_wire = field_expression_wire;
  if (field_expression_wire) {
    const auto encoded = in.blob();
    if (FieldExchangeExpressions::decoded_storage_bound(
            encoded, field_expression_limits_(capacity_)) > capacity_)
      throw std::length_error("field expression decode workspace exceeds continuation capacity");
    table.field_expressions =
        FieldExchangeExpressions::decode_roots(encoded, field_expression_limits_(capacity_));
    table.field_expression_used.resize(table.field_expressions.size());
    if (table.field_expressions.empty() ||
        std::find(table.field_expressions.begin(), table.field_expressions.end(),
                  FieldExchangeExpressions::Ref{}) != table.field_expressions.end())
      throw std::invalid_argument("CONT4 has no retained field expression");
  }
  const auto realm_count = in.word();
  const auto primary_realm = multi_issuer ? in.word() : 0;
  if (primary_realm > realm_count)
    throw std::invalid_argument("finite allocator realm is absent from its authority table");
  if (realm_count > (in.bytes.size() - in.at) / 24)
    throw std::invalid_argument("truncated finite realm table");
  for (std::uint64_t i = 0; i < realm_count; ++i) {
    const std::array<std::uint64_t, 2> origin{in.word(), in.word()};
    const auto replicated = in.word();
    if (replicated > 1)
      throw std::invalid_argument("invalid finite ownership tag");
    for (const auto& prior : table.realms)
      if (prior->unexported_identity() == origin)
        throw std::invalid_argument("duplicate finite realm origin");
    auto realm = FiniteDomainRealmRef(new FiniteDomainRealm(origin, bool(replicated)));
    if (decode_context_)
      realm = decode_context_->intern_realm(std::move(realm));
    if (replicated && (!multi_issuer || primary_realm == i + 1)) {
      if (out.realm)
        throw std::invalid_argument("multiple finite allocator authorities in one continuation");
      out.realm = realm;
    }
    if (multi_issuer && primary_realm == i + 1 && !replicated)
      throw std::invalid_argument("finite allocator realm is not replicated");
    table.realms.push_back(std::move(realm));
  }
  const auto finite_count = in.word();
  if (finite_count > (in.bytes.size() - in.at) / 40)
    throw std::invalid_argument("truncated finite domain table");
  for (std::uint64_t i = 0; i < finite_count; ++i) {
    const auto realm = in.word();
    const auto size = signed_word_(in), cap = signed_word_(in), family = signed_word_(in),
               version = signed_word_(in);
    if (realm >= table.realms.size() || size < 0 || cap < size || family < 0 || version <= 0)
      throw std::invalid_argument("invalid finite domain descriptor");
    for (const auto& prior : table.finite)
      if (prior->realm() == table.realms[realm] && prior->version == version)
        throw std::invalid_argument("duplicate finite domain version");
    auto domain =
        FiniteDomainRef(new FiniteDomain(size, cap, family, version, table.realms[realm]));
    if (decode_context_)
      domain = decode_context_->intern_domain(std::move(domain));
    table.finite.push_back(std::move(domain));
    if (table.realms[realm] == out.realm)
      out.max_domain_version = std::max(out.max_domain_version, version);
  }
  const auto read_registry = [&] {
    if (!lane_)
      throw std::logic_error("distributed restore requires an installed execution lane");
    const std::array<std::uint64_t, 2> issuer{in.word(), in.word()};
    const auto bytes = in.word(), versions = in.word();
    if (!table.registries.empty() && table.registries.back()->issuer() >= issuer)
      throw std::invalid_argument("distributed issuer table is duplicated or not canonical");
    table.registries.push_back(
        decode_context_ ? decode_context_->registry(*lane_, issuer, bytes, versions)
                        : distributed::Registry::restore_local(*lane_, issuer, bytes, versions));
  };
  if (multi_issuer) {
    const auto count = in.word(), primary = in.word();
    if (count > (in.bytes.size() - in.at) / 32 || primary > count)
      throw std::invalid_argument("distributed issuer table is truncated or invalid");
    for (std::uint64_t i = 0; i < count; ++i)
      read_registry();
    if (primary)
      out.distributed_registry = table.registries[primary - 1];
  }
  const auto distributed_count = in.word();
  if (distributed_count > (in.bytes.size() - in.at) / 40)
    throw std::invalid_argument("truncated distributed domain table");
  if (distributed_count) {
    if (!multi_issuer) {
      read_registry();
      out.distributed_registry = table.registries.front();
    }
    for (std::uint64_t i = 0; i < distributed_count; ++i) {
      const auto issuer_slot = multi_issuer ? in.word() : 0;
      if (issuer_slot >= table.registries.size())
        throw std::invalid_argument("distributed domain has an absent issuer authority");
      const auto& registry = table.registries[issuer_slot];
      const auto ordinal = in.word();
      const auto family = signed_word_(in), cap = signed_word_(in);
      const auto n = in.word();
      if (n != std::size_t(lane_->size() + 1) || n > (in.bytes.size() - in.at) / 8)
        throw std::invalid_argument("checkpoint ownership/rank count differs from installed lane");
      std::vector<std::int64_t> offsets;
      offsets.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t j = 0; j < n; ++j)
        offsets.push_back(signed_word_(in));
      const distributed::Identity id{registry->issuer(), ordinal};
      table.distributed.push_back(decode_context_
                                      ? decode_context_->domain(*registry, id, family, cap, offsets)
                                      : registry->restore_domain_local(id, family, cap, offsets));
    }
  }
  const auto authority = [&](distributed::Identity id) -> distributed::Registry& {
    for (const auto& registry : table.registries)
      if (registry->issuer() == id.issuer)
        return *registry;
    throw std::invalid_argument("checkpoint Array has an absent issuer authority");
  };
  std::map<std::string, ContinuationType> array_types;
  continuation_detail::StructuralBudget type_budget(capacity_);
  continuation_detail::walk_structure(
      &type_, [](const ContinuationType& node) { return node.children.size(); },
      [](const ContinuationType& node, std::size_t i) { return &node.children[i]; },
      [&](const ContinuationType& type) {
        if (type.kind == ContinuationKind::DistributedArray)
          array_types.emplace(continuation_transport_type_identity(type, true, capacity_), type);
        return true;
      },
      [](const ContinuationType&) {}, type_budget);
  const auto arrays = in.word();
  if (arrays > (in.bytes.size() - in.at) / 40 || (arrays && table.registries.empty()))
    throw std::invalid_argument("truncated or authority-free distributed Array table");
  for (std::uint64_t i = 0; i < arrays; ++i) {
    const auto ordinal = in.word();
    const auto type_id = in.text();
    const auto slot = in.word(), target = in.word(), charge = in.word(), count = in.word();
    const auto found = array_types.find(type_id);
    if (found == array_types.end() || slot >= table.distributed.size())
      throw std::invalid_argument("Array table type/domain absent from installed schema");
    const auto& type = found->second;
    const auto& domain = table.distributed[slot];
    if (domain->family() != type.family || domain->capacity() != type.capacity ||
        count != std::size_t(domain->local_size()) || count > (in.bytes.size() - in.at) / 8 ||
        charge < 32)
      throw std::invalid_argument("Array table has inconsistent ownership, type or charge");
    auto array = std::make_shared<ContinuationDistributedArray<Dim>>();
    array->domain = domain;
    array->element = type.children[0];
    if (target) {
      if (target - 1 >= table.distributed.size())
        throw std::invalid_argument("Array Index target witness is absent");
      array->index_domain = table.distributed[target - 1];
    }
    auto& registry = authority(domain->identity());
    array->version = distributed::Identity{registry.issuer(), ordinal};
    array->values.reserve(static_cast<std::size_t>(count));
    const auto begin = in.at;
    for (std::uint64_t j = 0; j < count; ++j)
      array->values.push_back(decode_extended_value_(array->element, table, in));
    if (in.at - begin > charge - 32)
      throw std::invalid_argument("Array checkpoint understates its retained allocation charge");
    const auto actual_charge =
        continuation_distributed_array_charge<Dim>(array, registry, capacity_, table.registries);
    if (charge != actual_charge)
      throw std::invalid_argument(
          "Array checkpoint retained charge differs from the native allocation policy");
    if (decode_context_) {
      table.arrays.push_back(
          decode_context_->array(registry, std::move(array), charge, capacity_, table.registries));
    } else {
      registry.restore_value_local(array->version, array, distributed::array_type_tag<Dim>(),
                                   charge);
      table.arrays.push_back(std::move(array));
    }
  }
  out.root = decode_extended_value_(type_, table, in);
  if (in.at != in.bytes.size())
    throw std::invalid_argument("extended continuation image has trailing bytes");
  validate(out.root);
  ExtendedTable used;
  collect_extended_(out.root, used);
  for (const auto& array : table.arrays) {
    validate_(array_type_(array), Value::distributed_array(array));
    collect_extended_(Value::distributed_array(array), used);
  }
  if (out.realm &&
      std::find(used.realms.begin(), used.realms.end(), out.realm) == used.realms.end())
    used.realms.push_back(out.realm);
  if (used.finite.size() != table.finite.size() || used.realms.size() != table.realms.size())
    throw std::invalid_argument("checkpoint tables contain unreachable authorities or values");
  if (std::find(table.field_expression_used.begin(), table.field_expression_used.end(), false) !=
      table.field_expression_used.end())
    throw std::invalid_argument("checkpoint field expression table contains an unreferenced root");
  // Exact global reachability is checked collectively by the restart transaction before
  // publication; a rank-local shard alone cannot establish or refute it.
  auto closure = std::make_shared<ContinuationClosure<Dim>>();
  closure->domains = table.distributed;
  closure->arrays = table.arrays;
  if (multi_issuer) {
    std::sort(closure->arrays.begin(), closure->arrays.end(), [](const auto& a, const auto& b) {
      return std::tie(a->version.issuer, a->version.ordinal) <
             std::tie(b->version.issuer, b->version.ordinal);
    });
  }
  closure->registries = table.registries;
  closure->replicated_realm = out.realm;
  out.closure = std::move(closure);
  out.realms = std::move(table.realms);
  return out;
}
Ref decode_extended_value_(const ContinuationType& type, const ExtendedTable& table,
                           continuation_detail::Reader& in) const {
  using Frame = continuation_detail::DecodeFrame<Dim>;
  return continuation_detail::decode_structure<Dim>(
      type, capacity_, [&](const ContinuationType& type) -> Frame {
        const auto kind = type.kind;
        if (kind == ContinuationKind::Field || kind == ContinuationKind::Boolean ||
            kind == ContinuationKind::Int64 || kind == ContinuationKind::Float64 ||
            kind == ContinuationKind::Domain || kind == ContinuationKind::Index) {
          auto leaf = decode_(type, table.finite, in);
          if (kind == ContinuationKind::Field && table.field_expression_wire) {
            const auto slot = in.word();
            if (slot > table.field_expressions.size())
              throw std::invalid_argument("checkpoint Field expression alias is absent");
            if (slot) {
              table.field_expression_used[slot - 1] = true;
              leaf = Value::with_field_exchange_expression(leaf, table.field_expressions[slot - 1]);
            }
          }
          return Frame::leaf(std::move(leaf));
        }
        if (in.word() != static_cast<unsigned>(kind))
          throw std::invalid_argument("extended continuation wire kind differs");
        if (kind == ContinuationKind::DistributedDomain ||
            kind == ContinuationKind::DistributedIndex) {
          const auto slot = in.word();
          if (slot >= table.distributed.size())
            throw std::invalid_argument("distributed domain alias is absent");
          const auto& domain = table.distributed[slot];
          if (domain->family() != type.family || domain->capacity() != type.capacity)
            throw std::invalid_argument("distributed domain alias changes family");
          if (kind == ContinuationKind::DistributedDomain)
            return Frame::leaf(Value::distributed_domain(domain));
          return Frame::leaf(Value::distributed_index(domain, signed_word_(in)));
        }
        if (kind == ContinuationKind::DistributedArray) {
          const auto slot = in.word();
          if (slot >= table.arrays.size())
            throw std::invalid_argument("Array dependency is cyclic or forward-referenced");
          const auto& array = table.arrays[slot];
          if (array_type_(array) != type)
            throw std::invalid_argument("Array alias has a different installed type");
          return Frame::leaf(Value::distributed_array(array));
        }
        std::uint64_t arm = 0;
        if (kind == ContinuationKind::Variant) {
          arm = in.word();
          if (arm >= type.children.size())
            throw std::invalid_argument("invalid Variant tag");
        }
        FiniteDomainRef domain;
        if (kind == ContinuationKind::Array) {
          const auto slot = in.word();
          if (slot >= table.finite.size())
            throw std::invalid_argument("finite Array domain is absent");
          domain = table.finite[slot];
          continuation_detail::domain(domain, type);
        }
        const auto count = in.word();
        const auto expected = kind == ContinuationKind::Product   ? type.children.size()
                              : kind == ContinuationKind::Variant ? 1
                              : domain                            ? std::uint64_t(domain->size)
                                                                  : UINT64_MAX;
        if (count != expected || count > (in.bytes.size() - in.at) / 8)
          throw std::invalid_argument("extended continuation member count differs");

        return Frame::container(type, std::move(domain), arm, static_cast<std::size_t>(count));
      });
}
