// Included in ContinuationCodec<Dim>. Structural checkpoint of multiple finite issuers
// and immutable distributed Array DAGs; construction remains entirely rank-local.
  struct ExtendedTable {
    std::vector<FiniteDomainRef> finite;
    std::vector<FiniteDomainRealmRef> realms;
    std::vector<distributed::DomainRef> distributed;
    std::vector<ContinuationDistributedArrayRef<Dim>> arrays; // dependencies before users
  };
  static bool requires_extended_(const Ref& root) {
    FiniteDomainRealmRef first; bool needed = false;
    auto visit = [&](auto&& self, const Ref& value) -> void {
      if (continuation_distributed_kind(value->kind())) needed = true;
      if (const auto& domain = value->domain_ref()) {
        if (!domain->realm()->replicated()) needed = true;
        if (first && first != domain->realm()) needed = true;
        else first = domain->realm();
      }
      for (const auto& child : value->members()) self(self, child);
    };
    visit(visit, root); return needed;
  }
  template<class T> static std::size_t table_slot_(const std::vector<T>& values, const T& value) {
    const auto found = std::find(values.begin(), values.end(), value);
    if (found == values.end()) throw std::invalid_argument("continuation alias table is incomplete");
    return static_cast<std::size_t>(found-values.begin());
  }
  static ContinuationType array_type_(const ContinuationDistributedArrayRef<Dim>& array) {
    ContinuationType type; type.kind = ContinuationKind::DistributedArray;
    type.family = array->domain->family(); type.capacity = array->domain->capacity();
    type.children.push_back(array->element); return type;
  }
  static void collect_extended_(const Ref& value, ExtendedTable& table, unsigned depth = 0) {
    if (depth > 128) throw std::length_error("continuation dependency graph exceeds depth bound");
    if (const auto& domain = value->domain_ref(); domain &&
        std::find(table.finite.begin(), table.finite.end(), domain) == table.finite.end()) {
      table.finite.push_back(domain);
      if (std::find(table.realms.begin(), table.realms.end(), domain->realm()) == table.realms.end())
        table.realms.push_back(domain->realm());
    }
    if (const auto& domain = value->distributed_domain_ref(); domain &&
        std::find(table.distributed.begin(), table.distributed.end(), domain) == table.distributed.end())
      table.distributed.push_back(domain);
    const auto& array = value->distributed_array_ref();
    if (array && array->index_domain &&
        std::find(table.distributed.begin(), table.distributed.end(), array->index_domain) == table.distributed.end())
      table.distributed.push_back(array->index_domain);
    if (array && std::find(table.arrays.begin(), table.arrays.end(), array) != table.arrays.end()) return;
    for (const auto& child : value->members()) collect_extended_(child, table, depth+1);
    if (array) table.arrays.push_back(array);
  }
  std::vector<std::uint8_t> encode_extended_(const Ref& root, std::uint64_t publication) const {
    ExtendedTable table; collect_extended_(root, table);
    const bool distributed_image = !table.distributed.empty() ||
        (closure_ && (!closure_->domains.empty() || !closure_->arrays.empty()));
    if (distributed_image && !registry_)
      throw std::logic_error("distributed checkpoint lacks its installed registry authority");
    if (distributed_image) {
      if (!closure_) throw std::logic_error("distributed checkpoint requires a complete collective dependency closure");
      for (const auto& array : closure_->arrays) collect_extended_(Value::distributed_array(array), table);
      // The shared closure is in native publication order; dependencies necessarily precede
      // their immutable users. All ranks retain the same version table and their own shards.
      table.distributed=closure_->domains; table.arrays=closure_->arrays;
    }
    if(closure_ && closure_->replicated_realm &&
        std::find(table.realms.begin(),table.realms.end(),closure_->replicated_realm)==table.realms.end())
      table.realms.push_back(closure_->replicated_realm);
    // Preflight the authority table before exporting identities or writing bytes.
    // One continuation has one replicated allocator, but may retain many local
    // issuers carried by different shards. Match the decoder's exact identities.
    FiniteDomainRealmRef replicated;
    for (std::size_t i = 0; i < table.realms.size(); ++i) {
      const auto& realm = table.realms[i];
      if (!realm || !realm->visible())
        throw std::invalid_argument("continuation contains a revoked finite authority");
      if (realm->replicated()) {
        if (replicated)
          throw std::invalid_argument("multiple replicated finite authorities in one continuation");
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
    continuation_detail::Writer out(capacity_);
    out.text("POPSCONT2"); out.text(identity_); out.word(Dim); out.word(publication);
    out.word(table.realms.size());
    for (const auto& realm : table.realms) {
      const auto origin = realm->transport_identity();
      out.word(origin[0]); out.word(origin[1]); out.word(realm->replicated());
    }
    out.word(table.finite.size());
    for (const auto& domain : table.finite) {
      out.word(table_slot_(table.realms, domain->realm()));
      out.word(domain->size); out.word(domain->capacity); out.word(domain->family); out.word(domain->version);
    }
    out.word(table.distributed.size());
    if (!table.distributed.empty()) {
      out.word(registry_->issuer()[0]); out.word(registry_->issuer()[1]);
      out.word(registry_->storage_bytes()); out.word(registry_->storage_versions());
      for (const auto& domain : table.distributed) {
        registry_->authenticate(domain); const auto id = domain->identity();
        out.word(id.ordinal); out.word(domain->family()); out.word(domain->capacity());
        out.word(domain->partition_offsets().size());
        for (auto offset : domain->partition_offsets()) out.word(offset);
      }
    }
    out.word(table.arrays.size());
    for (const auto& array : table.arrays) {
      const auto held = registry_->resolve_value(array->version, distributed::array_type_tag<Dim>());
      if (held.get() != array.get()) throw std::logic_error("checkpoint Array is not its published immutable version");
      out.word(array->version.ordinal);
      out.text(continuation_transport_type_identity(array_type_(array), true));
      out.word(table_slot_(table.distributed, array->domain));
      out.word(array->index_domain ? table_slot_(table.distributed,array->index_domain)+1 : 0);
      out.word(registry_->value_charge(array->version, distributed::array_type_tag<Dim>()));
      out.word(array->values.size());
      for (const auto& item : array->values) encode_extended_value_(item, table, out);
    }
    encode_extended_value_(root, table, out);
    return std::move(out.bytes);
  }
  static void encode_extended_value_(const Ref& value, const ExtendedTable& table,
                                      continuation_detail::Writer& out) {
    const auto kind = value->kind();
    if (kind == ContinuationKind::Field || kind == ContinuationKind::Boolean ||
        kind == ContinuationKind::Int64 || kind == ContinuationKind::Float64 ||
        kind == ContinuationKind::Domain || kind == ContinuationKind::Index) {
      encode_(value, table.finite, out); return;
    }
    out.word(static_cast<unsigned>(kind));
    if (kind == ContinuationKind::DistributedDomain || kind == ContinuationKind::DistributedIndex) {
      out.word(table_slot_(table.distributed, value->distributed_domain_ref()));
      if (kind == ContinuationKind::DistributedIndex)
        out.word(std::bit_cast<std::uint64_t>(value->distributed_index_offset()));
      return;
    }
    if (kind == ContinuationKind::DistributedArray) {
      out.word(table_slot_(table.arrays, value->distributed_array_ref())); return;
    }
    if (kind == ContinuationKind::Variant) out.word(value->arm());
    if (kind == ContinuationKind::Array) out.word(table_slot_(table.finite, value->domain_ref()));
    out.word(value->members().size());
    for (const auto& child : value->members()) encode_extended_value_(child, table, out);
  }
  Decoded decode_extended_(continuation_detail::Reader& in) const {
    if (in.text() != identity_ || in.word() != Dim)
      throw std::invalid_argument("extended continuation schema or dimension differs");
    Decoded out; out.publication = in.word(); ExtendedTable table;
    const auto realm_count = in.word();
    if (realm_count > (in.bytes.size()-in.at)/24) throw std::invalid_argument("truncated finite realm table");
    for (std::uint64_t i=0; i<realm_count; ++i) {
      const std::array<std::uint64_t,2> origin{in.word(),in.word()}; const auto replicated=in.word();
      if (replicated > 1) throw std::invalid_argument("invalid finite ownership tag");
      for (const auto& prior : table.realms)
        if (prior->unexported_identity() == origin) throw std::invalid_argument("duplicate finite realm origin");
      auto realm = FiniteDomainRealmRef(new FiniteDomainRealm(origin, bool(replicated)));
      if (replicated) {
        if (out.realm) throw std::invalid_argument("multiple replicated finite authorities in one continuation");
        out.realm = realm;
      }
      table.realms.push_back(std::move(realm));
    }
    const auto finite_count = in.word();
    if (finite_count > (in.bytes.size()-in.at)/40) throw std::invalid_argument("truncated finite domain table");
    for (std::uint64_t i=0; i<finite_count; ++i) {
      const auto realm=in.word(); const auto size=signed_word_(in),cap=signed_word_(in),family=signed_word_(in),version=signed_word_(in);
      if (realm>=table.realms.size() || size<0 || cap<size || family<0 || version<=0)
        throw std::invalid_argument("invalid finite domain descriptor");
      for (const auto& prior : table.finite)
        if (prior->realm()==table.realms[realm] && prior->version==version)
          throw std::invalid_argument("duplicate finite domain version");
      table.finite.push_back(FiniteDomainRef(new FiniteDomain(size,cap,family,version,table.realms[realm])));
      if (table.realms[realm] == out.realm) out.max_domain_version=std::max(out.max_domain_version,version);
    }
    const auto distributed_count = in.word();
    if (distributed_count > (in.bytes.size()-in.at)/40)
      throw std::invalid_argument("truncated distributed domain table");
    if (distributed_count) {
      if (!lane_) throw std::logic_error("distributed restore requires an installed execution lane");
      const std::array<std::uint64_t,2> issuer{in.word(),in.word()};
      const auto bytes=in.word(), versions=in.word();
      out.distributed_registry=distributed::Registry::restore_local(*lane_,issuer,bytes,versions);
      for (std::uint64_t i=0; i<distributed_count; ++i) {
        const auto ordinal=in.word(); const auto family=signed_word_(in),cap=signed_word_(in); const auto n=in.word();
        if (n!=std::size_t(lane_->size()+1) || n>(in.bytes.size()-in.at)/8)
          throw std::invalid_argument("checkpoint ownership/rank count differs from installed lane");
        std::vector<std::int64_t> offsets; offsets.reserve(static_cast<std::size_t>(n));
        for(std::uint64_t j=0;j<n;++j) offsets.push_back(signed_word_(in));
        table.distributed.push_back(out.distributed_registry->restore_domain_local(
            distributed::Identity{issuer,ordinal},family,cap,offsets));
      }
    }
    std::map<std::string, ContinuationType> array_types;
    auto collect_types = [&](auto&& self,const ContinuationType& type)->void {
      if (type.kind==ContinuationKind::DistributedArray)
        array_types.emplace(continuation_transport_type_identity(type,true),type);
      for(const auto& child:type.children) self(self,child);
    }; collect_types(collect_types,type_);
    const auto arrays=in.word();
    if (arrays>(in.bytes.size()-in.at)/40 || (arrays && !out.distributed_registry))
      throw std::invalid_argument("truncated or authority-free distributed Array table");
    for(std::uint64_t i=0;i<arrays;++i) {
      const auto ordinal=in.word(); const auto type_id=in.text(); const auto slot=in.word(),target=in.word(),charge=in.word(),count=in.word();
      const auto found=array_types.find(type_id);
      if(found==array_types.end() || slot>=table.distributed.size())
        throw std::invalid_argument("Array table type/domain absent from installed schema");
      const auto& type=found->second; const auto& domain=table.distributed[slot];
      if(domain->family()!=type.family || domain->capacity()!=type.capacity ||
          count!=std::size_t(domain->local_size()) || count>(in.bytes.size()-in.at)/8 || charge<32)
        throw std::invalid_argument("Array table has inconsistent ownership, type or charge");
      auto array=std::make_shared<ContinuationDistributedArray<Dim>>();
      array->domain=domain; array->element=type.children[0];
      if(target) {
        if(target-1>=table.distributed.size()) throw std::invalid_argument("Array Index target witness is absent");
        array->index_domain=table.distributed[target-1];
      }
      array->version=distributed::Identity{out.distributed_registry->issuer(),ordinal};
      array->values.reserve(static_cast<std::size_t>(count));
      const auto begin=in.at;
      for(std::uint64_t j=0;j<count;++j) array->values.push_back(decode_extended_value_(array->element,table,in));
      if(in.at-begin>charge-32) throw std::invalid_argument("Array checkpoint understates its retained allocation charge");
      const auto actual_charge=continuation_distributed_array_charge<Dim>(array,*out.distributed_registry,capacity_);
      if(charge!=actual_charge)
        throw std::invalid_argument("Array checkpoint retained charge differs from the native allocation policy");
      out.distributed_registry->restore_value_local(array->version,array,distributed::array_type_tag<Dim>(),charge);
      table.arrays.push_back(std::move(array));
    }
    out.root=decode_extended_value_(type_,table,in);
    if(in.at!=in.bytes.size()) throw std::invalid_argument("extended continuation image has trailing bytes");
    validate(out.root); ExtendedTable used; collect_extended_(out.root,used);
    for(const auto& array:table.arrays) {
      validate_(array_type_(array),Value::distributed_array(array));
      collect_extended_(Value::distributed_array(array),used);
    }
    if(out.realm && std::find(used.realms.begin(),used.realms.end(),out.realm)==used.realms.end()) used.realms.push_back(out.realm);
    if(used.finite.size()!=table.finite.size() || used.realms.size()!=table.realms.size())
      throw std::invalid_argument("checkpoint tables contain unreachable authorities or values");
    // Exact global reachability is checked collectively by the restart transaction before
    // publication; a rank-local shard alone cannot establish or refute it.
    auto closure=std::make_shared<ContinuationClosure<Dim>>();
    closure->domains=table.distributed;closure->arrays=table.arrays;closure->replicated_realm=out.realm;out.closure=std::move(closure);
    out.realms=std::move(table.realms); return out;
  }
  Ref decode_extended_value_(const ContinuationType& type,const ExtendedTable& table,
                            continuation_detail::Reader& in) const {
    const auto kind=type.kind;
    if(kind==ContinuationKind::Field || kind==ContinuationKind::Boolean || kind==ContinuationKind::Int64 ||
       kind==ContinuationKind::Float64 || kind==ContinuationKind::Domain || kind==ContinuationKind::Index)
      return decode_(type,table.finite,in);
    if(in.word()!=static_cast<unsigned>(kind)) throw std::invalid_argument("extended continuation wire kind differs");
    if(kind==ContinuationKind::DistributedDomain || kind==ContinuationKind::DistributedIndex) {
      const auto slot=in.word(); if(slot>=table.distributed.size()) throw std::invalid_argument("distributed domain alias is absent");
      const auto& domain=table.distributed[slot];
      if(domain->family()!=type.family || domain->capacity()!=type.capacity)
        throw std::invalid_argument("distributed domain alias changes family");
      if(kind==ContinuationKind::DistributedDomain) return Value::distributed_domain(domain);
      return Value::distributed_index(domain,signed_word_(in));
    }
    if(kind==ContinuationKind::DistributedArray) {
      const auto slot=in.word(); if(slot>=table.arrays.size()) throw std::invalid_argument("Array dependency is cyclic or forward-referenced");
      const auto& array=table.arrays[slot];
      if(array_type_(array)!=type) throw std::invalid_argument("Array alias has a different installed type");
      return Value::distributed_array(array);
    }
    std::uint64_t arm=0;
    if(kind==ContinuationKind::Variant) {arm=in.word();if(arm>=type.children.size()) throw std::invalid_argument("invalid Variant tag");}
    FiniteDomainRef domain;
    if(kind==ContinuationKind::Array) {
      const auto slot=in.word();if(slot>=table.finite.size()) throw std::invalid_argument("finite Array domain is absent");
      domain=table.finite[slot];continuation_detail::domain(domain,type);
    }
    const auto count=in.word();
    const auto expected=kind==ContinuationKind::Product?type.children.size():
        kind==ContinuationKind::Variant?1:domain?std::uint64_t(domain->size):UINT64_MAX;
    if(count!=expected || count>(in.bytes.size()-in.at)/8) throw std::invalid_argument("extended continuation member count differs");
    std::vector<Ref> values;values.reserve(static_cast<std::size_t>(count));
    for(std::size_t i=0;i<count;++i) values.push_back(decode_extended_value_(
        type.children[kind==ContinuationKind::Product?i:kind==ContinuationKind::Variant?arm:0],table,in));
    if(kind==ContinuationKind::Product) return Value::product(std::move(values));
    if(kind==ContinuationKind::Variant) return Value::variant(static_cast<std::uint32_t>(arm),std::move(values[0]));
    return Value::array(std::move(domain),std::move(values));
  }
