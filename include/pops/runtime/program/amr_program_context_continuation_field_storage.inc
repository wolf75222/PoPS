// Structural conversion preserves storage authority as well as field values.
// A weak lifetime token prevents an address reused by a later allocation from
// inheriting the owner of a destroyed immutable image or materialized copy.
private:
struct ContinuationFieldStorageAuthority {
  std::weak_ptr<const void> lifetime;
  int runtime_owner, level;
  std::uint64_t epoch, generation;
  ContinuationFieldSupport::Ref support;
  ContinuationFieldAllocations allocations;
  bool revoked = false;
  std::optional<PreparedResourceAttempt> producer_attempt;
  FieldRepresentationObligation::Ref representation;
};
mutable std::map<const field_type*, ContinuationFieldStorageAuthority> continuation_field_storage_;

ContinuationFieldAllocations continuation_field_allocations_(const field_type& field) const {
  ContinuationFieldAllocations result{field.storage_identity(), {}};
  result.local.reserve(field.local_size());
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result.local.push_back(field.fab(local).storage());
  return result;
}

#include <pops/runtime/program/amr_program_context_history_deferred_authority.inc>

std::optional<ContinuationFieldStorageAuthority> continuation_field_storage_authority_(
    const field_type& field) const {
  const auto& manager = runtime_state().hist_;
  for (const auto& [key, ring] : manager.histories)
    for (std::size_t slot = 0; slot < ring.size(); ++slot)
      if (&ring[slot] == &field) {
        const auto decoded = decode_history_key_(key);
        if (!decoded)
          throw std::invalid_argument("retained History Field has no exact level key");
        HistoryFieldMetadata metadata;
        if (const auto entry = manager.slot_fields.find(key); entry != manager.slot_fields.end())
          metadata = entry->second.at(slot);
        const auto& support = metadata.support;
        if (support) {
          support->validate();
          if (support->program_identity != facade_->installed_program_hash())
            throw std::invalid_argument("retained History Field has a foreign Program support");
        }
        const int owner =
            support ? (support->field_provider.empty() ? sys_block(support->program_owner) : -1)
                    : manager.owner.at(key);
        return ContinuationFieldStorageAuthority{field.storage_identity(),
                                                 owner,
                                                 support ? support->producer_level : decoded->first,
                                                 runtime_->topology_epoch(),
                                                 runtime_->materialization_generation(),
                                                 support,
                                                 continuation_field_allocations_(field),
                                                 false,
                                                 std::nullopt,
                                                 metadata.representation};
      }
  if (auto cached = deferred_history_field_authority_(field))
    return cached;
  if (const auto found = continuation_field_storage_.find(&field);
      found != continuation_field_storage_.end()) {
    if (found->second.lifetime.expired()) {
      continuation_field_storage_.erase(found);
    } else {
      const auto& authority = found->second;

      if (!authority.allocations.matches(field))
        throw std::invalid_argument("Continuation Field storage allocation was reincarnated");
      // A successfully retained immutable Store may now own this exact leaf.
      // Only that fallback can survive expiration of its producing attempt.
      if (!authority.revoked &&
          (!authority.producer_attempt || authority.producer_attempt->visible()))
        return authority;
    }
  }

  // A decoded/restarted Store already validated every Field against its exact
  // native level prototype. Match the actual immutable leaf, never its shape.
  std::optional<ContinuationFieldStorageAuthority> result;
  const auto inspect = [&](const auto& stores, std::uint64_t epoch, std::uint64_t generation,
                           int producer_levels) {
    for (const auto& [level, store] : stores) {
      if (!store.initialized())
        continue;
      continuation_detail::walk_value<Dim>(
          store.read(), store.configuration().byte_capacity,
          [&](const ContinuationValue<Dim>& value) {
            if (value.kind() == ContinuationKind::Field && &value.field() == &field) {
              const int candidate =
                  value.field_provider().empty() ? sys_block(value.field_owner()) : -1;
              const auto& support = value.field_support();
              const int producer = support ? support->producer_level : level;
              if (producer < 0 || producer >= producer_levels ||
                  (support && (support->program_identity != facade_->installed_program_hash() ||
                               support->program_owner != value.field_owner() ||
                               support->field_provider != value.field_provider())))
                throw std::invalid_argument("Continuation Field has a foreign producer support");
              if (result && (result->runtime_owner != candidate || result->level != producer ||
                             bool(result->support) != bool(support) ||
                             (support && *result->support != *support)))
                throw std::logic_error("Continuation Field aliases different scientific owners");
              result = ContinuationFieldStorageAuthority{value.field_storage_owner(),
                                                         candidate,
                                                         producer,
                                                         epoch,
                                                         generation,
                                                         support,
                                                         continuation_field_allocations_(field),
                                                         false,
                                                         std::nullopt,
                                                         value.field_representation()};
            }
            return true;
          },
          [](const ContinuationValue<Dim>&) {});
    }
  };
  // The source forest retains its original storage authority while target
  // geometry is being built. Presence in an accepted Store is not a remap.
  if (continuation_regrid_ && continuation_regrid_->attempt.visible()) {
    const auto& frame = *continuation_regrid_;
    inspect(frame.sources, frame.source_epoch, frame.source_generation,
            static_cast<int>(frame.sources.size()) - 1);
    if (result)
      return result;
  }
  inspect(amr_continuation_accepted_, runtime_->topology_epoch(),
          runtime_->materialization_generation(), nlev());
  inspect(level_continuation_working_, runtime_->topology_epoch(),
          runtime_->materialization_generation(), nlev());
  inspect(level_continuation_macro_bases_, runtime_->topology_epoch(),
          runtime_->materialization_generation(), nlev());
  if (!result) {
    if (const auto snapshot = hierarchy_field_observation_source(field))
      result = ContinuationFieldStorageAuthority{snapshot,
                                                 -1,
                                                 snapshot->origin_level_,
                                                 snapshot->epoch_,
                                                 snapshot->generation_,
                                                 snapshot->support_,
                                                 continuation_field_allocations_(field),
                                                 false,
                                                 snapshot->attempt_};
  }
  if (!result && continuation_field_storage_.contains(&field))
    throw std::invalid_argument("Continuation Field instance has been revoked");
  return result;
}

void revoke_continuation_field_copies_() const noexcept {
  for (auto it = continuation_field_storage_.begin(); it != continuation_field_storage_.end();) {
    if (it->second.lifetime.expired())
      it = continuation_field_storage_.erase(it);
    else {
      it->second.revoked = true;
      ++it;
    }
  }
}

std::optional<int> continuation_field_storage_owner_(const field_type& field) const {
  const auto authority = continuation_field_storage_authority_(field);
  if (!authority || authority->runtime_owner < 0)
    return std::nullopt;
  require_program_field_access(field);
  return authority->runtime_owner;
}

public:
const field_type& continuation_field_prototype_(
    int block, int scope_level, const ContinuationFieldSupport::Ref& support) const {
  int level = scope_level;
  if (support) {
    support->validate();
    if (support->program_identity != facade_->installed_program_hash() ||
        support->program_owner != block || !support->field_provider.empty())
      throw std::invalid_argument("Continuation Field prototype has a foreign scientific support");
    level = support->producer_level;
  }
  if (level < 0 || level >= nlev())
    throw std::invalid_argument("Continuation Field prototype requires an exact producer level");
  return facade_->prepared_amr_block_state(sys_block(block), level);
}

// This is a read permission, independent of the persisted scientific origin.
// An old temporal sample on the same live support is valid; another level or a
// superseded storage generation is not implicitly transferred by shape equality.
void require_program_field_access(const field_type& field) const {
  const auto authority = continuation_field_storage_authority_(field);
  if (!authority) {
    if (recognizes_hierarchy_field_observation(field))
      return;
    // Independent prepared fields are qualified by their actual provider
    // allocation, never by a matching shape or component count.
    for (const auto& [key, resource] : hierarchy_field_resources_) {
      if (!resource.solver || resource.topology_epoch != runtime_->topology_epoch() ||
          resource.generation != runtime_->materialization_generation())
        continue;
      if (active_level_ >= 0 && active_level_ < resource.solver->level_count()) {
        if (&field == &resource.solver->solution(active_level_))
          return;
        // An assembled RHS/coefficient is owned by its declared provider slot,
        // independently of any State. Match its actual allocation at this level.
        for (const auto& slot : resource.selection.assembly_field_slots)
          if (&field == &resource.solver->assembly_target(slot, active_level_))
            return;
      }
      for (const auto& [slot, scratch] : resource.scratches)
        if (std::get<0>(slot) == active_level_ && &field == &scratch)
          return;
    }
    (void)scratch_prototype_owner_(field, false);
    return;
  }
  if (authority->level != active_level_ || authority->epoch != runtime_->topology_epoch() ||
      authority->generation != runtime_->materialization_generation())
    throw std::invalid_argument(
        "Continuation Field numerical read requires its exact producer support or an explicit "
        "transfer");
}

private:
ContinuationFieldStorageAuthority continuation_source_field_authority_(
    const field_type& source) const {
  if (continuation_regrid_phase()) {
    if (const auto qualified = regrid_field_qualification_(source)) {
      const auto& frame = require_continuation_regrid_();
      const bool old = qualified->side == FieldTopology::Side::Source;
      auto authority = continuation_field_storage_authority_(source);
      return {{},
              qualified->provider.empty() ? sys_block(qualified->block) : -1,
              qualified->level,
              old ? frame.source_epoch : runtime_->topology_epoch(),
              old ? frame.source_generation : runtime_->materialization_generation(),
              qualified->support ? qualified->support : (authority ? authority->support : nullptr),
              continuation_field_allocations_(source),
              false,
              std::nullopt,
              qualified->representation};
    }
  }
  if (auto authority = continuation_field_storage_authority_(source))
    return *authority;
  return {{},
          scratch_prototype_owner_(source),
          active_level_,
          runtime_->topology_epoch(),
          runtime_->materialization_generation(),
          {},
          continuation_field_allocations_(source)};
}

void continuation_require_field_type_(const field_type& source,
                                      const ContinuationFieldStorageAuthority& authority,
                                      const ContinuationType& type) const {
  if (type.kind != ContinuationKind::Field || type.field_components != source.ncomp() ||
      type.scientific_contract.empty())
    throw std::invalid_argument(
        "Continuation Field source differs from its declared scientific type");
  if (type.field_provider.empty()) {
    if (type.field_owner < 0 || sys_block(type.field_owner) != authority.runtime_owner ||
        (authority.support && (!authority.support->field_provider.empty() ||
                               authority.support->program_owner != type.field_owner)))
      throw std::invalid_argument("Continuation Field source has a foreign State owner");
  } else {
    const auto& provider = continuation_field_provider_(type.field_provider);
    if (type.field_owner != -1 || authority.runtime_owner != -1 || !authority.support ||
        authority.support->field_provider != type.field_provider ||
        authority.support->program_owner != -1 || provider.components != source.ncomp())
      throw std::invalid_argument("Continuation Field source has foreign FieldProblem authority");
  }
  if (authority.support) {
    authority.support->validate();
    if (authority.support->program_identity != facade_->installed_program_hash() ||
        authority.support->producer_level != authority.level)
      throw std::invalid_argument("Continuation Field has foreign program or producer authority");
  }
}

void continuation_require_field_metadata_(const ContinuationFieldStorageAuthority& authority,
                                          const ContinuationFieldMetadata& metadata) const {
  if (metadata.expression)
    throw std::invalid_argument(
        "Uniform field exchange provenance cannot be rebound as an AMR flux basis");
  if (bool(metadata.support) != bool(authority.support) ||
      (authority.support && *metadata.support != *authority.support) ||
      bool(metadata.representation) != bool(authority.representation) ||
      (authority.representation && *metadata.representation != *authority.representation))
    throw std::invalid_argument(
        "Continuation Field metadata differs from its exact source instance");
}

public:
void continuation_preflight_field(const field_type& source, const ContinuationType& type) const {
  continuation_require_field_type_(source, continuation_source_field_authority_(source), type);
}
void continuation_preflight_field(const field_type& source, const ContinuationType& type,
                                  const ContinuationFieldMetadata& metadata) const {
  const auto authority = continuation_source_field_authority_(source);
  continuation_require_field_type_(source, authority, type);
  continuation_require_field_metadata_(authority, metadata);
}

private:
template <class Lifetime>
ContinuationFieldMetadata continuation_copy_field_support_(
    const field_type& source, const field_type& destination, const Lifetime& lifetime,
    const ContinuationType* type, const ContinuationFieldMetadata* metadata) const {
  const std::shared_ptr<const void> retained = lifetime;
  if (!retained)
    throw std::invalid_argument("Continuation Field copy has no storage lifetime");
  auto authority = continuation_source_field_authority_(source);
  if (metadata)
    continuation_require_field_metadata_(authority, *metadata);
  if (type) {
    continuation_require_field_type_(source, authority, *type);
    if (!authority.support)
      authority.support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
          facade_->installed_program_hash(), type->scientific_contract, type->field_owner,
          authority.level, authority.epoch, authority.generation, type->field_provider});
  }
  if (authority.support)
    authority.support->validate();
  continuation_bind_field_provenance(source, destination);
  for (auto it = continuation_field_storage_.begin(); it != continuation_field_storage_.end();) {
    if (it->second.lifetime.expired())
      it = continuation_field_storage_.erase(it);
    else
      ++it;
  }
  authority.lifetime = retained;
  authority.allocations = continuation_field_allocations_(destination);
  authority.revoked = false;
  continuation_field_storage_.insert_or_assign(&destination, authority);
  return {{}, authority.support, authority.representation};
}

public:
template <class Lifetime>
ContinuationFieldMetadata continuation_bind_field_provenance(const field_type& source,
                                                             const field_type& destination,
                                                             const Lifetime& lifetime) const {
  return continuation_copy_field_support_(source, destination, lifetime, nullptr, nullptr);
}
template <class Lifetime>
ContinuationFieldMetadata continuation_bind_field_provenance(const field_type& source,
                                                             const field_type& destination,
                                                             const Lifetime& lifetime,
                                                             const ContinuationType& type) const {
  return continuation_copy_field_support_(source, destination, lifetime, &type, nullptr);
}
template <class Lifetime>
void continuation_bind_field_provenance(const field_type& source, const field_type& destination,
                                        const Lifetime& lifetime,
                                        const ContinuationFieldMetadata& metadata) const {
  (void)continuation_copy_field_support_(source, destination, lifetime, nullptr, &metadata);
}

template <class Lifetime>
void continuation_bind_field_provenance(
    const field_type& source, const field_type& destination, const Lifetime& lifetime,
    const FieldExchangeExpressions::Ref& uniform_expression) const {
  if (uniform_expression)
    throw std::invalid_argument(
        "Uniform field exchange provenance cannot be rebound as an AMR flux basis");
  continuation_bind_field_provenance(source, destination, lifetime);
}

// This adapter moves no mathematical point: it only owns a structural copy.
// Numerical use remains subject to require_program_field_access at the consumer.
std::shared_ptr<field_type> own_structural_field(const field_type& source) const {
  (void)continuation_source_field_authority_(source);
  auto result = std::make_shared<field_type>(source);
  continuation_bind_field_provenance(source, *result, result);
  return result;
}
