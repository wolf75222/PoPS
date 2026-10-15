// Structural conversion preserves storage authority as well as field values.
// A weak lifetime token prevents an address reused by a later allocation from
// inheriting the owner of a destroyed immutable image or materialized copy.
private:
struct ContinuationFieldStorageAuthority {
  std::weak_ptr<const void> lifetime;
  int runtime_owner, level;
  std::uint64_t epoch, generation;
  ContinuationFieldSupport::Ref support;
  std::vector<const void*> allocations;
  bool revoked = false;
  std::optional<PreparedResourceAttempt> producer_attempt;
};
mutable std::map<const field_type*, ContinuationFieldStorageAuthority> continuation_field_storage_;

std::vector<const void*> continuation_field_allocations_(const field_type& field) const {
  std::vector<const void*> result;
  result.reserve(field.local_size());
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result.push_back(field.fab(local).storage().data());
  return result;
}

std::optional<ContinuationFieldStorageAuthority> continuation_field_storage_authority_(
    const field_type& field) const {
  if (const auto found = continuation_field_storage_.find(&field);
      found != continuation_field_storage_.end()) {
    if (found->second.lifetime.expired()) {
      continuation_field_storage_.erase(found);
    } else {
      const auto& authority = found->second;
      if (authority.producer_attempt && !authority.producer_attempt->visible())
        throw std::invalid_argument("Continuation Field producer attempt has been revoked");
      if (authority.allocations.size() != field.local_size())
        throw std::invalid_argument("Continuation Field storage replaced its exact allocation set");
      for (std::size_t local = 0; local < field.local_size(); ++local)
        if (authority.allocations[local] != field.fab(local).storage().data())
          throw std::invalid_argument("Continuation Field storage allocation was reincarnated");
      if (!authority.revoked)
        return authority;
    }
  }

  // A decoded/restarted Store already validated every Field against its exact
  // native level prototype. Match the actual immutable leaf, never its shape.
  std::optional<ContinuationFieldStorageAuthority> result;
  const auto inspect = [&](const auto& stores, std::uint64_t epoch, std::uint64_t generation) {
    for (const auto& [level, store] : stores) {
      if (!store.initialized())
        continue;
      continuation_detail::walk_value<Dim>(
          store.read(), store.configuration().byte_capacity,
          [&](const ContinuationValue<Dim>& value) {
            if (value.kind() == ContinuationKind::Field && &value.field() == &field) {
              const int candidate = sys_block(value.field_owner());
              const auto& support = value.field_support();
              const int producer = support ? support->producer_level : level;
              if (producer < 0 || producer >= nlev() ||
                  (support && (support->program_identity != facade_->installed_program_hash() ||
                               support->program_owner != value.field_owner())))
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
                                                         continuation_field_allocations_(field)};
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
    inspect(frame.sources, frame.source_epoch, frame.source_generation);
    if (result)
      return result;
  }
  inspect(amr_continuation_accepted_, runtime_->topology_epoch(),
          runtime_->materialization_generation());
  inspect(level_continuation_working_, runtime_->topology_epoch(),
          runtime_->materialization_generation());
  inspect(level_continuation_macro_bases_, runtime_->topology_epoch(),
          runtime_->materialization_generation());
  if (!result) {
    if (const auto snapshot = hierarchy_field_observation_source(field))
      result = ContinuationFieldStorageAuthority{snapshot,
                                                 -1,
                                                 snapshot->origin_level_,
                                                 snapshot->epoch_,
                                                 snapshot->generation_,
                                                 {},
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
        support->program_owner != block)
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
      if (active_level_ >= 0 && active_level_ < resource.solver->level_count() &&
          &field == &resource.solver->solution(active_level_))
        return;
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
template <class Lifetime>
ContinuationFieldMetadata continuation_copy_field_support_(
    const field_type& source, const field_type& destination, const Lifetime& lifetime,
    const ContinuationType* type, const ContinuationFieldMetadata* metadata) const {
  const std::shared_ptr<const void> retained = lifetime;
  if (!retained)
    throw std::invalid_argument("Continuation Field copy has no storage lifetime");
  int owner = -1, level = active_level_;
  auto epoch = runtime_->topology_epoch();
  auto generation = runtime_->materialization_generation();
  auto authority = continuation_field_storage_authority_(source);
  ContinuationFieldSupport::Ref support = authority ? authority->support : nullptr;
  const auto regrid = continuation_regrid_phase() ? regrid_field_qualification_(source)
                                                  : std::optional<RegridFieldQualification>{};
  if (regrid) {
    const auto& frame = require_continuation_regrid_();
    owner = sys_block(regrid->block);
    level = regrid->level;
    if (regrid->side == FieldTopology::Side::Source) {
      epoch = frame.source_epoch;
      generation = frame.source_generation;
    }
  } else if (authority) {
    owner = authority->runtime_owner;
    level = authority->level;
    epoch = authority->epoch;
    generation = authority->generation;
  } else {
    owner = scratch_prototype_owner_(source);
  }
  if (metadata) {
    if (metadata->expression)
      throw std::invalid_argument(
          "Uniform field exchange provenance cannot be rebound as an AMR flux basis");
    if (bool(metadata->support) != bool(support) || (support && *metadata->support != *support))
      throw std::invalid_argument(
          "Continuation Field metadata differs from its exact source instance");
  }
  if (type) {
    if (type->kind != ContinuationKind::Field || sys_block(type->field_owner) != owner ||
        type->field_components != source.ncomp() || type->scientific_contract.empty())
      throw std::invalid_argument(
          "Continuation Field source differs from its declared scientific type");
    if (!support)
      support = std::make_shared<const ContinuationFieldSupport>(
          ContinuationFieldSupport{facade_->installed_program_hash(), type->scientific_contract,
                                   type->field_owner, level, epoch, generation});
  }
  if (support)
    support->validate();
  continuation_bind_field_provenance(source, destination);
  for (auto it = continuation_field_storage_.begin(); it != continuation_field_storage_.end();) {
    if (it->second.lifetime.expired())
      it = continuation_field_storage_.erase(it);
    else
      ++it;
  }
  continuation_field_storage_.insert_or_assign(
      &destination,
      ContinuationFieldStorageAuthority{retained, owner, level, epoch, generation, support,
                                        continuation_field_allocations_(destination), false,
                                        authority ? authority->producer_attempt : std::nullopt});
  return {{}, std::move(support)};
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
  auto result = std::make_shared<field_type>(source);
  continuation_bind_field_provenance(source, *result, result);
  return result;
}
