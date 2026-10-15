// Structural conversion preserves storage authority as well as field values.
// A weak lifetime token prevents an address reused by a later allocation from
// inheriting the owner of a destroyed immutable image or materialized copy.
private:
struct ContinuationFieldStorageAuthority {
  std::weak_ptr<const void> lifetime;
  int runtime_owner, level;
  std::uint64_t epoch, generation;
};
mutable std::map<const field_type*, ContinuationFieldStorageAuthority> continuation_field_storage_;

std::optional<int> continuation_field_storage_owner_(const field_type& field) const {
  if (const auto found = continuation_field_storage_.find(&field);
      found != continuation_field_storage_.end()) {
    if (found->second.lifetime.expired()) {
      continuation_field_storage_.erase(found);
    } else {
      const auto& authority = found->second;
      if (authority.level != active_level_ || authority.epoch != runtime_->topology_epoch() ||
          authority.generation != runtime_->materialization_generation())
        throw std::invalid_argument(
            "Continuation Field storage belongs to another level or generation");
      return authority.runtime_owner;
    }
  }

  // A decoded/restarted Store already validated every Field against its exact
  // native level prototype. Match the actual immutable leaf, never its shape.
  std::optional<int> owner;
  const auto inspect = [&](const auto& stores) {
    const auto found = stores.find(active_level_);
    if (found == stores.end() || !found->second.initialized())
      return;
    const auto& store = found->second;
    continuation_detail::walk_value<Dim>(
        store.read(), store.configuration().byte_capacity,
        [&](const ContinuationValue<Dim>& value) {
          if (value.kind() == ContinuationKind::Field && &value.field() == &field) {
            const int candidate = sys_block(value.field_owner());
            if (owner && *owner != candidate)
              throw std::logic_error("Continuation Field aliases different scientific owners");
            owner = candidate;
          }
          return true;
        },
        [](const ContinuationValue<Dim>&) {});
  };
  inspect(amr_continuation_accepted_);
  inspect(level_continuation_working_);
  inspect(level_continuation_macro_bases_);
  return owner;
}

public:
template <class Lifetime>
void continuation_bind_field_provenance(const field_type& source, const field_type& destination,
                                        const Lifetime& lifetime) const {
  const std::shared_ptr<const void> retained = lifetime;
  if (!retained)
    throw std::invalid_argument("Continuation Field copy has no storage lifetime");
  int owner = -1, level = active_level_;
  auto epoch = runtime_->topology_epoch();
  auto generation = runtime_->materialization_generation();
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
  } else {
    owner = scratch_prototype_owner_(source);
  }
  continuation_bind_field_provenance(source, destination);
  for (auto it = continuation_field_storage_.begin(); it != continuation_field_storage_.end();) {
    if (it->second.lifetime.expired())
      it = continuation_field_storage_.erase(it);
    else
      ++it;
  }
  continuation_field_storage_.insert_or_assign(
      &destination, ContinuationFieldStorageAuthority{retained, owner, level, epoch, generation});
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
