// Representation metadata never supplies storage permission. Keep the existing
// level, incarnation, attempt and weak-lifetime checks before reading or changing
// the description of a privately owned Field value.
public:
std::string field_provider_read_authority(std::string_view consumer, std::size_t slot,
                                          std::uint64_t capacity) const {
  return facade_->prepared_amr_auxiliary_read_authority(consumer, active_level_, slot, capacity);
}
FieldRepresentationObligation::Ref field_representation_obligation(const field_type& field) const {
  if (const auto authority = continuation_field_storage_authority_(field))
    return authority->representation;
  // An exact native provider allocation can be read without being a persistent
  // State-owned Field. Its assembly/solution/scratch has no retained obligation;
  // it must still pass the provider's actual level and storage authority.
  require_program_field_access(field);
  return {};
}

std::string field_transform_support(const field_type& field, const ContinuationType& type,
                                    std::uint64_t workspace_byte_capacity,
                                    std::int64_t production_id = -1) const {
  require_program_field_access(field);
  const auto authority = continuation_source_field_authority_(field);
  continuation_require_field_type_(field, authority, type);
  const auto& support = authority.support;
  // Only the issuer may assign the first point. A later discharge without a
  // retained origin is rejected, rather than re-labelled at its current clock.
  std::string production_origin;
  if (!support) {
    if (production_id < 0 || production_id > std::numeric_limits<int>::max())
      throw std::invalid_argument("Field transformation has no retained production origin");
    const auto source_charge = sizeof(multiblock::BoundaryEvaluationPoint) + primary_clock_.size();
    if (source_charge > workspace_byte_capacity)
      throw std::length_error("Field production point exceeds its workspace");
    workspace_byte_capacity -= source_charge;
    production_origin = field_transform_production_origin(
        type.scientific_contract, production_id,
        boundary_evaluation_point(static_cast<int>(production_id)), workspace_byte_capacity);
    if (production_origin.size() > workspace_byte_capacity)
      throw std::length_error("Field production origin exceeds its workspace");
    workspace_byte_capacity -= production_origin.size();
  }
  const FieldTransformSupport description =
      support ? FieldTransformSupport{support->program_identity,
                                      support->origin_point,
                                      support->field_provider,
                                      support->program_owner,
                                      support->producer_level,
                                      support->topology_epoch,
                                      support->materialization_generation}
              : FieldTransformSupport{facade_->program_runtime_state_().installed_hash_,
                                      production_origin,
                                      type.field_provider,
                                      type.field_owner,
                                      authority.level,
                                      authority.epoch,
                                      authority.generation};
  return description.encode(geometry(), field.ncomp(), workspace_byte_capacity);
}

void require_discharged_field_representation(const field_type& field) const {
  if (field_representation_obligation(field))
    throw std::invalid_argument(
        "conservative Field publication requires its exact representation transformation");
}

private:
void preflight_field_representation_copy_(const field_type& source, const ContinuationType& type,
                                          std::uint64_t copy_byte_capacity) const {
  // This allowance covers the owned Field/support record. Flux-expression
  // retention continues to use the installed AMR flux-expression budget;
  // journals and closed strings are reserved by the enclosing transformation.
  const auto charge = [&](std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > copy_byte_capacity / width)
      throw std::length_error("Field representation copy exceeds its explicit byte capacity");
    copy_byte_capacity -= count * width;
  };
  charge(sizeof(field_type) + sizeof(ContinuationFieldStorageAuthority));
  charge(field_type::storage_identity_payload_bytes());
  charge(sizeof(ContinuationFieldSupport));
  charge(facade_->program_runtime_state_().installed_hash_.size());
  charge(type.scientific_contract.size());
  charge(type.field_provider.size());
  charge(source.layout().size(), 2 * sizeof(Box<Dim>) + sizeof(std::size_t));
  charge(source.distribution().owners().size(), sizeof(Index<Dim>));
  charge(source.local_size(), sizeof(typename field_type::fab_type) + sizeof(std::size_t) +
                                  sizeof(typename field_type::fab_type::storage_type::const_type));
  for (std::size_t i = 0; i < source.local_size(); ++i)
    charge(source.fab(i).size(), sizeof(Real));
}

void replace_field_representation_obligation_(
    const field_type& field, const FieldRepresentationObligation::Ref& expected,
    FieldRepresentationObligation::Ref replacement) const {
  const auto authority = continuation_source_field_authority_(field);
  const auto found = continuation_field_storage_.find(&field);
  if (found == continuation_field_storage_.end() || found->second.lifetime.expired() ||
      found->second.revoked || !found->second.allocations.matches(field) ||
      (found->second.producer_attempt && !found->second.producer_attempt->visible()) ||
      bool(authority.representation) != bool(expected) ||
      (expected && *authority.representation != *expected))
    throw std::invalid_argument("Field representation replacement has no exact live source");
  if (replacement)
    replacement->validate();
  // No allocation or publication is interleaved with this final metadata swap.
  found->second.representation = std::move(replacement);
}

public:
std::shared_ptr<field_type> own_field_representation(const field_type& source,
                                                     const ContinuationType& type,
                                                     FieldRepresentationObligation::Ref obligation,
                                                     std::uint64_t copy_byte_capacity,
                                                     std::int64_t production_id) const {
  if (!obligation)
    throw std::invalid_argument("Field representation issuance requires a closed transformation");
  obligation->validate();
  continuation_preflight_field(source, type);
  preflight_field_representation_copy_(source, type, copy_byte_capacity);
  if (field_representation_obligation(source))
    throw std::invalid_argument("Field representation issuance cannot replace a pending mapping");
  const auto existing_support = continuation_source_field_authority_(source).support;
  ContinuationFieldSupport::Ref issued_support = existing_support;
  if (!issued_support) {
    if (production_id < 0 || production_id > std::numeric_limits<int>::max())
      throw std::invalid_argument("Field production requires its compiled SSA identity");
    const auto source_charge = sizeof(multiblock::BoundaryEvaluationPoint) + primary_clock_.size();
    if (source_charge > copy_byte_capacity)
      throw std::length_error("Field production point exceeds its copy workspace");
    auto origin = field_transform_production_origin(
        type.scientific_contract, production_id,
        boundary_evaluation_point(static_cast<int>(production_id)),
        copy_byte_capacity - source_charge);
    if (origin.size() > (copy_byte_capacity - source_charge) / 2)
      throw std::length_error("Field production support exceeds its copy workspace");
    preflight_field_representation_copy_(source, type,
                                         copy_byte_capacity - source_charge - 2 * origin.size());
    issued_support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
        facade_->program_runtime_state_().installed_hash_, std::move(origin), type.field_owner,
        active_level_, runtime_->topology_epoch(), runtime_->materialization_generation(),
        type.field_provider});
  }
  auto owned = std::make_shared<field_type>(source);
  continuation_bind_field_provenance(source, *owned, owned, type);
  continuation_field_storage_.at(owned.get()).support = std::move(issued_support);
  replace_field_representation_obligation_(*owned, {}, std::move(obligation));
  return owned;
}

std::shared_ptr<field_type> own_discharged_field_representation(
    const field_type& numerical_result, const field_type& coordinate, const ContinuationType& type,
    const FieldRepresentationObligation::Ref& evaluated, std::uint64_t copy_byte_capacity) const {
  require_program_field_access(numerical_result);
  require_program_field_access(coordinate);
  continuation_preflight_field(numerical_result, type);
  continuation_preflight_field(coordinate, type);
  preflight_field_representation_copy_(numerical_result, type, copy_byte_capacity);
  const auto expected = field_representation_obligation(coordinate);
  if (!expected || !evaluated || *expected != *evaluated)
    throw std::invalid_argument("Field representation differs from the executed transformation");
  auto owned = std::make_shared<field_type>(numerical_result);
  continuation_bind_field_provenance(coordinate, *owned, owned, type);
  replace_field_representation_obligation_(*owned, expected, {});
  return owned;
}
