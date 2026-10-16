// A representation obligation is metadata of an exact owned Field instance.
// Its strings describe a generated transformation and captured inputs; they do
// not grant storage authority. All callers converge failures before publication.
public:
std::string field_provider_read_authority(std::string_view consumer, std::size_t slot,
                                          std::uint64_t capacity) const {
  return system_->prepared_auxiliary_read_authority(consumer, slot, capacity);
}
FieldRepresentationObligation::Ref field_representation_obligation(const field_type& field) const {
  const auto instance = continuation_field_instance_(field);
  return instance ? instance->metadata.representation : FieldRepresentationObligation::Ref{};
}

std::string field_transform_support(const field_type& field, const ContinuationType& type,
                                    std::uint64_t workspace_byte_capacity,
                                    std::int64_t production_id = -1) const {
  continuation_preflight_field(field, type);
  const auto instance = continuation_field_instance_(field);
  const auto support = instance ? instance->metadata.support : ContinuationFieldSupport::Ref{};
  const auto& runtime = runtime_state();
  const std::string_view program =
      !runtime.installed_hash_.empty()
          ? std::string_view(runtime.installed_hash_)
          : (runtime.continuation_.configured()
                 ? std::string_view(runtime.continuation_.configuration().identity)
                 : std::string_view{});
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
              : FieldTransformSupport{
                    program, production_origin, type.field_provider, type.field_owner, 0, 0, 0};
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
  const auto charge = [&](std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > copy_byte_capacity / width)
      throw std::length_error("Field representation copy exceeds its explicit byte capacity");
    copy_byte_capacity -= count * width;
  };
  // The caller reserves this copy allowance separately from the two journals
  // and their closed strings. Existing exchange-expression storage retains its
  // own installed budget. Charge declared storage, not allocator internals.
  charge(sizeof(field_type) + sizeof(ContinuationFieldInstance));
  charge(field_type::storage_identity_payload_bytes());
  charge(sizeof(ContinuationFieldSupport));
  const auto& runtime = runtime_state();
  const auto program_bytes = !runtime.installed_hash_.empty()
                                 ? runtime.installed_hash_.size()
                                 : (runtime.continuation_.configured()
                                        ? runtime.continuation_.configuration().identity.size()
                                        : 0);
  charge(program_bytes);
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
  const auto instance = continuation_field_instance_(field);
  const auto found = continuation_field_instances_->values.find(&field);
  if (!instance || found == continuation_field_instances_->values.end() ||
      found->second.lifetime.expired() || !found->second.sealed ||
      !found->second.allocations.matches(field) ||
      (found->second.attempt && !found->second.attempt->visible()) ||
      bool(instance->metadata.representation) != bool(expected) ||
      (expected && *instance->metadata.representation != *expected))
    throw std::invalid_argument("Field representation replacement has no exact live source");
  if (replacement)
    replacement->validate();
  // Validation above checks the exact allocation, live weak owner and attempt.
  // A shared_ptr assignment cannot fail, and no other metadata is modified.
  found->second.metadata.representation = std::move(replacement);
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
  if (type.scientific_contract.empty())
    throw std::invalid_argument("Field representation requires its exact scientific type");
  continuation_preflight_field(source, type);
  preflight_field_representation_copy_(source, type, copy_byte_capacity);
  if (field_representation_obligation(source))
    throw std::invalid_argument("Field representation issuance cannot replace a pending mapping");
  const auto existing_support =
      (continuation_field_instance_(source) ? continuation_field_instance_(source)->metadata.support
                                            : ContinuationFieldSupport::Ref{});
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
    issued_support = std::make_shared<const ContinuationFieldSupport>(
        ContinuationFieldSupport{continuation_field_program_identity_(), std::move(origin),
                                 type.field_owner, 0, 0, 0, type.field_provider});
  }
  auto owned = std::make_shared<field_type>(source);
  auto metadata = continuation_bind_field_provenance(source, *owned, owned, type);
  metadata.support = std::move(issued_support);
  metadata.representation = std::move(obligation);
  register_continuation_field_copy_(*owned, owned, std::move(metadata));
  return owned;
}

std::shared_ptr<field_type> own_discharged_field_representation(
    const field_type& numerical_result, const field_type& coordinate, const ContinuationType& type,
    const FieldRepresentationObligation::Ref& evaluated, std::uint64_t copy_byte_capacity) const {
  continuation_preflight_field(numerical_result, type);
  continuation_preflight_field(coordinate, type);
  preflight_field_representation_copy_(numerical_result, type, copy_byte_capacity);
  const auto expected = field_representation_obligation(coordinate);
  if (!expected || !evaluated || *expected != *evaluated)
    throw std::invalid_argument("Field representation differs from the executed transformation");
  // The numerical result comes from the generated equation kernel. Only its
  // metadata is inherited from the authenticated coordinate: there is no
  // second implementation of the transformation in this service.
  auto owned = std::make_shared<field_type>(numerical_result);
  continuation_bind_field_provenance(coordinate, *owned, owned, type);
  replace_field_representation_obligation_(*owned, expected, {});
  return owned;
}
