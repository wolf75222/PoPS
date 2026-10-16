// Independent Uniform Field producers. Scientific support is persisted; live access
// requires the exact allocation and lifetime issued by this context or its accepted Store.
private:
struct ContinuationFieldAllocations {
  using local_storage_type = typename field_type::fab_type::storage_type::const_type;
  std::shared_ptr<const void> incarnation;
  // Pin the exact allocations, not only their addresses: a mutable Fab can be
  // replaced independently of its MultiFab and must not regain authority by ABA.
  std::vector<local_storage_type> local;
  bool matches(const field_type& field) const noexcept {
    if (incarnation != field.storage_identity() || local.size() != field.local_size())
      return false;
    for (std::size_t i = 0; i < local.size(); ++i)
      if (local[i].data() != field.fab(i).storage().data())
        return false;
    return true;
  }
};
struct ContinuationFieldInstance {
  std::weak_ptr<const void> lifetime;
  ContinuationFieldAllocations allocations;
  ContinuationFieldMetadata metadata;
  std::optional<PreparedResourceAttempt> attempt;
  bool sealed = false;
};
struct ContinuationFieldInstances {
  std::map<const field_type*, ContinuationFieldInstance> values;
};
std::shared_ptr<ContinuationFieldInstances> continuation_field_instances_ =
    std::make_shared<ContinuationFieldInstances>();
using ContinuationFieldProviders = std::map<std::string, std::shared_ptr<field_type>, std::less<>>;
mutable ContinuationFieldProviders continuation_field_providers_;

static ContinuationFieldAllocations continuation_field_allocations_(const field_type& value) {
  ContinuationFieldAllocations allocations{value.storage_identity(), {}};
  allocations.local.reserve(value.local_size());
  for (std::size_t i = 0; i < value.local_size(); ++i)
    allocations.local.push_back(value.fab(i).storage());
  return allocations;
}
static std::string continuation_field_program_identity_(runtime_type* system) {
  auto identity = system->installed_program_hash();
  const auto& store = system->program_runtime_state_().continuation_;
  if (identity.empty() && store.configured())
    identity = store.configuration().identity;
  if (identity.empty())
    throw std::logic_error("Field observation requires an installed Program identity");
  return identity;
}
std::string continuation_field_program_identity_() const {
  return continuation_field_program_identity_(system_);
}
const field_type& continuation_field_provider_(std::string_view contract) const {
  const auto found = continuation_field_providers_.find(contract);
  if (found == continuation_field_providers_.end())
    throw std::invalid_argument("Continuation Field names an uninstalled independent provider");
  return *found->second;
}
static void require_continuation_field_support_(runtime_type* system, const ContinuationType& type,
                                                const ContinuationFieldSupport::Ref& support) {
  if (type.kind != ContinuationKind::Field ||
      !((type.field_owner >= 0 && type.field_provider.empty()) ||
        (type.field_owner == -1 && !type.field_provider.empty())))
    throw std::invalid_argument("Continuation Field has an invalid scientific owner");
  if (support) {
    support->validate();
    if (support->program_identity != continuation_field_program_identity_(system) ||
        support->producer_level != 0 || support->topology_epoch != 0 ||
        support->materialization_generation != 0 || support->program_owner != type.field_owner ||
        support->field_provider != type.field_provider)
      throw std::invalid_argument(
          "Uniform Field restoration requires an explicit support transfer");
  } else if (!type.field_provider.empty()) {
    throw std::invalid_argument("Independent Field has no authenticated producer support");
  }
}
static const field_type& continuation_field_prototype_(
    runtime_type* system, const ContinuationFieldProviders& providers, const ContinuationType& type,
    const ContinuationFieldSupport::Ref& support) {
  require_continuation_field_support_(system, type, support);
  if (type.field_provider.empty()) {
    const auto& map = system->program_block_map();
    return system->block_state(map.at(static_cast<std::size_t>(type.field_owner)));
  }
  const auto found = providers.find(type.field_provider);
  if (found == providers.end())
    throw std::invalid_argument("Continuation Field names an uninstalled independent provider");
  const auto& prototype = *found->second;
  if (prototype.ncomp() != type.field_components)
    throw std::invalid_argument("Continuation Field provider changes its component schema");
  return prototype;
}
const field_type& continuation_field_prototype_(
    const ContinuationType& type, const ContinuationFieldSupport::Ref& support) const {
  return continuation_field_prototype_(system_, continuation_field_providers_, type, support);
}
std::optional<ContinuationFieldInstance> continuation_field_instance_(
    const field_type& value) const {
  // History owns the current slot incarnation. Its empty metadata is meaningful:
  // never inherit a prior producer merely because a ring reuses an address.
  const auto& histories = runtime_state().hist_;
  for (const auto& [name, ring] : histories.histories)
    for (std::size_t slot = 0; slot < ring.size(); ++slot)
      if (&ring[slot] == &value) {
        ContinuationFieldMetadata metadata;
        if (const auto fields = histories.slot_fields.find(name);
            fields != histories.slot_fields.end()) {
          metadata.support = fields->second.at(slot).support;
          metadata.representation = fields->second.at(slot).representation;
        }
        if (metadata.support) {
          metadata.support->validate();
          if (metadata.support->program_identity != continuation_field_program_identity_())
            throw std::invalid_argument("retained History Field has a foreign Program support");
        }
        if (const auto exchanges = histories.slot_exchanges.find(name);
            exchanges != histories.slot_exchanges.end())
          metadata.expression = exchanges->second.at(slot);
        return ContinuationFieldInstance{value.storage_identity(),
                                         continuation_field_allocations_(value),
                                         std::move(metadata),
                                         {},
                                         true};
      }
  auto& values = continuation_field_instances_->values;
  bool revoked = false;
  if (const auto found = values.find(&value); found != values.end()) {
    if (found->second.lifetime.expired()) {
      values.erase(found);
    } else {
      const auto& instance = found->second;
      if (!instance.sealed || !instance.allocations.matches(value))
        throw std::invalid_argument("Continuation Field allocation is unsealed or replaced");
      revoked = instance.attempt && !instance.attempt->visible();
      if (!revoked)
        return instance;
    }
  }
  std::optional<ContinuationFieldInstance> result;
  const auto& store = runtime_state().continuation_;
  if (store.initialized())
    continuation_detail::walk_value<Dim>(
        store.read(), store.configuration().byte_capacity,
        [&](const ContinuationValue<Dim>& leaf) {
          if (leaf.kind() == ContinuationKind::Field && &leaf.field() == &value) {
            ContinuationType type(ContinuationKind::Field);
            type.field_owner = leaf.field_owner();
            type.field_components = value.ncomp();
            type.field_provider = leaf.field_provider();
            (void)continuation_field_prototype_(type, leaf.field_support());
            result = ContinuationFieldInstance{leaf.field_storage_owner(),
                                               continuation_field_allocations_(value),
                                               leaf.field_metadata(),
                                               {},
                                               true};
          }
          return true;
        },
        [](const ContinuationValue<Dim>&) {});
  if (!result && revoked)
    throw std::invalid_argument("Continuation Field producer attempt has been revoked");
  return result;
}
template <class Lifetime>
void register_continuation_field_copy_(const field_type& destination, const Lifetime& lifetime,
                                       ContinuationFieldMetadata metadata) const {
  if (!metadata.support && !metadata.representation)
    return;
  auto attempt = resource_attempt();
  continuation_field_instances_->values.insert_or_assign(
      &destination, ContinuationFieldInstance{
                        lifetime, continuation_field_allocations_(destination), std::move(metadata),
                        attempt.visible() ? std::optional(attempt) : std::nullopt, true});
}

public:
void configure_continuation_field_provider(std::string_view contract, int components,
                                           int ghost_depth) const {
  if (contract.empty() || components <= 0 || ghost_depth < 0)
    throw std::invalid_argument("Independent Field provider declaration is incomplete");
  if (const auto found = continuation_field_providers_.find(contract);
      found != continuation_field_providers_.end()) {
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    if (found->second->ncomp() != components || found->second->ghosts() != ghosts)
      throw std::invalid_argument("Independent Field provider declaration changed");
    return;
  }
  // Uniform geometry is shared; this allocation borrows its grid, never its State ownership.
  auto prototype = std::make_shared<field_type>(alloc_scalar_field(components, ghost_depth));
  continuation_field_providers_.emplace(std::string(contract), std::move(prototype));
}
std::shared_ptr<field_type> allocate_continuation_field_observation(
    std::string_view contract, std::uint64_t byte_capacity) const {
  const auto& prototype = continuation_field_provider_(contract);
  const auto attempt = resource_attempt();
  if (!attempt.visible())
    throw std::logic_error("Independent Field observation requires an active Program attempt");
  const auto program = continuation_field_program_identity_();
  std::uint64_t remaining = byte_capacity;
  const auto charge = [&](std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > remaining / width)
      throw std::length_error("Independent Field observation exceeds its byte capacity");
    remaining -= count * width;
  };
  // Charge declared storage and metadata before any candidate allocation. As with
  // ContinuationCodec budgets this excludes allocator-specific bookkeeping.
  charge(sizeof(field_type));
  charge(field_type::storage_identity_payload_bytes());
  charge(sizeof(std::pair<const field_type* const, ContinuationFieldInstance>));
  // Pending and sealed support coexist during the failure-atomic seal prepare.
  charge(2, sizeof(ContinuationFieldSupport));
  charge(2, contract.size());
  charge(2, program.size());
  charge(sizeof("pending") - 1);
  charge(sizeof("pops.uniform-field-observation.point@1:sha256:") - 1 + 64);
  // MultiFab owns both its layout and Distribution's layout, an owner table,
  // global/local index maps, Fab objects and the registry's allocation addresses.
  charge(prototype.layout().size(), 2 * sizeof(Box<Dim>) + sizeof(std::size_t));
  charge(prototype.distribution().owners().size(), sizeof(Index<Dim>));
  charge(prototype.local_size(),
         sizeof(typename field_type::fab_type) + sizeof(std::size_t) +
             sizeof(typename ContinuationFieldAllocations::local_storage_type));
  for (std::size_t i = 0; i < prototype.local_size(); ++i) {
    charge(prototype.fab(i).size(), sizeof(Real));
  }
  auto& instances = continuation_field_instances_->values;
  for (auto it = instances.begin(); it != instances.end();)
    if (it->second.lifetime.expired())
      it = instances.erase(it);
    else
      ++it;
  auto result =
      std::make_shared<field_type>(prototype.layout(), prototype.distribution(),
                                   prototype.local_rank(), prototype.ncomp(), prototype.ghosts());
  auto support = std::make_shared<const ContinuationFieldSupport>(
      ContinuationFieldSupport{program, "pending", -1, 0, 0, 0, std::string(contract)});
  continuation_field_instances_->values.insert_or_assign(
      result.get(),
      ContinuationFieldInstance{result,
                                continuation_field_allocations_(*result),
                                {{}, std::move(support), {}},
                                attempt.visible() ? std::optional(attempt) : std::nullopt,
                                false});
  return result;
}
void seal_continuation_field_observation(std::string_view contract, std::int64_t value_id,
                                         const std::shared_ptr<field_type>& value) const {
  auto prepared = continuation_local_phase(
      [&] {
        if (!value || value_id < 0)
          throw std::invalid_argument("Independent Field observation has an invalid identity");
        auto& entries = continuation_field_instances_->values;
        const auto found = entries.find(value.get());
        if (found == entries.end() || found->second.lifetime.expired() || found->second.sealed ||
            !found->second.metadata.support ||
            found->second.metadata.support->field_provider != contract ||
            !found->second.allocations.matches(*value) || !found->second.attempt ||
            !found->second.attempt->visible())
          throw std::invalid_argument("Independent Field observation is not its issued allocation");
        require_same_field_contract_(*value, continuation_field_provider_(contract),
                                     "Field observation");
        // Persistent scientific identity must survive restart. The attempt's
        // private revocation capability belongs only to the live registry above.
        ExactContractBuilder point;
        point.text("pops.uniform-field-observation.point@1")
            .text(primary_clock_)
            .scalar(value_id)
            .scalar(static_cast<std::int64_t>(macro_step()))
            .scalar(physical_time() + logical_physical_time_offset_)
            .scalar(current_dt_)
            .scalar(logical_phase_begin_.numerator)
            .scalar(logical_phase_begin_.denominator)
            .scalar(logical_phase_span_.numerator)
            .scalar(logical_phase_span_.denominator)
            .scalar(stage_time_.numerator)
            .scalar(stage_time_.denominator);
        const auto point_bytes = std::move(point).release();
        const std::vector<std::uint8_t> point_data(point_bytes.begin(), point_bytes.end());
        const auto origin = std::string("pops.uniform-field-observation.point@1:sha256:") +
                            identity::sha256_hex(point_data);
        auto support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
            continuation_field_program_identity_(), origin, -1, 0, 0, 0, std::string(contract)});
        return std::make_pair(&found->second, std::move(support));
      },
      "independent Field observation publication");
  // All allocation, validation and device completion precede the collective vote.
  prepared.first->metadata.support = std::move(prepared.second);
  prepared.first->sealed = true;
}

void continuation_preflight_field(const field_type& source, const ContinuationType& type) const {
  const auto instance = continuation_field_instance_(source);
  const auto support = instance ? instance->metadata.support : ContinuationFieldSupport::Ref{};
  const auto& prototype = continuation_field_prototype_(type, support);
  if (!type.field_provider.empty() && !instance)
    throw std::invalid_argument("Independent Field source has no exact live storage authority");
  if (type.field_provider.empty())
    require_same_layout_(source, prototype, "Continuation Field provider");
  else
    require_same_field_contract_(source, prototype, "Continuation independent Field provider");
  if (source.ncomp() != type.field_components)
    throw std::invalid_argument("Continuation Field component schema changed");
}
void continuation_preflight_field(const field_type& source, const ContinuationType& type,
                                  const ContinuationFieldMetadata& metadata) const {
  continuation_preflight_field(source, type);
  require_continuation_field_support_(system_, type, metadata.support);
  const auto instance = continuation_field_instance_(source);
  if ((instance && (instance->metadata.support != metadata.support ||
                    instance->metadata.expression != metadata.expression ||
                    instance->metadata.representation != metadata.representation)) ||
      (!instance && (metadata.support || metadata.representation)))
    throw std::invalid_argument("Continuation Field metadata is not owned by its exact source");
}
