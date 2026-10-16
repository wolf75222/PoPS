// Scientific FieldProblem identity is separate from native solve invocation IDs.
private:
struct ContinuationFieldAllocations {
  using local_storage_type = typename field_type::fab_type::storage_type::const_type;
  std::shared_ptr<const void> incarnation;
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
struct ContinuationFieldProvider {
  std::set<std::pair<std::int64_t, int>> routes;
  std::vector<field_type> prototypes;
  std::uint64_t epoch = 0, generation = 0;
  int components = 0, ghosts = 0;
};
mutable std::map<std::string, ContinuationFieldProvider, std::less<>> continuation_field_providers_;
struct ContinuationFieldObservationAllocation {
  std::weak_ptr<field_type> lifetime;
  std::string contract;
  PreparedResourceAttempt attempt;
  int level;
  std::uint64_t epoch, generation;
  ContinuationFieldAllocations allocations;
  std::uint64_t byte_capacity = 0;
};
mutable std::map<const field_type*, ContinuationFieldObservationAllocation>
    continuation_field_observation_allocations_;

void require_hierarchy_field_observation_storage_(const field_type& prototype,
                                                  std::string_view contract,
                                                  std::size_t origin_bytes,
                                                  std::uint64_t capacity) const {
  const auto charge = [&](std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > capacity / width)
      throw std::length_error("FieldProblem observation exceeds its byte capacity");
    capacity -= count * width;
  };
  charge(sizeof(field_type) + field_type::storage_identity_payload_bytes());
  // The pending and sealed map records coexist until collective publication.
  // Tree/control-block allocator bookkeeping is outside this declared-storage budget.
  charge(sizeof(std::pair<const field_type* const, ContinuationFieldObservationAllocation>));
  charge(sizeof(std::pair<const field_type* const, ContinuationFieldStorageAuthority>));
  charge(sizeof(ContinuationFieldSupport));
  charge(2, contract.size());
  charge(facade_->installed_program_hash().size());
  charge(origin_bytes);
  charge(prototype.layout().size(), 2 * sizeof(Box<Dim>) + sizeof(std::size_t));
  charge(prototype.distribution().owners().size(), sizeof(Index<Dim>));
  charge(prototype.local_size(),
         sizeof(typename field_type::fab_type) + sizeof(std::size_t) +
             2 * sizeof(typename ContinuationFieldAllocations::local_storage_type));
  for (std::size_t patch = 0; patch < prototype.local_size(); ++patch)
    charge(prototype.fab(patch).size(), sizeof(Real));
}

const ContinuationFieldProvider& continuation_field_provider_(std::string_view contract) const {
  const auto found = continuation_field_providers_.find(contract);
  if (found == continuation_field_providers_.end())
    throw std::invalid_argument("Continuation FieldProblem has no installed provider catalogue");
  auto& provider = found->second;
  if (provider.epoch != runtime_->topology_epoch() ||
      provider.generation != runtime_->materialization_generation() ||
      provider.prototypes.size() != static_cast<std::size_t>(nlev())) {
    // This is storage rematerialization from the authenticated native carrier.
    // It neither solves the field nor authorizes any old or foreign allocation.
    std::vector<field_type> prepared;
    prepared.reserve(static_cast<std::size_t>(nlev()));
    for (int level = 0; level < nlev(); ++level)
      prepared.push_back(make_scratch_(runtime_->hierarchy().state(level), provider.components,
                                       uniform_ghosts_(provider.ghosts)));
    provider.prototypes.swap(prepared);
    provider.epoch = runtime_->topology_epoch();
    provider.generation = runtime_->materialization_generation();
  }
  return found->second;
}

public:
void configure_hierarchy_continuation_field_provider(std::string_view contract, std::int64_t solve,
                                                     int component, int components = 1,
                                                     int ghosts = 1) const {
  auto& provider = configured_hierarchy_field_solver_(solve);
  decltype(continuation_field_providers_) staged;
  std::string identity;
  continuation_local_phase(
      [&] {
        if (contract.empty() || component < 0 || components < 1 || ghosts < 0 ||
            provider.level_count() != nlev())
          throw std::invalid_argument("Continuation FieldProblem requires an exact unknown");
        ExactContractBuilder proof;
        proof.text("pops.continuation.field-provider@1")
            .text(contract)
            .scalar(solve)
            .scalar(component)
            .scalar(components)
            .scalar(ghosts);
        identity = std::move(proof).release();
        ContinuationFieldProvider pending{
            {},         {},    runtime_->topology_epoch(), runtime_->materialization_generation(),
            components, ghosts};
        const auto previous = continuation_field_providers_.find(contract);
        if (previous != continuation_field_providers_.end()) {
          if (previous->second.components != components || previous->second.ghosts != ghosts)
            throw std::invalid_argument(
                "Continuation FieldProblem catalogue changed its storage schema");
          pending.routes = previous->second.routes;
        }
        pending.routes.emplace(solve, component);
        pending.prototypes.reserve(static_cast<std::size_t>(nlev()));
        for (int level = 0; level < nlev(); ++level) {
          const auto& source = provider.solution(level);
          if (component >= source.ncomp())
            throw std::invalid_argument("Continuation FieldProblem selects an absent component");
          pending.prototypes.push_back(make_scratch_(source, components, uniform_ghosts_(ghosts)));
          if (previous != continuation_field_providers_.end() &&
              previous->second.epoch == pending.epoch &&
              previous->second.generation == pending.generation)
            require_same_field_contract_(pending.prototypes.back(),
                                         previous->second.prototypes.at(level),
                                         "Continuation FieldProblem routes disagree on storage");
        }
        staged.emplace(std::string(contract), std::move(pending));
      },
      "Continuation FieldProblem catalogue preparation");
  const std::array<std::pair<std::string_view, std::string_view>, 1> proof{
      {{"provider", identity}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(proof, prepared_execution_lane()))
    throw std::invalid_argument("Continuation FieldProblem catalogue differs across ranks");
  auto prepared = staged.extract(staged.begin());
  const auto previous = continuation_field_providers_.find(contract);
  if (previous != continuation_field_providers_.end())
    continuation_field_providers_.erase(previous);
  continuation_field_providers_.insert(std::move(prepared));
}

const field_type& continuation_field_prototype_(
    const ContinuationType& type, int scope_level,
    const ContinuationFieldSupport::Ref& support) const {
  if (type.field_provider.empty())
    return continuation_field_prototype_(type.field_owner, scope_level, support);
  if (type.kind != ContinuationKind::Field || type.field_owner != -1 || !support ||
      support->program_identity != facade_->installed_program_hash() ||
      support->field_provider != type.field_provider || support->program_owner != -1)
    throw std::invalid_argument(
        "Continuation FieldProblem prototype has foreign scientific authority");
  support->validate();
  const auto& provider = continuation_field_provider_(type.field_provider);
  if (support->producer_level < 0 || support->producer_level >= nlev())
    throw std::invalid_argument("Continuation FieldProblem producer level is absent");
  const auto& prototype = provider.prototypes.at(support->producer_level);
  if (prototype.ncomp() != type.field_components)
    throw std::invalid_argument("Continuation FieldProblem component schema changed");
  return prototype;
}

std::shared_ptr<field_type> allocate_hierarchy_continuation_field_observation(
    std::string_view contract, std::uint64_t byte_capacity) const {
  const field_type* prototype = nullptr;
  continuation_local_phase(
      [&] {
        const auto& provider = continuation_field_provider_(contract);
        if (active_level_ < 0 || active_level_ >= nlev() || !resource_attempt().visible())
          throw std::invalid_argument(
              "FieldProblem observation allocation requires an exact attempt");
        prototype = &provider.prototypes.at(active_level_);
        require_hierarchy_field_observation_storage_(*prototype, contract, 0, byte_capacity);
      },
      "FieldProblem observation capacity");
  std::shared_ptr<field_type> result;
  decltype(continuation_field_observation_allocations_) staged;
  continuation_local_phase(
      [&] {
        result = std::make_shared<field_type>(*prototype);
        staged.emplace(result.get(),
                       ContinuationFieldObservationAllocation{
                           result, std::string(contract), resource_attempt(), active_level_,
                           runtime_->topology_epoch(), runtime_->materialization_generation(),
                           continuation_field_allocations_(*result), byte_capacity});
      },
      "FieldProblem observation allocation");
  std::erase_if(continuation_field_observation_allocations_,
                [](const auto& item) { return item.second.lifetime.expired(); });
  continuation_field_observation_allocations_.merge(staged);
  return result;
}

void seal_hierarchy_continuation_field_observation(
    std::string_view contract, std::int64_t observation, const std::shared_ptr<field_type>& field,
    const std::shared_ptr<const HierarchyFieldObservationSnapshot>& source) const {
  decltype(continuation_field_storage_) staged;
  continuation_local_phase(
      [&] {
        require_hierarchy_field_observation_(source);
        const auto issued = continuation_field_observation_allocations_.find(field.get());
        if (!field || observation < 0 || !source->support_ ||
            issued == continuation_field_observation_allocations_.end())
          throw std::invalid_argument(
              "derived FieldProblem observation lacks exact issued storage");
        const auto& allocation = issued->second;
        if (allocation.lifetime.lock() != field || allocation.contract != contract ||
            !allocation.attempt.visible() || !allocation.attempt.same_attempt(resource_attempt()) ||
            allocation.level != active_level_ || allocation.epoch != runtime_->topology_epoch() ||
            allocation.generation != runtime_->materialization_generation() ||
            !allocation.allocations.matches(*field))
          throw std::invalid_argument(
              "derived FieldProblem observation replaced its issued allocation");
        const auto& provider = continuation_field_provider_(contract);
        if (!provider.routes.contains({source->solve_, source->component_}))
          throw std::invalid_argument(
              "derived FieldProblem observation has a foreign source route");
        // The source origin is only known at seal. Charge its exact bytes before
        // constructing support or the second allocation-reference vector.
        require_hierarchy_field_observation_storage_(
            *field, contract, source->support_->origin_point.size(), allocation.byte_capacity);
        require_same_field_contract_(*field, provider.prototypes.at(active_level_),
                                     "derived FieldProblem observation prototype");
        auto support = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
            facade_->installed_program_hash(), source->support_->origin_point, -1, active_level_,
            runtime_->topology_epoch(), runtime_->materialization_generation(),
            std::string(contract)});
        support->validate();
        staged.emplace(field.get(),
                       ContinuationFieldStorageAuthority{
                           field, -1, active_level_, runtime_->topology_epoch(),
                           runtime_->materialization_generation(), std::move(support),
                           continuation_field_allocations_(*field), false, source->attempt_});
        Kokkos::fence();
      },
      "derived FieldProblem observation authority");
  continuation_field_observation_allocations_.erase(field.get());
  continuation_field_storage_.erase(field.get());
  continuation_field_storage_.merge(staged);
}
