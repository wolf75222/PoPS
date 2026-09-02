// Accepted-context implementation fragment; included inside AcceptedContextSnapshot.

static std::unique_ptr<AcceptedContextSnapshot> from_forward(DetachedState staged) {
  if (!staged.interface_flux_ledger || staged.interface_flux_ledger->in_transaction())
    throw std::logic_error("AMR Program detached accepted context requires a sealed ledger");
  // A ledger copy preserves its dense images but not the reserve of its transaction-contract
  // strings.  Forward preparation is cold; re-prime it here, before this detached image can be
  // rebound and used by an accepted attempt.  Refresh, publish and finalization deliberately
  // have no route to this bind-only operation.
  staged.interface_flux_ledger->prime_hot_carriers_at_bind();
  return std::unique_ptr<AcceptedContextSnapshot>(new AcceptedContextSnapshot(std::move(staged)));
}

/// One rollback transaction retains a separate accepted image.  Every owned clone is charged:
/// the adapter's interface ledger and subcycling state remain live while this snapshot owns its
/// distinct rollback copies, so omitting either would understate the simultaneous ceiling.
[[nodiscard]] static std::uint64_t resident_storage_bytes_for_owner(
    const AmrStorageTopologyAdapter& owner,
    const PreparedSubcyclingBundle* prepared_subcycling_bundle = nullptr,
    const typename multiblock_subcycling_type::MutableStateImage* forward_subcycling_state =
        nullptr) {
  const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("AMR Program accepted snapshot storage overflows uint64");
    total += value;
  };
  const auto vector_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("AMR Program accepted snapshot vector storage overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  };

  std::uint64_t total = sizeof(AcceptedContextSnapshot);
  const auto effects_without_staging = owner.history_effects_resident_storage_bytes_(false);
  const auto effects_with_staging = owner.history_effects_resident_storage_bytes_(true);
  if (effects_with_staging < effects_without_staging)
    throw std::logic_error("AMR Program accepted snapshot staging footprint underflowed");
  // The owner-only candidate byte buffer is not cloned into the rollback snapshot.  Everything
  // else in this delta is the snapshot's own POPSAND5 staging envelope, including its level
  // clock and flat serialization carriers.
  const auto candidate_bytes = vector_bytes(owner.accepted_checkpoint_candidate_bytes_);
  const auto staging_delta = effects_with_staging - effects_without_staging;
  if (staging_delta < candidate_bytes)
    throw std::logic_error("AMR Program accepted snapshot staging footprint is malformed");
  checked_add(total, effects_without_staging);

  // A snapshot owns another complete prepared scratch image.  The live image's field payload
  // is represented by generated rows; this clone is rollback-only and must retain both the
  // field carriers and their complete MultiFab storage here.
  checked_add(total, vector_bytes(owner.prepared_scratch_));
  checked_add(total, vector_bytes(owner.prepared_scratch_descriptors_));
  if (owner.prepared_scratch_.size() != owner.prepared_scratch_descriptors_.size())
    throw std::logic_error("AMR Program accepted snapshot lost prepared scratch descriptors");
  for (std::size_t slot = 0; slot < owner.prepared_scratch_.size(); ++slot)
    for (std::size_t family = 0; family < owner.prepared_scratch_[slot].size(); ++family) {
      const auto& fields = owner.prepared_scratch_[slot][family];
      const auto& descriptors = owner.prepared_scratch_descriptors_[slot][family];
      checked_add(total, vector_bytes(fields));
      checked_add(total, vector_bytes(descriptors));
      if (fields.size() != descriptors.size())
        throw std::logic_error("AMR Program accepted snapshot scratch family is incomplete");
      for (std::size_t subslot = 0; subslot < fields.size(); ++subslot) {
        const auto& entry = fields[subslot];
        if (static_cast<bool>(entry) != static_cast<bool>(descriptors[subslot]))
          throw std::logic_error("AMR Program accepted snapshot scratch entry is incomplete");
        if (!entry)
          continue;
        checked_add(total, vector_bytes(*entry));
        for (const field_type& field : *entry)
          checked_add(total, field.resident_storage_bytes());
      }
    }

  checked_add(total, owner.hot_path_workspace_.resident_storage_bytes());
  checked_add(total, vector_bytes(owner.accepted_checkpoint_level_clock_slots_));
  checked_add(total, owner.clock_schedule_.resident_storage_bytes());
  if (!owner.interface_flux_ledger_)
    throw std::logic_error("AMR Program accepted snapshot has no interface-flux ledger");
  checked_add(total, sizeof(interface_flux_ledger_type));
  checked_add(total, owner.interface_flux_ledger_->resident_storage_bytes());
  if (owner.multiblock_subcycling_)
    checked_add(total, owner.multiblock_subcycling_->mutable_state_image_resident_storage_bytes());
  // During Candidate the A image remains in the accepted owner while the snapshot owns a
  // complete B engine and its rollback image.  The bundle owns B's engine/tables/workspace;
  // its separate MutableStateImage is an additional copy used by the inverse publication.
  if (prepared_subcycling_bundle != nullptr) {
    checked_add(total, prepared_subcycling_bundle->resident_storage_bytes());
    if (forward_subcycling_state != nullptr) {
      if (!prepared_subcycling_bundle->engine)
        throw std::logic_error("AMR Program prepared subcycling bundle has no engine");
      checked_add(total,
                  prepared_subcycling_bundle->engine->mutable_state_image_resident_storage_bytes());
    }
  } else if (forward_subcycling_state != nullptr) {
    throw std::logic_error("AMR Program forward subcycling state has no prepared bundle");
  }

  const auto string_bytes = [&](const std::string& value) -> std::uint64_t {
    const auto begin = reinterpret_cast<std::uintptr_t>(&value);
    const auto end = begin + sizeof(value);
    const auto data = reinterpret_cast<std::uintptr_t>(value.data());
    if (data >= begin && data < end)
      return 0;
    if (value.capacity() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program accepted snapshot string storage overflows uint64");
    return static_cast<std::uint64_t>(value.capacity()) + 1U;
  };
  // The snapshot retains a second cold slot image for accepted effects so a later contraction
  // and regrowth stays allocation-free.  history_effects_resident_storage_bytes_ above already
  // charged the snapshot's logical accepted copy; charge this distinct slot mirror once here.
  for (const auto& fragments : owner.accepted_face_flux_commit_slots_) {
    checked_add(total, vector_bytes(fragments));
    for (const auto& fragment : fragments) {
      checked_add(total, string_bytes(fragment.key.owner));
      checked_add(total, string_bytes(fragment.key.state));
      checked_add(total, string_bytes(fragment.key.stage));
      checked_add(total, vector_bytes(fragment.payload));
    }
  }
  checked_add(total, vector_bytes(owner.accepted_synchronization_event_commit_slots_));
  for (const auto& event : owner.accepted_synchronization_event_commit_slots_)
    checked_add(total, string_bytes(event.phase));
  checked_add(total, string_bytes(owner.accepted_temporal_partition_.provider_identity));
  checked_add(total, vector_bytes(owner.accepted_temporal_partition_.cells));
  if (owner.cell_temporal_configuration_) {
    const auto& configuration = *owner.cell_temporal_configuration_;
    checked_add(total, string_bytes(configuration.clock));
    checked_add(total, string_bytes(configuration.exact_contract));
    checked_add(total, vector_bytes(configuration.level_rungs));
    checked_add(total, vector_bytes(configuration.routes));
    checked_add(total, vector_bytes(configuration.level_cell_counts));
  }
  checked_add(total, staging_delta - candidate_bytes);
  return total;
}

/// Candidate-aware receipt used whenever this snapshot owns a forward bundle.  The owner is
/// the still-live A authority; this image adds B and its separate mutable rollback image
/// without recharging A's mutable image a second time.
[[nodiscard]] std::uint64_t resident_storage_bytes_for(
    const AmrStorageTopologyAdapter& owner) const {
  std::uint64_t total = resident_storage_bytes_for_owner(
      owner, prepared_subcycling_bundle_ ? std::addressof(*prepared_subcycling_bundle_) : nullptr,
      forward_subcycling_state_ ? std::addressof(*forward_subcycling_state_) : nullptr);
  const auto checked_add = [](std::uint64_t& target, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - target)
      throw std::overflow_error("AMR Program forward tensor snapshot storage overflows uint64");
    target += value;
  };
  const auto vector_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("AMR Program forward tensor snapshot vector overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  };
  const auto string_bytes = [](const std::string& value) -> std::uint64_t {
    return ::pops::amr::reflux::detail::external_string_storage_bytes(value);
  };
  if (forward_hierarchy_tensor_solver_) {
    const auto storage = forward_hierarchy_tensor_solver_->resident_storage();
    if (!storage.is_exact())
      throw std::logic_error("AMR Program forward hierarchy tensor solver has no exact storage");
    checked_add(total, storage.bytes);
    checked_add(total, vector_bytes(forward_hierarchy_tensor_boundaries_));
  }
  if (forward_hierarchy_tensor_selection_) {
    const auto& selection = *forward_hierarchy_tensor_selection_;
    checked_add(total, string_bytes(selection.provider_identity));
    checked_add(total, string_bytes(selection.plan_identity));
    checked_add(total, string_bytes(selection.operator_contract_identity));
    checked_add(total, vector_bytes(selection.assembly_field_slots));
    for (const auto& field_slot : selection.assembly_field_slots)
      checked_add(total, string_bytes(field_slot));
    checked_add(total, string_bytes(selection.solution_field_slot));
    checked_add(total, string_bytes(selection.exact_contract));
  }
  checked_add(total, vector_bytes(forward_prepared_history_mutation_slots_));
  for (const auto& slot : forward_prepared_history_mutation_slots_) {
    checked_add(total, string_bytes(slot.name));
    checked_add(total, string_bytes(slot.key));
    checked_add(total, string_bytes(slot.clock_identity));
    checked_add(total, vector_bytes(slot.rollback_ring));
    for (const auto& field : slot.rollback_ring)
      checked_add(total, field.resident_storage_bytes());
    checked_add(total, vector_bytes(slot.dts));
    checked_add(total, vector_bytes(slot.expressions));
    checked_add(total, string_bytes(slot.store_contract));
  }
  checked_add(total, string_bytes(forward_prepared_history_rotation_contract_));
  checked_add(total, vector_bytes(forward_accepted_history_binding_mutation_slots_));
  checked_add(total, vector_bytes(forward_accepted_pending_history_ordinal_sources_));
  return total;
}

std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> detach_for_forward(
    std::uint64_t topology_epoch, std::uint64_t materialization_generation,
    void*& rebind_token) const override {
  rebind_token = static_cast<void*>(&rebind_owner());
  auto detached = from_forward(detach_for_forward(topology_epoch, materialization_generation));
  detached->prime_accepted_state_staging_from_cold_source_(*this);
  return detached;
}

void rebind_after_forward_publish(void* rebind_token) noexcept override {
  auto* owner = static_cast<AmrStorageTopologyAdapter*>(rebind_token);
  if (owner == nullptr || owner_ != nullptr || !interface_flux_ledger_ ||
      interface_flux_ledger_->in_transaction())
    std::terminate();
  owner_ = owner;
}

void publish_prepared_installation_temporal_authority(void* rebind_token) noexcept override {
  auto* owner = static_cast<AmrStorageTopologyAdapter*>(rebind_token);
  if (owner == nullptr || owner_ != nullptr || !interface_flux_ledger_ ||
      interface_flux_ledger_->in_transaction() || resource_epoch_ != owner->resource_epoch_ ||
      resource_generation_ != owner->resource_generation_)
    std::terminate();
  // The owner's staging/workspace images were primed from the complete host capacity after DSO
  // preparation.  Publishing the whole snapshot here would exchange those witnesses with a
  // copy made before the bootstrap authority was known.  Only these four values are products of
  // prepare_forward_temporal_partition(); all are no-throw swaps/scalars.
  std::swap(owner->accepted_temporal_partition_, accepted_temporal_partition_);
  std::swap(owner->cell_temporal_configuration_, cell_temporal_configuration_);
  owner->accepted_flux_budget_contract_.swap(accepted_flux_budget_contract_);
  owner->accepted_coupling_contract_.swap(accepted_coupling_contract_);
  std::swap(owner->accepted_state_revision_, accepted_state_revision_);
  owner_ = owner;
}

void prepare_forward_hierarchy_refresh(std::uint64_t topology_epoch,
                                       std::uint64_t materialization_generation) override {
  require_detached_forward_authority_(topology_epoch, materialization_generation,
                                      "hierarchy refresh");
  if (!history_levels_.empty() || !history_flux_expressions_.empty() ||
      !pending_history_remaps_.empty())
    throw std::logic_error(
        "AMR Program forward hierarchy refresh requires an explicit detached history remap");
  invalidate_forward_topology_resources_();
  reset_forward_accepted_state_staging_(topology_epoch, materialization_generation);
}

void prepare_forward_history_remap(const AmrProgramHistoryRemapDescriptor& descriptor) override {
  if (descriptor.published_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.published_materialization_generation == std::numeric_limits<std::uint64_t>::max())
    throw std::logic_error("AMR Program forward history remap has no published authority");
  // A cumulative regrid retains one detached image of the final forward hierarchy, while each
  // descriptor remains the direct transition which created its child ring.  Requiring every
  // descriptor to equal the final epoch would forge non-direct checkpoint markers.  Instead,
  // each transition is bounded by the detached final authority; the carrier authenticates it
  // against its own staged transaction and the checkpoint validator requires the complete
  // contiguous chain to end at this final image.
  if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
      descriptor.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.prior_topology_epoch + 1U != descriptor.published_topology_epoch ||
      descriptor.prior_materialization_generation + 1U !=
          descriptor.published_materialization_generation ||
      descriptor.published_topology_epoch > resource_epoch_ ||
      descriptor.published_materialization_generation > resource_generation_)
    throw std::logic_error("AMR Program detached accepted context cannot prepare history remap");
  prepare_detached_history_remap_(descriptor);
  reset_forward_accepted_state_staging_(descriptor.published_topology_epoch,
                                        descriptor.published_materialization_generation);
}

void prepare_forward_full_history_reseed(
    const AmrProgramFullHistoryReseedDescriptor& descriptor) override {
  prepare_detached_full_rebuild_history_reseed_(descriptor);
  reset_forward_accepted_state_staging_(descriptor.published_topology_epoch,
                                        descriptor.published_materialization_generation);
}

/// Rebuild the topology-bound cell-local provider solely from the forward hierarchy authority.
/// This runs during Candidate: every allocation is confined to local replacement values and the
/// only mutation of the detached image is the final no-throw swap/publication of those values.
void prepare_forward_temporal_partition(
    const PreparedForwardAmrTemporalAuthority& authority) override {
  require_detached_forward_authority_(authority.topology_epoch,
                                      authority.materialization_generation, "temporal partition");
  if (authority.accepted_state_revision == std::numeric_limits<std::uint64_t>::max() ||
      authority.spatial_contract.empty() || authority.lane_identity.empty() ||
      authority.collective_contract.empty() || authority.level_count == 0 ||
      authority.block_count == 0 ||
      authority.block_count > std::numeric_limits<std::size_t>::max() / authority.level_count ||
      authority.block_level_cell_counts.size() != authority.block_count * authority.level_count ||
      authority.periodic_faces.size() != static_cast<std::size_t>(2 * Dim) ||
      authority.temporal_provider_identity.empty() || authority.flux_budget_contract.empty() ||
      authority.coupling_contract.empty() ||
      authority.interface_flux_ledger_budget.exact_contract.empty() ||
      interface_flux_ledger_->topology_epoch() != authority.topology_epoch)
    throw std::logic_error("AMR Program forward temporal authority is incomplete");

  std::optional<CellTemporalConfiguration> next_configuration;
  CellTemporalPartitionAcceptedState next_partition;
  std::string next_flux_budget_contract;
  std::string next_coupling_contract;

  if (accepted_temporal_partition_.kind == TemporalPartitionKind::Global) {
    if (cell_temporal_configuration_ ||
        authority.temporal_provider_identity != kGlobalTemporalPartitionProvider ||
        accepted_temporal_partition_.provider_identity != kGlobalTemporalPartitionProvider ||
        !accepted_temporal_partition_.cells.empty())
      throw std::logic_error(
          "AMR Program forward global temporal partition changed its provider authority");
    // Multiblock coupling and its interface-flux provider are independent of the temporal
    // partition provider.  Their exact contracts and the bind-sealed ledger budget above are
    // refreshed as part of this forward authority; only a cell-local executor may not appear
    // beneath a global partition without an explicit transfer declaration.
    next_partition = accepted_temporal_partition_;
    validate_cell_temporal_partition_state(next_partition);
  } else if (accepted_temporal_partition_.kind == TemporalPartitionKind::CellLocal) {
    if (!cell_temporal_configuration_ ||
        authority.temporal_provider_identity != kSameLevelTransportEulerStageFluxProvider ||
        accepted_temporal_partition_.provider_identity != kSameLevelTransportEulerStageFluxProvider)
      throw std::logic_error(
          "AMR Program forward cell-local temporal provider is not rematerializable");
    // A cell-local partition is only a serialized witness for the resident
    // provider/executor pair.  This detached checkpoint has no value-transfer declaration for
    // that pair (in particular its per-level field/geometry/diagnostic arenas), so rewriting
    // the witness for another epoch would leave the accepted executor bound to the old
    // topology.  The installation bootstrap has the same epoch and remains valid; every real
    // topology transition must be rejected by the host preflight before publication.
    if (cell_temporal_configuration_->topology_epoch != authority.topology_epoch ||
        cell_temporal_configuration_->materialization_generation !=
            authority.materialization_generation ||
        accepted_temporal_partition_.topology_epoch != authority.topology_epoch)
      throw std::logic_error(
          "AMR topology regrid has no declared transfer provider for the cell-local resident "
          "executor");
    next_configuration.emplace(*cell_temporal_configuration_);
    next_partition = prepare_forward_cell_temporal_partition_(
        *next_configuration, authority, accepted_temporal_partition_.synchronization_tick);
  } else {
    throw std::logic_error("AMR Program forward temporal partition has an unsupported kind");
  }
  next_flux_budget_contract = authority.flux_budget_contract;
  next_coupling_contract = authority.coupling_contract;

  // This is the sole cold boundary at which a forward hierarchy may enlarge its accepted
  // checkpoint clocks.  Refresh borrows the published adapter envelope and must never reserve.
  if (accepted_checkpoint_level_clock_slots_.capacity() < authority.level_count)
    accepted_checkpoint_level_clock_slots_.reserve(authority.level_count);
  accepted_checkpoint_level_clock_slots_.clear();

  // ``prepare_budget`` allocates at Candidate time.  Once it has succeeded the ledger replaces
  // only its detached bound image; the subsequent string/configuration/partition swaps cannot
  // allocate or consult the sealed owner.
  auto prepared_budget =
      interface_flux_ledger_->prepare_budget(authority.interface_flux_ledger_budget);
  interface_flux_ledger_->publish_prepared_budget(prepared_budget);
  std::swap(cell_temporal_configuration_, next_configuration);
  std::swap(accepted_temporal_partition_, next_partition);
  accepted_flux_budget_contract_.swap(next_flux_budget_contract);
  accepted_coupling_contract_.swap(next_coupling_contract);
  accepted_state_revision_ = authority.accepted_state_revision;
}
