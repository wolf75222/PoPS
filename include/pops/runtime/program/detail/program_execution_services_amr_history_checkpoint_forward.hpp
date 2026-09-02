// Accepted-context implementation fragment; included inside AcceptedContextSnapshot.

static void require_forward_storage_ceilings_(
    std::uint64_t candidate_live_bytes, std::uint64_t snapshot_a_plus_b_bytes,
    const AmrProgramAcceptedStateStagingCapacity<Dim>& envelope) {
  if (candidate_live_bytes > envelope.configured_live_subcycling_bytes)
    throw std::length_error(
        "AMR Program forward subcycling candidate exceeds its configured live ceiling");
  if (snapshot_a_plus_b_bytes > envelope.configured_forward_snapshot_bytes)
    throw std::length_error(
        "AMR Program forward subcycling candidate exceeds its configured A+B snapshot ceiling");
}

/// Exact candidate-side peak after all forward cold preparation and before the first B
/// emplacement.  `history_effects_resident_storage_bytes_(true)` deliberately includes the
/// rebuilt face/event pools and the mutable POPSAND5 staging image; those are distinct from the
/// bundle's engine/ledger/workspace ownership and therefore charged here exactly once.
[[nodiscard]] std::uint64_t forward_candidate_resident_storage_bytes_(
    const PreparedSubcyclingBundle& bundle,
    const typename multiblock_subcycling_type::MutableStateImage& mutable_state) const {
  const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("AMR Program forward candidate storage overflows uint64");
    total += value;
  };
  const auto vector_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("AMR Program forward candidate vector storage overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  };
  const auto string_bytes = [](const std::string& value) -> std::uint64_t {
    return ::pops::amr::reflux::detail::external_string_storage_bytes(value);
  };
  if (!bundle.engine)
    throw std::logic_error("AMR Program forward candidate has no subcycling engine");
  std::uint64_t total = 0;
  checked_add(total, bundle.resident_storage_bytes());
  checked_add(total, bundle.engine->mutable_state_image_resident_storage_bytes());
  const auto add_face_slots = [&](const auto& axes) {
    for (const auto& slots : axes) {
      checked_add(total, vector_bytes(slots));
      for (const auto& slot : slots) {
        checked_add(total, string_bytes(slot.key.owner));
        checked_add(total, string_bytes(slot.key.state));
        checked_add(total, string_bytes(slot.key.stage));
        checked_add(total, vector_bytes(slot.payload));
      }
    }
  };
  const auto add_events = [&](const auto& events) {
    checked_add(total, vector_bytes(events));
    for (const auto& event : events)
      checked_add(total, string_bytes(event.phase));
  };
  // These are precisely the forward-rebuilt B effect carriers.  Logical vectors retain their
  // own capacity separately from the full cold slot pools, and checkpoint staging keeps both
  // its configured clock vector and its transient serialized candidate buffer resident.
  add_face_slots(accepted_face_flux_);
  add_face_slots(accepted_face_flux_slots_);
  add_events(accepted_synchronization_events_);
  add_events(accepted_synchronization_event_slots_);
  checked_add(total, vector_bytes(accepted_checkpoint_level_clock_slots_));
  checked_add(total, vector_bytes(accepted_state_staging_.state.level_clocks));
  checked_add(total, vector_bytes(accepted_state_staging_.state.accepted_interface_flux));
  checked_add(total, vector_bytes(accepted_state_staging_.state.synchronization_events));
  checked_add(total, vector_bytes(accepted_state_staging_.accepted_interface_flux_slots));
  for (const auto& ordinals : accepted_face_flux_ordinals_)
    checked_add(total, vector_bytes(ordinals));
  checked_add(total, vector_bytes(accepted_interface_flux_wire_ordinals_));
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
  checked_add(total, vector_bytes(accepted_state_staging_.synchronization_event_slots));
  const auto& staging = accepted_state_staging_;
  const auto& state = staging.state;
  checked_add(total, string_bytes(state.spatial_contract));
  checked_add(total, vector_bytes(state.history_slots));
  for (const auto& slot : state.history_slots)
    checked_add(total, string_bytes(slot.name));
  checked_add(total, vector_bytes(state.pending_history_remaps));
  for (const auto& remap : state.pending_history_remaps)
    checked_add(total, string_bytes(remap.key));
  checked_add(total, vector_bytes(state.history_flux_payload));
  checked_add(total, string_bytes(state.temporal_partition.provider_identity));
  checked_add(total, vector_bytes(state.temporal_partition.cells));
  checked_add(total, vector_bytes(state.tagging_hysteresis_state));
  checked_add(total, string_bytes(state.flux_budget_contract));
  checked_add(total, string_bytes(state.coupling_contract));
  for (const auto& fragment : state.accepted_interface_flux) {
    checked_add(total, string_bytes(fragment.key.interface_identity));
    checked_add(total, string_bytes(fragment.key.stage_identity));
    checked_add(total, string_bytes(fragment.key.graph_identity));
    checked_add(total, string_bytes(fragment.key.rate_identity));
    checked_add(total, string_bytes(fragment.key.application_identity));
    checked_add(total, vector_bytes(fragment.payload));
  }
  add_events(state.synchronization_events);
  checked_add(total, vector_bytes(staging.history_slot_bindings));
  for (const auto& binding : staging.history_slot_bindings)
    checked_add(total, string_bytes(binding.key));
  checked_add(total, vector_bytes(staging.history_slot_pool));
  for (const auto& slot : staging.history_slot_pool)
    checked_add(total, string_bytes(slot.name));
  checked_add(total, vector_bytes(staging.history_slot_active_indices));
  checked_add(total, vector_bytes(staging.pending_history_keys));
  for (const auto& key : staging.pending_history_keys)
    checked_add(total, string_bytes(key));
  checked_add(total, vector_bytes(staging.pending_history_remap_slots));
  for (const auto& remap : staging.pending_history_remap_slots)
    checked_add(total, string_bytes(remap.key));
  checked_add(total, vector_bytes(staging.pending_history_active_slots));
  add_face_slots(staging.accepted_face_flux_slots);
  for (const auto& sources : staging.accepted_face_flux_sources)
    checked_add(total, vector_bytes(sources));
  for (const auto& active : staging.accepted_face_flux_active_slots)
    checked_add(total, vector_bytes(active));
  checked_add(total, vector_bytes(staging.accepted_interface_flux_active_slots));
  checked_add(total, vector_bytes(staging.synchronization_event_active_indices));
  if (forward_hierarchy_tensor_solver_) {
    const auto storage = forward_hierarchy_tensor_solver_->resident_storage();
    if (!storage.is_exact())
      throw std::logic_error("AMR Program forward candidate tensor has no exact storage");
    checked_add(total, storage.bytes);
    checked_add(total, vector_bytes(forward_hierarchy_tensor_boundaries_));
  }
  if (forward_hierarchy_tensor_selection_) {
    const auto& selection = *forward_hierarchy_tensor_selection_;
    checked_add(total, string_bytes(selection.provider_identity));
    checked_add(total, string_bytes(selection.plan_identity));
    checked_add(total, string_bytes(selection.operator_contract_identity));
    checked_add(total, vector_bytes(selection.assembly_field_slots));
    for (const auto& slot : selection.assembly_field_slots)
      checked_add(total, string_bytes(slot));
  }
  // `mutable_state` is passed by reference to make the pre-emplace ownership explicit; its
  // exact storage is provided by the engine helper above and must agree with this image.
  (void)mutable_state;
  return total;
}

/// Rebuild the B topology's accepted-effect envelopes while Candidate owns the detached
/// engine.  `detach_for_forward` deliberately preserves A's pools for inverse publication,
/// but merely clearing the logical vectors would otherwise publish their A route keys after a
/// regrid.  This routine consumes only B's bind-sealed candidate-ledger templates and the
/// typed forward authority; it neither observes a facade nor runs on a hot publication path.
void rebuild_forward_accepted_effect_slots_(const PreparedSubcyclingBundle& bundle,
                                            const PreparedForwardAmrExecutionAuthority& authority) {
  if (!bundle.engine || bundle.block_count == 0 || authority.active_level_count() == 0)
    throw std::logic_error("AMR Program forward accepted-effect envelope has no bundle authority");
  if (bundle.block_count > std::numeric_limits<std::size_t>::max() / authority.active_level_count())
    throw std::overflow_error("AMR Program forward accepted-effect envelope exceeds size_t");

  decltype(accepted_face_flux_slots_) next_face_slots;
  bundle.engine->bind_candidate_ledger_slots(
      [&](std::size_t, std::size_t, std::size_t, multiblock_flux_ledger_type& ledger) {
        if (!ledger.resident_slots_bound())
          throw std::logic_error(
              "AMR Program forward accepted-effect ledger slots were not cold-bound");
        for (int axis = 0; axis < Dim; ++axis) {
          auto& destination = next_face_slots[static_cast<std::size_t>(axis)];
          const auto templates = ledger.resident_slot_templates(axis);
          if (templates.size() > std::numeric_limits<std::size_t>::max() - destination.size())
            throw std::length_error("AMR Program forward accepted-effect face slots exceed size_t");
          destination.insert(destination.end(), templates.begin(), templates.end());
        }
      });
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    auto& slots = next_face_slots[index];
    std::sort(slots.begin(), slots.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
    accepted_face_flux_slots_[index].swap(slots);
    accepted_face_flux_[index].clear();
    if (accepted_face_flux_[index].capacity() < accepted_face_flux_slots_[index].size())
      accepted_face_flux_[index].reserve(accepted_face_flux_slots_[index].size());
    auto& staging_slots = accepted_state_staging_.accepted_face_flux_slots[index];
    staging_slots = accepted_face_flux_slots_[index];
    auto& staging_logical = accepted_state_staging_.state.accepted_face_flux[index];
    staging_logical.clear();
    staging_logical.reserve(staging_slots.size());
    auto& staging_sources = accepted_state_staging_.accepted_face_flux_sources[index];
    staging_sources.clear();
    staging_sources.reserve(staging_slots.size());
    auto& staging_active = accepted_state_staging_.accepted_face_flux_active_slots[index];
    staging_active.clear();
    staging_active.reserve(staging_slots.size());
  }

  if (accepted_state_staging_.configured_level_count < authority.active_level_count())
    throw std::logic_error(
        "AMR Program forward synchronization envelope is shallower than its active hierarchy");
  const std::size_t transitions = accepted_state_staging_.configured_level_count - 1U;
  if (transitions != 0 &&
      bundle.block_count > std::numeric_limits<std::size_t>::max() / transitions / 2U)
    throw std::overflow_error("AMR Program forward synchronization slots exceed size_t");
  std::vector<AmrProgramSynchronizationEvent> next_events;
  next_events.reserve(bundle.block_count * transitions * 2U);
  for (std::size_t block = 0; block < bundle.block_count; ++block)
    for (std::size_t parent = 0; parent < transitions; ++parent)
      for (const std::string_view phase :
           {std::string_view("reflux"), std::string_view("average_down")})
        next_events.push_back({static_cast<int>(parent),
                               static_cast<int>(parent + 1U),
                               static_cast<int>(block),
                               std::string(phase),
                               {}});
  accepted_synchronization_event_slots_.swap(next_events);
  accepted_synchronization_events_.clear();
  if (accepted_synchronization_events_.capacity() < accepted_synchronization_event_slots_.size())
    accepted_synchronization_events_.reserve(accepted_synchronization_event_slots_.size());
  accepted_state_staging_.synchronization_event_slots = accepted_synchronization_event_slots_;
  accepted_state_staging_.state.synchronization_events.clear();
  accepted_state_staging_.state.synchronization_events.reserve(
      accepted_state_staging_.synchronization_event_slots.size());
  accepted_state_staging_.synchronization_event_active_indices.clear();
  accepted_state_staging_.synchronization_event_active_indices.reserve(
      accepted_state_staging_.synchronization_event_slots.size());
}

/// Cold-prepare B's provider-owned hierarchy tensor carrier from the forward topology image.
/// In particular, no accepted facade/runtime is available or consulted on this path.
void prepare_forward_hierarchy_tensor_solver_(
    const PreparedForwardAmrExecutionAuthority& authority) {
  forward_hierarchy_tensor_solver_.reset();
  forward_hierarchy_tensor_boundaries_.clear();
  if (!forward_hierarchy_tensor_selection_)
    return;
  const auto* typed =
      dynamic_cast<const PreparedForwardAmrExecutionAuthorityView<Dim>*>(std::addressof(authority));
  if (typed == nullptr)
    throw std::logic_error("AMR Program forward tensor solver has no typed topology authority");
  const auto& topology = *typed->topology();
  const auto& selection = *forward_hierarchy_tensor_selection_;
  hierarchy_tensor_request_type request;
  std::vector<HierarchyTensorLevelBoundary> boundaries;
  std::exception_ptr local_error;
  try {
    if (topology.lane == nullptr || !topology.lane->active() || owner_ == nullptr ||
        owner_->hierarchy_tensor_solver_registry_ == nullptr || selection.program_block < 0 ||
        static_cast<std::size_t>(selection.program_block) >= topology.program_block_map.size())
      throw std::invalid_argument("AMR Program forward tensor authority is incomplete");
    const int runtime_block =
        topology.program_block_map.at(static_cast<std::size_t>(selection.program_block));
    if (runtime_block < 0 ||
        static_cast<std::size_t>(runtime_block) >= topology.block_prototypes.size() ||
        topology.level_geometries.empty() ||
        topology.spatial_refinement_ratios.size() + 1U != topology.level_geometries.size())
      throw std::invalid_argument("AMR Program forward tensor topology is malformed");
    const auto& levels = topology.block_prototypes.at(static_cast<std::size_t>(runtime_block));
    if (levels.size() != topology.level_geometries.size() ||
        topology.periodic_faces.size() != static_cast<std::size_t>(2 * Dim))
      throw std::invalid_argument("AMR Program forward tensor levels are incomplete");
    std::array<bool, Dim> periodic{};
    for (int axis = 0; axis < Dim; ++axis) {
      const auto lower = static_cast<std::size_t>(2 * axis);
      if (topology.periodic_faces[lower] != topology.periodic_faces[lower + 1U])
        throw std::invalid_argument("AMR Program forward tensor periodic authority is asymmetric");
      periodic[static_cast<std::size_t>(axis)] = topology.periodic_faces[lower];
    }
    const BoundaryTopology<Dim> boundary_topology = BoundaryTopology<Dim>::axis_periodic(periodic);
    request.block = static_cast<std::size_t>(runtime_block);
    request.components = selection.components;
    request.plan_identity = selection.plan_identity;
    request.operator_contract_identity = selection.operator_contract_identity;
    request.assembly_field_slots = selection.assembly_field_slots;
    request.solution_field_slot = selection.solution_field_slot;
    request.options = selection.options;
    request.ratios = topology.spatial_refinement_ratios;
    request.levels.reserve(levels.size());
    boundaries.reserve(levels.size());
    for (std::size_t level = 0; level < levels.size(); ++level) {
      const Geometry<Dim>& geometry = topology.level_geometries[level];
      std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
      RealVector<Dim> spacing{};
      for (int axis = 0; axis < Dim; ++axis) {
        spacing[axis] = geometry.spacing(axis);
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          if (boundary_topology.is_physical(face))
            faces[static_cast<std::size_t>(face.ordinal())] = {PhysicalBoundaryKind::dirichlet,
                                                               Real(0), Real(1), Real(0)};
        }
      }
      const PhysicalBoundaryConditions<Dim> boundary{boundary_topology, faces, spacing};
      const field_type& state = levels[level];
      request.levels.push_back(
          {geometry, boundary, state.layout(), state.distribution(), state.local_rank()});
      boundaries.push_back({geometry, boundary});
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = *topology.lane;
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program forward tensor preparation failed collectively");
  }
  forward_hierarchy_tensor_solver_ = prepare_hierarchy_tensor_solver_collectively(
      *owner_->hierarchy_tensor_solver_registry_, selection.provider_identity, std::move(request),
      lane);
  forward_hierarchy_tensor_boundaries_ = std::move(boundaries);
  forward_hierarchy_tensor_topology_epoch_ = authority.topology_epoch();
  forward_hierarchy_tensor_materialization_generation_ = authority.materialization_generation();
}

void prepare_forward_accepted_state_staging_(
    const PreparedForwardAmrExecutionAuthority& authority) {
  const std::size_t configured_levels = authority.configured_level_count();
  const std::size_t active_levels = authority.active_level_count();
  if (!accepted_state_staging_.prepared_envelope || configured_levels == 0 || active_levels == 0 ||
      active_levels > configured_levels ||
      accepted_state_staging_.configured_level_count != configured_levels)
    throw std::logic_error(
        "AMR Program forward accepted-state staging has no matching prepared capacity");
  reset_forward_accepted_state_staging_(authority.topology_epoch(),
                                        authority.materialization_generation());
}

void seal_forward_accepted_state_staging_() {
  auto& staging = accepted_state_staging_;
  // The forward image is the only cold boundary at which the replacement ledger templates and
  // their accepted checkpoint envelope coexist.  Seal the non-ledger portion of that image here
  // as well: after HiddenPublish the first Program step can mutate the resident ledger clocks, so
  // replaying the allocating/string-key cold prime there would compare dynamic keys against the
  // bind-time wire templates and reject a valid replacement.
  copy_temporal_partition_preallocated_(staging.state.temporal_partition,
                                        accepted_temporal_partition_);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    const auto& slots = staging.accepted_face_flux_slots[index];
    if (!staging.state.accepted_face_flux[index].empty() ||
        !staging.accepted_face_flux_sources[index].empty() ||
        !staging.accepted_face_flux_active_slots[index].empty() ||
        staging.state.accepted_face_flux[index].capacity() < slots.size() ||
        staging.accepted_face_flux_sources[index].capacity() < slots.size() ||
        staging.accepted_face_flux_active_slots[index].capacity() < slots.size() ||
        accepted_face_flux_[index].capacity() < accepted_face_flux_slots_[index].size() ||
        accepted_face_flux_slots_[index].size() != slots.size())
      throw std::logic_error(
          "AMR Program forward accepted face-flux image was not sealed before publication");
  }
  if (!staging.state.synchronization_events.empty() ||
      !staging.synchronization_event_active_indices.empty() ||
      !staging.state.accepted_interface_flux.empty() ||
      !staging.accepted_interface_flux_active_slots.empty() ||
      !accepted_interface_flux_staging_sources_.empty() ||
      accepted_synchronization_events_.capacity() <
          accepted_synchronization_event_slots_.size() ||
      accepted_synchronization_event_slots_.size() !=
          staging.synchronization_event_slots.size())
    throw std::logic_error(
        "AMR Program forward accepted effect image was not sealed before publication");
  staging.primed = true;
}

void reset_forward_accepted_state_staging_(std::uint64_t topology_epoch,
                                           std::uint64_t materialization_generation) {
  auto& staging = accepted_state_staging_;
  if (!staging.prepared_envelope || staging.configured_level_count == 0 ||
      topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
      materialization_generation == std::numeric_limits<std::uint64_t>::max())
    throw std::logic_error(
        "AMR Program forward accepted-state staging reset has no prepared authority");

  // The detached copy may contain the old logical image.  Return its string/payload ownership
  // to the presealed pools before making the new topology's logical image empty.  This is a
  // Candidate-only compaction; the accepted owner is never mutated here.
  for (std::size_t index = 0; index < staging.state.history_slots.size(); ++index) {
    if (index >= staging.history_slot_active_indices.size())
      throw std::logic_error("AMR Program forward staging history active slots are malformed");
    const std::size_t pool = staging.history_slot_active_indices[index];
    if (pool >= staging.history_slot_pool.size())
      throw std::logic_error("AMR Program forward staging history pool is malformed");
    staging.state.history_slots[index].name.swap(staging.history_slot_pool[pool].name);
  }
  staging.state.history_slots.clear();
  staging.history_slot_active_indices.clear();
  for (std::size_t index = 0; index < staging.state.pending_history_remaps.size(); ++index) {
    if (index >= staging.pending_history_active_slots.size())
      throw std::logic_error("AMR Program forward staging pending slots are malformed");
    const std::size_t pool = staging.pending_history_active_slots[index];
    if (pool >= staging.pending_history_remap_slots.size())
      throw std::logic_error("AMR Program forward staging pending pool is malformed");
    staging.state.pending_history_remaps[index].key.swap(
        staging.pending_history_remap_slots[pool].key);
  }
  staging.state.pending_history_remaps.clear();
  staging.pending_history_active_slots.clear();
  for (int axis = 0; axis < Dim; ++axis) {
    const auto slot_axis = static_cast<std::size_t>(axis);
    auto& logical = staging.state.accepted_face_flux[slot_axis];
    auto& slots = staging.accepted_face_flux_slots[slot_axis];
    auto& active = staging.accepted_face_flux_active_slots[slot_axis];
    if (logical.size() != active.size())
      throw std::logic_error("AMR Program forward staging face-flux active slots are malformed");
    for (std::size_t index = 0; index < logical.size(); ++index) {
      if (active[index] >= slots.size())
        throw std::logic_error("AMR Program forward staging face-flux pool is malformed");
      auto& pool = slots[active[index]];
      pool.key.owner = std::move(logical[index].key.owner);
      pool.key.state = std::move(logical[index].key.state);
      pool.key.stage = std::move(logical[index].key.stage);
      pool.payload = std::move(logical[index].payload);
    }
    logical.clear();
    active.clear();
    staging.accepted_face_flux_sources[slot_axis].clear();
    if (logical.capacity() < slots.size())
      logical.reserve(slots.size());
    if (staging.accepted_face_flux_sources[slot_axis].capacity() < slots.size())
      staging.accepted_face_flux_sources[slot_axis].reserve(slots.size());
    if (active.capacity() < slots.size())
      active.reserve(slots.size());
  }
  for (std::size_t index = 0; index < staging.state.synchronization_events.size(); ++index) {
    if (index >= staging.synchronization_event_active_indices.size())
      throw std::logic_error("AMR Program forward staging event active slots are malformed");
    const std::size_t pool = staging.synchronization_event_active_indices[index];
    if (pool >= staging.synchronization_event_slots.size())
      throw std::logic_error("AMR Program forward staging event pool is malformed");
    staging.state.synchronization_events[index].phase.swap(
        staging.synchronization_event_slots[pool].phase);
  }
  staging.state.synchronization_events.clear();
  staging.synchronization_event_active_indices.clear();
  staging.state.accepted_interface_flux.clear();
  staging.accepted_interface_flux_slots.clear();
  staging.accepted_interface_flux_active_slots.clear();
  accepted_interface_flux_staging_sources_.clear();
  staging.state.level_clocks.clear();
  if (staging.state.level_clocks.capacity() < staging.configured_level_count)
    staging.state.level_clocks.reserve(staging.configured_level_count);
  staging.topology_epoch = topology_epoch;
  staging.materialization_generation = materialization_generation;
  staging.primed = false;
  staging.valid = false;
}

void prime_accepted_state_staging_from_cold_source_(const AcceptedContextSnapshot& source) {
  prime_accepted_state_staging_from_cold_staging_(source.accepted_state_staging_);
}

void prime_accepted_state_staging_from_cold_staging_(const AcceptedStateStaging<Dim>& cold) {
  auto& staging = accepted_state_staging_;
  if (!cold.prepared_envelope)
    return;
  if (!staging.prepared_envelope || staging.configured_level_count != cold.configured_level_count)
    throw std::logic_error("AMR Program accepted-state cold copy changed its prepared envelope");
  const auto reserve_like = [](auto& destination, const auto& source_values) {
    if (destination.capacity() < source_values.capacity())
      destination.reserve(source_values.capacity());
  };
  const auto reserve_string = [](std::string& destination, const std::string& source_value) {
    if (destination.capacity() < source_value.capacity())
      destination.reserve(source_value.capacity());
  };
  const auto require_same_size = [](const auto& destination, const auto& source_values,
                                    const char* what) {
    if (destination.size() != source_values.size())
      throw std::logic_error(std::string("AMR Program accepted-state cold copy changed ") + what);
  };
  reserve_string(staging.state.spatial_contract, cold.state.spatial_contract);
  reserve_like(staging.state.level_clocks, cold.state.level_clocks);
  reserve_like(staging.state.histories, cold.state.histories);
  require_same_size(staging.state.histories, cold.state.histories, "history descriptors");
  for (std::size_t index = 0; index < cold.state.histories.size(); ++index) {
    reserve_string(staging.state.histories[index].name, cold.state.histories[index].name);
    reserve_string(staging.state.histories[index].state_identity,
                   cold.state.histories[index].state_identity);
    reserve_string(staging.state.histories[index].space_identity,
                   cold.state.histories[index].space_identity);
    reserve_string(staging.state.histories[index].clock_identity,
                   cold.state.histories[index].clock_identity);
    reserve_string(staging.state.histories[index].interpolation_identity,
                   cold.state.histories[index].interpolation_identity);
  }
  reserve_like(staging.state.history_slots, cold.state.history_slots);
  reserve_like(staging.history_slot_pool, cold.history_slot_pool);
  require_same_size(staging.history_slot_pool, cold.history_slot_pool, "history slot pool");
  for (std::size_t index = 0; index < cold.history_slot_pool.size(); ++index)
    reserve_string(staging.history_slot_pool[index].name, cold.history_slot_pool[index].name);
  reserve_like(staging.history_slot_bindings, cold.history_slot_bindings);
  require_same_size(staging.history_slot_bindings, cold.history_slot_bindings,
                    "history slot bindings");
  for (std::size_t index = 0; index < cold.history_slot_bindings.size(); ++index)
    reserve_string(staging.history_slot_bindings[index].key, cold.history_slot_bindings[index].key);
  reserve_like(staging.history_slot_active_indices, cold.history_slot_active_indices);
  reserve_like(staging.state.pending_history_remaps, cold.state.pending_history_remaps);
  require_same_size(staging.state.pending_history_remaps, cold.state.pending_history_remaps,
                    "pending remap image");
  for (std::size_t index = 0; index < cold.state.pending_history_remaps.size(); ++index)
    reserve_string(staging.state.pending_history_remaps[index].key,
                   cold.state.pending_history_remaps[index].key);
  reserve_like(staging.pending_history_remap_slots, cold.pending_history_remap_slots);
  require_same_size(staging.pending_history_remap_slots, cold.pending_history_remap_slots,
                    "pending remap slots");
  for (std::size_t index = 0; index < cold.pending_history_remap_slots.size(); ++index)
    reserve_string(staging.pending_history_remap_slots[index].key,
                   cold.pending_history_remap_slots[index].key);
  reserve_like(staging.pending_history_keys, cold.pending_history_keys);
  require_same_size(staging.pending_history_keys, cold.pending_history_keys, "pending remap keys");
  for (std::size_t index = 0; index < cold.pending_history_keys.size(); ++index)
    reserve_string(staging.pending_history_keys[index], cold.pending_history_keys[index]);
  reserve_like(staging.pending_history_active_slots, cold.pending_history_active_slots);
  reserve_like(staging.state.history_flux_payload, cold.state.history_flux_payload);
  reserve_string(staging.state.temporal_partition.provider_identity,
                 cold.state.temporal_partition.provider_identity);
  reserve_like(staging.state.temporal_partition.cells, cold.state.temporal_partition.cells);
  reserve_like(staging.state.tagging_hysteresis_state, cold.state.tagging_hysteresis_state);
  reserve_string(staging.state.flux_budget_contract, cold.state.flux_budget_contract);
  reserve_string(staging.state.coupling_contract, cold.state.coupling_contract);
  reserve_like(staging.state.synchronization_events, cold.state.synchronization_events);
  require_same_size(staging.state.synchronization_events, cold.state.synchronization_events,
                    "synchronization events");
  for (std::size_t index = 0; index < cold.state.synchronization_events.size(); ++index)
    reserve_string(staging.state.synchronization_events[index].phase,
                   cold.state.synchronization_events[index].phase);
  reserve_like(staging.synchronization_event_slots, cold.synchronization_event_slots);
  require_same_size(staging.synchronization_event_slots, cold.synchronization_event_slots,
                    "synchronization event slots");
  for (std::size_t index = 0; index < cold.synchronization_event_slots.size(); ++index)
    reserve_string(staging.synchronization_event_slots[index].phase,
                   cold.synchronization_event_slots[index].phase);
  reserve_like(staging.synchronization_event_active_indices,
               cold.synchronization_event_active_indices);
  reserve_like(staging.state.accepted_interface_flux, cold.state.accepted_interface_flux);
  require_same_size(staging.state.accepted_interface_flux, cold.state.accepted_interface_flux,
                    "interface-flux image");
  const auto prime_interface_fragment = [&](auto& destination, const auto& source_fragment) {
    reserve_string(destination.key.interface_identity, source_fragment.key.interface_identity);
    reserve_string(destination.key.stage_identity, source_fragment.key.stage_identity);
    reserve_string(destination.key.graph_identity, source_fragment.key.graph_identity);
    reserve_string(destination.key.rate_identity, source_fragment.key.rate_identity);
    reserve_string(destination.key.application_identity, source_fragment.key.application_identity);
    reserve_like(destination.payload, source_fragment.payload);
  };
  for (std::size_t index = 0; index < cold.state.accepted_interface_flux.size(); ++index)
    prime_interface_fragment(staging.state.accepted_interface_flux[index],
                             cold.state.accepted_interface_flux[index]);
  reserve_like(staging.accepted_interface_flux_slots, cold.accepted_interface_flux_slots);
  require_same_size(staging.accepted_interface_flux_slots, cold.accepted_interface_flux_slots,
                    "interface-flux slots");
  for (std::size_t index = 0; index < cold.accepted_interface_flux_slots.size(); ++index)
    prime_interface_fragment(staging.accepted_interface_flux_slots[index],
                             cold.accepted_interface_flux_slots[index]);
  reserve_like(staging.accepted_interface_flux_active_slots,
               cold.accepted_interface_flux_active_slots);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    reserve_like(staging.state.accepted_face_flux[index], cold.state.accepted_face_flux[index]);
    reserve_like(staging.accepted_face_flux_slots[index], cold.accepted_face_flux_slots[index]);
    require_same_size(staging.state.accepted_face_flux[index], cold.state.accepted_face_flux[index],
                      "face-flux image");
    require_same_size(staging.accepted_face_flux_slots[index], cold.accepted_face_flux_slots[index],
                      "face-flux slots");
    const auto prime_face_fragment = [&](auto& destination, const auto& source_fragment) {
      reserve_string(destination.key.owner, source_fragment.key.owner);
      reserve_string(destination.key.state, source_fragment.key.state);
      reserve_string(destination.key.stage, source_fragment.key.stage);
      reserve_like(destination.payload, source_fragment.payload);
    };
    for (std::size_t slot = 0; slot < cold.state.accepted_face_flux[index].size(); ++slot)
      prime_face_fragment(staging.state.accepted_face_flux[index][slot],
                          cold.state.accepted_face_flux[index][slot]);
    for (std::size_t slot = 0; slot < cold.accepted_face_flux_slots[index].size(); ++slot)
      prime_face_fragment(staging.accepted_face_flux_slots[index][slot],
                          cold.accepted_face_flux_slots[index][slot]);
    reserve_like(staging.accepted_face_flux_sources[index], cold.accepted_face_flux_sources[index]);
    reserve_like(staging.accepted_face_flux_active_slots[index],
                 cold.accepted_face_flux_active_slots[index]);
  }
}
