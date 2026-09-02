// Accepted-history lifecycle is cold/requalification work, kept outside the hot checkpoint staging authority.
// This is intentionally a class-definition fragment included by history_checkpoint_runtime.hpp.

void refresh_accepted_hierarchy_state_(bool prepare_subcycling = true) const {
  require_facade_execution_();
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR Program accepted-state refresh crossed an active attempt");
  refresh_resources_();
  const std::size_t levels = runtime_->hierarchy().num_levels();
  if (accepted_checkpoint_level_clock_slots_.capacity() < levels) {
    // Explicit bootstrap invokes the v5 hierarchy-refresh hook after the candidate adapter has
    // been bound to its final facade but before the first accepted snapshot is captured.  This
    // is the only legal allocating refresh: the synchronised resource epoch/generation, invalid
    // accepted-state revision and empty byte arena together authenticate that pre-snapshot state.
    // Every accepted/attempt path has a revision or a primed byte arena and must continue to
    // fail closed rather than silently growing its hot checkpoint storage.
    const bool detached_bootstrap_before_first_snapshot =
        accepted_state_revision_ == std::numeric_limits<std::uint64_t>::max() &&
        accepted_checkpoint_candidate_bytes_.capacity() == 0 &&
        resource_epoch_ == runtime_->topology_epoch() &&
        resource_generation_ == runtime_->materialization_generation();
    if (!detached_bootstrap_before_first_snapshot)
      throw std::logic_error("AMR Program checkpoint level-clock storage was not primed");
    bind_accepted_checkpoint_candidate_buffer_();
  }
  requalify_cell_temporal_configuration_();
  if (prepare_subcycling)
    prepare_multiblock_subcycling_engine_();
  const bool cold_prime = !accepted_state_staging_.primed;
  if (cold_prime)
    prime_accepted_state_staging_at_bind_();
  const auto& history_manager = runtime_state().hist_;
  if (prepared_history_mutation_epoch_ != resource_epoch_ ||
      prepared_history_mutation_generation_ != resource_generation_ ||
      accepted_history_ordinal_owner_ != std::addressof(history_manager) ||
      accepted_history_ordinal_epoch_ != resource_epoch_ ||
      accepted_history_ordinal_generation_ != resource_generation_ ||
      prepared_history_mutation_slots_.size() != history_manager.histories.size() ||
      accepted_history_binding_mutation_slots_.size() !=
          accepted_state_staging_.history_slot_bindings.size() ||
      accepted_pending_history_ordinal_sources_.size() !=
          accepted_state_staging_.pending_history_keys.size())
    prime_history_mutation_workspace_at_bind_();
  fill_accepted_state_staging_();
  AmrProgramAcceptedState<Dim>* state = std::addressof(accepted_state_staging_.state);
  if (state->flux_budget_contract.empty() || state->coupling_contract.empty())
    throw std::logic_error("AMR Program accepted-state staging produced empty temporal contracts");
  require_accepted_state_staging_commit_preallocated_(*state);
  if (interface_flux_ledger_) {
    runtime::program::serialize_prevalidated_amr_program_accepted_state_with_interface_views_into(
        *state, accepted_interface_flux_staging_sources_.size(),
        [&](const auto& visitor) {
          for (const auto& source : accepted_interface_flux_staging_sources_)
            visitor(source);
        },
        accepted_checkpoint_candidate_bytes_);
  } else {
    runtime::program::serialize_prevalidated_amr_program_accepted_state_into(
        *state, accepted_checkpoint_candidate_bytes_);
  }
  facade_->program_publish_prevalidated_accepted_state_(accepted_checkpoint_candidate_bytes_);
  accepted_state_revision_ = facade_->program_accepted_state_revision_();
  commit_accepted_state_staging_noexcept_(*state);
  if (accepted_flux_budget_contract_.empty() || accepted_coupling_contract_.empty())
    throw std::logic_error("AMR Program accepted-state staging commit erased temporal contracts");
}

void refresh_accepted_hierarchy_state_after_remap_(
    const AmrProgramHistoryRemapDescriptor& descriptor) const {
  struct RemapCandidate {
    std::map<std::string, AmrProgramPendingHistoryRemap> pending;
    std::map<std::string, field_type> deferred_scratches;
    std::string exact_contract;
  };
  RemapCandidate candidate = prepare_history_mutation_collectively_(
      [&]() {
        require_facade_execution_();
        if (!active_attempt_states_.empty())
          throw std::logic_error("AMR Program accepted history remap crossed an active attempt");
        RemapCandidate staged{pending_history_remaps_, deferred_history_lag_scratches_, {}};
        if (descriptor.child_physical_layout_changed && descriptor.child_published &&
            (descriptor.temporal_numerator == 1 || descriptor.temporal_numerator == 2) &&
            descriptor.temporal_denominator == 1 && descriptor.integral_only) {
          for (const AmrProgramHistoryRemapEntry& entry : descriptor.history_plan) {
            if (entry.source != AmrProgramHistoryRemapSource::ParentDeferred)
              continue;
            const std::string& key = entry.key;
            const auto found = runtime_state().hist_.histories.find(key);
            if (found == runtime_state().hist_.histories.end())
              throw std::runtime_error(
                  "AMR Program deferred history remap plan lacks its child ring");
            const auto& ring = found->second;
            const auto decoded = decode_history_key_(key);
            if (!decoded || decoded->first != descriptor.child_level ||
                !runtime_state().hist_.initialized.at(key))
              continue;
            const auto& dts = runtime_state().hist_.slot_dt.at(key);
            if (ring.size() != 2 || dts.size() != 2 || runtime_state().hist_.depth.at(key) != 2 ||
                runtime_state().hist_.store_pending.at(key) || !(dts[1] > Real(0)))
              throw std::runtime_error(
                  "AMR Program deferred history remap has an unsupported child ring");
            const auto prior_pending = staged.pending.find(key);
            if (prior_pending != staged.pending.end() && !prior_pending->second.consumed)
              throw std::runtime_error(
                  "AMR Program deferred history remap would supersede a pending lag");
            const double source_dt = static_cast<double>(dts[1]);
            AmrProgramPendingHistoryRemap next{
                key, descriptor.parent_level, descriptor.child_level,
                descriptor.prior_topology_epoch, descriptor.prior_materialization_generation,
                descriptor.published_topology_epoch,
                descriptor.published_materialization_generation, descriptor.accepted_macro_step,
                descriptor.temporal_numerator, descriptor.temporal_denominator, source_dt,
                source_dt / static_cast<double>(descriptor.temporal_numerator), false};
            if (prior_pending == staged.pending.end())
              staged.pending.emplace(key, std::move(next));
            else
              prior_pending->second = std::move(next);
            auto [scratch, inserted] = staged.deferred_scratches.try_emplace(key, ring.front());
            (void)inserted;
            require_same_field_contract_(scratch->second, ring.front(),
                                         "AMR Program deferred history remap scratch");
          }
        }
        ExactContractBuilder contract;
        contract.text("pops.amr.accepted-history-remap.v1")
            .scalar(descriptor.parent_level)
            .scalar(descriptor.child_level)
            .scalar(descriptor.child_published)
            .scalar(descriptor.child_physical_layout_changed)
            .scalar(descriptor.prior_topology_epoch)
            .scalar(descriptor.prior_materialization_generation)
            .scalar(descriptor.published_topology_epoch)
            .scalar(descriptor.published_materialization_generation)
            .scalar(descriptor.accepted_macro_step)
            .scalar(descriptor.temporal_numerator)
            .scalar(descriptor.temporal_denominator)
            .scalar(descriptor.integral_only)
            .scalar(static_cast<std::uint64_t>(descriptor.history_plan.size()))
            .scalar(static_cast<std::uint64_t>(staged.pending.size()))
            .scalar(static_cast<std::uint64_t>(staged.deferred_scratches.size()));
        for (const AmrProgramHistoryRemapEntry& entry : descriptor.history_plan)
          contract.bytes(entry.key).bytes(entry.parent_key).scalar(entry.source);
        for (const auto& [key, marker] : staged.pending)
          contract.bytes(key)
              .scalar(marker.parent_level)
              .scalar(marker.child_level)
              .scalar(std::bit_cast<std::uint64_t>(marker.source_dt))
              .scalar(std::bit_cast<std::uint64_t>(marker.target_dt));
        staged.exact_contract = std::move(contract).release();
        return staged;
      },
      [](const RemapCandidate& staged) -> const std::string& { return staged.exact_contract; },
      "AMR Program accepted history remap");
  try {
    refresh_resources_after_accepted_history_remap_(descriptor);
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted history remap cannot refresh resources: " +
                             std::string(exception.what()));
  }
  try {
    requalify_cell_temporal_configuration_();
  } catch (const std::exception& exception) {
    throw std::runtime_error(
        "AMR Program accepted history remap cannot requalify temporal state: " +
        std::string(exception.what()));
  }
  try {
    prepare_multiblock_subcycling_engine_();
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted history remap cannot prepare flux budget: " +
                             std::string(exception.what()));
  }
  struct AcceptedImageCandidate {
    AmrProgramAcceptedState<Dim> state;
    std::vector<std::uint8_t> bytes;
    std::string exact_contract;
  };
  AcceptedImageCandidate image = prepare_history_mutation_collectively_(
      [&]() {
        AcceptedImageCandidate staged;
        staged.state = accepted_state_();
        staged.state.pending_history_remaps.clear();
        staged.state.pending_history_remaps.reserve(candidate.pending.size());
        for (const auto& [key, marker] : candidate.pending) {
          if (key != marker.key)
            throw std::logic_error("AMR Program accepted history remap staged a foreign key");
          if (marker.consumed)
            continue;
          staged.state.pending_history_remaps.push_back(marker);
        }
        staged.bytes = serialize_amr_program_accepted_state(staged.state);
        staged.exact_contract.assign(reinterpret_cast<const char*>(staged.bytes.data()),
                                     staged.bytes.size());
        return staged;
      },
      [](const AcceptedImageCandidate& staged) -> const std::string& {
        return staged.exact_contract;
      },
      "AMR Program accepted history remap image");
  try {
    // All allocation, serialization, and exact byte agreement completed above.  Every prepared
    // rank now enters the facade's collective publication with the same immutable image.
    facade_->restore_program_accepted_state(image.bytes);
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted history remap cannot publish accepted state: " +
                             std::string(exception.what()));
  }
  accepted_state_revision_ = facade_->program_accepted_state_revision_();
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_flux_budget_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_coupling_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
  std::swap(accepted_temporal_partition_, image.state.temporal_partition);
  accepted_flux_budget_contract_.swap(image.state.flux_budget_contract);
  accepted_coupling_contract_.swap(image.state.coupling_contract);
  std::swap(accepted_face_flux_, image.state.accepted_face_flux);
  accepted_synchronization_events_.swap(image.state.synchronization_events);
  static_assert(std::is_nothrow_swappable_v<decltype(pending_history_remaps_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(deferred_history_lag_scratches_)>);
  pending_history_remaps_.swap(candidate.pending);
  deferred_history_lag_scratches_.swap(candidate.deferred_scratches);
  // The detached candidate owns value maps; restore the live adapter's non-owning dense ordinals
  // only after their successful accepted publication and value swaps.  A rejected candidate never
  // reaches this point and therefore leaves the prior ordinal/image pair untouched.
  prime_history_mutation_workspace_at_bind_();
  rebind_accepted_face_flux_ordinals_preallocated_noexcept_();
  rebind_accepted_interface_flux_ordinals_preallocated_noexcept_();
}

void preflight_restart_regrid_() const {
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR RegridOnRestart requires an accepted Program boundary");
  refresh_resources_();
  requalify_cell_temporal_configuration_();
  import_accepted_state_(true);
  if (accepted_temporal_partition_.kind == TemporalPartitionKind::Global) {
    require_regrid_rematerializable_temporal_partition(accepted_temporal_partition_);
    return;
  }
  if (accepted_temporal_partition_.provider_identity != kSameLevelTransportEulerStageFluxProvider ||
      !cell_temporal_configuration_ ||
      accepted_temporal_partition_ !=
          cell_temporal_full_partition_(*cell_temporal_configuration_,
                                        accepted_temporal_partition_.synchronization_tick))
    throw std::runtime_error(
        "AMR RegridOnRestart cannot rematerialize this cell-local temporal provider");
}

void restart_regrid_() const {
  preflight_restart_regrid_();
  begin_restart_regrid_history_sequence();
  try {
    for (int parent = 0; parent + 1 < facade_->program_configured_n_levels_(); ++parent) {
      (void)facade_->execute_prepared_tagging(parent);
      if (!facade_->regrid_from_prepared_tagging(parent))
        break;
    }
    end_restart_regrid_history_sequence();
  } catch (...) {
    end_restart_regrid_history_sequence();
    throw;
  }
  multiblock_subcycling_.reset();
  accepted_face_flux_ = {};
  interface_flux_commit_guard_.reset();
  interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
      runtime_->topology_epoch(), inactive_interface_flux_budget_());
  accepted_synchronization_events_.clear();
  // Publish the cleared image before recreating the subcycling engine. Preparing first
  // would make accepted_state_() synthesize reflux/average_down reports for a topology
  // that has not taken a step.
  refresh_accepted_hierarchy_state_(false);
  prepare_multiblock_subcycling_engine_();
}

void resync_after_restart_() const {
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR restart resynchronization crossed an active Program attempt");
  multiblock_subcycling_.reset();
  // rebuild_hierarchy publishes a temporary newer topology epoch; restore_checkpoint_counters
  // then republishes the authenticated checkpoint epoch. The interface-flux ledger followed
  // the temporary epoch and must be requalified rather than treated as a live regression.
  if (interface_flux_ledger_ &&
      interface_flux_ledger_->topology_epoch() > runtime_->topology_epoch()) {
    interface_flux_commit_guard_.reset();
    interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
        runtime_->topology_epoch(), inactive_interface_flux_budget_());
  }
  try {
    synchronize_resource_generation_();
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR restart resynchronization resource refresh failed: " +
                             std::string(exception.what()));
  }
  try {
    import_accepted_state_(true);
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR restart resynchronization accepted-state import failed: " +
                             std::string(exception.what()));
  }
  // The pre-import stamp makes the freshly rebuilt live hierarchy available to the
  // authenticated decoder.  It intentionally cannot qualify the decoded rings: at
  // that point the native restart materializer has not yet installed their complete
  // all-level provenance.  Stamp them only after the full accepted image (including
  // its exact flux payload) has passed validation and been published.
  try {
    synchronize_resource_generation_();
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR restart resynchronization history qualification failed: " +
                             std::string(exception.what()));
  }
}

/// Validate the manager that the AMR engine has already remapped and exchanged in its outer
/// transaction.  This does not prepare, allocate, or register histories: it merely authenticates
/// the exact replacement against the previously live logical registry and the now-live hierarchy.
///
/// The metadata preflight is deliberately facade-free.  A rank-local malformed candidate must
/// converge before another rank can reach one of the later hierarchy/provider observations.
void validate_accepted_history_remap_metadata_() const {
  const auto& manager = runtime_state().hist_;
  if (history_levels_.empty() || manager.histories.empty())
    throw std::runtime_error(
        "AMR Program accepted history remap requires both prior and remapped history registries");

  const auto require_exact_metadata_keys = [&](const auto& metadata) {
    if (metadata.size() != manager.histories.size())
      throw std::runtime_error("AMR Program accepted history remap metadata is incomplete");
    for (const auto& [key, ring] : manager.histories) {
      (void)ring;
      if (!metadata.contains(key))
        throw std::runtime_error("AMR Program accepted history remap metadata is incomplete");
    }
  };
  require_exact_metadata_keys(manager.depth);
  require_exact_metadata_keys(manager.initialized);
  require_exact_metadata_keys(manager.fill_count);
  require_exact_metadata_keys(manager.store_pending);
  require_exact_metadata_keys(manager.owner);
  require_exact_metadata_keys(manager.state_identity);
  require_exact_metadata_keys(manager.space_identity);
  require_exact_metadata_keys(manager.clock_identity);
  require_exact_metadata_keys(manager.interpolation_identity);
  require_exact_metadata_keys(manager.slot_dt);

  const std::size_t live_levels = runtime_->hierarchy().num_levels();
  if (live_levels == 0)
    throw std::runtime_error("AMR Program accepted history remap has no live hierarchy level");
  for (const auto& [key, ring] : manager.histories) {
    const auto decoded = decode_history_key_(key);
    if (!decoded || decoded->first < 0 || static_cast<std::size_t>(decoded->first) >= live_levels ||
        key != history_key_(decoded->second, decoded->first))
      throw std::runtime_error("AMR Program accepted history remap has a malformed ring key");
    const int depth = manager.depth.at(key);
    const int owner = manager.owner.at(key);
    const bool initialized = manager.initialized.at(key);
    const int fill_count = manager.fill_count.at(key);
    const auto& slot_dt = manager.slot_dt.at(key);
    if (depth < 1 || ring.size() != static_cast<std::size_t>(depth) || fill_count < 0 ||
        fill_count > depth || initialized != (fill_count > 0) ||
        slot_dt.size() != static_cast<std::size_t>(depth))
      throw std::runtime_error("AMR Program accepted history remap has invalid ring metadata");
    for (const Real dt : slot_dt) {
      const double value = static_cast<double>(dt);
      if (!std::isfinite(value) || (initialized ? !(value > 0.0) : value != 0.0))
        throw std::runtime_error(
            "AMR Program accepted history remap has invalid exact outgoing-dt metadata");
    }
    if (owner < 0 || owner >= n_blocks() || manager.state_identity.at(key).empty() ||
        manager.space_identity.at(key).empty() || manager.clock_identity.at(key).empty() ||
        manager.interpolation_identity.at(key).empty())
      throw std::runtime_error("AMR Program accepted history remap has an invalid descriptor");
  }
}

void validate_accepted_history_remap_() const {
  const auto& manager = runtime_state().hist_;
  std::set<std::string> prior_names;
  for (const auto& [key, level] : history_levels_) {
    const auto decoded = decode_history_key_(key);
    if (!decoded || decoded->first != level)
      throw std::runtime_error("AMR Program retained history registry is malformed before remap");
    prior_names.insert(decoded->second);
  }

  const std::size_t live_levels = runtime_->hierarchy().num_levels();
  std::set<std::string> remapped_names;
  struct Descriptor {
    int owner = -1;
    int depth = 0;
    int components = 0;
    std::string state;
    std::string space;
    std::string clock;
    std::string interpolation;

    bool operator==(const Descriptor&) const = default;
  };
  std::map<std::string, Descriptor> descriptors;
  std::map<std::string, std::set<int>> levels_by_name;
  for (const auto& [key, ring] : manager.histories) {
    const auto decoded = decode_history_key_(key);
    const int level = decoded->first;
    const std::string& name = decoded->second;
    const int depth = manager.depth.at(key);
    const int owner = manager.owner.at(key);
    const field_type& live = facade_->program_prepared_amr_block_state_(owner, level);
    for (const field_type& slot : ring)
      if (slot.layout() != live.layout() || slot.distribution() != live.distribution() ||
          slot.local_rank() != live.local_rank() || slot.local_size() != live.local_size() ||
          slot.ghosts() != live.ghosts() || slot.ncomp() < 1)
        throw std::runtime_error(
            "AMR Program accepted history remap ring disagrees with the live hierarchy");
    const Descriptor descriptor{owner,
                                depth,
                                ring.front().ncomp(),
                                manager.state_identity.at(key),
                                manager.space_identity.at(key),
                                manager.clock_identity.at(key),
                                manager.interpolation_identity.at(key)};
    for (const field_type& slot : ring)
      if (slot.ncomp() != descriptor.components)
        throw std::runtime_error(
            "AMR Program accepted history remap ring has inconsistent components");
    auto [descriptor_it, inserted] = descriptors.emplace(name, descriptor);
    if (!inserted && descriptor_it->second != descriptor)
      throw std::runtime_error(
          "AMR Program accepted history remap descriptor differs between hierarchy levels");
    if (!levels_by_name[name].insert(level).second)
      throw std::runtime_error("AMR Program accepted history remap duplicates a hierarchy level");
    remapped_names.insert(name);
  }
  if (prior_names != remapped_names)
    throw std::runtime_error(
        "AMR Program accepted history remap changes the logical history registry");
  for (const auto& [name, levels] : levels_by_name) {
    (void)name;
    if (levels.size() != live_levels)
      throw std::runtime_error("AMR Program accepted history remap omits a live hierarchy level");
    for (std::size_t level = 0; level < live_levels; ++level)
      if (!levels.contains(static_cast<int>(level)))
        throw std::runtime_error("AMR Program accepted history remap omits a live hierarchy level");
  }
}

void rethrow_accepted_history_remap_collective_failure_(std::exception_ptr local_error,
                                                        const ExecutionLane& lane) const {
  if (all_reduce_max(local_error ? 1L : 0L, lane) == 0)
    return;
  if (lane.size() == 1 && local_error)
    std::rethrow_exception(local_error);
  throw std::runtime_error("AMR Program accepted history remap failed collectively");
}

void refresh_resources_after_accepted_history_remap_(
    const AmrProgramHistoryRemapDescriptor& descriptor) const {
  if (facade_ != nullptr)
    facade_->refresh_prepared_amr_levels();
  if (resource_epoch_ == runtime_->topology_epoch() &&
      resource_generation_ == runtime_->materialization_generation())
    throw std::runtime_error(
        "AMR Program accepted history remap requires a newly published hierarchy generation");
  // The numeric rings were just rematerialized by the native transaction.  Build the matching
  // provenance map off-line: no retained expression is replaced until all local validation and
  // the prepared-lane agreement below have completed.
  struct RemappedFluxCandidate {
    std::map<std::string, std::vector<FluxExpression>> expressions;
    std::string exact_contract;
  };
  RemappedFluxCandidate remapped = prepare_history_mutation_collectively_(
      [&]() {
        RemappedFluxCandidate staged;
        std::map<std::string, const AmrProgramHistoryRemapEntry*> plan;
        for (const AmrProgramHistoryRemapEntry& entry : descriptor.history_plan) {
          if (entry.key.empty() || !plan.emplace(entry.key, &entry).second)
            throw std::runtime_error("AMR Program history flux remap has a non-canonical plan");
          if (entry.source == AmrProgramHistoryRemapSource::ParentDeferred &&
              entry.parent_key.empty())
            throw std::runtime_error("AMR Program history flux remap has no parent provenance key");
          if (entry.source != AmrProgramHistoryRemapSource::ParentDeferred &&
              !entry.parent_key.empty())
            throw std::runtime_error(
                "AMR Program history flux remap has a non-canonical retained/removal parent key");
        }
        ExactContractBuilder contract;
        contract.text("pops.amr.history-flux-remap.v1")
            .scalar(runtime_->topology_epoch())
            .scalar(runtime_->materialization_generation())
            .scalar(descriptor.parent_level)
            .scalar(descriptor.child_level)
            .scalar(descriptor.child_published)
            .scalar(descriptor.child_physical_layout_changed)
            .scalar(static_cast<std::uint64_t>(runtime_state().hist_.histories.size()));
        for (const auto& [key, ring] : runtime_state().hist_.histories) {
          const auto decoded = decode_history_key_(key);
          if (!decoded)
            throw std::runtime_error("AMR Program history flux remap has a malformed ring key");
          const auto planned = plan.find(key);
          if (decoded->first == descriptor.child_level && descriptor.child_published &&
              planned == plan.end())
            throw std::runtime_error("AMR Program history flux remap omits an affected child ring");
          if (decoded->first == descriptor.child_level && !descriptor.child_published)
            throw std::runtime_error(
                "AMR Program history flux remap retained a removed child-level ring");
          const auto retained = history_flux_expressions_.find(key);
          const bool source_from_parent =
              planned != plan.end() &&
              planned->second->source == AmrProgramHistoryRemapSource::ParentDeferred;
          if (!source_from_parent && retained != history_flux_expressions_.end() &&
              retained->second.size() == ring.size()) {
            staged.expressions.emplace(key, retained->second);
          } else {
            if (!source_from_parent)
              throw std::runtime_error(
                  "AMR Program history flux remap lacks its retained history provenance");
            const auto parent = history_flux_expressions_.find(planned->second->parent_key);
            if (parent == history_flux_expressions_.end() || parent->second.size() != ring.size())
              throw std::runtime_error(
                  "AMR Program history flux remap lacks its authenticated parent sample");
            staged.expressions.emplace(key, parent->second);
          }
          const auto& slots = staged.expressions.at(key);
          contract.bytes(key).scalar(static_cast<std::uint64_t>(slots.size()));
          for (const FluxExpression& expression : slots)
            contract.scalar(static_cast<std::uint64_t>(expression.size()));
        }
        for (const auto& [key, entry] : plan) {
          const bool live = runtime_state().hist_.histories.contains(key);
          if ((entry->source == AmrProgramHistoryRemapSource::Removed && live) ||
              (entry->source != AmrProgramHistoryRemapSource::Removed && !live))
            throw std::runtime_error("AMR Program history flux remap plan differs from live rings");
        }
        staged.exact_contract = std::move(contract).release();
        return staged;
      },
      [](const RemappedFluxCandidate& staged) -> const std::string& {
        return staged.exact_contract;
      },
      "AMR Program accepted history flux remap");
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr metadata_error;
  try {
    validate_accepted_history_remap_metadata_();
  } catch (...) {
    metadata_error = std::current_exception();
  }
  rethrow_accepted_history_remap_collective_failure_(metadata_error, lane);
  std::exception_ptr validation_error;
  try {
    validate_accepted_history_remap_();
  } catch (...) {
    validation_error = std::current_exception();
  }
  rethrow_accepted_history_remap_collective_failure_(validation_error, lane);
  std::exception_ptr synchronization_error;
  try {
    synchronize_resource_generation_();
  } catch (...) {
    synchronization_error = std::current_exception();
  }
  // The resource-generation tail can allocate a replacement ledger after its own prepared-lane
  // work.  Converge that failure before the caller enters temporal/subcycling collectives; the
  // enclosing AcceptedSnapshot remains the rollback authority for any partial resource refresh.
  rethrow_accepted_history_remap_collective_failure_(synchronization_error, lane);
  if (active_level_ >= nlev())
    active_level_ = 0;
  static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
  history_flux_expressions_.swap(remapped.expressions);
  scratches_.clear();
}
