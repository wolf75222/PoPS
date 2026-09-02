// Accepted-context implementation fragment; included inside AcceptedContextSnapshot.

void require_detached_forward_authority_(std::uint64_t topology_epoch,
                                         std::uint64_t materialization_generation,
                                         const char* operation) const {
  if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
      resource_epoch_ != topology_epoch || resource_generation_ != materialization_generation ||
      history_epoch_ != topology_epoch || history_generation_ != materialization_generation)
    throw std::logic_error(std::string("AMR Program detached accepted context cannot prepare ") +
                           operation);
}

[[nodiscard]] static std::uint64_t forward_cell_count_(
    const PreparedForwardAmrTemporalAuthority& authority, std::size_t block, std::size_t level) {
  if (block >= authority.block_count || level >= authority.level_count ||
      block > (std::numeric_limits<std::size_t>::max() - level) / authority.level_count)
    throw std::logic_error("AMR Program forward temporal authority has an invalid cell index");
  return authority.block_level_cell_counts[block * authority.level_count + level];
}

[[nodiscard]] static std::vector<int> forward_cell_temporal_level_rungs_(
    int finest_rung, const PreparedForwardAmrTemporalAuthority& authority) {
  if (finest_rung < 0 || finest_rung > 30 || authority.level_count == 0 ||
      authority.temporal_relations.size() + 1 != authority.level_count)
    throw std::logic_error(
        "AMR Program forward temporal authority lacks one relation per level transition");
  std::vector<int> rungs(authority.level_count, finest_rung);
  for (std::size_t child = authority.temporal_relations.size(); child != 0; --child) {
    const auto& relation = authority.temporal_relations[child - 1];
    if (relation.parent_level() != static_cast<int>(child - 1) ||
        relation.child_level() != static_cast<int>(child))
      throw std::logic_error(
          "AMR Program forward temporal authority has a non-canonical level relation");
    const auto ratio = relation.temporal_ratio();
    if (ratio.numerator <= 0 || ratio.denominator != 1)
      throw std::invalid_argument(
          "cell-local AMR requires integral power-of-two temporal refinement ratios");
    const auto refinement = static_cast<std::uint64_t>(ratio.numerator);
    if ((refinement & (refinement - 1)) != 0)
      throw std::invalid_argument(
          "cell-local AMR requires power-of-two temporal refinement ratios");
    int exponent = 0;
    for (std::uint64_t value = refinement; value > 1; value >>= 1)
      ++exponent;
    if (rungs[child] > 30 - exponent)
      throw std::invalid_argument("cell-local AMR derived rung exceeds its bounded domain");
    rungs[child - 1] = rungs[child] + exponent;
  }
  return rungs;
}

[[nodiscard]] static std::uint64_t forward_block_major_offset_(
    const CellTemporalConfiguration& configuration,
    const PreparedForwardAmrTemporalAuthority& authority, std::size_t route, std::size_t level) {
  std::uint64_t offset = 0;
  for (std::size_t prior = 0; prior <= route; ++prior) {
    const std::size_t stop = prior == route ? level : authority.level_count;
    const int block = configuration.routes[prior].runtime_block;
    if (block < 0)
      throw std::logic_error("AMR Program forward temporal route has a negative block");
    for (std::size_t prior_level = 0; prior_level < stop; ++prior_level) {
      const std::uint64_t count =
          forward_cell_count_(authority, static_cast<std::size_t>(block), prior_level);
      if (count > std::numeric_limits<std::uint64_t>::max() - offset)
        throw std::overflow_error("cell-local AMR block-major identity exceeds uint64_t");
      offset += count;
    }
  }
  return offset;
}

[[nodiscard]] static CellTemporalPartitionAcceptedState prepare_forward_cell_temporal_partition_(
    CellTemporalConfiguration& configuration, const PreparedForwardAmrTemporalAuthority& authority,
    std::int64_t synchronization_tick) {
  if constexpr (!cell_temporal_host_execution_supported_)
    throw std::invalid_argument(
        "cell-local AMR execution requires a host default execution and memory space");
  if (configuration.clock.empty() || configuration.tick_denominator <= 0 ||
      configuration.rung < 0 || configuration.rung > 30 || configuration.routes.empty() ||
      configuration.routes.size() != authority.block_count || authority.coupling_count != 0 ||
      authority.has_interface_flux_provider ||
      authority.periodic_faces.size() != static_cast<std::size_t>(2 * Dim) ||
      !std::all_of(authority.periodic_faces.begin(), authority.periodic_faces.end(),
                   [](bool periodic) { return periodic; }))
    throw std::invalid_argument(
        "cell-local AMR forward hierarchy requires periodic uncoupled complete routes");

  std::sort(configuration.routes.begin(), configuration.routes.end(),
            [](const auto& left, const auto& right) {
              return std::tie(left.runtime_block, left.program_block, left.rhs_id) <
                     std::tie(right.runtime_block, right.program_block, right.rhs_id);
            });
  for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
    const auto& entry = configuration.routes[route];
    if (entry.program_block < 0 || entry.runtime_block != static_cast<int>(route) ||
        entry.rhs_id < 0 ||
        (route != 0 && configuration.routes[route - 1].program_block == entry.program_block))
      throw std::logic_error(
          "AMR Program forward cell-local routes are not a complete block bijection");
  }

  configuration.topology_epoch = authority.topology_epoch;
  configuration.materialization_generation = authority.materialization_generation;
  configuration.level_rungs = forward_cell_temporal_level_rungs_(configuration.rung, authority);
  configuration.level_cell_counts.clear();
  configuration.level_cell_counts.reserve(authority.level_count);
  for (std::size_t level = 0; level < authority.level_count; ++level) {
    const std::uint64_t count = forward_cell_count_(authority, 0, level);
    for (std::size_t block = 1; block < authority.block_count; ++block)
      if (forward_cell_count_(authority, block, level) != count)
        throw std::logic_error(
            "AMR Program forward cell-local blocks do not share one staged topology");
    configuration.level_cell_counts.push_back(count);
  }

  ExactContractBuilder contract;
  contract.text("pops.amr-program.cell-local-forward-euler")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(configuration.clock)
      .scalar(configuration.tick_denominator)
      .scalar(std::int32_t{configuration.rung})
      .scalar(configuration.topology_epoch)
      .scalar(configuration.materialization_generation)
      .text(authority.lane_identity)
      .bytes(authority.spatial_contract)
      .text("host-default-execution-and-memory")
      .presence(cell_temporal_host_execution_supported_)
      .scalar(std::uint64_t{2 * Dim})
      .scalar(static_cast<std::uint64_t>(authority.coupling_count))
      .presence(authority.has_interface_flux_provider)
      .scalar(static_cast<std::uint64_t>(configuration.routes.size()))
      .sequence(configuration.level_rungs,
                [](ExactContractBuilder& item, int rung) { item.scalar(std::int32_t{rung}); })
      .sequence(configuration.level_cell_counts,
                [](ExactContractBuilder& item, std::uint64_t count) { item.scalar(count); });
  for (bool periodic : authority.periodic_faces)
    contract.presence(periodic);
  for (const auto& route : configuration.routes)
    contract.scalar(std::int32_t{route.program_block})
        .scalar(std::int32_t{route.runtime_block})
        .scalar(std::int32_t{route.rhs_id});
  configuration.exact_contract = std::move(contract).release();

  if (synchronization_tick < 0 ||
      synchronization_tick % (std::int64_t{1} << configuration.level_rungs.front()) != 0)
    throw std::invalid_argument(
        "cell-local AMR forward accepted time is not aligned to its coarsest derived rung");

  CellTemporalPartitionAcceptedState partition;
  partition.kind = TemporalPartitionKind::CellLocal;
  partition.provider_identity = std::string(kSameLevelTransportEulerStageFluxProvider);
  partition.topology_epoch = authority.topology_epoch;
  partition.synchronization_tick = synchronization_tick;
  partition.tick_denominator = configuration.tick_denominator;
  std::uint64_t total_cells = 0;
  for (std::size_t level = 0; level < authority.level_count; ++level)
    for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
      const int block = configuration.routes[route].runtime_block;
      const std::uint64_t count =
          forward_cell_count_(authority, static_cast<std::size_t>(block), level);
      if (count > std::numeric_limits<std::uint64_t>::max() - total_cells)
        throw std::overflow_error("cell-local AMR forward partition exceeds uint64_t");
      total_cells += count;
    }
  if (total_cells == 0 || total_cells > std::numeric_limits<std::size_t>::max())
    throw std::logic_error("cell-local AMR forward partition has an invalid cell capacity");
  partition.cells.reserve(static_cast<std::size_t>(total_cells));
  for (std::size_t level = 0; level < authority.level_count; ++level)
    for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
      std::uint64_t cell = forward_block_major_offset_(configuration, authority, route, level);
      const int block = configuration.routes[route].runtime_block;
      const std::uint64_t count =
          forward_cell_count_(authority, static_cast<std::size_t>(block), level);
      if (count > std::numeric_limits<std::uint64_t>::max() - cell)
        throw std::overflow_error("cell-local AMR forward cell identity exceeds uint64_t");
      for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal)
        partition.cells.push_back({static_cast<int>(level), cell++,
                                   configuration.level_rungs[level], synchronization_tick});
    }
  validate_cell_temporal_partition_state(partition);
  return partition;
}

/// Regrid preparation may allocate and build replacement provenance maps, but it must never
/// query the bound adapter: that adapter still names the last sealed hierarchy.  The descriptor
/// is the sole authority for affected keys; its deferred markers were prepared from the numeric
/// HistoryManager before the forward topology was staged.
void prepare_detached_history_remap_(const AmrProgramHistoryRemapDescriptor& descriptor) {
  if (descriptor.parent_level < 0 || descriptor.child_level < 0 ||
      descriptor.child_level != descriptor.parent_level + 1 ||
      descriptor.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.accepted_macro_step < 0 || descriptor.temporal_numerator <= 0 ||
      descriptor.temporal_denominator <= 0 || descriptor.operation_identity.empty())
    throw std::logic_error("AMR Program detached history remap descriptor is incomplete");

  std::map<std::string, const AmrProgramHistoryRemapEntry*> plan;
  for (const AmrProgramHistoryRemapEntry& entry : descriptor.history_plan) {
    if (entry.key.empty() || !plan.emplace(entry.key, &entry).second)
      throw std::logic_error("AMR Program detached history remap has a non-canonical plan");
    if (entry.source == AmrProgramHistoryRemapSource::ParentDeferred) {
      if (!descriptor.child_published || entry.parent_key.empty())
        throw std::logic_error(
            "AMR Program detached history remap has an invalid parent-deferred entry");
    } else if (!entry.parent_key.empty()) {
      throw std::logic_error(
          "AMR Program detached history remap has a non-canonical retained/removal parent key");
    }
  }

  // Every prior child provenance carrier must be mentioned by the exact plan.  Leaving one
  // outside the plan would retain a pointer-qualified history/lag image across a hierarchy
  // publication merely because the live callback used to rediscover it later.
  for (const auto& [key, level] : history_levels_)
    if (level == descriptor.child_level && !plan.contains(key))
      throw std::logic_error(
          "AMR Program detached history remap omits a prior child history carrier");
  for (const auto& [key, expressions] : history_flux_expressions_) {
    (void)expressions;
    const auto level = history_levels_.find(key);
    if (level == history_levels_.end())
      throw std::logic_error(
          "AMR Program detached history remap found flux provenance without a history carrier");
    if (level->second == descriptor.child_level && !plan.contains(key))
      throw std::logic_error(
          "AMR Program detached history remap omits prior child flux provenance");
  }
  for (const auto& [key, marker] : pending_history_remaps_)
    if (marker.child_level == descriptor.child_level && !plan.contains(key))
      throw std::logic_error(
          "AMR Program detached history remap omits a prior child deferred marker");

  auto next_levels = history_levels_;
  auto next_flux = history_flux_expressions_;
  auto next_pending = pending_history_remaps_;
  auto next_deferred_scratches = deferred_history_lag_scratches_;

  for (const auto& [key, entry] : plan) {
    const auto level = history_levels_.find(key);
    const auto flux = history_flux_expressions_.find(key);
    switch (entry->source) {
      case AmrProgramHistoryRemapSource::Removed:
        if (descriptor.child_published ||
            (level != history_levels_.end() && level->second != descriptor.child_level) ||
            (flux != history_flux_expressions_.end() && level == history_levels_.end()))
          throw std::logic_error("AMR Program detached history remap removes a foreign ring");
        next_levels.erase(key);
        next_flux.erase(key);
        next_pending.erase(key);
        next_deferred_scratches.erase(key);
        break;
      case AmrProgramHistoryRemapSource::RetainedChild:
        if (!descriptor.child_published || level == history_levels_.end() ||
            level->second != descriptor.child_level || flux == history_flux_expressions_.end())
          throw std::logic_error(
              "AMR Program detached history remap lacks retained child provenance");
        // A lag marker refers to the old child geometry and cannot survive an accepted
        // topology transition without a fresh, descriptor-authenticated replacement.
        next_pending.erase(key);
        next_deferred_scratches.erase(key);
        break;
      case AmrProgramHistoryRemapSource::ParentDeferred: {
        if (!descriptor.child_physical_layout_changed)
          throw std::logic_error(
              "AMR Program detached history remap defers a parent without a physical child "
              "change");
        const auto parent_level = history_levels_.find(entry->parent_key);
        const auto parent_flux = history_flux_expressions_.find(entry->parent_key);
        if (parent_level == history_levels_.end() ||
            parent_level->second != descriptor.parent_level ||
            parent_flux == history_flux_expressions_.end() || parent_flux->second.empty())
          throw std::logic_error(
              "AMR Program detached history remap lacks authenticated parent provenance");
        if (level != history_levels_.end() && level->second != descriptor.child_level)
          throw std::logic_error(
              "AMR Program detached history remap reuses a foreign child history key");
        if (flux != history_flux_expressions_.end() &&
            flux->second.size() != parent_flux->second.size())
          throw std::logic_error(
              "AMR Program detached history remap changes the child provenance ring depth");
        next_levels.insert_or_assign(key, descriptor.child_level);
        next_flux.insert_or_assign(key, parent_flux->second);
        next_pending.erase(key);
        next_deferred_scratches.erase(key);
        break;
      }
    }
  }

  std::set<std::string> marker_keys;
  for (const AmrProgramPendingHistoryRemap& marker : descriptor.prepared_pending_history_remaps) {
    const auto entry = plan.find(marker.key);
    if (marker.key.empty() || !marker_keys.insert(marker.key).second || entry == plan.end() ||
        entry->second->source != AmrProgramHistoryRemapSource::ParentDeferred ||
        marker.parent_level != descriptor.parent_level ||
        marker.child_level != descriptor.child_level ||
        marker.prior_topology_epoch != descriptor.prior_topology_epoch ||
        marker.prior_materialization_generation != descriptor.prior_materialization_generation ||
        marker.published_topology_epoch != descriptor.published_topology_epoch ||
        marker.published_materialization_generation !=
            descriptor.published_materialization_generation ||
        marker.accepted_macro_step != descriptor.accepted_macro_step ||
        marker.temporal_numerator != descriptor.temporal_numerator ||
        marker.temporal_denominator != descriptor.temporal_denominator || marker.consumed ||
        !(marker.source_dt > 0.0) || !(marker.target_dt > 0.0) ||
        marker.target_dt != marker.source_dt / static_cast<double>(marker.temporal_numerator))
      throw std::logic_error(
          "AMR Program detached history remap has a non-canonical deferred marker");
    if (!next_levels.contains(marker.key) || !next_flux.contains(marker.key))
      throw std::logic_error(
          "AMR Program detached history remap marker has no prepared child provenance");
    next_pending.insert_or_assign(marker.key, marker);
  }

  // A marker can only be prepared by the exact direct-child IntegralOnly route.  If that route
  // was not selected, a non-empty marker DTO is a construction error rather than a late live
  // fallback.
  if (!descriptor.prepared_pending_history_remaps.empty() &&
      (!descriptor.child_physical_layout_changed || !descriptor.child_published ||
       !descriptor.integral_only || descriptor.temporal_denominator != 1 ||
       (descriptor.temporal_numerator != 1 && descriptor.temporal_numerator != 2)))
    throw std::logic_error(
        "AMR Program detached history remap has an unsupported deferred temporal relation");

  history_levels_.swap(next_levels);
  history_flux_expressions_.swap(next_flux);
  pending_history_remaps_.swap(next_pending);
  deferred_history_lag_scratches_.swap(next_deferred_scratches);
  invalidate_forward_topology_resources_();
}

/// A direct hierarchy replacement has no parent/child transition to replay.  It still needs an
/// explicit detached operation when numeric history has been rematerialized: this operation
/// authenticates the complete replacement image and replaces every topology-qualified history
/// provenance map in one Candidate-time swap.  Its typed internal descriptor travels only
/// through the retained snapshot, so no public facade or second transaction protocol can reach it.
void prepare_detached_full_rebuild_history_reseed_(
    const AmrProgramFullHistoryReseedDescriptor& descriptor) {
  if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
      descriptor.accepted_macro_step < 0 || descriptor.level_count == 0 ||
      descriptor.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
      descriptor.published_topology_epoch != resource_epoch_ ||
      descriptor.published_materialization_generation != resource_generation_ ||
      descriptor.published_topology_epoch != history_epoch_ ||
      descriptor.published_materialization_generation != history_generation_ ||
      descriptor.prior_topology_epoch + 1U != descriptor.published_topology_epoch ||
      descriptor.prior_materialization_generation + 1U !=
          descriptor.published_materialization_generation)
    throw std::logic_error(
        "AMR Program detached full-rebuild history reseed has no exact forward authority");

  std::map<std::string, const AmrProgramFullHistoryReseedEntry*> plan;
  std::map<std::string, std::set<int>> planned_levels;
  std::set<std::string> planned_identities;
  for (const AmrProgramFullHistoryReseedEntry& entry : descriptor.history_plan) {
    const auto decoded = decode_full_rebuild_history_key_(entry.key);
    if (entry.source_key.empty() || entry.history_identity.empty() ||
        !plan.emplace(entry.key, &entry).second || !decoded || decoded->first != entry.level ||
        decoded->first < 0 || decoded->second != entry.history_identity ||
        !planned_levels[decoded->second].insert(decoded->first).second)
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed has a non-canonical plan");
    const auto source = history_levels_.find(entry.source_key);
    const auto source_flux = history_flux_expressions_.find(entry.source_key);
    const auto source_key = decode_full_rebuild_history_key_(entry.source_key);
    if (source == history_levels_.end() || source_flux == history_flux_expressions_.end() ||
        !source_key || source_key->first < 0 || source_key->second != decoded->second)
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed lacks its source identity");
    planned_identities.insert(decoded->second);
  }
  if (plan.empty())
    throw std::logic_error("AMR Program detached full-rebuild history reseed is empty");

  std::set<std::string> retained_identities;
  for (const auto& [key, level] : history_levels_) {
    const auto decoded = decode_full_rebuild_history_key_(key);
    if (!decoded || level != decoded->first || decoded->first < 0 ||
        !history_flux_expressions_.contains(key))
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed found unauthenticated provenance");
    retained_identities.insert(decoded->second);
  }
  for (const auto& [key, expressions] : history_flux_expressions_) {
    (void)expressions;
    if (!history_levels_.contains(key))
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed found foreign flux provenance");
  }
  if (retained_identities != planned_identities)
    throw std::logic_error(
        "AMR Program detached full-rebuild history reseed changed historical identities");

  if (plan.size() != planned_identities.size() * descriptor.level_count)
    throw std::logic_error(
        "AMR Program detached full-rebuild history reseed has an invalid level count");
  for (const auto& [identity, levels] : planned_levels) {
    if (levels.size() != descriptor.level_count)
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed has incomplete levels");
    for (std::size_t level = 0; level < descriptor.level_count; ++level)
      if (!levels.contains(static_cast<int>(level)))
        throw std::logic_error(
            "AMR Program detached full-rebuild history reseed has non-contiguous levels");
    (void)identity;
  }

  std::map<std::string, int> next_levels;
  std::map<std::string, std::vector<FluxExpression>> next_flux;
  for (const auto& [key, entry] : plan) {
    const auto decoded = decode_full_rebuild_history_key_(key);
    if (!decoded)
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed lost a validated level");
    const auto source_flux = history_flux_expressions_.find(entry->source_key);
    if (source_flux == history_flux_expressions_.end())
      throw std::logic_error(
          "AMR Program detached full-rebuild history reseed lost its source provenance");
    next_levels.emplace(key, decoded->first);
    next_flux.emplace(key, source_flux->second);
  }
  history_levels_.swap(next_levels);
  history_flux_expressions_.swap(next_flux);
  pending_history_remaps_.clear();
  deferred_history_lag_scratches_.clear();
  invalidate_forward_topology_resources_();
}

[[nodiscard]] static std::optional<std::pair<int, std::string>> decode_full_rebuild_history_key_(
    std::string_view key) {
  constexpr std::string_view prefix = "pops.amr.level-history.v1/";
  if (!key.starts_with(prefix))
    return std::nullopt;
  key.remove_prefix(prefix.size());
  const std::size_t level_end = key.find('/');
  const std::size_t length_end = key.find(':', level_end == std::string_view::npos ? 0 : level_end);
  if (level_end == std::string_view::npos || length_end == std::string_view::npos)
    return std::nullopt;
  try {
    std::size_t consumed = 0;
    const int level = std::stoi(std::string(key.substr(0, level_end)), &consumed);
    if (consumed != level_end || level < 0 || std::to_string(level) != key.substr(0, level_end))
      return std::nullopt;
    const std::string length_text(key.substr(level_end + 1, length_end - level_end - 1));
    consumed = 0;
    const unsigned long long encoded_length = std::stoull(length_text, &consumed);
    const std::string name(key.substr(length_end + 1));
    if (consumed != length_text.size() || std::to_string(encoded_length) != length_text ||
        encoded_length != name.size() || name.empty())
      return std::nullopt;
    return std::pair<int, std::string>{level, std::move(name)};
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

/// These values carry patch addresses or a completed flux publication.  They are not valid on
/// a new hierarchy.  Their replacement is prepared by the forward graph before HiddenPublish;
/// clearing the detached image is therefore a deliberate invalidation, never a live refresh.
void invalidate_forward_topology_resources_() noexcept {
  deferred_history_lag_scratches_.clear();
  for (auto& axis : accepted_face_flux_)
    axis.clear();
  accepted_synchronization_events_.clear();
}

explicit AcceptedContextSnapshot(DetachedState staged)
    : clock_schedule_(std::move(staged.clock_schedule)),
      resource_epoch_(staged.resource_epoch),
      resource_generation_(staged.resource_generation),
      history_epoch_(staged.history_epoch),
      history_generation_(staged.history_generation),
      forward_hierarchy_tensor_topology_epoch_(staged.resource_epoch),
      forward_hierarchy_tensor_materialization_generation_(staged.resource_generation),
      forward_execution_bundle_epoch_(staged.forward_execution_bundle_epoch),
      forward_execution_bundle_generation_(staged.forward_execution_bundle_generation),
      detached_accepted_snapshot_bytes_(staged.detached_accepted_snapshot_bytes),
      test_forward_storage_ceiling_override_(
          std::move(staged.test_forward_storage_ceiling_override)),
      history_levels_(std::move(staged.history_levels)),
      forward_history_runtime_owners_(std::move(staged.history_runtime_owners)),
      history_flux_expressions_(std::move(staged.history_flux_expressions)),
      pending_history_remaps_(std::move(staged.pending_history_remaps)),
      deferred_history_lag_scratches_(std::move(staged.deferred_history_lag_scratches)),
      prepared_scratch_(std::move(staged.prepared_scratch)),
      prepared_scratch_descriptors_(std::move(staged.prepared_scratch_descriptors)),
      hot_path_workspace_(std::move(staged.hot_path_workspace)),
      accepted_state_staging_(std::move(staged.accepted_state_staging)),
      forward_storage_capacity_(std::move(staged.forward_storage_capacity)),
      forward_flux_tables_(std::move(staged.forward_flux_tables)),
      prepared_subcycling_bundle_(std::move(staged.prepared_subcycling_bundle)),
      forward_hierarchy_tensor_selection_(std::move(staged.hierarchy_tensor_selection)),
      accepted_face_flux_ordinals_(std::move(staged.accepted_face_flux_ordinals)),
      accepted_interface_flux_staging_sources_(
          std::move(staged.accepted_interface_flux_staging_sources)),
      accepted_interface_flux_wire_ordinals_(
          std::move(staged.accepted_interface_flux_wire_ordinals)),
      accepted_checkpoint_level_clock_slots_(
          std::move(staged.accepted_checkpoint_level_clock_slots)),
      accepted_temporal_partition_(std::move(staged.accepted_temporal_partition)),
      cell_temporal_configuration_(std::move(staged.cell_temporal_configuration)),
      accepted_flux_budget_contract_(std::move(staged.accepted_flux_budget_contract)),
      accepted_coupling_contract_(std::move(staged.accepted_coupling_contract)),
      accepted_face_flux_(std::move(staged.accepted_face_flux)),
      interface_flux_ledger_(std::move(staged.interface_flux_ledger)),
      accepted_synchronization_events_(std::move(staged.accepted_synchronization_events)),
      accepted_face_flux_slots_(std::move(staged.accepted_face_flux_slots)),
      accepted_synchronization_event_slots_(std::move(staged.accepted_synchronization_event_slots)),
      multiblock_subcycling_state_(std::move(staged.multiblock_subcycling_state)),
      accepted_state_revision_(staged.accepted_state_revision) {}
