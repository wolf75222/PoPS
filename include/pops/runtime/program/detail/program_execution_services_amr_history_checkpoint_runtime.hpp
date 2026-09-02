void reconcile_multiblock_reflux_(multiblock_reflux_context_type& context) const {
  if (context.flux.published_size() == 0)
    return;
  const Geometry<Dim> geometry =
      facade_->program_prepared_amr_level_geometry_(static_cast<int>(context.parent_level));
  bool found_route = false;
  for (auto& route : hot_path_workspace_.prepared_metric_reflux_routes) {
    if (route.ledger != std::addressof(context.flux))
      continue;
    if (route.block != context.block || route.parent_level != context.parent_level ||
        route.query.owner != context.block_identity ||
        route.query.levels !=
            ::pops::amr::reflux::LevelTransition{static_cast<int>(context.parent_level),
                                                 static_cast<int>(context.parent_level + 1)} ||
        route.ratio != context.spatial_ratio || route.mapping != context.face_mapping)
      throw std::logic_error("AMR Program prepared metric reflux route lost its static authority");
    // Routes are bound once per exact parent subwindow.  Requalification restores all of them;
    // advancing one substep must therefore update and consume only its own immutable phase
    // window, never collapse sibling windows onto one mutable query.
    if (route.bound_window_begin != context.parent_window.begin.phase ||
        route.bound_window_end != context.parent_window.end.phase)
      continue;
    route.query.attempt = context.attempt;
    route.query.macro_step = context.parent_window.begin.macro_step;
    route.query.window_begin = context.parent_window.begin.phase;
    route.query.window_end = context.parent_window.end.phase;
    found_route = true;
  }
  if (!found_route)
    throw std::runtime_error(
        "AMR Program published a reflux ledger without an exact prepared metric route");

  // Authenticate every published fragment before any resident result/correction image is
  // rewritten.  A matching route is unique by its static coarse interface key; accepting an
  // unbound or ambiguous entry would silently bypass the prepared coverage/temporal authority.
  for (int axis = 0; axis < Dim; ++axis)
    for (const auto& entry : context.flux.published_entries(axis)) {
      std::size_t matches = 0;
      for (const auto& route : hot_path_workspace_.prepared_metric_reflux_routes) {
        if (route.ledger != std::addressof(context.flux))
          continue;
        const auto& query = route.query;
        const bool dynamic_match = entry.key.attempt == query.attempt &&
                                   entry.key.clock.macro_step == query.macro_step &&
                                   !(entry.key.clock.phase < query.window_begin) &&
                                   !(query.window_end < entry.key.clock.phase);
        if (dynamic_match && entry.key.owner == query.owner && entry.key.state == query.state &&
            entry.key.levels == query.levels && entry.key.centering == query.centering &&
            entry.key.axis == query.axis && entry.key.coarse_face == query.coarse_face)
          ++matches;
      }
      if (matches != 1)
        throw std::runtime_error(
            "AMR Program published reflux fragment has no unique prepared metric route");
    }

  for (auto& route : hot_path_workspace_.prepared_metric_reflux_routes) {
    if (route.ledger != std::addressof(context.flux))
      continue;
    if (route.bound_window_begin != context.parent_window.begin.phase ||
        route.bound_window_end != context.parent_window.end.phase)
      continue;
    const auto& reflux = ::pops::amr::reflux::metric_reflux_prepared(
        route.workspace, context.flux, route.query, route.ratio, route.mapping, route.budget,
        payload_axpy_);
    const auto& correction = route.workspace.reconcile_coarse_cell_correction(
        cell_measure_(geometry), route.interface.side, payload_axpy_);
    apply_reflux_payload_(context.parent, route.interface.coarse_cell, correction);
    (void)reflux;
  }
}

static std::optional<std::pair<int, std::string>> decode_history_key_(std::string_view key) {
  constexpr std::string_view prefix = "pops.amr.level-history.v1/";
  if (!key.starts_with(prefix))
    return std::nullopt;
  key.remove_prefix(prefix.size());
  const std::size_t slash = key.find('/');
  const std::size_t colon = key.find(':', slash == std::string_view::npos ? 0 : slash);
  if (slash == std::string_view::npos || colon == std::string_view::npos)
    throw std::invalid_argument("AMR Program history storage key is malformed");
  std::size_t consumed = 0;
  const int level = std::stoi(std::string(key.substr(0, slash)), &consumed);
  if (level < 0 || consumed != slash)
    throw std::invalid_argument("AMR Program history storage key has an invalid level");
  const std::string length_text(key.substr(slash + 1, colon - slash - 1));
  consumed = 0;
  const std::size_t length = std::stoull(length_text, &consumed);
  const std::string name(key.substr(colon + 1));
  if (consumed != length_text.size() || name.empty() || name.size() != length)
    throw std::invalid_argument("AMR Program history storage key has an invalid name");
  return std::pair<int, std::string>{level, name};
}

AmrProgramAcceptedState<Dim> accepted_state_(
    std::vector<::pops::amr::ClockStamp>* reusable_level_clocks = nullptr) const {
  require_facade_execution_();
  AmrProgramAcceptedState<Dim> state;
  state.spatial_contract = runtime_->spatial_contract();
  state.topology_epoch = runtime_->topology_epoch();
  state.materialization_generation = runtime_->materialization_generation();
  const std::size_t levels = runtime_->hierarchy().num_levels();
  if (reusable_level_clocks != nullptr) {
    if (reusable_level_clocks->capacity() < levels)
      throw std::logic_error("AMR Program checkpoint level-clock storage was not primed");
    state.level_clocks.swap(*reusable_level_clocks);
    state.level_clocks.clear();
  } else {
    state.level_clocks.reserve(levels);
  }
  for (std::size_t level = 0; level < levels; ++level)
    state.level_clocks.push_back({static_cast<int>(level),
                                  facade_->program_macro_step_(),
                                  {0, 1},
                                  facade_->program_time_()});
  state.logical_clock_ticks =
      clock_schedule_.accepted_ticks(static_cast<std::int64_t>(facade_->program_macro_step_()));
  state.temporal_partition = accepted_temporal_partition_;

  const auto& manager = runtime_state().hist_;
  struct AccumulatedHistory {
    AmrProgramHistoryDescriptor descriptor;
    std::set<int> levels;
  };
  std::map<std::string, AccumulatedHistory> histories;
  for (const auto& [key, ring] : manager.histories) {
    const auto decoded = decode_history_key_(key);
    if (!decoded || ring.empty())
      throw std::runtime_error("AMR Program accepted history registry is malformed");
    const auto& [level, name] = *decoded;
    const int runtime_owner = manager.owner.at(key);
    int program_owner = -1;
    const auto& block_map = facade_->program_block_map_();
    for (std::size_t program = 0; program < block_map.size(); ++program)
      if (block_map[program] == runtime_owner) {
        program_owner = static_cast<int>(program);
        break;
      }
    if (program_owner < 0)
      throw std::runtime_error("AMR Program history lost its authenticated block owner");
    AmrProgramHistoryDescriptor descriptor{name,
                                           program_owner,
                                           manager.state_identity.at(key),
                                           manager.space_identity.at(key),
                                           manager.clock_identity.at(key),
                                           manager.interpolation_identity.at(key),
                                           manager.depth.at(key),
                                           ring.front().ncomp()};
    auto [entry, inserted] =
        histories.try_emplace(name, AccumulatedHistory{descriptor, std::set<int>{level}});
    if (!inserted) {
      const auto& retained = entry->second.descriptor;
      if (retained.program_owner != descriptor.program_owner ||
          retained.state_identity != descriptor.state_identity ||
          retained.space_identity != descriptor.space_identity ||
          retained.clock_identity != descriptor.clock_identity ||
          retained.interpolation_identity != descriptor.interpolation_identity ||
          retained.depth != descriptor.depth || retained.components != descriptor.components ||
          !entry->second.levels.insert(level).second)
        throw std::runtime_error("AMR Program history differs between active levels");
    }
    const auto& dts = manager.slot_dt.at(key);
    if (dts.size() != ring.size())
      throw std::runtime_error("AMR Program history dt provenance has the wrong depth");
    for (std::size_t slot = 0; slot < ring.size(); ++slot)
      state.history_slots.push_back({name, level, static_cast<int>(slot),
                                     static_cast<double>(dts[slot]), manager.initialized.at(key),
                                     manager.fill_count.at(key)});
  }
  for (auto& [name, accumulated] : histories) {
    (void)name;
    if (accumulated.levels.size() != runtime_->hierarchy().num_levels())
      throw std::runtime_error("AMR Program history omits an active hierarchy level");
    state.histories.push_back(std::move(accumulated.descriptor));
  }
  std::sort(state.history_slots.begin(), state.history_slots.end(),
            [](const auto& left, const auto& right) {
              return std::tie(left.name, left.level, left.slot) <
                     std::tie(right.name, right.level, right.slot);
            });
  state.pending_history_remaps.reserve(pending_history_remaps_.size());
  for (const auto& [key, pending] : pending_history_remaps_) {
    if (key != pending.key)
      throw std::runtime_error("AMR Program pending history remap has a foreign key");
    if (pending.consumed)
      continue;
    state.pending_history_remaps.push_back(pending);
  }
  state.history_flux_payload = serialize_history_flux_payload_();

  if (multiblock_subcycling_) {
    state.flux_budget_contract = multiblock_subcycling_program_budget_contract_;
    state.coupling_contract =
        std::string(facade_->prepared_amr_multiblock_hierarchy_().collective_contract());
    const std::string_view interface_contract =
        facade_->prepared_amr_multiblock_hierarchy_().interface_flux_provider_contract();
    if (!interface_contract.empty()) {
      ExactContractBuilder accepted_coupling;
      accepted_coupling.text("pops.amr-program.accepted-coupling")
          .scalar(std::uint32_t{1})
          .bytes(state.coupling_contract)
          .bytes(interface_contract);
      state.coupling_contract = std::move(accepted_coupling).release();
    }
    if (interface_flux_ledger_) {
      ExactContractBuilder budgeted_coupling;
      budgeted_coupling.text("pops.amr-program.accepted-budgeted-coupling")
          .scalar(std::uint32_t{1})
          .bytes(state.coupling_contract)
          .bytes(interface_flux_ledger_->budget().exact_contract)
          .scalar(
              static_cast<std::uint64_t>(interface_flux_ledger_->budget().max_fragments_per_window))
          .scalar(static_cast<std::uint64_t>(
              interface_flux_ledger_->budget().max_payload_terms_per_window))
          .scalar(
              static_cast<std::uint64_t>(interface_flux_ledger_->budget().max_transaction_depth));
      state.coupling_contract = std::move(budgeted_coupling).release();
    }
    const auto seal_contract = [](std::string_view prefix, std::string_view contract) {
      const auto* begin = reinterpret_cast<const std::uint8_t*>(contract.data());
      std::vector<std::uint8_t> bytes(begin, begin + contract.size());
      return std::string(prefix) + identity::sha256_hex(bytes);
    };
    state.flux_budget_contract = seal_contract("pops.amr-program.complete-flux-budget.v1:sha256:",
                                               state.flux_budget_contract);
    state.coupling_contract =
        seal_contract("pops.amr-program.accepted-coupling.v1:sha256:", state.coupling_contract);
    for (std::size_t block = 0; block < facade_->prepared_amr_multiblock_hierarchy_().block_count();
         ++block) {
      for (std::size_t parent = 0; parent + 1 < runtime_->hierarchy().num_levels(); ++parent) {
        for (const auto& ledger : multiblock_subcycling_->ledgers(block, parent))
          for (int axis = 0; axis < Dim; ++axis) {
            const auto& entries = ledger.published_entries(axis);
            auto& flux = state.accepted_face_flux[static_cast<std::size_t>(axis)];
            flux.insert(flux.end(), entries.begin(), entries.end());
          }
        const ::pops::amr::ClockStamp clock{static_cast<int>(parent),
                                            facade_->program_macro_step_(),
                                            {0, 1},
                                            facade_->program_time_()};
        state.synchronization_events.push_back({static_cast<int>(parent),
                                                static_cast<int>(parent + 1),
                                                static_cast<int>(block), "reflux", clock});
        state.synchronization_events.push_back({static_cast<int>(parent),
                                                static_cast<int>(parent + 1),
                                                static_cast<int>(block), "average_down", clock});
      }
    }
    for (int axis = 0; axis < Dim; ++axis) {
      auto by_key = [](const auto& left, const auto& right) { return left.key < right.key; };
      std::sort(state.accepted_face_flux[static_cast<std::size_t>(axis)].begin(),
                state.accepted_face_flux[static_cast<std::size_t>(axis)].end(), by_key);
    }
  } else {
    state.flux_budget_contract = accepted_flux_budget_contract_;
    state.coupling_contract = accepted_coupling_contract_;
    state.accepted_face_flux = accepted_face_flux_;
    state.synchronization_events = accepted_synchronization_events_;
  }
  if (interface_flux_ledger_) {
    if (interface_flux_ledger_->in_transaction())
      throw std::logic_error(
          "AMR Program checkpoint cannot observe an active interface-flux transaction");
    state.accepted_interface_flux = interface_flux_ledger_->cold_published_fragments();
    std::sort(state.accepted_interface_flux.begin(), state.accepted_interface_flux.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
  }
  return state;
}

void import_accepted_state_(bool force) const {
  require_facade_execution_();
  const std::uint64_t revision = facade_->program_accepted_state_revision_();
  if (!force && revision == accepted_state_revision_)
    return;
  try {
    prepare_multiblock_subcycling_engine_();
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted-state import cannot prepare its live budget: " +
                             std::string(exception.what()));
  }
  std::vector<std::uint8_t> bytes;
  try {
    bytes = facade_->program_accepted_state_();
  } catch (const std::exception& exception) {
    throw std::runtime_error(
        "AMR Program accepted-state import cannot read the native accepted image: " +
        std::string(exception.what()));
  }
  if (bytes.empty())
    throw std::runtime_error("AMR Program accepted-state import received an empty image");
  AmrProgramAcceptedState<Dim> state;
  try {
    state = deserialize_amr_program_accepted_state<Dim>(bytes, &interface_flux_ledger_->budget());
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted-state import cannot decode the native image: " +
                             std::string(exception.what()));
  }
  try {
    require_live_amr_program_checkpoint(state, *runtime_);
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted-state import lost live hierarchy authority: " +
                             std::string(exception.what()));
  }
  std::map<std::string, std::vector<FluxExpression>> restored_history_flux;
  try {
    restored_history_flux = prepare_history_flux_payload_restore_(state.history_flux_payload);
  } catch (const std::exception& exception) {
    throw std::runtime_error("AMR Program accepted-state import cannot restore history flux: " +
                             std::string(exception.what()));
  }
  std::map<std::string, AmrProgramPendingHistoryRemap> restored_pending;
  std::map<std::string, field_type> restored_deferred_history_lag_scratches;
  for (const auto& pending : state.pending_history_remaps) {
    const auto ring = runtime_state().hist_.histories.find(pending.key);
    const auto decoded = decode_history_key_(pending.key);
    const auto owner = runtime_state().hist_.owner.find(pending.key);
    const auto initialized = runtime_state().hist_.initialized.find(pending.key);
    const auto depth = runtime_state().hist_.depth.find(pending.key);
    const auto stored = runtime_state().hist_.store_pending.find(pending.key);
    const auto dts = runtime_state().hist_.slot_dt.find(pending.key);
    if (pending.key.empty() || !decoded || ring == runtime_state().hist_.histories.end() ||
        owner == runtime_state().hist_.owner.end() ||
        initialized == runtime_state().hist_.initialized.end() ||
        depth == runtime_state().hist_.depth.end() ||
        stored == runtime_state().hist_.store_pending.end() ||
        dts == runtime_state().hist_.slot_dt.end() || decoded->first != pending.child_level ||
        pending.parent_level < 0 || pending.child_level != pending.parent_level + 1 ||
        pending.child_level >= static_cast<int>(state.level_clocks.size()) ||
        !initialized->second || depth->second != 2 || ring->second.size() != 2 ||
        dts->second.size() != 2 || stored->second ||
        pending.accepted_macro_step !=
            state.level_clocks[static_cast<std::size_t>(pending.child_level)].macro_step ||
        static_cast<double>(dts->second[1]) != pending.source_dt ||
        pending.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        pending.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
        pending.prior_topology_epoch + 1 != pending.published_topology_epoch ||
        pending.prior_materialization_generation + 1 !=
            pending.published_materialization_generation ||
        pending.consumed || pending.published_topology_epoch != runtime_->topology_epoch() ||
        pending.published_materialization_generation != runtime_->materialization_generation() ||
        pending.temporal_denominator != 1 ||
        (pending.temporal_numerator != 1 && pending.temporal_numerator != 2) ||
        !(pending.source_dt > 0.0) || !(pending.target_dt > 0.0) ||
        pending.target_dt != pending.source_dt / static_cast<double>(pending.temporal_numerator) ||
        !restored_pending.emplace(pending.key, pending).second)
      throw std::runtime_error("AMR Program accepted-state import has an invalid pending remap");
    if (!restored_deferred_history_lag_scratches.try_emplace(pending.key, ring->second.front())
             .second)
      throw std::runtime_error(
          "AMR Program accepted-state import has a duplicate deferred scratch");
  }
  AmrProgramAcceptedState<Dim> expected;
  try {
    expected = accepted_state_();
  } catch (const std::exception& exception) {
    throw std::runtime_error(
        "AMR Program accepted-state import cannot construct live history provenance: " +
        std::string(exception.what()));
  }
  if (state.histories != expected.histories || state.history_slots != expected.history_slots)
    throw std::runtime_error(
        "AMR Program accepted-state history provenance differs from live restored rings");
  if (!expected.flux_budget_contract.empty() &&
      state.flux_budget_contract != expected.flux_budget_contract)
    throw std::runtime_error("AMR Program accepted-state flux budget is no longer authentic");
  if (!expected.coupling_contract.empty() && state.coupling_contract != expected.coupling_contract)
    throw std::runtime_error("AMR Program accepted-state coupling contract is no longer authentic");
  if (state.level_clocks.empty())
    throw std::runtime_error("AMR Program accepted-state import lacks its level clocks");
  const std::int64_t accepted_macro_step = state.level_clocks.front().macro_step;
  for (const auto& clock : state.level_clocks)
    if (clock.macro_step != accepted_macro_step)
      throw std::runtime_error(
          "AMR Program accepted-state levels disagree on their accepted macro step");
  if (state.temporal_partition.kind == TemporalPartitionKind::CellLocal) {
    if (!cell_temporal_configuration_ ||
        state.temporal_partition.provider_identity != kSameLevelTransportEulerStageFluxProvider ||
        state.temporal_partition.tick_denominator != cell_temporal_configuration_->tick_denominator)
      throw std::runtime_error(
          "AMR Program restored cell-local partition lacks its generated route authority");
    for (const auto& clock : state.level_clocks) {
      const double scaled =
          clock.physical_time * static_cast<double>(state.temporal_partition.tick_denominator);
      if (!std::isfinite(scaled) || scaled < 0.0 ||
          !(scaled < static_cast<double>(std::numeric_limits<std::int64_t>::max())) ||
          std::floor(scaled) != scaled)
        throw std::runtime_error("AMR Program restored cell-local clock has no bounded exact tick");
      const auto tick = static_cast<std::int64_t>(scaled);
      if (tick != state.temporal_partition.synchronization_tick ||
          clock.phase != ::pops::amr::Rational{0, 1})
        throw std::runtime_error(
            "AMR Program restored cell-local clocks are not at one exact macro barrier");
    }
    const auto expected_partition = cell_temporal_full_partition_(
        *cell_temporal_configuration_, state.temporal_partition.synchronization_tick);
    if (state.temporal_partition != expected_partition)
      throw std::runtime_error(
          "AMR Program restored cell-local partition differs from its generated topology");
  }
  // This validates every accepted field while it is still intact.  In particular, the temporal
  // partition and interface entries must not be observed after either has been moved into the
  // context.  The candidate owns its allocation before the no-throw member publication below.
  auto interface_budget = interface_flux_ledger_->budget();
  auto restored_interface_flux = std::make_unique<interface_flux_ledger_type>(
      restore_amr_program_interface_flux_ledger(state, std::move(interface_budget)));

  // Checkpoint decoding owns compact event vectors.  They must never replace the accepted
  // bind-sealed envelope directly: a later candidate can activate any prepared event route, not
  // only the routes that happened to be present in this checkpoint.  Build the cold replacement
  // from the current prepared slots and retain its complete capacity before publishing either
  // half of the accepted event pair.
  const auto& staging = accepted_state_staging_;
  if (!staging.prepared_envelope ||
      staging.synchronization_event_slots.size() < state.synchronization_events.size())
    throw std::runtime_error(
        "AMR Program accepted-state import has no prepared synchronization-event envelope");
  auto restored_synchronization_event_commit_slots = staging.synchronization_event_slots;
  if (restored_synchronization_event_commit_slots.capacity() <
      staging.synchronization_event_slots.capacity())
    restored_synchronization_event_commit_slots.reserve(
        staging.synchronization_event_slots.capacity());
  for (std::size_t index = 0; index < staging.synchronization_event_slots.size(); ++index) {
    auto& destination = restored_synchronization_event_commit_slots[index];
    const auto& source = staging.synchronization_event_slots[index];
    if (destination.parent_level != source.parent_level ||
        destination.child_level != source.child_level ||
        destination.runtime_block != source.runtime_block || destination.phase != source.phase)
      throw std::runtime_error(
          "AMR Program accepted-state import changed a prepared synchronization-event identity");
    if (destination.phase.capacity() < source.phase.capacity())
      destination.phase.reserve(source.phase.capacity());
  }
  std::vector<AmrProgramSynchronizationEvent> restored_synchronization_events;
  restored_synchronization_events.reserve(restored_synchronization_event_commit_slots.size());
  for (std::size_t index = 0; index < state.synchronization_events.size(); ++index) {
    const auto& source = state.synchronization_events[index];
    const auto found = std::find_if(
        restored_synchronization_event_commit_slots.begin(),
        restored_synchronization_event_commit_slots.end(), [&](const auto& slot) {
          return slot.parent_level == source.parent_level &&
                 slot.child_level == source.child_level &&
                 slot.runtime_block == source.runtime_block && slot.phase == source.phase;
        });
    if (found == restored_synchronization_event_commit_slots.end())
      throw std::runtime_error(
          "AMR Program accepted-state import has an event outside its prepared envelope");
    if (std::any_of(state.synchronization_events.begin(),
                    state.synchronization_events.begin() + static_cast<std::ptrdiff_t>(index),
                    [&](const auto& prior) {
                      return prior.parent_level == source.parent_level &&
                             prior.child_level == source.child_level &&
                             prior.runtime_block == source.runtime_block &&
                             prior.phase == source.phase;
                    }))
      throw std::runtime_error(
          "AMR Program accepted-state import duplicates a prepared synchronization event");
    restored_synchronization_events.push_back(*found);
    auto& destination = restored_synchronization_events.back();
    if (destination.phase.capacity() < found->phase.capacity())
      destination.phase.reserve(found->phase.capacity());
    if (source.phase.size() > destination.phase.capacity())
      throw std::runtime_error(
          "AMR Program accepted-state import event exceeds its prepared phase capacity");
    destination.phase.resize(source.phase.size());
    std::copy(source.phase.begin(), source.phase.end(), destination.phase.begin());
    destination.parent_level = source.parent_level;
    destination.child_level = source.child_level;
    destination.runtime_block = source.runtime_block;
    destination.clock = source.clock;
  }
  clock_schedule_.restore_accepted_ticks(state.logical_clock_ticks, accepted_macro_step);
  static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(deferred_history_lag_scratches_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_flux_budget_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_coupling_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
  history_flux_expressions_.swap(restored_history_flux);
  std::swap(accepted_temporal_partition_, state.temporal_partition);
  pending_history_remaps_.swap(restored_pending);
  deferred_history_lag_scratches_.swap(restored_deferred_history_lag_scratches);
  for (const auto& diagnostic : cell_temporal_diagnostics_)
    if (diagnostic)
      diagnostic->invalidate_accepted_publication(accepted_temporal_partition_.synchronization_tick,
                                                  accepted_temporal_partition_.tick_denominator);
  accepted_flux_budget_contract_.swap(state.flux_budget_contract);
  accepted_coupling_contract_.swap(state.coupling_contract);
  std::swap(accepted_face_flux_, state.accepted_face_flux);
  interface_flux_commit_guard_.reset();
  interface_flux_ledger_.swap(restored_interface_flux);
  accepted_synchronization_events_.swap(restored_synchronization_events);
  accepted_synchronization_event_commit_slots_.swap(restored_synchronization_event_commit_slots);
  accepted_state_revision_ = revision;
  // `restored_*` containers are value-owned images.  Rebuild the adapter-only non-owning
  // ordinals only after all no-throw swaps have installed the complete accepted authority.
  // This is a cold checkpoint-import boundary; Candidate never repairs these references.
  prime_history_mutation_workspace_at_bind_();
  rebind_accepted_face_flux_ordinals_at_cold_prime_();
  rebind_accepted_interface_flux_ordinals_at_cold_prime_();
}

void reset_accepted_state_staging_for_cold_prime_() const {
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR Program accepted-state staging cannot reset during an attempt");

  auto& staging = accepted_state_staging_;
  staging.valid = false;
  if (!staging.prepared_envelope) {
    staging.primed = false;
    return;
  }

  // Validate every resident ownership index before exchanging the first string or payload.  This
  // is a cold bind boundary, but it still must leave the previous image intact when its envelope
  // is malformed.
  const auto require_unique_slots = [](const auto& slots, const char* description) {
    for (std::size_t index = 0; index < slots.size(); ++index)
      for (std::size_t previous = 0; previous < index; ++previous)
        if (slots[index] == slots[previous])
          throw std::logic_error(std::string("AMR Program accepted-state staging ") + description +
                                 " reuse one resident slot");
  };
  if (staging.state.history_slots.size() != staging.history_slot_active_indices.size())
    throw std::logic_error("AMR Program accepted-state staging history slots are malformed");
  for (const std::size_t slot : staging.history_slot_active_indices)
    if (slot >= staging.history_slot_pool.size())
      throw std::logic_error("AMR Program accepted-state staging history pool is malformed");
  require_unique_slots(staging.history_slot_active_indices, "history indices");
  if (staging.state.pending_history_remaps.size() != staging.pending_history_active_slots.size())
    throw std::logic_error("AMR Program accepted-state staging pending slots are malformed");
  for (const std::size_t slot : staging.pending_history_active_slots)
    if (slot >= staging.pending_history_remap_slots.size())
      throw std::logic_error("AMR Program accepted-state staging pending pool is malformed");
  require_unique_slots(staging.pending_history_active_slots, "pending indices");
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    if (staging.state.accepted_face_flux[index].size() !=
        staging.accepted_face_flux_active_slots[index].size())
      throw std::logic_error("AMR Program accepted-state staging face-flux slots are malformed");
    for (const std::size_t slot : staging.accepted_face_flux_active_slots[index])
      if (slot >= staging.accepted_face_flux_slots[index].size())
        throw std::logic_error("AMR Program accepted-state staging face-flux pool is malformed");
    require_unique_slots(staging.accepted_face_flux_active_slots[index], "face-flux indices");
  }
  if (staging.state.synchronization_events.size() !=
      staging.synchronization_event_active_indices.size())
    throw std::logic_error("AMR Program accepted-state staging event slots are malformed");
  for (const std::size_t slot : staging.synchronization_event_active_indices)
    if (slot >= staging.synchronization_event_slots.size())
      throw std::logic_error("AMR Program accepted-state staging event pool is malformed");
  require_unique_slots(staging.synchronization_event_active_indices, "event indices");
  if (staging.state.accepted_interface_flux.size() !=
      staging.accepted_interface_flux_active_slots.size())
    throw std::logic_error("AMR Program accepted-state staging interface slots are malformed");
  for (const std::size_t slot : staging.accepted_interface_flux_active_slots)
    if (slot >= staging.accepted_interface_flux_slots.size())
      throw std::logic_error("AMR Program accepted-state staging interface pool is malformed");
  require_unique_slots(staging.accepted_interface_flux_active_slots, "interface indices");

  for (std::size_t index = 0; index < staging.state.history_slots.size(); ++index)
    staging.state.history_slots[index].name.swap(
        staging.history_slot_pool[staging.history_slot_active_indices[index]].name);
  staging.state.history_slots.clear();
  staging.history_slot_active_indices.clear();

  for (std::size_t index = 0; index < staging.state.pending_history_remaps.size(); ++index)
    staging.state.pending_history_remaps[index].key.swap(
        staging.pending_history_remap_slots[staging.pending_history_active_slots[index]].key);
  staging.state.pending_history_remaps.clear();
  staging.pending_history_active_slots.clear();

  for (int axis = 0; axis < Dim; ++axis) {
    const auto slot_axis = static_cast<std::size_t>(axis);
    auto& logical = staging.state.accepted_face_flux[slot_axis];
    auto& slots = staging.accepted_face_flux_slots[slot_axis];
    const auto& active = staging.accepted_face_flux_active_slots[slot_axis];
    for (std::size_t index = 0; index < logical.size(); ++index) {
      auto& resident = slots[active[index]];
      resident.key.owner = std::move(logical[index].key.owner);
      resident.key.state = std::move(logical[index].key.state);
      resident.key.stage = std::move(logical[index].key.stage);
      resident.payload = std::move(logical[index].payload);
    }
    logical.clear();
    staging.accepted_face_flux_active_slots[slot_axis].clear();
    staging.accepted_face_flux_sources[slot_axis].clear();
  }

  for (std::size_t index = 0; index < staging.state.synchronization_events.size(); ++index)
    staging.state.synchronization_events[index].phase.swap(
        staging.synchronization_event_slots[staging.synchronization_event_active_indices[index]]
            .phase);
  staging.state.synchronization_events.clear();
  staging.synchronization_event_active_indices.clear();

  for (std::size_t index = 0; index < staging.state.accepted_interface_flux.size(); ++index) {
    auto& logical = staging.state.accepted_interface_flux[index];
    auto& resident =
        staging.accepted_interface_flux_slots[staging.accepted_interface_flux_active_slots[index]];
    resident.key.interface_identity = std::move(logical.key.interface_identity);
    resident.key.stage_identity = std::move(logical.key.stage_identity);
    resident.key.graph_identity = std::move(logical.key.graph_identity);
    resident.key.rate_identity = std::move(logical.key.rate_identity);
    resident.key.application_identity = std::move(logical.key.application_identity);
    resident.payload = std::move(logical.payload);
  }
  staging.state.accepted_interface_flux.clear();
  staging.accepted_interface_flux_active_slots.clear();
  accepted_interface_flux_staging_sources_.clear();
  staging.state.level_clocks.clear();
  staging.primed = false;
}

/// Rebind the compact resident-ledger location to the already sorted checkpoint wire slots.
/// This is cold topology work: template scans and sorting are deliberately performed here, while
/// Candidate refresh below follows only `source_slot -> staging_slot` ordinals and never builds a
/// string key or a temporary pointer/sort carrier.
void rebind_accepted_face_flux_ordinals_at_cold_prime_() const {
  auto& staging = accepted_state_staging_;
  std::array<std::vector<AcceptedFaceFluxOrdinal>, Dim> next;
  if (multiblock_subcycling_) {
    // Attempt and macro-step are candidate coordinates rewritten by the resident ledger.  The
    // ordinal binds the immutable route identity; comparing either mutable coordinate would make
    // an accepted slot disappear on the first retry or on the next macro-step.
    const auto same_wire_identity = [](const auto& left, const auto& right) {
      return left.key.owner == right.key.owner && left.key.state == right.key.state &&
             left.key.levels == right.key.levels && left.key.centering == right.key.centering &&
             left.key.axis == right.key.axis && left.key.face == right.key.face &&
             left.key.coarse_face == right.key.coarse_face && left.key.stage == right.key.stage &&
             left.key.role == right.key.role && left.key.contribution == right.key.contribution &&
             left.key.clock.level == right.key.clock.level &&
             left.key.clock.phase == right.key.clock.phase;
    };
    multiblock_subcycling_->bind_candidate_ledger_slots([&](std::size_t, std::size_t, std::size_t,
                                                            multiblock_flux_ledger_type& ledger) {
      for (int axis = 0; axis < Dim; ++axis) {
        const auto dimension = static_cast<std::size_t>(axis);
        const auto templates = ledger.resident_slot_templates(axis);
        auto& destination = next[dimension];
        const auto& slots = staging.accepted_face_flux_slots[dimension];
        if (destination.size() > slots.size() ||
            templates.size() > slots.size() - destination.size())
          throw std::logic_error("AMR Program accepted face-flux ordinal exceeds slot envelope");
        for (std::size_t source_slot = 0; source_slot < templates.size(); ++source_slot) {
          std::size_t match = slots.size();
          for (std::size_t slot = 0; slot < slots.size(); ++slot) {
            if (!same_wire_identity(templates[source_slot], slots[slot]))
              continue;
            if (match != slots.size())
              throw std::logic_error("AMR Program accepted face-flux ordinal is ambiguous");
            match = slot;
          }
          if (match == slots.size()) {
            const auto& key = templates[source_slot].key;
            throw std::logic_error(
                "AMR Program accepted face-flux ordinal has no wire slot: owner=" + key.owner +
                ", state=" + key.state + ", stage=" + key.stage +
                ", clock-level=" + std::to_string(key.clock.level) +
                ", clock-step=" + std::to_string(key.clock.macro_step) +
                ", clock-phase=" + std::to_string(key.clock.phase.numerator) + "/" +
                std::to_string(key.clock.phase.denominator) + ", attempt=" +
                std::to_string(key.attempt) + ", source-slot=" + std::to_string(source_slot) +
                ", wire-slots=" + std::to_string(slots.size()));
          }
          destination.push_back({std::addressof(ledger), source_slot, match});
        }
      }
    });
    for (int axis = 0; axis < Dim; ++axis) {
      const auto dimension = static_cast<std::size_t>(axis);
      auto& ordinals = next[dimension];
      const auto& slots = staging.accepted_face_flux_slots[dimension];
      if (ordinals.size() != slots.size())
        throw std::logic_error("AMR Program accepted face-flux ordinal count changed after bind");
      std::sort(ordinals.begin(), ordinals.end(), [](const auto& left, const auto& right) {
        return left.staging_slot < right.staging_slot;
      });
      for (std::size_t index = 0; index < ordinals.size(); ++index)
        if (ordinals[index].staging_slot != index)
          throw std::logic_error("AMR Program accepted face-flux wire ordinals are not unique");
    }
  } else {
    for (int axis = 0; axis < Dim; ++axis)
      if (!staging.accepted_face_flux_slots[static_cast<std::size_t>(axis)].empty())
        throw std::logic_error("AMR Program accepted face-flux slots have no cold ledger owner");
  }
  accepted_face_flux_ordinals_.swap(next);
  accepted_face_flux_ordinal_owner_ = multiblock_subcycling_.get();
  accepted_face_flux_ordinal_epoch_ = resource_epoch_;
  accepted_face_flux_ordinal_generation_ = resource_generation_;
}

/// Interface ledgers expose a dense source image rather than stable per-fragment Entry objects.
/// Seal the finite source-to-wire permutation while no candidate is active.  The dense ledger
/// ingress/contract owns identity authentication; the hot path follows only this permutation.
void rebind_accepted_interface_flux_ordinals_at_cold_prime_() const {
  if (interface_flux_ledger_ && interface_flux_ledger_->in_transaction())
    throw std::logic_error("AMR Program interface-flux ordinals cannot bind an active ledger");
  const std::size_t bound = accepted_interface_flux_staging_sources_.capacity();
  std::vector<std::size_t> next;
  next.resize(bound);
  for (std::size_t ordinal = 0; ordinal < bound; ++ordinal)
    next[ordinal] = ordinal;
  accepted_interface_flux_wire_ordinals_.swap(next);
  accepted_interface_flux_ordinal_owner_ = interface_flux_ledger_.get();
  accepted_interface_flux_ordinal_epoch_ = resource_epoch_;
  accepted_interface_flux_ordinal_generation_ = resource_generation_;
}

/// The topology aggregate has already been published when this route is used.  It therefore
/// cannot allocate, sort, or reject a partially rebuilt ordinal image: the cold capacity seal
/// must have reserved every ordinal slot before this no-throw publication boundary.
void rebind_accepted_face_flux_ordinals_preallocated_noexcept_() const noexcept {
  auto& staging = accepted_state_staging_;
  // Keep this no-throw matcher identical to the cold route above: attempt and macro-step are
  // mutable candidate coordinates, not part of the resident wire-slot identity.
  const auto same_wire_identity = [](const auto& left, const auto& right) noexcept {
    return left.key.owner == right.key.owner && left.key.state == right.key.state &&
           left.key.levels == right.key.levels && left.key.centering == right.key.centering &&
           left.key.axis == right.key.axis && left.key.face == right.key.face &&
           left.key.coarse_face == right.key.coarse_face && left.key.stage == right.key.stage &&
           left.key.role == right.key.role && left.key.contribution == right.key.contribution &&
           left.key.clock.level == right.key.clock.level &&
           left.key.clock.phase == right.key.clock.phase;
  };
  if (!multiblock_subcycling_) {
    for (int axis = 0; axis < Dim; ++axis) {
      const auto dimension = static_cast<std::size_t>(axis);
      if (!staging.accepted_face_flux_slots[dimension].empty())
        std::terminate();
      accepted_face_flux_ordinals_[dimension].clear();
    }
  } else {
    for (int axis = 0; axis < Dim; ++axis) {
      const auto dimension = static_cast<std::size_t>(axis);
      auto& ordinals = accepted_face_flux_ordinals_[dimension];
      const auto& slots = staging.accepted_face_flux_slots[dimension];
      if (ordinals.capacity() < slots.size())
        std::terminate();
      ordinals.resize(slots.size());
      for (auto& ordinal : ordinals)
        ordinal = {};
    }
    multiblock_subcycling_->bind_candidate_ledger_slots(
        [&](std::size_t, std::size_t, std::size_t, multiblock_flux_ledger_type& ledger) noexcept {
          for (int axis = 0; axis < Dim; ++axis) {
            const auto dimension = static_cast<std::size_t>(axis);
            auto& ordinals = accepted_face_flux_ordinals_[dimension];
            const auto& slots = staging.accepted_face_flux_slots[dimension];
            const auto templates = ledger.resident_slot_templates(axis);
            for (std::size_t source_slot = 0; source_slot < templates.size(); ++source_slot) {
              std::size_t match = slots.size();
              for (std::size_t slot = 0; slot < slots.size(); ++slot) {
                if (!same_wire_identity(templates[source_slot], slots[slot]))
                  continue;
                if (match != slots.size())
                  std::terminate();
                match = slot;
              }
              if (match == slots.size() || ordinals[match].ledger != nullptr)
                std::terminate();
              ordinals[match] = {std::addressof(ledger), source_slot, match};
            }
          }
        });
    for (int axis = 0; axis < Dim; ++axis)
      for (const auto& ordinal : accepted_face_flux_ordinals_[static_cast<std::size_t>(axis)])
        if (ordinal.ledger == nullptr)
          std::terminate();
  }
  accepted_face_flux_ordinal_owner_ = multiblock_subcycling_.get();
  accepted_face_flux_ordinal_epoch_ = resource_epoch_;
  accepted_face_flux_ordinal_generation_ = resource_generation_;
}

void rebind_accepted_interface_flux_ordinals_preallocated_noexcept_() const noexcept {
  if (interface_flux_ledger_ && interface_flux_ledger_->in_transaction())
    std::terminate();
  const std::size_t bound = accepted_interface_flux_staging_sources_.capacity();
  if (accepted_interface_flux_wire_ordinals_.capacity() < bound)
    std::terminate();
  accepted_interface_flux_wire_ordinals_.resize(bound);
  for (std::size_t ordinal = 0; ordinal < bound; ++ordinal)
    accepted_interface_flux_wire_ordinals_[ordinal] = ordinal;
  accepted_interface_flux_ordinal_owner_ = interface_flux_ledger_.get();
  accepted_interface_flux_ordinal_epoch_ = resource_epoch_;
  accepted_interface_flux_ordinal_generation_ = resource_generation_;
}

void prime_accepted_state_staging_at_bind_() const {
  if (active_attempt_states_.size() != 0)
    throw std::logic_error("AMR Program accepted-state staging cannot prime during an attempt");

  auto& staging = accepted_state_staging_;
  staging.valid = false;
  // Bind/regrid is deliberately not a reconstruction boundary.  The detached image created all
  // history keys, state-slot bindings, pending-remap slots, and nested string envelopes before
  // the sole resource-plan seal.  In particular, do not resolve ProgramRuntimeState here: while
  // an accepted facade is being replaced that resolver can still name the prior Program image.
  if (!staging.prepared_envelope)
    throw std::logic_error("AMR Program accepted-state staging was not prepared before seal");
  if (staging.topology_epoch != resource_epoch_)
    throw std::logic_error(
        "AMR Program accepted-state staging detached topology epoch is stale: staging=" +
        std::to_string(staging.topology_epoch) + ", resource=" + std::to_string(resource_epoch_));
  if (staging.materialization_generation != resource_generation_)
    throw std::logic_error(
        "AMR Program accepted-state staging detached materialization generation is stale: "
        "staging=" +
        std::to_string(staging.materialization_generation) +
        ", resource=" + std::to_string(resource_generation_));
  if (staging.configured_level_count == 0 ||
      staging.state.level_clocks.capacity() < staging.configured_level_count)
    throw std::logic_error("AMR Program accepted-state staging detached level envelope is stale");
  if (staging.history_slot_bindings.size() != staging.history_slot_pool.size() ||
      staging.pending_history_keys.size() != staging.pending_history_remap_slots.size())
    throw std::logic_error("AMR Program accepted-state staging detached slot envelope is stale");
  if (!staging.state.history_slots.empty() || !staging.history_slot_active_indices.empty() ||
      !staging.state.pending_history_remaps.empty() ||
      !staging.pending_history_active_slots.empty())
    throw std::logic_error("AMR Program accepted-state staging detached logical image is stale");

  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    const auto& slots = staging.accepted_face_flux_slots[index];
    const auto& state_axis = staging.state.accepted_face_flux[index];
    const auto& sources = staging.accepted_face_flux_sources[index];
    const auto& active_slots = staging.accepted_face_flux_active_slots[index];
    // The detached candidate has already copied and sorted the complete resident templates.
    // Bind only verifies that no previous logical accepted image leaked into this candidate.
    if (!state_axis.empty() || !sources.empty() || !active_slots.empty() ||
        state_axis.capacity() < slots.size() || sources.capacity() < slots.size() ||
        active_slots.capacity() < slots.size())
      throw std::logic_error(
          "AMR Program accepted face-flux envelope was not prepared before seal");
  }
  if (!staging.state.synchronization_events.empty() ||
      !staging.synchronization_event_active_indices.empty() ||
      !staging.state.accepted_interface_flux.empty() ||
      !staging.accepted_interface_flux_active_slots.empty() ||
      !accepted_interface_flux_staging_sources_.empty())
    throw std::logic_error(
        "AMR Program accepted synchronization slots differ from their envelope: events=" +
        std::to_string(staging.state.synchronization_events.size()) +
        ", event-active=" + std::to_string(staging.synchronization_event_active_indices.size()) +
        ", interface-logical=" + std::to_string(staging.state.accepted_interface_flux.size()) +
        ", interface-active=" +
        std::to_string(staging.accepted_interface_flux_active_slots.size()) +
        ", interface-sources=" + std::to_string(accepted_interface_flux_staging_sources_.size()));
  for (int axis = 0; axis < Dim; ++axis) {
    auto& accepted = accepted_face_flux_[static_cast<std::size_t>(axis)];
    const auto& slots = accepted_face_flux_commit_slots_[static_cast<std::size_t>(axis)];
    if (accepted.capacity() < slots.size() ||
        slots.size() != staging.accepted_face_flux_slots[static_cast<std::size_t>(axis)].size())
      throw std::logic_error(
          "AMR Program accepted face-flux commit envelope was not prepared before seal");
  }
  if (accepted_synchronization_events_.capacity() <
          accepted_synchronization_event_commit_slots_.size() ||
      accepted_synchronization_event_commit_slots_.size() !=
          staging.synchronization_event_slots.size())
    throw std::logic_error(
        "AMR Program accepted synchronization commit envelope was not prepared before seal");

  // Seed the immutable temporal-partition identity once, while binding is still cold.  The hot
  // copier below intentionally refuses a different kind/topology/cell route; leaving this state
  // value-initialized would turn the first accepted refresh into either an allocating assignment
  // or a false shape failure.
  const auto& accepted_partition = accepted_temporal_partition_;
  auto& staged_partition = staging.state.temporal_partition;
  if (accepted_partition.provider_identity.size() > staged_partition.provider_identity.capacity() ||
      accepted_partition.cells.size() > staged_partition.cells.capacity())
    throw std::logic_error(
        "AMR Program accepted-state temporal partition envelope was not primed: provider=" +
        std::to_string(accepted_partition.provider_identity.size()) + "/" +
        std::to_string(staged_partition.provider_identity.capacity()) +
        ", cells=" + std::to_string(accepted_partition.cells.size()) + "/" +
        std::to_string(staged_partition.cells.capacity()));
  staged_partition.kind = accepted_partition.kind;
  staged_partition.topology_epoch = accepted_partition.topology_epoch;
  staged_partition.synchronization_tick = accepted_partition.synchronization_tick;
  staged_partition.tick_denominator = accepted_partition.tick_denominator;
  staged_partition.provider_identity.resize(accepted_partition.provider_identity.size());
  std::copy(accepted_partition.provider_identity.begin(),
            accepted_partition.provider_identity.end(), staged_partition.provider_identity.begin());
  staged_partition.cells.resize(accepted_partition.cells.size());
  std::copy(accepted_partition.cells.begin(), accepted_partition.cells.end(),
            staged_partition.cells.begin());
  // The post-publication topology handoff may only resize these carriers within their cold-sealed
  // capacities.  Reserve against the complete staging envelope while this adapter is still the
  // sole owner.
  for (int axis = 0; axis < Dim; ++axis) {
    const auto dimension = static_cast<std::size_t>(axis);
    const std::size_t face_slots = staging.accepted_face_flux_slots[dimension].size();
    if (accepted_face_flux_ordinals_[dimension].capacity() < face_slots)
      accepted_face_flux_ordinals_[dimension].reserve(face_slots);
  }
  const std::size_t interface_slots = accepted_interface_flux_staging_sources_.capacity();
  if (accepted_interface_flux_wire_ordinals_.capacity() < interface_slots)
    accepted_interface_flux_wire_ordinals_.reserve(interface_slots);
  rebind_accepted_face_flux_ordinals_at_cold_prime_();
  rebind_accepted_interface_flux_ordinals_at_cold_prime_();
  staging.primed = true;
}

void fill_accepted_state_staging_() const {
  auto& staging = accepted_state_staging_;
  staging.valid = false;
  if (!staging.primed || staging.topology_epoch != runtime_->topology_epoch() ||
      staging.materialization_generation != runtime_->materialization_generation())
    throw std::logic_error("AMR Program accepted-state staging is stale or was not cold-primed");
  auto& state = staging.state;
  const auto copy_string = [](std::string& destination, std::string_view source, const char* what) {
    if (source.size() > destination.capacity())
      throw std::logic_error(std::string("AMR Program accepted-state staging ") + what +
                             " capacity was not primed");
    destination.resize(source.size());
    std::copy(source.begin(), source.end(), destination.begin());
  };
  copy_string(state.spatial_contract, runtime_->spatial_contract(), "spatial contract");
  state.topology_epoch = staging.topology_epoch;
  state.materialization_generation = staging.materialization_generation;
  const std::size_t levels = runtime_->hierarchy().num_levels();
  if (levels == 0 || levels > staging.configured_level_count ||
      levels > state.level_clocks.capacity())
    throw std::logic_error("AMR Program accepted-state staging level shape changed after bind");
  state.level_clocks.resize(levels);
  for (std::size_t level = 0; level < levels; ++level)
    state.level_clocks[level] = {static_cast<int>(level), macro_step(), {0, 1}, physical_time()};
  clock_schedule_.accepted_ticks_in_wire_order_into(state.logical_clock_ticks,
                                                    static_cast<std::int64_t>(macro_step()));

  // This is an accepted-step copy, not a value assignment: assignment is permitted to replace
  // the nested string/vector allocation even when the apparent outer capacity is sufficient.
  // Keep the equivalent of the later cold-copy helper local here: this fragment is parsed before
  // the accepted-snapshot capacity definitions, so depending on that member would create an
  // include-order cycle.
  const auto& partition = accepted_temporal_partition_;
  auto& staged_partition = state.temporal_partition;
  if (staged_partition.kind != partition.kind ||
      staged_partition.topology_epoch != partition.topology_epoch ||
      staged_partition.cells.size() != partition.cells.size() ||
      partition.provider_identity.size() > staged_partition.provider_identity.capacity())
    throw std::logic_error(
        "AMR Program accepted-state staging temporal partition changed after bind");
  for (std::size_t index = 0; index < partition.cells.size(); ++index)
    if (staged_partition.cells[index].level != partition.cells[index].level ||
        staged_partition.cells[index].cell != partition.cells[index].cell ||
        staged_partition.cells[index].rung != partition.cells[index].rung)
      throw std::logic_error(
          "AMR Program accepted-state staging temporal partition route changed after bind");
  copy_string(staged_partition.provider_identity, partition.provider_identity,
              "temporal provider identity");
  staged_partition.synchronization_tick = partition.synchronization_tick;
  staged_partition.tick_denominator = partition.tick_denominator;
  for (std::size_t index = 0; index < partition.cells.size(); ++index)
    staged_partition.cells[index].accepted_tick = partition.cells[index].accepted_tick;
  // Flux and coupling contracts are part of the accepted temporal authority, not derived
  // opportunistically from a live facade during the hot refresh.  The installation/regrid
  // candidate has already prepared their exact values and primed both destination strings; copy
  // them into the staging image explicitly so the subsequent no-throw swap cannot publish an
  // empty contract left over from the cold envelope template.
  copy_string(state.flux_budget_contract, accepted_flux_budget_contract_, "flux-budget contract");
  copy_string(state.coupling_contract, accepted_coupling_contract_, "coupling contract");
  // The facade owns the accepted tagging state.  Copy its canonical payload through the private
  // preallocated seam: retaining a prior staging payload would silently publish stale hysteresis
  // after a regrid, while constructing `encode()` here would allocate in Candidate.
  facade_->program_copy_tagging_hysteresis_state_into_(state.tagging_hysteresis_state);

  const auto& manager = runtime_state().hist_;
  if (accepted_history_ordinal_owner_ != std::addressof(manager) ||
      accepted_history_ordinal_epoch_ != resource_epoch_ ||
      accepted_history_ordinal_generation_ != resource_generation_ ||
      accepted_history_binding_mutation_slots_.size() != staging.history_slot_bindings.size() ||
      accepted_pending_history_ordinal_sources_.size() != staging.pending_history_keys.size())
    throw std::logic_error(
        "AMR Program accepted-state history ordinals were not rebound at the cold boundary");
  std::size_t expected_history_rings = 0;
  for (std::size_t binding_index = 0; binding_index < staging.history_slot_bindings.size();
       ++binding_index) {
    const auto& binding = staging.history_slot_bindings[binding_index];
    if (binding.state_slot >= staging.history_slot_pool.size())
      throw std::logic_error("AMR Program accepted-state staging history pool is malformed");
    const int configured_level = staging.history_slot_pool[binding.state_slot].level;
    if (configured_level < 0)
      throw std::logic_error("AMR Program accepted-state history level is malformed");
    const std::size_t mutation_slot = accepted_history_binding_mutation_slots_[binding_index];
    if (static_cast<std::size_t>(configured_level) >= levels) {
      if (mutation_slot != kInactiveHistoryMutationOrdinal_)
        throw std::logic_error("AMR Program inactive history ordinal was rebound prematurely");
      continue;
    }
    if (mutation_slot >= prepared_history_mutation_slots_.size())
      throw std::logic_error("AMR Program accepted-state history ordinal is malformed");
    const auto& live = prepared_history_mutation_slots_[mutation_slot];
    if (live.live_ring == nullptr || live.live_dts == nullptr || live.live_initialized == nullptr ||
        live.live_fill_count == nullptr || binding.source_slot >= live.live_ring->size() ||
        binding.source_slot >= live.live_dts->size())
      throw std::logic_error("AMR Program accepted-state history ordinal changed after bind");
    if (binding.source_slot == 0)
      ++expected_history_rings;
  }
  if (expected_history_rings != manager.histories.size())
    throw std::logic_error(
        "AMR Program accepted-state staging history ring count changed after bind");
  for (std::size_t index = 0; index < state.history_slots.size(); ++index) {
    const std::size_t pool = staging.history_slot_active_indices[index];
    if (pool >= staging.history_slot_pool.size())
      throw std::logic_error("AMR Program accepted-state staging history active slot is invalid");
    state.history_slots[index].name.swap(staging.history_slot_pool[pool].name);
  }
  state.history_slots.clear();
  staging.history_slot_active_indices.clear();
  std::size_t active_history_slots = 0;
  for (const auto& binding : staging.history_slot_bindings)
    if (static_cast<std::size_t>(staging.history_slot_pool[binding.state_slot].level) < levels)
      ++active_history_slots;
  if (active_history_slots > state.history_slots.capacity() ||
      active_history_slots > staging.history_slot_active_indices.capacity())
    throw std::logic_error("AMR Program accepted-state staging history envelope was not primed");
  state.history_slots.resize(active_history_slots);
  staging.history_slot_active_indices.resize(active_history_slots);
  std::size_t active_history_index = 0;
  for (std::size_t binding_index = 0; binding_index < staging.history_slot_bindings.size();
       ++binding_index) {
    const auto& binding = staging.history_slot_bindings[binding_index];
    const auto& pool = staging.history_slot_pool[binding.state_slot];
    if (static_cast<std::size_t>(pool.level) >= levels)
      continue;
    const std::size_t mutation_slot = accepted_history_binding_mutation_slots_[binding_index];
    if (mutation_slot == kInactiveHistoryMutationOrdinal_ ||
        mutation_slot >= prepared_history_mutation_slots_.size())
      throw std::logic_error("AMR Program active history ordinal was not rebound cold");
    const auto& live = prepared_history_mutation_slots_[mutation_slot];
    auto& target = state.history_slots[active_history_index];
    target.name.swap(staging.history_slot_pool[binding.state_slot].name);
    target.level = pool.level;
    target.slot = pool.slot;
    target.outgoing_dt = static_cast<double>((*live.live_dts)[binding.source_slot]);
    target.initialized = *live.live_initialized;
    target.fill_count = *live.live_fill_count;
    staging.history_slot_active_indices[active_history_index] = binding.state_slot;
    ++active_history_index;
  }
  const std::size_t active_pending_count = static_cast<std::size_t>(std::count_if(
      accepted_pending_history_ordinal_sources_.begin(),
      accepted_pending_history_ordinal_sources_.end(),
      [](const AmrProgramPendingHistoryRemap* row) { return row != nullptr && !row->consumed; }));
  if (active_pending_count > state.pending_history_remaps.capacity() ||
      active_pending_count > staging.pending_history_remap_slots.size())
    throw std::logic_error(
        "AMR Program accepted-state staging pending-remap capacity was not primed");
  const std::size_t prior_pending_count = state.pending_history_remaps.size();
  for (std::size_t index = active_pending_count; index < prior_pending_count; ++index) {
    const std::size_t slot = staging.pending_history_active_slots[index];
    state.pending_history_remaps[index].key.swap(staging.pending_history_remap_slots[slot].key);
  }
  state.pending_history_remaps.resize(active_pending_count);
  staging.pending_history_active_slots.resize(active_pending_count);
  std::size_t index = 0;
  for (std::size_t ordinal = 0; ordinal < accepted_pending_history_ordinal_sources_.size();
       ++ordinal) {
    const auto* pending = accepted_pending_history_ordinal_sources_[ordinal];
    if (pending == nullptr)
      continue;
    if (pending->consumed)
      continue;
    const std::size_t slot = ordinal;
    if (state.pending_history_remaps[index].key != pending->key) {
      if (index < prior_pending_count) {
        const std::size_t prior_slot = staging.pending_history_active_slots[index];
        if (prior_slot < staging.pending_history_remap_slots.size())
          state.pending_history_remaps[index].key.swap(
              staging.pending_history_remap_slots[prior_slot].key);
      }
      state.pending_history_remaps[index].key.swap(staging.pending_history_remap_slots[slot].key);
    }
    if (pending->key.size() > state.pending_history_remaps[index].key.capacity())
      throw std::logic_error(
          "AMR Program accepted-state staging pending-remap key capacity was not primed");
    auto& target = state.pending_history_remaps[index];
    target.parent_level = pending->parent_level;
    target.child_level = pending->child_level;
    target.prior_topology_epoch = pending->prior_topology_epoch;
    target.prior_materialization_generation = pending->prior_materialization_generation;
    target.published_topology_epoch = pending->published_topology_epoch;
    target.published_materialization_generation = pending->published_materialization_generation;
    target.accepted_macro_step = pending->accepted_macro_step;
    target.temporal_numerator = pending->temporal_numerator;
    target.temporal_denominator = pending->temporal_denominator;
    target.source_dt = pending->source_dt;
    target.target_dt = pending->target_dt;
    target.consumed = pending->consumed;
    staging.pending_history_active_slots[index] = slot;
    ++index;
  }
  serialize_history_flux_payload_into_(state.history_flux_payload);

  for (std::size_t index = 0; index < state.synchronization_events.size(); ++index) {
    const std::size_t pool = staging.synchronization_event_active_indices[index];
    if (pool >= staging.synchronization_event_slots.size())
      throw std::logic_error("AMR Program accepted-state staging event active slot is invalid");
    state.synchronization_events[index].phase.swap(staging.synchronization_event_slots[pool].phase);
  }
  state.synchronization_events.clear();
  staging.synchronization_event_active_indices.clear();
  const std::size_t blocks = facade_->prepared_amr_multiblock_hierarchy_().block_count();
  const std::size_t active_event_count = blocks * (levels - 1U) * 2U;
  if (active_event_count > state.synchronization_events.capacity() ||
      active_event_count > staging.synchronization_event_active_indices.capacity())
    throw std::logic_error("AMR Program accepted-state staging event envelope was not primed");
  state.synchronization_events.resize(active_event_count);
  staging.synchronization_event_active_indices.resize(active_event_count);
  std::size_t active_event_index = 0;
  for (std::size_t block = 0; block < blocks; ++block)
    for (std::size_t parent = 0; parent + 1 < levels; ++parent)
      for (std::size_t phase = 0; phase < 2; ++phase) {
        const std::size_t pool =
            (block * (staging.configured_level_count - 1U) + parent) * 2U + phase;
        if (pool >= staging.synchronization_event_slots.size())
          throw std::logic_error("AMR Program accepted-state staging event pool is malformed");
        const auto& source = staging.synchronization_event_slots[pool];
        auto& destination = state.synchronization_events[active_event_index];
        destination.phase.swap(staging.synchronization_event_slots[pool].phase);
        destination.parent_level = source.parent_level;
        destination.child_level = source.child_level;
        destination.runtime_block = source.runtime_block;
        destination.clock = {destination.parent_level, macro_step(), {0, 1}, physical_time()};
        staging.synchronization_event_active_indices[active_event_index] = pool;
        ++active_event_index;
      }

  if (accepted_face_flux_ordinal_owner_ != multiblock_subcycling_.get() ||
      accepted_face_flux_ordinal_epoch_ != resource_epoch_ ||
      accepted_face_flux_ordinal_generation_ != resource_generation_)
    throw std::logic_error(
        "AMR Program accepted face-flux ordinals were not rebound at the cold boundary");
  for (int axis = 0; axis < Dim; ++axis) {
    const auto dimension = static_cast<std::size_t>(axis);
    // Non-owning source pointers belong only to the cold adapter ordinals.  The staging image is
    // value-owned and intentionally carries no source reference across a snapshot/candidate copy.
    staging.accepted_face_flux_sources[dimension].clear();
    const auto& ordinals = accepted_face_flux_ordinals_[dimension];
    auto& target = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    auto& slots = staging.accepted_face_flux_slots[static_cast<std::size_t>(axis)];
    auto& active_slots = staging.accepted_face_flux_active_slots[static_cast<std::size_t>(axis)];
    if (ordinals.size() != slots.size() || target.size() != active_slots.size() ||
        target.capacity() < ordinals.size() || active_slots.capacity() < ordinals.size())
      throw std::logic_error("AMR Program accepted face-flux logical envelope was not primed");
    const std::size_t prior = target.size();
    for (std::size_t index = 0; index < prior; ++index) {
      const std::size_t slot = active_slots[index];
      if (slot >= slots.size())
        throw std::logic_error("AMR Program accepted face-flux active slot is invalid");
      slots[slot].key.owner = std::move(target[index].key.owner);
      slots[slot].key.state = std::move(target[index].key.state);
      slots[slot].key.stage = std::move(target[index].key.stage);
      slots[slot].payload = std::move(target[index].payload);
    }
    // The logical accepted vector is normally empty at a freshly published topology.  Its
    // storage is bind-reserved, but indexing reserved raw storage before constructing the
    // entries is undefined behaviour and can expose moved-from identity strings to the
    // checkpoint serializer.  Materialize the complete finite envelope in place first; the
    // active prefix is compacted below without growing either vector.
    target.resize(ordinals.size());
    active_slots.resize(ordinals.size());
    std::size_t active_count = 0;
    for (const auto& ordinal : ordinals) {
      if (ordinal.ledger == nullptr || ordinal.staging_slot >= slots.size())
        throw std::logic_error("AMR Program accepted face-flux ordinal is malformed");
      const auto published = ordinal.ledger->published_entries(axis);
      if (ordinal.source_slot >= published.size())
        continue;
      const auto& source = published[ordinal.source_slot];
      const std::size_t slot = ordinal.staging_slot;
      auto& destination = target[active_count];
      destination.key.owner = std::move(slots[slot].key.owner);
      destination.key.state = std::move(slots[slot].key.state);
      destination.key.stage = std::move(slots[slot].key.stage);
      destination.payload = std::move(slots[slot].payload);
      if (source.key.owner.size() > destination.key.owner.capacity() ||
          source.key.state.size() > destination.key.state.capacity() ||
          source.key.stage.size() > destination.key.stage.capacity() ||
          source.payload.size() > destination.payload.capacity())
        throw std::logic_error("AMR Program accepted face-flux capacity was not primed");
      copy_string(destination.key.owner, source.key.owner, "face owner");
      copy_string(destination.key.state, source.key.state, "face state");
      copy_string(destination.key.stage, source.key.stage, "face stage");
      destination.key.clock = source.key.clock;
      destination.key.attempt = source.key.attempt;
      destination.key.levels = source.key.levels;
      destination.key.centering = source.key.centering;
      destination.key.axis = source.key.axis;
      destination.key.face = source.key.face;
      destination.key.coarse_face = source.key.coarse_face;
      destination.key.role = source.key.role;
      destination.key.contribution = source.key.contribution;
      destination.measure = source.measure;
      destination.payload.resize(source.payload.size());
      std::copy(source.payload.begin(), source.payload.end(), destination.payload.begin());
      active_slots[active_count] = slot;
      ++active_count;
    }
    target.resize(active_count);
    active_slots.resize(active_count);
  }
  if (interface_flux_ledger_ && interface_flux_ledger_->in_transaction())
    throw std::logic_error(
        "AMR Program accepted-state staging observed an active interface ledger");
  if (interface_flux_ledger_) {
    if (accepted_interface_flux_ordinal_owner_ != interface_flux_ledger_.get() ||
        accepted_interface_flux_ordinal_epoch_ != resource_epoch_ ||
        accepted_interface_flux_ordinal_generation_ != resource_generation_)
      throw std::logic_error("AMR Program accepted interface-flux ordinals were not rebound cold");
    auto& sources = accepted_interface_flux_staging_sources_;
    sources.clear();
    std::size_t source_ordinal = 0;
    interface_flux_ledger_->for_each_published(
        [&](typename interface_flux_ledger_type::FragmentView source) {
          if (sources.size() == sources.capacity() ||
              source_ordinal >= accepted_interface_flux_wire_ordinals_.size())
            throw std::logic_error(
                "AMR Program accepted interface-flux source exceeded its bind budget");
          const std::size_t wire_ordinal = accepted_interface_flux_wire_ordinals_[source_ordinal++];
          if (wire_ordinal != sources.size())
            throw std::logic_error(
                "AMR Program accepted interface-flux wire ordinal changed after bind");
          sources.push_back({source.key, source.measure, source.payload, wire_ordinal});
        });
    // The cold-bound ledger contract seals dense source order and this ordinal permutation.
    // Candidate only copies that finite order; sorting or comparing identity text here would
    // manufacture a second authority after the accepted writer lease is held.
    if (sources.size() > sources.capacity())
      throw std::logic_error("AMR Program accepted interface-flux envelope was not primed");
    std::size_t identity_characters = 0;
    std::size_t payload_terms = 0;
    for (const auto& source : sources) {
      if (source.key.topology_epoch != state.topology_epoch ||
          source.key.interface_identity.empty() || source.key.stage_identity.empty() ||
          source.key.graph_identity.empty() || source.key.rate_identity.empty() ||
          source.key.application_identity.empty() || source.key.coarse_level < 0 ||
          source.key.fine_level != source.key.coarse_level + 1 ||
          source.key.left_block == source.key.right_block ||
          !source.measure.stage_weight_resolved || source.payload.empty())
        throw std::logic_error("AMR Program accepted interface-flux source is not resolved");
      const std::size_t characters =
          source.key.interface_identity.size() + source.key.stage_identity.size() +
          source.key.graph_identity.size() + source.key.rate_identity.size() +
          source.key.application_identity.size();
      if (characters >
              interface_flux_ledger_->budget().max_identity_characters - identity_characters ||
          source.payload.size() >
              interface_flux_ledger_->budget().max_payload_terms_per_window - payload_terms)
        throw std::logic_error(
            "AMR Program accepted interface-flux source exceeds its bind budget");
      identity_characters += characters;
      payload_terms += source.payload.size();
      for (Real component : source.payload)
        if (!std::isfinite(static_cast<double>(component)))
          throw std::logic_error("AMR Program accepted interface-flux payload is not finite");
    }
    if (identity_characters > interface_flux_ledger_->budget().max_identity_characters ||
        payload_terms > interface_flux_ledger_->budget().max_payload_terms_per_window)
      throw std::logic_error("AMR Program accepted interface-flux source exceeds its bind budget");
    state.accepted_interface_flux.clear();
    staging.accepted_interface_flux_active_slots.clear();
  }
  staging.valid = true;
}

void require_accepted_state_staging_commit_preallocated_(
    const AmrProgramAcceptedState<Dim>& state) const {
  if (state.temporal_partition.provider_identity.size() >
          accepted_temporal_partition_.provider_identity.capacity() ||
      state.temporal_partition.cells.size() > accepted_temporal_partition_.cells.capacity() ||
      state.flux_budget_contract.size() > accepted_flux_budget_contract_.capacity() ||
      state.coupling_contract.size() > accepted_coupling_contract_.capacity())
    throw std::logic_error("AMR Program accepted-state commit capacity was not primed");
  for (int axis = 0; axis < Dim; ++axis) {
    const auto& source = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    const auto& destination = accepted_face_flux_[static_cast<std::size_t>(axis)];
    const auto& slots = accepted_face_flux_commit_slots_[static_cast<std::size_t>(axis)];
    if (source.size() > destination.capacity() || source.size() > slots.size())
      throw std::logic_error("AMR Program accepted face-flux commit capacity was not primed");
    for (std::size_t index = 0; index < source.size(); ++index) {
      const auto& target = index < destination.size() ? destination[index] : slots[index];
      if (source[index].key.owner.size() > target.key.owner.capacity() ||
          source[index].key.state.size() > target.key.state.capacity() ||
          source[index].key.stage.size() > target.key.stage.capacity() ||
          source[index].payload.size() > target.payload.capacity())
        throw std::logic_error("AMR Program accepted face-flux commit envelope was not primed");
    }
  }
  if (state.synchronization_events.size() > accepted_synchronization_events_.capacity() ||
      state.synchronization_events.size() > accepted_synchronization_event_commit_slots_.size())
    throw std::logic_error("AMR Program accepted synchronization commit capacity was not primed");
}

void commit_accepted_state_staging_noexcept_(AmrProgramAcceptedState<Dim>& state) const noexcept {
  const auto copy_string = [](std::string& destination, std::string_view source) noexcept {
    if (source.size() > destination.capacity())
      std::terminate();
    destination.resize(source.size());
    std::copy(source.begin(), source.end(), destination.begin());
  };
  // These staging fields have been preflighted before collective publication.  Swap retains both
  // authenticated envelopes and makes the scalar accepted authority update terminal/no-throw.
  using std::swap;
  swap(accepted_temporal_partition_, state.temporal_partition);
  accepted_flux_budget_contract_.swap(state.flux_budget_contract);
  accepted_coupling_contract_.swap(state.coupling_contract);
  for (int axis = 0; axis < Dim; ++axis) {
    auto& destination = accepted_face_flux_[static_cast<std::size_t>(axis)];
    auto& slots = accepted_face_flux_commit_slots_[static_cast<std::size_t>(axis)];
    const auto& source = state.accepted_face_flux[static_cast<std::size_t>(axis)];
    const std::size_t prior = destination.size();
    for (std::size_t index = source.size(); index < prior; ++index) {
      destination[index].key.owner.swap(slots[index].key.owner);
      destination[index].key.state.swap(slots[index].key.state);
      destination[index].key.stage.swap(slots[index].key.stage);
      destination[index].payload.swap(slots[index].payload);
    }
    destination.resize(source.size());
    for (std::size_t index = prior; index < source.size(); ++index) {
      destination[index].key.owner.swap(slots[index].key.owner);
      destination[index].key.state.swap(slots[index].key.state);
      destination[index].key.stage.swap(slots[index].key.stage);
      destination[index].payload.swap(slots[index].payload);
    }
    for (std::size_t index = 0; index < source.size(); ++index) {
      auto& target = destination[index];
      const auto& input = source[index];
      copy_string(target.key.owner, input.key.owner);
      copy_string(target.key.state, input.key.state);
      copy_string(target.key.stage, input.key.stage);
      target.key.clock = input.key.clock;
      target.key.attempt = input.key.attempt;
      target.key.levels = input.key.levels;
      target.key.centering = input.key.centering;
      target.key.axis = input.key.axis;
      target.key.face = input.key.face;
      target.key.coarse_face = input.key.coarse_face;
      target.key.role = input.key.role;
      target.key.contribution = input.key.contribution;
      target.measure = input.measure;
      target.payload.resize(input.payload.size());
      std::copy(input.payload.begin(), input.payload.end(), target.payload.begin());
    }
  }
  const std::size_t prior_events = accepted_synchronization_events_.size();
  for (std::size_t index = state.synchronization_events.size(); index < prior_events; ++index)
    accepted_synchronization_events_[index].phase.swap(
        accepted_synchronization_event_commit_slots_[index].phase);
  accepted_synchronization_events_.resize(state.synchronization_events.size());
  for (std::size_t index = prior_events; index < state.synchronization_events.size(); ++index)
    accepted_synchronization_events_[index].phase.swap(
        accepted_synchronization_event_commit_slots_[index].phase);
  for (std::size_t index = 0; index < state.synchronization_events.size(); ++index) {
    auto& target = accepted_synchronization_events_[index];
    const auto& input = state.synchronization_events[index];
    copy_string(target.phase, input.phase);
    target.parent_level = input.parent_level;
    target.child_level = input.child_level;
    target.runtime_block = input.runtime_block;
    target.clock = input.clock;
  }
}

// clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_lifecycle.hpp>
// clang-format on
