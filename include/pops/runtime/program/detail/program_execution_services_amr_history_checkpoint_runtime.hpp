void reconcile_multiblock_reflux_(multiblock_reflux_context_type& context) const {
  if (context.flux.published_size() == 0)
    return;
  const Geometry<Dim> geometry =
      facade_->program_prepared_amr_level_geometry_(static_cast<int>(context.parent_level));
  std::size_t maximum_fine_faces = 1;
  for (int axis = 0; axis < Dim; ++axis)
    maximum_fine_faces =
        checked_product_(maximum_fine_faces, static_cast<std::size_t>(context.spatial_ratio[axis]),
                         "AMR Program reflux fine-face budget");
  const ::pops::amr::reflux::MetricRefluxBudget budget{
      maximum_fine_faces, std::max<std::size_t>(context.flux.published_size(), 1),
      std::max<std::size_t>(context.flux.published_size(), 1)};
  const std::string state_identity = std::string(context.block_identity) + "/state";
  for (const ProgramInterfaceFace& interface : program_interface_faces_(context.parent_level)) {
    ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
    query.owner = std::string(context.block_identity);
    query.state = state_identity;
    query.levels = {static_cast<int>(context.parent_level),
                    static_cast<int>(context.parent_level + 1)};
    query.axis = interface.axis;
    query.coarse_face = interface.coarse_face;
    query.attempt = context.attempt;
    query.macro_step = context.parent_window.begin.macro_step;
    query.window_begin = context.parent_window.begin.phase;
    query.window_end = context.parent_window.end.phase;
    bool found_coarse = false;
    const auto& entries = context.flux.published_entries(interface.axis);
    const auto matches_query = [&](const auto& entry) {
      return entry.key.owner == query.owner && entry.key.state == query.state &&
             entry.key.levels == query.levels && entry.key.axis == query.axis &&
             entry.key.coarse_face == query.coarse_face && entry.key.attempt == query.attempt &&
             entry.key.clock.macro_step == query.macro_step &&
             !(entry.key.clock.phase < query.window_begin) &&
             !(query.window_end < entry.key.clock.phase);
    };
    for (const auto& entry : entries) {
      if (!matches_query(entry))
        continue;
      found_coarse = found_coarse || entry.key.role == ::pops::amr::reflux::FaceLedgerRole::Coarse;
    }
    if (!found_coarse)
      throw std::runtime_error(
          "AMR Program reflux ledger lacks its block-qualified coarse face: owner=" + query.owner +
          " axis=" + std::to_string(query.axis) +
          " published=" + std::to_string(context.flux.published_size()));
    const auto reflux =
        facade_->prepared_amr_multiblock_hierarchy_().topology_runtime().reconcile_reflux_for_owner(
            context.flux, query, context.block_identity, state_identity, context.face_mapping,
            budget, payload_axpy_);
    const std::vector<Real> correction = ::pops::amr::reflux::coarse_cell_reflux_correction(
        reflux, cell_measure_(geometry), interface.side, payload_axpy_);
    apply_reflux_payload_(context.parent, interface.coarse_cell, correction);
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

AmrProgramAcceptedState<Dim> accepted_state_() const {
  require_facade_execution_();
  AmrProgramAcceptedState<Dim> state;
  state.spatial_contract = runtime_->spatial_contract();
  state.topology_epoch = runtime_->topology_epoch();
  state.materialization_generation = runtime_->materialization_generation();
  state.level_clocks.reserve(runtime_->hierarchy().num_levels());
  for (std::size_t level = 0; level < runtime_->hierarchy().num_levels(); ++level)
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
  clock_schedule_.restore_accepted_ticks(state.logical_clock_ticks, accepted_macro_step);
  static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_flux_budget_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_coupling_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
  history_flux_expressions_.swap(restored_history_flux);
  std::swap(accepted_temporal_partition_, state.temporal_partition);
  pending_history_remaps_.swap(restored_pending);
  deferred_history_lag_scratches_.clear();
  for (const auto& diagnostic : cell_temporal_diagnostics_)
    if (diagnostic)
      diagnostic->invalidate_accepted_publication(accepted_temporal_partition_.synchronization_tick,
                                                  accepted_temporal_partition_.tick_denominator);
  accepted_flux_budget_contract_.swap(state.flux_budget_contract);
  accepted_coupling_contract_.swap(state.coupling_contract);
  std::swap(accepted_face_flux_, state.accepted_face_flux);
  interface_flux_commit_guard_.reset();
  interface_flux_ledger_.swap(restored_interface_flux);
  accepted_synchronization_events_.swap(state.synchronization_events);
  accepted_state_revision_ = revision;
}

void refresh_accepted_hierarchy_state_(bool prepare_subcycling = true) const {
  require_facade_execution_();
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR Program accepted-state refresh crossed an active attempt");
  refresh_resources_();
  requalify_cell_temporal_configuration_();
  if (prepare_subcycling)
    prepare_multiblock_subcycling_engine_();
  const auto state = accepted_state_();
  facade_->restore_program_accepted_state(serialize_amr_program_accepted_state(state));
  accepted_state_revision_ = facade_->program_accepted_state_revision_();
  accepted_temporal_partition_ = state.temporal_partition;
  accepted_flux_budget_contract_ = state.flux_budget_contract;
  accepted_coupling_contract_ = state.coupling_contract;
  accepted_face_flux_ = state.accepted_face_flux;
  accepted_synchronization_events_ = state.synchronization_events;
}

void refresh_accepted_hierarchy_state_after_remap_(
    const AmrProgramHistoryRemapDescriptor& descriptor) const {
  struct RemapCandidate {
    std::map<std::string, AmrProgramPendingHistoryRemap> pending;
    std::string exact_contract;
  };
  RemapCandidate candidate = prepare_history_mutation_collectively_(
      [&]() {
        require_facade_execution_();
        if (!active_attempt_states_.empty())
          throw std::logic_error("AMR Program accepted history remap crossed an active attempt");
        RemapCandidate staged{pending_history_remaps_, {}};
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
            if (staged.pending.contains(key))
              throw std::runtime_error(
                  "AMR Program deferred history remap would supersede a pending lag");
            const double source_dt = static_cast<double>(dts[1]);
            staged.pending.emplace(
                key,
                AmrProgramPendingHistoryRemap{
                    key, descriptor.parent_level, descriptor.child_level,
                    descriptor.prior_topology_epoch, descriptor.prior_materialization_generation,
                    descriptor.published_topology_epoch,
                    descriptor.published_materialization_generation, descriptor.accepted_macro_step,
                    descriptor.temporal_numerator, descriptor.temporal_denominator, source_dt,
                    source_dt / static_cast<double>(descriptor.temporal_numerator), false});
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
            .scalar(static_cast<std::uint64_t>(staged.pending.size()));
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
  pending_history_remaps_.swap(candidate.pending);
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
