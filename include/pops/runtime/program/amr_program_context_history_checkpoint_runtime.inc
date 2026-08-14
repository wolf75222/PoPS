void reconcile_multiblock_reflux_(multiblock_reflux_context_type& context) const {
  if (context.flux.published_size() == 0)
    return;
  const Geometry<Dim> geometry =
      facade_->prepared_amr_level_geometry(static_cast<int>(context.parent_level));
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
    state.level_clocks.push_back(
        {static_cast<int>(level), facade_->macro_step(), {0, 1}, facade_->time()});
  state.logical_clock_ticks =
      clock_schedule_.accepted_ticks(static_cast<std::int64_t>(facade_->macro_step()));
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
    const auto& block_map = facade_->program_block_map();
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
        const ::pops::amr::ClockStamp clock{
            static_cast<int>(parent), facade_->macro_step(), {0, 1}, facade_->time()};
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
    state.accepted_interface_flux = interface_flux_ledger_->published_entries();
    std::sort(state.accepted_interface_flux.begin(), state.accepted_interface_flux.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
  }
  return state;
}

void import_accepted_state_(bool force) const {
  require_facade_execution_();
  const std::uint64_t revision = facade_->program_accepted_state_revision();
  if (!force && revision == accepted_state_revision_)
    return;
  prepare_multiblock_subcycling_engine_();
  const std::vector<std::uint8_t> bytes = facade_->program_accepted_state();
  if (bytes.empty())
    throw std::runtime_error("AMR Program accepted-state import received an empty image");
  AmrProgramAcceptedState<Dim> state =
      deserialize_amr_program_accepted_state<Dim>(bytes, &interface_flux_ledger_->budget());
  require_live_amr_program_checkpoint(state, *runtime_);
  const auto expected = accepted_state_();
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
  clock_schedule_.restore_accepted_ticks(state.logical_clock_ticks, accepted_macro_step);
  accepted_temporal_partition_ = std::move(state.temporal_partition);
  for (const auto& diagnostic : cell_temporal_diagnostics_)
    if (diagnostic)
      diagnostic->invalidate_accepted_publication(accepted_temporal_partition_.synchronization_tick,
                                                  accepted_temporal_partition_.tick_denominator);
  accepted_flux_budget_contract_ = std::move(state.flux_budget_contract);
  accepted_coupling_contract_ = std::move(state.coupling_contract);
  accepted_face_flux_ = std::move(state.accepted_face_flux);
  interface_flux_commit_guard_.reset();
  auto interface_budget = interface_flux_ledger_->budget();
  interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
      restore_amr_program_interface_flux_ledger(state, std::move(interface_budget)));
  accepted_synchronization_events_ = std::move(state.synchronization_events);
  accepted_state_revision_ = revision;
}

void refresh_accepted_hierarchy_state_() const {
  require_facade_execution_();
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR Program accepted-state refresh crossed an active attempt");
  refresh_resources_();
  requalify_cell_temporal_configuration_();
  const auto state = accepted_state_();
  facade_->restore_program_accepted_state(serialize_amr_program_accepted_state(state));
  accepted_state_revision_ = facade_->program_accepted_state_revision();
  accepted_temporal_partition_ = state.temporal_partition;
  accepted_flux_budget_contract_ = state.flux_budget_contract;
  accepted_coupling_contract_ = state.coupling_contract;
  accepted_face_flux_ = state.accepted_face_flux;
  accepted_synchronization_events_ = state.synchronization_events;
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
    for (int parent = 0; parent + 1 < facade_->configured_n_levels(); ++parent) {
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
  refresh_accepted_hierarchy_state_();
}

void resync_after_restart_() const {
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR restart resynchronization crossed an active Program attempt");
  multiblock_subcycling_.reset();
  synchronize_resource_generation_();
  import_accepted_state_(true);
}
