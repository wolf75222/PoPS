void require_refresh_preallocated_() const {
  if (resource_epoch_ != owner_->resource_epoch_ ||
      resource_generation_ != owner_->resource_generation_ ||
      history_epoch_ != owner_->history_epoch_ ||
      history_generation_ != owner_->history_generation_ ||
      interface_flux_ledger_->topology_epoch() !=
          owner_->interface_flux_ledger_->topology_epoch() ||
      interface_flux_ledger_->budget() != owner_->interface_flux_ledger_->budget())
    throw std::logic_error(
        "AMR Program accepted context changed after its resident transaction image was primed");
  require_history_levels_preallocated_(history_levels_, owner_->history_levels_);
  require_history_flux_expressions_preallocated_(history_flux_expressions_,
                                                 owner_->history_flux_expressions_);
  require_pending_history_remaps_preallocated_(pending_history_remaps_,
                                               owner_->pending_history_remaps_);
  require_deferred_history_lag_scratches_preallocated_(deferred_history_lag_scratches_,
                                                       owner_->deferred_history_lag_scratches_);
  require_temporal_partition_preallocated_(accepted_temporal_partition_,
                                           owner_->accepted_temporal_partition_);
  require_cell_temporal_configuration_preallocated_(cell_temporal_configuration_,
                                                    owner_->cell_temporal_configuration_);
  if (accepted_flux_budget_contract_ != owner_->accepted_flux_budget_contract_ ||
      accepted_coupling_contract_ != owner_->accepted_coupling_contract_)
    throw std::logic_error("AMR Program accepted flux authority changed after prime");
  require_events_preallocated_(accepted_synchronization_events_,
                               accepted_synchronization_event_slots_,
                               owner_->accepted_synchronization_events_);
  require_face_flux_preallocated_(accepted_face_flux_, accepted_face_flux_slots_,
                                  owner_->accepted_face_flux_);
  interface_flux_ledger_->require_preallocated_copy_from(*owner_->interface_flux_ledger_);
}

static void copy_string_preallocated_(std::string& destination, const std::string& source) {
  if (source.size() > destination.capacity())
    throw std::logic_error("AMR Program accepted context string capacity was not primed");
  destination.resize(source.size());
  std::copy(source.begin(), source.end(), destination.begin());
}

/// Cold-only copy companion.  A standard string/vector copy is intentionally allowed to use a
/// smaller capacity than its source; accepted rollback images must instead retain the finite
/// envelope authenticated at bind.  Do not call these helpers from refresh/publish/finalize.
static void prime_string_capacity_from_cold_source_(std::string& destination,
                                                    const std::string& source) {
  if (destination.capacity() < source.capacity())
    destination.reserve(source.capacity());
}

static void prime_event_capacities_from_cold_source_(
    std::vector<AmrProgramSynchronizationEvent>& destination,
    const std::vector<AmrProgramSynchronizationEvent>& source) {
  if (destination.size() != source.size())
    throw std::logic_error("AMR Program cold event copy changed its logical shape");
  if (destination.capacity() < source.capacity())
    destination.reserve(source.capacity());
  for (std::size_t index = 0; index < source.size(); ++index) {
    const auto& input = source[index];
    auto& target = destination[index];
    if (target.parent_level != input.parent_level || target.child_level != input.child_level ||
        target.runtime_block != input.runtime_block || target.phase != input.phase)
      throw std::logic_error("AMR Program cold event copy changed its identity");
    prime_string_capacity_from_cold_source_(target.phase, input.phase);
  }
}

static void prime_face_flux_capacities_from_cold_source_(
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>&
        destination,
    const std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>& source) {
  for (int axis = 0; axis < Dim; ++axis) {
    auto& target_axis = destination[static_cast<std::size_t>(axis)];
    const auto& input_axis = source[static_cast<std::size_t>(axis)];
    if (target_axis.size() != input_axis.size())
      throw std::logic_error("AMR Program cold face-flux copy changed its logical shape");
    if (target_axis.capacity() < input_axis.capacity())
      target_axis.reserve(input_axis.capacity());
    for (std::size_t index = 0; index < input_axis.size(); ++index) {
      auto& target = target_axis[index];
      const auto& input = input_axis[index];
      if (target.key.owner != input.key.owner || target.key.state != input.key.state ||
          target.key.levels != input.key.levels || target.key.centering != input.key.centering ||
          target.key.axis != input.key.axis || target.key.face != input.key.face ||
          target.key.coarse_face != input.key.coarse_face || target.key.stage != input.key.stage ||
          target.key.role != input.key.role || target.key.contribution != input.key.contribution ||
          target.payload.size() != input.payload.size())
        throw std::logic_error("AMR Program cold face-flux copy changed its identity");
      prime_string_capacity_from_cold_source_(target.key.owner, input.key.owner);
      prime_string_capacity_from_cold_source_(target.key.state, input.key.state);
      prime_string_capacity_from_cold_source_(target.key.stage, input.key.stage);
      if (target.payload.capacity() < input.payload.capacity())
        target.payload.reserve(input.payload.capacity());
    }
  }
}

static void prime_detached_state_capacities_from_cold_source_(
    DetachedState& destination, const AcceptedContextSnapshot& source) {
  if (!destination.interface_flux_ledger)
    throw std::logic_error("AMR Program detached cold copy lost its interface-flux ledger");
  prime_string_capacity_from_cold_source_(destination.accepted_flux_budget_contract,
                                          source.accepted_flux_budget_contract_);
  prime_string_capacity_from_cold_source_(destination.accepted_coupling_contract,
                                          source.accepted_coupling_contract_);
  prime_face_flux_capacities_from_cold_source_(destination.accepted_face_flux,
                                               source.accepted_face_flux_);
  prime_event_capacities_from_cold_source_(destination.accepted_synchronization_events,
                                           source.accepted_synchronization_events_);
  prime_string_capacity_from_cold_source_(destination.accepted_temporal_partition.provider_identity,
                                          source.accepted_temporal_partition_.provider_identity);
  if (destination.cell_temporal_configuration) {
    if (!source.cell_temporal_configuration_)
      throw std::logic_error("AMR Program detached cold copy changed cell-temporal authority");
    prime_string_capacity_from_cold_source_(destination.cell_temporal_configuration->clock,
                                            source.cell_temporal_configuration_->clock);
    prime_string_capacity_from_cold_source_(destination.cell_temporal_configuration->exact_contract,
                                            source.cell_temporal_configuration_->exact_contract);
  }
}

void prime_copied_capacities_from_owner_at_bind_() {
  require_owner_cold_prime_();
  prime_string_capacity_from_cold_source_(accepted_flux_budget_contract_,
                                          owner_->accepted_flux_budget_contract_);
  prime_string_capacity_from_cold_source_(accepted_coupling_contract_,
                                          owner_->accepted_coupling_contract_);
  prime_face_flux_capacities_from_cold_source_(accepted_face_flux_, owner_->accepted_face_flux_);
  prime_event_capacities_from_cold_source_(accepted_synchronization_events_,
                                           owner_->accepted_synchronization_events_);
  prime_face_flux_capacities_from_cold_source_(accepted_face_flux_slots_,
                                               owner_->accepted_face_flux_);
  prime_event_capacities_from_cold_source_(accepted_synchronization_event_slots_,
                                           owner_->accepted_synchronization_events_);
  prime_string_capacity_from_cold_source_(accepted_temporal_partition_.provider_identity,
                                          owner_->accepted_temporal_partition_.provider_identity);
  if (cell_temporal_configuration_) {
    if (!owner_->cell_temporal_configuration_)
      throw std::logic_error("AMR Program cold copy changed cell-temporal authority");
    prime_string_capacity_from_cold_source_(cell_temporal_configuration_->clock,
                                            owner_->cell_temporal_configuration_->clock);
    prime_string_capacity_from_cold_source_(cell_temporal_configuration_->exact_contract,
                                            owner_->cell_temporal_configuration_->exact_contract);
  } else if (owner_->cell_temporal_configuration_) {
    throw std::logic_error("AMR Program cold copy changed cell-temporal authority");
  }
  interface_flux_ledger_->prime_hot_carriers_at_bind();
}

void prime_copied_capacities_from_cold_source_(const AcceptedContextSnapshot& source) {
  if (!interface_flux_ledger_ || !source.interface_flux_ledger_ ||
      interface_flux_ledger_->in_transaction() || source.interface_flux_ledger_->in_transaction())
    throw std::logic_error("AMR Program accepted context cannot cold-prime an active ledger");
  prime_string_capacity_from_cold_source_(accepted_flux_budget_contract_,
                                          source.accepted_flux_budget_contract_);
  prime_string_capacity_from_cold_source_(accepted_coupling_contract_,
                                          source.accepted_coupling_contract_);
  prime_face_flux_capacities_from_cold_source_(accepted_face_flux_, source.accepted_face_flux_);
  prime_event_capacities_from_cold_source_(accepted_synchronization_events_,
                                           source.accepted_synchronization_events_);
  prime_face_flux_capacities_from_cold_source_(accepted_face_flux_slots_,
                                               source.accepted_face_flux_slots_);
  prime_event_capacities_from_cold_source_(accepted_synchronization_event_slots_,
                                           source.accepted_synchronization_event_slots_);
  prime_string_capacity_from_cold_source_(accepted_temporal_partition_.provider_identity,
                                          source.accepted_temporal_partition_.provider_identity);
  if (cell_temporal_configuration_) {
    if (!source.cell_temporal_configuration_)
      throw std::logic_error("AMR Program cold copy changed cell-temporal authority");
    prime_string_capacity_from_cold_source_(cell_temporal_configuration_->clock,
                                            source.cell_temporal_configuration_->clock);
    prime_string_capacity_from_cold_source_(cell_temporal_configuration_->exact_contract,
                                            source.cell_temporal_configuration_->exact_contract);
  } else if (source.cell_temporal_configuration_) {
    throw std::logic_error("AMR Program cold copy changed cell-temporal authority");
  }
  interface_flux_ledger_->prime_hot_carriers_at_bind();
}

static void copy_history_levels_preallocated_(std::map<std::string, int>& destination,
                                              const std::map<std::string, int>& source) {
  require_history_levels_preallocated_(destination, source);
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first)
      throw std::logic_error("AMR Program accepted history identity changed after prime");
    target->second = input->second;
  }
}

static void require_history_levels_preallocated_(const std::map<std::string, int>& destination,
                                                 const std::map<std::string, int>& source) {
  if (destination.size() != source.size())
    throw std::logic_error("AMR Program accepted history map shape changed after prime");
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target)
    if (target->first != input->first)
      throw std::logic_error("AMR Program accepted history identity changed after prime");
}

static void copy_history_flux_expressions_preallocated_(
    std::map<std::string, std::vector<FluxExpression>>& destination,
    const std::map<std::string, std::vector<FluxExpression>>& source) {
  require_history_flux_expressions_preallocated_(destination, source);
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first || target->second.size() != input->second.size())
      throw std::logic_error("AMR Program history flux registry identity changed after prime");
    for (std::size_t slot = 0; slot < input->second.size(); ++slot) {
      FluxExpression& target_expression = target->second[slot];
      const FluxExpression& input_expression = input->second[slot];
      if (target_expression.size() != input_expression.size())
        throw std::logic_error("AMR Program history flux basis shape changed after prime");
      auto target_term = target_expression.begin();
      auto input_term = input_expression.begin();
      for (; input_term != input_expression.end(); ++input_term, ++target_term) {
        if (target_term->first != input_term->first ||
            target_term->second.basis != input_term->second.basis ||
            target_term->second.coefficient.size() != input_term->second.coefficient.size())
          throw std::logic_error("AMR Program history flux basis identity changed after prime");
        auto target_coefficient = target_term->second.coefficient.begin();
        auto input_coefficient = input_term->second.coefficient.begin();
        for (; input_coefficient != input_term->second.coefficient.end();
             ++input_coefficient, ++target_coefficient) {
          if (target_coefficient->first != input_coefficient->first)
            throw std::logic_error("AMR Program history flux polynomial changed after prime");
          target_coefficient->second = input_coefficient->second;
        }
      }
    }
  }
}

static void require_history_flux_expressions_preallocated_(
    const std::map<std::string, std::vector<FluxExpression>>& destination,
    const std::map<std::string, std::vector<FluxExpression>>& source) {
  if (destination.size() != source.size())
    throw std::logic_error("AMR Program history flux registry shape changed after prime");
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first || target->second.size() != input->second.size())
      throw std::logic_error("AMR Program history flux registry identity changed after prime");
    for (std::size_t slot = 0; slot < input->second.size(); ++slot) {
      const FluxExpression& target_expression = target->second[slot];
      const FluxExpression& input_expression = input->second[slot];
      if (target_expression.size() != input_expression.size())
        throw std::logic_error("AMR Program history flux basis shape changed after prime");
      auto target_term = target_expression.begin();
      auto input_term = input_expression.begin();
      for (; input_term != input_expression.end(); ++input_term, ++target_term) {
        if (target_term->first != input_term->first ||
            target_term->second.basis != input_term->second.basis ||
            target_term->second.coefficient.size() != input_term->second.coefficient.size())
          throw std::logic_error("AMR Program history flux basis identity changed after prime");
        auto target_coefficient = target_term->second.coefficient.begin();
        auto input_coefficient = input_term->second.coefficient.begin();
        for (; input_coefficient != input_term->second.coefficient.end();
             ++input_coefficient, ++target_coefficient)
          if (target_coefficient->first != input_coefficient->first)
            throw std::logic_error("AMR Program history flux polynomial changed after prime");
      }
    }
  }
}

static void copy_pending_history_remaps_preallocated_(
    std::map<std::string, AmrProgramPendingHistoryRemap>& destination,
    const std::map<std::string, AmrProgramPendingHistoryRemap>& source) {
  require_pending_history_remaps_preallocated_(destination, source);
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first || input->second.key != input->first ||
        target->second.key != target->first)
      throw std::logic_error("AMR Program pending history remap identity changed after prime");
    target->second.parent_level = input->second.parent_level;
    target->second.child_level = input->second.child_level;
    target->second.prior_topology_epoch = input->second.prior_topology_epoch;
    target->second.prior_materialization_generation =
        input->second.prior_materialization_generation;
    target->second.published_topology_epoch = input->second.published_topology_epoch;
    target->second.published_materialization_generation =
        input->second.published_materialization_generation;
    target->second.accepted_macro_step = input->second.accepted_macro_step;
    target->second.temporal_numerator = input->second.temporal_numerator;
    target->second.temporal_denominator = input->second.temporal_denominator;
    target->second.source_dt = input->second.source_dt;
    target->second.target_dt = input->second.target_dt;
    target->second.consumed = input->second.consumed;
  }
}

static void require_pending_history_remaps_preallocated_(
    const std::map<std::string, AmrProgramPendingHistoryRemap>& destination,
    const std::map<std::string, AmrProgramPendingHistoryRemap>& source) {
  if (destination.size() != source.size())
    throw std::logic_error("AMR Program pending history remap shape changed after prime");
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target)
    if (target->first != input->first || input->second.key != input->first ||
        target->second.key != target->first)
      throw std::logic_error("AMR Program pending history remap identity changed after prime");
}

static void copy_deferred_history_lag_scratches_preallocated_(
    std::map<std::string, field_type>& destination,
    const std::map<std::string, field_type>& source) {
  require_deferred_history_lag_scratches_preallocated_(destination, source);
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first)
      throw std::logic_error("AMR Program deferred history scratch identity changed after prime");
    copy_field_preallocated_(target->second, input->second);
  }
}

static void require_deferred_history_lag_scratches_preallocated_(
    const std::map<std::string, field_type>& destination,
    const std::map<std::string, field_type>& source) {
  if (destination.size() != source.size())
    throw std::logic_error("AMR Program deferred history scratch shape changed after prime");
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first)
      throw std::logic_error("AMR Program deferred history scratch identity changed after prime");
    require_field_preallocated_(target->second, input->second);
  }
}

static void copy_temporal_partition_preallocated_(
    CellTemporalPartitionAcceptedState& destination,
    const CellTemporalPartitionAcceptedState& source) {
  require_temporal_partition_preallocated_(destination, source);
  copy_string_preallocated_(destination.provider_identity, source.provider_identity);
  destination.synchronization_tick = source.synchronization_tick;
  destination.tick_denominator = source.tick_denominator;
  for (std::size_t index = 0; index < source.cells.size(); ++index) {
    if (destination.cells[index].level != source.cells[index].level ||
        destination.cells[index].cell != source.cells[index].cell ||
        destination.cells[index].rung != source.cells[index].rung)
      throw std::logic_error("AMR Program temporal partition identity changed after prime");
    destination.cells[index].accepted_tick = source.cells[index].accepted_tick;
  }
}

static void require_temporal_partition_preallocated_(
    const CellTemporalPartitionAcceptedState& destination,
    const CellTemporalPartitionAcceptedState& source) {
  if (destination.kind != source.kind || destination.topology_epoch != source.topology_epoch ||
      destination.cells.size() != source.cells.size() ||
      source.provider_identity.size() > destination.provider_identity.capacity())
    throw std::logic_error("AMR Program temporal partition shape changed after prime");
  for (std::size_t index = 0; index < source.cells.size(); ++index)
    if (destination.cells[index].level != source.cells[index].level ||
        destination.cells[index].cell != source.cells[index].cell ||
        destination.cells[index].rung != source.cells[index].rung)
      throw std::logic_error("AMR Program temporal partition identity changed after prime");
}

static void copy_cell_temporal_configuration_preallocated_(
    std::optional<CellTemporalConfiguration>& destination,
    const std::optional<CellTemporalConfiguration>& source) {
  require_cell_temporal_configuration_preallocated_(destination, source);
  if (!source)
    return;
  if (destination->level_rungs.size() != source->level_rungs.size() ||
      destination->routes.size() != source->routes.size() ||
      destination->level_cell_counts.size() != source->level_cell_counts.size())
    throw std::logic_error("AMR Program cell-temporal configuration shape changed after prime");
  copy_string_preallocated_(destination->clock, source->clock);
  copy_string_preallocated_(destination->exact_contract, source->exact_contract);
  destination->tick_denominator = source->tick_denominator;
  destination->rung = source->rung;
  destination->topology_epoch = source->topology_epoch;
  destination->materialization_generation = source->materialization_generation;
  std::copy(source->level_rungs.begin(), source->level_rungs.end(),
            destination->level_rungs.begin());
  std::copy(source->routes.begin(), source->routes.end(), destination->routes.begin());
  std::copy(source->level_cell_counts.begin(), source->level_cell_counts.end(),
            destination->level_cell_counts.begin());
}

static void require_cell_temporal_configuration_preallocated_(
    const std::optional<CellTemporalConfiguration>& destination,
    const std::optional<CellTemporalConfiguration>& source) {
  if (destination.has_value() != source.has_value())
    throw std::logic_error("AMR Program cell-temporal configuration changed after prime");
  if (!source)
    return;
  if (destination->level_rungs.size() != source->level_rungs.size() ||
      destination->routes.size() != source->routes.size() ||
      destination->level_cell_counts.size() != source->level_cell_counts.size() ||
      source->clock.size() > destination->clock.capacity() ||
      source->exact_contract.size() > destination->exact_contract.capacity())
    throw std::logic_error("AMR Program cell-temporal configuration shape changed after prime");
  if (!std::equal(destination->routes.begin(), destination->routes.end(), source->routes.begin()))
    throw std::logic_error("AMR Program cell-temporal route identity changed after prime");
}

static void copy_events_preallocated_(std::vector<AmrProgramSynchronizationEvent>& destination,
                                      std::vector<AmrProgramSynchronizationEvent>& slots,
                                      const std::vector<AmrProgramSynchronizationEvent>& source) {
  require_events_preallocated_(destination, slots, source);
  const std::size_t prior_size = destination.size();
  for (std::size_t index = source.size(); index < prior_size; ++index)
    destination[index].phase.swap(slots[index].phase);
  destination.resize(source.size());
  for (std::size_t index = prior_size; index < source.size(); ++index) {
    destination[index].phase.swap(slots[index].phase);
    destination[index].parent_level = slots[index].parent_level;
    destination[index].child_level = slots[index].child_level;
    destination[index].runtime_block = slots[index].runtime_block;
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    if (destination[index].parent_level != source[index].parent_level ||
        destination[index].child_level != source[index].child_level ||
        destination[index].runtime_block != source[index].runtime_block ||
        destination[index].phase != source[index].phase)
      throw std::logic_error("AMR Program synchronization event shape changed after prime");
    destination[index].clock = source[index].clock;
  }
}

static void require_events_preallocated_(
    const std::vector<AmrProgramSynchronizationEvent>& destination,
    const std::vector<AmrProgramSynchronizationEvent>& slots,
    const std::vector<AmrProgramSynchronizationEvent>& source) {
  if (source.size() > destination.capacity() || source.size() > slots.size())
    throw std::logic_error("AMR Program synchronization event capacity was not primed");
  for (std::size_t index = 0; index < source.size(); ++index) {
    const auto& expected = index < destination.size() ? destination[index] : slots[index];
    if (expected.parent_level != source[index].parent_level ||
        expected.child_level != source[index].child_level ||
        expected.runtime_block != source[index].runtime_block ||
        expected.phase != source[index].phase)
      throw std::logic_error("AMR Program synchronization event shape changed after prime");
  }
}

static void copy_face_flux_preallocated_(
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>&
        destination,
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>&
        slots,
    const std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>& source) {
  require_face_flux_preallocated_(destination, slots, source);
  for (int axis = 0; axis < Dim; ++axis) {
    auto& target_axis = destination[static_cast<std::size_t>(axis)];
    auto& slot_axis = slots[static_cast<std::size_t>(axis)];
    const auto& input_axis = source[static_cast<std::size_t>(axis)];
    const std::size_t prior_size = target_axis.size();
    for (std::size_t index = input_axis.size(); index < prior_size; ++index) {
      target_axis[index].key.owner.swap(slot_axis[index].key.owner);
      target_axis[index].key.state.swap(slot_axis[index].key.state);
      target_axis[index].key.stage.swap(slot_axis[index].key.stage);
      target_axis[index].payload.swap(slot_axis[index].payload);
    }
    target_axis.resize(input_axis.size());
    for (std::size_t index = prior_size; index < input_axis.size(); ++index) {
      auto& target = target_axis[index];
      auto& slot = slot_axis[index];
      target.key.owner.swap(slot.key.owner);
      target.key.state.swap(slot.key.state);
      target.key.stage.swap(slot.key.stage);
      target.payload.swap(slot.payload);
      target.key.levels = slot.key.levels;
      target.key.centering = slot.key.centering;
      target.key.axis = slot.key.axis;
      target.key.face = slot.key.face;
      target.key.coarse_face = slot.key.coarse_face;
      target.key.role = slot.key.role;
      target.key.contribution = slot.key.contribution;
    }
    for (std::size_t index = 0; index < input_axis.size(); ++index) {
      auto& target = target_axis[index];
      const auto& input = input_axis[index];
      if (target.key.owner != input.key.owner || target.key.state != input.key.state ||
          target.key.levels != input.key.levels || target.key.centering != input.key.centering ||
          target.key.axis != input.key.axis || target.key.face != input.key.face ||
          target.key.coarse_face != input.key.coarse_face || target.key.stage != input.key.stage ||
          target.key.role != input.key.role || target.key.contribution != input.key.contribution)
        throw std::logic_error("AMR Program accepted face-flux identity changed after prime");
      // ClockStamp is a fixed-size value authority, not a dynamically allocated clock name.
      // Its exact phase/time qualification is copied atomically with the face-flux key.
      target.key.clock = input.key.clock;
      target.key.attempt = input.key.attempt;
      target.key.role = input.key.role;
      target.key.contribution = input.key.contribution;
      target.measure.stage_weight = input.measure.stage_weight;
      target.measure.substep_begin = input.measure.substep_begin;
      target.measure.substep_end = input.measure.substep_end;
      target.measure.substep_duration = input.measure.substep_duration;
      target.measure.face_measure = input.measure.face_measure;
      if (input.payload.size() > target.payload.capacity())
        throw std::logic_error("AMR Program accepted face-flux payload capacity was not primed");
      target.payload.resize(input.payload.size());
      std::copy(input.payload.begin(), input.payload.end(), target.payload.begin());
    }
  }
}

static void require_face_flux_preallocated_(
    const std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>& destination,
    const std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>& slots,
    const std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>,
                     Dim>& source) {
  for (int axis = 0; axis < Dim; ++axis) {
    const auto& target_axis = destination[static_cast<std::size_t>(axis)];
    const auto& slot_axis = slots[static_cast<std::size_t>(axis)];
    const auto& input_axis = source[static_cast<std::size_t>(axis)];
    if (input_axis.size() > target_axis.capacity() || input_axis.size() > slot_axis.size())
      throw std::logic_error("AMR Program accepted face-flux capacity was not primed");
    for (std::size_t index = 0; index < input_axis.size(); ++index) {
      const auto& target = index < target_axis.size() ? target_axis[index] : slot_axis[index];
      const auto& input = input_axis[index];
      if (target.key.owner != input.key.owner || target.key.state != input.key.state ||
          target.key.levels != input.key.levels || target.key.centering != input.key.centering ||
          target.key.axis != input.key.axis || target.key.face != input.key.face ||
          target.key.coarse_face != input.key.coarse_face || target.key.stage != input.key.stage ||
          target.key.role != input.key.role || target.key.contribution != input.key.contribution ||
          input.payload.size() > target.payload.capacity())
        throw std::logic_error("AMR Program accepted face-flux identity changed after prime");
    }
  }
}

static void copy_field_preallocated_(field_type& destination, const field_type& source) {
  require_field_preallocated_(destination, source);
  for (std::size_t local = 0; local < source.local_size(); ++local)
    Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
}

static void require_field_preallocated_(const field_type& destination, const field_type& source) {
  if (destination.layout() != source.layout() ||
      destination.distribution() != source.distribution() ||
      destination.local_rank() != source.local_rank() || destination.ncomp() != source.ncomp() ||
      destination.ghosts() != source.ghosts() || destination.local_size() != source.local_size())
    throw std::logic_error("AMR Program deferred history field shape changed after prime");
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    if (destination.global_index(local) != source.global_index(local) ||
        destination.fab(local).box() != source.fab(local).box() ||
        destination.fab(local).grown_box() != source.fab(local).grown_box() ||
        destination.fab(local).size() != source.fab(local).size())
      throw std::logic_error("AMR Program deferred history field patch changed after prime");
  }
}
