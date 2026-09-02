// Cold AMR adapter preparation and accepted-context binding.
// This is intentionally a class-definition fragment included by amr_backend.hpp.

void bind_preparation_hot_path_workspace_() const {
  if (preparation_view_ == nullptr)
    return;
  const std::size_t blocks = preparation_view_->block_prototypes.size();
  const std::size_t levels = preparation_view_->level_geometries.size();
  hot_path_workspace_.bind(blocks, levels,
                           [this](std::size_t block, std::size_t level) -> const field_type& {
                             return preparation_view_->block_prototypes.at(block).at(level);
                           });
  hot_path_workspace_.bind_sum_reduction(
      blocks, levels, [this](std::size_t block, std::size_t level) -> const field_type& {
        return preparation_view_->block_prototypes.at(block).at(level);
      });
}

void bind_accepted_hot_path_workspace_() const {
  if (facade_ == nullptr || runtime_ == nullptr)
    return;
  const std::size_t blocks = static_cast<std::size_t>(n_blocks());
  const std::size_t levels = static_cast<std::size_t>(nlev());
  hot_path_workspace_.bind(blocks, levels,
                           [this](std::size_t block, std::size_t level) -> const field_type& {
                             return facade_->program_prepared_amr_block_state_(
                                 static_cast<int>(block), static_cast<int>(level));
                           });
  hot_path_workspace_.bind_sum_reduction(
      blocks, levels, [this](std::size_t block, std::size_t level) -> const field_type& {
        return facade_->program_prepared_amr_block_state_(static_cast<int>(block),
                                                          static_cast<int>(level));
      });
  hot_path_workspace_.bind_coupling_invocation(
      std::max(facade_->program_prepared_amr_program_flux_expression_budget_()
                   .interface_coupling_identity_character_bound,
               primary_clock_.size()));
}

/// The accepted checkpoint has a distinct rollback image in the facade.  Keep this adapter-owned
/// candidate arena equally large so refreshing POPSAND5 can serialize, agree, then publish
/// without touching either image until every rank has accepted the bytes.
void bind_accepted_checkpoint_candidate_buffer_() const {
  if (facade_ == nullptr || runtime_ == nullptr)
    throw std::logic_error("AMR Program checkpoint candidate requires an accepted facade");
  const std::size_t capacity = facade_->program_checkpoint_state_capacity_().first;
  if (accepted_checkpoint_candidate_bytes_.capacity() < capacity)
    accepted_checkpoint_candidate_bytes_.reserve(capacity);
  const std::size_t levels = runtime_->hierarchy().num_levels();
  if (accepted_checkpoint_level_clock_slots_.capacity() < levels)
    accepted_checkpoint_level_clock_slots_.reserve(levels);
  accepted_checkpoint_candidate_bytes_.clear();
  accepted_checkpoint_level_clock_slots_.clear();
  // Regrid/restart calls this cold bind boundary before the next accepted image is assembled.
  // A staging witness is topology-bound; return every logical string/payload to its resident
  // pool before invalidating it.  Merely clearing `primed` would leave the prior accepted
  // synchronization image resident and make the next cold qualification fail (or allocate).
  reset_accepted_state_staging_for_cold_prime_();
}

AmrStorageTopologyAdapter() = default;
explicit AmrStorageTopologyAdapter(const PreparedAmrTopologyView* preparation_view)
    : preparation_view_(preparation_view), preparation_mode_(true) {
  if (preparation_view_ == nullptr)
    throw std::invalid_argument("AMR preparation adapter has no topology view");
  preparation_view_->validate();
  runtime_ = preparation_view_->runtime;
  hierarchy_tensor_solver_registry_ = std::make_shared<hierarchy_tensor_registry_type>(
      *preparation_view_->hierarchy_tensor_registry);
  // Forward regrid/bootstrap providers retain a complete detached topology image but no accepted
  // AmrRuntime.  Their resource generation is supplied by that image and must not fall back to
  // a live facade.
  if (runtime_ != nullptr)
    synchronize_resource_generation_();
  bind_preparation_hot_path_workspace_();
}

/// Bind the finite ProgramResourcePlan slot space before a DSO prelude can request scratch.
/// Slots are deliberately dense indexes, never generated value ids.
void bind_prepared_scratch_slots(std::size_t count) const {
  if (preparation_view_ == nullptr)
    throw std::logic_error("AMR scratch slots can only bind on a detached preparation image");
  prepared_scratch_.clear();
  prepared_scratch_.resize(count);
  prepared_scratch_descriptors_.clear();
  prepared_scratch_descriptors_.resize(count);
  bind_preparation_hot_path_workspace_();
}

/// Cold-bind the copied v5 flux authority.  The host has already rejected malformed ABI rows;
/// repeat the compact cross-check here because this is the last boundary before the generated
/// Program receives services.  No table string or value-id map survives this conversion.
void bind_prepared_amr_flux_tables(
    const std::vector<ProgramInstallationTables::ResourcePlan>& resource_plan,
    const std::vector<ProgramInstallationTables::FluxBasisOccurrence>& basis_occurrences,
    const std::vector<ProgramInstallationTables::FaceFluxStage>& face_flux_stages) const {
  if (preparation_view_ == nullptr)
    throw std::logic_error("AMR flux tables can only bind on a detached preparation image");
  const std::size_t blocks = preparation_view_->program_block_map.size();
  if (blocks == 0 || preparation_view_->block_prototypes.size() != blocks)
    throw std::invalid_argument("AMR flux table bind has no exact Program block topology");
  PreparedFluxTableCarrier candidate;
  candidate.bases.reserve(basis_occurrences.size());
  candidate.terms.reserve(face_flux_stages.size());
  candidate.basis_slots_by_runtime_block.resize(blocks);
  candidate.term_slots_by_runtime_block.resize(blocks);
  candidate.next_basis_by_runtime_block.resize(blocks, 0);
  // Build the complete replacement image off to the side.  A malformed late row must not
  // clear the currently resident carrier: callers may report that rejection and retain the
  // previous prepared Program unchanged.
  std::vector<FluxBasis> candidate_basis_payloads(basis_occurrences.size());
  std::vector<std::uint8_t> candidate_basis_active(basis_occurrences.size(), std::uint8_t{0});
  for (std::size_t slot = 0; slot < basis_occurrences.size(); ++slot) {
    const auto& row = basis_occurrences[slot];
    if (row.basis_slot != slot || row.block < 0 || static_cast<std::size_t>(row.block) >= blocks ||
        row.expression_slot >= resource_plan.size())
      throw std::invalid_argument("AMR flux table bind has a non-dense basis occurrence");
    const auto& resource = resource_plan[row.expression_slot];
    if (resource.slot != row.expression_slot ||
        resource.value_id != static_cast<std::uint64_t>(row.rhs_identity) ||
        resource.owner != row.owner || resource.clock != row.clock || resource.level != row.level ||
        row.provider > 3 || row.stage_denominator <= 0 || row.stage_numerator < 0 ||
        row.stage_numerator > row.stage_denominator)
      throw std::invalid_argument("AMR flux table bind has mismatched basis provenance");
    const int runtime_block =
        preparation_view_->program_block_map[static_cast<std::size_t>(row.block)];
    if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= blocks)
      throw std::invalid_argument("AMR flux table bind has an invalid runtime block route");
    candidate.bases.push_back({row.basis_slot,
                               row.expression_slot,
                               static_cast<std::uint32_t>(runtime_block),
                               row.level,
                               row.rhs_identity,
                               static_cast<std::uint8_t>(row.provider),
                               resource.components,
                               {row.stage_numerator, row.stage_denominator}});
    candidate.basis_slots_by_runtime_block[static_cast<std::size_t>(runtime_block)].push_back(
        row.basis_slot);
  }
  for (std::size_t slot = 0; slot < face_flux_stages.size(); ++slot) {
    const auto& row = face_flux_stages[slot];
    if (row.slot != slot || row.basis_slot >= candidate.bases.size() ||
        row.expression_slot >= resource_plan.size() || row.dt_power != 1 ||
        row.coefficient_denominator <= 0 || row.coefficient_numerator == 0)
      throw std::invalid_argument("AMR flux table bind has an invalid final term");
    const auto& basis = basis_occurrences[row.basis_slot];
    const auto& resource = resource_plan[row.expression_slot];
    if (resource.slot != row.expression_slot || resource.owner != row.owner ||
        resource.clock != row.clock || (resource.level != -1 && resource.level != basis.level) ||
        row.owner != basis.owner || row.clock != basis.clock)
      throw std::invalid_argument("AMR flux table bind has mismatched final provenance");
    const std::size_t runtime_block = candidate.bases[row.basis_slot].runtime_block;
    const auto& declared_basis = candidate.bases[row.basis_slot];
    const std::string stage_identity =
        "pops.program-flux-expression.v1/provider/" +
        std::to_string(static_cast<unsigned int>(declared_basis.provider)) + "/rhs/" +
        std::to_string(declared_basis.rhs_identity) + "/basis/" +
        std::to_string(declared_basis.basis_slot) + "/expression/" +
        std::to_string(row.expression_slot) + "/dt-power/1/weight/" +
        std::to_string(row.coefficient_numerator) + "/" +
        std::to_string(row.coefficient_denominator) + "/stage/" +
        std::to_string(declared_basis.stage.numerator) + "/" +
        std::to_string(declared_basis.stage.denominator);
    candidate.terms.push_back({row.slot,
                               row.basis_slot,
                               row.expression_slot,
                               {row.coefficient_numerator, row.coefficient_denominator}});
    candidate.terms.back().stage_identity = stage_identity;
    candidate.term_slots_by_runtime_block[runtime_block].push_back(row.slot);
  }
  candidate.bound = true;
  using std::swap;
  swap(static_flux_tables_, candidate);
  swap(static_flux_basis_payloads_, candidate_basis_payloads);
  swap(static_flux_basis_active_, candidate_basis_active);
}

/// Report the cold-resident v5 carrier as explicit host resource families.  This is called only
/// after subcycling has expanded every topology route and prepared every ledger slot, while the
/// installation image is still detached.  Exact value rows remain valid: these host-only
/// families are recorded in ``prepared_layouts`` and included in the single exact global ceiling
/// during resource-plan sealing, never grown as a hidden post-publication arena.
[[nodiscard]] std::vector<ProgramInstallationTables::ResourcePrototype>
prepared_amr_flux_resident_resource_prototypes() const {
  using prototype_type = ProgramInstallationTables::ResourcePrototype;
  using kind_type = ProgramInstallationTables::ResourcePrototypeKind;
  if (!static_flux_tables_.bound)
    return {};

  const std::size_t resource_count = prepared_scratch_.size();
  std::vector<std::uint64_t> basis_bytes(resource_count, 0);
  std::vector<std::uint64_t> ledger_bytes(resource_count, 0);
  const auto checked_add = [](std::uint64_t& target, std::uint64_t value, std::string_view what) {
    if (value > std::numeric_limits<std::uint64_t>::max() - target)
      throw std::overflow_error(std::string("AMR Program resident ") + std::string(what) +
                                " footprint overflows uint64");
    target += value;
  };
  const auto require_slot = [&](std::uint32_t slot) {
    if (slot >= resource_count)
      throw std::logic_error("AMR Program resident footprint has an absent expression slot");
  };
  const auto vector_bytes = [](std::size_t capacity, std::size_t element_size,
                               std::string_view what) -> std::uint64_t {
    if (element_size != 0 && capacity > std::numeric_limits<std::uint64_t>::max() / element_size)
      throw std::overflow_error(std::string("AMR Program resident ") + std::string(what) +
                                " footprint overflows uint64");
    return static_cast<std::uint64_t>(capacity) * element_size;
  };
  const auto add_basis = [&](std::uint32_t slot, std::uint64_t bytes) {
    require_slot(slot);
    checked_add(basis_bytes[slot], bytes, "basis");
  };
  const auto add_ledger = [&](std::uint32_t slot, std::uint64_t bytes) {
    require_slot(slot);
    checked_add(ledger_bytes[slot], bytes, "ledger");
  };

  if (!static_flux_tables_.bases.empty()) {
    add_basis(static_flux_tables_.bases.front().expression_slot,
              vector_bytes(static_flux_tables_.bases.capacity(),
                           sizeof(typename PreparedFluxTableCarrier::Basis), "basis table"));
    add_basis(static_flux_tables_.bases.front().expression_slot,
              ::pops::amr::reflux::detail::external_string_storage_bytes(
                  static_flux_collective_contract_));
    add_basis(static_flux_tables_.bases.front().expression_slot,
              vector_bytes(static_flux_basis_payloads_.capacity(), sizeof(FluxBasis),
                           "basis payload table"));
    add_basis(static_flux_tables_.bases.front().expression_slot,
              vector_bytes(static_flux_basis_active_.capacity(), sizeof(std::uint8_t),
                           "basis activity table"));
    add_basis(static_flux_tables_.bases.front().expression_slot,
              vector_bytes(static_flux_tables_.basis_slots_by_runtime_block.capacity(),
                           sizeof(std::vector<std::uint32_t>), "basis block index"));
    add_basis(static_flux_tables_.bases.front().expression_slot,
              vector_bytes(static_flux_tables_.next_basis_by_runtime_block.capacity(),
                           sizeof(std::size_t), "basis cursor index"));
  }
  if (!static_flux_tables_.terms.empty()) {
    add_ledger(static_flux_tables_.terms.front().expression_slot,
               vector_bytes(static_flux_tables_.terms.capacity(),
                            sizeof(typename PreparedFluxTableCarrier::Term), "final term table"));
    add_ledger(static_flux_tables_.terms.front().expression_slot,
               vector_bytes(static_flux_tables_.term_slots_by_runtime_block.capacity(),
                            sizeof(std::vector<std::uint32_t>), "term block index"));
  } else if (!static_flux_tables_.bases.empty()) {
    // The empty term index itself is still resident in an authenticated all-cancel Program.
    add_basis(static_flux_tables_.bases.front().expression_slot,
              vector_bytes(static_flux_tables_.term_slots_by_runtime_block.capacity(),
                           sizeof(std::vector<std::uint32_t>), "empty term block index"));
  }

  for (const auto& basis : static_flux_tables_.bases) {
    require_slot(basis.expression_slot);
    add_basis(basis.expression_slot,
              vector_bytes(basis.face_routes.capacity(),
                           sizeof(typename PreparedFluxTableCarrier::Basis::FaceRoute),
                           "basis face-route table"));
    for (const auto& route : basis.face_routes)
      add_basis(basis.expression_slot,
                vector_bytes(route.faces.capacity(),
                             sizeof(typename PreparedFluxTableCarrier::Basis::Face),
                             "basis face-route faces"));
    if (basis.basis_slot >= static_flux_basis_payloads_.size())
      throw std::logic_error("AMR Program resident footprint basis payload is not dense");
    const FluxBasis& payload = static_flux_basis_payloads_[basis.basis_slot];
    add_basis(basis.expression_slot, vector_bytes(payload.faces.capacity(), sizeof(FluxBasisFace),
                                                  "basis face payload table"));
    for (const FluxBasisFace& face : payload.faces)
      add_basis(basis.expression_slot, vector_bytes(face.flux_density.capacity(), sizeof(Real),
                                                    "basis face density payload"));
  }
  for (std::size_t runtime_block = 0;
       runtime_block < static_flux_tables_.basis_slots_by_runtime_block.size(); ++runtime_block) {
    const auto& slots = static_flux_tables_.basis_slots_by_runtime_block[runtime_block];
    if (!slots.empty()) {
      const auto slot = slots.front();
      if (slot >= static_flux_tables_.bases.size())
        throw std::logic_error("AMR Program resident basis block index is not dense");
      add_basis(static_flux_tables_.bases[slot].expression_slot,
                vector_bytes(slots.capacity(), sizeof(std::uint32_t), "basis block slots"));
    }
  }
  std::vector<std::pair<multiblock_flux_ledger_type*, std::uint32_t>> ledgers;
  ledgers.reserve(static_flux_tables_.terms.size());
  for (const auto& term : static_flux_tables_.terms) {
    require_slot(term.expression_slot);
    add_ledger(term.expression_slot,
               ::pops::amr::reflux::detail::external_string_storage_bytes(term.stage_identity));
    add_ledger(term.expression_slot,
               vector_bytes(term.ledger_routes.capacity(),
                            sizeof(typename PreparedFluxTableCarrier::LedgerRoute),
                            "term ledger-route table"));
    for (const auto& route : term.ledger_routes) {
      if (route.ledger == nullptr)
        throw std::logic_error("AMR Program resident ledger route has no ledger");
      add_ledger(term.expression_slot, vector_bytes(route.slots.capacity(), sizeof(std::uint32_t),
                                                    "term ledger slot index"));
      const auto found = std::find_if(ledgers.begin(), ledgers.end(), [&](const auto& candidate) {
        return candidate.first == route.ledger;
      });
      if (found == ledgers.end())
        ledgers.push_back({route.ledger, term.expression_slot});
      else
        found->second = std::min(found->second, term.expression_slot);
    }
  }
  // Every candidate ledger is cold-bound, even for an authenticated all-cancel basis with no
  // final term route.  Terms select their natural expression receipt when present; otherwise
  // attach the empty resident ledger image to the deterministic first basis receipt.
  if (multiblock_subcycling_ && !static_flux_tables_.bases.empty()) {
    const std::uint32_t cancellation_slot = static_flux_tables_.bases.front().expression_slot;
    multiblock_subcycling_->bind_candidate_ledger_slots([&](std::size_t, std::size_t, std::size_t,
                                                            multiblock_flux_ledger_type& ledger) {
      const auto found = std::find_if(ledgers.begin(), ledgers.end(), [&](const auto& candidate) {
        return candidate.first == &ledger;
      });
      if (found == ledgers.end())
        ledgers.push_back({&ledger, cancellation_slot});
    });
  }
  for (const auto& [ledger, expression_slot] : ledgers)
    add_ledger(expression_slot, ledger->resident_storage_bytes());
  for (const auto& slots : static_flux_tables_.term_slots_by_runtime_block) {
    if (!slots.empty()) {
      const auto slot = slots.front();
      if (slot >= static_flux_tables_.terms.size())
        throw std::logic_error("AMR Program resident term block index is not dense");
      add_ledger(static_flux_tables_.terms[slot].expression_slot,
                 vector_bytes(slots.capacity(), sizeof(std::uint32_t), "term block slots"));
    }
  }

  std::vector<prototype_type> result;
  result.reserve(resource_count * 2U);
  const auto append = [&](std::uint32_t slot, std::uint64_t bytes, std::int32_t subslot,
                          kind_type kind) {
    if (bytes == 0)
      return;
    result.push_back({slot, subslot, {bytes, 1, 1, 0, bytes, bytes}, kind});
  };
  for (std::size_t slot = 0; slot < resource_count; ++slot) {
    append(static_cast<std::uint32_t>(slot), basis_bytes[slot], 1, kind_type::flux_basis);
    append(static_cast<std::uint32_t>(slot), ledger_bytes[slot], 2, kind_type::flux_ledger);
  }
  return result;
}

/// Host-owned arenas are deliberately kept out of generated value rows.  Their slot is a
/// stable host namespace token, while kind/subslot identifies one independently auditable
/// family.  Flux carriers are collected by the dedicated flux method above and are excluded
/// here to prevent charging the same resident image twice.
[[nodiscard]] std::vector<ProgramInstallationTables::ResourcePrototype>
prepared_host_resident_resource_prototypes() const {
  using prototype = ProgramInstallationTables::ResourcePrototype;
  using kind = ProgramInstallationTables::ResourcePrototypeKind;
  if (preparation_view_ == nullptr || !hot_path_workspace_.bound)
    throw std::logic_error("AMR Program host resident footprint has no prepared workspace");
  const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("AMR Program host resident footprint overflows uint64");
    total += value;
  };
  const auto vector_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("AMR Program subcycling vector storage overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  };
  std::vector<prototype> result;
  result.reserve(10);
  const auto append_host_family = [&](std::uint64_t bytes, kind resource_kind) {
    if (bytes != 0)
      result.push_back({0, 0, {bytes, 1, 1, 0, bytes, bytes}, resource_kind});
  };
  if (preparation_view_->candidate_accepted_state_staging_capacity == nullptr)
    throw std::logic_error(
        "AMR Program host resident footprint has no configured forward storage envelope");
  const auto& forward_envelope = *preparation_view_->candidate_accepted_state_staging_capacity;
  const bool tensor_receipt_active = static_cast<bool>(hierarchy_tensor_selection_);
  const bool tensor_receipt_canonical =
      tensor_receipt_active
          ? forward_envelope.configured_tensor_provider_bytes != 0 &&
                !forward_envelope.configured_tensor_provider_request_contract.empty() &&
                !forward_envelope.configured_tensor_provider_limit_contract.empty()
          : forward_envelope.configured_tensor_provider_bytes == 0 &&
                forward_envelope.configured_tensor_provider_request_contract.empty() &&
                forward_envelope.configured_tensor_provider_limit_contract.empty();
  if (forward_envelope.configured_level_cell_bounds.size() != forward_envelope.level_count ||
      forward_envelope.configured_patch_bounds.size() != forward_envelope.level_count ||
      forward_envelope.configured_parent_child_pair_bounds.size() + 1U !=
          forward_envelope.level_count ||
      forward_envelope.configured_route_bounds.size() + 1U != forward_envelope.level_count ||
      forward_envelope.configured_event_bounds.size() + 1U != forward_envelope.level_count ||
      forward_envelope.configured_hierarchy_contract_characters_by_level.size() !=
          forward_envelope.level_count ||
      forward_envelope.configured_coupling_workspace_bytes_by_level.size() !=
          forward_envelope.level_count ||
      forward_envelope.configured_forward_storage_counts.level_cell_bounds !=
          forward_envelope.configured_level_cell_bounds ||
      forward_envelope.configured_forward_storage_counts.multifab_value_counts.size() !=
          runtime::program::PreparedAmrForwardStorageCounts::multifab_family_count ||
      forward_envelope.configured_live_subcycling_bytes == 0 ||
      forward_envelope.configured_forward_snapshot_bytes == 0 ||
      forward_envelope.configured_rank_bound == 0 ||
      forward_envelope.configured_subcycling_storage_contract.empty() || !tensor_receipt_canonical)
    throw std::logic_error(
        "AMR Program host resident footprint has an incomplete configured forward envelope");
  const auto workspace_bytes = hot_path_workspace_.resident_storage_bytes();
  const auto reduction_bytes = hot_path_workspace_.sum_reduction.resident_storage_bytes();
  if (workspace_bytes != 0)
    result.push_back(
        {0, 0, {workspace_bytes, 1, 1, 0, workspace_bytes, workspace_bytes}, kind::hot_snapshot});
  if (reduction_bytes != 0)
    result.push_back(
        {0, 1, {reduction_bytes, 1, 1, 0, reduction_bytes, reduction_bytes}, kind::reduction});

  const auto prepared_coupling_bytes =
      preparation_view_->candidate_prepared_coupling_workspace_bytes;
  if (prepared_coupling_bytes != 0) {
    // `prepared_coupling` is the explicit ownership fold A + sum(F_i + I_i).  Its configured
    // maximum already includes the eventual accepted A=C(L), so do not add the install-time
    // one-level A image a second time.
    const std::uint64_t prepared_coupling_peak_bytes =
        forward_envelope.configured_forward_storage_bytes.prepared_coupling;
    if (prepared_coupling_bytes > prepared_coupling_peak_bytes)
      throw std::logic_error(
          "AMR Program prepared coupling image exceeds its configured ownership ceiling");
    result.push_back(
        {0,
         0,
         {prepared_coupling_bytes, 1, 1, 0, prepared_coupling_bytes, prepared_coupling_peak_bytes},
         kind::prepared_coupling});
  }

  std::uint64_t subcycling_bytes = 0;
  checked_add(subcycling_bytes, vector_bytes(active_attempt_states_));
  checked_add(subcycling_bytes, vector_bytes(active_staged_parents_));
  checked_add(subcycling_bytes, vector_bytes(active_incoming_flux_));
  checked_add(subcycling_bytes, vector_bytes(active_outgoing_flux_));
  checked_add(subcycling_bytes, vector_bytes(active_block_identities_));
  checked_add(subcycling_bytes, vector_bytes(active_flux_basis_counts_));
  checked_add(subcycling_bytes, vector_bytes(prepared_rhs_basis_bounds_));
  checked_add(subcycling_bytes, vector_bytes(prepared_coefficient_term_bounds_));
  // The generic engine owns the program's candidate/history/average-down workspace.  Its
  // interface-ledger payload remains in the separately staged flux family.
  if (multiblock_subcycling_) {
    checked_add(subcycling_bytes, sizeof(*multiblock_subcycling_));
    checked_add(subcycling_bytes, multiblock_subcycling_->resident_storage_bytes_excluding_flux());
  }
  checked_add(subcycling_bytes, ::pops::amr::reflux::detail::external_string_storage_bytes(
                                    multiblock_subcycling_program_budget_contract_));
  if (subcycling_bytes != 0)
    result.push_back(
        {0,
         0,
         {subcycling_bytes, 1, 1, 0, subcycling_bytes,
          std::max(subcycling_bytes, forward_envelope.configured_live_subcycling_bytes)},
         kind::amr_subcycling});

  std::uint64_t cell_temporal_bytes = 0;
  checked_add(cell_temporal_bytes, vector_bytes(cell_temporal_resident_levels_));
  checked_add(cell_temporal_bytes, vector_bytes(cell_temporal_diagnostic_workspace_));
  checked_add(cell_temporal_bytes, vector_bytes(cell_temporal_diagnostics_));
  checked_add(cell_temporal_bytes, vector_bytes(cell_temporal_diagnostic_rollback_));
  for (const auto& resident : cell_temporal_resident_levels_) {
    if (!resident || !resident->runtime || !resident->executor)
      throw std::logic_error("AMR Program cell-temporal resident level is incomplete");
    checked_add(cell_temporal_bytes, sizeof(*resident));
    checked_add(cell_temporal_bytes, sizeof(*resident->runtime));
    checked_add(cell_temporal_bytes, resident->runtime->resident_storage_bytes());
    checked_add(cell_temporal_bytes, sizeof(*resident->executor));
    checked_add(cell_temporal_bytes, resident->executor->resident_storage_bytes());
  }
  const auto add_pool = [&](const auto& pool) {
    for (const auto& diagnostic : pool) {
      if (!diagnostic)
        throw std::logic_error("AMR Program cell-temporal diagnostic pool has a null entry");
      checked_add(cell_temporal_bytes, sizeof(*diagnostic));
      checked_add(cell_temporal_bytes, diagnostic->resident_storage_bytes());
    }
  };
  add_pool(cell_temporal_diagnostic_workspace_);
  add_pool(cell_temporal_diagnostics_);
  add_pool(cell_temporal_diagnostic_rollback_);
  if (cell_temporal_bytes != 0)
    result.push_back({0,
                      0,
                      {cell_temporal_bytes, 1, 1, 0, cell_temporal_bytes, cell_temporal_bytes},
                      kind::cell_temporal});

  // The generated rhs/state/scalar rows already own the MultiFab payload of these entries.
  // This host family deliberately records only the carrier vectors, descriptors and the
  // separately allocated field objects which make that finite generated storage reachable.
  std::uint64_t prepared_scratch_bytes = 0;
  checked_add(prepared_scratch_bytes, vector_bytes(prepared_scratch_));
  checked_add(prepared_scratch_bytes, vector_bytes(prepared_scratch_descriptors_));
  if (prepared_scratch_.size() != prepared_scratch_descriptors_.size())
    throw std::logic_error("AMR Program prepared scratch storage lost its descriptor slots");
  for (std::size_t slot = 0; slot < prepared_scratch_.size(); ++slot)
    for (std::size_t family = 0; family < prepared_scratch_[slot].size(); ++family) {
      const auto& fields = prepared_scratch_[slot][family];
      const auto& descriptors = prepared_scratch_descriptors_[slot][family];
      checked_add(prepared_scratch_bytes, vector_bytes(fields));
      checked_add(prepared_scratch_bytes, vector_bytes(descriptors));
      if (fields.size() != descriptors.size())
        throw std::logic_error("AMR Program prepared scratch family lost its descriptors");
      for (std::size_t subslot = 0; subslot < fields.size(); ++subslot) {
        const auto& entry = fields[subslot];
        if (static_cast<bool>(entry) != static_cast<bool>(descriptors[subslot]))
          throw std::logic_error("AMR Program prepared scratch entry lost its descriptor");
        if (entry) {
          checked_add(prepared_scratch_bytes, vector_bytes(*entry));
          for (const field_type& field : *entry) {
            const std::uint64_t storage = field.resident_storage_bytes();
            const std::uint64_t payload = field.resident_payload_bytes();
            if (storage < payload)
              throw std::logic_error(
                  "AMR Program prepared scratch storage is smaller than its payload");
            checked_add(prepared_scratch_bytes, storage - payload);
          }
        }
      }
    }
  // The topology-refresh map is not a second generated scratch authority: v5 hot accesses use
  // prepared_scratch_ exclusively.  It can nevertheless retain cold restore allocations until a
  // generation swap moves them into AcceptedContextSnapshot, so charge the live allocation once
  // here; the no-throw swap does not create a duplicate.
  if (scratches_.size() >
      std::numeric_limits<std::uint64_t>::max() / sizeof(typename decltype(scratches_)::value_type))
    throw std::overflow_error("AMR Program cold scratch map storage overflows uint64");
  checked_add(prepared_scratch_bytes, static_cast<std::uint64_t>(scratches_.size()) *
                                          sizeof(typename decltype(scratches_)::value_type));
  for (const auto& [key, scratch] : scratches_) {
    (void)key;
    checked_add(prepared_scratch_bytes, scratch.resident_storage_bytes());
  }
  append_host_family(prepared_scratch_bytes, kind::prepared_scratch);

  std::uint64_t coupled_jacvec_bytes = 0;
  {
    std::scoped_lock lock(coupled_jacvec_mutex_);
    if (coupled_jacvec_scratch_) {
      checked_add(coupled_jacvec_bytes, sizeof(*coupled_jacvec_scratch_));
      checked_add(coupled_jacvec_bytes, vector_bytes(coupled_jacvec_scratch_->levels));
      for (const CoupledJacvecLevelScratch& level : coupled_jacvec_scratch_->levels)
        for (const auto* field : {level.residual[0].get(), level.residual[1].get(),
                                  level.coupled[0].get(), level.coupled[1].get()}) {
          if (field == nullptr)
            throw std::logic_error("AMR Program coupled Jacobian scratch is incomplete");
          checked_add(coupled_jacvec_bytes, sizeof(*field));
          checked_add(coupled_jacvec_bytes, field->resident_storage_bytes());
        }
    }
  }
  append_host_family(coupled_jacvec_bytes, kind::coupled_jacvec);

  // The schedule owns its own primary identity; `primary_clock_` is the adapter's distinct
  // externally allocated execution identity and must therefore be charged separately.
  std::uint64_t clock_bytes = clock_schedule_.resident_storage_bytes();
  const auto string_is_external = [](const std::string& value) {
    const auto begin = reinterpret_cast<std::uintptr_t>(&value);
    const auto end = begin + sizeof(value);
    const auto data = reinterpret_cast<std::uintptr_t>(value.data());
    return data < begin || data >= end;
  };
  if (string_is_external(primary_clock_)) {
    if (primary_clock_.capacity() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program primary clock storage overflows uint64");
    checked_add(clock_bytes, static_cast<std::uint64_t>(primary_clock_.capacity()) + 1U);
  }
  if (clock_bytes != 0)
    result.push_back(
        {0, 0, {clock_bytes, 1, 1, 0, clock_bytes, clock_bytes}, kind::clock_schedule});

  const auto history_effects_bytes = history_effects_resident_storage_bytes_(true);
  append_host_family(history_effects_bytes, kind::history_effects);

  const auto accepted_snapshot_bytes = accepted_context_snapshot_resident_storage_bytes_();
  const auto forward_snapshot_bytes = [&] {
    std::uint64_t total = accepted_snapshot_bytes;
    // `amr_subcycling` is the accepted live A engine family.  `accepted_snapshot` is a
    // different rollback carrier: it retains snapshot(A) and must reserve one independently
    // prepared B bundle.  The following addition is therefore the intentional A+B peak, not a
    // double charge of the same engine object; the plan keeps the two resource kinds separate.
    checked_add(total, forward_envelope.configured_live_subcycling_bytes);
    return std::max(total, forward_envelope.configured_forward_snapshot_bytes);
  }();
  if (accepted_snapshot_bytes != 0)
    result.push_back(
        {0,
         0,
         {accepted_snapshot_bytes, 1, 1, 0, accepted_snapshot_bytes, forward_snapshot_bytes},
         kind::accepted_snapshot});

  // Provider-owned handles are not represented by generated value rows.  Charge their exact
  // retained storage as one host family: the vector distribution reports its erased Model, and
  // a hierarchy solver reports its sealed provider payload (including its optional
  // PreparedProvider callback).  The solver hook includes its concrete heap object so this
  // type-erased owner never guesses the size of a derived allocation.
  std::uint64_t provider_storage_bytes = 0;
  const auto add_exact_provider_storage = [&](const PreparedResidentStorage storage,
                                              const char* provider) {
    if (!storage.is_exact())
      throw std::invalid_argument(std::string("AMR Program provider '") + provider +
                                  "' has no exact resident-storage contract");
    checked_add(provider_storage_bytes, storage.bytes);
  };
  add_exact_provider_storage(vector_distribution_.resident_storage(), "vector-distribution");
  if (static_cast<bool>(hierarchy_tensor_solver_) != tensor_receipt_active)
    throw std::logic_error(
        "AMR Program hierarchy tensor selection and prepared provider disagree at bind");
  if (hierarchy_tensor_solver_) {
    // `prepare_hierarchy_tensor_solver_collectively()` seals this candidate before the DSO
    // prelude returns, so resident_storage() includes both publication images at this sole
    // plan-seal observation point.  A missing solver is the canonical inactive provider and
    // contributes exact zero.
    const PreparedResidentStorage tensor_storage = hierarchy_tensor_solver_->resident_storage();
    if (!tensor_storage.is_exact() || tensor_storage.bytes == 0 ||
        tensor_storage.bytes > forward_envelope.configured_tensor_provider_bytes)
      throw std::logic_error(
          "AMR Program hierarchy tensor provider exceeds its configured storage receipt");
    add_exact_provider_storage(tensor_storage, "hierarchy-tensor-solver");
  }
  append_host_family(provider_storage_bytes, kind::provider_storage);

  // Every prepared AMR Program retains this ledger, including the canonical inactive image for
  // a topology without an interface provider.  The topology view decides which authenticated
  // budget shape is legal; neither image may disappear from the global host ceiling.
  if (!interface_flux_ledger_ ||
      interface_flux_ledger_->topology_epoch() != preparation_view_->topology_epoch)
    throw std::logic_error("AMR Program interface-flux resident footprint has no prepared ledger");
  const auto& interface_budget = interface_flux_ledger_->budget();
  if (preparation_view_->candidate_interface_flux_ledger_budget == nullptr ||
      interface_budget != *preparation_view_->candidate_interface_flux_ledger_budget)
    throw std::logic_error(
        "AMR Program interface-flux resident footprint differs from its prepared budget");
  if (preparation_view_->has_interface_flux_provider) {
    if (interface_budget.max_fragments_per_window == 0 ||
        interface_budget.max_payload_terms_per_window == 0 ||
        interface_budget.max_identity_characters == 0 || interface_budget.exact_contract.empty())
      throw std::logic_error(
          "AMR Program interface-flux resident footprint has no authenticated active ledger");
  } else if (interface_budget.max_fragments_per_window != 0 ||
             interface_budget.max_payload_terms_per_window != 0 ||
             interface_budget.max_identity_characters != 0 ||
             interface_budget.max_transaction_depth != 1 ||
             interface_budget.exact_contract.empty()) {
    throw std::logic_error(
        "AMR Program interface-flux resident footprint has a non-canonical inactive ledger");
  }
  // The ledger's method reports its dynamic arenas only.  The unique_ptr owns one separately
  // allocated ledger object, so include that inline carrier in this host-resident family.
  std::uint64_t interface_ledger_bytes = interface_flux_ledger_->resident_storage_bytes();
  checked_add(interface_ledger_bytes, sizeof(*interface_flux_ledger_));
  if (interface_ledger_bytes == 0)
    throw std::logic_error(
        "AMR Program interface-flux resident footprint has an empty prepared ledger");
  // Static expression ledgers use flux_ledger/slot/subslot = expression/2.  This separate
  // topology-owned ledger is a single host family and therefore reserves subslot 3.
  result.push_back(
      {0,
       3,
       {interface_ledger_bytes, 1, 1, 0, interface_ledger_bytes, interface_ledger_bytes},
       kind::flux_ledger});
  return result;
}

/// Adopt the complete DSO-prepared cell-temporal shape before the facade becomes observable.
/// The image has already validated and collectively sealed this pair, so this intentionally
/// performs only no-throw moves/swaps and never consults the accepted facade.
void adopt_prepared_cell_temporal_execution(PreparedCellTemporalExecution<Dim> staged) noexcept {
  static_assert(std::is_nothrow_move_constructible_v<CellTemporalConfiguration>);
  static_assert(std::is_nothrow_swappable_v<decltype(cell_temporal_configuration_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
  std::optional<CellTemporalConfiguration> configuration(std::move(staged.configuration));
  cell_temporal_configuration_.swap(configuration);
  std::swap(accepted_temporal_partition_, staged.partition);
  cell_temporal_diagnostic_workspace_.swap(staged.diagnostic_workspace);
  cell_temporal_diagnostics_.swap(staged.accepted_diagnostics);
  cell_temporal_diagnostic_rollback_.swap(staged.rollback_diagnostics);
  rebind_cell_temporal_resident_levels_noexcept_();
}

/// Transaction hooks for the finite cell-local diagnostic arena.  They intentionally operate
/// only on already-primed value buffers; an installation or a regrid must build a fresh image
/// instead of growing one of these pools during candidate execution.
void snapshot_cell_temporal_diagnostics_noexcept() const noexcept {
  copy_cell_temporal_diagnostics_noexcept_(cell_temporal_diagnostic_rollback_,
                                           cell_temporal_diagnostics_);
}
void publish_cell_temporal_diagnostics_noexcept() const noexcept {
  copy_cell_temporal_diagnostics_noexcept_(cell_temporal_diagnostics_,
                                           cell_temporal_diagnostic_workspace_);
}
void restore_cell_temporal_diagnostics_noexcept() const noexcept {
  copy_cell_temporal_diagnostics_noexcept_(cell_temporal_diagnostics_,
                                           cell_temporal_diagnostic_rollback_);
}

/// Called by the host only after every rank has accepted the complete image.  The retained DSO
/// provider stops borrowing the detached candidate view before the owner is made reachable.
void bind_accepted_facade(facade_type* facade,
                          accepted_runtime_state_resolver_type runtime_state_resolver) {
  if (preparation_view_ == nullptr)
    throw std::logic_error("AMR accepted binding requires its detached preparation topology");
  if (preparation_view_->forward_detached)
    throw std::logic_error("AMR forward-detached preparation cannot activate an accepted facade");
  if (runtime_state_resolver == nullptr)
    throw std::invalid_argument("AMR accepted binding requires a runtime-state resolver");
  preparation_view_->validate();
  // Activation occurs before the aggregate owner swap. Retain the image-owned tensor registry:
  // it includes DSO-defined providers which must remain owner-last and must never leak into the
  // global facade registry. Consulting the facade here would still observe the previous accepted
  // hierarchy during a cold installation.
  runtime_ = preparation_view_->runtime;
  if (!hierarchy_tensor_solver_registry_)
    throw std::logic_error("AMR accepted binding lost its image-owned tensor registry");
  facade_ = require_facade_(facade);
  accepted_runtime_state_resolver_ = runtime_state_resolver;
  accepted_runtime_state_ =
      std::addressof(accepted_runtime_state_resolver_(facade_));
  // The candidate may have replaced its hot workspace while preparing subcycling, forward
  // scratch, or a topology-bound execution bundle after the clock was first adopted.  Rebind the
  // finite point storage at this last cold activation boundary so direct RHS/solve operations can
  // never discover an empty clock (or allocate one) in the accepted hot path.
  if (!primary_clock_.empty())
    hot_path_workspace_.bind_boundary_point_clock(primary_clock_);
  {
    const ExecutionLane& lane = prepared_execution_lane();
    // Activation precedes the aggregate owner swap.  Reading the facade map here would observe
    // the previous accepted Program (or an empty map on first installation) and leave every hot
    // projection/CFL route unbound.  The detached topology image is already collectively
    // validated and is the sole candidate authority at this point.
    const auto& map = preparation_view_->program_block_map;
    const std::size_t runtime_block_count = preparation_view_->block_prototypes.size();
    std::vector<int> candidate;
    candidate.reserve(map.size());
    for (const int runtime_block : map) {
      if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= runtime_block_count)
        throw std::logic_error("AMR projection/speed route targets no runtime block");
      candidate.push_back(runtime_block);
    }
    ExactContractBuilder receipt;
    receipt.text("pops.amr.program.projection-speed-routes")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(lane.identity())
        .scalar(static_cast<std::uint64_t>(candidate.size()));
    for (const int runtime_block : candidate)
      receipt.scalar(std::int32_t{runtime_block});
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("amr-program-projection-speed-routes"), receipt.view()}}, lane))
      throw std::runtime_error("AMR projection/speed block routes differ across MPI ranks");
    projection_speed_routes_.swap(candidate);
    projection_speed_routes_bound_ = true;
  }
  preparation_view_ = nullptr;
  preparation_mode_ = false;
  // Resource carriers, coupled scratch and the hot-path workspace were all built from the
  // detached topology in the adapter constructor.  Activation still precedes the aggregate
  // owner-last publication, so neither a history-free artifact nor a replacement may rescan the
  // old facade here.  The retained prepared image already has the exact candidate shape and
  // becomes the accepted carrier when the host performs its no-throw publication exchange.
}

/// Complete the accepted bind after the host has exchanged the prepared ProgramRuntimeState into
/// its stable Impl-owned storage.  This is pointer/ordinal rebinding only: every vector, map and
/// string capacity was sealed before the aggregate writer-held publication.
void rebind_accepted_runtime_state_after_publish() const noexcept {
  if (accepted_runtime_state_ == nullptr || preparation_view_ != nullptr)
    std::terminate();
  rebind_history_mutation_workspace_preallocated_after_restore_();
}
// A resident level runtime retains a raw pointer to this adapter. Its lifetime is therefore
// bounded by this one owner: copying or moving would leave that pointer stale or would share
// mutable cell-temporal authority between service images. Prepared and accepted images retain
// the public service through shared ownership; forward/restart images use the explicit detached
// snapshot protocol below, never a value copy.
AmrStorageTopologyAdapter(const AmrStorageTopologyAdapter&) = delete;
AmrStorageTopologyAdapter& operator=(const AmrStorageTopologyAdapter&) = delete;
AmrStorageTopologyAdapter(AmrStorageTopologyAdapter&&) = delete;
AmrStorageTopologyAdapter& operator=(AmrStorageTopologyAdapter&&) = delete;

[[nodiscard]] ProgramHostDescriptor program_host_descriptor() const {
  if (facade_ == nullptr)
    throw std::logic_error(
        "AMR Program host descriptor is unavailable from a detached preparation image");
  return const_cast<facade_type*>(facade_)->program_host_descriptor();
}

// During DSO preparation the facade redirects this declaration to its non-owning candidate
// graph.  The DSO never receives a registry pointer or an extra native callback.
void stage_auxiliary_consumer_plan(runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) const {
  if (facade_ == nullptr)
    throw std::logic_error("AMR auxiliary consumer plan requires one execution facade");
  facade_->install_auxiliary_consumer_plan(std::move(plan));
}

class ForwardScratchTopology final : public PreparedForwardAmrScratchTopology {
 public:
  explicit ForwardScratchTopology(std::vector<std::vector<const field_type*>> values)
      : values_(std::move(values)) {}
  [[nodiscard]] const std::vector<std::vector<const field_type*>>& values() const noexcept {
    return values_;
  }

 private:
  std::vector<std::vector<const field_type*>> values_;
};

/// Dense, adapter-owned route from one canonical checkpoint slot to the resident face ledger.
/// Forward accepted snapshots carry this trivially destructible image only to preserve its
/// bind-sealed vector capacity; every pointer is overwritten against the newly published bundle
/// before the first hot read.
struct AcceptedFaceFluxOrdinal {
  const multiblock_flux_ledger_type* ledger = nullptr;
  std::size_t source_slot = 0;
  std::size_t staging_slot = 0;
};

// Class-scope responsibility fragments preserve the public nested-type identities and member
// layout of AmrStorageTopologyAdapter while making each semantic authority independently auditable.
#include <pops/runtime/program/detail/program_execution_services_amr_spatial.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_public.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_public.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_spatial_operations.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_public.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_solver.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_private.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_polynomial.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_cell_temporal_level_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_definitions.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_services.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_cell_temporal_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_subcycling_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_basis.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_flux_expression_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_runtime.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_field_runtime_services.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_services.hpp>
#include <pops/runtime/program/detail/program_execution_services_amr_spatial_operations_services.hpp>

public:
using cell_temporal_provider_type =
    PreparedSameLevelTransportEulerPackStageFluxProvider<Dim, CellTemporalLevelRuntime>;
using cell_temporal_executor_type =
    PreparedBatchedCellTemporalExecutor<cell_temporal_provider_type>;

/// Immutable provider/executor pair for one exact AMR level.  All nested field images, route
/// records, partitions and device tables are constructed from the detached candidate topology;
/// the step path only rebinds the fixed candidate field pointers.
struct CellTemporalResidentLevel final {
  std::unique_ptr<CellTemporalLevelRuntime> runtime;
  std::unique_ptr<cell_temporal_executor_type> executor;
  std::size_t diagnostic_slot = 0;
};

void rebind_cell_temporal_resident_levels_noexcept_() const noexcept {
  if (!cell_temporal_configuration_ ||
      cell_temporal_resident_levels_.size() !=
          static_cast<std::size_t>(cell_temporal_configuration_->level_rungs.size()))
    std::terminate();
  for (const auto& resident : cell_temporal_resident_levels_) {
    if (!resident || !resident->runtime || !resident->executor)
      std::terminate();
    resident->runtime->rebind_configuration_noexcept(*cell_temporal_configuration_);
  }
}

/// Build the AMR accepted-service image for a forward topology without exposing the concrete
/// snapshot implementation to the system transaction carrier.  The returned image is detached
/// and therefore cannot restore or observe a facade until HiddenPublish rebinds it below.
[[nodiscard]] static std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>
detach_accepted_context_for_forward(const AcceptedProgramExecutionServicesSnapshot& accepted,
                                    std::uint64_t topology_epoch,
                                    std::uint64_t materialization_generation,
                                    AmrStorageTopologyAdapter*& rebind_owner) {
  void* opaque_rebind_owner = nullptr;
  auto detached =
      accepted.detach_for_forward(topology_epoch, materialization_generation, opaque_rebind_owner);
  rebind_owner = static_cast<AmrStorageTopologyAdapter*>(opaque_rebind_owner);
  if (!detached || rebind_owner == nullptr)
    throw std::logic_error("AMR Program forward regrid has no AMR accepted-service image");
  return detached;
}

/// Rebuild only declared transient scratch families against candidate field prototypes. The
/// detached accepted image owns the replacement until HiddenPublish and cannot consult a
/// facade or live hierarchy.
static void prepare_forward_scratch_rematerialization(
    AcceptedProgramExecutionServicesSnapshot& accepted,
    std::vector<std::vector<const field_type*>> forward_prototypes) {
  ForwardScratchTopology topology(std::move(forward_prototypes));
  accepted.prepare_forward_scratch_rematerialization(topology);
}

/// Complete the detached image only after the forward hierarchy has become the live authority.
/// This is pointer rebinding only; all state and capacity were prepared in Candidate.
static void rebind_detached_accepted_context_after_publish(
    AcceptedProgramExecutionServicesSnapshot& accepted, AmrStorageTopologyAdapter& owner) noexcept {
  accepted.rebind_after_forward_publish(static_cast<void*>(&owner));
}

/// Cold-bind the finite interface-flux snapshot storage exactly once.  Transaction refresh and
/// finalization intentionally have no access to this operation.
static void prime_accepted_context_at_bind(AcceptedProgramExecutionServicesSnapshot& accepted) {
  accepted.prime_at_bind();
}

static void prime_copied_accepted_context_at_bind(
    AcceptedProgramExecutionServicesSnapshot& accepted) {
  accepted.prime_copied_image_at_bind();
}

template <int TestDim>
friend struct detail::AmrProgramHistoryRemapCollectiveTestAccess;
template <int TestDim>
friend class ::pops::runtime::program::ProgramExecutionServices;
