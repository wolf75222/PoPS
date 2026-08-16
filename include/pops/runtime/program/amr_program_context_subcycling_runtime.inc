void clear_active_flux_expression_(const field_type& field) const noexcept {
  if (!active_attempt_states_.empty())
    active_flux_expressions_.erase(&field);
}

template <class Body>
void advance_prepared_hierarchy_(double dt, Body&& body, std::string_view operation) const {
  if (!std::isfinite(dt) || !(dt > 0.0))
    throw std::invalid_argument("AMR Program step requires a finite positive dt");
  const int prior_level = active_level_;
  const double prior_dt = current_dt_;
  const double prior_interval_start = current_interval_start_time_;
  const ::pops::amr::Rational prior_interval_begin = current_interval_begin_phase_;
  const ::pops::amr::Rational prior_interval_end = current_interval_end_phase_;
  const int prior_substep = logical_substep_;
  const ::pops::amr::Rational prior_stage = stage_time_;
  refresh_resources_();
  require_facade_execution_();
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR Program hierarchy advance cannot nest another attempt");
  // The prior candidate is only discarded when a later accepted transaction has already
  // captured its own rollback image.  During the transaction that published it, this guard kept
  // the pre-commit ledger alive even after the live ledger crossed its noexcept swap boundary.
  interface_flux_commit_guard_.reset();
  try {
    begin_step(dt);
    prepare_multiblock_subcycling_engine_();
    const ::pops::amr::ClockWindow root{
        {0, static_cast<std::int64_t>(facade_->macro_step()), {0, 1}, facade_->time()},
        {0, static_cast<std::int64_t>(facade_->macro_step()), {1, 1}, facade_->time() + dt}};
    const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
    std::optional<typename interface_flux_ledger_type::PreparedBegin> prepared_begin;
    std::exception_ptr ledger_error;
    try {
      prepared_begin.emplace(interface_flux_ledger_->prepare_begin());
    } catch (...) {
      ledger_error = std::current_exception();
    }
    if (all_reduce_max(ledger_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && ledger_error)
        std::rethrow_exception(ledger_error);
      throw std::runtime_error(
          "AMR Program interface-ledger begin preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("amr-program-interface-ledger-begin"),
              prepared_begin->exact_contract()}},
            lane))
      throw std::runtime_error(
          "AMR Program interface-ledger begin contract differs between MPI ranks");
    interface_flux_ledger_->publish_prepared_begin(*prepared_begin);
    multiblock_subcycling_->advance(
        root,
        [&](multiblock_level_group_type group) { advance_multiblock_level_group_(group, body); },
        [&](multiblock_reflux_context_type& reflux) { reconcile_multiblock_reflux_(reflux); },
        [&](std::size_t runtime_block, std::size_t level, const field_type& candidate) {
          facade_->validate_prepared_amr_state_publication_candidate(
              static_cast<int>(runtime_block), static_cast<int>(level), candidate);
        },
        [&](std::size_t level, std::span<field_type*> pack) {
          stage_prepared_publication_candidates_(level, pack);
        });
    std::optional<typename interface_flux_ledger_type::PreparedCommit> prepared_commit;
    ledger_error = nullptr;
    try {
      prepared_commit.emplace(interface_flux_ledger_->prepare_commit());
    } catch (...) {
      ledger_error = std::current_exception();
    }
    if (all_reduce_max(ledger_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && ledger_error)
        std::rethrow_exception(ledger_error);
      throw std::runtime_error(
          "AMR Program interface-ledger commit preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("amr-program-interface-ledger-commit"),
              prepared_commit->exact_contract()}},
            lane))
      throw std::runtime_error(
          "AMR Program interface-ledger commit contract differs between MPI ranks");
    static_assert(
        std::is_nothrow_move_constructible_v<typename interface_flux_ledger_type::PreparedCommit>);
    static_assert(noexcept(interface_flux_commit_guard_.swap(prepared_commit)));
    interface_flux_commit_guard_.swap(prepared_commit);
    interface_flux_ledger_->publish_prepared_commit(*interface_flux_commit_guard_);
  } catch (...) {
    if (interface_flux_ledger_ && interface_flux_ledger_->in_transaction())
      interface_flux_ledger_->rollback();
    active_level_ = prior_level;
    current_dt_ = prior_dt;
    current_interval_start_time_ = prior_interval_start;
    current_interval_begin_phase_ = prior_interval_begin;
    current_interval_end_phase_ = prior_interval_end;
    logical_substep_ = prior_substep;
    stage_time_ = prior_stage;
    clear_active_multiblock_group_();
    throw;
  }
  active_level_ = prior_level;
  current_dt_ = prior_dt;
  current_interval_start_time_ = prior_interval_start;
  current_interval_begin_phase_ = prior_interval_begin;
  current_interval_end_phase_ = prior_interval_end;
  logical_substep_ = prior_substep;
  stage_time_ = prior_stage;
  (void)operation;
}

static std::size_t checked_product_(std::size_t left, std::size_t right,
                                    std::string_view operation) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::length_error(std::string(operation) + " exceeds size_t");
  return left * right;
}

static ::pops::amr::InterfaceFluxLedgerBudget inactive_interface_flux_budget_() {
  return {0, 0, 1, "pops.amr-program.interface-flux-budget/inactive"};
}

void prepare_multiblock_subcycling_engine_() const {
  auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  std::vector<std::size_t> rhs_basis_bounds;
  std::vector<std::size_t> coefficient_term_bounds;
  std::string program_budget_contract;
  std::exception_ptr local_error;
  try {
    const flux_expression_budget_type& expression_budget =
        facade_->prepared_amr_program_flux_expression_budget();
    const auto& prepared_map = facade_->prepared_amr_program_block_map();
    const auto& program_map = facade_->program_block_map();
    const std::size_t blocks = hierarchy.block_count();
    if (expression_budget.program_hash.empty() ||
        expression_budget.program_hash != facade_->installed_program_hash() ||
        expression_budget.generation != runtime_->materialization_generation() ||
        expression_budget.exact_contract.empty() ||
        expression_budget.program_block_map.canonical_indices != prepared_map.canonical_indices ||
        expression_budget.program_block_map.hierarchy_contract != prepared_map.hierarchy_contract ||
        expression_budget.program_block_map.exact_contract != prepared_map.exact_contract ||
        expression_budget.blocks.size() != blocks || program_map.size() != blocks ||
        prepared_map.canonical_indices.size() != blocks)
      throw std::logic_error(
          "AMR Program flux-expression budget is not authentic for the active carrier");

    rhs_basis_bounds.assign(blocks, 0);
    coefficient_term_bounds.assign(blocks, 0);
    std::vector<bool> seen(blocks, false);
    for (std::size_t program_block = 0; program_block < blocks; ++program_block) {
      const int mapped = program_map[program_block];
      const std::size_t canonical = prepared_map.canonical_indices[program_block];
      if (mapped < 0 || static_cast<std::size_t>(mapped) != canonical || canonical >= blocks ||
          seen[canonical])
        throw std::logic_error(
            "AMR Program flux-expression budget differs from its exact block permutation");
      seen[canonical] = true;
      const auto& block_budget = expression_budget.blocks[program_block];
      const bool active =
          block_budget.rhs_basis_bound != 0 || block_budget.coefficient_term_bound != 0;
      if (active && (block_budget.rhs_basis_bound == 0 || block_budget.coefficient_term_bound == 0))
        throw std::logic_error("AMR Program flux-expression budget has a partial per-block bound");
      rhs_basis_bounds[canonical] = block_budget.rhs_basis_bound;
      coefficient_term_bounds[canonical] = block_budget.coefficient_term_bound;
    }
    program_budget_contract = expression_budget.exact_contract;
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = hierarchy.lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (hierarchy.lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program flux-expression budget validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-flux-expression-budget"),
            std::string_view(program_budget_contract)}},
          lane))
    throw std::invalid_argument(
        "AMR Program flux-expression budget differs between prepared-lane ranks");

  std::vector<::pops::amr::ParentChildClockRelation> relations;
  ::pops::amr::InterfaceFluxLedgerBudget interface_budget;
  std::optional<typename interface_flux_ledger_type::PreparedBudget> prepared_interface_budget;
  local_error = nullptr;
  try {
    relations = facade_->prepared_program_temporal_relations();
    if (relations.size() + 1 != hierarchy.level_count())
      throw std::logic_error(
          "AMR Program subcycling lacks one temporal relation per live transition");
    interface_budget = facade_->prepared_amr_interface_flux_ledger_budget();
    prepared_interface_budget.emplace(interface_flux_ledger_->prepare_budget(interface_budget));
    ExactContractBuilder complete;
    complete.text("pops.amr-program.complete-flux-budget")
        .scalar(std::uint32_t{1})
        .bytes(program_budget_contract)
        .bytes(interface_budget.exact_contract);
    program_budget_contract = std::move(complete).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program interface budget preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-interface-ledger-budget"),
            std::string_view(interface_budget.exact_contract)}},
          lane))
    throw std::invalid_argument(
        "AMR Program interface ledger budgets differ between prepared-lane ranks");
  interface_flux_ledger_->publish_prepared_budget(*prepared_interface_budget);

  if (multiblock_subcycling_ != nullptr &&
      multiblock_subcycling_epoch_ == hierarchy.topology_runtime().topology_epoch() &&
      multiblock_subcycling_generation_ ==
          hierarchy.topology_runtime().materialization_generation() &&
      multiblock_subcycling_program_budget_contract_ == program_budget_contract)
    return;

  std::unique_ptr<multiblock_subcycling_type> prepared;
  local_error = nullptr;
  try {
    std::size_t maximum_patches = 1;
    for (std::size_t level = 0; level < hierarchy.level_count(); ++level)
      maximum_patches = std::max(
          maximum_patches, hierarchy.topology_runtime().hierarchy().layout(level).patches().size());
    const std::size_t overlap_pairs = maximum_patches < 2
                                          ? 1
                                          : checked_product_(maximum_patches, maximum_patches - 1,
                                                             "AMR Program patch-overlap budget") /
                                                2;

    // The ledger implementation requires a positive structural capacity even for an
    // authenticated source-only Program.  Flux-producing Programs derive every retained entry
    // from the exact per-block RHS-basis bound below; no evaluation-count fallback is used.
    std::size_t maximum_entries = 1;
    for (std::size_t parent = 0; parent < relations.size(); ++parent) {
      const auto ratio =
          hierarchy.topology_runtime().hierarchy().layout(parent + 1).ratio_from_parent();
      const auto temporal = relations[parent].temporal_ratio();
      const std::size_t substeps =
          static_cast<std::size_t>(temporal.numerator / temporal.denominator +
                                   (temporal.numerator % temporal.denominator == 0 ? 0 : 1));
      for (std::size_t block = 0; block < hierarchy.block_count(); ++block) {
        std::size_t block_entries = 0;
        for (const ProgramInterfaceFace& interface : program_interface_faces_(parent)) {
          std::size_t fine_faces = 1;
          for (int axis = 0; axis < Dim; ++axis)
            if (axis != interface.axis)
              fine_faces = checked_product_(fine_faces, static_cast<std::size_t>(ratio[axis]),
                                            "AMR Program fine-face budget");
          const std::size_t fine_evaluations =
              checked_product_(substeps, fine_faces, "AMR Program temporal fine-flux budget");
          if (fine_evaluations == std::numeric_limits<std::size_t>::max())
            throw std::length_error("AMR Program face-fragment budget exceeds size_t");
          const std::size_t fragments_per_basis = 1 + fine_evaluations;
          const std::size_t face_entries =
              checked_product_(rhs_basis_bounds[block], fragments_per_basis,
                               "AMR Program authenticated face-flux expression budget");
          if (block_entries > std::numeric_limits<std::size_t>::max() - face_entries)
            throw std::length_error("AMR Program face-flux ledger budget exceeds size_t");
          block_entries += face_entries;
        }
        maximum_entries = std::max(maximum_entries, block_entries);
      }
    }

    ::pops::numerics::time::amr::MultiBlockAmrSubcyclingBudget budget;
    budget.transitions = {relations.size(), {maximum_patches, overlap_pairs}};
    budget.flux_ledger = {maximum_entries, maximum_entries, 1};
    prepared = std::make_unique<multiblock_subcycling_type>(
        multiblock_subcycling_type::prepare(hierarchy, relations, budget));
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (hierarchy.lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program subcycling preparation failed collectively");
  }

  multiblock_subcycling_ = std::move(prepared);
  prepared_rhs_basis_bounds_ = std::move(rhs_basis_bounds);
  prepared_coefficient_term_bounds_ = std::move(coefficient_term_bounds);
  multiblock_subcycling_program_budget_contract_ = std::move(program_budget_contract);
  // This engine is prepared against the facade's rebuilt multi-block hierarchy.  Regrid and
  // restart rematerialize that authority before the raw runtime's convenience view necessarily
  // observes the new generation, so cache only against the exact authority used to size faces.
  multiblock_subcycling_epoch_ = hierarchy.topology_runtime().topology_epoch();
  multiblock_subcycling_generation_ = hierarchy.topology_runtime().materialization_generation();
}

template <class Body>
void advance_multiblock_level_group_(multiblock_level_group_type group, Body& body) const {
  const std::size_t blocks = facade_->prepared_amr_multiblock_hierarchy_().block_count();
  if (group.size() != blocks || group.empty())
    throw std::logic_error("AMR Program level group lost its complete block pack");

  if (prepared_rhs_basis_bounds_.size() != blocks ||
      prepared_coefficient_term_bounds_.size() != blocks)
    throw std::logic_error(
        "AMR Program level group lacks its authenticated flux-expression budget");
  active_attempt_states_.assign(blocks, nullptr);
  active_staged_parents_.assign(blocks, nullptr);
  active_incoming_flux_.assign(blocks, nullptr);
  active_outgoing_flux_.assign(blocks, nullptr);
  active_block_identities_.assign(blocks, {});
  active_flux_basis_counts_.assign(blocks, 0);
  active_flux_expressions_.clear();
  next_active_flux_basis_identity_ = 0;
  for (auto& current : group) {
    if (current.block >= blocks || current.level != group.front().level ||
        current.substep != group.front().substep || current.attempt != group.front().attempt ||
        current.window.begin != group.front().window.begin ||
        current.window.end != group.front().window.end ||
        active_attempt_states_[current.block] != nullptr)
      throw std::logic_error("AMR Program level group is not canonical and simultaneous");
    active_attempt_states_[current.block] = &current.candidate;
    active_staged_parents_[current.block] = current.staged_parent;
    active_incoming_flux_[current.block] = current.incoming_flux;
    active_outgoing_flux_[current.block] = current.outgoing_flux;
    active_block_identities_[current.block] = current.block_identity;
  }
  if (std::find(active_attempt_states_.begin(), active_attempt_states_.end(), nullptr) !=
      active_attempt_states_.end())
    throw std::logic_error("AMR Program level group omits a runtime block candidate");

  active_level_ = static_cast<int>(group.front().level);
  logical_substep_ = group.front().substep;
  active_subcycling_attempt_ = group.front().attempt;
  active_subcycling_window_ = group.front().window;
  current_dt_ = group.front().window.end.physical_time - group.front().window.begin.physical_time;
  current_interval_start_time_ = group.front().window.begin.physical_time;
  current_interval_begin_phase_ = group.front().window.begin.phase;
  current_interval_end_phase_ = group.front().window.end.phase;
  stage_time_ = {0, 1};
  facade_->clear_prepared_amr_level_evaluations();
  try {
    body(current_dt_);
    for (std::size_t block = 0; block < blocks; ++block)
      materialize_active_flux_expression_(block, *active_attempt_states_[block]);
  } catch (...) {
    clear_active_multiblock_group_();
    throw;
  }
  clear_active_multiblock_group_();
}

void stage_prepared_publication_candidates_(std::size_t level, std::span<field_type*> pack) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::size_t blocks = 0;
  std::vector<const field_type*> actives;
  std::exception_ptr lookup_error;
  try {
    blocks = static_cast<std::size_t>(n_blocks());
    if (pack.size() != blocks)
      throw std::invalid_argument(
          "AMR Program publication staging pack lost its complete block set");
    actives.assign(blocks, nullptr);
    for (std::size_t block = 0; block < blocks; ++block) {
      if (pack[block] == nullptr)
        throw std::invalid_argument("AMR Program publication staging pack contains a null state");
      actives[block] = facade_->prepared_amr_block_level_active_mask(static_cast<int>(block),
                                                                     static_cast<int>(level));
    }
  } catch (...) {
    lookup_error = std::current_exception();
  }
  if (all_reduce_max(lookup_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && lookup_error)
      std::rethrow_exception(lookup_error);
    throw std::runtime_error("AMR Program publication staging mask lookup failed collectively");
  }
  for (std::size_t block = 0; block < blocks; ++block) {
    const long masked = actives[block] != nullptr ? 1L : 0L;
    if (all_reduce_min(masked, lane) != all_reduce_max(masked, lane))
      throw std::runtime_error(
          "AMR Program publication staging mask classification differs between ranks");
  }

  std::vector<std::optional<field_type>> staged;
  std::exception_ptr stage_error;
  try {
    staged.resize(blocks);
    for (std::size_t block = 0; block < blocks; ++block) {
      const field_type* const active = actives[block];
      if (active == nullptr)
        continue;
      const int runtime_block = static_cast<int>(block);
      const int live_level = static_cast<int>(level);
      const field_type& accepted = facade_->prepared_amr_block_state(runtime_block, live_level);
      require_same_field_contract_(*pack[block], accepted,
                                   "AMR Program publication staging candidate");
      require_same_layout_(*active, accepted, "AMR Program publication staging active mask");
      if (active->ncomp() != 1)
        throw std::invalid_argument(
            "AMR Program publication staging active mask requires one component");
      staged[block].emplace(accepted);
      const int components = pack[block]->ncomp();
      for (std::size_t local = 0; local < pack[block]->local_size(); ++local) {
        const auto& source_fab = pack[block]->fab(local);
        auto& destination_fab = staged[block]->fab(local);
        const auto& active_fab = active->fab(local);
        if (pack[block]->global_index(local) != accepted.global_index(local) ||
            pack[block]->global_index(local) != active->global_index(local) ||
            source_fab.box() != destination_fab.box() || source_fab.box() != active_fab.box())
          throw std::invalid_argument(
              "AMR Program publication staging encountered different exact local patch ownership");
        const FieldView<const Real, Dim> source = std::as_const(source_fab).view();
        const FieldView<Real, Dim> destination = destination_fab.view();
        const FieldView<const Real, Dim> mask = std::as_const(active_fab).view();
        for_each_cell(source_fab.box(), [=] POPS_HD(const Index<Dim>& cell) {
          if (mask(cell, 0) < Real(0.5))
            return;
          for (int component = 0; component < components; ++component)
            destination(cell, component) = source(cell, component);
        });
      }
    }
  } catch (...) {
    stage_error = std::current_exception();
  }
  try {
    Kokkos::fence();
  } catch (...) {
    stage_error = std::current_exception();
  }
  if (stage_error)
    std::rethrow_exception(stage_error);
  for (std::size_t block = 0; block < blocks; ++block)
    if (staged[block])
      *pack[block] = std::move(*staged[block]);
}

void clear_active_multiblock_group_() const noexcept {
  active_attempt_states_.clear();
  active_staged_parents_.clear();
  active_incoming_flux_.clear();
  active_outgoing_flux_.clear();
  active_block_identities_.clear();
  active_flux_basis_counts_.clear();
  active_flux_expressions_.clear();
  next_active_flux_basis_identity_ = 0;
  active_subcycling_attempt_ = 0;
}

struct ProgramInterfaceFace {
  int axis = 0;
  Index<Dim> coarse_face{};
  Index<Dim> coarse_cell{};
  ::pops::amr::reflux::CoarseCellFaceSide side = ::pops::amr::reflux::CoarseCellFaceSide::Lower;
};

static std::array<int, Dim> index_key_(const Index<Dim>& index) {
  std::array<int, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[static_cast<std::size_t>(axis)] = index[axis];
  return result;
}

static std::vector<Index<Dim>> cells_in_box_(const Box<Dim>& box) {
  const std::size_t cells = static_cast<std::size_t>(box.numPts());
  std::vector<Index<Dim>> result;
  result.reserve(cells);
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    std::size_t remainder = ordinal;
    Index<Dim> cell{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t length = static_cast<std::size_t>(box.length(axis));
      cell[axis] = box.lo[axis] + static_cast<int>(remainder % length);
      remainder /= length;
    }
    result.push_back(cell);
  }
  return result;
}

std::vector<ProgramInterfaceFace> program_interface_faces_(std::size_t parent_level) const {
  const auto& parent = runtime_->hierarchy().layout(parent_level);
  const auto& child = runtime_->hierarchy().layout(parent_level + 1);
  const BoundaryTopology<Dim> topology = facade_->prepared_amr_boundary_topology();
  Extent<Dim> ratio{};
  for (int axis = 0; axis < Dim; ++axis)
    ratio[axis] = child.ratio_from_parent()[axis];
  std::set<std::array<int, Dim>> covered;
  for (const Box<Dim>& fine_patch : child.patches().boxes())
    for (const Index<Dim>& cell : cells_in_box_(pops::coarsen(fine_patch, ratio)))
      covered.insert(index_key_(cell));

  std::vector<ProgramInterfaceFace> result;
  for (const auto& coordinate : covered) {
    Index<Dim> inside{};
    for (int axis = 0; axis < Dim; ++axis)
      inside[axis] = coordinate[static_cast<std::size_t>(axis)];
    for (int axis = 0; axis < Dim; ++axis) {
      for (int direction : {-1, 1}) {
        Index<Dim> outside = inside;
        outside[axis] += direction;
        if (!parent.domain().contains(outside)) {
          const BoundarySide side = direction < 0 ? BoundarySide::lower : BoundarySide::upper;
          if (!topology.is_periodic(Face<Dim>{axis, side}))
            continue;
          outside[axis] = direction < 0 ? parent.domain().hi[axis] : parent.domain().lo[axis];
        }
        const bool parent_cell =
            std::any_of(parent.patches().boxes().begin(), parent.patches().boxes().end(),
                        [&](const Box<Dim>& patch) { return patch.contains(outside); });
        if (!parent_cell || covered.contains(index_key_(outside)))
          continue;
        Index<Dim> face = inside;
        if (direction > 0)
          ++face[axis];
        result.push_back({axis, face, outside,
                          direction > 0 ? ::pops::amr::reflux::CoarseCellFaceSide::Lower
                                        : ::pops::amr::reflux::CoarseCellFaceSide::Upper});
      }
    }
  }
  return result;
}

std::vector<Real> collective_face_payload_(const level_evaluation_type& evaluation,
                                           const field_type& field, int axis,
                                           const Index<Dim>& face) const {
  std::vector<Real> payload;
  std::exception_ptr local_error;
  try {
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR Program interface face has no level flux patch");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    payload.assign(static_cast<std::size_t>(field.ncomp()), Real(0));
    if (owner == field.local_rank()) {
      const std::size_t local = field.local_index_of(selected);
      if (local == field_type::not_local || local >= evaluation.integrated_face_fluxes.size())
        throw std::runtime_error("AMR Program interface flux storage lost its local patch");
      const FieldView<const Real, Dim> values =
          evaluation.integrated_face_fluxes[local].view().axes[axis];
      for (int component = 0; component < field.ncomp(); ++component) {
        Real value = Real(0);
        Kokkos::parallel_reduce(
            "pops_program_amr_interface_face", Kokkos::RangePolicy<>(0, 1),
            [=] POPS_HD(int, Real& sum) { sum += values(face, component); }, value);
        payload[static_cast<std::size_t>(component)] = value;
      }
      Kokkos::fence();
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program interface flux failed on another MPI rank");
  }
  for (Real& value : payload)
    value = all_reduce_sum(value, lane);
  return payload;
}

POPS_HD static Real named_flux_face_value_(const FieldView<const Real, Dim>& flux,
                                           const Index<Dim>& face, int axis, int component) {
  Index<Dim> lower = face;
  --lower[axis];
  return Real(0.5) * (flux(lower, component) + flux(face, component));
}

std::vector<Real> collective_named_face_payload_(const std::array<field_type*, Dim>& cell_fluxes,
                                                 const field_type& field, int axis,
                                                 const Index<Dim>& face) const {
  std::vector<Real> payload;
  std::exception_ptr local_error;
  try {
    if (axis < 0 || axis >= Dim || cell_fluxes[static_cast<std::size_t>(axis)] == nullptr)
      throw std::out_of_range("AMR Program named face-flux axis is outside its exact rank");
    const field_type& flux = *cell_fluxes[static_cast<std::size_t>(axis)];
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR Program named interface face has no level flux patch");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    payload.assign(static_cast<std::size_t>(field.ncomp()), Real(0));
    if (owner == field.local_rank()) {
      const std::size_t local = flux.local_index_of(selected);
      if (local == field_type::not_local || local >= flux.local_size())
        throw std::runtime_error("AMR Program named interface flux lost its local patch");
      const FieldView<const Real, Dim> values = std::as_const(flux).fab(local).view();
      for (int component = 0; component < field.ncomp(); ++component) {
        Real value = Real(0);
        Kokkos::parallel_reduce(
            "pops_program_amr_named_interface_face", Kokkos::RangePolicy<>(0, 1),
            [=] POPS_HD(int, Real& sum) {
              sum += named_flux_face_value_(values, face, axis, component);
            },
            value);
        payload[static_cast<std::size_t>(component)] = value;
      }
      Kokkos::fence();
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program named interface flux failed collectively");
  }
  for (Real& value : payload)
    value = all_reduce_sum(value, lane);
  return payload;
}

std::vector<Real> collective_face_payload_(
    const std::array<field_type, Dim>& integrated_face_fluxes, const field_type& field, int axis,
    const Index<Dim>& face) const {
  std::vector<Real> payload;
  std::exception_ptr local_error;
  try {
    if (axis < 0 || axis >= Dim)
      throw std::out_of_range("AMR Program exact face-flux axis is outside its rank");
    const field_type& faces = integrated_face_fluxes[static_cast<std::size_t>(axis)];
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR Program exact interface face has no level flux patch");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    payload.assign(static_cast<std::size_t>(field.ncomp()), Real(0));
    if (owner == field.local_rank()) {
      const std::size_t local = faces.local_index_of(selected);
      if (local == field_type::not_local || local >= faces.local_size())
        throw std::runtime_error("AMR Program exact interface flux lost its local patch");
      const FieldView<const Real, Dim> values = faces.fab(local).view();
      for (int component = 0; component < field.ncomp(); ++component) {
        Real value = Real(0);
        Kokkos::parallel_reduce(
            "pops_program_amr_exact_interface_face", Kokkos::RangePolicy<>(0, 1),
            [=] POPS_HD(int, Real& sum) { sum += values(face, component); }, value);
        payload[static_cast<std::size_t>(component)] = value;
      }
      Kokkos::fence();
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program exact interface flux failed collectively");
  }
  for (Real& value : payload)
    value = all_reduce_sum(value, lane);
  return payload;
}

static void payload_axpy_(std::vector<Real>& destination, double factor,
                          const std::vector<Real>& source) {
  if (destination.empty())
    destination.assign(source.size(), Real(0));
  if (destination.size() != source.size())
    throw std::invalid_argument("AMR Program reflux payload component counts differ");
  for (std::size_t component = 0; component < source.size(); ++component)
    destination[component] += static_cast<Real>(factor) * source[component];
}

static double face_measure_(const Geometry<Dim>& geometry, int normal_axis) {
  double result = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    if (axis != normal_axis)
      result *= static_cast<double>(geometry.spacing(axis));
  return result;
}

static double cell_measure_(const Geometry<Dim>& geometry) {
  double result = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<double>(geometry.spacing(axis));
  return result;
}
