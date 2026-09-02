const std::string& prepared_hot_fence_label_() const noexcept {
  // Kokkos 5.1 materializes a long default label for `fence()`; this short SSO resident label
  // is shared by every direct AMR hot fence below.
  static const std::string label{"pops.amr.fence"};
  return label;
}

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
    const ::pops::amr::ClockWindow root{{0,
                                         static_cast<std::int64_t>(facade_->program_macro_step_()),
                                         {0, 1},
                                         facade_->program_time_()},
                                        {0,
                                         static_cast<std::int64_t>(facade_->program_macro_step_()),
                                         {1, 1},
                                         facade_->program_time_() + dt}};
    const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
    if (!active_cell_temporal_execution_) {
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
    }
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
    if (!active_cell_temporal_execution_) {
      std::optional<typename interface_flux_ledger_type::PreparedCommit> prepared_commit;
      std::exception_ptr ledger_error;
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
      static_assert(std::is_nothrow_move_constructible_v<
                    typename interface_flux_ledger_type::PreparedCommit>);
      static_assert(noexcept(interface_flux_commit_guard_.swap(prepared_commit)));
      interface_flux_commit_guard_.swap(prepared_commit);
      interface_flux_ledger_->publish_prepared_commit(*interface_flux_commit_guard_);
    }
  } catch (...) {
    if (!active_cell_temporal_execution_ && interface_flux_ledger_ &&
        interface_flux_ledger_->in_transaction())
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
  return {0, 0, 1, 0, "pops.amr-program.interface-flux-budget/inactive"};
}

/// Complete cold-built subcycling image.  Nothing in this carrier aliases a preparation
/// authority; publication is one no-throw exchange after all routes and budgets agree.
struct SubcyclingPreparationAuthority final {
  prepared_multiblock_type* hierarchy = nullptr;
  std::vector<Geometry<Dim>> level_geometries;
  std::span<const ::pops::amr::ParentChildClockRelation> relations;
  const flux_expression_budget_type* expression_budget = nullptr;
  const typename prepared_multiblock_type::ProgramBlockMap* program_block_map = nullptr;
  ::pops::amr::InterfaceFluxLedgerBudget interface_budget{};
  std::string_view installed_hash;
  BoundaryTopology<Dim> boundary_topology{};

  [[nodiscard]] const hierarchy_type& topology_hierarchy() const {
    return hierarchy->topology_runtime().hierarchy();
  }
  [[nodiscard]] const std::vector<Geometry<Dim>>& prepared_level_geometries() const {
    return level_geometries;
  }
  [[nodiscard]] std::size_t block_count() const { return hierarchy->block_count(); }
  [[nodiscard]] std::string_view block_identity(std::size_t block) const {
    return hierarchy->block_identity(block);
  }
  [[nodiscard]] const ExecutionLane& lane() const { return hierarchy->lane(); }
  [[nodiscard]] std::string_view collective_contract() const {
    return hierarchy->collective_contract();
  }
  [[nodiscard]] std::uint64_t topology_epoch() const {
    return hierarchy->topology_runtime().topology_epoch();
  }
  [[nodiscard]] std::uint64_t materialization_generation() const {
    return hierarchy->topology_runtime().materialization_generation();
  }
  [[nodiscard]] multiblock_subcycling_type prepare_engine(
      std::span<const ::pops::amr::ParentChildClockRelation> value_relations,
      ::pops::numerics::time::amr::MultiBlockAmrSubcyclingBudget budget) const {
    return multiblock_subcycling_type::prepare(*hierarchy, value_relations, budget);
  }

  void validate() const {
    if (hierarchy == nullptr || expression_budget == nullptr || program_block_map == nullptr ||
        installed_hash.empty() || !lane().active() || collective_contract().empty() ||
        program_block_map->canonical_indices.size() != block_count() ||
        program_block_map->hierarchy_contract != collective_contract() ||
        program_block_map->exact_contract.empty() ||
        expression_budget->program_hash != installed_hash ||
        expression_budget->generation != materialization_generation() ||
        expression_budget->exact_contract.empty() ||
        expression_budget->program_block_map.canonical_indices !=
            program_block_map->canonical_indices ||
        expression_budget->program_block_map.hierarchy_contract !=
            program_block_map->hierarchy_contract ||
        expression_budget->program_block_map.exact_contract != program_block_map->exact_contract ||
        expression_budget->blocks.size() != block_count() ||
        interface_budget.exact_contract.empty() ||
        level_geometries.size() != topology_hierarchy().num_levels() ||
        relations.size() + 1 != topology_hierarchy().num_levels())
      throw std::invalid_argument("AMR Program subcycling preparation authority is incomplete");
  }
};

[[nodiscard]] SubcyclingPreparationAuthority detached_subcycling_authority_() const {
  if (preparation_view_ == nullptr)
    throw std::logic_error("AMR detached subcycling preparation has no topology image");
  preparation_view_->validate_subcycling_authority();
  std::array<bool, Dim> periodic{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t lower = static_cast<std::size_t>(2 * axis);
    const std::size_t upper = lower + 1U;
    if (preparation_view_->periodic_faces[lower] != preparation_view_->periodic_faces[upper])
      throw std::invalid_argument(
          "AMR detached subcycling preparation has asymmetric periodic faces");
    periodic[static_cast<std::size_t>(axis)] = preparation_view_->periodic_faces[lower];
  }
  SubcyclingPreparationAuthority authority{
      .hierarchy = preparation_view_->candidate_multiblock,
      .level_geometries = preparation_view_->level_geometries,
      .relations = preparation_view_->temporal_relations,
      .expression_budget = preparation_view_->candidate_flux_expression_budget,
      .program_block_map = preparation_view_->candidate_program_block_map,
      .interface_budget = *preparation_view_->candidate_interface_flux_ledger_budget,
      .installed_hash = *preparation_view_->candidate_installed_hash,
      .boundary_topology = BoundaryTopology<Dim>::axis_periodic(periodic),
  };
  authority.validate();
  return authority;
}

[[nodiscard]] SubcyclingPreparationAuthority accepted_subcycling_authority_() const {
  require_facade_execution_();
  auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  std::vector<Geometry<Dim>> level_geometries;
  level_geometries.reserve(hierarchy.topology_runtime().hierarchy().num_levels());
  for (std::size_t level = 0; level < hierarchy.topology_runtime().hierarchy().num_levels();
       ++level)
    level_geometries.push_back(
        facade_->program_prepared_amr_level_geometry_(static_cast<int>(level)));
  SubcyclingPreparationAuthority authority{
      .hierarchy = std::addressof(hierarchy),
      .level_geometries = std::move(level_geometries),
      .relations = facade_->program_prepared_temporal_relations_(),
      .expression_budget =
          std::addressof(facade_->program_prepared_amr_program_flux_expression_budget_()),
      .program_block_map = std::addressof(facade_->program_prepared_amr_program_block_map_()),
      .interface_budget = facade_->program_prepared_amr_interface_flux_ledger_budget_(),
      .installed_hash = facade_->program_installed_hash_(),
      .boundary_topology = facade_->program_prepared_amr_boundary_topology_(),
  };
  authority.validate();
  return authority;
}

template <class Authority>
[[nodiscard]] static PreparedSubcyclingBundle prepare_multiblock_subcycling_bundle_from_authority_(
    const Authority& authority, const PreparedFluxTableCarrier& flux_seed,
    PreparedHotPathWorkspace workspace, std::string_view primary_clock) {
  authority.validate();
  PreparedSubcyclingBundle bundle;
  bundle.hot_path_workspace = std::move(workspace);
  const auto& hierarchy = authority.topology_hierarchy();
  if (authority.prepared_level_geometries().size() != hierarchy.num_levels())
    throw std::logic_error("AMR Program subcycling geometry authority is incomplete");
  bundle.level_geometries = authority.prepared_level_geometries();
  // A bound v5 Program has no dynamic ledger fallback.  The rollback image and retry path own
  // only resident slots, so reject a missing exact table collectively before preparing or
  // publishing any candidate subcycling/ledger authority.
  std::exception_ptr static_table_error;
  try {
    if (primary_clock.empty())
      throw std::logic_error("AMR Program subcycling requires a prepared primary clock");
    // The completed subcycling bundle replaces the adapter's prior hot workspace by noexcept
    // swap.  Prime the replacement itself while construction is still cold; priming only the
    // pre-swap adapter would discard the clock storage and make the first accepted RHS allocate
    // or fail after publication.
    bundle.hot_path_workspace.bind_boundary_point_clock(primary_clock);
    if (!flux_seed.bound)
      throw std::logic_error(
          "AMR Program subcycling requires bind-sealed static face-flux resident tables");
  } catch (...) {
    static_table_error = std::current_exception();
  }
  if (all_reduce_max(static_table_error ? 1L : 0L, authority.lane()) != 0) {
    if (authority.lane().size() == 1 && static_table_error)
      std::rethrow_exception(static_table_error);
    throw std::runtime_error(
        "AMR Program subcycling lacks static face-flux resident tables collectively");
  }
  const std::uint64_t topology_epoch = authority.topology_epoch();
  const std::uint64_t materialization_generation = authority.materialization_generation();
  std::vector<std::size_t> rhs_basis_bounds;
  std::vector<std::size_t> coefficient_term_bounds;
  std::string program_budget_contract;
  std::exception_ptr local_error;
  try {
    const flux_expression_budget_type& expression_budget = *authority.expression_budget;
    const auto& prepared_map = *authority.program_block_map;
    const std::size_t blocks = authority.block_count();
    if (expression_budget.program_hash.empty() ||
        expression_budget.program_hash != authority.installed_hash ||
        expression_budget.generation != materialization_generation ||
        expression_budget.exact_contract.empty() ||
        expression_budget.program_block_map.canonical_indices != prepared_map.canonical_indices ||
        expression_budget.program_block_map.hierarchy_contract != prepared_map.hierarchy_contract ||
        expression_budget.program_block_map.exact_contract != prepared_map.exact_contract ||
        expression_budget.blocks.size() != blocks ||
        prepared_map.canonical_indices.size() != blocks)
      throw std::logic_error(
          "AMR Program flux-expression budget is not authentic for the active carrier");

    rhs_basis_bounds.assign(blocks, 0);
    coefficient_term_bounds.assign(blocks, 0);
    std::vector<bool> seen(blocks, false);
    for (std::size_t program_block = 0; program_block < blocks; ++program_block) {
      const std::size_t canonical = prepared_map.canonical_indices[program_block];
      if (canonical >= blocks || seen[canonical])
        throw std::logic_error(
            "AMR Program flux-expression budget differs from its exact block permutation");
      seen[canonical] = true;
      const auto& block_budget = expression_budget.blocks[program_block];
      if (block_budget.rhs_basis_bound == 0 && block_budget.coefficient_term_bound != 0)
        throw std::logic_error("AMR Program coefficient terms require a non-zero RHS-basis bound");
      rhs_basis_bounds[canonical] = block_budget.rhs_basis_bound;
      coefficient_term_bounds[canonical] = block_budget.coefficient_term_bound;
    }
    program_budget_contract = expression_budget.exact_contract;
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = authority.lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (authority.lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program flux-expression budget validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-flux-expression-budget"),
            std::string_view(program_budget_contract)}},
          lane))
    throw std::invalid_argument(
        "AMR Program flux-expression budget differs between prepared-lane ranks");

  std::span<const ::pops::amr::ParentChildClockRelation> relations;
  ::pops::amr::InterfaceFluxLedgerBudget interface_budget;
  local_error = nullptr;
  try {
    // This detached candidate has just passed collective exact-budget authentication.  Reserve
    // the hot invocation strings from that immutable authority before any carrier is published.
    bundle.hot_path_workspace.bind_coupling_invocation(
        std::max(authority.expression_budget->interface_coupling_identity_character_bound,
                 primary_clock.size()));
    relations = authority.relations;
    if (relations.size() + 1 != hierarchy.num_levels())
      throw std::logic_error(
          "AMR Program subcycling lacks one temporal relation per live transition");
    interface_budget = authority.interface_budget;
    bundle.interface_ledger =
        std::make_unique<interface_flux_ledger_type>(topology_epoch, interface_budget);
    bundle.interface_ledger->prime_snapshot_arenas_at_bind();
    bundle.interface_ledger->prime_snapshot_slots_at_bind();
    bundle.interface_ledger->prime_hot_carriers_at_bind();
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
  std::unique_ptr<multiblock_subcycling_type> prepared;
  PreparedFluxTableCarrier staged_flux_tables;
  std::vector<FluxBasis> staged_flux_basis_payloads;
  std::vector<std::uint8_t> staged_flux_basis_active;
  std::string staged_flux_collective_contract;
  local_error = nullptr;
  try {
    std::size_t maximum_patches = 1;
    for (std::size_t level = 0; level < hierarchy.num_levels(); ++level)
      maximum_patches = std::max(maximum_patches, hierarchy.layout(level).patches().size());
    const std::size_t overlap_pairs = maximum_patches < 2
                                          ? 1
                                          : checked_product_(maximum_patches, maximum_patches - 1,
                                                             "AMR Program patch-overlap budget") /
                                                2;

    // The ledger implementation requires a positive structural capacity even for an
    // authenticated source-only Program.  Flux-producing Programs derive every retained entry
    // from the exact per-block RHS-basis and coefficient-term bounds below; no evaluation-count
    // fallback is used.  An authenticated all-cancel table has a zero term bound and therefore
    // retains only this structural minimum.
    std::size_t maximum_entries = 1;
    for (std::size_t parent = 0; parent < relations.size(); ++parent) {
      const auto ratio = hierarchy.layout(parent + 1).ratio_from_parent();
      const auto temporal = relations[parent].temporal_ratio();
      const std::size_t substeps =
          static_cast<std::size_t>(temporal.numerator / temporal.denominator +
                                   (temporal.numerator % temporal.denominator == 0 ? 0 : 1));
      for (std::size_t block = 0; block < authority.block_count(); ++block) {
        std::size_t block_entries = 0;
        for (const ProgramInterfaceFace& interface :
             program_interface_faces_(hierarchy, authority.boundary_topology, parent)) {
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
          const std::size_t terms_per_block =
              checked_product_(rhs_basis_bounds[block], coefficient_term_bounds[block],
                               "AMR Program authenticated coefficient-term budget");
          const std::size_t face_entries =
              checked_product_(terms_per_block, fragments_per_basis,
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
    prepared =
        std::make_unique<multiblock_subcycling_type>(authority.prepare_engine(relations, budget));
    if (flux_seed.bound) {
      // The topology expansion is cold but still transactional.  Build every route and payload
      // in a detached image so an invalid row/topology cannot damage a resident Program that is
      // still valid for no-clobber rejection handling.
      staged_flux_tables = flux_seed;
      staged_flux_basis_payloads.resize(staged_flux_tables.bases.size());
      staged_flux_basis_active.assign(staged_flux_tables.bases.size(), std::uint8_t{0});
      using prepared_slot_type = typename multiblock_flux_ledger_type::PreparedSlot;
      for (auto& term : staged_flux_tables.terms)
        term.ledger_routes.clear();
      for (auto& basis : staged_flux_tables.bases)
        basis.face_routes.clear();

      prepared->bind_candidate_ledger_slots([&](std::size_t block, std::size_t parent,
                                                std::size_t invocation,
                                                multiblock_flux_ledger_type& ledger) {
        if (block >= authority.block_count() || parent >= relations.size() ||
            parent + 1 >= hierarchy.num_levels())
          throw std::logic_error("AMR Program resident flux slot has no transition");
        const std::string owner{authority.block_identity(block)};
        const std::string state = owner + "/state";
        const auto ratio = hierarchy.layout(parent + 1).ratio_from_parent();
        const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
            hierarchy.layout(parent).domain().lo, hierarchy.layout(parent + 1).domain().lo};
        const auto interfaces =
            program_interface_faces_(hierarchy, authority.boundary_topology, parent);
        const auto temporal = relations[parent].temporal_ratio();
        if (temporal.numerator <= 0 || temporal.denominator <= 0)
          throw std::invalid_argument(
              "AMR Program resident flux slot has an invalid temporal subcycle relation");
        const std::int64_t whole_substeps = temporal.numerator / temporal.denominator;
        const std::size_t fine_substeps = static_cast<std::size_t>(
            whole_substeps + (temporal.numerator % temporal.denominator == 0 ? 0 : 1));
        if (fine_substeps == 0 || fine_substeps > std::numeric_limits<std::uint32_t>::max())
          throw std::length_error(
              "AMR Program resident fine-flux substep count exceeds its compact carrier");
        const ::pops::amr::ClockWindow parent_window =
            prepared->candidate_ledger_window(parent, invocation);
        const auto fine_windows = relations[parent].partition(parent_window);
        if (fine_windows.size() != fine_substeps)
          throw std::logic_error(
              "AMR Program resident flux windows differ from their prepared subcycle count");
        std::vector<prepared_slot_type> slots;
        auto& term_slots = staged_flux_tables.term_slots_by_runtime_block.at(block);
        std::sort(term_slots.begin(), term_slots.end(),
                  [&](std::uint32_t left, std::uint32_t right) {
                    const auto& left_term = staged_flux_tables.terms.at(left);
                    const auto& right_term = staged_flux_tables.terms.at(right);
                    return std::tie(left_term.basis_slot, left_term.slot) <
                           std::tie(right_term.basis_slot, right_term.slot);
                  });
        for (const std::uint32_t term_slot : term_slots) {
          auto& term = staged_flux_tables.terms.at(term_slot);
          auto& basis = staged_flux_tables.bases.at(term.basis_slot);
          if (basis.components == 0)
            throw std::invalid_argument(
                "AMR Program resident flux slot has no sealed payload components");
          const auto append_route = [&](::pops::amr::reflux::FaceLedgerRole role,
                                        std::size_t level) {
            if (basis.level >= 0 && basis.level != static_cast<int>(level))
              return;
            typename PreparedFluxTableCarrier::LedgerRoute route;
            route.ledger = &ledger;
            route.level = static_cast<std::int32_t>(level);
            route.role = role;
            route.substep_count = static_cast<std::uint32_t>(
                role == ::pops::amr::reflux::FaceLedgerRole::Fine ? fine_substeps : 1U);
            auto face_route = std::find_if(
                basis.face_routes.begin(), basis.face_routes.end(), [&](const auto& candidate) {
                  return candidate.ledger == &ledger &&
                         candidate.level == static_cast<int>(level) && candidate.role == role;
                });
            if (face_route == basis.face_routes.end()) {
              typename PreparedFluxTableCarrier::Basis::FaceRoute candidate;
              candidate.ledger = &ledger;
              candidate.level = static_cast<std::int32_t>(level);
              candidate.role = role;
              for (const ProgramInterfaceFace& interface : interfaces) {
                const auto append = [&](const Index<Dim>& face) {
                  candidate.faces.push_back({interface.axis, face, interface.coarse_face, 0.0});
                };
                if (role == ::pops::amr::reflux::FaceLedgerRole::Coarse) {
                  append(interface.coarse_face);
                  continue;
                }
                ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
                query.owner = owner;
                query.state = state;
                query.levels = {static_cast<int>(parent), static_cast<int>(parent + 1)};
                query.axis = interface.axis;
                query.coarse_face = interface.coarse_face;
                query.attempt = 0;
                query.macro_step = 0;
                query.window_begin = {0, 1};
                query.window_end = {1, 1};
                std::size_t fine_count = 1;
                for (int axis = 0; axis < Dim; ++axis)
                  if (axis != interface.axis)
                    fine_count = checked_product_(fine_count, static_cast<std::size_t>(ratio[axis]),
                                                  "AMR Program resident fine-face enumeration");
                const ::pops::amr::reflux::MetricRefluxBudget fine_budget{fine_count, fine_count,
                                                                          1};
                for (const Index<Dim>& face : ::pops::amr::reflux::fine_faces_for_coarse_face(
                         query, ratio, mapping, fine_budget))
                  append(face);
              }
              std::sort(candidate.faces.begin(), candidate.faces.end(),
                        [](const auto& left, const auto& right) {
                          if (left.axis != right.axis)
                            return left.axis < right.axis;
                          for (int axis = 0; axis < Dim; ++axis) {
                            if (left.face[axis] != right.face[axis])
                              return left.face[axis] < right.face[axis];
                            if (left.coarse_face[axis] != right.coarse_face[axis])
                              return left.coarse_face[axis] < right.coarse_face[axis];
                          }
                          return false;
                        });
              basis.face_routes.push_back(std::move(candidate));
              face_route = std::prev(basis.face_routes.end());
            }
            if (face_route->faces.size() >
                std::numeric_limits<std::size_t>::max() / route.substep_count)
              throw std::length_error(
                  "AMR Program resident flux route substep slices exceed size_t");
            route.slots.assign(face_route->faces.size() * route.substep_count,
                               std::numeric_limits<std::uint32_t>::max());
            term.ledger_routes.push_back(std::move(route));
          };
          append_route(::pops::amr::reflux::FaceLedgerRole::Coarse, parent);
          append_route(::pops::amr::reflux::FaceLedgerRole::Fine, parent + 1);
        }
        // The ledger's prepared order is part of its resident authority.  A coarse group runs
        // once, then every fine child group reuses this ledger in ascending local substep order.
        // Materialize the complete cold sequence accordingly; route slots retain their own
        // dense [substep][face] index so the warm path never searches a key or grows a vector.
        constexpr std::array<::pops::amr::reflux::FaceLedgerRole, 2> roles{
            ::pops::amr::reflux::FaceLedgerRole::Coarse, ::pops::amr::reflux::FaceLedgerRole::Fine};
        for (int axis = 0; axis < Dim; ++axis)
          for (const auto role : roles) {
            const std::size_t substep_count =
                role == ::pops::amr::reflux::FaceLedgerRole::Fine ? fine_substeps : 1U;
            for (std::size_t substep = 0; substep < substep_count; ++substep)
              for (const std::uint32_t term_slot : term_slots) {
                auto& term = staged_flux_tables.terms.at(term_slot);
                auto& basis = staged_flux_tables.bases.at(term.basis_slot);
                const auto route = std::find_if(
                    term.ledger_routes.begin(), term.ledger_routes.end(),
                    [&](const auto& candidate) {
                      return candidate.ledger == &ledger &&
                             candidate.level ==
                                 static_cast<int>(
                                     role == ::pops::amr::reflux::FaceLedgerRole::Coarse
                                         ? parent
                                         : parent + 1) &&
                             candidate.role == role;
                    });
                if (route == term.ledger_routes.end())
                  continue;
                const auto face_route = std::find_if(
                    basis.face_routes.begin(), basis.face_routes.end(), [&](const auto& candidate) {
                      return candidate.ledger == &ledger && candidate.level == route->level &&
                             candidate.role == role;
                    });
                if (face_route == basis.face_routes.end() ||
                    route->substep_count != substep_count ||
                    route->slots.size() != face_route->faces.size() * substep_count)
                  throw std::logic_error(
                      "AMR Program resident flux route lost its exact substep slice shape");
                for (std::size_t face_index = 0; face_index < face_route->faces.size();
                     ++face_index) {
                  const auto& face = face_route->faces[face_index];
                  if (face.axis != axis)
                    continue;
                  const std::size_t route_index = substep * face_route->faces.size() + face_index;
                  if (route->slots[route_index] != std::numeric_limits<std::uint32_t>::max() ||
                      slots.size() > std::numeric_limits<std::uint32_t>::max())
                    throw std::length_error(
                        "AMR Program resident flux slot index exceeds its compact carrier");
                  ::pops::amr::reflux::FaceFluxFragmentKey<Dim> key;
                  key.owner = owner;
                  key.state = state;
                  key.levels = {static_cast<int>(parent), static_cast<int>(parent + 1)};
                  key.axis = face.axis;
                  key.face = face.face;
                  key.coarse_face = face.coarse_face;
                  if (basis.stage < ::pops::amr::Rational{0, 1} ||
                      ::pops::amr::Rational{1, 1} < basis.stage)
                    throw std::logic_error(
                        "AMR Program resident flux slot has an invalid static stage fraction");
                  const ::pops::amr::ClockWindow& window =
                      role == ::pops::amr::reflux::FaceLedgerRole::Fine
                          ? fine_windows.at(substep).window
                          : parent_window;
                  if (window.begin.level != route->level || window.end.level != route->level)
                    throw std::logic_error(
                        "AMR Program resident flux slot has a mismatched prepared clock level");
                  const ::pops::amr::Rational stage_phase =
                      window.begin.phase + (window.end.phase - window.begin.phase) * basis.stage;
                  const double stage_time =
                      window.begin.physical_time +
                      basis.stage.value() * (window.end.physical_time - window.begin.physical_time);
                  key.clock = {route->level, window.begin.macro_step, stage_phase, stage_time};
                  key.stage = term.stage_identity;
                  key.role = role;
                  route->slots[route_index] = static_cast<std::uint32_t>(slots.size());
                  slots.push_back({std::move(key), basis.components});
                }
              }
          }
        for (const std::uint32_t term_slot : term_slots)
          for (const auto& route : staged_flux_tables.terms.at(term_slot).ledger_routes)
            if (route.ledger == &ledger &&
                std::any_of(route.slots.begin(), route.slots.end(), [](std::uint32_t slot) {
                  return slot == std::numeric_limits<std::uint32_t>::max();
                }))
              throw std::logic_error(
                  "AMR Program resident flux route has an unmaterialized substep slot");
        // An empty list is the authenticated cancellation case: it remains a prepared
        // ledger but has no publication slots to materialize during this transition.
        ledger.prepare_resident_slots(slots);
        // Bind one metric-reflux authority for each exact coarse interface that has a resident
        // coarse slot.  Resident slot templates are the cold payload authority while the
        // accepted publication remains empty; no synthetic accepted fragment is permitted.
        for (const ProgramInterfaceFace& interface : interfaces) {
          const auto coarse =
              std::find_if(slots.begin(), slots.end(), [&](const prepared_slot_type& slot) {
                return slot.key.role == ::pops::amr::reflux::FaceLedgerRole::Coarse &&
                       slot.key.owner == owner && slot.key.state == state &&
                       slot.key.levels ==
                           ::pops::amr::reflux::LevelTransition{static_cast<int>(parent),
                                                                static_cast<int>(parent + 1)} &&
                       slot.key.axis == interface.axis &&
                       slot.key.coarse_face == interface.coarse_face;
              });
          if (coarse == slots.end())
            continue;
          std::size_t fine_faces = 1;
          for (int axis = 0; axis < Dim; ++axis)
            if (axis != interface.axis)
              fine_faces = checked_product_(fine_faces, static_cast<std::size_t>(ratio[axis]),
                                            "AMR Program prepared metric reflux fine-face budget");
          typename PreparedHotPathWorkspace::PreparedMetricRefluxRoute route;
          route.ledger = &ledger;
          route.block = block;
          route.parent_level = parent;
          route.interface = interface;
          route.query.owner = owner;
          route.query.state = state;
          route.query.levels = {static_cast<int>(parent), static_cast<int>(parent + 1)};
          route.query.axis = interface.axis;
          route.query.coarse_face = interface.coarse_face;
          route.query.attempt = coarse->key.attempt;
          route.query.macro_step = coarse->key.clock.macro_step;
          route.query.window_begin = parent_window.begin.phase;
          route.query.window_end = parent_window.end.phase;
          route.bound_window_begin = parent_window.begin.phase;
          route.bound_window_end = parent_window.end.phase;
          route.ratio = ratio;
          route.mapping = mapping;
          route.budget = {fine_faces, slots.size(), slots.size()};

          // The candidate-slot walk visits the same resident ledger once for each descendant
          // invocation. Metric reflux is instead an authority per exact coarse interface and
          // parent clock window. Collapse only byte-for-byte identical cold routes; a duplicate
          // key with different authority is rejected before it can reach the hot path.
          std::size_t equivalent_routes = 0;
          for (const auto& existing : bundle.hot_path_workspace.prepared_metric_reflux_routes) {
            const bool same_key = existing.ledger == route.ledger &&
                                  existing.block == route.block &&
                                  existing.parent_level == route.parent_level &&
                                  existing.interface.axis == route.interface.axis &&
                                  existing.interface.coarse_face == route.interface.coarse_face &&
                                  existing.interface.coarse_cell == route.interface.coarse_cell &&
                                  existing.interface.side == route.interface.side &&
                                  existing.bound_window_begin == route.bound_window_begin &&
                                  existing.bound_window_end == route.bound_window_end;
            if (!same_key)
              continue;
            const bool same_authority =
                existing.query.owner == route.query.owner &&
                existing.query.state == route.query.state &&
                existing.query.levels == route.query.levels &&
                existing.query.centering == route.query.centering &&
                existing.query.axis == route.query.axis &&
                existing.query.coarse_face == route.query.coarse_face &&
                existing.query.attempt == route.query.attempt &&
                existing.query.macro_step == route.query.macro_step &&
                existing.query.window_begin == route.query.window_begin &&
                existing.query.window_end == route.query.window_end &&
                existing.ratio == route.ratio && existing.mapping == route.mapping &&
                existing.budget.max_fine_faces == route.budget.max_fine_faces &&
                existing.budget.max_published_entries == route.budget.max_published_entries &&
                existing.budget.max_clock_stage_slices == route.budget.max_clock_stage_slices;
            if (!same_authority)
              throw std::logic_error(
                  "AMR Program prepared metric reflux route has a conflicting cold authority");
            ++equivalent_routes;
          }
          if (equivalent_routes > 1)
            throw std::logic_error(
                "AMR Program prepared metric reflux route duplicated its cold authority");
          if (equivalent_routes == 1)
            continue;
          route.workspace.prepare(ledger, route.query, route.ratio, route.mapping, route.budget);
          bundle.hot_path_workspace.prepared_metric_reflux_routes.push_back(std::move(route));
        }
        (void)invocation;
      });
      prepared->mirror_candidate_ledger_slots_into_accepted_at_bind();
      for (const auto& basis : staged_flux_tables.bases) {
        if (basis.basis_slot >= staged_flux_basis_payloads.size())
          throw std::logic_error("AMR Program resident basis payload slot is not dense");
        FluxBasis& resident = staged_flux_basis_payloads[basis.basis_slot];
        resident.faces.clear();
        // The retained history carrier has one exact cold face image, not a maximum-sized tail
        // of uninitialized slots.  Preserve the same Coarse-then-Fine ordering used by the hot
        // RHS attachment; this is the stable occurrence ordering for regrid transfer.
        constexpr std::array<::pops::amr::reflux::FaceLedgerRole, 2> face_roles{
            ::pops::amr::reflux::FaceLedgerRole::Coarse, ::pops::amr::reflux::FaceLedgerRole::Fine};
        for (const auto role : face_roles) {
          for (const auto& route : basis.face_routes) {
            if (route.role != role)
              continue;
            if (route.level < 0)
              throw std::logic_error("AMR Program resident basis route has a negative level");
            if (static_cast<std::size_t>(route.level) >= bundle.level_geometries.size())
              throw std::logic_error(
                  "AMR Program resident basis route level exceeds its sealed geometry image");
            const Geometry<Dim>& geometry =
                bundle.level_geometries.at(static_cast<std::size_t>(route.level));
            for (const auto& template_face : route.faces) {
              const double measure = face_measure_(geometry, template_face.axis);
              if (!(measure > 0.0) || !std::isfinite(measure))
                throw std::invalid_argument(
                    "AMR Program resident basis route has an invalid sealed face measure");
              FluxBasisFace face;
              face.role = role;
              face.axis = template_face.axis;
              face.face = template_face.face;
              face.coarse_face = template_face.coarse_face;
              face.face_measure = measure;
              face.flux_density.resize(basis.components, Real(0));
              resident.faces.push_back(std::move(face));
            }
          }
        }
        resident.face_count = resident.faces.size();
      }
      ExactContractBuilder flux_contract;
      flux_contract.text("pops.amr-program.static-flux-helper-preflight")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(static_cast<std::uint64_t>(staged_flux_tables.bases.size()))
          .scalar(static_cast<std::uint64_t>(staged_flux_tables.terms.size()));
      for (const auto& basis : staged_flux_tables.bases) {
        flux_contract.scalar(basis.basis_slot)
            .scalar(basis.expression_slot)
            .scalar(basis.runtime_block)
            .scalar(basis.level)
            .scalar(basis.rhs_identity)
            .scalar(basis.provider)
            .scalar(basis.components)
            .scalar(basis.stage.numerator)
            .scalar(basis.stage.denominator)
            .scalar(static_cast<std::uint64_t>(basis.face_routes.size()));
        for (const auto& route : basis.face_routes) {
          flux_contract.scalar(route.level)
              .scalar(static_cast<std::uint8_t>(route.role))
              .scalar(static_cast<std::uint64_t>(route.faces.size()));
          for (const auto& face : route.faces) {
            flux_contract.scalar(face.axis);
            for (int axis = 0; axis < Dim; ++axis)
              flux_contract.scalar(face.face[axis]);
            for (int axis = 0; axis < Dim; ++axis)
              flux_contract.scalar(face.coarse_face[axis]);
          }
        }
      }
      for (const auto& term : staged_flux_tables.terms) {
        flux_contract.scalar(term.slot)
            .scalar(term.basis_slot)
            .scalar(term.expression_slot)
            .scalar(term.coefficient.numerator)
            .scalar(term.coefficient.denominator)
            .text(term.stage_identity)
            .scalar(static_cast<std::uint64_t>(term.ledger_routes.size()));
        for (const auto& route : term.ledger_routes) {
          flux_contract.scalar(route.level)
              .scalar(static_cast<std::uint8_t>(route.role))
              .scalar(route.substep_count)
              .scalar(static_cast<std::uint64_t>(route.slots.size()));
          for (const std::uint32_t slot : route.slots)
            flux_contract.scalar(slot);
        }
      }
      flux_contract.scalar(static_cast<std::uint64_t>(
          bundle.hot_path_workspace.prepared_metric_reflux_routes.size()));
      for (const auto& route : bundle.hot_path_workspace.prepared_metric_reflux_routes) {
        flux_contract.text(route.query.owner)
            .text(route.query.state)
            .scalar(route.block)
            .scalar(route.parent_level)
            .scalar(route.interface.axis)
            .scalar(static_cast<std::uint8_t>(route.interface.side))
            .scalar(route.query.levels.coarse)
            .scalar(route.query.levels.fine)
            .scalar(route.budget.max_fine_faces)
            .scalar(route.budget.max_published_entries)
            .scalar(route.budget.max_clock_stage_slices);
        for (int axis = 0; axis < Dim; ++axis) {
          flux_contract.scalar(route.interface.coarse_face[axis])
              .scalar(route.interface.coarse_cell[axis])
              .scalar(route.mapping.coarse_origin[axis])
              .scalar(route.mapping.fine_origin[axis])
              .scalar(route.ratio[axis]);
        }
      }
      staged_flux_collective_contract = std::move(flux_contract).release();
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (authority.lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program subcycling preparation failed collectively");
  }
  if (flux_seed.bound &&
      !all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-program-static-flux-helper-preflight", staged_flux_collective_contract}}, lane))
    throw std::invalid_argument(
        "AMR Program static flux helper routes differ between execution ranks");

  // The six active level-group packs are intentionally empty between callbacks: their size is
  // also the public marker that no candidate attempt is live.  Reserve their complete block
  // envelope only while this authority is cold, then every hot resize is capacity-checked below.
  const auto prime_active_group = [&](auto& storage) {
    if (!storage.empty())
      throw std::logic_error("AMR Program cold subcycling bind crossed an active level group");
    storage.reserve(authority.block_count());
  };
  try {
    prime_active_group(bundle.active_attempt_states);
    prime_active_group(bundle.active_staged_parents);
    prime_active_group(bundle.active_incoming_flux);
    prime_active_group(bundle.active_outgoing_flux);
    prime_active_group(bundle.active_block_identities);
    prime_active_group(bundle.active_flux_basis_counts);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (authority.lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(
        "AMR Program active level-group storage preparation failed collectively");
  }

  // Publish only after all cold construction and rank agreement have succeeded.  Both swaps are
  // no-throw and make the expanded carrier/budget visible as one resident generation.
  bundle.engine = std::move(prepared);
  bundle.flux_tables = std::move(staged_flux_tables);
  bundle.flux_basis_payloads = std::move(staged_flux_basis_payloads);
  bundle.flux_basis_active = std::move(staged_flux_basis_active);
  bundle.flux_collective_contract = std::move(staged_flux_collective_contract);
  bundle.rhs_basis_bounds = std::move(rhs_basis_bounds);
  bundle.coefficient_term_bounds = std::move(coefficient_term_bounds);
  bundle.program_budget_contract = std::move(program_budget_contract);
  // This engine is prepared against the facade's rebuilt multi-block hierarchy.  Regrid and
  // restart rematerialize that authority before the raw runtime's convenience view necessarily
  // observes the new generation, so cache only against the exact authority used to size faces.
  bundle.topology_epoch = topology_epoch;
  bundle.materialization_generation = materialization_generation;
  bundle.block_count = authority.block_count();
  return bundle;
}

void publish_prepared_subcycling_bundle_noexcept(PreparedSubcyclingBundle&& bundle) const noexcept {
  static_assert(std::is_nothrow_swappable_v<decltype(multiblock_subcycling_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(static_flux_tables_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(static_flux_basis_payloads_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(static_flux_basis_active_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(static_flux_collective_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(prepared_rhs_basis_bounds_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(prepared_coefficient_term_bounds_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(interface_flux_ledger_)>);
  static_assert(
      std::is_nothrow_swappable_v<decltype(multiblock_subcycling_program_budget_contract_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(hot_path_workspace_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(active_attempt_states_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(active_staged_parents_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(active_incoming_flux_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(active_outgoing_flux_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(active_block_identities_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(active_flux_basis_counts_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(multiblock_subcycling_epoch_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(multiblock_subcycling_generation_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(multiblock_subcycling_block_count_)>);
  if (!bundle.engine || !bundle.interface_ledger)
    std::terminate();
  using std::swap;
  swap(multiblock_subcycling_, bundle.engine);
  swap(interface_flux_ledger_, bundle.interface_ledger);
  swap(static_flux_tables_, bundle.flux_tables);
  swap(static_flux_basis_payloads_, bundle.flux_basis_payloads);
  swap(static_flux_basis_active_, bundle.flux_basis_active);
  static_flux_collective_contract_.swap(bundle.flux_collective_contract);
  swap(prepared_rhs_basis_bounds_, bundle.rhs_basis_bounds);
  swap(prepared_coefficient_term_bounds_, bundle.coefficient_term_bounds);
  multiblock_subcycling_program_budget_contract_.swap(bundle.program_budget_contract);
  swap(hot_path_workspace_, bundle.hot_path_workspace);
  swap(active_attempt_states_, bundle.active_attempt_states);
  swap(active_staged_parents_, bundle.active_staged_parents);
  swap(active_incoming_flux_, bundle.active_incoming_flux);
  swap(active_outgoing_flux_, bundle.active_outgoing_flux);
  swap(active_block_identities_, bundle.active_block_identities);
  swap(active_flux_basis_counts_, bundle.active_flux_basis_counts);
  swap(multiblock_subcycling_epoch_, bundle.topology_epoch);
  swap(multiblock_subcycling_generation_, bundle.materialization_generation);
  swap(multiblock_subcycling_block_count_, bundle.block_count);
}

/// Rebuild the accepted effect envelope for a freshly published live subcycling bundle.  The
/// detached forward path has its own equivalent because it retains A for rollback.  Here B is
/// already the live ledger owner, so no A ordinal may be inspected while rebuilding its wire
/// slots.  This remains a cold regrid boundary: every allocation happens before the later
/// ordinal rebind and accepted-step execution observes only the rebuilt resident pools.
void rebuild_live_accepted_effect_slots_from_published_bundle_cold_() const {
  if (!multiblock_subcycling_ || multiblock_subcycling_block_count_ == 0 || runtime_ == nullptr ||
      !accepted_forward_storage_capacity_)
    throw std::logic_error(
        "AMR Program live accepted-effect envelope has no published bundle authority");
  const auto& capacity = *accepted_forward_storage_capacity_;
  const std::size_t active_levels = runtime_->hierarchy().num_levels();
  if (capacity.level_count == 0 || active_levels == 0 || active_levels > capacity.level_count)
    throw std::logic_error(
        "AMR Program live accepted-effect envelope exceeds its configured hierarchy");

  auto& staging = accepted_state_staging_;
  if (!staging.prepared_envelope || staging.configured_level_count != capacity.level_count)
    throw std::logic_error(
        "AMR Program live accepted-effect envelope has no authenticated staging capacity");

  // Return any logical A image to its own resident pool while that pool is still self-owned.
  // In particular, this clears the interface serialization views before B replaces the ledger
  // pointers which they once borrowed.  It never visits the adapter-owned A ordinals.
  reset_accepted_state_staging_for_cold_prime_();

  decltype(accepted_face_flux_commit_slots_) next_face_slots;
  multiblock_subcycling_->bind_candidate_ledger_slots([&](std::size_t, std::size_t, std::size_t,
                                                          multiblock_flux_ledger_type& ledger) {
    if (!ledger.resident_slots_bound())
      throw std::logic_error("AMR Program live accepted-effect ledger slots were not cold-bound");
    for (int axis = 0; axis < Dim; ++axis) {
      auto& destination = next_face_slots[static_cast<std::size_t>(axis)];
      const auto templates = ledger.resident_slot_templates(axis);
      if (templates.size() > std::numeric_limits<std::size_t>::max() - destination.size())
        throw std::length_error("AMR Program live accepted-effect face slots exceed size_t");
      destination.insert(destination.end(), templates.begin(), templates.end());
    }
  });

  std::size_t prepared_face_slots = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    auto& slots = next_face_slots[static_cast<std::size_t>(axis)];
    std::sort(slots.begin(), slots.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
    if (slots.size() > std::numeric_limits<std::size_t>::max() - prepared_face_slots)
      throw std::length_error("AMR Program live accepted-effect face slots exceed size_t");
    prepared_face_slots += slots.size();
  }
  if (prepared_face_slots > capacity.face_fragment_count)
    throw std::logic_error(
        "AMR Program live accepted-effect face slots exceed their configured capacity");

  if (multiblock_subcycling_block_count_ > 0 && capacity.level_count > 1 &&
      multiblock_subcycling_block_count_ >
          std::numeric_limits<std::size_t>::max() / (capacity.level_count - 1U) / 2U)
    throw std::overflow_error("AMR Program live synchronization slots exceed size_t");
  const std::size_t transitions = capacity.level_count - 1U;
  const std::size_t event_count = multiblock_subcycling_block_count_ * transitions * 2U;
  if (event_count != capacity.synchronization_event_count)
    throw std::logic_error(
        "AMR Program live synchronization slots differ from their configured capacity");
  std::vector<AmrProgramSynchronizationEvent> next_event_slots;
  next_event_slots.reserve(event_count);
  for (std::size_t block = 0; block < multiblock_subcycling_block_count_; ++block)
    for (std::size_t parent = 0; parent < transitions; ++parent)
      for (const std::string_view phase :
           {std::string_view("reflux"), std::string_view("average_down")})
        next_event_slots.push_back({static_cast<int>(parent),
                                    static_cast<int>(parent + 1U),
                                    static_cast<int>(block),
                                    std::string(phase),
                                    {}});

  // Build every replacement carrier before exchanging any A-owned slot.  The staging image and
  // accepted commit image intentionally own separate copies: the former feeds serialization,
  // the latter feeds the no-throw accepted-state commit.
  decltype(staging.accepted_face_flux_slots) next_staging_face_slots = next_face_slots;
  decltype(accepted_face_flux_) next_accepted_face_flux;
  decltype(staging.state.accepted_face_flux) next_staging_face_flux;
  decltype(staging.accepted_face_flux_sources) next_staging_face_sources;
  decltype(staging.accepted_face_flux_active_slots) next_staging_face_active_slots;
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    const std::size_t slots = next_face_slots[index].size();
    next_accepted_face_flux[index].reserve(slots);
    next_staging_face_flux[index].reserve(slots);
    next_staging_face_sources[index].reserve(slots);
    next_staging_face_active_slots[index].reserve(slots);
  }
  decltype(staging.synchronization_event_slots) next_staging_event_slots = next_event_slots;
  decltype(accepted_synchronization_events_) next_accepted_synchronization_events;
  decltype(staging.state.synchronization_events) next_staging_synchronization_events;
  decltype(staging.synchronization_event_active_indices) next_staging_event_active_indices;
  next_accepted_synchronization_events.reserve(event_count);
  next_staging_synchronization_events.reserve(event_count);
  next_staging_event_active_indices.reserve(event_count);

  static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_commit_slots_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
  static_assert(
      std::is_nothrow_swappable_v<decltype(accepted_synchronization_event_commit_slots_)>);
  static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
  using std::swap;
  swap(accepted_face_flux_commit_slots_, next_face_slots);
  swap(accepted_face_flux_, next_accepted_face_flux);
  swap(staging.accepted_face_flux_slots, next_staging_face_slots);
  swap(staging.state.accepted_face_flux, next_staging_face_flux);
  swap(staging.accepted_face_flux_sources, next_staging_face_sources);
  swap(staging.accepted_face_flux_active_slots, next_staging_face_active_slots);
  swap(accepted_synchronization_event_commit_slots_, next_event_slots);
  swap(accepted_synchronization_events_, next_accepted_synchronization_events);
  swap(staging.synchronization_event_slots, next_staging_event_slots);
  swap(staging.state.synchronization_events, next_staging_synchronization_events);
  swap(staging.synchronization_event_active_indices, next_staging_event_active_indices);

  // The former ordinal arenas may still contain A pointers.  Discard those values before the
  // cold bind below can inspect B; their capacities remain resident for the rebind.
  for (auto& ordinals : accepted_face_flux_ordinals_)
    ordinals.clear();
  accepted_face_flux_ordinal_owner_ = nullptr;
  accepted_face_flux_ordinal_epoch_ = std::numeric_limits<std::uint64_t>::max();
  accepted_face_flux_ordinal_generation_ = std::numeric_limits<std::uint64_t>::max();
  accepted_interface_flux_wire_ordinals_.clear();
  accepted_interface_flux_ordinal_owner_ = nullptr;
  accepted_interface_flux_ordinal_epoch_ = std::numeric_limits<std::uint64_t>::max();
  accepted_interface_flux_ordinal_generation_ = std::numeric_limits<std::uint64_t>::max();
  staging.topology_epoch = resource_epoch_;
  staging.materialization_generation = resource_generation_;
  staging.prepared_envelope = true;
  staging.primed = false;
  staging.valid = false;
}

/// Build the complete mutable POPSAND5 envelope while the adapter still owns only detached
/// topology.  The configured checkpoint shape is the authority for variable interface/event
/// storage; resident face-ledger templates supply the exact per-axis routing and identity shape.
/// No accepted facade or published checkpoint is observed here.
void prime_accepted_state_staging_envelope_from_prepared_capacity_() const {
  if (preparation_view_ == nullptr || preparation_view_->runtime == nullptr ||
      preparation_view_->candidate_multiblock == nullptr ||
      preparation_view_->candidate_accepted_state_staging_capacity == nullptr ||
      !multiblock_subcycling_)
    throw std::logic_error("AMR Program staging envelope has no detached capacity authority");
  const auto& capacity = *preparation_view_->candidate_accepted_state_staging_capacity;
  const auto& interface_budget = *preparation_view_->candidate_interface_flux_ledger_budget;
  const std::size_t active_levels = preparation_view_->runtime->hierarchy().num_levels();
  if (active_levels == 0 || active_levels > capacity.level_count ||
      capacity.interface_fragment_count != interface_budget.max_fragments_per_window ||
      capacity.interface_payload_terms != interface_budget.max_payload_terms_per_window)
    throw std::logic_error(
        "AMR Program staging envelope differs from its prepared capacity authority");

  const auto reserve_string = [](std::string& value, std::size_t bound) {
    value.reserve(bound);
    value.clear();
  };
  auto& staging = accepted_state_staging_;
  staging.valid = false;
  staging.primed = false;
  staging.configured_level_count = capacity.level_count;
  staging.state = {};
  reserve_string(staging.state.spatial_contract, capacity.spatial_contract_characters);
  staging.state.level_clocks.reserve(capacity.level_count);
  staging.state.level_clocks.resize(active_levels);
  staging.state.logical_clock_ticks.clear();
  for (const std::string& identity : capacity.logical_clock_identities)
    staging.state.logical_clock_ticks.emplace(identity, 0);
  staging.state.histories.reserve(capacity.histories.size());
  std::size_t history_slots = 0;
  for (const auto& descriptor : capacity.histories) {
    if (descriptor.depth < 0 ||
        static_cast<std::size_t>(descriptor.depth) >
            (std::numeric_limits<std::size_t>::max() - history_slots) / capacity.level_count)
      throw std::length_error("AMR Program staging history-slot capacity exceeds size_t");
    history_slots += static_cast<std::size_t>(descriptor.depth) * capacity.level_count;
    staging.state.histories.push_back(descriptor);
  }
  staging.history_slot_pool.resize(history_slots);
  staging.history_slot_bindings.resize(history_slots);
  std::size_t history_slot = 0;
  for (const auto& descriptor : capacity.histories)
    for (std::size_t level = 0; level < capacity.level_count; ++level)
      for (int slot = 0; slot < descriptor.depth; ++slot) {
        auto& value = staging.history_slot_pool.at(history_slot++);
        reserve_string(value.name, descriptor.name.size());
        value.name = descriptor.name;
        value.level = static_cast<int>(level);
        value.slot = slot;
      }
  std::sort(staging.history_slot_pool.begin(), staging.history_slot_pool.end(),
            [](const auto& left, const auto& right) {
              return std::tie(left.name, left.level, left.slot) <
                     std::tie(right.name, right.level, right.slot);
            });
  for (std::size_t index = 0; index < staging.history_slot_pool.size(); ++index) {
    const auto& slot = staging.history_slot_pool[index];
    auto& binding = staging.history_slot_bindings[index];
    binding.key = "pops.amr.level-history.v1/" + std::to_string(slot.level) + "/" +
                  std::to_string(slot.name.size()) + ":" + slot.name;
    binding.state_slot = index;
    binding.source_slot = static_cast<std::size_t>(slot.slot);
  }
  staging.state.history_slots.reserve(history_slots);
  staging.state.history_slots.clear();
  staging.history_slot_active_indices.reserve(history_slots);
  staging.state.pending_history_remaps.resize(capacity.pending_history_remap_count);
  for (auto& remap : staging.state.pending_history_remaps)
    reserve_string(remap.key, capacity.pending_history_remap_key_characters);
  staging.pending_history_remap_slots.resize(capacity.pending_history_remap_count);
  for (auto& remap : staging.pending_history_remap_slots)
    reserve_string(remap.key, capacity.pending_history_remap_key_characters);
  staging.pending_history_keys.resize(capacity.pending_history_remap_count);
  std::size_t pending_index = 0;
  for (const auto& descriptor : capacity.histories)
    for (std::size_t child = 1; child < capacity.level_count; ++child) {
      if (pending_index == staging.pending_history_keys.size())
        throw std::logic_error("AMR Program detached pending-remap slots exceed their capacity");
      auto& key = staging.pending_history_keys[pending_index++];
      reserve_string(key, capacity.pending_history_remap_key_characters);
      key = "pops.amr.level-history.v1/" + std::to_string(child) + "/" +
            std::to_string(descriptor.name.size()) + ":" + descriptor.name;
      if (key.size() > capacity.pending_history_remap_key_characters)
        throw std::logic_error("AMR Program detached pending-remap key exceeds its capacity");
    }
  if (pending_index != staging.pending_history_keys.size())
    throw std::logic_error("AMR Program detached pending-remap shape differs from its capacity");
  staging.pending_history_active_slots.reserve(capacity.pending_history_remap_count);
  // Logical pending remaps begin empty; their separately retained slot pool keeps the configured
  // key capacities available for the first 0 -> N transition.
  staging.state.pending_history_remaps.clear();
  staging.pending_history_active_slots.reserve(capacity.pending_history_remap_count);
  staging.state.history_flux_payload.reserve(capacity.history_flux_payload_bytes);
  reserve_string(staging.state.temporal_partition.provider_identity,
                 capacity.temporal_provider_identity.size());
  staging.state.temporal_partition.cells.reserve(capacity.temporal_cell_count);
  staging.state.tagging_hysteresis_state.reserve(capacity.tagging_hysteresis_bytes);
  reserve_string(staging.state.flux_budget_contract, capacity.flux_budget_contract_characters);
  reserve_string(staging.state.coupling_contract, capacity.coupling_contract_characters);
  accepted_checkpoint_candidate_bytes_.reserve(capacity.checkpoint_byte_capacity);
  accepted_checkpoint_candidate_bytes_.clear();
  accepted_checkpoint_level_clock_slots_.reserve(capacity.level_count);
  accepted_checkpoint_level_clock_slots_.clear();

  std::size_t prepared_face_slots = 0;
  std::array<std::vector<typename multiblock_flux_ledger_type::Entry>, Dim>
      prepared_face_slot_images;
  multiblock_subcycling_->bind_candidate_ledger_slots(
      [&](std::size_t, std::size_t, std::size_t, multiblock_flux_ledger_type& ledger) {
        for (int axis = 0; axis < Dim; ++axis) {
          auto& destination = prepared_face_slot_images[static_cast<std::size_t>(axis)];
          const auto templates = ledger.resident_slot_templates(axis);
          if (templates.size() > std::numeric_limits<std::size_t>::max() - destination.size())
            throw std::length_error("AMR Program staging face-slot envelope exceeds size_t");
          destination.insert(destination.end(), templates.begin(), templates.end());
        }
      });
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t axis_index = static_cast<std::size_t>(axis);
    auto& slots = staging.accepted_face_flux_slots[axis_index];
    slots.swap(prepared_face_slot_images[axis_index]);
    std::sort(slots.begin(), slots.end(),
              [](const auto& left, const auto& right) { return left.key < right.key; });
    if (slots.size() > std::numeric_limits<std::size_t>::max() - prepared_face_slots)
      throw std::length_error("AMR Program staging face-slot capacity exceeds size_t");
    prepared_face_slots += slots.size();
    auto& state_axis = staging.state.accepted_face_flux[axis_index];
    state_axis.reserve(slots.size());
    staging.accepted_face_flux_sources[axis_index].reserve(slots.size());
    staging.accepted_face_flux_active_slots[axis_index].reserve(slots.size());
    accepted_face_flux_commit_slots_[axis_index] = slots;
    accepted_face_flux_[axis_index].reserve(slots.size());
  }
  if (prepared_face_slots > capacity.face_fragment_count)
    throw std::logic_error(
        "AMR Program prepared face slots exceed the authenticated checkpoint bound");

  auto& interface_slots = staging.accepted_interface_flux_slots;
  interface_slots.clear();
  staging.state.accepted_interface_flux.reserve(capacity.interface_fragment_count);
  accepted_interface_flux_staging_sources_.reserve(capacity.interface_fragment_count);

  staging.synchronization_event_slots.resize(capacity.synchronization_event_count);
  for (auto& event : staging.synchronization_event_slots)
    reserve_string(event.phase, capacity.synchronization_phase_characters);
  const std::size_t configured_event_count =
      preparation_view_->candidate_multiblock->block_count() * (capacity.level_count - 1U) * 2U;
  if (configured_event_count != staging.synchronization_event_slots.size())
    throw std::logic_error("AMR Program staging synchronization events exceed their capacity");
  for (std::size_t block = 0, index = 0;
       block < preparation_view_->candidate_multiblock->block_count(); ++block)
    for (std::size_t parent = 0; parent + 1 < capacity.level_count; ++parent) {
      for (const std::string_view phase :
           {std::string_view("reflux"), std::string_view("average_down")}) {
        auto& event = staging.synchronization_event_slots[index++];
        event.parent_level = static_cast<int>(parent);
        event.child_level = static_cast<int>(parent + 1);
        event.runtime_block = static_cast<int>(block);
        event.phase.assign(phase);
      }
    }
  staging.state.synchronization_events.reserve(capacity.synchronization_event_count);
  staging.state.synchronization_events.clear();
  staging.synchronization_event_active_indices.reserve(capacity.synchronization_event_count);
  accepted_synchronization_event_commit_slots_ = staging.synchronization_event_slots;
  accepted_synchronization_events_.reserve(capacity.synchronization_event_count);
  staging.topology_epoch = preparation_view_->topology_epoch;
  staging.materialization_generation = preparation_view_->materialization_generation;
  staging.prepared_envelope = true;
}

void prime_prepared_subcycling_engine() const {
  if (!preparation_mode_ || preparation_view_ == nullptr || facade_ != nullptr)
    throw std::logic_error(
        "AMR Program detached subcycling prime requires an unactivated preparation image");
  if (preparation_view_->candidate_accepted_state_staging_capacity == nullptr)
    throw std::logic_error("AMR Program detached subcycling prime has no forward storage capacity");
  // The topology view borrows the host's installation-local capacity only through this cold
  // prime.  Retain an immutable copy before the installation stack unwinds; accepted snapshots
  // and forward candidates subsequently share this control block and never dereference the view.
  accepted_forward_storage_capacity_ =
      std::make_shared<const AmrProgramAcceptedStateStagingCapacity<Dim>>(
          *preparation_view_->candidate_accepted_state_staging_capacity);
  if (auto bundle = prepare_multiblock_subcycling_bundle_from_authority_(
          detached_subcycling_authority_(), static_flux_tables_, hot_path_workspace_,
          primary_clock_);
      bundle.engine) {
    auto history_flux = prepare_static_history_flux_provenance_at_bind_(bundle.flux_tables,
                                                                        bundle.flux_basis_payloads);
    publish_prepared_subcycling_bundle_noexcept(std::move(bundle));
    static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
    history_flux_expressions_.swap(history_flux);
  }
  prime_accepted_state_staging_envelope_from_prepared_capacity_();
  prime_history_mutation_workspace_at_bind_();
  rebind_accepted_face_flux_ordinals_at_cold_prime_();
  rebind_accepted_interface_flux_ordinals_at_cold_prime_();
}

void prepare_multiblock_subcycling_engine_() const {
  if (preparation_view_ != nullptr || accepted_runtime_state_ == nullptr)
    throw std::logic_error(
        "AMR Program live subcycling preparation has no stable accepted runtime-state authority");
  const auto accepted_effect_slots_match_current = [&] {
    const auto& staging = accepted_state_staging_;
    if (!staging.prepared_envelope || staging.topology_epoch != resource_epoch_ ||
        staging.materialization_generation != resource_generation_ ||
        accepted_face_flux_ordinal_owner_ != multiblock_subcycling_.get() ||
        accepted_face_flux_ordinal_epoch_ != resource_epoch_ ||
        accepted_face_flux_ordinal_generation_ != resource_generation_ ||
        accepted_interface_flux_ordinal_owner_ != interface_flux_ledger_.get() ||
        accepted_interface_flux_ordinal_epoch_ != resource_epoch_ ||
        accepted_interface_flux_ordinal_generation_ != resource_generation_ ||
        accepted_synchronization_event_commit_slots_.size() !=
            staging.synchronization_event_slots.size() ||
        accepted_synchronization_events_.capacity() <
            accepted_synchronization_event_commit_slots_.size())
      return false;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto index = static_cast<std::size_t>(axis);
      if (accepted_face_flux_commit_slots_[index].size() !=
              staging.accepted_face_flux_slots[index].size() ||
          accepted_face_flux_[index].capacity() < accepted_face_flux_commit_slots_[index].size() ||
          accepted_face_flux_ordinals_[index].size() !=
              staging.accepted_face_flux_slots[index].size())
        return false;
    }
    return true;
  };

  // A current engine is usable only with the matching B effect envelope and non-owning ordinal
  // image.  A raw topology publish can leave those carriers at A even when the scalar engine
  // witnesses have already converged, so cold-rebuild them before a pre-attempt consumer can
  // dereference an ordinal.
  if (multiblock_subcycling_ != nullptr && runtime_ != nullptr &&
      multiblock_subcycling_epoch_ == runtime_->topology_epoch() &&
      multiblock_subcycling_generation_ == runtime_->materialization_generation() &&
      multiblock_subcycling_block_count_ != 0 &&
      prepared_rhs_basis_bounds_.size() == multiblock_subcycling_block_count_ &&
      prepared_coefficient_term_bounds_.size() == multiblock_subcycling_block_count_ &&
      !multiblock_subcycling_program_budget_contract_.empty()) {
    if (!accepted_effect_slots_match_current()) {
      rebuild_live_accepted_effect_slots_from_published_bundle_cold_();
      prime_history_mutation_workspace_at_bind_();
    }
    // A topology-static restore can exchange a cold-prepared staging image whose logical
    // bootstrap bit is false, even though every resident slot and ordinal still names the
    // current engine.  That requires only the preallocated prime below; treating it as a B
    // topology mismatch would rebuild the complete effect envelope on the first hot step.
    if (!accepted_state_staging_.primed)
      prime_accepted_state_staging_at_bind_();
    // A snapshot restore swaps the accepted resource image without copying these adapter-owned
    // non-owning history ordinals.  The engine itself can therefore still be current while its
    // history workspace names the prior image.  Repair only at this pre-attempt cold boundary;
    // store_history_/rotate_histories_ remain fail-closed hot consumers.
    const auto& history_manager = accepted_runtime_state_->hist_;
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
    return;
  }
  auto bundle = prepare_multiblock_subcycling_bundle_from_authority_(
      accepted_subcycling_authority_(), static_flux_tables_, hot_path_workspace_, primary_clock_);
  if (!bundle.engine)
    throw std::logic_error("AMR Program live subcycling preparation produced no bundle");
  auto history_flux = prepare_static_history_flux_provenance_at_bind_(bundle.flux_tables,
                                                                      bundle.flux_basis_payloads);
  publish_prepared_subcycling_bundle_noexcept(std::move(bundle));
  rebuild_live_accepted_effect_slots_from_published_bundle_cold_();
  static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
  history_flux_expressions_.swap(history_flux);
  prime_history_mutation_workspace_at_bind_();
  prime_accepted_state_staging_at_bind_();
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
  const auto require_active_group_capacity = [&](const auto& storage) {
    return storage.capacity() >= blocks;
  };
  if (!require_active_group_capacity(active_attempt_states_) ||
      !require_active_group_capacity(active_staged_parents_) ||
      !require_active_group_capacity(active_incoming_flux_) ||
      !require_active_group_capacity(active_outgoing_flux_) ||
      !require_active_group_capacity(active_block_identities_) ||
      !require_active_group_capacity(active_flux_basis_counts_))
    throw std::logic_error(
        "AMR Program active level-group arena was not primed during candidate preparation");
  active_attempt_states_.resize(blocks);
  active_staged_parents_.resize(blocks);
  active_incoming_flux_.resize(blocks);
  active_outgoing_flux_.resize(blocks);
  active_block_identities_.resize(blocks);
  active_flux_basis_counts_.resize(blocks);
  std::fill(active_attempt_states_.begin(), active_attempt_states_.end(), nullptr);
  std::fill(active_staged_parents_.begin(), active_staged_parents_.end(), nullptr);
  std::fill(active_incoming_flux_.begin(), active_incoming_flux_.end(), nullptr);
  std::fill(active_outgoing_flux_.begin(), active_outgoing_flux_.end(), nullptr);
  std::fill(active_block_identities_.begin(), active_block_identities_.end(), std::string_view{});
  std::fill(active_flux_basis_counts_.begin(), active_flux_basis_counts_.end(), std::size_t{0});
  active_flux_expressions_.clear();
  reset_static_flux_active_state_();
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
  facade_->program_clear_prepared_amr_level_evaluations_();
  try {
    // A static carrier must agree on its complete helper route before one rank can enter a
    // per-face SUM/payload collective.  The witness was assembled only during detached prime;
    // this hot preflight borrows its resident bytes and performs no route/table allocation.
    if (static_flux_tables_.bound) {
      if (static_flux_collective_contract_.empty())
        throw std::logic_error("AMR Program static flux carrier has no helper preflight witness");
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{"amr-program-static-flux-helper-preflight", static_flux_collective_contract_}},
              prepared_execution_lane()))
        throw std::invalid_argument(
            "AMR Program static flux helper routes differ between execution ranks");
    }
    body(current_dt_);
    // Cell-local FE has already committed its finite, provider-owned face diagnostics.  It is
    // explicitly refused at preparation when interface/coupling flux authority is requested, so
    // constructing the generic symbolic registry here would be both a second authority and a
    // hot map/string allocation.  Ordinary AMR Program routes retain the existing materializer.
    if (!active_cell_temporal_execution_)
      for (std::size_t block = 0; block < blocks; ++block)
        materialize_active_flux_expression_(block, *active_attempt_states_[block]);
  } catch (...) {
    clear_active_multiblock_group_();
    throw;
  }
  clear_active_multiblock_group_();
}

// The exact interface-route builder and its paired payload collectors are separated from the
// subcycling state machine.  It remains a class-definition fragment, so this include introduces
// no runtime dispatch or compatibility surface.
// clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_subcycling_interface_payload.hpp>
// clang-format on
