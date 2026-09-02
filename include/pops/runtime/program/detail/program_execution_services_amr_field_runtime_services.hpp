field_type* live_attempt_state_(int runtime_block, int level) const {
  if (runtime_block < 0 || level < 0)
    return nullptr;
  if (multiblock_subcycling_ != nullptr && multiblock_subcycling_->has_attempt_candidates())
    return &multiblock_subcycling_->attempt_state(static_cast<std::size_t>(runtime_block),
                                                  static_cast<std::size_t>(level));
  if (!active_attempt_states_.empty() && level == active_level_)
    return active_attempt_states_.at(static_cast<std::size_t>(runtime_block));
  return nullptr;
}

bool is_live_attempt_candidate_(const field_type* field) const {
  if (field == nullptr)
    return false;
  if (multiblock_subcycling_ != nullptr && multiblock_subcycling_->has_attempt_candidates()) {
    const int blocks = n_blocks();
    const int levels = nlev();
    for (int block = 0; block < blocks; ++block)
      for (int level = 0; level < levels; ++level)
        if (live_attempt_state_(block, level) == field)
          return true;
    return false;
  }
  if (active_attempt_states_.empty())
    return false;
  return std::find(active_attempt_states_.begin(), active_attempt_states_.end(), field) !=
         active_attempt_states_.end();
}

void require_history_free_for_topology_change_(std::string_view operation) const {
  // The adapter's level index is the fast path while a detached Program image is being
  // assembled.  Once the image has been bound to the accepted facade, the authoritative
  // registry is ProgramRuntimeState::hist_: an artifact replacement can otherwise leave this
  // adapter's retained index empty while exact-ranked rings are already live.  Topology must be
  // rejected before handing a PreparedRegrid to AmrRuntime; publishing first would invalidate the
  // active transaction image and turn an ordinary refusal into a fail-stop rollback.
  const bool accepted_history_is_registered =
      facade_ != nullptr && !runtime_state().hist_.histories.empty();
  if (!history_levels_.empty() || accepted_history_is_registered)
    throw std::runtime_error(
        "AmrStorageTopologyAdapter cannot " + std::string(operation) +
        " while exact-ranked history rings lack a prepared rematerialization transaction");
}
[[noreturn]] static void unavailable_(std::string_view provider) {
  throw std::runtime_error("AmrStorageTopologyAdapter has no prepared " + std::string(provider));
}
/// Provider-owned physical law used by both the authenticated build request and the generated
/// flat Krylov boundary session. Keep one retained instance per prepared level; consumers never
/// reconstruct this law from topology alone.
PhysicalBoundaryConditions<Dim> hierarchy_tensor_boundary_(const Geometry<Dim>& geometry) const {
  const BoundaryTopology<Dim> topology = prepared_boundary_topology_();
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      if (topology.is_physical(face))
        faces[static_cast<std::size_t>(face.ordinal())] = {PhysicalBoundaryKind::dirichlet, Real(0),
                                                           Real(1), Real(0)};
    }
  }
  return PhysicalBoundaryConditions<Dim>{topology, faces, spacing};
}

PreparedHierarchyTensorState prepare_hierarchy_tensor_solver_(
    const HierarchyTensorSelection& selection) const {
  hierarchy_tensor_request_type request;
  std::vector<HierarchyTensorLevelBoundary> boundaries;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    const int runtime_block = sys_block(selection.program_block);
    if (runtime_block < 0)
      throw std::invalid_argument("AMR hierarchy tensor solver has an invalid runtime block");
    request.block = static_cast<std::size_t>(runtime_block);
    request.components = selection.components;
    request.plan_identity = selection.plan_identity;
    request.operator_contract_identity = selection.operator_contract_identity;
    request.assembly_field_slots = selection.assembly_field_slots;
    request.solution_field_slot = selection.solution_field_slot;
    request.options = selection.options;
    const std::size_t levels = preparation_view_ != nullptr
                                   ? preparation_view_->level_geometries.size()
                                   : runtime_->hierarchy().num_levels();
    request.levels.reserve(levels);
    boundaries.reserve(levels);
    if (levels > 1)
      request.ratios.reserve(levels - 1U);
    for (std::size_t level = 0; level < levels; ++level) {
      const field_type& level_state =
          preparation_view_ != nullptr
              ? preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block))
                    .at(level)
              : runtime_->hierarchy().state(level);
      const Geometry<Dim> level_geometry =
          preparation_view_ != nullptr
              ? preparation_view_->level_geometries.at(level)
              : facade_->program_prepared_amr_level_geometry_(static_cast<int>(level));
      const PhysicalBoundaryConditions<Dim> boundary = hierarchy_tensor_boundary_(level_geometry);
      request.levels.push_back({level_geometry, boundary, level_state.layout(),
                                level_state.distribution(), level_state.local_rank()});
      boundaries.push_back({level_geometry, boundary});
      if (level != 0)
        request.ratios.push_back(preparation_view_ != nullptr
                                     ? preparation_view_->spatial_refinement_ratios.at(level - 1U)
                                     : runtime_->hierarchy().layout(level).ratio_from_parent());
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_failure, lane) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(
        "AMR hierarchy tensor request construction failed on another MPI rank");
  }
  return {prepare_hierarchy_tensor_solver_collectively(*hierarchy_tensor_solver_registry_,
                                                       selection.provider_identity,
                                                       std::move(request), lane),
          std::move(boundaries)};
}

hierarchy_tensor_solver_type& configured_hierarchy_tensor_solver_() const {
  if (!hierarchy_tensor_selection_)
    throw std::logic_error(
        "AMR hierarchy tensor solver must be configured before hierarchy access");
  if (preparation_view_ == nullptr)
    refresh_resources_();
  const std::uint64_t topology_epoch =
      preparation_view_ != nullptr ? preparation_view_->topology_epoch : runtime_->topology_epoch();
  const std::uint64_t materialization_generation =
      preparation_view_ != nullptr ? preparation_view_->materialization_generation
                                   : runtime_->materialization_generation();
  if (!hierarchy_tensor_solver_ || hierarchy_tensor_topology_epoch_ != topology_epoch ||
      hierarchy_tensor_materialization_generation_ != materialization_generation) {
    PreparedHierarchyTensorState prepared =
        prepare_hierarchy_tensor_solver_(*hierarchy_tensor_selection_);
    hierarchy_tensor_solver_ = std::move(prepared.solver);
    hierarchy_tensor_boundaries_ = std::move(prepared.boundaries);
    hierarchy_tensor_topology_epoch_ = topology_epoch;
    hierarchy_tensor_materialization_generation_ = materialization_generation;
  }
  return *hierarchy_tensor_solver_;
}

void require_hierarchy_tensor_binding_(int program_block, int components) const {
  if (!hierarchy_tensor_selection_ || hierarchy_tensor_selection_->program_block != program_block ||
      hierarchy_tensor_selection_->components != components)
    throw std::logic_error(
        "AMR hierarchy tensor block/component binding differs from its prepared solver");
}

void synchronize_resource_generation_() const {
  if (interface_flux_ledger_ &&
      interface_flux_ledger_->topology_epoch() != runtime_->topology_epoch())
    interface_flux_commit_guard_.reset();
  prepare_coupled_jacvec_scratch_();
  if (!interface_flux_ledger_) {
    interface_flux_ledger_ = std::make_unique<interface_flux_ledger_type>(
        runtime_->topology_epoch(), inactive_interface_flux_budget_());
  } else {
    interface_flux_ledger_->advance_topology_epoch(runtime_->topology_epoch());
  }
  resource_epoch_ = runtime_->topology_epoch();
  resource_generation_ = runtime_->materialization_generation();
  std::map<std::string, int> indexed_histories;
  if (facade_ != nullptr) {
    for (const auto& [key, ring] : runtime_state().hist_.histories) {
      (void)ring;
      const auto decoded = decode_history_key_(key);
      if (!decoded ||
          static_cast<std::size_t>(decoded->first) >= runtime_->hierarchy().num_levels())
        throw std::runtime_error(
            "AMR Program history registry is not qualified by the live hierarchy");
      indexed_histories.emplace(key, decoded->first);
    }
  }
  history_levels_.swap(indexed_histories);
  if (history_levels_.empty()) {
    history_epoch_ = std::numeric_limits<std::uint64_t>::max();
    history_generation_ = std::numeric_limits<std::uint64_t>::max();
  } else {
    history_epoch_ = resource_epoch_;
    history_generation_ = resource_generation_;
  }
}
void refresh_resources_() const {
  if (facade_ != nullptr) {
    runtime_type* const live_runtime = require_runtime_(*facade_);
    if (runtime_ != live_runtime) {
      if (!active_attempt_states_.empty())
        throw std::logic_error(
            "AMR Program hierarchy refresh cannot rebind a runtime during an active attempt");
      runtime_ = live_runtime;
    }
    if (!preparation_mode_)
      facade_->refresh_prepared_amr_levels();
  }
  if (resource_epoch_ == runtime_->topology_epoch() &&
      resource_generation_ == runtime_->materialization_generation())
    return;
  if (!history_levels_.empty() && (history_epoch_ != runtime_->topology_epoch() ||
                                   history_generation_ != runtime_->materialization_generation())) {
    // A fresh restart bind registers its level-zero ring before the Python restart transaction
    // materializes the checkpoint's complete all-level history image.  That empty registration
    // has no accepted numerical or flux provenance and may follow the raw hierarchy rebuild.
    // Any stored/cold-filled/pending or flux-authenticated sample remains a hard refusal: only
    // the native restart materializer may replace it.
    const auto& manager = runtime_state().hist_;
    std::map<std::string, std::vector<FluxExpression>> provisional_flux;
    std::exception_ptr provisional_error;
    try {
      const std::size_t live_levels = runtime_->hierarchy().num_levels();
      std::map<std::string, std::set<int>> levels_by_name;
      std::map<std::string,
               std::tuple<int, int, int, std::string, std::string, std::string, std::string>>
          descriptors;
      const auto exact_metadata = [&](const auto& metadata) {
        return metadata.size() == manager.histories.size() &&
               std::all_of(manager.histories.begin(), manager.histories.end(),
                           [&](const auto& entry) { return metadata.contains(entry.first); });
      };
      if (manager.histories.empty() || live_levels == 0 || !exact_metadata(manager.depth) ||
          !exact_metadata(manager.initialized) || !exact_metadata(manager.fill_count) ||
          !exact_metadata(manager.store_pending) || !exact_metadata(manager.slot_dt) ||
          !exact_metadata(manager.owner) || !exact_metadata(manager.state_identity) ||
          !exact_metadata(manager.space_identity) || !exact_metadata(manager.clock_identity) ||
          !exact_metadata(manager.interpolation_identity))
        throw std::runtime_error(
            "AMR Program topology changed while retained histories still name the prior layouts");
      for (const auto& [key, ring] : manager.histories) {
        const auto decoded = decode_history_key_(key);
        const auto retained = history_flux_expressions_.find(key);
        if (!decoded || decoded->first < 0 ||
            static_cast<std::size_t>(decoded->first) >= live_levels || ring.empty() ||
            manager.depth.at(key) != static_cast<int>(ring.size()) || manager.initialized.at(key) ||
            manager.fill_count.at(key) != 0 || manager.store_pending.at(key) ||
            manager.slot_dt.at(key).size() != ring.size() ||
            !std::all_of(
                manager.slot_dt.at(key).begin(), manager.slot_dt.at(key).end(),
                [](Real dt) { return dt == Real(0); }) ||
            manager.owner.at(key) < 0 || manager.state_identity.at(key).empty() ||
            manager.space_identity.at(key).empty() || manager.clock_identity.at(key).empty() ||
            manager.interpolation_identity.at(key).empty() ||
            !std::all_of(
                ring.begin(), ring.end(),
                [&](const field_type& slot) { return slot.ncomp() == ring.front().ncomp(); }) ||
            (retained != history_flux_expressions_.end() &&
             (retained->second.size() != ring.size() ||
              !std::all_of(
                  retained->second.begin(), retained->second.end(),
                  [](const FluxExpression& expression) { return expression.empty(); }))))
          throw std::runtime_error(
              "AMR Program topology changed while retained histories still name the prior layouts");
        const auto descriptor =
            std::make_tuple(manager.owner.at(key), manager.depth.at(key), ring.front().ncomp(),
                            manager.state_identity.at(key), manager.space_identity.at(key),
                            manager.clock_identity.at(key), manager.interpolation_identity.at(key));
        auto [descriptor_it, inserted] = descriptors.emplace(decoded->second, descriptor);
        if (!inserted && descriptor_it->second != descriptor)
          throw std::runtime_error(
              "AMR Program topology changed while retained histories have mixed provisional "
              "descriptors");
        levels_by_name[decoded->second].insert(decoded->first);
        provisional_flux.emplace(key, std::vector<FluxExpression>(ring.size()));
      }
      for (const auto& [name, levels] : levels_by_name) {
        (void)name;
        if (levels.size() != live_levels)
          throw std::runtime_error(
              "AMR Program topology changed while retained histories still name the prior layouts");
        for (std::size_t level = 0; level < live_levels; ++level)
          if (!levels.contains(static_cast<int>(level)))
            throw std::runtime_error(
                "AMR Program topology changed while retained histories still name the prior "
                "layouts");
      }
      for (const auto& [key, level] : history_levels_)
        if (!manager.histories.contains(key) || level != decode_history_key_(key)->first)
          throw std::runtime_error(
              "AMR Program topology changed while retained histories still name the prior layouts");
      for (const auto& [key, expressions] : history_flux_expressions_)
        if (!manager.histories.contains(key) ||
            expressions.size() != manager.histories.at(key).size() ||
            !std::all_of(expressions.begin(), expressions.end(),
                         [](const FluxExpression& expression) { return expression.empty(); }))
          throw std::runtime_error(
              "AMR Program topology changed while retained histories still name the prior layouts");
    } catch (...) {
      provisional_error = std::current_exception();
    }
    rethrow_accepted_history_remap_collective_failure_(provisional_error,
                                                       prepared_execution_lane());
    history_flux_expressions_.swap(provisional_flux);
  }
  scratches_.clear();
  std::exception_ptr synchronization_error;
  try {
    synchronize_resource_generation_();
  } catch (...) {
    synchronization_error = std::current_exception();
  }
  // synchronize_resource_generation_ may allocate a ledger after its own prepared-lane work.
  // Converge that local tail failure before refresh_accepted_hierarchy_state_after_remap_ enters
  // its later temporal/subcycling collectives; the enclosing AcceptedSnapshot owns rollback.
  rethrow_accepted_history_remap_collective_failure_(synchronization_error,
                                                     prepared_execution_lane());
  if (active_level_ >= nlev())
    active_level_ = 0;
}

static Extent<Dim> uniform_ghosts_(int depth) {
  if (depth < 0)
    throw std::invalid_argument("AMR Program ghost depth must be non-negative");
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = depth;
  return result;
}

static field_type make_scratch_(const field_type& prototype, int ncomp, Extent<Dim> ghosts) {
  if (ncomp < 1)
    throw std::invalid_argument("AMR Program scratch requires positive components");
  field_type result(prototype.layout(), prototype.distribution(), prototype.local_rank(), ncomp,
                    ghosts);
  result.set_val(Real(0));
  return result;
}

struct CoupledJacvecLevelScratch {
  std::array<std::unique_ptr<field_type>, 2> residual;
  std::array<std::unique_ptr<field_type>, 2> coupled;
};

struct CoupledJacvecScratch {
  std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
  std::vector<CoupledJacvecLevelScratch> levels;
};

void prepare_coupled_jacvec_scratch_() const {
  if (facade_ == nullptr && preparation_view_ == nullptr)
    return;
  const std::uint64_t topology_epoch = runtime_->topology_epoch();
  const std::uint64_t materialization_generation = runtime_->materialization_generation();
  if (coupled_jacvec_scratch_ && coupled_jacvec_scratch_->topology_epoch == topology_epoch &&
      coupled_jacvec_scratch_->materialization_generation == materialization_generation)
    return;

  const ExecutionLane& lane = prepared_execution_lane();
  std::unique_ptr<CoupledJacvecScratch> candidate;
  std::exception_ptr preparation_error;
  try {
    const std::size_t block_count = preparation_view_ != nullptr
                                        ? preparation_view_->block_prototypes.size()
                                        : static_cast<std::size_t>(facade_->program_n_blocks_());
    if (block_count == 2) {
      candidate = std::make_unique<CoupledJacvecScratch>();
      candidate->topology_epoch = topology_epoch;
      candidate->materialization_generation = materialization_generation;
      candidate->levels.resize(runtime_->hierarchy().num_levels());
      for (std::size_t level = 0; level < candidate->levels.size(); ++level)
        for (int runtime_block = 0; runtime_block < 2; ++runtime_block) {
          const field_type& prototype =
              preparation_view_ != nullptr
                  ? preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block))
                        .at(level)
                  : facade_->program_prepared_amr_block_state_(runtime_block,
                                                               static_cast<int>(level));
          auto residual = std::make_unique<field_type>(prototype.layout(), prototype.distribution(),
                                                       prototype.local_rank(), prototype.ncomp(),
                                                       prototype.ghosts());
          auto coupled = std::make_unique<field_type>(prototype.layout(), prototype.distribution(),
                                                      prototype.local_rank(), prototype.ncomp(),
                                                      prototype.ghosts());
          residual->set_val(Real(0));
          coupled->set_val(Real(0));
          candidate->levels[level].residual[static_cast<std::size_t>(runtime_block)] =
              std::move(residual);
          candidate->levels[level].coupled[static_cast<std::size_t>(runtime_block)] =
              std::move(coupled);
        }
    }
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR coupled Jacobian scratch preparation failed collectively");
  }
  coupled_jacvec_scratch_ = std::move(candidate);
}

CoupledJacvecLevelScratch& require_coupled_jacvec_scratch_(
    int first_block, const field_type& first_state, const field_type& first_result,
    int second_block, const field_type& second_state, const field_type& second_result) const {
  if (!coupled_jacvec_scratch_ ||
      coupled_jacvec_scratch_->topology_epoch != runtime_->topology_epoch() ||
      coupled_jacvec_scratch_->materialization_generation !=
          runtime_->materialization_generation() ||
      active_level_ < 0 ||
      static_cast<std::size_t>(active_level_) >= coupled_jacvec_scratch_->levels.size())
    throw std::logic_error("AMR coupled Jacobian scratch is stale or unprepared");
  const int first_runtime = sys_block(first_block);
  const int second_runtime = sys_block(second_block);
  auto& level = coupled_jacvec_scratch_->levels[static_cast<std::size_t>(active_level_)];
  const auto require_block = [&](int runtime_block, const field_type& state,
                                 const field_type& result) {
    const std::size_t index = static_cast<std::size_t>(runtime_block);
    if (runtime_block < 0 || runtime_block >= 2 || !level.residual[index] || !level.coupled[index])
      throw std::logic_error("AMR coupled Jacobian block scratch is incomplete");
    const field_type& prepared_state = *level.coupled[index];
    const field_type& prepared_result = *level.residual[index];
    require_same_field_contract_(state, prepared_state, "AMR coupled Jacobian state scratch");
    require_same_field_contract_(result, prepared_result, "AMR coupled Jacobian residual scratch");
    if (state.ghosts() != prepared_state.ghosts() || result.ghosts() != prepared_result.ghosts() ||
        state.shares_storage_with(prepared_state) || state.shares_storage_with(prepared_result) ||
        result.shares_storage_with(prepared_state) || result.shares_storage_with(prepared_result))
      throw std::invalid_argument(
          "AMR coupled Jacobian scratch changed or aliases an invocation field");
  };
  require_block(first_runtime, first_state, first_result);
  require_block(second_runtime, second_state, second_result);
  return level;
}

field_type& persistent_scratch_(ScratchKind kind, ProgramCacheSlot slot, int subslot,
                                const field_type& prototype, int ncomp, Extent<Dim> ghosts,
                                bool reset = true) const {
  if (subslot < 0)
    throw std::invalid_argument("AMR Program scratch identity must be non-negative");
  if (slot >= prepared_scratch_.size() || active_level_ < 0)
    throw std::logic_error("AMR Program scratch is outside the bind-sealed resource plan");
  auto& family = prepared_scratch_[slot][static_cast<std::size_t>(kind)];
  auto& descriptors = prepared_scratch_descriptors_[slot][static_cast<std::size_t>(kind)];
  const auto index = static_cast<std::size_t>(subslot);
  if (index >= family.size() || index >= descriptors.size() || !family[index] ||
      !descriptors[index])
    throw std::logic_error("AMR Program scratch was not primed during installation");
  const PreparedScratchDescriptor& declaration = *descriptors[index];
  std::size_t storage_level = 0;
  if (declaration.declared_level < 0)
    storage_level = static_cast<std::size_t>(active_level_);
  else if (active_level_ != declaration.declared_level)
    throw std::logic_error("AMR Program scratch was used outside its declared hierarchy level");
  if (storage_level >= family[index]->size())
    throw std::logic_error("AMR Program scratch was not primed during installation");
  field_type& result = family[index]->at(storage_level);
  if (result.layout() != prototype.layout() || result.distribution() != prototype.distribution() ||
      result.local_rank() != prototype.local_rank() || result.ncomp() != ncomp ||
      result.ghosts() != ghosts)
    throw std::runtime_error("AMR Program scratch identity changed its exact field contract");
  if (reset) {
    result.set_val(Real(0));
    clear_active_flux_expression_(result);
  }
  return result;
}

void prime_prepared_scratch(std::uint8_t kind_code, std::size_t slot, int subslot,
                            int program_block, int declared_level, int ncomp,
                            int ghost_depth) const {
  if (preparation_view_ == nullptr || subslot < 0 || ncomp < 1 || ghost_depth < 0 ||
      kind_code > static_cast<std::uint8_t>(ScratchKind::Scalar) ||
      slot >= prepared_scratch_.size())
    throw std::invalid_argument("AMR Program scratch prime is outside the detached resource image");
  const ScratchKind kind = static_cast<ScratchKind>(kind_code);
  const int runtime_block = sys_block(program_block);
  const auto required = static_cast<std::size_t>(subslot) + 1;
  auto& family = prepared_scratch_[slot][static_cast<std::size_t>(kind)];
  auto& descriptors = prepared_scratch_descriptors_[slot][static_cast<std::size_t>(kind)];
  if (family.size() < required)
    family.resize(required);
  if (descriptors.size() < required)
    descriptors.resize(required);
  auto& entry = family[static_cast<std::size_t>(subslot)];
  auto& descriptor = descriptors[static_cast<std::size_t>(subslot)];
  const Extent<Dim> ghosts = uniform_ghosts_(ghost_depth);
  const PreparedScratchDescriptor requested{runtime_block, declared_level, ncomp, ghosts};
  if (descriptor &&
      (descriptor->runtime_block != requested.runtime_block ||
       descriptor->declared_level != requested.declared_level ||
       descriptor->ncomp != requested.ncomp || descriptor->ghosts != requested.ghosts))
    throw std::logic_error("AMR Program scratch prime changed its sealed owner or shape");
  const auto& prototypes =
      preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block));
  if (declared_level < -1 ||
      (declared_level >= 0 && static_cast<std::size_t>(declared_level) >= prototypes.size()))
    throw std::invalid_argument("AMR Program scratch prime has an invalid declared level");
  const std::size_t first_level =
      declared_level < 0 ? 0U : static_cast<std::size_t>(declared_level);
  const std::size_t level_count = declared_level < 0 ? prototypes.size() : 1U;
  if (!entry) {
    entry.emplace();
    entry->reserve(level_count);
    for (std::size_t index = 0; index < level_count; ++index)
      entry->emplace_back(make_scratch_(prototypes.at(first_level + index), ncomp, ghosts));
    descriptor.emplace(requested);
    return;
  }
  if (!descriptor)
    throw std::logic_error("AMR Program scratch prime lost its sealed declaration");
  if (entry->size() != level_count)
    throw std::logic_error("AMR Program scratch prime changed its prepared level count");
  for (std::size_t index = 0; index < level_count; ++index) {
    const field_type& prepared = entry->at(index);
    const field_type& prototype = prototypes.at(first_level + index);
    if (prepared.layout() != prototype.layout() ||
        prepared.distribution() != prototype.distribution() ||
        prepared.local_rank() != prototype.local_rank() || prepared.ncomp() != ncomp ||
        prepared.ghosts() != ghosts)
      throw std::logic_error("AMR Program scratch prime changed an exact layout");
  }
}

int scratch_prototype_owner_(const field_type& prototype) const {
  std::optional<int> owner;
  const auto record = [&](int candidate) {
    if (candidate < 0 || candidate >= n_blocks())
      throw std::logic_error("AMR Program scratch prototype has an invalid runtime owner");
    if (owner && *owner != candidate)
      throw std::logic_error("AMR Program scratch prototype aliases multiple runtime owners");
    owner = candidate;
  };
  for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block) {
    const field_type* const active = live_attempt_state_(runtime_block, active_level_);
    const field_type* const accepted =
        &facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
    if (&prototype == active || &prototype == accepted)
      record(runtime_block);
  }
  for (const auto& [key, scratch] : scratches_)
    if (&prototype == &scratch)
      record(std::get<2>(key));
  const auto& manager = runtime_state().hist_;
  for (const auto& [key, ring] : manager.histories) {
    const bool is_history_slot = std::any_of(
        ring.begin(), ring.end(), [&](const field_type& slot) { return &prototype == &slot; });
    if (!is_history_slot)
      continue;
    const auto level = history_levels_.find(key);
    const auto decoded = decode_history_key_(key);
    const auto history_owner = manager.owner.find(key);
    if (level == history_levels_.end() || !decoded || decoded->first != active_level_ ||
        level->second != active_level_ || history_owner == manager.owner.end() ||
        history_epoch_ != runtime_->topology_epoch() ||
        history_generation_ != runtime_->materialization_generation())
      throw std::invalid_argument(
          "AMR Program scratch prototype names a stale or foreign-level history ring");
    record(history_owner->second);
  }
  if (!owner)
    throw std::invalid_argument(
        "AMR Program scratch prototype has no authenticated runtime block owner");
  return *owner;
}

int projection_candidate_owner_(const field_type& detached_candidate) const {
  std::optional<int> owner;
  const auto record = [&](int candidate_owner) {
    if (candidate_owner < 0 || candidate_owner >= n_blocks())
      throw std::logic_error("AMR Program projection scratch has an invalid runtime owner");
    if (owner && *owner != candidate_owner)
      throw std::logic_error("AMR Program projection candidate aliases multiple scratch owners");
    owner = candidate_owner;
  };
  // Accepted Program execution is bound only to the dense prepared-scratch arena.  The legacy
  // map is a preparation-only carrier; probing it here would turn a projection callback into a
  // hot associative lookup and permit post-bind scratch drift.
  if (preparation_mode_) {
    for (const auto& [key, scratch] : scratches_) {
      if (&detached_candidate != &scratch)
        continue;
      if (std::get<0>(key) != ScratchKind::State || std::get<1>(key) != active_level_)
        throw std::invalid_argument(
            "AMR Program projection candidate is not an active-level state scratch");
      record(std::get<2>(key));
    }
  }
  constexpr std::size_t state_family = static_cast<std::size_t>(ScratchKind::State);
  for (std::size_t slot = 0; slot < prepared_scratch_.size(); ++slot) {
    const auto& family = prepared_scratch_[slot][state_family];
    const auto& descriptors = prepared_scratch_descriptors_[slot][state_family];
    if (family.size() != descriptors.size())
      throw std::logic_error("AMR Program projection scratch lost its descriptors");
    for (std::size_t subslot = 0; subslot < family.size(); ++subslot) {
      if (!family[subslot] || !descriptors[subslot])
        continue;
      const PreparedScratchDescriptor& declaration = *descriptors[subslot];
      const std::size_t storage_level =
          declaration.declared_level < 0 ? static_cast<std::size_t>(active_level_) : std::size_t{0};
      if ((declaration.declared_level >= 0 && declaration.declared_level != active_level_) ||
          storage_level >= family[subslot]->size())
        continue;
      if (&detached_candidate == &family[subslot]->at(storage_level))
        record(declaration.runtime_block);
    }
  }
  if (!owner)
    throw std::invalid_argument(
        "AMR Program projection requires an owner-qualified detached state scratch");
  return *owner;
}
