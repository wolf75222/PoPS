void require_history_free_for_topology_change_(std::string_view operation) const {
  if (!history_levels_.empty())
    throw std::runtime_error(
        "AmrProgramContext cannot " + std::string(operation) +
        " while exact-ranked history rings lack a prepared rematerialization transaction");
}
[[noreturn]] static void unavailable_(std::string_view provider) {
  throw std::runtime_error("AmrProgramContext has no prepared " + std::string(provider));
}
/// Provider-owned physical law used by both the authenticated build request and the generated
/// flat Krylov boundary session. Keep one retained instance per prepared level; consumers never
/// reconstruct this law from topology alone.
PhysicalBoundaryConditions<Dim> hierarchy_tensor_boundary_(const Geometry<Dim>& geometry) const {
  const BoundaryTopology<Dim> topology = facade_->prepared_amr_boundary_topology();
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
    request.levels.reserve(runtime_->hierarchy().num_levels());
    boundaries.reserve(runtime_->hierarchy().num_levels());
    if (runtime_->hierarchy().num_levels() > 1)
      request.ratios.reserve(runtime_->hierarchy().num_levels() - 1);
    for (std::size_t level = 0; level < runtime_->hierarchy().num_levels(); ++level) {
      const field_type& level_state = runtime_->hierarchy().state(level);
      const Geometry<Dim> level_geometry =
          facade_->prepared_amr_level_geometry(static_cast<int>(level));
      const PhysicalBoundaryConditions<Dim> boundary = hierarchy_tensor_boundary_(level_geometry);
      request.levels.push_back({level_geometry, boundary, level_state.layout(),
                                level_state.distribution(), level_state.local_rank()});
      boundaries.push_back({level_geometry, boundary});
      if (level != 0)
        request.ratios.push_back(runtime_->hierarchy().layout(level).ratio_from_parent());
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
  refresh_resources_();
  if (!hierarchy_tensor_solver_ || hierarchy_tensor_topology_epoch_ != runtime_->topology_epoch() ||
      hierarchy_tensor_materialization_generation_ != runtime_->materialization_generation()) {
    PreparedHierarchyTensorState prepared =
        prepare_hierarchy_tensor_solver_(*hierarchy_tensor_selection_);
    hierarchy_tensor_solver_ = std::move(prepared.solver);
    hierarchy_tensor_boundaries_ = std::move(prepared.boundaries);
    hierarchy_tensor_topology_epoch_ = runtime_->topology_epoch();
    hierarchy_tensor_materialization_generation_ = runtime_->materialization_generation();
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
  if (facade_ != nullptr)
    facade_->refresh_prepared_amr_levels();
  if (resource_epoch_ == runtime_->topology_epoch() &&
      resource_generation_ == runtime_->materialization_generation())
    return;
  if (!history_levels_.empty() && (history_epoch_ != runtime_->topology_epoch() ||
                                   history_generation_ != runtime_->materialization_generation()))
    throw std::runtime_error(
        "AMR Program topology changed while retained histories still name the prior layouts");
  scratches_.clear();
  synchronize_resource_generation_();
  if (active_level_ >= nlev())
    active_level_ = 0;
}

/// Validate the manager that the AMR engine has already remapped and exchanged in its outer
/// transaction.  This does not prepare, allocate, or register histories: it merely authenticates
/// the exact replacement against the previously live logical registry and the now-live hierarchy.
void validate_accepted_history_remap_() const {
  const auto& manager = runtime_state().hist_;
  if (history_levels_.empty() || manager.histories.empty())
    throw std::runtime_error(
        "AMR Program accepted history remap requires both prior and remapped history registries");

  std::set<std::string> prior_names;
  for (const auto& [key, level] : history_levels_) {
    const auto decoded = decode_history_key_(key);
    if (!decoded || decoded->first != level)
      throw std::runtime_error("AMR Program retained history registry is malformed before remap");
    prior_names.insert(decoded->second);
  }

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
    if (!decoded || decoded->first < 0 || static_cast<std::size_t>(decoded->first) >= live_levels ||
        key != history_key_(decoded->second, decoded->first))
      throw std::runtime_error("AMR Program accepted history remap has a malformed ring key");
    const int level = decoded->first;
    const std::string& name = decoded->second;
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
    const field_type& live = facade_->prepared_amr_block_state(owner, level);
    if (ring.empty())
      throw std::runtime_error("AMR Program accepted history remap has an empty ring");
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

void refresh_resources_after_accepted_history_remap_() const {
  if (facade_ != nullptr)
    facade_->refresh_prepared_amr_levels();
  if (resource_epoch_ == runtime_->topology_epoch() &&
      resource_generation_ == runtime_->materialization_generation())
    throw std::runtime_error(
        "AMR Program accepted history remap requires a newly published hierarchy generation");
  validate_accepted_history_remap_();
  scratches_.clear();
  synchronize_resource_generation_();
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
  if (facade_ == nullptr)
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
    if (facade_->n_blocks() == 2) {
      candidate = std::make_unique<CoupledJacvecScratch>();
      candidate->topology_epoch = topology_epoch;
      candidate->materialization_generation = materialization_generation;
      candidate->levels.resize(runtime_->hierarchy().num_levels());
      for (std::size_t level = 0; level < candidate->levels.size(); ++level)
        for (int runtime_block = 0; runtime_block < 2; ++runtime_block) {
          const field_type& prototype =
              facade_->prepared_amr_block_state(runtime_block, static_cast<int>(level));
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

field_type& persistent_scratch_(ScratchKind kind, std::int64_t value_id, int subslot,
                                const field_type& prototype, int ncomp, Extent<Dim> ghosts) const {
  if (value_id < 0 || subslot < 0)
    throw std::invalid_argument("AMR Program scratch identity must be non-negative");
  refresh_resources_();
  const int runtime_owner = scratch_prototype_owner_(prototype);
  const ScratchKey key{kind, active_level_, runtime_owner, value_id, subslot};
  auto found = scratches_.find(key);
  if (found == scratches_.end())
    found = scratches_.emplace(key, make_scratch_(prototype, ncomp, ghosts)).first;
  field_type& result = found->second;
  if (result.layout() != prototype.layout() || result.distribution() != prototype.distribution() ||
      result.local_rank() != prototype.local_rank() || result.ncomp() != ncomp ||
      result.ghosts() != ghosts)
    throw std::runtime_error("AMR Program scratch identity changed its exact field contract");
  result.set_val(Real(0));
  clear_active_flux_expression_(result);
  return result;
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
    const field_type* active = nullptr;
    if (!active_attempt_states_.empty())
      active = active_attempt_states_.at(static_cast<std::size_t>(runtime_block));
    const field_type* const accepted =
        &facade_->prepared_amr_block_state(runtime_block, active_level_);
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
  for (const auto& [key, scratch] : scratches_) {
    if (&detached_candidate != &scratch)
      continue;
    if (std::get<0>(key) != ScratchKind::State || std::get<1>(key) != active_level_)
      throw std::invalid_argument(
          "AMR Program projection candidate is not an active-level state scratch");
    const int candidate_owner = std::get<2>(key);
    if (owner && *owner != candidate_owner)
      throw std::logic_error("AMR Program projection candidate aliases multiple scratch owners");
    owner = candidate_owner;
  }
  if (!owner)
    throw std::invalid_argument(
        "AMR Program projection requires an owner-qualified detached state scratch");
  return *owner;
}

static std::string history_key_(const std::string& name, int level) {
  if (name.empty() || level < 0)
    throw std::invalid_argument("AMR Program history key is invalid");
  return "pops.amr.level-history.v1/" + std::to_string(level) + "/" + std::to_string(name.size()) +
         ":" + name;
}

void require_history_owner_(int program_owner) const {
  if (program_owner < 0 || sys_block(program_owner) < 0)
    throw std::invalid_argument("AMR Program history has a foreign block owner");
}

field_type& history_slot_(const std::string& name, int lag, bool zero_start, int components) const {
  const std::string key = history_key_(name, active_level_);
  auto& manager = runtime_state().hist_;
  const auto found = manager.histories.find(key);
  if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(key))
    throw std::out_of_range("AMR Program history slot is absent");
  field_type& result = found->second[static_cast<std::size_t>(lag)];
  if (components >= 0 && result.ncomp() != components)
    throw std::invalid_argument("AMR Program history component contract differs");
  if (!manager.initialized.at(key)) {
    if (!zero_start)
      throw std::runtime_error("AMR Program history has not been initialized");
  }
  return result;
}
