void apply_coupling_operators(std::string_view graph_identity, std::string_view rate_identity,
                              std::string_view application_identity, Real dt,
                              std::initializer_list<CouplingStateOverride> candidates) const {
  require_facade_execution_();
  auto& workspace = hot_path_workspace_;
  workspace.require_bound(candidates.size(), "AMR Program coupling");
  std::fill(workspace.coupling_states.begin(), workspace.coupling_states.end(), nullptr);
  std::optional<runtime::multiblock::BoundaryEvaluationPoint> prepared_point;
  std::optional<runtime::multiblock::InterfaceFluxFragmentPublication> prepared_publication;
  std::exception_ptr local_error;
  try {
    if (graph_identity.empty() || graph_identity != facade_->program_installed_hash_() ||
        rate_identity.empty() || application_identity.empty())
      throw std::invalid_argument(
          "AMR Program coupling requires exact graph, rate, and application identities");
    const auto& coupling_budget =
        facade_->program_prepared_amr_program_flux_expression_budget_();
    std::size_t identity_characters = graph_identity.size();
    if (rate_identity.size() > std::numeric_limits<std::size_t>::max() - identity_characters)
      throw std::length_error("AMR Program coupling identity characters exceed size_t");
    identity_characters += rate_identity.size();
    if (application_identity.size() > std::numeric_limits<std::size_t>::max() - identity_characters)
      throw std::length_error("AMR Program coupling identity characters exceed size_t");
    identity_characters += application_identity.size();
    if (identity_characters > coupling_budget.interface_coupling_identity_character_bound)
      throw std::length_error(
          "AMR Program coupling identities exceed the frozen artifact character bound");
    for (const CouplingStateOverride& candidate : candidates) {
      const int runtime_block = sys_block(candidate.program_block);
      if (candidate.state == nullptr ||
          workspace.coupling_states[static_cast<std::size_t>(runtime_block)] != nullptr)
        throw std::invalid_argument(
            "AMR Program coupling candidates are incomplete, duplicate, or null");
      require_same_field_contract_(*candidate.state,
                                   facade_->program_prepared_amr_block_state_(runtime_block,
                                                                                active_level_),
                                   "AMR Program coupling candidate");
      if (!active_attempt_states_.empty()) {
        const bool detached_group_candidate = is_live_attempt_candidate_(candidate.state);
        const bool prepared_scratch =
            std::any_of(scratches_.begin(), scratches_.end(),
                        [&](const auto& entry) { return &entry.second == candidate.state; });
        if (!detached_group_candidate && !prepared_scratch)
          throw std::invalid_argument(
              "active AMR Program coupling requires detached group candidates or prepared "
              "scratches");
        for (int accepted_block = 0; accepted_block < n_blocks(); ++accepted_block)
          if (candidate.state ==
              &facade_->program_prepared_amr_block_state_(accepted_block, active_level_))
            throw std::invalid_argument(
                "active AMR Program coupling cannot mutate an accepted block carrier");
      }
      workspace.coupling_states[static_cast<std::size_t>(runtime_block)] = candidate.state;
    }
    if (std::find(workspace.coupling_states.begin(), workspace.coupling_states.end(), nullptr) !=
        workspace.coupling_states.end())
      throw std::invalid_argument(
          "AMR Program coupling requires every authenticated Program block candidate");
    const ::pops::amr::Rational interval_phase =
        active_subcycling_window_.begin.phase +
        (active_subcycling_window_.end.phase - active_subcycling_window_.begin.phase) * stage_time_;
    const double interval_duration =
        active_subcycling_window_.end.physical_time - active_subcycling_window_.begin.physical_time;
    const double evaluation_time =
        active_subcycling_window_.begin.physical_time + stage_time_.value() * interval_duration;
    if (!std::isfinite(interval_duration) || !(interval_duration > 0.0) ||
        !std::isfinite(static_cast<double>(dt)) || !(dt > Real(0)))
      throw std::logic_error("AMR Program interface publication has an invalid exact interval");
    prepared_point.emplace(runtime::multiblock::BoundaryEvaluationPoint{
        primary_clock_, static_cast<std::int64_t>(facade_->program_macro_step_()), active_level_,
        logical_substep_, 0, stage_time_, interval_duration, evaluation_time,
        std::string(graph_identity), std::string(rate_identity),
        std::string(application_identity)});
    prepared_publication.emplace(runtime::multiblock::InterfaceFluxFragmentPublication{
        interface_flux_ledger_.get(),
        runtime_->topology_epoch(),
        nlev(),
        {active_level_, static_cast<std::int64_t>(facade_->program_macro_step_()), interval_phase,
         evaluation_time},
        "program-stage:" + std::to_string(stage_time_.numerator) + "/" +
            std::to_string(stage_time_.denominator),
        active_subcycling_window_,
        exact_binary_rational_(dt / interval_duration),
        true});
  } catch (...) {
    local_error = std::current_exception();
  }
  const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program coupling candidate pack failed collectively");
  }
  count_kernel_(static_cast<std::int64_t>(facade_->apply_prepared_amr_program_candidates(
      active_level_, dt, std::span<field_type* const>(workspace.coupling_states), *prepared_point,
      interface_flux_ledger_->in_transaction() ? &*prepared_publication : nullptr)));
}

private:
/// Visit every live hierarchy level of an authenticated Program owner exactly once.  The seed
/// field identifies the owner's state, scratch, history, or current-level layout; each visit
/// receives that level's matching field and prepared active-cell mask (null when Cartesian).
template <class Visitor>
void for_each_owner_active_level_(int program_block, const field_type& left,
                                  const field_type* right, Visitor&& visitor,
                                  bool cover_all_scratch_levels = false) const {
  refresh_resources_();
  const int runtime_block = sys_block(program_block);
  const int levels = nlev();
  if (levels < 1)
    return;

  const auto same_layout = [](const field_type& a, const field_type& b) {
    return a.layout() == b.layout() && a.distribution() == b.distribution() &&
           a.local_rank() == b.local_rank() && a.local_size() == b.local_size();
  };

  enum class Family : std::uint8_t { Unset, State, Scratch, History, Direct };
  struct Identity {
    Family family = Family::Unset;
    int scratch_kind = -1;
    int scratch_owner = -1;
    std::int64_t scratch_value_id = -1;
    int scratch_subslot = -1;
    std::string history_name;
    int history_lag = -1;
    int direct_level = -1;
  };

  const auto classify = [&](const field_type& seed, const char* role) {
    Identity identity;
    for (int level = 0; level < levels; ++level) {
      if (&seed == &facade_->program_prepared_amr_block_state_(runtime_block, level))
        identity.family = Family::State;
      if (const field_type* attempt = live_attempt_state_(runtime_block, level);
          attempt != nullptr && &seed == attempt)
        identity.family = Family::State;
    }
    for (const auto& [key, scratch] : scratches_) {
      if (&seed != &scratch)
        continue;
      if (std::get<2>(key) != runtime_block)
        throw std::invalid_argument(std::string(role) +
                                    " received a foreign AMR Program owner field");
      if (identity.family != Family::Unset && identity.family != Family::Scratch)
        throw std::logic_error(std::string(role) +
                               " aliases multiple AMR Program field identities");
      identity.family = Family::Scratch;
      identity.scratch_kind = static_cast<int>(std::get<0>(key));
      identity.direct_level = std::get<1>(key);
      identity.scratch_owner = std::get<2>(key);
      identity.scratch_value_id = std::get<3>(key);
      identity.scratch_subslot = std::get<4>(key);
    }
    const auto& manager = runtime_state().hist_;
    for (const auto& [key, ring] : manager.histories) {
      for (std::size_t lag = 0; lag < ring.size(); ++lag) {
        if (&seed != &ring[lag])
          continue;
        const auto decoded = decode_history_key_(key);
        const auto owner = manager.owner.find(key);
        if (!decoded || owner == manager.owner.end() || owner->second != runtime_block)
          throw std::invalid_argument(std::string(role) +
                                      " received a foreign AMR Program history field");
        if (identity.family != Family::Unset && identity.family != Family::History)
          throw std::logic_error(std::string(role) +
                                 " aliases multiple AMR Program field identities");
        identity.family = Family::History;
        identity.history_name = decoded->second;
        identity.history_lag = static_cast<int>(lag);
      }
    }
    if (identity.family == Family::Unset) {
      for (int level = 0; level < levels; ++level) {
        if (!same_layout(seed,
                         facade_->program_prepared_amr_block_state_(runtime_block, level)))
          continue;
        if (identity.direct_level >= 0)
          throw std::logic_error(std::string(role) + " matches multiple AMR Program owner levels");
        identity.direct_level = level;
      }
      if (identity.direct_level < 0)
        throw std::invalid_argument(std::string(role) +
                                    " has no authenticated AMR Program owner layout");
      identity.family = Family::Direct;
    }
    return identity;
  };

  const auto resolve = [&](const field_type& seed, const Identity& identity, int level,
                           const char* role) -> const field_type& {
    const field_type& accepted =
        facade_->program_prepared_amr_block_state_(runtime_block, level);
    const field_type* current = nullptr;
    if (identity.family == Family::State) {
      current = live_attempt_state_(runtime_block, level);
      if (current == nullptr)
        current = &accepted;
    } else if (identity.family == Family::Scratch) {
      for (const auto& [key, scratch] : scratches_) {
        if (static_cast<int>(std::get<0>(key)) == identity.scratch_kind &&
            std::get<1>(key) == level && std::get<2>(key) == identity.scratch_owner &&
            std::get<3>(key) == identity.scratch_value_id &&
            std::get<4>(key) == identity.scratch_subslot) {
          current = &scratch;
          break;
        }
      }
      if (current == nullptr)
        throw std::runtime_error(std::string(role) +
                                 " scratch is missing a live AMR Program owner level");
    } else if (identity.family == Family::History) {
      const auto& manager = runtime_state().hist_;
      const auto found = manager.histories.find(history_key_(identity.history_name, level));
      if (found == manager.histories.end() || identity.history_lag < 0 ||
          static_cast<std::size_t>(identity.history_lag) >= found->second.size())
        throw std::runtime_error(std::string(role) +
                                 " history is missing a live AMR Program owner level");
      current = &found->second[static_cast<std::size_t>(identity.history_lag)];
    } else {
      if (level != identity.direct_level)
        throw std::runtime_error(std::string(role) +
                                 " cannot cover every live AMR Program owner level");
      current = &seed;
    }
    require_same_layout_(*current, accepted, role);
    return *current;
  };

  const Identity left_identity = classify(left, "AMR Program reduction");
  const std::optional<Identity> right_identity =
      right == nullptr ? std::nullopt
                       : std::optional<Identity>(classify(*right, "AMR Program reduction"));
  // State and history reductions cover every live level.  A Program scratch or a
  // layout-authenticated direct field is produced on one active_level_ at a time
  // (the generated body walks for_each_program_resource_level).  Reducing the
  // same SSA id across leftover sibling-level scratches would mix stale values
  // from the previous level iteration into the current guard or CFL scalar.
  // pointwise_status_max is the exception: it must see every sibling scratch of
  // the same SSA id so an active invalid on any live level is fail-closed.
  const auto covers_level = [&](const Identity& identity, int level) {
    if (identity.family == Family::Scratch)
      return cover_all_scratch_levels || level == identity.direct_level;
    if (identity.family == Family::Direct)
      return level == identity.direct_level;
    return true;
  };
  for (int level = 0; level < levels; ++level) {
    if (!covers_level(left_identity, level))
      continue;
    if (right_identity && !covers_level(*right_identity, level))
      continue;
    const field_type& accepted =
        facade_->program_prepared_amr_block_state_(runtime_block, level);
    const field_type& left_field = resolve(left, left_identity, level, "AMR Program reduction");
    const field_type* right_field = nullptr;
    if (right_identity)
      right_field = &resolve(*right, *right_identity, level, "AMR Program reduction");
    const field_type* const active =
        facade_->program_prepared_amr_block_level_active_mask_(runtime_block, level);
    if (active != nullptr)
      require_same_layout_(*active, accepted, "AMR Program reduction active mask");
    visitor(left_field, right_field, active);
  }
}

void converge_owner_reduction_(std::exception_ptr local_error, const ExecutionLane& lane,
                               const char* operation) const {
  if (all_reduce_max(local_error ? 1L : 0L, lane) == 0)
    return;
  if (lane.size() == 1 && local_error)
    std::rethrow_exception(local_error);
  throw std::runtime_error(std::string(operation) + " failed collectively");
}

public:
Real sum_component(const field_type& field, int component) const {
  return static_cast<Real>(
      all_reduce_sum(pops::reduce_sum_local(field, component), prepared_execution_lane()));
}
Real max_component(const field_type& field, int component) const {
  return static_cast<Real>(
      all_reduce_max(pops::reduce_max_local(field, component), prepared_execution_lane()));
}
Real min_component(const field_type& field, int component) const {
  return static_cast<Real>(
      all_reduce_min(pops::reduce_min_local(field, component), prepared_execution_lane()));
}

/// Generated AMR reductions authenticate the Program owner and exclude inactive cells on every
/// live level exactly once.  They stay raw active-domain algebra: no kappa, volume, or geometry
/// weighting.  Empty active domains use sum/abs_sum/dot/norm2/norm_inf = 0, max = -inf, min = +inf.
Real sum_component(int program_block, const field_type& field, int component) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = Real(0);
  try {
    for_each_owner_active_level_(
        program_block, field, nullptr,
        [&](const field_type& level_field, const field_type*, const field_type* active) {
          local += pops::reduce_active_sum_local(level_field, component, active);
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program sum reduction");
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane));
}
Real abs_sum_component(int program_block, const field_type& field, int component) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = Real(0);
  try {
    for_each_owner_active_level_(
        program_block, field, nullptr,
        [&](const field_type& level_field, const field_type*, const field_type* active) {
          local += pops::reduce_active_abs_sum_local(level_field, component, active);
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program abs-sum reduction");
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane));
}
Real max_component(int program_block, const field_type& field, int component) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = -std::numeric_limits<Real>::infinity();
  long participation = 0;
  try {
    for_each_owner_active_level_(
        program_block, field, nullptr,
        [&](const field_type& level_field, const field_type*, const field_type* active) {
          const MaskedMaxLocalResult probe =
              pops::reduce_masked_max_local(level_field, component, active);
          if (!probe.has_active)
            return;
          ++participation;
          local = std::max(local, pops::reduce_active_max_local(level_field, component, active));
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program max reduction");
  const long global_participation = all_reduce_sum(participation, lane);
  const Real reduced = static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
  return global_participation == 0 ? -std::numeric_limits<Real>::infinity() : reduced;
}
Real min_component(int program_block, const field_type& field, int component) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = std::numeric_limits<Real>::infinity();
  long participation = 0;
  try {
    for_each_owner_active_level_(
        program_block, field, nullptr,
        [&](const field_type& level_field, const field_type*, const field_type* active) {
          const MaskedMaxLocalResult probe =
              pops::reduce_masked_max_local(level_field, component, active);
          if (!probe.has_active)
            return;
          ++participation;
          local = std::min(local, pops::reduce_active_min_local(level_field, component, active));
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program min reduction");
  const long global_participation = all_reduce_sum(participation, lane);
  const Real reduced = static_cast<Real>(all_reduce_min(static_cast<double>(local), lane));
  return global_participation == 0 ? std::numeric_limits<Real>::infinity() : reduced;
}
Real norm2(int program_block, const field_type& field) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = Real(0);
  try {
    for_each_owner_active_level_(
        program_block, field, nullptr,
        [&](const field_type& level_field, const field_type*, const field_type* active) {
          local += pops::dot_active_local(level_field, level_field, 0, active);
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program norm2 reduction");
  return std::sqrt(static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane)));
}
Real norm_inf(int program_block, const field_type& field) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = Real(0);
  try {
    for_each_owner_active_level_(
        program_block, field, nullptr,
        [&](const field_type& level_field, const field_type*, const field_type* active) {
          local = std::max(local, pops::reduce_active_norm_inf_local(level_field, 0, active));
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program norm-inf reduction");
  return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
}
Real dot(int program_block, const field_type& left, const field_type& right) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr local_error;
  Real local = Real(0);
  try {
    for_each_owner_active_level_(
        program_block, left, &right,
        [&](const field_type& left_field, const field_type* right_field, const field_type* active) {
          if (right_field == nullptr)
            throw std::logic_error("AMR Program dot is missing its right owner field");
          require_same_field_contract_(left_field, *right_field, "AMR Program dot");
          local += pops::dot_active_local(left_field, *right_field, 0, active);
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  converge_owner_reduction_(local_error, lane, "AMR Program dot reduction");
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane));
}

Geometry<Dim> geometry() const {
  require_facade_execution_();
  return facade_->program_prepared_amr_level_geometry_(active_level_);
}

field_type& assembly_target(field_type& field, std::string_view identity) const {
  if (identity.empty())
    throw std::invalid_argument("AMR Program assembly target requires an identity");
  if (!hierarchy_tensor_selection_)
    return field;
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback)
    return field;
  if (std::find(hierarchy_tensor_selection_->assembly_field_slots.begin(),
                hierarchy_tensor_selection_->assembly_field_slots.end(),
                identity) == hierarchy_tensor_selection_->assembly_field_slots.end())
    throw std::invalid_argument("AMR hierarchy assembly used an undeclared provider field slot");
  return solver.assembly_target(identity, active_level_);
}

field_type& assembly_source(field_type& field, std::string_view identity) const {
  if (identity.empty())
    throw std::invalid_argument("AMR Program assembly source requires an identity");
  if (!hierarchy_tensor_selection_)
    return field;
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback)
    return field;
  if (identity != hierarchy_tensor_selection_->solution_field_slot)
    throw std::invalid_argument("AMR hierarchy read used an undeclared provider solution slot");
  return solver.solution(active_level_);
}

std::shared_ptr<scalar_boundary_session_type> prepare_mesh_boundary_session(
    field_type& prototype, const ExecutionLane& lane) const {
  require_prepared_lane_(lane, "AMR scalar boundary preparation");
  return std::make_shared<scalar_boundary_session_type>(
      geometry(), facade_->program_prepared_amr_boundary_topology_(), prototype, lane,
      next_boundary_generation_());
}

std::shared_ptr<block_boundary_session_type> prepare_block_boundary_session(
    int program_block, field_type& prototype,
    const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
  require_prepared_lane_(lane, "AMR block boundary preparation");
  const int runtime_block = sys_block(program_block);
  require_boundary_point_(point, "AMR block scalar boundary");
  require_same_field_contract_(prototype, state(program_block), "AMR block boundary prototype");
  auto transport = scalar_boundary_session_type::prepare_block(
      geometry(), facade_->program_prepared_amr_boundary_topology_(), prototype, lane,
      next_boundary_generation_());
  return std::shared_ptr<block_boundary_session_type>(
      new block_boundary_session_type(facade_, runtime_block, point, lane, std::move(transport)));
}

std::shared_ptr<tensor_boundary_session_type> prepare_tensor_boundary_session(
    int program_block, field_type& prototype,
    const runtime::multiblock::BoundaryEvaluationPoint& point, const ExecutionLane& lane) const {
  const ExecutionLane& runtime_lane = prepared_execution_lane();
  std::exception_ptr local_error;
  int runtime_block = -1;
  const field_type* runtime_block_owner = nullptr;
  const HierarchyTensorLevelBoundary* provider_boundary = nullptr;
  std::string runtime_lane_identity;
  try {
    if (!lane.active() || lane.identity().empty() || !runtime_lane.active() ||
        runtime_lane.identity().empty() || !lane.congruent_with(runtime_lane))
      throw std::invalid_argument(
          "AMR tensor boundary preparation requires a live lane in the exact runtime rank "
          "space");
    runtime_lane_identity.assign(runtime_lane.identity());
    if (!hierarchy_tensor_selection_ ||
        hierarchy_tensor_selection_->program_block != program_block ||
        hierarchy_tensor_selection_->components != 1 || !hierarchy_tensor_solver_ ||
        hierarchy_tensor_solver_->execution_path() !=
            HierarchyTensorSolverExecutionPath::PreparedKrylovFallback ||
        hierarchy_tensor_solver_->level_count() != 1 || nlev() != 1 || active_level_ != 0 ||
        hierarchy_tensor_topology_epoch_ != runtime_->topology_epoch() ||
        hierarchy_tensor_materialization_generation_ != runtime_->materialization_generation() ||
        hierarchy_tensor_boundaries_.size() != 1)
      throw std::logic_error(
          "AMR tensor boundary preparation requires its live one-level Krylov provider");
    runtime_block = sys_block(program_block);
    require_current_boundary_point_exact_(point, "AMR tensor boundary preparation");
    if (prototype.ncomp() != 1)
      throw std::invalid_argument("AMR tensor boundary prototype must be scalar");
    for (int axis = 0; axis < Dim; ++axis)
      if (prototype.ghosts()[axis] != 1)
        throw std::invalid_argument(
            "AMR tensor boundary prototype requires its exact one-cell ghost layout");
    runtime_block_owner =
        &facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
    require_same_layout_(prototype, *runtime_block_owner,
                         "AMR tensor boundary prototype authority");
    if (facade_->program_prepared_amr_block_level_active_mask_(runtime_block, active_level_) !=
        nullptr)
      throw std::logic_error(
          "AMR Cartesian tensor boundary has no embedded-boundary cut-face authority");
    provider_boundary = &hierarchy_tensor_boundaries_.front();
    if (provider_boundary->geometry != geometry())
      throw std::logic_error("AMR tensor boundary provider geometry is stale");
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR tensor boundary preparation failed collectively");
  }

  return tensor_boundary_session_type::prepare(
      provider_boundary->geometry, provider_boundary->conditions, prototype, lane,
      next_boundary_generation_(),
      PreparedTensorBoundaryAuthority{
          this, runtime_, reinterpret_cast<std::uintptr_t>(runtime_block_owner),
          std::move(runtime_lane_identity), program_block, runtime_block, active_level_,
          runtime_->topology_epoch(), runtime_->materialization_generation()},
      point);
}

void fill_boundary(field_type& field) const {
  fill_boundary(field, prepared_execution_lane());
}

void fill_boundary(field_type& field, const ExecutionLane& lane) const {
  require_prepared_lane_(lane, "AMR boundary fill");
  scalar_boundary_session_type session(geometry(), facade_->program_prepared_amr_boundary_topology_(), field,
                                       lane, next_boundary_generation_());
  session.fill(field);
}

void laplacian(field_type& output, field_type& input) const {
  require_scalar_stencil_(output, input, 1, "AMR Program Laplacian");
  fill_boundary(input);
  const Geometry<Dim> geom = geometry();
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const FieldView<Real, Dim> result = output.fab(local).view();
    const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      Real image = Real(0);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell;
        Index<Dim> upper = cell;
        --lower[axis];
        ++upper[axis];
        const Real spacing = geom.spacing(axis);
        image +=
            (value(upper, 0) - Real(2) * value(cell, 0) + value(lower, 0)) / (spacing * spacing);
      }
      result(cell, 0) = image;
    });
  }
  count_kernel_();
}

void laplacian(field_type& output, field_type& input,
               const scalar_boundary_session_type& boundary) const {
  require_prepared_lane_(boundary.lane(), "AMR Laplacian boundary");
  boundary.fill(input);
  laplacian_without_fill_(output, input, boundary.geometry());
}
void laplacian(field_type& output, field_type& input, const scalar_boundary_session_type& boundary,
               const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  require_boundary_point_(point, "AMR Program Laplacian");
  laplacian(output, input, boundary);
}

void gradient(field_type& output, field_type& input) const {
  require_scalar_stencil_(output, input, Dim, "AMR Program gradient");
  fill_boundary(input);
  const Geometry<Dim> geom = geometry();
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const FieldView<Real, Dim> result = output.fab(local).view();
    const FieldView<const Real, Dim> value = std::as_const(input).fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell;
        Index<Dim> upper = cell;
        --lower[axis];
        ++upper[axis];
        result(cell, axis) = (value(upper, 0) - value(lower, 0)) / (Real(2) * geom.spacing(axis));
      }
    });
  }
  count_kernel_();
}

void gradient(field_type& output, field_type& input,
              const scalar_boundary_session_type& boundary) const {
  require_prepared_lane_(boundary.lane(), "AMR gradient boundary");
  boundary.fill(input);
  gradient_without_fill_(output, input, boundary.geometry());
}
void gradient(field_type& output, field_type& input, const scalar_boundary_session_type& boundary,
              const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  require_boundary_point_(point, "AMR Program gradient");
  gradient(output, input, boundary);
}

void divergence(field_type& output, field_type& flux) const {
  auto boundary = prepare_mesh_boundary_session(flux, prepared_execution_lane());
  divergence(output, flux, *boundary);
}
void divergence(field_type& output, field_type& flux,
                const scalar_boundary_session_type& boundary) const {
  require_prepared_lane_(boundary.lane(), "AMR divergence boundary");
  if (output.ncomp() != 1 || flux.ncomp() != Dim)
    throw std::invalid_argument("AMR Program divergence requires one exact native vector field");
  require_same_layout_(output, flux, "AMR Program divergence");
  boundary.fill(flux);
  const Geometry<Dim> geom = boundary.geometry();
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const FieldView<Real, Dim> result = output.fab(local).view();
    const FieldView<const Real, Dim> vector = std::as_const(flux).fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      Real value = Real(0);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell;
        Index<Dim> upper = cell;
        --lower[axis];
        ++upper[axis];
        value += (vector(upper, axis) - vector(lower, axis)) / (Real(2) * geom.spacing(axis));
      }
      result(cell, 0) = value;
    });
  }
  count_kernel_();
}
void divergence(field_type& output, field_type& flux, const scalar_boundary_session_type& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  require_boundary_point_(point, "AMR Program divergence");
  divergence(output, flux, boundary);
}

void pack_vector(field_type& output, const std::array<const field_type*, Dim>& components) const {
  if (output.ncomp() != Dim)
    throw std::invalid_argument("AMR Program vector packing requires Dim output components");
  for (const field_type* component : components)
    if (component == nullptr || component->ncomp() != 1)
      throw std::invalid_argument("AMR Program vector packing requires scalar components");
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    std::array<FieldView<const Real, Dim>, Dim> values{};
    for (int axis = 0; axis < Dim; ++axis) {
      require_same_layout_(output, *components[static_cast<std::size_t>(axis)],
                           "AMR Program vector packing");
      values[static_cast<std::size_t>(axis)] =
          components[static_cast<std::size_t>(axis)]->fab(local).view();
    }
    const FieldView<Real, Dim> result = output.fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int axis = 0; axis < Dim; ++axis)
        result(cell, axis) = values[static_cast<std::size_t>(axis)](cell, 0);
    });
  }
  count_kernel_();
}

void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                      const tensor_boundary_session_type& boundary) const {
  tensor_laplacian(output, input, tensor, boundary, boundary.point());
}

void tensor_laplacian(field_type& output, field_type& input, const field_type& tensor,
                      const tensor_boundary_session_type& boundary,
                      const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  const ExecutionLane& lane = boundary.lane();
  const ExecutionLane& runtime_lane = prepared_execution_lane();
  std::exception_ptr local_error;
  try {
    if (!hierarchy_tensor_selection_ || !hierarchy_tensor_solver_ ||
        hierarchy_tensor_selection_->components != 1 ||
        hierarchy_tensor_topology_epoch_ != runtime_->topology_epoch() ||
        hierarchy_tensor_materialization_generation_ != runtime_->materialization_generation() ||
        hierarchy_tensor_solver_->execution_path() !=
            HierarchyTensorSolverExecutionPath::PreparedKrylovFallback ||
        hierarchy_tensor_solver_->level_count() != 1 || hierarchy_tensor_boundaries_.size() != 1 ||
        nlev() != 1 || active_level_ != 0)
      throw std::logic_error(
          "AMR Program tensor Laplacian requires its prepared one-level Krylov fallback");

    const int program_block = hierarchy_tensor_selection_->program_block;
    const int runtime_block = sys_block(program_block);
    const field_type& accepted =
        facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
    const PreparedTensorBoundaryAuthority& authority = boundary.authority();
    if (authority.program_owner != this || authority.runtime_owner != runtime_ ||
        authority.block_owner_identity != reinterpret_cast<std::uintptr_t>(&accepted) ||
        authority.runtime_lane_identity != runtime_lane.identity() ||
        authority.program_block != program_block || authority.runtime_block != runtime_block ||
        authority.level != active_level_ ||
        authority.topology_epoch != runtime_->topology_epoch() ||
        authority.materialization_generation != runtime_->materialization_generation() ||
        !lane.active() || lane.identity().empty() || !runtime_lane.active() ||
        runtime_lane.identity().empty() || !lane.congruent_with(runtime_lane) ||
        boundary.generation() == 0)
      throw std::invalid_argument(
          "AMR Program tensor Laplacian received a foreign or stale prepared session");
    const HierarchyTensorLevelBoundary& provider_boundary = hierarchy_tensor_boundaries_.front();
    if (boundary.geometry() != provider_boundary.geometry ||
        boundary.conditions() != provider_boundary.conditions || boundary.point() != point)
      throw std::invalid_argument(
          "AMR Program tensor Laplacian changed its provider boundary authority");
    require_current_boundary_point_exact_(point, "AMR Program tensor Laplacian");

    require_scalar_stencil_(output, input, 1, "AMR Program tensor Laplacian");
    require_same_layout_(input, tensor, "AMR Program tensor Laplacian tensor");
    if (tensor.ncomp() != Dim * Dim || output.ghosts() != input.ghosts() ||
        tensor.ghosts() != input.ghosts())
      throw std::invalid_argument(
          "AMR Program tensor Laplacian requires one exact row-major Dim*Dim stencil layout");
    for (int axis = 0; axis < Dim; ++axis)
      if (input.ghosts()[axis] != 1)
        throw std::invalid_argument(
            "AMR Program tensor Laplacian requires its exact one-cell ghost layout");
    boundary.authenticate_field(input);

    require_same_layout_(output, accepted, "AMR Program tensor Laplacian output authority");
    require_same_layout_(input, accepted, "AMR Program tensor Laplacian input authority");
    require_same_layout_(tensor, accepted, "AMR Program tensor Laplacian tensor authority");
    if (facade_->program_prepared_amr_block_level_active_mask_(runtime_block, active_level_) !=
        nullptr)
      throw std::logic_error(
          "AMR Cartesian tensor Laplacian has no active embedded-boundary cut-face authority");
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program tensor Laplacian validation failed collectively");
  }

  // Generated condensed coefficients are frozen with their allocated ghosts before Krylov. Only
  // the iterate belongs to this prepared boundary transaction; do not refill or repack A here.
  boundary.fill(input);
  const Geometry<Dim>& geom = boundary.geometry();
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const FieldView<Real, Dim> result = output.fab(local).view();
    const auto tensor_operator = elliptic::nd::make_cartesian_tensor_operator<
        elliptic::nd::CartesianTensorDivergenceSign::positive_divergence>(
        std::as_const(input.fab(local)).view(),
        elliptic::nd::packed_cartesian_tensor_coefficients<Dim>(
            std::as_const(tensor.fab(local)).view()),
        geom);
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      result(cell, 0) = tensor_operator.image(cell);
    });
  }
  count_kernel_();
}
