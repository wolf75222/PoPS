/// Materialize the centered face value 0.5*(F_left + F_right) from each already-filled named
/// cell flux. The same device-callable face provider drives both the divergence and the active
/// hierarchy's transactional reflux basis, so the conservative correction cannot observe a
/// reconstructed or differently rounded flux route.
void neg_div_named_flux_into(int program_block, field_type& stage_state, field_type& rhs,
                             const std::array<field_type*, Dim>& fluxes, int rate_id) const {
  const ExecutionLane& lane = prepared_execution_lane();
  const long active = active_attempt_states_.empty() ? 0L : 1L;
  const long active_minimum = all_reduce_min(active, lane);
  const long active_maximum = all_reduce_max(active, lane);
  if (active_minimum != active_maximum)
    throw std::logic_error("AMR named-flux activity differs between execution ranks");

  int runtime_block = -1;
  std::optional<Geometry<Dim>> prepared_geometry;
  std::optional<runtime::multiblock::BoundaryEvaluationPoint> point;
  std::optional<::pops::amr::ClockWindow> interval;
  FluxExpressionRegistry candidate_registry;
  std::vector<std::size_t> candidate_counts;
  std::uint64_t candidate_identity = 0;
  std::exception_ptr preparation_error;
  try {
    runtime_block = sys_block(program_block);
    require_rate_identity_(rate_id);
    // This is deliberately enforced here as well as in generated preflight: direct and
    // hand-authored callers may not bypass the EB/shared-interface refusal and mutate RHS or
    // flux metadata first.
    require_named_flux_execution_envelope_(runtime_block);
    require_same_field_contract_(stage_state, rhs, "AMR Program named-flux residual");
    prepared_geometry.emplace(geometry());
    for (int axis = 0; axis < Dim; ++axis)
      if (const field_type* flux = fluxes[static_cast<std::size_t>(axis)];
          flux == nullptr || flux->layout() != rhs.layout() ||
          flux->distribution() != rhs.distribution() || flux->local_rank() != rhs.local_rank() ||
          flux->local_size() != rhs.local_size() || flux->ncomp() != rhs.ncomp() ||
          flux->ghosts()[axis] < 1)
        throw std::invalid_argument("AMR named flux differs from its exact residual layout");
    if (active_maximum != 0) {
      candidate_registry = active_flux_expressions_;
      candidate_counts = active_flux_basis_counts_;
      candidate_identity = next_active_flux_basis_identity_;
      point.emplace(boundary_evaluation_point(rate_id));
      interval.emplace(::pops::amr::ClockWindow{
          {active_level_, point->tick, current_interval_begin_phase_, current_interval_start_time_},
          {active_level_, point->tick, current_interval_end_phase_,
           current_interval_start_time_ + current_dt_}});
    }
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR named-flux local preparation failed collectively");
  }

  if (active_maximum != 0)
    prepare_active_flux_basis_impl_(
        runtime_block, *point, rate_id, FluxBasisProvider::NamedCell, runtime_->topology_epoch(),
        runtime_->materialization_generation(), rhs, nullptr, nullptr, &fluxes, *interval,
        candidate_registry, candidate_counts, candidate_identity);

  std::exception_ptr execution_error;
  try {
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      std::array<FieldView<const Real, Dim>, Dim> views{};
      for (int axis = 0; axis < Dim; ++axis)
        views[static_cast<std::size_t>(axis)] =
            std::as_const(*fluxes[static_cast<std::size_t>(axis)]).fab(local).view();
      const FieldView<Real, Dim> output = rhs.fab(local).view();
      const int components = rhs.ncomp();
      const Geometry<Dim> geom = *prepared_geometry;
      for_each_cell(rhs.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component) {
          Real divergence = Real(0);
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> lower_face = cell;
            Index<Dim> upper_face = cell;
            ++upper_face[axis];
            divergence += (named_flux_face_value_(views[static_cast<std::size_t>(axis)], upper_face,
                                                  axis, component) -
                           named_flux_face_value_(views[static_cast<std::size_t>(axis)], lower_face,
                                                  axis, component)) /
                          geom.spacing(axis);
          }
          output(cell, component) = -divergence;
        }
      });
    }
    device_fence();
    count_kernel_();
  } catch (...) {
    execution_error = std::current_exception();
  }
  if (all_reduce_max(execution_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && execution_error)
      std::rethrow_exception(execution_error);
    throw std::runtime_error("AMR named-flux divergence failed collectively");
  }

  if (active_maximum != 0) {
    static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
    static_assert(std::is_nothrow_swappable_v<std::vector<std::size_t>>);
    active_flux_expressions_.swap(candidate_registry);
    active_flux_basis_counts_.swap(candidate_counts);
    next_active_flux_basis_identity_ = candidate_identity;
  }
}

void apply_projection(int program_block, field_type& detached_candidate) const {
  const int runtime_block = sys_block(program_block);
  const int candidate_owner = projection_candidate_owner_(detached_candidate);
  facade_->project_prepared_amr_block_level_state(runtime_block, active_level_, candidate_owner,
                                                  detached_candidate);
}

Real max_wave_speed(int program_block, const field_type& stage_state) const {
  return facade_->prepared_amr_block_level_maximum_speed(sys_block(program_block), active_level_,
                                                         stage_state);
}

Real hmin() const {
  const Geometry<Dim> geom = geometry();
  Real result = geom.spacing(0);
  for (int axis = 1; axis < Dim; ++axis)
    result = std::min(result, geom.spacing(axis));
  return result;
}

RuntimeParams program_params(int program_block) const {
  (void)sys_block(program_block);
  return facade_->program_params(program_block);
}

void axpy(field_type& destination, Real factor, const field_type& source) const {
  require_same_field_contract_(destination, source, "AMR Program axpy");
  auto expression_update = prepare_active_axpy_flux_expression_(
      destination, source, exact_runtime_axpy_coefficient_(factor, source));
  pops::saxpy(destination, factor, source);
  publish_active_flux_expression_update_(std::move(expression_update));
  count_kernel_();
}
void axpy(field_type& destination, Real factor, const field_type& source, Real reference_dt,
          std::initializer_list<ExactCoefficientTerm> terms) const {
  require_same_field_contract_(destination, source, "AMR Program exact axpy");
  const auto coefficient = exact_coefficient_(factor, reference_dt, terms);
  auto expression_update = prepare_active_axpy_flux_expression_(destination, source, coefficient);
  pops::saxpy(destination, factor, source);
  publish_active_flux_expression_update_(std::move(expression_update));
  count_kernel_();
}

void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
             const field_type& right) const {
  require_same_field_contract_(destination, left, "AMR Program linear combination");
  require_same_field_contract_(destination, right, "AMR Program linear combination");
  auto expression_update = prepare_active_lincomb_flux_expression_(
      destination, left, exact_runtime_coefficient_(left_factor), right,
      exact_runtime_coefficient_(right_factor));
  pops::lincomb(destination, left_factor, left, right_factor, right);
  publish_active_flux_expression_update_(std::move(expression_update));
  count_kernel_();
}
void lincomb(field_type& destination, Real left_factor, const field_type& left, Real right_factor,
             const field_type& right, Real reference_dt,
             std::initializer_list<ExactCoefficientTerm> left_terms,
             std::initializer_list<ExactCoefficientTerm> right_terms) const {
  require_same_field_contract_(destination, left, "AMR Program exact linear combination");
  require_same_field_contract_(destination, right, "AMR Program exact linear combination");
  const auto left_coefficient = exact_coefficient_(left_factor, reference_dt, left_terms);
  const auto right_coefficient = exact_coefficient_(right_factor, reference_dt, right_terms);
  auto expression_update = prepare_active_lincomb_flux_expression_(
      destination, left, left_coefficient, right, right_coefficient);
  pops::lincomb(destination, left_factor, left, right_factor, right);
  publish_active_flux_expression_update_(std::move(expression_update));
  count_kernel_();
}

void commit_many(std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
  const auto& commit_lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
  std::vector<field_type*> targets;
  std::vector<field_type> snapshots;
  std::vector<FluxExpression> snapshot_flux_expressions;
  targets.reserve(commits.size());
  snapshots.reserve(commits.size());
  snapshot_flux_expressions.reserve(commits.size());
  std::exception_ptr snapshot_error;
  try {
    for (const auto& [target, source] : commits) {
      if (target == nullptr || source == nullptr ||
          std::find(targets.begin(), targets.end(), target) != targets.end())
        throw std::invalid_argument("AMR Program commit has null or duplicate storage");
      require_same_field_contract_(*target, *source, "AMR Program commit");
      targets.push_back(target);
      snapshots.emplace_back(*source);
      snapshot_flux_expressions.push_back(active_flux_expression_(*source));
    }
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L, commit_lane) != 0) {
    if (commit_lane.size() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error("AMR Program commit snapshot failed collectively");
  }

  const long commit_count = static_cast<long>(targets.size());
  if (all_reduce_min(commit_count, commit_lane) != all_reduce_max(commit_count, commit_lane))
    throw std::runtime_error("AMR Program commit count differs between MPI ranks");

  std::vector<std::optional<int>> runtime_blocks;
  runtime_blocks.reserve(targets.size());
  std::exception_ptr classification_error;
  try {
    for (field_type* target : targets)
      runtime_blocks.push_back(authenticated_runtime_block_for_state_target_(*target));
  } catch (...) {
    classification_error = std::current_exception();
  }
  if (all_reduce_max(classification_error ? 1L : 0L, commit_lane) != 0) {
    if (commit_lane.size() == 1 && classification_error)
      std::rethrow_exception(classification_error);
    throw std::runtime_error("AMR Program commit target classification failed collectively");
  }

  if (!active_attempt_states_.empty()) {
    std::map<const field_type*, FluxExpression> expression_candidate;
    std::exception_ptr active_commit_error;
    try {
      expression_candidate = active_flux_expressions_;
      for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
        const bool detached_group_candidate =
            std::find(active_attempt_states_.begin(), active_attempt_states_.end(),
                      targets[candidate]) != active_attempt_states_.end();
        const bool prepared_scratch =
            std::any_of(scratches_.begin(), scratches_.end(),
                        [&](const auto& entry) { return &entry.second == targets[candidate]; });
        if (!detached_group_candidate && !prepared_scratch)
          throw std::invalid_argument(
              "active AMR Program commit target is neither a detached group candidate nor a "
              "prepared scratch");
        for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block)
          if (targets[candidate] ==
              &facade_->prepared_amr_block_state(runtime_block, active_level_))
            throw std::invalid_argument(
                "active AMR Program commit cannot target an accepted block carrier");
        expression_candidate[targets[candidate]] = snapshot_flux_expressions[candidate];
      }
    } catch (...) {
      active_commit_error = std::current_exception();
    }
    if (all_reduce_max(active_commit_error ? 1L : 0L, commit_lane) != 0) {
      if (commit_lane.size() == 1 && active_commit_error)
        std::rethrow_exception(active_commit_error);
      throw std::runtime_error("active AMR Program commit failed collectively");
    }
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
      *targets[candidate] = std::move(snapshots[candidate]);
    active_flux_expressions_.swap(expression_candidate);
    return;
  }

  for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
    const std::optional<int> runtime_block = runtime_blocks[candidate];
    const long state_target = runtime_block ? 1L : 0L;
    if (all_reduce_min(state_target, commit_lane) != all_reduce_max(state_target, commit_lane))
      throw std::runtime_error(
          "AMR Program commit target state classification differs between MPI ranks");
    if (runtime_block)
      facade_->validate_prepared_amr_state_publication_candidate(*runtime_block, active_level_,
                                                                 snapshots[candidate]);
  }

  std::vector<std::size_t> accepted_snapshot_by_runtime(static_cast<std::size_t>(n_blocks()),
                                                        snapshots.size());
  std::size_t accepted_targets = 0;
  for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
    if (!runtime_blocks[candidate])
      continue;
    const int runtime_block = *runtime_blocks[candidate];
    if (targets[candidate] != &facade_->prepared_amr_block_state(runtime_block, active_level_))
      continue;
    ++accepted_targets;
    accepted_snapshot_by_runtime[static_cast<std::size_t>(runtime_block)] = candidate;
  }
  if (accepted_targets != 0) {
    std::vector<field_type> publication_candidates;
    std::exception_ptr publication_error;
    try {
      publication_candidates.reserve(static_cast<std::size_t>(n_blocks()));
      for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block) {
        const std::size_t snapshot =
            accepted_snapshot_by_runtime[static_cast<std::size_t>(runtime_block)];
        if (snapshot == snapshots.size())
          publication_candidates.emplace_back(
              facade_->prepared_amr_block_state(runtime_block, active_level_));
        else
          publication_candidates.emplace_back(snapshots[snapshot]);
      }
    } catch (...) {
      publication_error = std::current_exception();
    }
    if (all_reduce_max(publication_error ? 1L : 0L, commit_lane) != 0) {
      if (commit_lane.size() == 1 && publication_error)
        std::rethrow_exception(publication_error);
      throw std::runtime_error("AMR Program accepted-state publication pack failed collectively");
    }
    std::vector<field_type*> program_candidates(static_cast<std::size_t>(n_blocks()), nullptr);
    for (int program_block = 0; program_block < n_blocks(); ++program_block) {
      const int runtime_block = sys_block(program_block);
      program_candidates[static_cast<std::size_t>(program_block)] =
          &publication_candidates[static_cast<std::size_t>(runtime_block)];
    }
    facade_->publish_prepared_amr_program_candidates(
        active_level_, std::span<field_type* const>(program_candidates));
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
      if (!runtime_blocks[candidate])
        *targets[candidate] = std::move(snapshots[candidate]);
    return;
  }

  for (std::size_t candidate = 0; candidate < targets.size(); ++candidate)
    *targets[candidate] = std::move(snapshots[candidate]);
}
