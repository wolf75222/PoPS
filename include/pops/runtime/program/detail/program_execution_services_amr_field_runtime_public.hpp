/// Bind one generated consumer's compact provider view for the active AMR hierarchy level.
/// The program block is authenticated before storage lookup; the qid resolves through that
/// level's sealed plan, so neither generated code nor this context can fall back to ``ctx.aux``.
/// This is a rank-local Fab hot path: collective resource refresh belongs to the enclosing
/// resource traversal, never to one invocation of the local provider callback.
template <int Count>
[[nodiscard]] ProviderStorageView<Dim, Count> provider_values_view(std::string_view consumer_qid,
                                                                   int program_block,
                                                                   std::size_t local_fab) const {
  static_assert(Count >= 0, "a provider consumer count cannot be negative");
  if constexpr (Count == 0) {
    (void)consumer_qid;
    (void)program_block;
    (void)local_fab;
    return {};
  } else {
    const int runtime_block = sys_block(program_block);
    if (resource_epoch_ != runtime_->topology_epoch() ||
        resource_generation_ != runtime_->materialization_generation())
      throw std::logic_error(
          "AMR Program provider binding requires a collectively refreshed resource traversal");
    if (active_level_ < 0 || active_level_ >= nlev())
      throw std::out_of_range("AMR Program provider level lies outside the live hierarchy");
    const field_type& state_field =
        facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
    const auto* const groups =
        facade_->program_prepared_amr_provider_storage_groups_(active_level_);
    const auto& plan =
        facade_->program_prepared_amr_auxiliary_consumer_plan_(consumer_qid, active_level_);
    runtime::system::require_pointwise_provider_groups<Dim, Count>(
        state_field, groups, &plan, "AmrStorageTopologyAdapter provider values");
    return runtime::system::bind_provider_storage_view<Dim, Count>(&plan, groups, local_fab);
  }
}

field_type rhs_scratch_like(const field_type& prototype) const {
  return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
}
field_type scratch_state_like(const field_type& prototype) const {
  return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
}

field_type& rhs_scratch(ProgramCacheSlot slot, int subslot, const field_type& prototype) const {
  return persistent_scratch_(ScratchKind::Rhs, slot, subslot, prototype, prototype.ncomp(),
                             prototype.ghosts());
}
field_type& prepared_rhs_scratch(ProgramCacheSlot slot, int subslot,
                                 const field_type& prototype) const {
  return persistent_scratch_(ScratchKind::Rhs, slot, subslot, prototype, prototype.ncomp(),
                             prototype.ghosts(), false);
}
field_type& scratch_state(ProgramCacheSlot slot, int subslot, const field_type& prototype) const {
  return persistent_scratch_(ScratchKind::State, slot, subslot, prototype, prototype.ncomp(),
                             prototype.ghosts());
}
field_type& prepared_state_scratch(ProgramCacheSlot slot, int subslot,
                                   const field_type& prototype) const {
  return persistent_scratch_(ScratchKind::State, slot, subslot, prototype, prototype.ncomp(),
                             prototype.ghosts(), false);
}
field_type& scalar_scratch(ProgramCacheSlot slot, int subslot, const field_type& prototype,
                           int ncomp = 1, int ghost_depth = 1) const {
  return persistent_scratch_(ScratchKind::Scalar, slot, subslot, prototype, ncomp,
                             uniform_ghosts_(ghost_depth));
}
field_type& prepared_scalar_scratch(ProgramCacheSlot slot, int subslot, const field_type& prototype,
                                    int ncomp = 1, int ghost_depth = 1) const {
  return persistent_scratch_(ScratchKind::Scalar, slot, subslot, prototype, ncomp,
                             uniform_ghosts_(ghost_depth), false);
}

field_type alloc_scalar_field(int ncomp = 1, int ghost_depth = 1) const {
  if (ncomp < 1 || ghost_depth < 0)
    throw std::invalid_argument("AMR Program scalar allocation has an invalid shape");
  const field_type& prototype =
      runtime_->hierarchy().state(static_cast<std::size_t>(active_level_));
  return make_scratch_(prototype, ncomp, uniform_ghosts_(ghost_depth));
}

void rhs_into(int program_block, field_type& stage_state, field_type& rhs, int rate_id) const {
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  require_same_field_contract_(stage_state, rhs, "AMR Program residual");
  auto& point = hot_path_workspace_.direct_point;
  write_boundary_evaluation_point_into(point, rate_id);
  const auto& evaluation = active_attempt_states_.empty()
                               ? facade_->program_evaluate_prepared_amr_block_level_at_(
                                     runtime_block, point, stage_state)
                               : facade_->program_evaluate_prepared_amr_block_level_at_(
                                     runtime_block, point, stage_state, active_level_ - 1,
                                     staged_parent_for_block_(runtime_block));
  copy_valid_(evaluation.residual, rhs);
  if (!active_attempt_states_.empty())
    attach_active_flux_basis_(runtime_block, evaluation, rhs, rate_id,
                              FluxBasisProvider::PreparedResidual);
  count_kernel_();
}

void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
  auto& workspace = hot_path_workspace_;
  workspace.require_bound(requests.size(), "AMR Program RHS group");
  const ExecutionLane& lane = prepared_execution_lane();
  workspace.rhs_ordered.resize(requests.size());
  workspace.rhs_runtime_blocks.resize(requests.size());
  workspace.rhs_evaluation_targets.resize(requests.size());
  workspace.rhs_staged_parents.resize(requests.size());
  workspace.rhs_evaluations.resize(requests.size());
  workspace.rhs_candidates.resize(requests.size());
  workspace.rhs_backups.resize(requests.size());
  std::exception_ptr preparation_error;
  try {
    require_rate_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("AMR Program RHS group cannot be empty");
    if (active_level_ < 0 || static_cast<std::size_t>(active_level_) >= workspace.level_capacity)
      throw std::logic_error("AMR Program RHS group selected level is outside its prepared image");
    std::size_t index = 0;
    for (const RhsGroupRequest& request : requests) {
      require_rate_identity_(request.rate_id);
      const int runtime_block = sys_block(request.block);
      if (request.state == nullptr || request.rhs == nullptr || request.rate_id == group_id ||
          (request.flux_only != 0 && request.flux_only != 1) ||
          std::find(workspace.rhs_runtime_blocks.begin(),
                    workspace.rhs_runtime_blocks.begin() + index,
                    runtime_block) != workspace.rhs_runtime_blocks.begin() + index ||
          std::find(workspace.rhs_rates.begin(), workspace.rhs_rates.begin() + index,
                    request.rate_id) != workspace.rhs_rates.begin() + index ||
          std::find(workspace.rhs_residuals.begin(), workspace.rhs_residuals.begin() + index,
                    request.rhs) != workspace.rhs_residuals.begin() + index)
        throw std::invalid_argument("AMR Program RHS group contains an unsupported request");
      require_same_field_contract_(*request.state, *request.rhs, "AMR Program grouped residual");
      workspace.rhs_ordered[index] = &request;
      workspace.rhs_runtime_blocks[index] = runtime_block;
      workspace.rhs_rates[index] = request.rate_id;
      workspace.rhs_flux_only[index] = request.flux_only;
      workspace.rhs_residuals[index] = request.rhs;
      write_boundary_evaluation_point_into(workspace.rhs_points[index], request.rate_id);
      workspace.rhs_evaluation_targets[index] = {runtime_block, workspace.rhs_points[index].level};
      workspace.rhs_candidates[index] = &workspace.candidate(
          static_cast<std::size_t>(active_level_), static_cast<std::size_t>(runtime_block));
      workspace.rhs_backups[index] = &workspace.backup(static_cast<std::size_t>(active_level_),
                                                       static_cast<std::size_t>(runtime_block));
      ++index;
    }
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR Program RHS group preparation failed collectively");
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if (all_reduce_min(static_cast<long>(workspace.rhs_runtime_blocks[index]), lane) !=
            all_reduce_max(static_cast<long>(workspace.rhs_runtime_blocks[index]), lane) ||
        all_reduce_min(static_cast<long>(workspace.rhs_rates[index]), lane) !=
            all_reduce_max(static_cast<long>(workspace.rhs_rates[index]), lane) ||
        all_reduce_min(static_cast<long>(workspace.rhs_flux_only[index]), lane) !=
            all_reduce_max(static_cast<long>(workspace.rhs_flux_only[index]), lane))
      throw std::invalid_argument("AMR Program RHS group requests differ between execution ranks");
  }

  const long active = active_attempt_states_.empty() ? 0L : 1L;
  if (all_reduce_min(active, lane) != all_reduce_max(active, lane))
    throw std::logic_error("AMR Program RHS group activity differs between execution ranks");
  std::exception_ptr parent_error;
  try {
    std::fill(workspace.rhs_staged_parents.begin(),
              workspace.rhs_staged_parents.begin() + requests.size(), nullptr);
    if (active != 0)
      for (std::size_t index = 0; index < requests.size(); ++index)
        workspace.rhs_staged_parents[index] =
            staged_parent_for_block_(workspace.rhs_runtime_blocks[index]);
  } catch (...) {
    parent_error = std::current_exception();
  }
  if (all_reduce_max(parent_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && parent_error)
      std::rethrow_exception(parent_error);
    throw std::runtime_error("AMR Program RHS group parent preparation failed collectively");
  }

  for (std::size_t index = 0; index < requests.size(); ++index) {
    std::exception_ptr evaluation_error;
    try {
      const RhsGroupRequest& request = *workspace.rhs_ordered[index];
      const auto& evaluation =
          request.flux_only != 0
              ? (active != 0
                     ? facade_->prepare_prepared_amr_block_level_flux_at(
                           workspace.rhs_runtime_blocks[index], workspace.rhs_points[index],
                           *request.state, active_level_ - 1, workspace.rhs_staged_parents[index])
                     : facade_->prepare_prepared_amr_block_level_flux_at(
                           workspace.rhs_runtime_blocks[index], workspace.rhs_points[index],
                           *request.state))
              : (active != 0
                     ? facade_->prepare_prepared_amr_block_level_at(
                           workspace.rhs_runtime_blocks[index], workspace.rhs_points[index],
                           *request.state, active_level_ - 1, workspace.rhs_staged_parents[index])
                     : facade_->prepare_prepared_amr_block_level_at(
                           workspace.rhs_runtime_blocks[index], workspace.rhs_points[index],
                           *request.state));
      workspace.rhs_evaluations[index] = &evaluation;
      copy_valid_(evaluation.residual, *workspace.rhs_candidates[index]);
      device_fence();
    } catch (...) {
      evaluation_error = std::current_exception();
    }
    if (all_reduce_max(evaluation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && evaluation_error)
        std::rethrow_exception(evaluation_error);
      throw std::runtime_error("AMR Program RHS group evaluation failed collectively");
    }
  }

  FluxExpressionRegistry candidate_registry;
  std::vector<std::size_t> candidate_counts;
  std::uint64_t candidate_identity = 0;
  if (active != 0) {
    std::exception_ptr flux_preparation_error;
    try {
      // Bound v5 tables own their dense cursors and resident payloads.  Do not copy the legacy
      // registry/count packs on this hot route: they are only the dynamic fallback authority.
      if (!static_flux_tables_.bound) {
        candidate_registry = active_flux_expressions_;
        candidate_counts = active_flux_basis_counts_;
        candidate_identity = next_active_flux_basis_identity_;
      }
      const ::pops::amr::ClockWindow interval{
          {active_level_, workspace.rhs_points.front().tick, current_interval_begin_phase_,
           current_interval_start_time_},
          {active_level_, workspace.rhs_points.front().tick, current_interval_end_phase_,
           current_interval_start_time_ + current_dt_}};
      for (std::size_t index = 0; index < requests.size(); ++index) {
        const RhsGroupRequest& request = *workspace.rhs_ordered[index];
        prepare_active_flux_basis_impl_(
            workspace.rhs_runtime_blocks[index], workspace.rhs_points[index], request.rate_id,
            request.flux_only != 0 ? FluxBasisProvider::PreparedDefaultFlux
                                   : FluxBasisProvider::PreparedResidual,
            workspace.rhs_evaluations[index]->topology_epoch,
            workspace.rhs_evaluations[index]->materialization_generation, *request.rhs,
            workspace.rhs_evaluations[index], nullptr, nullptr, interval, candidate_registry,
            static_flux_tables_.bound ? active_flux_basis_counts_ : candidate_counts,
            static_flux_tables_.bound ? next_active_flux_basis_identity_ : candidate_identity);
      }
    } catch (...) {
      flux_preparation_error = std::current_exception();
    }
    if (all_reduce_max(flux_preparation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && flux_preparation_error)
        std::rethrow_exception(flux_preparation_error);
      throw std::runtime_error("AMR Program RHS group flux preparation failed collectively");
    }
  }

  std::exception_ptr evaluation_publication_validation_error;
  try {
    facade_->validate_prepared_amr_block_level_batch(std::span<const std::pair<int, int>>(
        workspace.rhs_evaluation_targets.data(), requests.size()));
  } catch (...) {
    evaluation_publication_validation_error = std::current_exception();
  }
  if (all_reduce_max(evaluation_publication_validation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && evaluation_publication_validation_error)
      std::rethrow_exception(evaluation_publication_validation_error);
    throw std::runtime_error(
        "AMR Program RHS group prepared evaluation publication failed collectively");
  }

  std::exception_ptr backup_error;
  try {
    for (std::size_t index = 0; index < requests.size(); ++index)
      copy_valid_(*workspace.rhs_ordered[index]->rhs, *workspace.rhs_backups[index]);
    device_fence();
  } catch (...) {
    backup_error = std::current_exception();
  }
  if (all_reduce_max(backup_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && backup_error)
      std::rethrow_exception(backup_error);
    throw std::runtime_error("AMR Program RHS group output snapshot failed collectively");
  }

  // Profiling is permitted to allocate/throw.  Account for the completed candidate work before
  // the no-throw publication boundary so a profiler failure cannot expose a partial commit.
  std::exception_ptr kernel_count_error;
  try {
    count_kernel_(static_cast<std::int64_t>(requests.size()));
  } catch (...) {
    kernel_count_error = std::current_exception();
  }
  if (all_reduce_max(kernel_count_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && kernel_count_error)
      std::rethrow_exception(kernel_count_error);
    throw std::runtime_error("AMR Program RHS group profiling failed collectively");
  }

  std::exception_ptr publication_error;
  try {
    for (std::size_t index = 0; index < requests.size(); ++index)
      copy_valid_(*workspace.rhs_candidates[index], *workspace.rhs_ordered[index]->rhs);
    device_fence();
  } catch (...) {
    publication_error = std::current_exception();
  }
  if (all_reduce_max(publication_error ? 1L : 0L, lane) != 0) {
    std::exception_ptr rollback_error;
    try {
      for (std::size_t index = 0; index < requests.size(); ++index)
        copy_valid_(*workspace.rhs_backups[index], *workspace.rhs_ordered[index]->rhs);
      device_fence();
    } catch (...) {
      rollback_error = std::current_exception();
    }
    if (all_reduce_max(rollback_error ? 1L : 0L, lane) != 0)
      throw std::runtime_error("AMR Program RHS group output rollback failed collectively");
    if (lane.size() == 1 && publication_error)
      std::rethrow_exception(publication_error);
    throw std::runtime_error("AMR Program RHS group publication failed collectively");
  }

  if (active != 0) {
    facade_->publish_prepared_amr_block_level_batch(std::span<const std::pair<int, int>>(
        workspace.rhs_evaluation_targets.data(), requests.size()));
    if (!static_flux_tables_.bound) {
      static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
      static_assert(std::is_nothrow_swappable_v<std::vector<std::size_t>>);
      active_flux_expressions_.swap(candidate_registry);
      active_flux_basis_counts_.swap(candidate_counts);
      next_active_flux_basis_identity_ = candidate_identity;
    }
  } else {
    facade_->publish_prepared_amr_block_level_batch(std::span<const std::pair<int, int>>(
        workspace.rhs_evaluation_targets.data(), requests.size()));
  }
}

void neg_div_flux_default_into(int program_block, field_type& stage_state, field_type& rhs,
                               int rate_id) const {
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  require_same_field_contract_(stage_state, rhs, "AMR Program default flux residual");
  auto& point = hot_path_workspace_.direct_point;
  write_boundary_evaluation_point_into(point, rate_id);
  const auto& evaluation = active_attempt_states_.empty()
                               ? facade_->program_evaluate_prepared_amr_block_level_flux_at_(
                                     runtime_block, point, stage_state)
                               : facade_->program_evaluate_prepared_amr_block_level_flux_at_(
                                     runtime_block, point, stage_state, active_level_ - 1,
                                     staged_parent_for_block_(runtime_block));
  copy_valid_(evaluation.residual, rhs);
  if (!active_attempt_states_.empty())
    attach_active_flux_basis_(runtime_block, evaluation, rhs, rate_id,
                              FluxBasisProvider::PreparedDefaultFlux);
  count_kernel_();
}
void source_default_into(int program_block, field_type& stage_state, field_type& rhs) const {
  const int runtime_block = sys_block(program_block);
  require_same_field_contract_(stage_state, rhs, "AMR Program default source residual");
  auto& point = hot_path_workspace_.direct_point;
  write_boundary_evaluation_point_into(point, 0);
  if (active_attempt_states_.empty())
    facade_->prepared_amr_block_level_source_into_at(runtime_block, point, stage_state, rhs);
  else
    facade_->prepared_amr_block_level_source_into_at(runtime_block, point, stage_state, rhs,
                                                     active_level_ - 1,
                                                     staged_parent_for_block_(runtime_block));
  clear_active_flux_expression_(rhs);
  count_kernel_();
}

void apply_source_mask(field_type& rhs, std::initializer_list<int> keep) const {
  pops::runtime::program::apply_component_keep_mask(rhs, keep);
  count_kernel_();
}

[[nodiscard]] NewtonOptions block_newton_options(int program_block) const {
  return facade_->program_block_newton_options_(sys_block(program_block));
}

[[nodiscard]] SolveOutcome solve_source_default(int program_block, field_type& stage_state, Real dt,
                                                const NewtonOptions& options) const {
  const int runtime_block = sys_block(program_block);
  auto& point = hot_path_workspace_.direct_point;
  write_boundary_evaluation_point_into(point, 0);
  SolveOutcome outcome = active_attempt_states_.empty()
                             ? facade_->solve_prepared_amr_block_level_source_at(
                                   runtime_block, point, stage_state, dt, options)
                             : facade_->solve_prepared_amr_block_level_source_at(
                                   runtime_block, point, stage_state, dt, options,
                                   active_level_ - 1, staged_parent_for_block_(runtime_block));
  count_kernel_();
  return outcome;
}

void publish_newton_report(int program_block, const SolveReport& solve) const {
  facade_->publish_newton_report(sys_block(program_block), solve);
}

void require_cartesian_generated_operator(int program_block, const std::string& operation) const {
  const int runtime_block = sys_block(program_block);
  if (operation.empty())
    throw std::invalid_argument("AMR generated operator requires an operation identity");
  if (operation == "named_flux")
    require_named_flux_execution_envelope_(runtime_block);
}

void prepare_generated_state(int program_block, field_type& stage_state, int rate_id) const {
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  auto& point = hot_path_workspace_.direct_point;
  write_boundary_evaluation_point_into(point, rate_id);
  if (active_attempt_states_.empty())
    facade_->prepare_generated_amr_block_level_state(runtime_block, point, stage_state);
  else
    facade_->prepare_generated_amr_block_level_state(runtime_block, point, stage_state,
                                                     active_level_ - 1,
                                                     staged_parent_for_block_(runtime_block));
}
