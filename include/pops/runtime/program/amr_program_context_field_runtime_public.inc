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
    const field_type& state_field = facade_->prepared_amr_block_state(runtime_block, active_level_);
    const auto* const groups = facade_->prepared_amr_provider_storage_groups(active_level_);
    const auto& plan =
        facade_->prepared_amr_auxiliary_consumer_plan(std::string(consumer_qid), active_level_);
    runtime::system::require_pointwise_provider_groups<Dim, Count>(
        state_field, groups, &plan, "AmrProgramContext provider values");
    return runtime::system::bind_provider_storage_view<Dim, Count>(&plan, groups, local_fab);
  }
}

field_type rhs_scratch_like(const field_type& prototype) const {
  return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
}
field_type scratch_state_like(const field_type& prototype) const {
  return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
}

field_type& rhs_scratch(std::int64_t value_id, int subslot, const field_type& prototype) const {
  return persistent_scratch_(ScratchKind::Rhs, value_id, subslot, prototype, prototype.ncomp(),
                             prototype.ghosts());
}
field_type& scratch_state(std::int64_t value_id, int subslot, const field_type& prototype) const {
  return persistent_scratch_(ScratchKind::State, value_id, subslot, prototype, prototype.ncomp(),
                             prototype.ghosts());
}
field_type& scalar_scratch(std::int64_t value_id, int subslot, const field_type& prototype,
                           int ncomp = 1, int ghost_depth = 1) const {
  return persistent_scratch_(ScratchKind::Scalar, value_id, subslot, prototype, ncomp,
                             uniform_ghosts_(ghost_depth));
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
  const auto point = boundary_evaluation_point(rate_id);
  const auto& evaluation =
      active_attempt_states_.empty()
          ? facade_->evaluate_prepared_amr_block_level_at(runtime_block, point, stage_state)
          : facade_->evaluate_prepared_amr_block_level_at(runtime_block, point, stage_state,
                                                          active_level_ - 1,
                                                          staged_parent_for_block_(runtime_block));
  copy_valid_(evaluation.residual, rhs);
  if (!active_attempt_states_.empty())
    attach_active_flux_basis_(runtime_block, evaluation, rhs, rate_id,
                              FluxBasisProvider::PreparedResidual);
  count_kernel_();
}

void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
  require_rate_identity_(group_id);
  std::vector<field_type> candidates;
  candidates.reserve(requests.size());
  for (const RhsGroupRequest& request : requests) {
    if (request.state == nullptr || request.rhs == nullptr || request.rate_id < 0 ||
        request.rate_id == group_id || request.flux_only != 0)
      throw std::invalid_argument("AMR Program RHS group contains an unsupported request");
    candidates.push_back(rhs_scratch_like(*request.rhs));
  }
  std::size_t index = 0;
  for (const RhsGroupRequest& request : requests)
    rhs_into(request.block, *request.state, candidates[index++], request.rate_id);
  index = 0;
  for (const RhsGroupRequest& request : requests) {
    copy_valid_(candidates[index], *request.rhs);
    copy_active_flux_expression_(candidates[index], *request.rhs);
    clear_active_flux_expression_(candidates[index]);
    ++index;
  }
}

void neg_div_flux_default_into(int program_block, field_type& stage_state, field_type& rhs,
                               int rate_id) const {
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  require_same_field_contract_(stage_state, rhs, "AMR Program default flux residual");
  const auto point = boundary_evaluation_point(rate_id);
  const auto& evaluation =
      active_attempt_states_.empty()
          ? facade_->evaluate_prepared_amr_block_level_flux_at(runtime_block, point, stage_state)
          : facade_->evaluate_prepared_amr_block_level_flux_at(
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
  const auto point = boundary_evaluation_point(0);
  if (active_attempt_states_.empty())
    facade_->prepared_amr_block_level_source_into_at(runtime_block, point, stage_state, rhs);
  else
    facade_->prepared_amr_block_level_source_into_at(runtime_block, point, stage_state, rhs,
                                                     active_level_ - 1,
                                                     staged_parent_for_block_(runtime_block));
  clear_active_flux_expression_(rhs);
  count_kernel_();
}

[[nodiscard]] SolveOutcome solve_source_default(int program_block, field_type& stage_state, Real dt,
                                                const NewtonOptions& options) const {
  const int runtime_block = sys_block(program_block);
  const auto point = boundary_evaluation_point(0);
  SolveOutcome outcome = active_attempt_states_.empty()
                             ? facade_->solve_prepared_amr_block_level_source_at(
                                   runtime_block, point, stage_state, dt, options)
                             : facade_->solve_prepared_amr_block_level_source_at(
                                   runtime_block, point, stage_state, dt, options,
                                   active_level_ - 1, staged_parent_for_block_(runtime_block));
  count_kernel_();
  return outcome;
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
  const auto point = boundary_evaluation_point(rate_id);
  if (active_attempt_states_.empty())
    facade_->prepare_generated_amr_block_level_state(runtime_block, point, stage_state);
  else
    facade_->prepare_generated_amr_block_level_state(runtime_block, point, stage_state,
                                                     active_level_ - 1,
                                                     staged_parent_for_block_(runtime_block));
}
