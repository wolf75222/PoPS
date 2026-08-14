void register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const hierarchy_tensor_provider_type> provider) const {
  require_facade_execution_();
  if (!hierarchy_tensor_solver_registry_)
    throw std::logic_error("AMR hierarchy tensor-solver registry is unavailable");
  facade_->register_program_hierarchy_tensor_solver_provider(std::move(provider));
}

void configure_hierarchy_tensor_solver(int program_block, int components,
                                       const std::string& provider_identity,
                                       const std::string& plan_identity,
                                       const std::string& operator_contract_identity,
                                       const std::vector<std::string>& assembly_field_slots,
                                       const std::string& solution_field_slot,
                                       const PreparedProviderOptions& options) const {
  require_facade_execution_();
  if (!hierarchy_tensor_solver_registry_)
    throw std::logic_error("AMR hierarchy tensor-solver registry is unavailable");

  const ExecutionLane& lane = prepared_execution_lane();
  std::optional<HierarchyTensorSelection> staged;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    (void)sys_block(program_block);
    if (components < 1 || provider_identity.empty() || plan_identity.empty() ||
        operator_contract_identity.empty() || assembly_field_slots.empty() ||
        solution_field_slot.empty() ||
        std::any_of(
            assembly_field_slots.begin(), assembly_field_slots.end(),
            [](const std::string& slot) { return slot.empty(); }) ||
        std::set<std::string>(assembly_field_slots.begin(), assembly_field_slots.end()).size() !=
            assembly_field_slots.size())
      throw std::invalid_argument(
          "AMR hierarchy tensor solver requires a complete exact field-slot envelope");
    ExactContractBuilder contract;
    contract.text("pops.hierarchy.tensor-solver-selection")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .scalar(std::int32_t{program_block})
        .scalar(std::int32_t{components})
        .text(provider_identity)
        .text(plan_identity)
        .text(operator_contract_identity)
        .sequence(assembly_field_slots,
                  [](ExactContractBuilder& item, const std::string& slot) { item.text(slot); })
        .text(solution_field_slot)
        .bytes(options.exact_contract());
    staged.emplace(HierarchyTensorSelection{
        program_block, components, provider_identity, plan_identity, operator_contract_identity,
        assembly_field_slots, solution_field_slot, options, std::move(contract).release()});
    if (hierarchy_tensor_selection_ &&
        hierarchy_tensor_selection_->exact_contract != staged->exact_contract)
      throw std::logic_error("AMR hierarchy tensor solver is already configured differently");
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure, lane) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR hierarchy tensor selection failed on another MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-hierarchy-tensor-selection", staged->exact_contract}}, lane))
    throw std::invalid_argument("AMR hierarchy tensor selection differs between MPI ranks");

  if (hierarchy_tensor_selection_ &&
      hierarchy_tensor_selection_->exact_contract == staged->exact_contract) {
    (void)configured_hierarchy_tensor_solver_();
    return;
  }

  PreparedHierarchyTensorState prepared = prepare_hierarchy_tensor_solver_(*staged);
  hierarchy_tensor_selection_ = std::move(staged);
  hierarchy_tensor_solver_ = std::move(prepared.solver);
  hierarchy_tensor_boundaries_ = std::move(prepared.boundaries);
  hierarchy_tensor_topology_epoch_ = runtime_->topology_epoch();
  hierarchy_tensor_materialization_generation_ = runtime_->materialization_generation();
}

SolveOutcome solve_hierarchy_tensor(int program_block, int components, Real relative_tolerance,
                                    Real absolute_tolerance, int maximum_iterations) const {
  require_hierarchy_tensor_binding_(program_block, components);
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider) {
    SolveReport report;
    report.mark_failed(SolveStatus::kInvalidInput, SolveAction::kRejectAttempt);
    return SolveOutcome::collective_lane(std::move(report), prepared_execution_lane());
  }
  return solve_prepared_hierarchy_tensor_collectively(
      solver,
      HierarchyTensorSolveControls{relative_tolerance, absolute_tolerance, maximum_iterations},
      prepared_execution_lane());
}

field_type& hierarchy_solution() const {
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
    throw std::logic_error(
        "provider-owned hierarchy solution requested on a prepared Krylov fallback path");
  return solver.solution(active_level_);
}
field_type& linear_solution(field_type& fallback) const {
  require_same_layout_(fallback, state(0), "AMR Program linear solution");
  if (!hierarchy_tensor_selection_)
    return fallback;
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  return solver.execution_path() == HierarchyTensorSolverExecutionPath::DirectProvider
             ? solver.solution(active_level_)
             : fallback;
}
void stage_linear_initial_guess() const {
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
    throw std::logic_error("hierarchy initial guess requires direct provider execution");
  solver.stage_initial_guess(active_level_, nullptr);
}
void stage_linear_initial_guess(const field_type& guess) const {
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
    throw std::logic_error("hierarchy initial guess requires direct provider execution");
  solver.stage_initial_guess(active_level_, &guess);
}

ClockScheduleState::SubcycleScope subcycle_scope(const std::string& parent,
                                                 const std::string& child, int count) const {
  return clock_schedule_.subcycle(parent, child, count);
}
LogicalEvaluationScope logical_evaluation_scope(int iteration, int count) const {
  return LogicalEvaluationScope(*this, iteration, count);
}
void synchronize_sample_and_hold(const std::string& source, const std::string& target, int step,
                                 Real offset) const {
  clock_schedule_.synchronize_sample_and_hold(source, target, step, static_cast<double>(offset));
}

int macro_step() const {
  return facade_->macro_step();
}
Real physical_time() const {
  return static_cast<Real>(facade_->time());
}

void record_scalar(const std::string& name, Real value) const {
  facade_->record_program_diagnostic(name, static_cast<double>(value));
}
void record_balance_term(const std::string& route, const std::string& term, Real value) const {
  facade_->record_program_balance_term(route, term, static_cast<double>(value));
}
bool balance_consumer_is_due(const std::string& contract, const std::string& route,
                             int every_n) const {
  return facade_->program_balance_consumer_is_due(contract, route, every_n);
}
void note_automatic_balance_capture_due(bool due) const {
  runtime_state().note_automatic_balance_capture_due(due, "AmrProgramContext");
}
void note_step_projection(const std::string& name) const {
  runtime_state().note_step_projection(name);
}
void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  facade_->profiler_handle().record(name, std::chrono::duration<double>(elapsed).count());
}

runtime_state_type& runtime_state() const {
  return facade_->program_runtime_state_();
}

const PreparedVectorDistribution<Dim>& program_resource_vector_distribution() const {
  refresh_resources_();
  vector_distribution_ = runtime_->hierarchy()
                                 .layout(static_cast<std::size_t>(active_level_))
                                 .distribution()
                                 .replicated()
                             ? PreparedVectorDistribution<Dim>::replicated()
                             : PreparedVectorDistribution<Dim>::distributed();
  return vector_distribution_;
}
int program_resource_field_level() const noexcept {
  return active_level_;
}
void configure_program_resource_field_nullspace(FieldNullspacePlan<Dim>& plan) const {
  Real cell_measure = Real(1);
  const Geometry<Dim> geom = geometry();
  for (int axis = 0; axis < Dim; ++axis)
    cell_measure *= geom.spacing(axis);
  for (FieldNullspaceBasis<Dim>& basis : plan.bases)
    basis.cell_measure = {cell_measure};
}

SolveOutcome solve_prepared_linear(const PreparedAffineLinearProblem<Dim>& problem,
                                   KrylovWorkspace<Dim>& workspace, field_type& solution,
                                   const field_type& rhs,
                                   const KrylovControls<Dim>& controls) const {
  const ExecutionLane& lane = workspace.execution_lane();
  const ExecutionLane& runtime_lane = prepared_execution_lane();
  std::exception_ptr local_error;
  try {
    if (!lane.active() || lane.identity().empty() || !runtime_lane.active() ||
        runtime_lane.identity().empty() || !lane.congruent_with(runtime_lane))
      throw std::invalid_argument(
          "AMR prepared linear solve requires its runtime-authenticated private lane");
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR prepared linear solve lane validation failed collectively");
  }
  return pops::solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls);
}

OperatorEvaluationSnapshot operator_evaluation_snapshot(OperatorFingerprint authority,
                                                        const field_type& prototype,
                                                        OperatorFingerprint resources) const {
  if (operator_snapshot_revision_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AMR Program operator snapshot revision exhausted uint64_t");
  const OperatorFingerprint topology = operator_topology_(prototype);
  OperatorEvaluationSnapshot snapshot =
      current_operator_snapshot_(authority, topology, resources, ++operator_snapshot_revision_);
  if (!snapshot.valid())
    throw std::runtime_error("AMR Program produced an invalid exact operator snapshot");
  active_operator_snapshot_ = snapshot;
  return snapshot;
}

OperatorEvaluationSnapshot probe_operator_evaluation(OperatorFingerprint authority,
                                                     OperatorFingerprint topology,
                                                     OperatorFingerprint resources,
                                                     std::uint64_t revision) const {
  const bool active = active_operator_snapshot_ && active_operator_snapshot_->revision == revision;
  OperatorEvaluationSnapshot probe =
      current_operator_snapshot_(authority, topology, resources, active ? revision : 0);
  if (!active || probe != *active_operator_snapshot_) {
    active_operator_snapshot_.reset();
    probe.revision = 0;
  }
  return probe;
}

void set_field_boundary_kernel(const std::string& provider_slot,
                               const CompiledFieldBoundaryKernel<Dim>& kernel) const {
  facade_->set_field_boundary_kernel(provider_slot, kernel);
}
void set_field_logical_timepoint(const std::string& provider_slot,
                                 const FieldLogicalTimePoint& point) const {
  facade_->set_field_logical_timepoint(provider_slot, point);
}

bool has_boundary_linearization(int program_block) const noexcept {
  try {
    return facade_ != nullptr &&
           facade_->has_prepared_amr_block_boundary_linearization(sys_block(program_block));
  } catch (...) {
    return false;
  }
}
void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                               int program_block, field_type& state, field_type& residual,
                               const block_boundary_session_type& boundary) const {
  require_block_boundary_session_(point, program_block, boundary, "AMR boundary residual");
  facade_->prepared_amr_block_level_boundary_residual_into_at(boundary.runtime_block_, point, state,
                                                              residual);
}
void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                          int program_block, field_type& state, const field_type& direction,
                          field_type& result, const block_boundary_session_type& boundary) const {
  require_block_boundary_session_(point, program_block, boundary, "AMR boundary JVP");
  facade_->prepared_amr_block_level_boundary_jvp_into_at(boundary.runtime_block_, point, state,
                                                         direction, result);
}
void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int program_block,
                      field_type& state, field_type& residual, bool flux_only,
                      const block_boundary_session_type& boundary) const {
  require_block_boundary_session_(point, program_block, boundary, "AMR core RHS");
  facade_->prepared_amr_block_level_rhs_core_into_at(boundary.runtime_block_, point, state,
                                                     residual, flux_only);
}
void rhs_jacvec_pair_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                             int first_block, field_type& first_state, field_type& first_result,
                             bool first_flux_only, int second_block, field_type& second_state,
                             field_type& second_result, bool second_flux_only) const {
  require_boundary_point_(point, "AMR coupled Jacobian residual");
  if (first_block == second_block || n_blocks() != 2 || !(point.dt > 0.0))
    throw std::invalid_argument(
        "AMR coupled Jacobian residual requires two distinct complete Program blocks");
  std::unique_lock scratch_lock(coupled_jacvec_mutex_, std::try_to_lock);
  if (!scratch_lock.owns_lock())
    throw std::logic_error("AMR coupled Jacobian scratch is already in use");
  CoupledJacvecLevelScratch& scratch = require_coupled_jacvec_scratch_(
      first_block, first_state, first_result, second_block, second_state, second_result);
  // The generated block operators and the prepared multi-block coupling/interface scheduler
  // remain the only numerical engines. Evaluate the detached block candidates first, then run
  // the exact scheduler on copies and recover its rate from the transactional candidate delta.
  const int first_runtime = sys_block(first_block);
  const int second_runtime = sys_block(second_block);
  field_type& first_candidate = *scratch.residual[static_cast<std::size_t>(first_runtime)];
  field_type& second_candidate = *scratch.residual[static_cast<std::size_t>(second_runtime)];
  field_type& first_coupled = *scratch.coupled[static_cast<std::size_t>(first_runtime)];
  field_type& second_coupled = *scratch.coupled[static_cast<std::size_t>(second_runtime)];
  first_candidate.set_val(Real(0));
  second_candidate.set_val(Real(0));
  copy_full_(first_state, first_coupled);
  copy_full_(second_state, second_coupled);
  const auto& first_evaluation =
      first_flux_only
          ? facade_->evaluate_prepared_amr_block_level_flux_at(first_runtime, point, first_state)
          : facade_->evaluate_prepared_amr_block_level_at(first_runtime, point, first_state);
  copy_valid_(first_evaluation.residual, first_candidate);
  const auto& second_evaluation =
      second_flux_only
          ? facade_->evaluate_prepared_amr_block_level_flux_at(second_runtime, point, second_state)
          : facade_->evaluate_prepared_amr_block_level_at(second_runtime, point, second_state);
  copy_valid_(second_evaluation.residual, second_candidate);

  std::array<field_type*, 2> candidates{};
  candidates[static_cast<std::size_t>(first_block)] = &first_coupled;
  candidates[static_cast<std::size_t>(second_block)] = &second_coupled;
  if (!interface_flux_ledger_ || !interface_flux_ledger_->in_transaction())
    throw std::logic_error(
        "AMR coupled Jacobian residual requires its active interface-ledger transaction");
  runtime::multiblock::InterfaceFluxFragmentPublication publication{
      interface_flux_ledger_.get(),
      runtime_->topology_epoch(),
      nlev(),
      {active_level_, static_cast<std::int64_t>(facade_->macro_step()), point.stage_fraction,
       point.physical_time},
      "program-jacvec-pair",
      active_subcycling_window_,
      exact_binary_rational_(static_cast<Real>(point.dt) /
                             (active_subcycling_window_.end.physical_time -
                              active_subcycling_window_.begin.physical_time)),
      true};
  interface_flux_ledger_->begin();
  try {
    (void)facade_->apply_prepared_amr_program_candidates(
        active_level_, static_cast<Real>(point.dt),
        std::span<field_type* const>(candidates.data(), candidates.size()), point, &publication);
    interface_flux_ledger_->rollback();
  } catch (...) {
    if (interface_flux_ledger_->transaction_depth() > 1)
      interface_flux_ledger_->rollback();
    throw;
  }
  saxpy(first_coupled, Real(-1), first_state);
  saxpy(second_coupled, Real(-1), second_state);
  saxpy(first_candidate, Real(1) / static_cast<Real>(point.dt), first_coupled);
  saxpy(second_candidate, Real(1) / static_cast<Real>(point.dt), second_coupled);
  copy_valid_(first_candidate, first_result);
  copy_valid_(second_candidate, second_result);
}
template <class Function>
void evaluate_with_field_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  const std::string& provider_slot, int program_block,
                                  field_type& perturbed, const field_type& accepted,
                                  Function&& evaluate) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::function<void()> prepared_evaluate;
  std::string request_contract;
  std::exception_ptr local_error;
  try {
    require_boundary_point_(point, "AMR perturbed field-state provider");
    if (provider_slot.empty() || sys_block(program_block) != 0)
      throw std::invalid_argument(
          "AMR perturbed field-state provider requires one exact mono-block field route");
    require_same_field_contract_(perturbed, accepted, "AMR perturbed field state");
    prepared_evaluate = std::function<void()>(std::forward<Function>(evaluate));
    ExactContractBuilder request;
    request.text("pops.amr-program.scalar-field-candidate-route")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(provider_slot)
        .text(point.clock)
        .scalar(point.tick)
        .scalar(point.level)
        .scalar(point.substep)
        .scalar(point.stage)
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time)
        .scalar(std::int32_t{program_block})
        .scalar(std::int32_t{0})
        .bytes(elliptic_contract_detail::field_layout_contract(perturbed))
        .bytes(elliptic_contract_detail::field_layout_contract(accepted));
    request_contract = std::move(request).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::invalid_argument("AMR perturbed field-state route preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-scalar-field-candidate-route"),
            std::string_view(request_contract)}},
          lane))
    throw std::invalid_argument("AMR perturbed field-state route differs between MPI ranks");
  facade_->with_program_field_candidate_at(point, provider_slot, active_level_, perturbed,
                                           std::move(prepared_evaluate));
}

[[nodiscard]] SolveOutcome solve_fields_from_state_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int program_block, field_type& stage) const {
  refresh_resources_();
  const ExecutionLane& lane = prepared_execution_lane();
  std::vector<const field_type*> runtime_stages;
  std::string request_contract;
  std::exception_ptr local_error;
  try {
    require_boundary_point_(point, "AMR Program single-state field solve");
    if (provider_slot.empty())
      throw std::invalid_argument("AMR Program field solve requires an exact provider slot");
    const int runtime_block = sys_block(program_block);
    const field_type& live = facade_->prepared_amr_block_state(runtime_block, active_level_);
    require_same_field_contract_(stage, live, "AMR Program field stage override");
    runtime_stages.assign(static_cast<std::size_t>(n_blocks()), nullptr);
    runtime_stages[static_cast<std::size_t>(runtime_block)] = &stage;
    ExactContractBuilder request;
    request.text("pops.amr-program.single-field-route")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(provider_slot)
        .text(point.clock)
        .scalar(point.tick)
        .scalar(point.level)
        .scalar(point.substep)
        .scalar(point.stage)
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time)
        .scalar(std::int32_t{program_block})
        .scalar(std::int32_t{runtime_block})
        .presence(true)
        .bytes(elliptic_contract_detail::field_layout_contract(stage));
    request_contract = std::move(request).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::invalid_argument(
        "AMR Program single-state field route preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-single-field-route"),
            std::string_view(request_contract)}},
          lane))
    throw std::invalid_argument("AMR Program single-state field route differs between MPI ranks");
  return facade_->solve_program_field_from_blocks_at(point, provider_slot, active_level_,
                                                     runtime_stages);
}

[[nodiscard]] SolveOutcome solve_fields_from_blocks_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t value_id,
    std::string_view field, std::initializer_list<FieldStageOverride> overrides) const {
  refresh_resources_();
  const ExecutionLane& lane = prepared_execution_lane();
  std::optional<GeneratedFieldRoute> candidate;
  std::vector<const field_type*> runtime_stages;
  std::map<std::int64_t, GeneratedFieldRoute> detached_cache_entry;
  typename std::map<std::int64_t, GeneratedFieldRoute>::node_type detached_cache_node;
  std::string request_contract;
  std::exception_ptr local_error;
  try {
    require_boundary_point_(point, "AMR Program simultaneous field solve");
    if (value_id < 0 || field.empty() || overrides.size() == 0)
      throw std::invalid_argument(
          "AMR Program simultaneous field solve requires an IR identity, field, and stages");

    candidate.emplace();
    candidate->field.assign(field.data(), field.size());
    runtime_stages.assign(static_cast<std::size_t>(n_blocks()), nullptr);
    std::vector<const field_type*> unique_stages;
    unique_stages.reserve(overrides.size());
    ExactContractBuilder request;
    request.text("pops.amr-program.simultaneous-field-route")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(value_id)
        .text(candidate->field)
        .text(point.clock)
        .scalar(point.tick)
        .scalar(point.level)
        .scalar(point.substep)
        .scalar(point.stage)
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time)
        .scalar(static_cast<std::uint64_t>(overrides.size()));
    for (const FieldStageOverride& override_value : overrides) {
      if (std::find(candidate->program_blocks.begin(), candidate->program_blocks.end(),
                    override_value.program_block) != candidate->program_blocks.end())
        throw std::invalid_argument(
            "AMR Program simultaneous field solve contains a duplicate Program block");
      if (override_value.state != nullptr && std::find(unique_stages.begin(), unique_stages.end(),
                                                       override_value.state) != unique_stages.end())
        throw std::invalid_argument(
            "AMR Program simultaneous field solve aliases two stage overrides");
      const int runtime_block = sys_block(override_value.program_block);
      if (std::find(candidate->runtime_blocks.begin(), candidate->runtime_blocks.end(),
                    runtime_block) != candidate->runtime_blocks.end())
        throw std::invalid_argument(
            "AMR Program simultaneous field solve maps two stages to one runtime block");
      if (override_value.state != nullptr)
        require_same_field_contract_(
            *override_value.state, facade_->prepared_amr_block_state(runtime_block, active_level_),
            "AMR Program simultaneous field stage override");
      candidate->program_blocks.push_back(override_value.program_block);
      candidate->runtime_blocks.push_back(runtime_block);
      runtime_stages[static_cast<std::size_t>(runtime_block)] = override_value.state;
      if (override_value.state != nullptr)
        unique_stages.push_back(override_value.state);
      request.scalar(std::int32_t{override_value.program_block})
          .scalar(std::int32_t{runtime_block})
          .presence(override_value.state != nullptr);
      if (override_value.state != nullptr)
        request.bytes(elliptic_contract_detail::field_layout_contract(*override_value.state));
    }
    const auto existing = generated_field_routes_.find(value_id);
    request.presence(existing != generated_field_routes_.end());
    if (existing == generated_field_routes_.end()) {
      detached_cache_entry.emplace(value_id, *candidate);
      detached_cache_node = detached_cache_entry.extract(value_id);
    } else if (existing->second != *candidate) {
      throw std::logic_error(
          "AMR Program simultaneous field IR identity changed its qualified route");
    }
    request_contract = std::move(request).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::invalid_argument(
        "AMR Program simultaneous field route preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-simultaneous-field-route"),
            std::string_view(request_contract)}},
          lane))
    throw std::invalid_argument("AMR Program simultaneous field route differs between MPI ranks");

  SolveOutcome outcome = facade_->solve_program_field_from_blocks_at(point, candidate->field,
                                                                     active_level_, runtime_stages);
  if (!detached_cache_node.empty()) {
    const auto insertion = generated_field_routes_.insert(std::move(detached_cache_node));
    if (!insertion.inserted)
      throw std::logic_error(
          "AMR Program simultaneous field route cache changed during collective execution");
  }
  return outcome;
}

[[nodiscard]] SolveOutcome solve_default_field_on_coarse_level() const {
  if (active_level_ != 0)
    throw std::logic_error(
        "AMR Program coarse-to-fine auxiliary injection is not a fine-level solve");
  refresh_resources_();
  return facade_->solve_program_default_field(0);
}
