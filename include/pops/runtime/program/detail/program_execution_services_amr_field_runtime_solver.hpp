void register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const hierarchy_tensor_provider_type> provider) const {
  if (preparation_view_ == nullptr)
    throw std::logic_error("AMR hierarchy tensor provider registration is preparation-only");
  if (!hierarchy_tensor_solver_registry_)
    throw std::logic_error("AMR hierarchy tensor-solver registry is unavailable");
  hierarchy_tensor_solver_registry_->add(std::move(provider), prepared_execution_lane());
}

void configure_hierarchy_tensor_solver(int program_block, int components,
                                       const std::string& provider_identity,
                                       const std::string& plan_identity,
                                       const std::string& operator_contract_identity,
                                       const std::vector<std::string>& assembly_field_slots,
                                       const std::string& solution_field_slot,
                                       const PreparedProviderOptions& options) const {
  if (preparation_view_ == nullptr)
    throw std::logic_error("AMR hierarchy tensor solver configuration is preparation-only");
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
  hierarchy_tensor_topology_epoch_ =
      preparation_view_ != nullptr ? preparation_view_->topology_epoch : runtime_->topology_epoch();
  hierarchy_tensor_materialization_generation_ = preparation_view_ != nullptr
                                                     ? preparation_view_->materialization_generation
                                                     : runtime_->materialization_generation();
}

/// Seal the provider-owned configured storage ceiling while the Program image is still detached.
///
/// The configured envelope belongs to the selected provider, not to the generic AMR checkpoint
/// arithmetic.  In particular, a provider which cannot prove a nonzero exact limit is rejected
/// before the host can publish a forward-storage ceiling.  The concrete preparation request is
/// rebuilt from the detached topology and must fit that envelope, closing the gap between the
/// capacity promise and the materialized candidate that selected the provider.
[[nodiscard]] HierarchyTensorConfiguredStorageReceipt<Dim>
configured_hierarchy_tensor_storage_receipt(std::span<const std::uint64_t> level_cell_bounds,
                                            std::span<const std::uint64_t> patch_bounds,
                                            std::span<const std::uint64_t> parent_child_pair_bounds,
                                            std::uint64_t rank_bound) const {
  const ExecutionLane& lane = prepared_execution_lane();
  HierarchyTensorConfiguredStorageReceipt<Dim> candidate;
  std::string collective_contract;
  std::exception_ptr local_error;
  try {
    if (hierarchy_tensor_selection_) {
      if (preparation_view_ == nullptr || hierarchy_tensor_solver_registry_ == nullptr)
        throw std::logic_error(
            "AMR hierarchy tensor storage receipt has no detached provider authority");
      const HierarchyTensorSelection& selection = *hierarchy_tensor_selection_;
      const int runtime_block = sys_block(selection.program_block);
      if (runtime_block < 0 ||
          static_cast<std::size_t>(runtime_block) >= preparation_view_->block_prototypes.size())
        throw std::logic_error("AMR hierarchy tensor storage receipt has an invalid block route");
      const auto provider = hierarchy_tensor_solver_registry_->resolve(selection.provider_identity);
      if (!provider || provider->identity() != selection.provider_identity)
        throw std::logic_error("AMR hierarchy tensor storage receipt resolved a foreign provider");

      HierarchyTensorConfiguredStorageRequest<Dim> configured;
      configured.level_cell_bounds.assign(level_cell_bounds.begin(), level_cell_bounds.end());
      configured.patch_bounds.assign(patch_bounds.begin(), patch_bounds.end());
      configured.parent_child_pair_bounds.assign(parent_child_pair_bounds.begin(),
                                                 parent_child_pair_bounds.end());
      configured.rank_bound = rank_bound;
      configured.components = selection.components;
      configured.provider_identity = selection.provider_identity;
      configured.provider_interface_version = provider->interface_version();
      configured.execution_lane_identity = std::string(lane.identity());
      configured.plan_identity = selection.plan_identity;
      configured.operator_contract_identity = selection.operator_contract_identity;
      configured.assembly_field_slots = selection.assembly_field_slots;
      configured.solution_field_slot = selection.solution_field_slot;
      configured.options = selection.options;
      hierarchy_tensor_detail::validate_configured_storage_request(configured);

      HierarchyTensorSolverBuildRequest<Dim> concrete;
      concrete.block = static_cast<std::size_t>(runtime_block);
      concrete.components = selection.components;
      concrete.plan_identity = selection.plan_identity;
      concrete.operator_contract_identity = selection.operator_contract_identity;
      concrete.assembly_field_slots = selection.assembly_field_slots;
      concrete.solution_field_slot = selection.solution_field_slot;
      concrete.options = selection.options;
      const auto& levels =
          preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block));
      if (levels.empty() || levels.size() != preparation_view_->level_geometries.size() ||
          preparation_view_->spatial_refinement_ratios.size() + 1U != levels.size())
        throw std::logic_error(
            "AMR hierarchy tensor storage receipt has an incomplete detached topology");
      concrete.ratios = preparation_view_->spatial_refinement_ratios;
      concrete.levels.reserve(levels.size());
      for (std::size_t level = 0; level < levels.size(); ++level) {
        const Geometry<Dim>& geometry = preparation_view_->level_geometries.at(level);
        const field_type& state = levels.at(level);
        concrete.levels.push_back({geometry, hierarchy_tensor_boundary_(geometry), state.layout(),
                                   state.distribution(), state.local_rank()});
      }
      if (!hierarchy_tensor_detail::request_fits_configured_storage(concrete, configured))
        throw std::length_error(
            "AMR hierarchy tensor concrete request exceeds its configured storage envelope");

      const HierarchyTensorConfiguredStorageLimit limit =
          provider->configured_storage_limit(configured);
      if (!limit.is_exact() || limit.maximum_bytes == 0)
        throw std::length_error(
            "AMR hierarchy tensor provider has no finite nonzero configured storage ceiling");
      candidate.active = true;
      candidate.maximum_bytes = limit.maximum_bytes;
      candidate.configured_request_contract =
          hierarchy_tensor_detail::configured_storage_request_contract(configured);
      candidate.configured_limit_contract =
          hierarchy_tensor_detail::configured_storage_limit_contract_from_request_contract<Dim>(
              candidate.configured_request_contract, limit);
    }
    if (!candidate.active && !candidate.is_canonical_inactive())
      throw std::logic_error("AMR hierarchy tensor inactive storage receipt is non-canonical");
    if (candidate.active &&
        (candidate.maximum_bytes == 0 || candidate.configured_request_contract.empty() ||
         candidate.configured_limit_contract.empty()))
      throw std::logic_error("AMR hierarchy tensor active storage receipt is incomplete");
    ExactContractBuilder receipt;
    receipt.text("pops.amr-program.hierarchy-tensor-configured-storage-receipt")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint8_t>(candidate.active ? 1U : 0U))
        .scalar(candidate.maximum_bytes)
        .bytes(candidate.configured_request_contract)
        .bytes(candidate.configured_limit_contract);
    collective_contract = std::move(receipt).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(
        "AMR hierarchy tensor configured storage receipt preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-hierarchy-tensor-configured-storage"), collective_contract}},
          lane))
    throw std::runtime_error(
        "AMR hierarchy tensor configured storage receipt differs between MPI ranks");
  return candidate;
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
  if (!hierarchy_tensor_selection_)
    return fallback;
  hierarchy_tensor_solver_type& solver = configured_hierarchy_tensor_solver_();
  if (solver.execution_path() != HierarchyTensorSolverExecutionPath::DirectProvider)
    return fallback;
  field_type& solution = solver.solution(active_level_);
  require_same_layout_(fallback, solution, "AMR Program linear solution");
  return solution;
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
  return facade_->program_macro_step_();
}
Real physical_time() const {
  return static_cast<Real>(facade_->program_time_());
}

void record_scalar(std::string_view name, Real value) const {
  runtime_state().record_diagnostic(name, value);
}
void record_balance_term(const std::string& route, const std::string& term, Real value) const {
  facade_->record_program_balance_term(route, term, static_cast<double>(value));
}
bool balance_consumer_is_due(const std::string& contract, const std::string& route,
                             int every_n) const {
  return facade_->program_balance_consumer_is_due_(contract, route, every_n);
}
void note_automatic_balance_capture_due(bool due) const {
  runtime_state().note_automatic_balance_capture_due(due, "AmrStorageTopologyAdapter");
}
void note_step_projection(const std::string& name) const {
  runtime_state().note_step_projection(name);
}
void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  facade_->program_profiler_().record(name, std::chrono::duration<double>(elapsed).count());
}

runtime_state_type& runtime_state() const {
  if (preparation_view_ != nullptr) {
    if (preparation_view_->program_state == nullptr)
      throw std::logic_error("AMR Program preparation has no detached runtime-state image");
    return *preparation_view_->program_state;
  }
  if (accepted_runtime_state_ == nullptr)
    throw std::logic_error("AMR Program execution has no accepted runtime-state authority");
  return *accepted_runtime_state_;
}

const PreparedVectorDistribution<Dim>& program_resource_vector_distribution() const {
  const int runtime_block = sys_block(0);
  const auto& distribution =
      preparation_view_ != nullptr
          ? preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block))
                .at(static_cast<std::size_t>(active_level_))
                .distribution()
          : (refresh_resources_(),
             runtime_->hierarchy().layout(static_cast<std::size_t>(active_level_)).distribution());
  vector_distribution_ = distribution.replicated() ? PreparedVectorDistribution<Dim>::replicated()
                                                   : PreparedVectorDistribution<Dim>::distributed();
  return vector_distribution_;
}
int program_resource_field_level() const noexcept {
  return active_level_;
}
std::string program_resource_materialization_identity(std::string_view owner_identity) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::string identity;
  std::exception_ptr local_error;
  try {
    if (preparation_view_ == nullptr)
      refresh_resources_();
    if (owner_identity.empty() || active_level_ < 0 || active_level_ >= nlev())
      throw std::invalid_argument(
          "AMR Program resource materialization requires an active exact owner identity");
    ExactContractBuilder contract;
    contract.text("pops.amr-program-resource-materialization")
        .scalar(std::uint32_t{1})
        .text(owner_identity)
        .text(preparation_view_ != nullptr ? std::string_view(preparation_view_->spatial_contract)
                                           : std::string_view(runtime_->spatial_contract()))
        .scalar(preparation_view_ != nullptr ? preparation_view_->topology_epoch
                                             : runtime_->topology_epoch())
        .scalar(preparation_view_ != nullptr ? preparation_view_->materialization_generation
                                             : runtime_->materialization_generation())
        .scalar(std::int32_t{active_level_})
        .text(lane.identity());
    identity = std::move(contract).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(
        "AMR Program resource materialization identity preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-resource-materialization"), std::string_view(identity)}},
          lane))
    throw std::invalid_argument(
        "AMR Program resource materialization identity differs between MPI ranks");
  return identity;
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
  const ExecutionLane& runtime_lane = prepared_execution_lane();
  const ExecutionLane& workspace_lane =
      ::pops::detail::KrylovWorkspaceAccess::execution_lane(workspace);
  const std::string_view workspace_token =
      ::pops::detail::KrylovWorkspaceAccess::materialization_token(workspace);
  std::string lane_contract;
  std::exception_ptr local_error;
  try {
    const bool workspace_active = workspace_lane.active();
    const bool workspace_named = !workspace_lane.identity().empty();
    const bool workspace_tokened = !workspace_token.empty();
    const bool runtime_active = runtime_lane.active();
    const bool runtime_named = !runtime_lane.identity().empty();
    // This is a local communicator comparison only. Every failure is converged below on the
    // runtime-owned lane, so no rank can enter a workspace-lane collective conditionally.
    const bool lanes_congruent = workspace_lane.congruent_with(runtime_lane);
    ExactContractBuilder contract;
    contract.text("pops.prepared-linear-workspace-lane")
        .scalar(std::uint32_t{2})
        .text(workspace_token)
        .text(workspace_lane.identity())
        .presence(workspace_active)
        .presence(workspace_named)
        .presence(workspace_tokened)
        .text(runtime_lane.identity())
        .presence(runtime_active)
        .presence(runtime_named)
        .presence(lanes_congruent);
    lane_contract = std::move(contract).release();
    if (!workspace_active || !workspace_named || !workspace_tokened || !runtime_active ||
        !runtime_named || !lanes_congruent)
      throw std::invalid_argument(
          "AMR prepared linear solve requires a workspace lane congruent with its "
          "runtime-authenticated lane");
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, runtime_lane) != 0) {
    if (runtime_lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR prepared linear solve lane validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.prepared-linear-workspace-lane"),
            std::string_view(lane_contract)}},
          runtime_lane))
    throw std::invalid_argument(
        "AMR prepared linear solve workspace lane contract differs across MPI ranks");
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
    const int runtime_block = sys_block(program_block);
    if (preparation_view_ != nullptr)
      return runtime_block >= 0 &&
             static_cast<std::size_t>(runtime_block) <
                 preparation_view_->runtime_block_boundary_linearizations.size() &&
             preparation_view_
                 ->runtime_block_boundary_linearizations[static_cast<std::size_t>(runtime_block)];
    return facade_ != nullptr &&
           facade_->program_has_prepared_amr_block_boundary_linearization_(runtime_block);
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
  const auto& first_evaluation = first_flux_only
                                     ? facade_->program_evaluate_prepared_amr_block_level_flux_at_(
                                           first_runtime, point, first_state)
                                     : facade_->program_evaluate_prepared_amr_block_level_at_(
                                           first_runtime, point, first_state);
  copy_valid_(first_evaluation.residual, first_candidate);
  const auto& second_evaluation = second_flux_only
                                      ? facade_->program_evaluate_prepared_amr_block_level_flux_at_(
                                            second_runtime, point, second_state)
                                      : facade_->program_evaluate_prepared_amr_block_level_at_(
                                            second_runtime, point, second_state);
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
      {active_level_, static_cast<std::int64_t>(facade_->program_macro_step_()),
       point.stage_fraction, point.physical_time},
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
    const field_type& live =
        facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
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

void bind_prepared_generated_field_route_slots(std::size_t slot_count) const {
  if (!generated_field_routes_.empty() && generated_field_routes_.size() != slot_count)
    throw std::logic_error("AMR Program field-route plan changed after preparation");
  generated_field_routes_.resize(slot_count);
}

void prepare_generated_field_route(std::uint32_t slot, std::string_view field,
                                   std::initializer_list<int> program_blocks) const {
  if (slot >= generated_field_routes_.size() || field.empty() || program_blocks.size() == 0)
    throw std::invalid_argument("AMR Program generated field route is outside the sealed plan");
  auto& route = generated_field_routes_[slot];
  if (route.prepared) {
    if (route.field != field || route.program_blocks.size() != program_blocks.size() ||
        !std::equal(route.program_blocks.begin(), route.program_blocks.end(),
                    program_blocks.begin()))
      throw std::logic_error("AMR Program generated field route changed after preparation");
    return;
  }
  route.field.assign(field.data(), field.size());
  route.program_blocks.assign(program_blocks.begin(), program_blocks.end());
  route.runtime_blocks.reserve(route.program_blocks.size());
  route.runtime_stages.assign(static_cast<std::size_t>(n_blocks()), nullptr);
  route.unique_stages.reserve(route.program_blocks.size());
  for (std::size_t index = 0; index < route.program_blocks.size(); ++index) {
    const int program_block = route.program_blocks[index];
    if (std::find(route.program_blocks.begin(), route.program_blocks.begin() + index,
                  program_block) != route.program_blocks.begin() + index)
      throw std::invalid_argument("AMR Program generated field route contains duplicate blocks");
    const int runtime_block = sys_block(program_block);
    if (std::find(route.runtime_blocks.begin(), route.runtime_blocks.end(), runtime_block) !=
        route.runtime_blocks.end())
      throw std::invalid_argument("AMR Program generated field route maps two blocks to one owner");
    route.runtime_blocks.push_back(runtime_block);
  }
  route.prepared = true;
}

[[nodiscard]] SolveOutcome solve_fields_from_blocks_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::uint32_t slot,
    std::initializer_list<FieldStageOverride> overrides) const {
  if (preparation_view_ == nullptr &&
      (resource_epoch_ != runtime_->topology_epoch() ||
       resource_generation_ != runtime_->materialization_generation()))
    throw std::logic_error("AMR Program field route is stale; refresh resources before begin_step");
  require_boundary_point_(point, "AMR Program simultaneous field solve");
  if (slot >= generated_field_routes_.size() || !generated_field_routes_[slot].prepared)
    throw std::logic_error(
        "AMR Program simultaneous field solve route was not prepared during installation");
  auto& route = generated_field_routes_[slot];
  if (overrides.size() != route.program_blocks.size())
    throw std::logic_error("AMR Program simultaneous field solve changed its prepared route");
  std::fill(route.runtime_stages.begin(), route.runtime_stages.end(), nullptr);
  route.unique_stages.clear();
  for (std::size_t index = 0; index < route.program_blocks.size(); ++index) {
    const FieldStageOverride& override_value = *(overrides.begin() + index);
    const int program_block = route.program_blocks[index];
    const int runtime_block = route.runtime_blocks[index];
    if (override_value.program_block != program_block || override_value.state == nullptr)
      throw std::invalid_argument("AMR Program simultaneous field solve differs from its route");
    if (std::find(route.unique_stages.begin(), route.unique_stages.end(), override_value.state) !=
        route.unique_stages.end())
      throw std::invalid_argument("AMR Program simultaneous field solve aliases stages");
    require_same_field_contract_(
        *override_value.state,
        facade_->program_prepared_amr_block_state_(runtime_block, active_level_),
        "AMR Program simultaneous field stage override");
    route.unique_stages.push_back(override_value.state);
    route.runtime_stages[static_cast<std::size_t>(runtime_block)] = override_value.state;
  }
  return facade_->solve_program_field_from_blocks_at(point, route.field, active_level_,
                                                     route.runtime_stages);
}

[[nodiscard]] SolveOutcome solve_default_field_on_coarse_level() const {
  if (active_level_ != 0)
    throw std::logic_error(
        "AMR Program coarse-to-fine auxiliary injection is not a fine-level solve");
  refresh_resources_();
  return facade_->solve_program_default_field(0);
}
