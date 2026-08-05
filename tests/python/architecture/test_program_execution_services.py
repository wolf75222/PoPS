"""ADC-702: one source implementation for topology-independent Program operations."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROGRAM_DIR = ROOT / "include" / "pops" / "runtime" / "program"
SHARED = PROGRAM_DIR / "program_execution_services.hpp"
PROGRAM_RUNTIME_STATE = PROGRAM_DIR / "program_runtime_state.hpp"
UNIFORM = PROGRAM_DIR / "program_context.hpp"
AMR = PROGRAM_DIR / "amr_program_context.hpp"
UNIFORM_DRIVER = ROOT / "include" / "pops" / "runtime" / "system" / "system_program_driver.hpp"
AMR_RUNTIME = ROOT / "src" / "runtime" / "amr" / "amr_system.cpp"
BINDINGS = (
    ROOT / "python" / "bindings" / "core" / "init" / "init_system.cpp",
    ROOT / "python" / "bindings" / "core" / "init" / "init_amr.cpp",
)
PREPARED_AFFINE = (
    ROOT / "include" / "pops" / "numerics" / "elliptic" / "linear" / "prepared_affine_problem.hpp"
)
CODEGEN = ROOT / "python" / "pops" / "codegen"
HISTORY_CONTRACT = (
    ROOT / "tests" / "python" / "integration" / "amr" / "test_program_history_contract.py"
)
CODEGEN_CONTEXT_ROUTES = (
    CODEGEN / "program_codegen.py",
    CODEGEN / "program_emit_amr.py",
    CODEGEN / "program_emit_field_boundaries.py",
    CODEGEN / "program_emit_kernels.py",
)

SHARED_SIGNATURES = (
    "struct FieldStageOverride",
    "struct GeneratedFieldSolveWorkspace",
    "struct CouplingStateOverride",
    "struct RhsGroupRequest",
    "struct RhsGroupBatch",
    "enum class ScratchKind",
    "enum class SchedulerCacheOperation",
    "enum class HistoryReadMode",
    "enum class HistoryRotationAction",
    "struct HistoryRegistration",
    "struct HistoryStorePlan",
    "struct ProgramResourceStorage",
    "struct ProgramResourceTopology",
    "struct ProgramClockCoordinate",
    "class ExclusiveUseGuard",
    "static bool field_layout_matches_(",
    "void prepare_generated_field_solve_workspace_(",
    "void require_field_evaluation_point_(",
    "ProgramRuntimeState& program_runtime_state_()",
    "void install(std::function<void(double)> step)",
    "SolveOutcome solve_fields()",
    "SolveOutcome solve_fields_from_state_at(",
    "struct LogicalEvaluationInterval",
    "class LogicalEvaluationScope",
    "[[nodiscard]] auto logical_evaluation_scope(",
    "void evaluate_with_field_state_at(",
    "void rhs_group(",
    "void rhs_into(",
    "runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(",
    "bool has_boundary_linearization(",
    "void require_cartesian_generated_operator(",
    "void neg_div_flux_default_into(",
    "void source_default_into(",
    "void apply_projection(",
    "Real hmin(",
    "Real max_wave_speed(",
    "bool is_polar_geometry(",
    "Real radial_origin(",
    "Real radial_spacing(",
    "MultiFab rhs_scratch_like(",
    "MultiFab scratch_state_like(",
    "MultiFab& rhs_scratch(",
    "MultiFab& scratch_state(",
    "MultiFab& scalar_scratch(",
    "MultiFab& state(",
    "MultiFab& aux(",
    "GridContext grid_context(",
    "Geometry geom(",
    "std::shared_ptr<PreparedGridBoundarySession> prepare_mesh_boundary_session(",
    "std::shared_ptr<PreparedGridBoundarySession> prepare_block_boundary_session(",
    "MultiFab& assembly_target(",
    "MultiFab& assembly_source(",
    "MultiFab& linear_solution(",
    "::pops::detail::AuthenticatedProgramApplyToken authenticated_program_apply_token(",
    "OperatorEvaluationSnapshot operator_evaluation_snapshot(",
    "OperatorEvaluationSnapshot probe_operator_evaluation(",
    "SolveOutcome solve_prepared_linear(",
    "MultiFab alloc_scalar_field(",
    "ProgramResourceTopology program_resource_topology(",
    "int level(",
    "void set_level(",
    "void with_program_resource_level(",
    "void for_each_program_resource_level(",
    "const PreparedVectorDistribution& program_resource_vector_distribution(",
    "FieldDistribution program_resource_field_storage_distribution(",
    "int program_resource_field_level(",
    "void configure_program_resource_field_nullspace(",
    "const MultiFab* pointwise_active_mask(",
    "Real pointwise_status_max(",
    "Real norm2(",
    "Real norm_inf(",
    "Real dot(",
    "void commit_many(",
    "void apply_coupling_operators(",
    "void set_stage_time(",
    "void configure_primary_clock(",
    "void declare_clock_relation(",
    "bool schedule_domain_occurs(",
    "bool schedule_is_due(",
    "bool schedule_at_start(",
    "bool schedule_decision(",
    "bool cache_should_update(",
    "void cache_store_aux(",
    "void cache_restore_aux(",
    "void cache_store_scratch(",
    "void cache_restore_scratch(",
    "void cache_accumulate_dt(",
    "Real cache_effective_dt(",
    "ClockScheduleState::SubcycleScope subcycle_scope(",
    "void synchronize_sample_and_hold(",
    "void interpolate_history_linear(",
    "int sys_block(",
    "int n_blocks(",
    "Real physical_time(",
    "void record_scalar(",
    "void record_balance_term(",
    "bool balance_consumer_is_due(",
    "RuntimeParams program_params(",
    "void set_field_logical_timepoint(",
    "void set_field_boundary_parameters(",
    "void set_field_boundary_kernel(",
    "Profiler& profiler(",
    "ProfileScope profile_node(",
    "void profile_record(",
    "void count_kernel(",
    "void count_scratch(",
    "int macro_step(",
    "[[noreturn]] void scheduler_error(",
    "static void require_rate_identity_(",
    "static void require_group_identity_(",
    "static std::runtime_error block_map_error_(",
    "static SolveReport consume_field_outcome_(",
    "RelativeCellMeasure relative_cell_measure_(",
    "void require_unqualified_reduction_safe_(",
    "const MultiFab* active_domain_mask_(",
    "[[noreturn]] static void throw_field_solve_failure_(",
    "static bool embedded_domain_enabled_(",
    "static const MultiFab* active_mask_from_context_(",
)

SHARED_OVERLOAD_COUNTS = {
    "SolveOutcome solve_fields_from_state(": 1,
    "SolveOutcome solve_fields_from_blocks(": 1,
    "void neg_div_flux_into(": 4,
    "void rhs_core_into_at(": 2,
    "void boundary_residual_into_at(": 2,
    "void boundary_jvp_into_at(": 2,
    "void laplacian(": 4,
    "void tensor_laplacian(": 4,
    "void gradient(": 4,
    "void divergence(": 4,
    "void axpy(": 2,
    "void lincomb(": 2,
    "Real sum_component(": 2,
    "Real max_component(": 2,
    "Real min_component(": 2,
    "Real abs_sum_component(": 2,
    "Real sum(": 2,
    "Real max(": 2,
    "Real min(": 2,
    "Real abs_sum(": 2,
    "void fill_boundary(": 2,
    "void register_history(": 2,
    "MultiFab& history(": 2,
    "MultiFab& history_zero_start(": 2,
    "void store_history(": 2,
    "void rotate_histories(": 2,
}


def _read(path):
    return path.read_text(encoding="utf-8")


def test_uniform_and_amr_inherit_the_same_execution_service():
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert re.search(
        r"class\s+ProgramContext\s*:\s*public\s+"
        r"ProgramExecutionServices<ProgramContext>",
        uniform,
    )
    assert re.search(
        r"class\s+AmrProgramContext\s*:\s*public\s+"
        r"ProgramExecutionServices<AmrProgramContext>",
        amr,
    )


def test_uniform_and_amr_enter_one_shared_cadence_dispatcher():
    state = _read(PROGRAM_RUNTIME_STATE)
    uniform_driver = _read(UNIFORM_DRIVER)
    amr_runtime = _read(AMR_RUNTIME)

    assert state.count("void dispatch_cadence_step(") == 1
    for operation in (
        "prepare_cadence_step(",
        "validate_cadence_partition(",
        "prepare_cadence_substep(",
        "run_balance_due_window(",
        "commit_cadence_step(",
        "complete_balance_step(",
    ):
        assert operation in state
        assert operation not in uniform_driver
        assert operation not in amr_runtime

    assert (
        'P->program_.dispatch_cadence_step(P->t, P->macro_step_, dt, "System");'
        in uniform_driver
    )
    assert 'program_.dispatch_cadence_step(t, macro_step_, dt, "AmrSystem");' in amr_runtime


def test_balance_attempt_sink_is_not_python_bound():
    for binding in BINDINGS:
        assert "record_program_balance_term" not in _read(binding)


def test_codegen_uses_one_facade_selected_provider_factory_not_concrete_context_dispatch():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert shared.count("struct ProgramExecutionProviderFor;") == 1
    assert shared.count("make_program_execution_provider(") == 1
    assert shared.count("make_program_execution_view(") == 1
    assert "make_program_execution_provider(System<Dim>* system)" in uniform
    assert "make_program_execution_view(System<Dim>* system)" in uniform
    assert "class ProgramContext" in uniform
    assert "class AmrProgramContext" in amr
    assert "ProgramExecutionProviderFor<System>" not in uniform
    assert "ProgramExecutionProviderFor<AmrSystem>" not in amr
    assert "explicit ProgramContext(void*" not in uniform
    assert "explicit AmrProgramContext(void*" not in amr

    codegen = "\n".join(_read(path) for path in CODEGEN_CONTEXT_ROUTES)
    for forbidden in (
        "AmrProgramContext",
        "ProgramContext& ctx",
        "make_shared<pops::runtime::program::ProgramContext>",
    ):
        assert forbidden not in codegen
    assert codegen.count("make_program_execution_provider(sys)") >= 3
    assert codegen.count("make_program_execution_view(sys)") == 1
    assert "pops_install_program(pops::System<pops::kNativeDimension>* sys)" in codegen
    assert "pops_install_program_amr(" in codegen
    assert "pops::AmrSystem<pops::kNativeDimension>* sys" in codegen


def test_extracted_operations_have_one_source_definition():
    shared = _read(SHARED)
    context_headers = tuple(PROGRAM_DIR.glob("*program_context.hpp"))
    assert set(context_headers) == {UNIFORM, AMR}
    for signature in SHARED_SIGNATURES:
        assert shared.count(signature) == 1, (
            "%r must have one implementation in program_execution_services.hpp" % signature
        )
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in context_headers
            if signature in _read(path)
        ]
        assert not offenders, "%r was reimplemented in a topology context: %s" % (
            signature,
            ", ".join(offenders),
        )
    for signature, expected_count in SHARED_OVERLOAD_COUNTS.items():
        assert shared.count(signature) == expected_count, (
            "%r must have %d overloads in program_execution_services.hpp"
            % (signature, expected_count)
        )
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in context_headers
            if signature in _read(path)
        ]
        assert not offenders, "%r was reimplemented in a topology context: %s" % (
            signature,
            ", ".join(offenders),
        )


def test_clock_state_is_owned_only_by_the_shared_service():
    shared = _read(SHARED)
    declarations = (
        "mutable ClockScheduleState clock_schedule_;",
        "mutable std::string primary_clock_;",
        "mutable amr::Rational stage_time_{0, 1};",
    )
    for declaration in declarations:
        assert shared.count(declaration) == 1
        assert declaration not in _read(UNIFORM)
        assert declaration not in _read(AMR)


def test_operator_snapshot_revision_state_is_owned_only_by_the_shared_service():
    shared = _read(SHARED)
    declarations = (
        "mutable std::uint64_t operator_snapshot_revision_ = 0;",
        "mutable std::optional<OperatorEvaluationSnapshot> active_operator_snapshot_;",
    )
    for declaration in declarations:
        assert shared.count(declaration) == 1
        assert declaration not in _read(UNIFORM)
        assert declaration not in _read(AMR)
    assert shared.count("void invalidate_active_operator_snapshot_() const noexcept") == 1
    assert "probe != *active_operator_snapshot_" in shared
    assert "active_operator_snapshot_revision_" not in shared
    assert "invalidate_active_operator_snapshot_" not in _read(UNIFORM)
    assert "invalidate_active_operator_snapshot_" not in _read(AMR)


def test_contexts_expose_explicit_provider_hooks_for_the_shared_surface():
    for path, context in ((UNIFORM, "ProgramContext"), (AMR, "AmrProgramContext")):
        source = _read(path)
        assert "friend class ProgramExecutionServices<%s>;" % context in source
        for hook in (
            "program_execution_logical_parent_dt_",
            "program_execution_install_",
            "program_execution_boundary_point_",
            "program_execution_rhs_into_",
            "program_execution_has_boundary_linearization_",
            "program_execution_require_cartesian_generated_operator_",
            "program_execution_rhs_core_into_at_",
            "program_execution_boundary_residual_into_at_",
            "program_execution_boundary_jvp_into_at_",
            "program_execution_neg_div_flux_default_into_",
            "program_execution_neg_div_named_flux_into_",
            "program_execution_operator_topology_",
            "program_execution_operator_evaluation_snapshot_",
            "program_execution_rhs_group_",
            "program_execution_source_default_into_",
            "program_execution_apply_projection_",
            "program_execution_hmin_",
            "program_execution_max_wave_speed_",
            "program_execution_is_polar_geometry_",
            "program_execution_radial_origin_",
            "program_execution_radial_spacing_",
            "program_execution_apply_polar_tensor_",
            "program_execution_capture_logical_evaluation_",
            "program_execution_apply_logical_evaluation_",
            "program_execution_restore_logical_evaluation_",
            "program_execution_solve_fields_outcome_",
            "program_execution_solve_fields_from_state_outcome_",
            "program_execution_field_solve_from_state_at_outcome_",
            "program_execution_solve_fields_from_blocks_outcome_",
            "program_execution_solve_generated_field_from_blocks_outcome_",
            "program_execution_default_grid_context_",
            "program_execution_block_grid_context_",
            "program_execution_owns_operator_authority_",
            "program_execution_assembly_target_",
            "program_execution_assembly_source_",
            "program_execution_linear_solution_",
            "program_execution_state_",
            "program_execution_alloc_scalar_field_",
            "program_execution_apply_coupling_",
            "program_execution_register_history_storage_",
            "program_execution_read_history_storage_",
            "program_execution_history_initialized_storage_",
            "program_execution_history_slot_dt_storage_",
            "program_execution_set_history_initialized_storage_",
            "program_execution_history_store_plan_",
            "program_execution_store_history_storage_",
            "program_execution_history_supports_selective_rotation_",
            "program_execution_history_rotation_action_",
            "program_execution_defer_history_rotation_",
            "program_execution_rotate_history_storage_",
            "program_execution_cache_",
            "program_execution_resource_topology_",
            "program_execution_resource_level_",
            "program_execution_select_resource_level_",
            "program_execution_resource_storage_",
            "program_execution_resource_cell_measures_",
            "program_execution_publish_axpy_",
            "program_execution_publish_exact_axpy_",
            "program_execution_publish_lincomb_",
            "program_execution_publish_exact_lincomb_",
            "program_execution_validate_commit_aliases_",
            "program_execution_record_balance_term_",
            "program_execution_balance_consumer_is_due_",
            "program_execution_runtime_state_",
            "program_execution_clock_coordinate_",
            "program_execution_field_facade_",
        ):
            definitions = re.findall(rf"(?m)^  \S[^\n;=]*\b{re.escape(hook)}\s*\(", source)
            assert len(definitions) == 1, "%s must define exactly one explicit provider hook %s" % (
                context,
                hook,
            )


def test_boundary_point_provider_is_the_topology_primitive_not_a_mirrored_trampoline():
    for path in (UNIFORM, AMR):
        source = _read(path)
        assert "BoundaryEvaluationPoint boundary_point_(" not in source
        assert "return boundary_point_(stage_id);" not in source
        assert "BoundaryEvaluationPoint program_execution_boundary_point_(" in source


def test_field_state_evaluation_consumes_outcomes_in_the_shared_service():
    shared = _read(SHARED)
    providers = (_read(UNIFORM), _read(AMR))

    assert shared.count("consume_field_outcome_(") == 3
    assert shared.count("solve_fields_from_state_at(point, provider_slot, block,") == 2
    assert "program_execution_solve_fields_from_state_at_" not in shared
    assert all(
        "program_execution_solve_fields_from_state_at_" not in provider for provider in providers
    )


def test_generated_field_stage_workspace_is_one_shared_program_authority():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)

    for authority in (
        "struct GeneratedFieldSolveWorkspace",
        "prepare_generated_field_solve_workspace_",
        "generated_field_solve_workspaces_",
        "expected_program_blocks",
    ):
        assert authority in shared
        assert authority not in uniform
        assert authority not in amr

    for invariant in (
        "requires a non-negative IR identity",
        "requires at least one stage override",
        "IR identity was reused for a different field",
        "block map is not injective",
        "changed its ordered block pack",
        "contains a duplicate Program block",
        "generated field-solve stage does not match its exact runtime-block layout",
        "generated field-solve stage cannot alias another block's live state",
    ):
        assert shared.count(invariant) == 1
        assert invariant not in uniform
        assert invariant not in amr

    assert "ExclusiveUseGuard use(workspace.in_use," in shared
    assert "struct WorkspaceUse" not in shared
    assert "struct WorkspaceUse" not in uniform
    assert "struct WorkspaceUse" not in amr
    assert "sys_->solve_fields_from_blocks_at_in_place_(point, field, runtime_stages)" in uniform
    assert "eng_->solve_named_fields_from_states_at(point, field, runtime_stages)" in amr


def test_field_evaluation_point_validation_is_shared_before_provider_dispatch():
    shared = _read(SHARED)
    providers = (_read(UNIFORM), _read(AMR))
    validation = shared.split("void require_field_evaluation_point_(", 1)[1].split("\n public:", 1)[
        0
    ]

    assert shared.count("void require_field_evaluation_point_(") == 1
    for invariant in (
        "point.clock.empty()",
        "point.tick < 0",
        "point.substep < 0",
        "point.stage < 0",
        "!(point.dt > 0.0)",
        "!std::isfinite(point.dt)",
        "!std::isfinite(point.physical_time)",
        "point.stage_fraction < amr::Rational(0, 1)",
        "amr::Rational(1, 1) < point.stage_fraction",
    ):
        assert invariant in validation
        assert validation.index(invariant) < validation.index(
            "provider_().program_execution_resource_level_()"
        )
    assert (
        shared.count('require_field_evaluation_point_(point, "Program single-state field solve")')
        == 1
    )
    assert (
        shared.count('require_field_evaluation_point_(point, "Program simultaneous field solve")')
        == 1
    )
    assert all("require_field_evaluation_point_" not in provider for provider in providers)


def test_grid_free_program_state_services_are_shared_not_mirrored():
    shared = _read(SHARED)
    runtime_state = _read(PROGRAM_RUNTIME_STATE)
    providers = (_read(UNIFORM), _read(AMR))

    assert shared.count("program_runtime_state_().block_map()") == 3
    assert shared.count("program_runtime_state_().record_diagnostic(name, value)") == 1
    assert shared.count("program_runtime_state_().note_step_projection(name)") == 1
    assert shared.count("program_runtime_state_().params(block)") == 1
    assert shared.count("program_runtime_state_().profiler()") == 1
    assert "const std::vector<int>& block_map() const noexcept" in runtime_state
    assert "Profiler& profiler() noexcept" in runtime_state

    for retired_hook in (
        "program_execution_block_map_",
        "program_execution_record_scalar_",
        "program_execution_note_step_projection_",
        "program_execution_params_",
        "program_execution_profiler_",
    ):
        assert retired_hook not in shared
        assert all(retired_hook not in provider for provider in providers)


def test_field_configuration_uses_one_shared_facade_dispatch():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)

    for operation in (
        "set_field_logical_timepoint",
        "set_field_boundary_parameters",
        "set_field_boundary_kernel",
    ):
        assert shared.count("provider_().program_execution_field_facade_().%s" % operation) == 1
        assert operation not in uniform
        assert operation not in amr

    for retired_hook in (
        "program_execution_set_field_timepoint_",
        "program_execution_set_field_parameters_",
        "program_execution_set_field_kernel_",
    ):
        assert retired_hook not in shared
        assert retired_hook not in uniform
        assert retired_hook not in amr

    assert "System& program_execution_field_facade_() const { return *sys_; }" in uniform
    assert "AmrSystem& program_execution_field_facade_() const { return *facade_; }" in amr


def test_clock_coordinate_is_one_shared_contract_not_three_provider_queries():
    shared = _read(SHARED)
    providers = (_read(UNIFORM), _read(AMR))

    for field in (
        "Real physical_time = Real(0);",
        "int macro_step = 0;",
        "int active_level = -1;",
    ):
        assert field in shared
    assert "const ProgramClockCoordinate coordinate =" in shared
    assert "coordinate.active_level" in shared
    assert "coordinate.macro_step" in shared

    for retired_hook in (
        "program_execution_physical_time_",
        "program_execution_macro_step_",
        "program_execution_active_level_",
    ):
        assert retired_hook not in shared
        assert all(retired_hook not in provider for provider in providers)


def test_amr_grouped_rhs_registers_exact_membership_before_fragment_publication():
    amr = _read(AMR)
    hook = amr.split("void program_execution_rhs_group_", 1)[1].split(
        "void program_execution_source_default_into_", 1
    )[0]
    registration = (
        "register_interface_flux_group_(batch.group_id, batch.runtime_blocks, batch.rate_ids)"
    )
    publication = "eng_->publish_level_interface_flux_fragments("
    assert registration in hook
    assert hook.index(registration) < hook.index(publication)


def test_shared_service_uses_only_explicit_provider_hooks():
    shared = _read(SHARED)
    calls = re.findall(r"provider_\(\)\.(\w+)\(", shared)
    assert calls
    assert all(call.startswith("program_execution_") for call in calls), calls


def test_generated_cartesian_guard_is_shared_and_providers_own_only_terminal_topology():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert re.search(
        r"program_execution_require_cartesian_generated_operator_\(\s*"
        r"sys_block\(block\),\s*operation\)",
        shared,
    )
    assert "sys_->require_cartesian_generated_operator(runtime_block, operation);" in uniform
    assert "current AMR engine supports Cartesian hierarchy layouts only" in amr
    assert "void require_cartesian_generated_operator(" not in uniform
    assert "void require_cartesian_generated_operator(" not in amr


def test_named_flux_divergence_surface_is_shared_and_amr_fails_closed_at_runtime():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    emitter = _read(CODEGEN / "program_emit_ops.py")
    assert shared.count("void neg_div_flux_into(") == 4
    assert "program_execution_neg_div_named_flux_into_(" in shared
    assert "void neg_div_flux_into(" not in uniform
    assert "void neg_div_flux_into(" not in amr
    assert "fill_ghosts(flux_x, context.geom.domain, context.bc" in uniform
    assert "Program named-flux divergence scratch must match" in uniform
    assert '"a named-flux (-div F) Program on AMR is deferred' in amr
    assert "ctx.neg_div_flux_into(%s, %s, %s, %s);" in emitter


def test_prepared_operator_policy_is_shared_while_storage_stays_provider_owned():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    prepared = _read(PREPARED_AFFINE)

    assert 'validate_prepared_field_slot(field_slot_identity, "Program assembly_target")' in shared
    assert 'validate_prepared_field_slot(field_slot_identity, "Program assembly_source")' in shared
    assert "program_execution_block_grid_context_(block)" in shared
    assert "program_execution_owns_operator_authority_(authority)" in shared
    assert "solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls)" in shared

    assert "friend class ::pops::runtime::program::ProgramContext;" not in prepared
    assert "friend class ::pops::runtime::program::AmrProgramContext;" not in prepared
    assert "friend class ::pops::runtime::program::ProgramExecutionServices;" in prepared

    for provider in (uniform, amr):
        assert "validate_prepared_field_slot(" not in provider
        assert "solve_prepared_affine_outcome(" not in provider

    assert "return sys_ != nullptr && sys_->program_owns_operator_authority(authority);" in uniform
    assert "return field;" in uniform
    assert "hierarchy_tensor_assembly_field_slots_" in amr
    assert "hierarchy_tensor_solution_field_slot_" in amr
    assert "solver.assembly_target(field_slot_identity, level_)" in amr
    assert "solver.solution(level_)" in amr


def test_spatial_algorithms_are_shared_and_only_polar_stencil_is_provider_owned():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)

    for shared_authority in (
        "void apply_spatial_laplacian_(",
        "static void apply_spatial_gradient_(",
        "fill_grid_ghosts(in, context",
        "apply_divergence(fx, fy, context.geom, out",
        "Program tensor Laplacian requires all four authored coefficient fields",
    ):
        assert shared_authority in shared
        assert shared_authority not in uniform
        assert shared_authority not in amr

    assert "apply_polar_tensor(" in uniform
    assert "Cartesian Program provider cannot execute a polar tensor stencil" in uniform
    assert "AMR Program provider does not support polar tensor spatial operators" in amr
    assert "field_postprocess(" not in uniform
    assert "field_postprocess(" not in amr
    assert "apply_laplacian(" not in amr


def test_resource_topology_transaction_is_shared_while_raw_topology_and_scratch_stay_provider_owned():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    emitter = _read(ROOT / "python" / "pops" / "codegen" / "program_emit_amr.py")
    for shared_authority in (
        "struct ProgramResourceTopology",
        "int blocks = 0;",
        "ProgramResourceTopology program_resource_topology()",
        "void with_program_resource_level(",
        "void for_each_program_resource_level(",
        "Program resource topology requires at least one level",
        "Program resource topology requires at least one runtime block",
    ):
        assert shared_authority in shared
        assert shared_authority not in uniform
        assert shared_authority not in amr
    for provider_hook in (
        "program_execution_resource_topology_",
        "program_execution_resource_level_",
        "program_execution_select_resource_level_",
    ):
        assert shared.count(provider_hook) >= 1
        assert uniform.count(provider_hook) == 1
        assert amr.count(provider_hook) == 1
    assert "program_execution_block_count_" not in shared
    assert "program_execution_block_count_" not in uniform
    assert "program_execution_block_count_" not in amr
    for retired_direct_surface in (
        "program_resource_topology_epoch",
        "program_resource_topology_generation",
    ):
        assert retired_direct_surface not in shared
        assert retired_direct_surface not in uniform
        assert retired_direct_surface not in amr
        assert retired_direct_surface not in emitter
    for retired_provider_scratch in (
        "program_scratch_topology_epoch_",
        "program_scratch_materialization_generation_",
    ):
        assert retired_provider_scratch not in shared
        assert retired_provider_scratch not in uniform
        assert retired_provider_scratch not in amr
    assert "ctx.for_each_program_resource_level(" in emitter
    assert "ctx.with_program_resource_level(" in emitter
    assert "ctx.set_level(" not in emitter
    assert "const int saved_level" not in emitter
    assert "topology_materialization_generation()" in amr


def test_amr_resource_level_selection_keeps_the_active_clock_window_qualified():
    amr = _read(AMR)
    hook = re.search(
        r"void program_execution_select_resource_level_\(int selected\) const noexcept \{"
        r"(?P<body>.*?)"
        r"\n  \}",
        amr,
        re.DOTALL,
    )
    assert hook is not None
    body = hook.group("body")
    assert "level_ = selected;" in body
    assert "if (current_window_)" in body
    assert "current_window_->begin.level = selected;" in body
    assert "current_window_->end.level = selected;" in body
    assert "void set_level(" not in amr


def test_scheduler_cache_algorithm_is_shared_but_storage_lifecycle_is_provider_owned():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert shared.count("#include <pops/runtime/program/cache_manager.hpp>") == 1
    assert "#include <pops/runtime/program/cache_manager.hpp>" not in uniform
    assert "program_execution_cache_(SchedulerCacheOperation" in uniform
    assert "return sys_->program_cache();" in uniform
    assert "program_execution_cache_(SchedulerCacheOperation operation)" in amr
    assert "has no AMR checkpoint/regrid storage provider" in amr
    assert "CacheManager cache_" not in shared
    for counter in ("cache_misses", "nodes_due", "cache_hits", "nodes_skipped"):
        assert shared.count('count("%s")' % counter) == 1


def test_history_program_algorithms_are_shared_and_contexts_expose_only_storage_hooks():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    operations = {
        "register_history": 2,
        "history": 2,
        "history_zero_start": 2,
        "interpolate_history_linear": 1,
        "store_history": 2,
        "rotate_histories": 2,
    }
    for operation, count in operations.items():
        definition = re.compile(r"(?m)^\s{2}(?:void|MultiFab&)\s+%s\s*\(" % operation)
        assert len(definition.findall(shared)) == count
        assert definition.search(uniform) is None
        assert definition.search(amr) is None
    for hook in (
        "program_execution_register_history_storage_",
        "program_execution_read_history_storage_",
        "program_execution_history_initialized_storage_",
        "program_execution_history_slot_dt_storage_",
        "program_execution_set_history_initialized_storage_",
        "program_execution_history_store_plan_",
        "program_execution_store_history_storage_",
        "program_execution_history_supports_selective_rotation_",
        "program_execution_history_rotation_action_",
        "program_execution_defer_history_rotation_",
        "program_execution_rotate_history_storage_",
    ):
        assert hook in uniform
        assert hook in amr
    for amr_authority in (
        "rotate_pending_",
        "ring_clocks_",
        "ring_identities_",
        "rebind_history_flux_topology_",
    ):
        assert amr_authority not in shared
        assert amr_authority in amr


def test_real_uniform_and_amr_runtimes_execute_one_analytical_history_contract():
    source = _read(HISTORY_CONTRACT)
    assert "_ab2_factor" in source
    assert source.count("_check_history_contract(") == 3
    assert '_check_history_contract("System"' in source
    assert '_check_history_contract("AmrSystem"' in source
    assert "np.array_equal(sys_rho, amr_rho)" not in source
    assert "byte-faithful mirror" not in source


def test_shared_storage_facade_maps_blocks_once_and_owns_nullspace_assignment():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert "program_execution_state_(sys_block(block))" in shared
    assert "program_execution_state_(int runtime_block)" in uniform
    assert "program_execution_state_(int runtime_block)" in amr
    assert "basis.cell_measure = cell_measures;" in shared
    assert "basis.cell_measure =" not in uniform
    assert "basis.cell_measure =" not in amr


def test_shared_projection_maps_the_program_block_once_and_leaves_native_dispatch_to_providers():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    projection = shared.split("void apply_projection(int block, MultiFab& state) const {", 1)[
        1
    ].split("\n  }", 1)[0]
    assert projection.count("const int runtime_block = sys_block(block);") == 1
    assert "program_execution_apply_projection_(runtime_block, state)" in projection
    assert "program_execution_apply_projection_(sys_block(block), state)" not in projection
    assert (
        projection.count("program_execution_projection_balance_integrals_(runtime_block, state)")
        == 2
    )
    assert "program_execution_projection_balance_integrals_(block, state)" not in projection
    assert "sys_->block_project(runtime_block, state);" in uniform
    assert (
        "eng_->project_level_state(static_cast<std::size_t>(runtime_block), level_, state);" in amr
    )
    for provider in (uniform, amr):
        projection_hook = provider.split("program_execution_apply_projection_", 1)[1].split("}", 1)[
            0
        ]
        assert "sys_block(" not in projection_hook
        balance_hook = provider.split("program_execution_projection_balance_integrals_", 1)[
            1
        ].split("Real program_execution_hmin_", 1)[0]
        assert "sys_block(" not in balance_hook


def test_shared_cfl_dispatch_maps_the_program_block_once_and_leaves_topology_to_providers():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert "return provider_().program_execution_hmin_();" in shared
    assert "program_execution_max_wave_speed_(sys_block(block), state)" in shared
    assert "return sys_->cfl_min_dx();" in uniform
    assert "return sys_->block_max_speed(runtime_block, state);" in uniform
    assert "return eng_->level_hmin(level_);" in amr
    assert "eng_->level_max_speed(static_cast<std::size_t>(runtime_block), level_, state)" in amr
    for provider in (uniform, amr):
        wave_speed_hook = provider.split("program_execution_max_wave_speed_", 1)[1].split("}", 1)[0]
        assert "sys_block(" not in wave_speed_hook


def test_shared_geometry_queries_leave_only_terminal_metric_facts_in_providers():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    for query, hook in (
        ("bool is_polar_geometry()", "program_execution_is_polar_geometry_"),
        ("Real radial_origin()", "program_execution_radial_origin_"),
        ("Real radial_spacing()", "program_execution_radial_spacing_"),
    ):
        assert shared.count(query) == 1
        assert hook in shared
        assert query not in uniform
        assert query not in amr
    assert "return sys_->program_is_polar();" in uniform
    assert "sys_->program_polar_geometry().r_min" in uniform
    assert "sys_->program_polar_geometry().dr()" in uniform
    assert "program_execution_is_polar_geometry_() const noexcept { return false; }" in amr
    assert "program_execution_radial_origin_() const noexcept { return Real(0); }" in amr
    assert "return eng_->level_geom(level_).dx();" in amr


def test_shared_commit_many_owns_layout_and_alias_semantics():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert "target->dmap().ranks() != source->dmap().ranks()" in shared
    assert "std::vector<MultiFab> aliased_sources;" in shared
    assert "program_execution_validate_commit_aliases_(has_aliased_source)" in shared
    assert shared.count("lincomb(*target, Real(0), *target, Real(1), *source);") == 2
    assert "program_execution_commit_copy_" not in shared
    assert "program_execution_commit_copy_" not in uniform
    assert "program_execution_commit_copy_" not in amr
    assert "has_aliased_source && capturing()" in amr


def test_shared_coupling_owns_workspace_mapping_layout_alias_and_reentrancy():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    for authority in (
        "struct CouplingWorkspace",
        "prepare_coupling_workspace_",
        "coupling_workspace_",
    ):
        assert authority in shared
        assert authority not in uniform
        assert authority not in amr
    for invariant in (
        "Program coupling workspace is already in use",
        "does not cover each runtime block exactly once",
        "does not match its exact runtime layout",
        "cannot alias accepted live states",
    ):
        assert invariant in shared
    assert "ExclusiveUseGuard use(coupling_workspace_.in_use," in shared
    assert "struct WorkspaceUse" not in shared
    assert "program_execution_apply_coupling_(" in shared
    assert "sys_->apply_coupling_operators(dt, runtime_states)" in uniform
    assert "eng_->apply_coupling_operators_at_level(level_, dt, runtime_states)" in amr


def test_logical_subdivision_is_shared_and_provider_rollback_is_opaque():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert "ClockWindow" not in shared
    assert "LogicalEvaluationRollback" not in shared
    assert uniform.count("struct LogicalEvaluationRollback") == 1
    assert amr.count("struct LogicalEvaluationRollback") == 1
    assert shared.count("parent_dt / static_cast<double>(count)") == 1
    assert " / static_cast<double>(count)" not in uniform
    assert " / static_cast<double>(count)" not in amr
    assert "amr::Rational(iteration, count)" not in uniform
    assert "amr::Rational(iteration, count)" not in amr


def test_persistent_scratch_registry_is_one_shared_resource_service():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    for authority in (
        "struct ProgramScratchKey",
        "struct ProgramScratchSlot",
        "struct ProgramScratchRegistry",
        "MultiFab& persistent_scratch_",
    ):
        assert authority in shared
        assert authority not in uniform
        assert authority not in amr
    assert "program_execution_scratch_" not in shared
    assert "program_execution_scratch_" not in uniform
    assert "program_execution_scratch_" not in amr
    assert "const ProgramResourceTopology topology = program_resource_topology();" in shared
    assert "const int level = this->level();" in shared
    for invariant in (
        "non-negative IR value and sub-slot identities",
        "persistent scratch level is out of range",
    ):
        assert shared.count(invariant) == 1


def test_error_schedule_is_shared_not_an_amr_capability_deferral():
    amr = _read(AMR)
    support = _read(ROOT / "python" / "pops" / "runtime" / "amr_program_support.py")
    assert 'deferred_op("scheduler_error"' not in amr
    scheduler_group = support.split('"scheduler": {', 1)[1].split("},\n}", 1)[0]
    assert '"scheduler_error"' not in scheduler_group
