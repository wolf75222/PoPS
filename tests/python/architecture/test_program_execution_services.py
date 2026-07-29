"""ADC-702: one source implementation for topology-independent Program operations."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROGRAM_DIR = ROOT / "include" / "pops" / "runtime" / "program"
SHARED = PROGRAM_DIR / "program_execution_services.hpp"
UNIFORM = PROGRAM_DIR / "program_context.hpp"
AMR = PROGRAM_DIR / "amr_program_context.hpp"
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
    "struct LogicalEvaluationInterval",
    "class LogicalEvaluationScope",
    "[[nodiscard]] auto logical_evaluation_scope(",
    "void evaluate_with_field_state_at(",
    "void rhs_group(",
    "void rhs_into(",
    "runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(",
    "bool has_boundary_linearization(",
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
    "int sys_block(",
    "int n_blocks(",
    "Real physical_time(",
    "void record_scalar(",
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
    "RelativeCellMeasure relative_cell_measure_(",
    "void require_unqualified_reduction_safe_(",
    "const MultiFab* active_domain_mask_(",
    "[[noreturn]] static void throw_field_solve_failure_(",
    "static bool embedded_domain_enabled_(",
    "static const MultiFab* active_mask_from_context_(",
)

SHARED_OVERLOAD_COUNTS = {
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


def test_codegen_uses_one_facade_selected_provider_factory_not_concrete_context_dispatch():
    shared = _read(SHARED)
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert shared.count("struct ProgramExecutionProviderFor;") == 1
    assert shared.count("make_program_execution_provider(") == 1
    assert shared.count("make_program_execution_view(") == 1
    assert "ProgramExecutionProviderFor<System>" in uniform
    assert "ProgramExecutionProviderFor<AmrSystem>" in amr
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
    assert "pops_install_program(pops::System* sys)" in codegen
    assert "pops_install_program_amr(pops::AmrSystem* sys)" in codegen


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


def test_contexts_expose_explicit_provider_hooks_for_the_shared_surface():
    for path, context in ((UNIFORM, "ProgramContext"), (AMR, "AmrProgramContext")):
        source = _read(path)
        assert "friend class ProgramExecutionServices<%s>;" % context in source
        for hook in (
            "program_execution_logical_parent_dt_",
            "program_execution_boundary_point_",
            "program_execution_rhs_into_",
            "program_execution_has_boundary_linearization_",
            "program_execution_rhs_core_into_at_",
            "program_execution_boundary_residual_into_at_",
            "program_execution_boundary_jvp_into_at_",
            "program_execution_neg_div_flux_default_into_",
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
            "program_execution_solve_fields_from_state_at_",
            "program_execution_scratch_",
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
            "program_execution_commit_copy_",
            "program_execution_block_map_",
            "program_execution_block_count_",
            "program_execution_physical_time_",
            "program_execution_record_scalar_",
            "program_execution_params_",
            "program_execution_set_field_timepoint_",
            "program_execution_set_field_parameters_",
            "program_execution_set_field_kernel_",
            "program_execution_profiler_",
            "program_execution_macro_step_",
            "program_execution_active_level_",
        ):
            assert source.count(hook) == 1, (
                "%s must provide exactly one explicit provider hook %s" % (context, hook)
            )


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
        "ProgramResourceTopology program_resource_topology()",
        "void with_program_resource_level(",
        "void for_each_program_resource_level(",
        "Program resource topology requires at least one level",
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
    for retired_direct_surface in (
        "program_resource_topology_epoch",
        "program_resource_topology_generation",
    ):
        assert retired_direct_surface not in shared
        assert retired_direct_surface not in uniform
        assert retired_direct_surface not in amr
        assert retired_direct_surface not in emitter
    for provider_owned_scratch in (
        "program_scratch_topology_epoch_",
        "program_scratch_materialization_generation_",
    ):
        assert provider_owned_scratch not in shared
        assert provider_owned_scratch in amr
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
    assert "program_execution_apply_projection_(sys_block(block), state)" in shared
    assert "sys_->block_project(runtime_block, state);" in uniform
    assert (
        "eng_->project_level_state(static_cast<std::size_t>(runtime_block), level_, state);" in amr
    )
    for provider in (uniform, amr):
        projection_hook = provider.split("program_execution_apply_projection_", 1)[1].split("}", 1)[
            0
        ]
        assert "sys_block(" not in projection_hook


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
    amr = _read(AMR)
    assert "target->dmap().ranks() != source->dmap().ranks()" in shared
    assert "std::vector<MultiFab> aliased_sources;" in shared
    assert "program_execution_validate_commit_aliases_(has_aliased_source)" in shared
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


def test_error_schedule_is_shared_not_an_amr_capability_deferral():
    amr = _read(AMR)
    support = _read(ROOT / "python" / "pops" / "runtime" / "amr_program_support.py")
    assert 'deferred_op("scheduler_error"' not in amr
    scheduler_group = support.split('"scheduler": {', 1)[1].split("},\n}", 1)[0]
    assert '"scheduler_error"' not in scheduler_group
