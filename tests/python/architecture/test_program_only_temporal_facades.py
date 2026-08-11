"""ADC-700: fail closed until every temporal route has an explicit Program primitive.

The ordinary explicit runtime tests may install the small Programs from
``tests.python.support.explicit_program``.  Semantic implicit tests must use
typed Program primitives: forward Euler cannot stand in for backward Euler,
partial IMEX, ARS(2,2,2), or nonlinear local solves.

The blocker ledger is intentionally empty. Coupled sources, linear IMEX and nonlinear local IMEX
execute through ordinary Programs. The former dimension-erased polar runtime builder is retired;
polar algorithms remain standalone until an exact-ranked metric provider owns their topology.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_CPP = ROOT / "src/runtime/system/system.cpp"
SYSTEM_HEADER = ROOT / "include/pops/runtime/system.hpp"
SYSTEM_BINDING = ROOT / "python/bindings/core/init/init_system.cpp"
RETIRED_REFERENCE_SYSTEM_DRIVER = ROOT / "tests/cpp/support/reference_system_driver.hpp"
REFERENCE_TIME_SCHEDULER = ROOT / "tests/cpp/support/reference_time_scheduler.hpp"
LEGACY_PUBLIC_TIME_SCHEDULER = (
    ROOT / "include/pops/numerics/time/schemes/scheduler.hpp"
)
AMR_SYSTEM_CPP = ROOT / "src/runtime/amr/amr_system.cpp"
AMR_SYSTEM_HEADER = ROOT / "include/pops/runtime/amr_system.hpp"
AMR_RUNTIME = ROOT / "include/pops/runtime/amr/amr_runtime.hpp"
AMR_SUBCYCLING = ROOT / "include/pops/numerics/time/amr/levels/amr_subcycling.hpp"
PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/program_context.hpp"
AMR_PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/amr_program_context.hpp"
AMR_DSL_BLOCK = ROOT / "include/pops/runtime/builders/compiled/amr_dsl_block.hpp"
BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
POLAR_BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder_polar.hpp"
RETIRED_SYSTEM_BLOCK_SEAM = (
    ROOT / "include/pops/runtime/builders/block/block_seam.hpp"
)
SYSTEM_BLOCK_STORE = ROOT / "include/pops/runtime/system/system_block_store.hpp"
GRID_CONTEXT = ROOT / "include/pops/runtime/context/grid_context.hpp"
NUMERICAL_DEFAULTS = ROOT / "include/pops/runtime/numerical_defaults.hpp"
IMPLICIT_STEPPER = ROOT / "include/pops/numerics/time/integrators/implicit_stepper.hpp"
SYSTEM_IMPL = ROOT / "src/runtime/system/system_impl.hpp"
SYSTEM_INSTALL = ROOT / "src/runtime/system/system_install.cpp"
PYTHON_SYSTEM_INSTALL = ROOT / "python/pops/runtime/_system_install.py"
PYTHON_SYSTEM_RUNTIME = ROOT / "python/pops/runtime/_system.py"
BINDINGS_DETAIL = ROOT / "python/bindings/core/bindings_detail.hpp"
AMR_BINDING = ROOT / "python/bindings/core/init/init_amr.cpp"
LEGACY_AMR_ADVANCE_HEADER = ROOT / "include/pops/numerics/time/amr/advance/amr_advance.hpp"
HEADERS_MANIFEST = ROOT / "include/pops_headers.manifest"
PUBLIC_COUPLING_ROOT = ROOT / "include/pops/coupling"
POLAR_POISSON = ROOT / "include/pops/numerics/elliptic/polar/polar_poisson_solver.hpp"
POLAR_TENSOR = ROOT / "include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp"
ALGORITHMS_DOC = ROOT / "docs/ALGORITHMS.md"

LEGACY_DIRECT_AMR_STEP_TESTS = set()
NONLINEAR_AMR_TEST = ROOT / "tests/python/integration/amr/test_amr_newton_full.py"
V3_FEATURES_TEST = ROOT / "tests/python/unit/runtime/test_v3_features.py"


def _function_body(source: str, signature: str) -> str:
    """Extract one C++ definition with balanced braces."""
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth = 0
    for position in range(open_brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise AssertionError("unterminated C++ definition for %s" % signature)


def _python_function_source(source: str, name: str) -> str:
    """Extract one Python function without matching unrelated compatibility tests."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError("missing Python function %r" % name)


def _cpp_without_comments(source: str) -> str:
    """Remove C++ comments before matching executable temporal syntax."""
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)


def test_system_temporal_facades_dispatch_only_through_an_installed_program():
    source = SYSTEM_CPP.read_text(encoding="utf-8")
    for signature in (
        "void System<Dim>::step(double dt)",
        "void System<Dim>::advance(double dt, int nsteps)",
        "double System<Dim>::step_cfl(",
    ):
        body = _function_body(source, signature)
        assert "require_step_installed(" in body
        assert "SystemStepper" not in body
        assert "driver_->" not in body

    step = _function_body(source, "void System<Dim>::step(double dt)")
    step_cfl = _function_body(source, "double System<Dim>::step_cfl(")
    assert "dispatch_cadence_step(" in step
    assert "dispatch_cadence_step(" in step_cfl
    assert "program_driver_" not in source
    assert "step_adaptive" not in source
    assert "step_adaptive" not in SYSTEM_HEADER.read_text(encoding="utf-8")
    assert "step_adaptive" not in SYSTEM_BINDING.read_text(encoding="utf-8")


def test_python_system_temporal_facades_prepare_only_the_installed_program_run():
    source = PYTHON_SYSTEM_RUNTIME.read_text(encoding="utf-8")
    step = _python_function_source(source, "step")
    run = _python_function_source(source, "run")
    for body in (step, run):
        assert "prepare_program_run" in body
        assert "run_control_payload" not in body
        assert "run_step_attempt" not in body
    assert "FixedDt" not in step


def test_static_system_assembler_is_retired_from_the_final_runtime_surface():
    """The exact-ranked System owns assembly; no static 2D coupling facade remains."""
    production_sources = (
        tuple((ROOT / "include/pops").rglob("*.hpp"))
        + tuple((ROOT / "src").rglob("*.cpp"))
        + tuple((ROOT / "python/bindings").rglob("*.cpp"))
        + tuple((ROOT / "python/pops").rglob("*.py"))
    )
    retired_identity = re.compile(r"\b(?:SystemDriver|SystemCoupler|make_system_coupler)\b")
    violations = {
        path.relative_to(ROOT).as_posix()
        for path in production_sources
        if retired_identity.search(path.read_text(encoding="utf-8"))
    }
    assert violations == set()
    assert not (
        ROOT / "include/pops/coupling/system/system_coupler.hpp"
    ).exists()
    assert (
        "pops/coupling/system/system_coupler.hpp"
        not in HEADERS_MANIFEST.read_text(encoding="utf-8")
    )

    assert not RETIRED_REFERENCE_SYSTEM_DRIVER.exists()


def test_historical_block_scheduler_is_not_an_installed_temporal_authority():
    """The old TimePolicy scheduler remains only as a test oracle."""
    assert not LEGACY_PUBLIC_TIME_SCHEDULER.exists()
    assert (
        "pops/numerics/time/schemes/scheduler.hpp"
        not in HEADERS_MANIFEST.read_text(encoding="utf-8")
    )

    public_sources = tuple((ROOT / "include/pops").rglob("*.hpp"))
    violations = {
        path.relative_to(ROOT).as_posix()
        for path in public_sources
        if "advance_subcycled(" in _cpp_without_comments(path.read_text(encoding="utf-8"))
    }
    assert violations == set()

    reference_scheduler = REFERENCE_TIME_SCHEDULER.read_text(encoding="utf-8")
    assert "namespace pops::test_support" in reference_scheduler
    assert "void advance_subcycled(" in reference_scheduler
    assert REFERENCE_TIME_SCHEDULER.relative_to(ROOT).as_posix().startswith(
        "tests/cpp/support/"
    )


def test_public_coupling_headers_are_spatial_only():
    """Public coupling services may prepare fields/residuals, never select a time method."""
    public_headers = []
    for row in HEADERS_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not row or row.startswith("#"):
            continue
        surface, relative = row.split(maxsplit=1)
        if surface == "api" and relative.startswith("pops/coupling/"):
            public_headers.append(ROOT / "include" / relative)

    assert public_headers
    assert all(path.is_relative_to(PUBLIC_COUPLING_ROOT) for path in public_headers)

    forbidden = {
        "temporal driver call or method": re.compile(
            r"\b(?:advance(?:_[A-Za-z0-9_]+)?|step(?:_cfl|_adaptive)?)\s*\("
        ),
        "time integrator include": re.compile(
            r"#include\s+<pops/numerics/time/(?:integrators|schemes)/"
        ),
        "native stepper invocation": re.compile(r"\b(?:take_step|run_explicit_substeps)\s*\("),
        "native time-policy selector": re.compile(
            r"\b(?:TimePolicyTraits|PerStageCoupling|OncePerStepCoupling)\b"
        ),
    }
    violations = set()
    for path in public_headers:
        source = _cpp_without_comments(path.read_text(encoding="utf-8"))
        violations.update(
            (path.relative_to(ROOT).as_posix(), label)
            for label, pattern in forbidden.items()
            if pattern.search(source)
        )
    assert violations == set()

    retired_single = PUBLIC_COUPLING_ROOT / "single/coupler.hpp"
    assert not retired_single.exists()
    assert (
        "pops/coupling/single/coupler.hpp"
        not in HEADERS_MANIFEST.read_text(encoding="utf-8")
    )


def test_local_implicit_solve_has_one_typed_options_route():
    source = IMPLICIT_STEPPER.read_text(encoding="utf-8")
    assert "const NewtonOptions& options" in source
    assert "int iters = 2" not in source
    assert "Legacy signature with a bare iteration budget" not in source


def test_amr_temporal_facades_use_amr_runtime_only_as_the_spatial_engine():
    source = AMR_SYSTEM_CPP.read_text(encoding="utf-8")
    for signature in (
        "void AmrSystem<Dim>::step(double dt)",
        "void AmrSystem<Dim>::advance(double dt, int nsteps)",
        "double AmrSystem<Dim>::step_cfl(",
    ):
        body = _function_body(source, signature)
        assert "require_step_installed(" in body
        assert "AmrRuntime::step" not in body
        assert "runtime->step(" not in body
        assert "runtime->advance(" not in body

    step = _function_body(source, "void AmrSystem<Dim>::step(double dt)")
    step_cfl = _function_body(source, "double AmrSystem<Dim>::step_cfl(")
    assert "dispatch_cadence_step(" in step
    assert "step(selected)" in step_cfl


def test_amr_spatial_runtime_owns_no_cfl_or_temporal_advance_authority():
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    system = AMR_SYSTEM_CPP.read_text(encoding="utf-8")
    assert "cfl_dt(" not in runtime
    assert "void step(" not in runtime
    assert "step_cfl(" not in runtime
    step_cfl = _function_body(system, "double AmrSystem<Dim>::step_cfl(")
    assert "generated stability bound" in step_cfl
    assert "program.dt_bound_(" in step_cfl
    assert "step(selected);" in step_cfl


def test_amr_regrid_is_an_explicit_prepared_program_operation():
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    context = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    assert "void regrid_if_due(" not in runtime
    assert "regrid_interval" not in runtime
    assert "regrid_if_due" not in context
    prepare = _function_body(context, "  ::pops::amr::regridding::PreparedRegrid<Dim> prepare_regrid(")
    publish = _function_body(context, "  void publish_regrid(")
    assert "runtime_->prepare_regrid(" in prepare
    assert 'require_history_free_for_topology_change_("regrid")' in publish
    assert "runtime_->publish_regrid(" in publish


def test_amr_blocks_expose_program_spatial_primitives_without_hidden_step_closures():
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    builder = AMR_DSL_BLOCK.read_text(encoding="utf-8")
    header = AMR_SYSTEM_HEADER.read_text(encoding="utf-8")
    for legacy_closure in (
        "advance_with_temporal_plan",
        "imex_advance",
        "project_per_level",
        "PreparedAmrTemporalPlan",
    ):
        assert legacy_closure not in runtime
        assert legacy_closure not in builder
    assert "b.advance =" not in builder
    assert "b.imex =" not in builder
    assert "b.imex" not in runtime
    assert "bool imex" not in header
    assert "bimex" not in builder
    assert "project_level_state" not in runtime
    assert "project_level_state" not in builder
    assert "assemble_residual(" in builder
    assert "spatial_operator_.assemble_residual(" in builder


def test_uniform_blocks_expose_spatial_primitives_without_hidden_step_closures():
    source = BLOCK_BUILDER.read_text(encoding="utf-8")
    for legacy_closure in (
        "AdvanceExplicit",
        "AdvanceImex",
        "AdvanceImexRkArs222",
        "AdvanceExplicitMasked",
        "AdvanceExplicitEb",
        "AdvanceImexMasked",
        "AdvanceImexEb",
        "PolarAdvanceExplicit",
    ):
        assert legacy_closure not in source

    store = SYSTEM_BLOCK_STORE.read_text(encoding="utf-8")
    assert "advance_masked" not in store
    assert "advance_eb" not in store
    assert "std::function<void(MultiFab&, Real, int)> advance" not in store
    assert "rhs_into" in store
    assert not RETIRED_SYSTEM_BLOCK_SEAM.exists()
    assert "bool imex = false;" not in NUMERICAL_DEFAULTS.read_text(encoding="utf-8")
    assert "out.imex" not in SYSTEM_IMPL.read_text(encoding="utf-8")
    assert "opt.imex" not in SYSTEM_INSTALL.read_text(encoding="utf-8")
    assert 'd["imex"]' not in BINDINGS_DETAIL.read_text(encoding="utf-8")


def test_polar_runtime_builder_is_retired_until_an_exact_ranked_metric_provider_exists():
    assert not POLAR_BLOCK_BUILDER.exists()
    assert not GRID_CONTEXT.exists()


def test_standalone_polar_elliptic_algorithms_remain_explicit():
    assert POLAR_POISSON.exists()
    assert POLAR_TENSOR.exists()
    documentation = ALGORITHMS_DOC.read_text(encoding="utf-8")
    assert "no public runtime route claims" in documentation
    assert "metric-aware `Dim`-ranked provider" in documentation


def test_amr_spatial_runtime_does_not_carry_an_unexecuted_implicit_solve():
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    header = AMR_SYSTEM_HEADER.read_text(encoding="utf-8")
    binding = AMR_BINDING.read_text(encoding="utf-8")
    assert "NewtonReport" not in runtime
    assert "newton_report(" not in runtime
    assert "SourceNewtonReport" not in header
    assert "&AmrSystem::newton_report" not in binding
    assert '"newton_report"' not in binding
    assert "s.newton_report(" not in binding

    source = AMR_DSL_BLOCK.read_text(encoding="utf-8")
    assert "implicit_components" not in source
    assert "NewtonOptions" not in source
    assert "NewtonReport" not in source
    assert "resolve_implicit_components_amr" not in source
    assert "resolve_implicit_components_compiled" not in source


def test_uniform_legacy_model_route_and_unpublished_newton_diagnostics_fail_before_allocation():
    native = _function_body(
        SYSTEM_INSTALL.read_text(encoding="utf-8"),
        "void System<Dim>::add_block(",
    )
    python = PYTHON_SYSTEM_INSTALL.read_text(encoding="utf-8")
    python_add_equation = _python_function_source(python, "add_equation")

    assert "System::add_block(ModelSpec) was removed from the native core" in native
    assert "PreparedSystemBlock<Dim>" in native
    assert python_add_equation.index(
        "_reject_unpublished_newton_diagnostics(time"
    ) < python_add_equation.index("native_block_scalars(")


def test_amr_runtime_and_builders_do_not_decode_a_second_time_method():
    for path in (
        AMR_SYSTEM_HEADER,
        AMR_SYSTEM_CPP,
        AMR_RUNTIME,
        AMR_DSL_BLOCK,
    ):
        source = path.read_text(encoding="utf-8")
        assert "AmrTimeMethod" not in source
        assert "amr_time_method_from_wire" not in source


def test_production_has_no_second_amr_time_engine():
    assert not LEGACY_AMR_ADVANCE_HEADER.exists()

    production_sources = tuple((ROOT / "include/pops").rglob("*.hpp")) + tuple(
        (ROOT / "src").rglob("*.cpp")
    )
    forbidden = (
        "AmrTimeMethod",
        "amr_time_method_from_wire",
        "advance_amr(",
        "amr_step_multilevel_multipatch",
        "subcycle_level_mp",
        "PreparedAmrTemporalPlan",
        "PreparedAmrLevelAdvanceScratch",
        "PreparedAmrTransitionAdvanceScratch",
        "PreparedAmrAdvanceScratchPlan",
        "RegMP",
        "AmrImplicitSourceStepper",
        "AmrSystemDriver",
        "PoissonCadence",
        "SubcyclingSchedule",
        "mf_advance_faces(",
        "mf_apply_source_treatment(",
        "mf_fill_fine_ghosts_t(",
    )
    violations = {
        path.relative_to(ROOT).as_posix(): token
        for path in production_sources
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    assert violations == {}


def test_prepared_amr_subcycle_plan_is_the_only_spatial_reflux_route():
    source = AMR_SUBCYCLING.read_text(encoding="utf-8")
    context = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    assert "class PreparedAmrSubcyclePlan" in source
    assert "class PreparedAmrSubcycleTransition" in source
    assert "PreparedAmrProgramReflux" not in source
    assert "reconcile_reflux(" in source
    assert "runtime_->reconcile_reflux(" in context
    for retired_attempt_state in (
        "begin_attempt(",
        "publish_attempt(",
        "abort_attempt(",
        "stage_flux_x_",
        "stage_flux_y_",
        "level_scratch_",
        "std::vector<AmrLevelMP> attempt_",
    ):
        assert retired_attempt_state not in source


def test_nonlinear_amr_semantics_use_the_compiled_program_not_a_blocker():
    nonlinear = NONLINEAR_AMR_TEST.read_text(encoding="utf-8")
    v3 = V3_FEATURES_TEST.read_text(encoding="utf-8")
    assert "IMEX(" in nonlinear
    assert "LocalNewton(" in nonlinear
    assert 'getattr(node, "op", None) == "solve_outcome"' in nonlinear
    assert "program.to_graph()" in nonlinear
    assert "program._values" not in nonlinear
    assert "max_iterations=1" in nonlinear
    assert "action=fail_run" in nonlinear
    assert "covered_poison" in nonlinear
    assert "invalid_evaluation" in nonlinear
    assert "engine.IMEX(" not in nonlinear
    assert "expect_native_rejection" not in nonlinear
    assert "expect_amr_newton_rejection" not in v3
    d2_guard = _python_function_source(v3, "expect_imex_program_required")
    assert "engine.IMEX(" in d2_guard
    assert "installed whole-system Program" in d2_guard


def test_program_contexts_do_not_claim_missing_coupling_or_implicit_primitives():
    """Keep the missing native seams visible instead of silently reaching old engines."""
    for path in (PROGRAM_CONTEXT, AMR_PROGRAM_CONTEXT):
        source = path.read_text(encoding="utf-8")
        for legacy_engine_primitive in (
            "coupled_source_step(",
            "backward_euler_source(",
            "newton_report(",
        ):
            assert legacy_engine_primitive not in source


def test_ranked_program_context_owns_candidate_state_coupling_not_a_live_state_step():
    uniform = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    amr = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    retired = (
        ROOT / "include" / "pops" / "runtime" / "program" / "program_execution_services.hpp"
    )
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    assert not retired.exists()
    assert uniform.count("struct CouplingStateOverride") == 1
    assert uniform.count("void apply_coupling_operators(") == 1
    assert "ProgramContext coupling requires every runtime block candidate" in uniform
    assert "system_->apply_coupling_operators(dt, runtime_states)" in uniform
    assert "[[noreturn]] void apply_coupling_operators(" in amr
    assert 'unavailable_("exact-ranked multi-block AMR coupling provider")' in amr
    assert "void coupled_source_step(" not in runtime
    assert "void step(Real dt)" not in runtime


def test_direct_amr_runtime_step_callers_remain_absent():
    """No C++ test may recreate an AmrRuntime temporal authority."""
    direct_step = re.compile(
        r"(?:"
        r"\b(?:rt[0-9A-Za-z_]*|runtime|rational|integral)\s*(?:\.|->)"
        r"|\b[A-Za-z_]\w*\.engine\(\)\s*->\s*"
        r")step(?:_cfl)?\("
    )
    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests/cpp").rglob("*.cpp")
        if direct_step.search(path.read_text(encoding="utf-8"))
    }
    assert discovered == LEGACY_DIRECT_AMR_STEP_TESTS


def test_gpu_amr_step_harnesses_install_a_program_authority():
    for path in (ROOT / "tests/gpu/romeo").glob("*.cpp"):
        source = path.read_text(encoding="utf-8")
        if "AmrSystem" not in source or re.search(r"\b[A-Za-z_]\w*\.step\(", source) is None:
            continue
        assert any(
            installer in source
            for installer in (
                "install_forward_euler_program(",
                "install_program_step(",
                "install_program(",
            )
        ), path.relative_to(ROOT).as_posix()


def test_gpu_amr_program_harness_retains_the_exact_magnetic_provider_device_probe():
    source = (ROOT / "tests/gpu/romeo/amrmpi_integrated.cpp").read_text(encoding="utf-8")
    for marker in (
        "run_magnetic_provider_program_probe(",
        "AuxiliaryComponentKey",
        "install_prepared_auxiliary_provider(",
        "install_auxiliary_consumer_plan(",
        "stage_auxiliary_input(",
        '"B-x"',
        '"B-y"',
        '"B-z"',
        'block_level_state_global("magnetic", level)',
        "baseline.size() < 2",
        "three-component provider projection incorrect on device",
    ):
        assert marker in source
    assert "set_magnetic_field(" not in source

    for retired_harness in (
        "gpu_amr_bz_validate.cpp",
        "gpu_amr_bz_mpi_validate.cpp",
        "gpu_amrsys_facade_validate.cpp",
    ):
        assert not (ROOT / "tests/gpu/romeo" / retired_harness).exists()


def test_gpu_geometric_mg_harness_proves_only_the_exact_ranked_operator_family():
    source = (ROOT / "tests/gpu/romeo/gpu_epm_validate.cpp").read_text(encoding="utf-8")
    for required in (
        "pops::kNativeDimension",
        "GeometricMG<kDim>",
        "options.reaction = kReaction",
        "manufactured error decreases under refinement",
    ):
        assert required in source
    for retired in (
        "Box2D",
        "Array4",
        "BCRec",
        "SpatialProvider2D",
        "set_epsilon(",
        "set_epsilon_anisotropic(",
    ):
        assert retired not in source


def test_legacy_2d_elliptic_spatial_provider_headers_stay_retired():
    for retired in (
        "include/pops/numerics/elliptic/interface/spatial_provider.hpp",
        "include/pops/runtime/context/wall_predicate.hpp",
    ):
        assert not (ROOT / retired).exists()
