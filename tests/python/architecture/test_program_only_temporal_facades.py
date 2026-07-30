"""ADC-700: fail closed until every temporal route has an explicit Program primitive.

The ordinary explicit runtime tests may install the small Programs from
``tests.python.support.explicit_program``.  Semantic implicit tests must use
typed Program primitives: forward Euler cannot stand in for backward Euler,
partial IMEX, ARS(2,2,2), or nonlinear local solves.

The blocker ledger is intentionally empty.  Coupled sources, polar transport,
linear IMEX and nonlinear local IMEX now execute through ordinary Programs;
none borrows a spatial-runtime time integrator.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_CPP = ROOT / "src/runtime/system/system.cpp"
SYSTEM_HEADER = ROOT / "include/pops/runtime/system.hpp"
SYSTEM_BINDING = ROOT / "python/bindings/core/init/init_system.cpp"
STATIC_SYSTEM_ASSEMBLER = ROOT / "include/pops/coupling/system/system_coupler.hpp"
REFERENCE_SYSTEM_DRIVER = ROOT / "tests/cpp/support/reference_system_driver.hpp"
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
PROGRAM_EXECUTION_SERVICES = (
    ROOT / "include/pops/runtime/program/program_execution_services.hpp"
)
AMR_DSL_BLOCK = ROOT / "include/pops/runtime/builders/compiled/amr_dsl_block.hpp"
AMR_BLOCK_SEAM = ROOT / "include/pops/runtime/builders/block/amr_block_seam.hpp"
BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
POLAR_BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder_polar.hpp"
SYSTEM_BLOCK_SEAM = ROOT / "include/pops/runtime/builders/block/block_seam.hpp"
SYSTEM_BLOCK_STORE = ROOT / "include/pops/runtime/system/system_block_store.hpp"
GRID_CONTEXT = ROOT / "include/pops/runtime/context/grid_context.hpp"
NUMERICAL_DEFAULTS = ROOT / "include/pops/runtime/numerical_defaults.hpp"
IMPLICIT_STEPPER = ROOT / "include/pops/numerics/time/integrators/implicit_stepper.hpp"
SYSTEM_IMPL = ROOT / "src/runtime/system/system_impl.hpp"
SYSTEM_INSTALL = ROOT / "src/runtime/system/system_install.cpp"
PYTHON_SYSTEM_INSTALL = ROOT / "python/pops/runtime/_system_install.py"
BINDINGS_DETAIL = ROOT / "python/bindings/core/bindings_detail.hpp"
AMR_BINDING = ROOT / "python/bindings/core/init/init_amr.cpp"
LEGACY_AMR_ADVANCE_HEADER = ROOT / "include/pops/numerics/time/amr/advance/amr_advance.hpp"
HEADERS_MANIFEST = ROOT / "include/pops_headers.manifest"
PUBLIC_COUPLING_ROOT = ROOT / "include/pops/coupling"

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
        "void System::step(double dt)",
        "void System::advance(double dt, int nsteps)",
        "double System::step_cfl(",
    ):
        body = _function_body(source, signature)
        assert "require_step_installed(" in body
        assert "SystemStepper" not in body
        assert "driver_->" not in body

    assert "program_driver_.step(dt)" in _function_body(source, "void System::step(double dt)")
    assert "program_driver_.step_cfl(" in _function_body(source, "double System::step_cfl(")
    assert "step_adaptive" not in source
    assert "step_adaptive" not in SYSTEM_HEADER.read_text(encoding="utf-8")
    assert "step_adaptive" not in SYSTEM_BINDING.read_text(encoding="utf-8")


def test_static_system_temporal_driver_is_test_only():
    """The coupling header may assemble operators, never select a time scheme or cadence."""
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
    assert (
        "test-only pops/coupling/system/system_coupler.hpp"
        in HEADERS_MANIFEST.read_text(encoding="utf-8")
    )

    assembler = STATIC_SYSTEM_ASSEMBLER.read_text(encoding="utf-8")
    assert "class SystemAssembler" in assembler
    assert "block_residual(" in assembler
    for temporal_authority in (
        "advance_subcycled(",
        "step_adaptive(",
        "step_cfl(",
        "SSPRK2Step",
        "ImplicitSourceStepper",
    ):
        assert temporal_authority not in assembler

    reference = REFERENCE_SYSTEM_DRIVER.read_text(encoding="utf-8")
    assert "class ReferenceSystemDriver" in reference
    assert "Real step_adaptive(" in reference
    assert REFERENCE_SYSTEM_DRIVER.relative_to(ROOT).as_posix().startswith("tests/cpp/support/")


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
    reference_driver = REFERENCE_SYSTEM_DRIVER.read_text(encoding="utf-8")
    assert "namespace pops::test_support" in reference_scheduler
    assert "void advance_subcycled(" in reference_scheduler
    assert '#include "reference_time_scheduler.hpp"' in reference_driver
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

    single = (PUBLIC_COUPLING_ROOT / "single/coupler.hpp").read_text(encoding="utf-8")
    assert "void solve_fields(const MultiFab& U)" in single
    assert "void assemble_residual(MultiFab& state, MultiFab& residual)" in single


def test_local_implicit_solve_has_one_typed_options_route():
    source = IMPLICIT_STEPPER.read_text(encoding="utf-8")
    assert "const NewtonOptions& options" in source
    assert "int iters = 2" not in source
    assert "Legacy signature with a bare iteration budget" not in source


def test_amr_temporal_facades_use_amr_runtime_only_as_the_spatial_engine():
    source = AMR_SYSTEM_CPP.read_text(encoding="utf-8")
    for signature in (
        "void AmrSystem::step(double dt)",
        "void AmrSystem::advance(double dt, int nsteps)",
        "double AmrSystem::step_cfl(",
    ):
        body = _function_body(source, signature)
        assert "require_step_installed(" in body
        assert "AmrRuntime::step" not in body
        assert "runtime->step(" not in body
        assert "runtime->advance(" not in body

    assert "run_program_cadence_(dt)" in _function_body(source, "void AmrSystem::step(double dt)")
    assert "run_program_cadence_(dt)" in _function_body(source, "double AmrSystem::step_cfl(")


def test_amr_program_cfl_does_not_require_native_advance_closures():
    source = AMR_RUNTIME.read_text(encoding="utf-8")
    cfl = _function_body(source, "Real cfl_dt(")
    assert "preflight_program_cfl_state_()" in cfl
    assert "preflight_native_temporal_step_()" not in cfl
    assert "preflight_native_temporal_step_" not in source
    assert "void step(Real dt)" not in source
    assert "Real step_cfl(Real cfl" not in source
    assert "has_explicit_temporal_relations_" not in source
    cfl_preflight = _function_body(source, "void preflight_program_cfl_state_(")
    assert "temporal_relations_.size()" in cfl_preflight
    assert "hierarchy_.refinement_ratios.size()" in cfl_preflight
    assert "time-subcycling ratios" in cfl_preflight
    temporal_product = _function_body(source, "Real temporal_refinement_product_(")
    assert "temporal_relations_" in temporal_product
    assert "hierarchy_.refinement_ratios" not in temporal_product
    program_context = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    refinement_preflight = _function_body(
        program_context, "static void require_supported_program_refinement_ratios_("
    )
    assert "parent_child_temporal_relation(child)" in refinement_preflight


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
    assert "project_level_state" in runtime
    assert "project_level_state" in builder


def test_uniform_blocks_expose_spatial_primitives_without_hidden_step_closures():
    for path in (BLOCK_BUILDER, POLAR_BLOCK_BUILDER):
        source = path.read_text(encoding="utf-8")
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

    closures = GRID_CONTEXT.read_text(encoding="utf-8")
    store = SYSTEM_BLOCK_STORE.read_text(encoding="utf-8")
    seam = SYSTEM_BLOCK_SEAM.read_text(encoding="utf-8")
    for source in (closures, store):
        assert "advance_masked" not in source
        assert "advance_eb" not in source
        assert "std::function<void(MultiFab&, Real, int)> advance" not in source
    assert "bool imex;" not in seam
    assert "std::string method;" not in seam
    assert "bool imex = false;" not in NUMERICAL_DEFAULTS.read_text(encoding="utf-8")
    assert "out.imex" not in SYSTEM_IMPL.read_text(encoding="utf-8")
    assert "opt.imex" not in SYSTEM_INSTALL.read_text(encoding="utf-8")
    assert 'd["imex"]' not in BINDINGS_DETAIL.read_text(encoding="utf-8")
    assert "rhs_into" in closures
    assert "rhs_into" in store


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

    for path in (AMR_DSL_BLOCK, AMR_BLOCK_SEAM):
        source = path.read_text(encoding="utf-8")
        assert "implicit_components" not in source
        assert "NewtonOptions" not in source
        assert "NewtonReport" not in source
        assert "resolve_implicit_components_amr" not in source
        assert "resolve_implicit_components_compiled" not in source


def test_uniform_system_rejects_unpublished_newton_diagnostics_before_allocation():
    native = _function_body(
        SYSTEM_INSTALL.read_text(encoding="utf-8"),
        "void System::add_block(",
    )
    python = PYTHON_SYSTEM_INSTALL.read_text(encoding="utf-8")
    python_add_block = _python_function_source(python, "add_block")
    python_add_equation = _python_function_source(python, "add_equation")

    assert "newton_diagnostics=true is unavailable" in native
    assert "no typed implicit Program consumer publishes that report" in native
    assert "diagnostics_.newton_reports[name]" not in native
    for entrypoint in (python_add_block, python_add_equation):
        assert entrypoint.index("_reject_unpublished_newton_diagnostics(time") < entrypoint.index(
            "native_block_scalars("
        )


def test_amr_runtime_and_builders_do_not_decode_a_second_time_method():
    for path in (
        AMR_SYSTEM_HEADER,
        AMR_SYSTEM_CPP,
        AMR_RUNTIME,
        AMR_DSL_BLOCK,
        AMR_BLOCK_SEAM,
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


def test_prepared_amr_program_reflux_plan_is_spatial_only():
    source = AMR_SUBCYCLING.read_text(encoding="utf-8")
    assert "class PreparedAmrProgramRefluxPlan" in source
    assert "class PreparedAmrProgramRefluxTransition" in source
    assert "synchronize_integrated(" in source
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


def test_amr_pointwise_status_reduces_every_valid_level_cell():
    services = PROGRAM_EXECUTION_SERVICES.read_text(encoding="utf-8")

    pointwise = _function_body(
        services,
        "  const MultiFab* pointwise_active_mask(int block, const MultiFab& field) const",
    )
    assert "active_mask_from_context_(" in pointwise
    assert "program_execution_block_grid_context_(block)" in pointwise

    active_mask = _function_body(
        services,
        "  static const MultiFab* active_mask_from_context_(",
    )
    assert "if (context.domain_mask == nullptr)" in active_mask
    assert "return context.domain_mask;" in active_mask

    status = _function_body(
        services,
        "  Real pointwise_status_max(int block, const MultiFab& status,",
    )
    assert "const MultiFab* expected = pointwise_active_mask(block, status)" in status
    assert "pops::reduce_max(status, 0, RelativeCellMeasure{active_cells, nullptr})" in status


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


def test_shared_program_service_owns_candidate_state_coupling_not_a_live_state_step():
    uniform = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    amr = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    shared = (
        ROOT / "include" / "pops" / "runtime" / "program" / "program_execution_services.hpp"
    ).read_text(encoding="utf-8")
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    assert shared.count("struct CouplingStateOverride") == 1
    assert shared.count("void apply_coupling_operators(") == 1
    assert "complete candidate pack for every runtime block" in shared
    assert "cannot alias accepted live states" in shared
    for source in (uniform, amr):
        assert "void apply_coupling_operators(" not in source
        assert source.count("program_execution_apply_coupling_(") == 1
    assert "sys_->apply_coupling_operators(dt, runtime_states)" in uniform
    assert "eng_->apply_coupling_operators_at_level(level_, dt, runtime_states)" in amr
    assert "apply_coupling_operators_at_level(" in runtime
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


def test_gpu_amr_program_harness_retains_the_bz_device_probe():
    source = (ROOT / "tests/gpu/romeo/amrmpi_integrated.cpp").read_text(encoding="utf-8")
    for marker in (
        "run_bz_program_probe(",
        "set_magnetic_field(",
        'block_level_state_global("magnetic", level)',
        "baseline.size() < 2",
        "B_z not consumed on device",
    ):
        assert marker in source

    for retired_harness in (
        "gpu_amr_bz_validate.cpp",
        "gpu_amr_bz_mpi_validate.cpp",
        "gpu_amrsys_facade_validate.cpp",
    ):
        assert not (ROOT / "tests/gpu/romeo" / retired_harness).exists()
