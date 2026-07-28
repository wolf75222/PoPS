"""ADC-700: fail closed until every temporal route has an explicit Program primitive.

The ordinary explicit runtime tests may install the small Programs from
``tests.python.support.explicit_program``.  The remaining semantic implicit and polar tests below
must not use that bridge: forward Euler cannot stand in for backward Euler, partial IMEX,
ARS(2,2,2), Newton diagnostics, or a missing point-qualified polar RHS.

This source-only gate records the current blockers deliberately.  Once typed Program primitives for
one family land, its semantic test must be migrated to that real primitive and removed from
``SEMANTIC_BLOCKERS`` in the same change. Coupled sources now execute through an explicit
candidate-state Program primitive, so their tests install a real SSPRK2 transport + split-source
Program instead of borrowing the retired facade stepper. Polar transport likewise has a genuine
point-qualified residual and its spatial tests use explicitly authored SSPRK2/SSPRK3 Programs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_CPP = ROOT / "src/runtime/system/system.cpp"
SYSTEM_HEADER = ROOT / "include/pops/runtime/system.hpp"
SYSTEM_BINDING = ROOT / "python/bindings/core/init/init_system.cpp"
AMR_SYSTEM_CPP = ROOT / "src/runtime/amr/amr_system.cpp"
AMR_SYSTEM_HEADER = ROOT / "include/pops/runtime/amr_system.hpp"
AMR_RUNTIME = ROOT / "include/pops/runtime/amr/amr_runtime.hpp"
AMR_SUBCYCLING = ROOT / "include/pops/numerics/time/amr/levels/amr_subcycling.hpp"
PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/program_context.hpp"
AMR_PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/amr_program_context.hpp"
AMR_DSL_BLOCK = ROOT / "include/pops/runtime/builders/compiled/amr_dsl_block.hpp"
AMR_BLOCK_SEAM = ROOT / "include/pops/runtime/builders/block/amr_block_seam.hpp"
BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
POLAR_BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder_polar.hpp"
SYSTEM_BLOCK_SEAM = ROOT / "include/pops/runtime/builders/block/block_seam.hpp"
SYSTEM_BLOCK_STORE = ROOT / "include/pops/runtime/system/system_block_store.hpp"
GRID_CONTEXT = ROOT / "include/pops/runtime/context/grid_context.hpp"
NUMERICAL_DEFAULTS = ROOT / "include/pops/runtime/numerical_defaults.hpp"
SYSTEM_IMPL = ROOT / "src/runtime/system/system_impl.hpp"
SYSTEM_INSTALL = ROOT / "src/runtime/system/system_install.cpp"
BINDINGS_DETAIL = ROOT / "python/bindings/core/bindings_detail.hpp"
AMR_BINDING = ROOT / "python/bindings/core/init/init_amr.cpp"
LEGACY_AMR_ADVANCE_HEADER = ROOT / "include/pops/numerics/time/amr/advance/amr_advance.hpp"
MANIFEST = ROOT / "tests/test_manifest.toml"

EXPLICIT_TEST_BRIDGE = "tests.python.support.explicit_program"
LEGACY_DIRECT_AMR_STEP_TESTS = set()
MIXED_SEMANTIC_BLOCKERS = {
    "tests/python/unit/runtime/test_v3_features.py": (
        "expect_imex_program_required",
    ),
}

# These remain executable semantic tests in the normal manifest.  They are blockers, not candidates
# for an explicit-Euler compatibility rewrite.
SEMANTIC_BLOCKERS = {
    "tests/python/integration/amr/test_amr_newton_full.py": (
        "engine.IMEX(",
        "expect_native_rejection(",
    ),
    "tests/python/unit/runtime/test_implicit_vars.py": (
        "implicit_vars=",
        ".advance(",
    ),
    "tests/python/unit/runtime/test_v3_features.py": (
        "engine.IMEX(",
        "expect_amr_newton_rejection(",
    ),
    "tests/python/unit/time/test_imexrk.py": (
        "engine.IMEXRK(",
        ".step(",
    ),
}


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
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError("missing Python function %r" % name)


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


def test_unlowerable_semantic_tests_remain_real_manifest_tests_without_fe_bridge():
    manifest = MANIFEST.read_text(encoding="utf-8")
    for relative, markers in SEMANTIC_BLOCKERS.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'path = "%s"' % Path(relative).parent.as_posix() in manifest
        blocker_functions = MIXED_SEMANTIC_BLOCKERS.get(relative)
        if blocker_functions is None:
            assert EXPLICIT_TEST_BRIDGE not in source
        else:
            # A mixed integration file may use authored FE/SSPRK Programs for genuinely explicit
            # sections. Its still-unlowerable semantic helpers must remain fail-closed.
            for function_name in blocker_functions:
                blocker_source = _python_function_source(source, function_name)
                assert "install_forward_euler_program" not in blocker_source
                assert "install_ssprk2_program" not in blocker_source
                assert "installed whole-system Program" in blocker_source
        for marker in markers:
            assert marker in source, "%s lost semantic marker %r" % (relative, marker)


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


def test_program_contexts_expose_candidate_state_coupling_not_a_live_state_step():
    uniform = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    amr = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    for source in (uniform, amr):
        assert "apply_coupling_operators(" in source
        assert "CouplingStateOverride" in source
    assert "complete candidate pack for every System block" in uniform
    assert "complete candidate pack for every runtime block" in amr
    assert "cannot alias accepted live states" in uniform
    assert "cannot alias accepted live states" in amr
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
