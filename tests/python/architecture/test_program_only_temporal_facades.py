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

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_CPP = ROOT / "src/runtime/system/system.cpp"
AMR_SYSTEM_CPP = ROOT / "src/runtime/amr/amr_system.cpp"
AMR_SYSTEM_HEADER = ROOT / "include/pops/runtime/amr_system.hpp"
AMR_RUNTIME = ROOT / "include/pops/runtime/amr/amr_runtime.hpp"
PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/program_context.hpp"
AMR_PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/amr_program_context.hpp"
AMR_DSL_BLOCK = ROOT / "include/pops/runtime/builders/compiled/amr_dsl_block.hpp"
MANIFEST = ROOT / "tests/test_manifest.toml"

EXPLICIT_TEST_BRIDGE = "tests.python.support.explicit_program"
LEGACY_DIRECT_AMR_STEP_TESTS = set()

# These remain executable semantic tests in the normal manifest.  They are blockers, not candidates
# for an explicit-Euler compatibility rewrite.
SEMANTIC_BLOCKERS = {
    "tests/python/integration/amr/test_amr_newton_full.py": (
        "engine.IMEX(",
        "newton_report(",
    ),
    "tests/python/unit/runtime/test_implicit_vars.py": (
        "implicit_vars=",
        ".advance(",
    ),
    "tests/python/unit/runtime/test_v3_features.py": (
        "engine.IMEX(",
        "newton_report(",
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


def test_system_temporal_facades_dispatch_only_through_an_installed_program():
    source = SYSTEM_CPP.read_text(encoding="utf-8")
    for signature in (
        "void System::step(double dt)",
        "void System::advance(double dt, int nsteps)",
        "double System::step_cfl(",
        "double System::step_adaptive(double cfl)",
    ):
        body = _function_body(source, signature)
        assert "require_step_installed(" in body
        assert "SystemStepper" not in body
        assert "driver_->" not in body

    assert "program_driver_.step(dt)" in _function_body(
        source, "void System::step(double dt)"
    )
    assert "program_driver_.step_cfl(" in _function_body(
        source, "double System::step_cfl("
    )
    adaptive = _function_body(source, "double System::step_adaptive(double cfl)")
    assert "has no ProgramGraph lowering" in adaptive


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

    assert "run_program_cadence_(dt)" in _function_body(
        source, "void AmrSystem::step(double dt)"
    )
    assert "run_program_cadence_(dt)" in _function_body(
        source, "double AmrSystem::step_cfl("
    )


def test_amr_program_cfl_does_not_require_native_advance_closures():
    source = AMR_RUNTIME.read_text(encoding="utf-8")
    cfl = _function_body(source, "Real cfl_dt(")
    assert "preflight_program_cfl_state_()" in cfl
    assert "preflight_native_temporal_step_()" not in cfl
    assert "preflight_native_temporal_step_" not in source
    assert "void step(Real dt)" not in source
    assert "Real step_cfl(Real cfl" not in source


def test_amr_blocks_expose_program_spatial_primitives_without_hidden_step_closures():
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    builder = AMR_DSL_BLOCK.read_text(encoding="utf-8")
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
    assert "project_level_state" in runtime
    assert "project_level_state" in builder


def test_amr_runtime_and_builders_do_not_decode_a_second_time_method():
    for path in (
        AMR_SYSTEM_HEADER,
        AMR_SYSTEM_CPP,
        AMR_RUNTIME,
        AMR_DSL_BLOCK,
        ROOT / "include/pops/runtime/builders/block/amr_block_seam.hpp",
    ):
        source = path.read_text(encoding="utf-8")
        assert "AmrTimeMethod" not in source
        assert "amr_time_method_from_wire" not in source


def test_unlowerable_semantic_tests_remain_real_manifest_tests_without_fe_bridge():
    manifest = MANIFEST.read_text(encoding="utf-8")
    for relative, markers in SEMANTIC_BLOCKERS.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'path = "%s"' % Path(relative).parent.as_posix() in manifest
        assert EXPLICIT_TEST_BRIDGE not in source
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


def test_direct_amr_runtime_step_callers_are_a_closed_migration_inventory():
    """No new C++ test may make the retiring AmrRuntime temporal engine authoritative."""
    direct_step = re.compile(
        r"\b(?:rt[0-9A-Za-z_]*|runtime|rational|integral)\s*(?:\.|->)\s*step(?:_cfl)?\("
    )
    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests/cpp").rglob("*.cpp")
        if direct_step.search(path.read_text(encoding="utf-8"))
    }
    assert discovered == LEGACY_DIRECT_AMR_STEP_TESTS
