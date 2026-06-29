"""Spec-corrective architecture gates.

These tests are intentionally narrow but hard: they guard the final public route
and the final examples against reintroducing transitional APIs or false
operator-first lowering. They do not scan the historical test suite, which still
contains explicit legacy regression coverage.
"""

import importlib
from pathlib import Path

import pytest

import pops
from examples.spec_final import lib_time_predictor_corrector_poisson_lorentz as lib_time
from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as manual


LEGACY_TOP_LEVEL = (
    "compile",
    "bind",
    "Problem",
    "Case",
    "CompiledTime",
    "Explicit",
    "IMEX",
    "Strang",
    "CondensedSchur",
    "integrate",
    "dsl",
)

LEGACY_MODULES = (
    "pops.dsl",
    "pops.integrate",
    "pops.library",
    "pops.std",
    "pops.lib.std",
)

FORBIDDEN_EXAMPLE_TOKENS = (
    "try:",
    "except ",
    "skip",
    "pops.compile(",
    "pops.bind",
    "pops.Problem",
    "pops.Case",
    "add_equation",
    "install_program",
    "_get_state",
    "_set_state",
    "_eval_rhs",
    "P.rhs",
    "P.solve_fields",
    "P.linear_source",
)

FALSE_LOWERING_OPS = {"rhs", "solve_fields", "linear_source"}


def _example_modules():
    return (manual, lib_time)


def test_legacy_imports_fail():
    for name in LEGACY_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_top_level_has_no_legacy_symbols():
    assert hasattr(pops, "compile_problem")
    for name in LEGACY_TOP_LEVEL:
        assert not hasattr(pops, name), "pops.%s is a forbidden legacy public symbol" % name


def test_no_public_runtime_fragmented_api():
    sim = pops.System(n=4)
    for name in ("add_equation", "install_program", "add_block", "set_poisson"):
        assert not hasattr(sim, name), "System.%s must not be public" % name

    program = pops.time.Program("api_gate")
    for name in ("rhs", "solve_fields", "linear_source", "source"):
        assert not hasattr(program, name), "Program.%s must not be public" % name
    assert hasattr(program, "call")


def test_examples_no_skip_or_legacy_route_tokens():
    for module in _example_modules():
        text = Path(module.__file__).read_text(encoding="utf-8")
        offenders = [tok for tok in FORBIDDEN_EXAMPLE_TOKENS if tok in text]
        assert offenders == [], "%s contains forbidden public-route token(s): %s" % (
            module.__file__, offenders)


def test_pcall_creates_call_nodes_not_false_lowering_nodes():
    for module in _example_modules():
        model = module.build_model()
        program = module.build_program(model)
        ops = [v.op for v in program._values]
        assert "call" in ops
        offenders = [v for v in program._values if v.op in FALSE_LOWERING_OPS]
        assert offenders == [], "final example lowered P.call through legacy node(s): %r" % offenders


def test_generated_cpp_calls_generated_module():
    model = manual.build_model()
    program = manual.build_program(model)
    cpp = program._emit_cpp_program_for_target(model=model)

    for op in ("fields_from_state", "explicit_rate", "implicit_operator"):
        assert "GeneratedModule::Operators::%s" % op in cpp

    body_start = cpp.index("auto generated_program_body")
    body_end = cpp.index("ctx.install", body_start)
    program_body = cpp[body_start:body_end]
    assert "ctx.rhs_into" not in program_body
    assert "ctx.solve_fields_from_state" not in program_body
    assert "ctx.source_default_into" not in program_body
    assert "ctx.neg_div_flux_default_into" not in program_body


def test_no_false_lowering_tokens_in_call_dispatch_source():
    source = Path(pops.codegen.program_emit_ops.__file__).read_text(encoding="utf-8")
    call_start = source.index('elif v.op == "call":')
    call_end = source.index('elif v.op == "solve_fields":', call_start)
    call_branch = source[call_start:call_end]

    assert "GeneratedModule::Operators" in call_branch
    for forbidden in ("ctx.rhs_into", "ctx.solve_fields", "ctx.solve_fields_from_state",
                      "ctx.source_default_into", "ctx.neg_div_flux_default_into"):
        assert forbidden not in call_branch
