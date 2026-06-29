"""Integration: final lib-time predictor-corrector route."""

from pathlib import Path

import pytest

from examples.spec_final import lib_time_predictor_corrector_poisson_lorentz as lib_time
from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as manual


FORBIDDEN_PUBLIC_ROUTE_TOKENS = (
    "tr" + "y:",
    "ex" + "cept ",
    "sk" + "ip",
    "pops." + "compile(",
    "pops." + "bind",
    "pops." + "Problem",
    "pops." + "Case",
    "add_" + "equation",
    "install_" + "program",
    "_get_" + "state",
    "_set_" + "state",
    "_eval_" + "rhs",
    "P." + "rhs",
    "P." + "solve_fields",
    "P." + "linear_source",
)


def _operator_calls(program):
    return [v.attrs.get("operator") for v in program._values if v.op == "call"]


def _diagnostics(program):
    return [v.attrs.get("diagnostic") for v in program._values if v.op == "record_scalar"]


def _commit_blocks(program):
    return sorted(program._commits)


def test_lib_time_program_is_structurally_comparable_to_manual_program():
    manual_module = manual.build_model()
    lib_module = lib_time.build_model()
    manual_program = manual.build_program(manual_module)
    lib_program = lib_time.build_program(lib_module)

    assert _operator_calls(lib_program) == _operator_calls(manual_program)
    assert _diagnostics(lib_program) == _diagnostics(manual_program)
    assert _commit_blocks(lib_program) == _commit_blocks(manual_program) == ["plasma"]
    assert not any(v.op in {"rhs", "solve_fields", "linear_source"} for v in lib_program._values)


def test_lib_time_example_has_no_legacy_public_route_tokens():
    text = Path(lib_time.__file__).read_text(encoding="utf-8")
    offenders = [token for token in FORBIDDEN_PUBLIC_ROUTE_TOKENS if token in text]
    assert offenders == []


@pytest.mark.requires_toolchain
def test_lib_time_predictor_corrector_runs_one_cfl_step():
    result = lib_time.run_once(n=8, cfl=0.2)
    assert result["state"].shape == (3, 8, 8)
    assert {"phi", "grad_x", "grad_y"} <= set(result["fields"])
    assert {"mass", "rho_min", "rho_max"} <= set(result["diagnostics"])
    assert result["profile"] is not None
