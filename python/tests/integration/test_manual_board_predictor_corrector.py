"""Integration: final manual board predictor-corrector route."""

from pathlib import Path

import pytest

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


def _program_signature(program):
    return [
        (v.op, v.vtype, v.block, v.attrs.get("operator"), v.attrs.get("diagnostic"))
        for v in program._values
    ]


def test_manual_board_program_is_operator_first():
    module = manual.build_model()
    program = manual.build_program(module)
    signature = _program_signature(program)

    assert any(op == "call" and operator == "fields_from_state"
               for op, _, _, operator, _ in signature)
    assert any(op == "call" and operator == "explicit_rate"
               for op, _, _, operator, _ in signature)
    assert any(op == "call" and operator == "implicit_operator"
               for op, _, _, operator, _ in signature)
    assert not any(op in {"rhs", "solve_fields", "linear_source"} for op, *_ in signature)
    assert {"mass", "rho_min", "rho_max"} <= {
        diagnostic for *_, diagnostic in signature if diagnostic
    }


def test_manual_board_example_has_no_legacy_public_route_tokens():
    text = Path(manual.__file__).read_text(encoding="utf-8")
    offenders = [token for token in FORBIDDEN_PUBLIC_ROUTE_TOKENS if token in text]
    assert offenders == []


@pytest.mark.requires_toolchain
def test_manual_board_predictor_corrector_runs_one_cfl_step():
    result = manual.run_once(n=8, cfl=0.2)
    assert result["state"].shape == (3, 8, 8)
    assert {"phi", "grad_x", "grad_y"} <= set(result["fields"])
    assert {"mass", "rho_min", "rho_max"} <= set(result["diagnostics"])
    assert result["profile"] is not None
