"""RB-03 strong shock (in-memory exact Riemann oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "strong_shock"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
STAR_OVER_RIGHT_MIN = 1000.0


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width, width


def test_star_pressure_much_greater_than_right_pressure():
    exact = _load_case_module("exact")
    assert exact.RHO_L == 1.0
    assert exact.U_L == 0.0
    assert exact.P_L == 1000.0
    assert exact.RHO_R == 1.0
    assert exact.U_R == 0.0
    assert exact.P_R == 0.01
    assert exact.GAMMA == 1.4
    assert exact.T_END == 0.012
    star = exact.star_states()
    assert star["p_star"] > STAR_OVER_RIGHT_MIN * exact.P_R
    assert star["p_star"] > exact.P_R


def test_density_and_pressure_are_positive():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(64)
    for time in (0.0, 0.006, exact.T_END):
        primitives = exact.primitives_1d(x, time)
        assert primitives.shape[0] == 3
        assert np.all(primitives[0] > 0.0)
        assert np.all(primitives[2] > 0.0)


def test_shock_is_right_going():
    exact = _load_case_module("exact")
    star = exact.star_states()
    assert star["right_wave"] == "shock"
    assert star["shock_speed"] is not None
    assert star["shock_speed"] > 0.0
    assert star["u_star"] > 0.0
    positions = exact.wave_positions(exact.T_END)
    assert positions["shock"] > exact.X0
    assert positions["contact"] > exact.X0


def test_write_rb03_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_rb03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "shock" in loaded["not_applicable_reason"]["orders"]


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def _skip_native(exc: BaseException) -> None:
    from tests.python.support.requirements import missing_compiler_requirement

    missing = missing_compiler_requirement()
    if missing:
        pytest.skip(missing)
    pytest.skip(str(exc))


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(run.run_native(16, t_end=0.012), dtype=np.float64)
    except run.NativeUnavailable as exc:
        _skip_native(exc)
    assert field.shape == (3, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
