"""RB-02 double rarefaction / near-vacuum (in-memory exact Riemann oracle)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "double_rarefaction"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width, width


def test_star_states_are_positive_and_below_both_sides():
    exact = _load_case_module("exact")
    assert exact.RHO_L == 1.0
    assert exact.U_L == -2.0
    assert exact.P_L == 0.4
    assert exact.RHO_R == 1.0
    assert exact.U_R == 2.0
    assert exact.P_R == 0.4
    assert exact.GAMMA == 1.4
    assert exact.T_END == 0.15
    star = exact.star_states()
    assert star["p_star"] > 0.0
    assert star["rho_star_left"] > 0.0
    assert star["rho_star_right"] > 0.0
    assert star["rho_star_left"] < exact.RHO_L
    assert star["rho_star_left"] < exact.RHO_R
    assert star["rho_star_right"] < exact.RHO_L
    assert star["rho_star_right"] < exact.RHO_R
    assert star["left_wave"] == "rarefaction"
    assert star["right_wave"] == "rarefaction"


def test_sampled_exact_density_and_pressure_are_positive():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(64)
    for time in (0.0, 0.075, exact.T_END):
        primitives = exact.primitives_1d(x, time)
        assert primitives.shape[0] == 3
        assert np.all(primitives[0] > 0.0)
        assert np.all(primitives[2] > 0.0)


def test_write_rb02_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_rb02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "shock/rarefaction" in loaded["not_applicable_reason"]["orders"]


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


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


def test_native_scheme_is_first_order_with_positivity_floor():
    """MUSCL at the diaphragm reconstructs p<0; FirstOrder + density floor."""
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "reconstruction.FirstOrder()" in text
    assert "positivity_floor=" in text
    assert "reconstruction.MUSCL" not in text
    assert run.POSITIVITY_FLOOR > 0.0


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    from tests.python.support.requirements import missing_compiler_requirement

    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (3, 16)
    assert field.flags["C_CONTIGUOUS"]
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
