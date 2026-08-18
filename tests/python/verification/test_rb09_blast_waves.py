"""RB-09 Woodward–Colella blast waves (in-memory two-blast IC; no closed form)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "blast_waves"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width, width


def test_three_region_initial_conditions():
    exact = _load_case_module("exact")
    x, _ = _cell_centers()
    primitives = exact.primitives_1d(x, 0.0)
    left = x < exact.X_LEFT
    right = x > exact.X_RIGHT
    middle = ~left & ~right
    assert np.any(left) and np.any(middle) and np.any(right)
    np.testing.assert_allclose(primitives[0], exact.RHO, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[1], exact.U, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[2, left], exact.P_LEFT, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[2, middle], exact.P_MIDDLE, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[2, right], exact.P_RIGHT, rtol=0.0, atol=0.0)
    assert exact.RHO == 1.0
    assert exact.U == 0.0
    assert exact.P_LEFT == 1000.0
    assert exact.P_MIDDLE == 0.01
    assert exact.P_RIGHT == 100.0
    assert exact.GAMMA == 1.4
    assert exact.T_END == 0.038
    assert exact.X_LEFT == 0.1
    assert exact.X_RIGHT == 0.9


def test_left_right_high_pressure_locations():
    exact = _load_case_module("exact")
    x, _ = _cell_centers()
    primitives = exact.primitives_1d(x, 0.0)
    left = x < exact.X_LEFT
    right = x > exact.X_RIGHT
    middle = ~left & ~right
    assert np.max(x[left]) < 0.1
    assert np.min(x[right]) > 0.9
    assert np.all((x[middle] >= 0.1) & (x[middle] <= 0.9))
    assert exact.P_LEFT > exact.P_RIGHT > exact.P_MIDDLE
    assert np.all(primitives[2, left] > primitives[2, right].max())
    assert np.all(primitives[2, right] > primitives[2, middle].max())


def test_density_and_pressure_are_positive():
    exact = _load_case_module("exact")
    x, _ = _cell_centers()
    primitives = exact.primitives_1d(x, 0.0)
    assert primitives.shape[0] == 3
    assert np.all(primitives[0] > 0.0)
    assert np.all(primitives[2] > 0.0)


def test_write_rb09_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_rb09_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "interacting blasts / no closed form" in loaded["not_applicable_reason"]["orders"]


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
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
