"""RB-04 Shu–Osher (in-memory FLASH/literature IC; no reference-fine run)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "shu_osher"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(exact, n_cells: int = N_CELLS):
    span = float(exact.DOMAIN_RIGHT) - float(exact.DOMAIN_LEFT)
    width = span / float(n_cells)
    centers = float(exact.DOMAIN_LEFT) + (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    return centers, width


def test_left_right_initial_conditions():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(exact)
    primitives = exact.primitives_1d(x, 0.0)
    left = x < exact.X0
    right = x > exact.X0
    assert np.any(left) and np.any(right)
    np.testing.assert_allclose(primitives[0, left], exact.RHO_L, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[1, left], exact.U_L, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[2, left], exact.P_L, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[1, right], exact.U_R, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(primitives[2, right], exact.P_R, rtol=0.0, atol=0.0)
    assert exact.RHO_L == 3.857143
    assert exact.U_L == 2.629369
    assert exact.P_L == 10.33333
    assert exact.U_R == 0.0
    assert exact.P_R == 1.0
    assert exact.GAMMA == 1.4
    assert exact.X0 == -4.0
    assert exact.DOMAIN_LEFT == -5.0
    assert exact.DOMAIN_RIGHT == 5.0


def test_right_sinusoidal_density_has_amplitude_0_2():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(exact, 128)
    primitives = exact.primitives_1d(x, 0.0)
    right = x > exact.X0
    expected = 1.0 + 0.2 * np.sin(5.0 * x[right])
    np.testing.assert_allclose(primitives[0, right], expected, rtol=0.0, atol=0.0)
    assert exact.AMPLITUDE == 0.2
    perturbation = primitives[0, right] - 1.0
    np.testing.assert_allclose(perturbation, exact.AMPLITUDE * np.sin(exact.WAVE_NUMBER * x[right]))


def test_density_and_pressure_are_positive():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(exact)
    primitives = exact.primitives_1d(x, 0.0)
    assert primitives.shape[0] == 3
    assert np.all(primitives[0] > 0.0)
    assert np.all(primitives[2] > 0.0)


def test_write_rb04_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_rb04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "reference-fine not in this increment" in loaded["not_applicable_reason"]["orders"]


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
