"""CP-05 multifluid eigenmode generator (2×2 toy M(k); no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "multifluid_modes"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
CANONICAL_K = 2.0 * np.pi
ADVANCE_TIME = 0.125


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_eigenvector_identity():
    exact = _load_case_module("exact")
    matrix = exact.system_matrix(CANONICAL_K)
    assert matrix.shape == (2, 2)
    assert exact.MODES == ("plus", "minus")
    wave_speed = float(exact.WAVE_SPEED)
    expected = {
        "plus": 1.0j * wave_speed * CANONICAL_K,
        "minus": -1.0j * wave_speed * CANONICAL_K,
    }
    for mode in exact.MODES:
        eigenvalue = exact.eigenvalue(mode, CANONICAL_K)
        vector = exact.right_eigenvector(mode, CANONICAL_K)
        np.testing.assert_allclose(eigenvalue, expected[mode], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            matrix @ vector,
            eigenvalue * vector,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_time_advance_matches_closed_form():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(N_CELLS)
    background = np.asarray(exact.BACKGROUND, dtype=np.float64)
    amplitude = float(exact.EPS)
    for mode in exact.MODES:
        eigenvalue = exact.eigenvalue(mode, CANONICAL_K)
        vector = exact.right_eigenvector(mode, CANONICAL_K)
        evolved = exact.exact_state(
            centers,
            ADVANCE_TIME,
            mode=mode,
            k=CANONICAL_K,
            eps=amplitude,
        )
        phase = np.exp(1.0j * CANONICAL_K * centers + eigenvalue * ADVANCE_TIME)
        expected = background[:, None] + amplitude * np.real(vector[:, None] * phase[None, :])
        np.testing.assert_allclose(evolved, expected, rtol=0.0, atol=1.0e-14)
        hat0 = amplitude * vector
        hat_t = exact.advance_fourier(hat0, ADVANCE_TIME, k=CANONICAL_K)
        np.testing.assert_allclose(
            hat_t,
            hat0 * np.exp(eigenvalue * ADVANCE_TIME),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_write_cp05_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_siblings_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


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
    plan = run.resolve_plan(16, mode="plus")
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05, mode="plus"), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
