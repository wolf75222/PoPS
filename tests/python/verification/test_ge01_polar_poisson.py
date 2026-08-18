"""GE-01 polar manufactured Poisson (in-memory harmonic; capability-gated polar runtime)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "geometry" / "polar_poisson"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
R_MIN = 0.2
R_MAX = 1.0
MODE = 2
ORDERS_REASON = "capability-gated polar runtime"
POLAR_RUNTIME_REFUSAL = "public polar System not active"
SAMPLE_POINTS = (
    (0.2, 0.0),
    (0.5, 0.25 * np.pi),
    (0.7, 0.5 * np.pi),
    (1.0, np.pi),
    (0.3, 1.5 * np.pi),
)
LAPLACIAN_ATOL = 1.0e-12


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_call_owners(source: str) -> list[str]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    owners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pops"
        ):
            continue
        current: ast.AST | None = node
        owner = "<module>"
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = current.name
                break
            current = parents.get(current)
        owners.append(owner)
    return owners


def test_analytic_laplacian_is_zero_at_sample_points():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert exact.M == MODE
    radii = np.array([point[0] for point in SAMPLE_POINTS], dtype=np.float64)
    thetas = np.array([point[1] for point in SAMPLE_POINTS], dtype=np.float64)
    field = exact.phi(radii, thetas)
    np.testing.assert_allclose(
        field,
        radii**MODE * np.cos(MODE * thetas),
        rtol=0.0,
        atol=1.0e-14,
    )
    laplacian = exact.polar_laplacian(radii, thetas)
    np.testing.assert_allclose(laplacian, 0.0, rtol=0.0, atol=LAPLACIAN_ATOL)
    for radius, theta in SAMPLE_POINTS:
        np.testing.assert_allclose(
            exact.polar_laplacian(radius, theta),
            0.0,
            rtol=0.0,
            atol=LAPLACIAN_ATOL,
        )
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        np.testing.assert_allclose(
            exact.cartesian_equivalent(x, y),
            exact.phi(radius, theta),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_annulus_excludes_origin():
    exact = _load_case_module("exact")
    assert exact.R_MIN == R_MIN
    assert exact.R_MAX == R_MAX
    assert exact.R_MIN > 0.0
    assert not exact.in_annulus(0.0)
    assert exact.in_annulus(R_MIN)
    assert exact.in_annulus(R_MAX)
    assert not exact.in_annulus(R_MIN - 1.0e-12)
    assert not exact.in_annulus(R_MAX + 1.0e-12)
    for radius, _theta in SAMPLE_POINTS:
        assert radius >= exact.R_MIN
        assert exact.in_annulus(radius)


def test_refuse_public_polar_runtime_is_non_empty():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    reason = run.refuse_public_polar_runtime()
    assert isinstance(reason, str)
    assert reason == POLAR_RUNTIME_REFUSAL
    with pytest.raises(run.NativeUnavailable, match=POLAR_RUNTIME_REFUSAL):
        run.run_native()


def test_write_ge01_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_ge01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
