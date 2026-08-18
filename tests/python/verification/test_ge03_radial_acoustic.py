"""GE-03 radial acoustic wave in Cartesian (in-memory oracle; no live runtime)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "geometry" / "radial_acoustic"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RING_RADIUS = 0.5
N_THETA = 64
ANGULAR_STD_ATOL = 1.0e-12


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


def test_field_depends_only_on_radius_at_t0():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    x = RING_RADIUS * np.cos(theta)
    y = RING_RADIUS * np.sin(theta)
    field = exact.phi(x, y, 0.0)
    assert field.shape == theta.shape
    np.testing.assert_allclose(field, field[0], rtol=0.0, atol=ANGULAR_STD_ATOL)
    assert float(np.std(field)) <= ANGULAR_STD_ATOL
    np.testing.assert_allclose(
        exact.radius(x, y),
        np.hypot(x, y),
        rtol=0.0,
        atol=0.0,
    )
    sample_x, sample_y = 0.3, 0.4
    radial = exact.radius(sample_x, sample_y)
    np.testing.assert_allclose(
        exact.phi(sample_x, sample_y, 0.0),
        exact.phi(radial, 0.0, 0.0),
        rtol=0.0,
        atol=ANGULAR_STD_ATOL,
    )
    np.testing.assert_allclose(exact.angular_std(RING_RADIUS, 0.0, N_THETA), 0.0, atol=ANGULAR_STD_ATOL)


def test_omega_equals_c_times_k():
    exact = _load_case_module("exact")
    assert exact.C == 1.0
    assert exact.EPS == 1.0e-3
    np.testing.assert_allclose(exact.omega(), exact.C * exact.K, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(exact.omega(k=2.5), exact.C * 2.5, rtol=0.0, atol=0.0)


def test_write_ge03_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_ge03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


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


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    case = run.build_case(8)
    plan = run.resolve_plan(8)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 2


def test_run_native_refuses_non_dim2(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM") as exc_info:
        run.run_native(8, t_end=0.01)
    message = str(exc_info.value)
    assert "fallback" in message.lower()
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.run_native(8, t_end=0.01)


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(8, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (4, 8, 8)
    assert field.flags["C_CONTIGUOUS"]
    assert np.isfinite(field).all()
    primitives = run.conserved_to_primitives(field)
    assert np.all(primitives["rho"] > 0.0)
    assert np.all(primitives["p"] > 0.0)
