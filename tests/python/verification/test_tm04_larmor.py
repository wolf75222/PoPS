"""TM-04 Larmor / magnetized oscillator (in-memory ODE plus public Case resolve)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "larmor"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
DT = 0.25
SPEED_TOL = 1.0e-12


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


def _speed(u) -> float:
    vector = np.asarray(u, dtype=np.float64)
    return float(np.linalg.norm(vector))


def test_exact_speed_is_conserved():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    omega_c = float(exact.OMEGA_C)
    u0 = np.asarray(exact.U0, dtype=np.float64)
    assert omega_c == 1.0
    np.testing.assert_array_equal(u0, np.array((1.0, 0.0), dtype=np.float64))
    initial = exact.speed(u0)
    np.testing.assert_allclose(initial, 1.0)
    for time in (0.0, 0.1, 0.5, math.pi / omega_c, 2.0 * math.pi / omega_c):
        advanced = exact.exact_advance(u0, time, omega_c=omega_c)
        np.testing.assert_allclose(exact.speed(advanced), initial, atol=SPEED_TOL)


def test_exact_returns_to_u0_at_one_period():
    exact = _load_case_module("exact")
    omega_c = float(exact.OMEGA_C)
    u0 = np.asarray(exact.U0, dtype=np.float64)
    period = 2.0 * math.pi / omega_c
    returned = exact.exact_advance(u0, period, omega_c=omega_c)
    np.testing.assert_allclose(returned, u0, atol=SPEED_TOL)


def test_implicit_midpoint_conserves_speed_on_one_step():
    run = _load_case_module("run")
    exact = _load_case_module("exact")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    u0 = np.asarray(exact.U0, dtype=np.float64)
    advanced = run.implicit_midpoint(u0, DT, omega_c=exact.OMEGA_C)
    np.testing.assert_allclose(exact.speed(advanced), exact.speed(u0), atol=SPEED_TOL)


def test_explicit_euler_speed_grows():
    run = _load_case_module("run")
    exact = _load_case_module("exact")
    u0 = np.asarray(exact.U0, dtype=np.float64)
    advanced = run.explicit_euler(u0, DT, omega_c=exact.OMEGA_C)
    assert exact.speed(advanced) > exact.speed(u0)


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(DT)
    plan = run.resolve_plan(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


def test_write_tm04_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tm04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


@pytest.mark.compiler
def test_run_native_returns_finite_field_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(DT, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.size == 2 * run.N_CELLS
    assert np.isfinite(field).all()
