"""TM-06 multirate species substeps (in-memory BE plus public Case)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "multirate"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
DT = 0.25


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


def test_exact_solutions_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert "import pops" not in text
    assert "from pops" not in text
    exact = _load_case_module("exact")
    assert exact.LAMBDA_F == 8.0
    assert exact.LAMBDA_S == 1.0
    assert tuple(exact.RATIOS) == (1, 2, 4, 8)
    y0 = float(exact.Y0)
    z0 = float(exact.Z0)
    np.testing.assert_allclose(exact.exact_y(0.0, y0), y0)
    np.testing.assert_allclose(exact.exact_z(0.0, z0), z0)
    times = (0.0, 0.125, 0.25, 0.5, 1.0)
    for time in times:
        np.testing.assert_allclose(
            exact.exact_y(time, y0),
            y0 * math.exp(-float(exact.LAMBDA_F) * time),
        )
        np.testing.assert_allclose(
            exact.exact_z(time, z0),
            z0 * math.exp(-float(exact.LAMBDA_S) * time),
        )
        state = exact.exact_state(time, y0, z0)
        np.testing.assert_allclose(state[0], exact.exact_y(time, y0))
        np.testing.assert_allclose(state[1], exact.exact_z(time, z0))


def test_r1_matches_single_rate_backward_euler():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in run_text
    assert "from exact import" not in run_text
    y0 = float(exact.Y0)
    z0 = float(exact.Z0)
    dt = float(exact.DT)
    single = run.single_rate_step(y0, z0, dt)
    multi = run.multirate_step(y0, z0, dt, 1)
    expected_y = y0 / (1.0 + float(exact.LAMBDA_F) * dt)
    expected_z = z0 / (1.0 + float(exact.LAMBDA_S) * dt)
    np.testing.assert_allclose(single[0], expected_y)
    np.testing.assert_allclose(single[1], expected_z)
    np.testing.assert_allclose(multi[0], expected_y)
    np.testing.assert_allclose(multi[1], expected_z)
    np.testing.assert_allclose(multi, single)


def test_larger_r_reduces_fast_component_error():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    y0 = float(exact.Y0)
    z0 = float(exact.Z0)
    dt = float(exact.DT)
    y_exact = exact.exact_y(dt, y0)
    z_single = run.single_rate_step(y0, z0, dt)[1]
    errors = []
    for ratio in exact.RATIOS:
        y_num, z_num = run.multirate_step(y0, z0, dt, ratio)
        errors.append(abs(float(y_num) - float(y_exact)))
        np.testing.assert_allclose(z_num, z_single)
    assert tuple(exact.RATIOS)[0] == 1
    err_r1 = errors[0]
    for err in errors[1:]:
        assert err < err_r1
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))


def test_write_tm06_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tm06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(DT)
    plan = run.resolve_plan(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


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
        field = np.asarray(run.run_native(DT, t_end=DT), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.size == 2 * run.N_CELLS
    assert np.isfinite(field).all()
