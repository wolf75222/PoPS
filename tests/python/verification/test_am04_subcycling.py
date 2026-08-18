"""AM-04 AMR subcycling (public clocks plus leftover in-memory helpers)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "subcycling"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RATIOS = (1, 2, 4)
COARSE_DT = 1.0 / 128.0


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


def test_fine_dt_is_coarse_dt_over_ratio():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert tuple(exact.RATIOS) == RATIOS
    for ratio in RATIOS:
        expected = COARSE_DT / float(ratio)
        np.testing.assert_allclose(exact.fine_dt(COARSE_DT, ratio), expected)
        np.testing.assert_allclose(run.fine_dt(COARSE_DT, ratio), expected)


def test_ratio_two_has_two_fine_steps_per_coarse():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert exact.fine_steps_per_coarse(2) == 2
    assert run.fine_steps_per_coarse(2) == 2
    assert exact.fine_steps_per_coarse(1) == 1
    assert exact.fine_steps_per_coarse(4) == 4


def test_manufactured_temporal_error_observed_order_is_two():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    dts, errors = run.manufactured_error_series(coarse_dt=COARSE_DT, ratios=RATIOS)
    np.testing.assert_allclose(dts, [COARSE_DT / float(ratio) for ratio in RATIOS])
    np.testing.assert_allclose(errors, [float(dt) ** 2 for dt in dts])
    for dt, error in zip(dts, errors, strict=True):
        np.testing.assert_allclose(exact.manufactured_temporal_error(dt), error)
    orders = observed_order(errors, dts)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))


def test_write_am04_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"]
    assert all(row["kind"] == "temporal" for row in loaded["orders"])
    observed = [row["observed_order"] for row in loaded["orders"]]
    np.testing.assert_allclose(observed, np.full(len(observed), 2.0))


def test_siblings_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


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
    for ratio in (1, 2):
        case = run.build_case(n_coarse=8, ratio=ratio)
        plan = run.resolve_plan(n_coarse=8, ratio=ratio)
        assert case is not None
        assert plan is not None
        assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(run.run_native(8, t_end=0.05, ratio=2), dtype=np.float64)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert field.size > 0
    assert field.shape == (8,)
    assert np.isfinite(field).all()
