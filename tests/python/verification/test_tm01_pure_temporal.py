"""TM-01 pure temporal order (fixed N=64, RK2 Δt series)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "pure_temporal"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64
DT = 1.0 / 128.0
DT_SERIES = tuple(DT / factor for factor in (1, 2, 4, 8))


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


def test_exact_loads_tr01_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    tr01 = load_sibling_module(TR01_EXACT)
    centers, _ = tr01.uniform_cell_centers(N_CELLS)
    np.testing.assert_array_equal(
        exact.exact_sine(centers, 0.25),
        tr01.exact_sine_1d(centers, 0.25),
    )


def test_exact_sine_translation_is_periodic_identity_on_fine_grid():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(N_CELLS)
    assert centers.size == N_CELLS
    np.testing.assert_allclose(
        exact.exact_sine(centers, 1.0, a=1.0, k=1.0),
        exact.exact_sine(centers, 0.0),
    )


def test_manufactured_rk2_dt_series_observed_order_is_two(tmp_path: Path):
    dts = np.asarray(DT_SERIES, dtype=np.float64)
    errors = dts**2
    orders = observed_order(errors, dts)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))

    analyze = _load_case_module("analyze")
    analyze.analyze_series(errors, dts, tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    kinds = [row["kind"] for row in loaded["orders"]]
    observed = [row["observed_order"] for row in loaded["orders"]]
    assert kinds
    assert set(kinds) == {"temporal"}
    np.testing.assert_allclose(observed, np.full(len(observed), 2.0))


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(DT)
    plan = run.resolve_plan(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


def test_write_tm01_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_tm01_report(tmp_path)
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


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert owners
            assert set(owners) == {"run_native"}
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
    assert field.size == N_CELLS
    assert np.isfinite(field).all()
