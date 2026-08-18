"""AM-07 AMR vs uniform fine (in-memory series plus public Case authoring)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "vs_uniform_fine"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
H = 1.0 / 16.0
H_SERIES = (1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0)


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


def test_exact_documents_manufactured_second_order_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    assert exact.FINE_RATIO == 2
    np.testing.assert_allclose(exact.fine_spacing(H), H / 2.0)
    np.testing.assert_allclose(exact.manufactured_error(H), exact.ERROR_CONSTANT * H**2)
    np.testing.assert_allclose(
        exact.manufactured_error(exact.fine_spacing(H)),
        exact.manufactured_error(H) / 4.0,
    )


def test_three_series_are_uniform_h_amr_fine_h2_and_uniform_h2():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    series = run.three_series(H)
    assert set(series) == {
        run.SERIES_UNIFORM_H,
        run.SERIES_AMR_H_FINE_H2,
        run.SERIES_UNIFORM_H2,
    }
    np.testing.assert_allclose(run.local_spacing(run.SERIES_UNIFORM_H, H), H)
    np.testing.assert_allclose(run.local_spacing(run.SERIES_AMR_H_FINE_H2, H), H / 2.0)
    np.testing.assert_allclose(run.local_spacing(run.SERIES_UNIFORM_H2, H), H / 2.0)


def test_fine_region_amr_error_matches_uniform_h2():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    series = run.three_series(H)
    amr_fine = series[run.SERIES_AMR_H_FINE_H2]
    uniform_h2 = series[run.SERIES_UNIFORM_H2]
    uniform_h = series[run.SERIES_UNIFORM_H]
    np.testing.assert_allclose(amr_fine, uniform_h2)
    np.testing.assert_allclose(amr_fine, exact.manufactured_error(H / 2.0))
    np.testing.assert_allclose(uniform_h, exact.manufactured_error(H))
    np.testing.assert_allclose(amr_fine * 4.0, uniform_h)


def test_manufactured_series_observed_order_is_two():
    exact = _load_case_module("exact")
    spacings = np.asarray(H_SERIES, dtype=np.float64)
    errors = np.asarray([exact.manufactured_error(h) for h in spacings], dtype=np.float64)
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))


def test_write_am07_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["amr"]["order_retained"] is True
    observed = [row["observed_order"] for row in loaded["orders"]]
    assert observed
    assert all(row["kind"] == "spatial" for row in loaded["orders"])
    np.testing.assert_allclose(observed, np.full(len(observed), 2.0))


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
    assert set(run.SERIES) == {
        run.SERIES_UNIFORM_H,
        run.SERIES_AMR_H_FINE_H2,
        run.SERIES_UNIFORM_H2,
    }
    for series in run.SERIES:
        case = run.build_case(n_cells=8, series=series)
        plan = run.resolve_plan(n_cells=8, series=series)
        assert case is not None
        assert plan is not None
        assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(
            run.run_native(8, t_end=0.05, series=run.SERIES_AMR_H_FINE_H2),
            dtype=np.float64,
        )
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert field.size > 0
    assert field.shape == (8,)
    assert np.isfinite(field).all()
