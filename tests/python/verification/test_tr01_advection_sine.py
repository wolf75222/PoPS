"""TR-01 1-d periodic advection sine (Phase 1 first scientific case)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RESOLUTIONS = (16, 32, 64, 128)


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tr01_advection_sine_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = 32):
    count = int(n_cells)
    width = 1.0 / count
    return (np.arange(count, dtype=np.float64) + 0.5) * width, np.full(count, width)


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


def test_exact_sine_translation_is_periodic_identity():
    exact = _load_case_module("exact")
    x, _ = _cell_centers(32)
    np.testing.assert_allclose(
        exact.exact_sine(x, 1.0, a=1.0, k=1.0),
        exact.exact_sine(x, 0.0),
    )


def test_reference_errors_of_exact_vs_exact_are_zero():
    exact = _load_case_module("exact")
    x, volumes = _cell_centers(32)
    field = exact.exact_sine(x, 0.0)
    errors = reference_errors(field, field, volumes)
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0


def test_manufactured_second_order_series_observed_order_is_two(tmp_path: Path):
    n = np.asarray(RESOLUTIONS, dtype=np.float64)
    spacings = 1.0 / n
    errors = spacings**2
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))

    analyze = _load_case_module("analyze")
    analyze.analyze_series(errors, spacings, tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    observed = [row["observed_order"] for row in loaded["orders"]]
    assert observed
    np.testing.assert_allclose(observed, np.full(len(observed), 2.0))


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(16)
    plan = run.resolve_plan(16)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


def test_write_tr01_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_tr01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"


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
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.size > 0
    assert np.isfinite(field).all()
