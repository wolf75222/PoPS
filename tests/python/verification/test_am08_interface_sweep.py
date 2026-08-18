"""AM-08 interface placement sweep (public 1-d AMR authoring plus in-memory helpers)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "interface_sweep"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RESOLUTIONS = (16, 32, 64, 128)
FIXED_N = 32


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


def test_interface_placements_cover_half_open_unit_interval():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    placements = np.asarray(exact.interface_placements(), dtype=np.float64)
    assert placements.ndim == 1
    assert placements.size >= 2
    assert np.all(placements >= 0.0)
    assert np.all(placements < 1.0)
    assert 0.0 in placements


def test_error_envelope_is_finite():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert "interface_band_mask" in text
    assert "band_max_abs_error" in text
    emin, emax = run.error_envelope(FIXED_N)
    assert np.isfinite(emin)
    assert np.isfinite(emax)
    assert emin > 0.0
    assert emax >= emin


def test_worst_placement_manufactured_order_is_two():
    run = _load_case_module("run")
    errors, spacings, worst_x0 = run.worst_error_series(RESOLUTIONS)
    assert 0.0 <= float(worst_x0) < 1.0
    assert np.all(np.isfinite(errors))
    assert np.all(np.asarray(errors, dtype=np.float64) > 0.0)
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0), rtol=1.0e-9, atol=1.0e-9)


def test_write_am08_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am08_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["amr"]["interface_error"] is not None
    assert np.isfinite(loaded["amr"]["interface_error"])
    assert loaded["amr"]["order_retained"] is True
    observed = [row["observed_order"] for row in loaded["orders"]]
    assert observed
    np.testing.assert_allclose(observed, np.full(len(observed), 2.0), rtol=1.0e-9, atol=1.0e-9)


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
        assert "from exact import" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    case = run.build_case(n_cells=8, interface=0.5)
    plan = run.resolve_plan(n_cells=8, interface=0.5)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(
            run.run_native(8, t_end=0.05, interface=0.5),
            dtype=np.float64,
        )
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert field.size > 0
    assert field.shape == (8,)
    assert np.isfinite(field).all()
