"""GE-02 solid-body scalar rotation (in-memory polar→Cartesian oracle; no live runtime)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "geometry" / "solid_rotation"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
PERIOD = 1.0
RING_RADIUS = 0.5
START_PEAK = (0.5, 0.0)
QUARTER_TURN_PEAK = (0.0, 0.5)
ORDERS_REASON = "capability-gated polar runtime"
POLAR_RUNTIME_REFUSAL = "public polar System not active"


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _imports_pops(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "pops" or alias.name.startswith("pops.") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "pops" or node.module.startswith("pops."):
                return True
    return False


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


def test_return_error_is_zero_at_period():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert not _imports_pops(text)
    assert exact.PERIOD == PERIOD
    np.testing.assert_allclose(exact.OMEGA, 2.0 * math.pi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(exact.PERIOD * exact.OMEGA, 2.0 * math.pi, rtol=0.0, atol=0.0)
    fields = run.return_fields(N_CELLS)
    initial = np.asarray(fields["initial"], dtype=np.float64)
    returned = np.asarray(fields["returned"], dtype=np.float64)
    assert initial.shape == (N_CELLS, N_CELLS)
    np.testing.assert_array_equal(returned, initial)
    errors = reference_errors(returned, initial, fields["volumes"])
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    x, y, _ = exact.cell_centers(N_CELLS)
    np.testing.assert_array_equal(
        exact.exact_return(x, y),
        exact.exact_scalar(x, y, 0.0),
    )
    np.testing.assert_array_equal(
        exact.exact_scalar(x, y, PERIOD),
        exact.exact_scalar(x, y, 0.0),
    )


def test_quarter_period_peak_has_rotated_90_degrees():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    start_x, start_y = exact.cartesian_from_polar(RING_RADIUS, 0.0)
    np.testing.assert_allclose((start_x, start_y), START_PEAK, rtol=0.0, atol=1.0e-15)
    quarter_x, quarter_y = exact.cartesian_from_polar(RING_RADIUS, 0.5 * math.pi)
    np.testing.assert_allclose(
        (quarter_x, quarter_y), QUARTER_TURN_PEAK, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        exact.peak_location(0.0), START_PEAK, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        exact.peak_location(PERIOD / 4.0), QUARTER_TURN_PEAK, rtol=0.0, atol=1.0e-15
    )
    sample_x, sample_y = 0.3, 0.4
    u, v = exact.velocity(sample_x, sample_y)
    np.testing.assert_allclose(u, -exact.OMEGA * sample_y, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(v, exact.OMEGA * sample_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        exact.exact_scalar(*START_PEAK, 0.0),
        exact.AMP,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        exact.exact_scalar(*QUARTER_TURN_PEAK, PERIOD / 4.0),
        exact.AMP,
        rtol=0.0,
        atol=1.0e-15,
    )
    x, y, width = exact.cell_centers(N_CELLS)
    field = np.asarray(exact.exact_scalar(x, y, PERIOD / 4.0), dtype=np.float64)
    peak = np.unravel_index(int(np.argmax(field)), field.shape)
    assert math.isclose(float(x[peak]), QUARTER_TURN_PEAK[0], abs_tol=width)
    assert math.isclose(float(y[peak]), QUARTER_TURN_PEAK[1], abs_tol=width)


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


def test_write_ge02_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_ge02_report(tmp_path)
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
        assert "from exact import" not in text
