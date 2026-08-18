"""TR-03 reversible SingleVortex (in-memory manufactured flow; no live runtime)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "single_vortex"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
PERIOD = 1.0
X0 = 0.5
Y0 = 0.75
DIV_TOL = 1.0e-2


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


def test_discrete_divergence_is_near_zero_at_cell_centres():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert not _imports_pops(text)
    u, v = exact.velocity(0.5, 0.25, 0.0)
    np.testing.assert_allclose((u, v), (1.0, 0.0), rtol=0.0, atol=1.0e-15)
    u_rev, v_rev = exact.velocity(0.5, 0.25, PERIOD)
    np.testing.assert_allclose((u_rev, v_rev), (-1.0, 0.0), rtol=0.0, atol=1.0e-15)
    for time in (0.0, 0.25, 0.5, PERIOD):
        divergence = np.asarray(
            run.manufactured_divergence(N_CELLS, t=time), dtype=np.float64
        )
        assert divergence.shape == (N_CELLS, N_CELLS)
        assert np.isfinite(divergence).all()
        assert float(np.max(np.abs(divergence))) < DIV_TOL


def test_return_field_equals_initial_condition():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
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
    peak = np.unravel_index(int(np.argmax(initial)), initial.shape)
    assert math.isclose(float(x[peak]), X0, abs_tol=1.0 / N_CELLS)
    assert math.isclose(float(y[peak]), Y0, abs_tol=1.0 / N_CELLS)


def test_write_tr03_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tr03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert (
        loaded["not_applicable_reason"]["orders"]
        == "return-to-IC / no spatial-order campaign"
    )


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
