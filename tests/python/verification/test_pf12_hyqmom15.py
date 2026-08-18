"""PF-12 HyQMOM15 large-moment state stand-in (15-wide saxpy; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "hyqmom15"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 16
N_COMPONENTS = 15
EULER_COMPONENTS = 5
BYTES_PER_SCALAR = 8
BYTES_PER_CELL = 120
EULER_BYTES_PER_CELL = 40
ORDERS_REASON = "kernel microbench stand-in, not a timed PF run"
HYQMOM15_CASE_REFUSAL = "public 15-component HyQMOM15 Case is not available"


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


def test_fifteen_components():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert exact.N_COMPONENTS == N_COMPONENTS
    assert len(exact.COMPONENT_NAMES) == N_COMPONENTS
    state = np.asarray(run.wide_state()["state"], dtype=np.float64)
    assert state.shape == (N_CELLS, N_COMPONENTS)
    assert state.shape[1] == exact.N_COMPONENTS


def test_bytes_per_cell():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert exact.BYTES_PER_SCALAR == BYTES_PER_SCALAR
    assert exact.BYTES_PER_CELL == BYTES_PER_CELL
    assert exact.bytes_per_cell() == BYTES_PER_CELL
    assert exact.bytes_per_cell() == N_COMPONENTS * BYTES_PER_SCALAR
    assert run.bytes_per_cell() == BYTES_PER_CELL
    comparison = run.width_comparison()
    assert comparison["hyqmom15_components"] == N_COMPONENTS
    assert comparison["hyqmom15_bytes_per_cell"] == BYTES_PER_CELL
    assert comparison["euler_components"] == EULER_COMPONENTS
    assert comparison["euler_bytes_per_cell"] == EULER_BYTES_PER_CELL
    assert comparison["ratio"] == 3.0


def test_saxpy_exact():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    result = run.saxpy_fields()
    a = np.asarray(result["a"], dtype=np.float64)
    b = np.asarray(result["b"], dtype=np.float64)
    assert a.shape == (N_CELLS, N_COMPONENTS)
    assert b.shape == (N_CELLS, N_COMPONENTS)
    np.testing.assert_array_equal(a, 2.0 * b)
    errors = reference_errors(a, 2.0 * b, result["volumes"])
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    exact_errors = exact.saxpy_errors()
    assert exact_errors.l1 == 0.0
    assert exact_errors.l2 == 0.0
    assert exact_errors.linf == 0.0


def test_write_pf12_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf12_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_run_native_points_at_official_benchmarks():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "benchmarks/manifest.toml" in text
    with pytest.raises(run.NativeUnavailable, match="benchmarks/manifest.toml"):
        run.run_native()



def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
