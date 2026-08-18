"""PF-10 checkpoint / HDF5 stand-in (1-d numpy npz; no live HDF5)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "checkpoint_io"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
ORDERS_REASON = "npz checkpoint stand-in, not a timed PF run"


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _imported_module_names(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


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


def test_round_trip_identity(tmp_path: Path):
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    field = exact.manufactured_field()
    assert np.asarray(field, dtype=np.float64).size == N_CELLS
    dest = tmp_path / exact.ARTIFACT_NAME
    result = run.round_trip(dest)
    restored = np.asarray(result["restored"], dtype=np.float64)
    errors = reference_errors(restored, field, exact.cell_volumes())
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    np.testing.assert_array_equal(restored, field)
    assert dest.is_file()
    assert result["bytes"] == dest.stat().st_size
    assert result["bytes"] > 0
    assert result["write_time_s"] == exact.FAKE_WRITE_TIME_S
    assert result["write_time_s"] > 0.0


def test_no_output_path_does_not_write_artifact(tmp_path: Path):
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    dest = tmp_path / exact.ARTIFACT_NAME
    result = run.run_no_output(dest)
    assert dest.exists() is False
    assert list(tmp_path.iterdir()) == []
    assert result["wrote"] is False
    assert result["bytes"] == 0
    assert result["write_time_s"] == 0.0


def test_write_pf10_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf10_report(tmp_path)
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
        imported = _imported_module_names(text)
        assert not any(item == "h5py" or item.startswith("h5py.") for item in imported)
        assert "hdf5" not in text.lower()
        if name == "exact.py":
            assert not any(item == "pops" or item.startswith("pops.") for item in imported)
