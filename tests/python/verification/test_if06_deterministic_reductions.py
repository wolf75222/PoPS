"""IF-06 deterministic reductions (in-memory exact-representable sums; no live MPI)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = (
    REPO_ROOT / "verification" / "cases" / "infrastructure" / "deterministic_reductions"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
BLOCK_SIZE = 8
ORDERS_REASON = "reduction identity / no live MPI"


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
            if any(
                alias.name == "pops" or alias.name.startswith("pops.")
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "pops" or module.startswith("pops."):
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


def _float64_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float64).view(np.uint64).reshape(()))


def test_geometric_series_is_exactly_representable():
    exact = _load_case_module("exact")
    field = np.asarray(exact.geometric_field(N_CELLS), dtype=np.float64)
    assert field.shape == (N_CELLS,)
    assert field.dtype == np.float64
    for k, value in enumerate(field):
        expected = float(np.ldexp(np.float64(1.0), -int(k)))
        assert float(value) == expected
        assert _float64_bits(float(value)) == _float64_bits(expected)
    closed = float(exact.closed_form_sum(N_CELLS))
    expected_sum = float(np.ldexp(np.float64(1.0), 1) - np.ldexp(np.float64(1.0), 1 - N_CELLS))
    assert closed == expected_sum
    assert _float64_bits(closed) == _float64_bits(expected_sum)


def test_official_diagnostics_are_attached_instead_of_python_trees():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert "attach_case_diagnostics" in text
    assert "sequential_reduce" not in text
    assert "pairwise_reduce" not in text
    assert "blocked_reduce" not in text
    field = exact.geometric_field(N_CELLS)
    oracle = float(exact.closed_form_sum(N_CELLS))
    assert field.shape == (N_CELLS,)
    assert math.isfinite(oracle)
    assert analyze.exact_sums_agree(N_CELLS) == 0.0
    assert hasattr(run, "build_case")
    assert hasattr(run, "resolve_plan")


def test_write_if06_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_run_native_repeat_or_skips():
    run = _load_case_module("run")
    try:
        result = run.run_native(16, t_end=0.1)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert float(result["linf"]) == 0.0
    assert result["diagnostics"] == "pops.diagnostics"


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native", "advance_once"}
        else:
            assert owners == []
            assert "pops.run(" not in text
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert not _imports_pops(exact_text)
