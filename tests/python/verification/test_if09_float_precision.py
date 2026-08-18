"""IF-09 float32 vs float64 plateau (in-memory TR-01 sine; no live compile)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "float_precision"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
T = 0.25
LINF_BOUND = 1.0e-6
ORDERS_REASON = "float32 vs float64 plateau / no live compile"
FLOAT32_AUTHORING_REFUSAL = "public float32 Case authoring not active"


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


def test_dtype_contract():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    tr01 = load_sibling_module(TR01_EXACT)
    fields = run.evaluate_precisions(N_CELLS, t=T)
    f32 = np.asarray(fields["float32"])
    f64 = np.asarray(fields["float64"])
    assert f32.dtype == np.float32
    assert f64.dtype == np.float64
    assert np.all(np.isfinite(f32))
    assert np.all(np.isfinite(f64))
    centers = exact.cell_centers(N_CELLS)
    np.testing.assert_array_equal(f64, tr01.exact_sine(centers, T))
    assert exact.fields_are_finite(f32, f64)


def test_precision_difference_is_o_1e7():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    fields = run.evaluate_precisions(N_CELLS, t=T)
    errors = reference_errors(
        fields["float32"], fields["float64"], fields["volumes"]
    )
    assert exact.LINF_BOUND == LINF_BOUND
    assert 0.0 < errors.linf <= LINF_BOUND
    assert errors.l1 <= LINF_BOUND
    assert errors.l2 <= LINF_BOUND
    assert run.max_precision_difference(N_CELLS, t=T) <= LINF_BOUND
    assert analyze.precision_linf(N_CELLS, t=T) <= LINF_BOUND


def test_write_if09_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if09_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_run_native_is_capability_gated():
    run = _load_case_module("run")
    assert run.refuse_float32_case_authoring() == FLOAT32_AUTHORING_REFUSAL
    with pytest.raises(run.NativeUnavailable, match=FLOAT32_AUTHORING_REFUSAL):
        run.run_native(N_CELLS, t_end=T)


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
