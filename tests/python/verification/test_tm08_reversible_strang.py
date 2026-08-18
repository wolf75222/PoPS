"""TM-08 reversible Strang (advance T, flip velocity, return; exact error 0)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "reversible_strang"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64
T = 0.25
A = 1.0
DT = 1.0 / 128.0


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
        exact.exact_sine(centers, T),
        tr01.exact_sine_1d(centers, T),
    )


def test_advance_then_flip_velocity_returns_to_initial():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    cycle = run.reversible_cycle(n_cells=N_CELLS, t=T, a=A)
    initial = np.asarray(cycle["initial"], dtype=np.float64)
    forwarded = np.asarray(cycle["forwarded"], dtype=np.float64)
    returned = np.asarray(cycle["returned"], dtype=np.float64)
    assert initial.size == N_CELLS
    assert not np.array_equal(forwarded, initial)
    np.testing.assert_array_equal(returned, initial)


def test_return_error_is_identically_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    cycle = run.reversible_cycle(n_cells=N_CELLS, t=T, a=A)
    errors = reference_errors(cycle["returned"], cycle["initial"], cycle["volumes"])
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    exact_errors = exact.return_errors(n_cells=N_CELLS, t=T, a=A)
    assert exact_errors.l1 == 0.0
    assert exact_errors.l2 == 0.0
    assert exact_errors.linf == 0.0


def test_write_tm08_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tm08_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(DT)
    plan = run.resolve_plan(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
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
