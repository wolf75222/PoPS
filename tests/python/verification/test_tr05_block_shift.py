"""TR-05 translated block faces (in-memory exact fields; no live runtime)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "block_shift"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
INTERFACE_X = (0.25, 0.2578125, 0.375)
N_CELLS = 128


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


def test_run_authors_translated_amr_joins():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "amr_shift_layout" in text
    assert "uniform_periodic_layout" not in text
    assert "AMRHierarchy" in text
    plan = run.resolve_plan(16, interface=0.25)
    assert plan is not None


def test_canonical_interfaces_are_cell_faces_on_n128():
    exact = _load_case_module("exact")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    assert tuple(exact.INTERFACE_X) == INTERFACE_X
    width = 1.0 / float(N_CELLS)
    for interface in exact.INTERFACE_X:
        n_left = exact.n_left_cells(N_CELLS, interface)
        assert n_left * width == interface
        assert 0 < n_left < N_CELLS


def test_exact_fields_are_identical_across_decompositions():
    exact = _load_case_module("exact")
    fields = [
        exact.exact_on_decomposition(N_CELLS, interface, 0.25)
        for interface in exact.INTERFACE_X
    ]
    assert len(fields) == 3
    np.testing.assert_array_equal(fields[0], fields[1])
    np.testing.assert_array_equal(fields[1], fields[2])
    np.testing.assert_array_equal(fields[0], fields[2])


def test_decomposition_difference_is_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    volumes = exact.cell_volumes(N_CELLS)
    fields = run.exact_fields_for_interfaces(N_CELLS, t=0.25)
    assert tuple(fields) == INTERFACE_X
    for left, right in ((0.25, 0.2578125), (0.25, 0.375), (0.2578125, 0.375)):
        errors = reference_errors(fields[left], fields[right], volumes)
        assert errors.l1 == 0.0
        assert errors.l2 == 0.0
        assert errors.linf == 0.0
    assert run.max_decomposition_difference(N_CELLS, t=0.25) == 0.0


def test_write_tr05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tr05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


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
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    from tests.python.support.requirements import missing_compiler_requirement

    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (16,)
    assert np.isfinite(field).all()
