"""IF-05 output-cadence invariance (in-memory exact dumps; no live I/O)."""
from __future__ import annotations

import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "output_cadence"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
CADENCES = (1, 2, 10)
N_CELLS = 32
N_STEPS = 10
T = 0.25
MUTATION_AMPLITUDE = 0.25
ORDERS_REASON = "exact-field identity / no live output cadence"
OUTPUT_CADENCE_REFUSAL = "public output-cadence consumer attach not active"


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
    centers = exact.cell_centers(N_CELLS)
    np.testing.assert_array_equal(
        exact.exact_sine(centers, T),
        tr01.exact_sine(centers, T),
    )
    np.testing.assert_array_equal(
        exact.exact_field(centers, T),
        tr01.exact_sine(centers, T),
    )


def test_all_cadences_agree_exactly():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert tuple(exact.CADENCES) == CADENCES
    assert exact.T == T
    assert exact.N_STEPS == N_STEPS
    volumes = exact.cell_volumes(N_CELLS)
    centers = exact.cell_centers(N_CELLS)
    reference = exact.exact_field(centers, T)
    fields = run.exact_fields_for_cadences(N_CELLS, t=T)
    assert tuple(fields) == CADENCES
    for cadence, field in fields.items():
        sample = run.advance_cadence(cadence, n_cells=N_CELLS)
        assert sample["t"] == T
        assert len(sample["dumps"]) == N_STEPS // int(cadence)
        assert sample["dumps"][-1]["t"] == T
        np.testing.assert_array_equal(sample["field"], reference)
        np.testing.assert_array_equal(field, reference)
        errors = reference_errors(field, reference, volumes)
        assert errors.l1 == 0.0
        assert errors.l2 == 0.0
        assert errors.linf == 0.0
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        assert errors.l1 == 0.0
        assert errors.l2 == 0.0
        assert errors.linf == 0.0
        np.testing.assert_array_equal(left, right)
    assert analyze.cadences_agree(N_CELLS, t=T) == 0.0
    assert run.max_cadence_difference(N_CELLS, t=T) == 0.0


def test_mutated_leftover_dump_is_detected():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    centers = exact.cell_centers(N_CELLS)
    field = exact.exact_field(centers, T)
    snapshot = np.array(field, copy=True)
    dumped = run.dump_state(field)
    dumped[0] += MUTATION_AMPLITUDE
    np.testing.assert_array_equal(field, snapshot)
    leftover = run.mutate_dump(snapshot, amplitude=MUTATION_AMPLITUDE)
    volumes = exact.cell_volumes(N_CELLS)
    errors = reference_errors(leftover, field, volumes)
    assert errors.linf == MUTATION_AMPLITUDE
    assert errors.l1 > 0.0
    assert errors.l2 > 0.0
    assert run.leftover_dump_linf(amplitude=MUTATION_AMPLITUDE) == MUTATION_AMPLITUDE


def test_write_if05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if05_report(tmp_path)
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


def test_run_native_cadence_compare_or_skips():
    run = _load_case_module("run")
    assert run.refuse_output_cadence_consumer() == OUTPUT_CADENCE_REFUSAL
    try:
        result = run.run_native(16, t_end=0.1)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert float(result["linf"]) < 1.0e-12


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
