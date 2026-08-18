"""IF-03 Kokkos Serial vs OpenMP parity (in-memory exact fields; optional native)."""
from __future__ import annotations

import ast
import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "space_parity"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
EXECUTION_SPACES = ("KokkosSerial", "KokkosOpenMP")
N_CELLS = 32
T = 0.25
ORDERS_REASON = "exact-field identity / no live Kokkos"


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


def test_both_execution_space_labels_present():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    case = tomllib.loads((CASE_DIR / "case.toml").read_text(encoding="utf-8"))
    assert tuple(exact.EXECUTION_SPACES) == EXECUTION_SPACES
    assert list(case["execution_spaces"]) == list(EXECUTION_SPACES)
    fields = run.exact_fields_for_spaces(N_CELLS, t=T)
    assert tuple(fields) == EXECUTION_SPACES
    for name in EXECUTION_SPACES:
        assert name in fields
        assert fields[name].shape == (N_CELLS,)


def test_serial_and_openmp_fields_identical():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    volumes = exact.cell_volumes(N_CELLS)
    fields = run.exact_fields_for_spaces(N_CELLS, t=T)
    serial = fields["KokkosSerial"]
    openmp = fields["KokkosOpenMP"]
    errors = reference_errors(serial, openmp, volumes)
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0
    np.testing.assert_array_equal(serial, openmp)
    assert analyze.spaces_agree(N_CELLS, t=T) == 0.0
    assert run.max_space_difference(N_CELLS, t=T) == 0.0


def test_write_if03_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON
    assert loaded["execution_spaces"] == list(EXECUTION_SPACES)


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_run_native_reuses_tr01_serial_and_openmp_thread_labels():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "run_native" in text
    assert "OMP_NUM_THREADS" in text
    assert "resolve_artifact_dim" in text
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert exact.SPACE_THREADS["KokkosSerial"] == 1
    assert exact.SPACE_THREADS["KokkosOpenMP"] == 8
    assert run.space_threads("KokkosSerial") == 1
    assert run.space_threads("KokkosOpenMP") == int(os.environ.get("POPS_ORDER_OMP", "8"))


def test_gpu_space_is_capability_gated_before_native_run():
    run = _load_case_module("run")
    with pytest.raises(run.NativeUnavailable, match="CUDA"):
        run.space_threads("KokkosCuda")
    with pytest.raises(run.NativeUnavailable, match="CUDA"):
        run.run_native_gpu()
    with pytest.raises(run.NativeUnavailable, match="CUDA"):
        run.run_native(space="KokkosCuda")


def test_run_native_refuses_dim_mismatch_before_tr01(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM") as exc_info:
        run.run_native(N_CELLS, t_end=T)
    assert "fallback" in str(exc_info.value).lower()


@pytest.mark.compiler
def test_run_native_spaces_returns_finite_or_skips(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    try:
        fields = run.run_native_spaces(N_CELLS, t_end=T)
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert tuple(fields) == EXECUTION_SPACES
    for field in fields.values():
        assert field.shape == (N_CELLS,)
        assert np.isfinite(field).all()
