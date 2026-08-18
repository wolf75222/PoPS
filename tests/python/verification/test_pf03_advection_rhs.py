"""PF-03 1-d upwind advection FV RHS (numpy stand-in; optional TR-01 timer)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "advection_rhs"
TR01_RUN = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "run.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 16
HALO_WIDTH = 1
T_END = 0.05
ORDERS_REASON = "kernel microbench stand-in, not a timed PF run"
NATIVE_KEYS = ("elapsed_s", "n_cells", "n_steps", "cells_per_second", "field")


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


def test_halo_width_1_wrap():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert exact.HALO_WIDTH == HALO_WIDTH
    filled = run.periodic_halo_fill()
    halo = int(exact.HALO_WIDTH)
    interior = np.asarray(filled[halo:-halo], dtype=np.float64)
    left = np.asarray(filled[:halo], dtype=np.float64)
    right = np.asarray(filled[-halo:], dtype=np.float64)
    assert interior.size == N_CELLS
    np.testing.assert_array_equal(left, interior[-halo:])
    np.testing.assert_array_equal(right, interior[:halo])
    assert exact.halo_linf(filled) == 0.0


def test_interior_rhs_matches_analytic_to_fd_order():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    result = run.interior_rhs()
    rhs = np.asarray(result["rhs"], dtype=np.float64)
    centers = np.asarray(result["centers"], dtype=np.float64)
    volumes = np.asarray(result["volumes"], dtype=np.float64)
    analytic = exact.exact_rhs(centers, 0.0)
    assert rhs.shape == analytic.shape
    assert rhs.size == N_CELLS
    omega = 2.0 * np.pi * float(exact.K)
    np.testing.assert_allclose(
        analytic,
        -float(exact.A) * float(exact.EPS) * omega * np.cos(omega * centers),
    )
    dx = 1.0 / float(N_CELLS)
    bound = abs(float(exact.A)) * float(exact.EPS) * omega**2 * dx
    errors = reference_errors(rhs, analytic, volumes)
    assert errors.linf <= bound
    np.testing.assert_allclose(rhs, analytic, atol=bound)


def test_write_pf03_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf03_report(tmp_path)
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
            assert "pops.run(" not in text
        else:
            assert owners == []
            assert "pops.run(" not in text
