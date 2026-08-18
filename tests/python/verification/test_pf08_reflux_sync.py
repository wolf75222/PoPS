"""PF-08 reflux / AMR sync stand-in (AM-09 residual; time is an observation)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.conservation import conservation_residual
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "reflux_sync"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
ORDERS_REASON = "reflux/AMR sync stand-in, time is an observation"


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


def test_closed_statement_with_reflux_has_zero_residual():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "conservation_residual" in text
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    terms = exact.closed_balance_terms()
    residual = conservation_residual(
        terms["storage_change"],
        terms["outward_boundary_flux"],
        terms["sources"],
        reflux=terms["reflux"],
        projection=terms["projection"],
    )
    np.testing.assert_allclose(residual, 0.0)
    np.testing.assert_allclose(run.residual(reflux=True), 0.0)
    assert float(terms["reflux"]) != 0.0
    observed = run.timed_sync(reflux=True)
    np.testing.assert_allclose(observed["residual"], 0.0)
    assert "elapsed_s" in observed
    assert float(observed["elapsed_s"]) >= 0.0


def test_open_statement_without_reflux_is_nonzero_negative_control():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    terms = exact.open_balance_terms()
    residual = conservation_residual(
        terms["storage_change"],
        terms["outward_boundary_flux"],
        terms["sources"],
        reflux=terms["reflux"],
        projection=terms["projection"],
    )
    assert float(residual) != 0.0
    assert float(terms["reflux"]) == 0.0
    open_residual = run.residual(reflux=False)
    assert float(open_residual) != 0.0
    np.testing.assert_allclose(open_residual, residual)
    observed = run.timed_sync(reflux=False)
    assert float(observed["residual"]) != 0.0
    assert float(observed["elapsed_s"]) >= 0.0


def test_write_pf08_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf08_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON
    notes = loaded["performance"]["one_node"]["notes"]
    assert "observation" in notes
    assert loaded["performance"]["one_node"]["cells_per_second"] is None
    assert loaded["amr"]["invariants_ok"] is True


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
