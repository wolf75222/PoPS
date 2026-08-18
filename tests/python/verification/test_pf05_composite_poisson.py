"""PF-05 composite AMR Poisson stand-in (leaf residual; no live runtime)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.leaf_reference_errors import leaf_reference_errors
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "performance" / "composite_poisson"
AM10_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "composite_poisson"
PO01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
COVERED_PARENT_RESIDUAL = 1.0e6
ORDERS_REASON = "kernel stand-in, not a timed PF run"


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


def test_exact_reuses_am10_po01_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    assert "composite_poisson" in text or "periodic_trig" in text
    exact = _load_case_module("exact")
    po01 = load_sibling_module(PO01_EXACT)
    centers, _ = po01.uniform_cell_grid(32)
    np.testing.assert_array_equal(exact.phi_exact(centers), po01.phi_exact(centers))
    np.testing.assert_array_equal(exact.rhs_exact(centers), po01.rhs_exact(centers))
    np.testing.assert_array_equal(exact.e_exact(centers), po01.e_exact(centers))


def test_leaf_residual_defined():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    sample = run.two_level_residual()
    residual = np.asarray(sample["residual"], dtype=np.float64)
    leaf_mask = np.asarray(sample["leaf_mask"], dtype=np.bool_)
    assert leaf_mask.dtype == np.bool_
    assert residual.size == leaf_mask.size
    assert np.any(leaf_mask)
    assert np.any(~leaf_mask)
    errors = run.leaf_residual_errors(sample)
    assert np.isfinite(errors.l1)
    assert np.isfinite(errors.l2)
    assert np.isfinite(errors.linf)
    assert errors.l2 == pytest.approx(0.0, abs=1.0e-12)
    fraction = float(sample["refined_fraction"])
    assert 0.0 < fraction < 1.0


def test_covered_parent_not_double_counted():
    run = _load_case_module("run")
    am10 = load_sibling_module(AM10_DIR / "run.py")
    polluted = run.two_level_residual(parent_residual=COVERED_PARENT_RESIDUAL)
    clean = run.two_level_residual(parent_residual=0.0)
    polluted_errors = run.leaf_residual_errors(polluted)
    clean_errors = run.leaf_residual_errors(clean)
    assert polluted_errors.l1 == pytest.approx(clean_errors.l1)
    assert polluted_errors.l2 == pytest.approx(clean_errors.l2)
    assert polluted_errors.linf == pytest.approx(clean_errors.linf)
    expected = leaf_reference_errors(
        polluted["residual"],
        polluted["residual_exact"],
        polluted["volumes"],
        polluted["leaf_mask"],
    )
    assert polluted_errors.l1 == pytest.approx(expected.l1)
    assert polluted_errors.l2 == pytest.approx(expected.l2)
    assert polluted_errors.linf == pytest.approx(expected.linf)
    full = reference_errors(
        polluted["residual"], polluted["residual_exact"], polluted["volumes"]
    )
    assert full.linf == pytest.approx(COVERED_PARENT_RESIDUAL)
    assert polluted_errors.linf < 1.0
    assert polluted_errors.l2 == pytest.approx(0.0, abs=1.0e-12)
    am10_polluted = am10.two_level_residual(parent_residual=COVERED_PARENT_RESIDUAL)
    am10_errors = am10.leaf_residual_errors(am10_polluted)
    assert polluted_errors.l2 == pytest.approx(am10_errors.l2)
    assert polluted["n_levels"] == am10_polluted["n_levels"]


def test_write_pf05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_pf05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON
    assert loaded["poisson"]["residual_l2"] == pytest.approx(0.0, abs=1.0e-12)
    assert loaded["amr"]["invariants_ok"] is True
    one_node = loaded["performance"]["one_node"]
    assert one_node is not None
    assert 0.0 < float(one_node["refined_fraction"]) < 1.0


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
