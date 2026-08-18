"""AM-11 Euler–Poisson AMR sync (leaf-only charge plus public Case)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "ep_sync"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")


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


def test_leaf_only_charge_excludes_covered_parent():
    exact = _load_case_module("exact")
    hierarchy = exact.two_level_hierarchy()
    rho = np.asarray(hierarchy["charge_density"], dtype=np.float64)
    volumes = np.asarray(hierarchy["volumes"], dtype=np.float64)
    leaf_mask = np.asarray(hierarchy["leaf_mask"])
    assert leaf_mask.dtype == np.bool_
    assert np.any(~leaf_mask)

    leaf = exact.leaf_net_charge(rho, volumes, leaf_mask)
    expected = float(np.sum(rho[leaf_mask] * volumes[leaf_mask]))
    np.testing.assert_allclose(leaf, expected, rtol=0.0, atol=0.0)

    poisoned = rho.copy()
    poisoned[~leaf_mask] = 1.0e9
    poisoned_leaf = exact.leaf_net_charge(poisoned, volumes, leaf_mask)
    np.testing.assert_allclose(poisoned_leaf, leaf, rtol=0.0, atol=0.0)


def test_covered_parent_must_not_double_count_net_charge():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    hierarchy = exact.two_level_hierarchy()
    rho = np.asarray(hierarchy["charge_density"], dtype=np.float64)
    volumes = np.asarray(hierarchy["volumes"], dtype=np.float64)
    leaf_mask = np.asarray(hierarchy["leaf_mask"])

    leaf = exact.leaf_net_charge(rho, volumes, leaf_mask)
    parent = exact.covered_parent_charge(rho, volumes, leaf_mask)
    naive = exact.naive_net_charge(rho, volumes)
    composed = run.compose_charge(rho, volumes, leaf_mask)

    assert parent != 0.0
    np.testing.assert_allclose(naive, leaf + parent, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(composed, leaf, rtol=0.0, atol=0.0)
    assert composed != naive
    np.testing.assert_allclose(
        parent,
        exact.restricted_parent_charge(hierarchy),
        rtol=0.0,
        atol=1e-15,
    )


def test_write_am11_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_am11_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["amr"]["invariants_ok"] is True


def test_siblings_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


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
    case = run.build_case(16)
    plan = run.resolve_plan(16)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(n_cells=16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
