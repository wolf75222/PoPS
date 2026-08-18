"""AM-05 regrid frequency sweep (public 1-d AMR authoring plus in-memory helpers)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "amr" / "regrid_frequency"
TR02_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "gaussian_pulse"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
K_VALUES = (1, 2, 4, 8, 16)
N_STEPS = 16


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _load_tr02(name: str):
    return load_sibling_module(TR02_DIR / f"{name}.py")


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


def test_rebuilds_over_n_steps_equal_n_over_k():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "gaussian_pulse" in text
    assert "from exact import" not in text
    assert tuple(exact.K_VALUES) == K_VALUES
    assert exact.N_STEPS == N_STEPS
    for k in K_VALUES:
        expected = N_STEPS // k
        assert exact.expected_rebuilds(k) == expected
        assert run.count_rebuilds(k) == expected


def test_leftover_k_regrid_minus_k_requested_is_zero():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    for k in K_VALUES:
        observed = run.observed_regrid_interval(k)
        leftover = exact.interval_leftover(observed, k)
        assert leftover == pytest.approx(0.0, abs=1.0e-12)
        assert run.interval_leftover(k) == pytest.approx(0.0, abs=1.0e-12)


def test_exact_field_has_no_linear_leftover_vs_inv_k():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    tr02 = _load_tr02("exact")
    inv_k, leftovers, slope = run.field_leftover_vs_inv_k()
    np.testing.assert_allclose(inv_k, [1.0 / float(k) for k in K_VALUES])
    np.testing.assert_allclose(leftovers, np.zeros(len(K_VALUES)))
    assert slope == pytest.approx(0.0, abs=1.0e-12)
    centers, _ = run.uniform_cells()
    t_end = exact.N_STEPS * exact.DT
    reference = tr02.exact_gaussian(centers, t_end)
    for k in K_VALUES:
        field = run.exact_field_at_regrid_frequency(k)
        np.testing.assert_allclose(field, reference)


def test_write_am05_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_am05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["amr"]["invariants_ok"] is True


def test_siblings_reuse_tr02_via_load_sibling_module():
    for name in ("exact.py", "run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "from exact import" not in text
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in exact_text
    assert "gaussian_pulse" in exact_text


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
    case = run.build_case(n_cells=8, regrid_every=2)
    plan = run.resolve_plan(n_cells=8, regrid_every=2)
    assert case is not None
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 1


@pytest.mark.compiler
def test_run_native_returns_finite_or_skips():
    run = _load_case_module("run")
    try:
        field = np.asarray(
            run.run_native(8, t_end=0.05, regrid_every=2),
            dtype=np.float64,
        )
    except run.NativeUnavailable as exc:
        pytest.skip(str(exc))
    assert field.size > 0
    assert field.shape == (8,)
    assert np.isfinite(field).all()
