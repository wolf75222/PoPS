"""CP-12 two-species charge cancellation (in-memory oracle; no solver required)."""
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
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "charge_cancel"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int = N_CELLS) -> np.ndarray:
    width = 1.0 / float(n_cells)
    return (np.arange(n_cells, dtype=np.float64) + 0.5) * width


def test_net_charge_is_identically_zero():
    exact = _load_case_module("exact")
    x = _cell_centers()
    q_plus, q_minus = exact.charges()
    np.testing.assert_allclose(q_plus + q_minus, 0.0, rtol=0.0, atol=0.0)
    for delta in (0.0, 0.1):
        density = exact.density(x, delta=delta)
        net = exact.net_charge(x, delta=delta)
        swapped = exact.net_charge(x, delta=delta, order=("minus", "plus"))
        np.testing.assert_allclose(net, q_plus * density + q_minus * density, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(net, 0.0)
        np.testing.assert_array_equal(swapped, 0.0)
        np.testing.assert_array_equal(exact.poisson_rhs(x, delta=delta), 0.0)


def test_electric_field_is_identically_zero_and_phi_is_constant():
    exact = _load_case_module("exact")
    x = _cell_centers()
    field = exact.e_exact(x)
    potential = exact.phi_exact(x)
    np.testing.assert_array_equal(field, 0.0)
    np.testing.assert_array_equal(potential, exact.PHI0)
    assert np.unique(potential).size == 1


def test_write_cp12_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp12_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["poisson"]["field_error"] == 0.0
    assert loaded["poisson"]["potential_error"] == 0.0
    assert loaded["poisson"]["residual_l2"] == 0.0


def test_siblings_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_potential_or_skips():
    from tests.python.support.requirements import missing_compiler_requirement

    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        phi = np.asarray(run.run_native(16, delta=0.1), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert phi.shape == (16,)
    assert np.isfinite(phi).all()
    np.testing.assert_allclose(phi - float(np.mean(phi)), 0.0, atol=1.0e-10)
