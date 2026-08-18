"""PO-03 Neumann nullspace Poisson (1-d; mean-free comparison; no solver required)."""
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
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "neumann_nullspace"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
TWO_PI = 2.0 * np.pi
N_CELLS = 32
GAUGE_SHIFT = 3.25


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_grid(n_cells: int = N_CELLS):
    width = 1.0 / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    volumes = np.full(n_cells, width, dtype=np.float64)
    return centers, volumes


def _pops_run_call_outside_run_native(text: str) -> bool:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "run"):
            continue
        value = func.value
        if not (isinstance(value, ast.Name) and value.id == "pops"):
            continue
        if _enclosing_function(tree, node) != "run_native":
            return True
    return False


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str | None:
    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []
            self.found: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def generic_visit(self, node: ast.AST) -> None:
            if node is target and self.stack:
                self.found = self.stack[-1]
            super().generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.found


def test_phi_exact_is_cos_two_pi_x_and_rhs_is_two_pi_squared():
    exact = _load_case_module("exact")
    x, _ = _cell_grid(64)
    phi = exact.phi_exact(x)
    rhs = exact.rhs_exact(x)
    np.testing.assert_allclose(phi, np.cos(TWO_PI * x), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(rhs, (TWO_PI**2) * np.cos(TWO_PI * x), rtol=0.0, atol=1.0e-14)


def test_homogeneous_neumann_at_endpoints():
    exact = _load_case_module("exact")
    np.testing.assert_allclose(exact.dphi_exact(0.0), 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(exact.dphi_exact(1.0), 0.0, rtol=0.0, atol=1.0e-14)


def test_mean_free_comparison_removes_constant_nullspace():
    exact = _load_case_module("exact")
    x, volumes = _cell_grid()
    phi = exact.phi_exact(x)
    shifted = phi + GAUGE_SHIFT
    mean_free_phi = exact.mean_free(phi, volumes)
    mean_free_shifted = exact.mean_free(shifted, volumes)
    np.testing.assert_allclose(mean_free_shifted, mean_free_phi, rtol=0.0, atol=1.0e-14)
    errors = reference_errors(mean_free_shifted, mean_free_phi, volumes)
    assert errors.linf < 1.0e-15
    assert abs(float(np.average(mean_free_phi, weights=volumes))) < 1.0e-14


def test_constant_rhs_is_incompatible():
    run = _load_case_module("run")
    _, volumes = _cell_grid()
    constant = np.ones(N_CELLS, dtype=np.float64)
    with pytest.raises(run.IncompatibleRhs):
        run.require_compatible_rhs(constant, volumes)
    observation = run.incompatible_rhs_observation(constant, volumes)
    assert observation["compatible"] is False
    assert observation["reason"]


def test_oracle_rhs_is_compatible():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    x, volumes = _cell_grid()
    run.require_compatible_rhs(exact.rhs_exact(x), volumes)


def test_write_po03_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_po03_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_po03_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_po03_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    reasons = " ".join(item["reason"] for item in loaded["failures"]).lower()
    assert "native" in reasons
    assert loaded["poisson"]["potential_error"] is None


def test_case_modules_do_not_mention_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_po03_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 1
    expected = [float(value) for value in observed_order(linf, spacings)]
    observed = [row["observed_order"] for row in loaded["orders"]]
    np.testing.assert_allclose(observed, expected)
    assert not np.allclose(observed, np.full(len(observed), 2.0))


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="PO-03", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
