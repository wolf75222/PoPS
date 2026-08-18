"""PO-07 elliptic tolerance sweep (1-d PO-01 oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import ARTIFACTS
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "elliptic_tolerance"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
TWO_PI = 2.0 * np.pi
RESOLUTIONS = (16, 32, 64, 128)
FIXED_N = 32
DOUBLED_N = 64
TOLERANCE_SWEEP = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int) -> np.ndarray:
    width = 1.0 / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


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


def test_rhs_exact_is_two_pi_squared_times_phi_exact():
    exact = _load_case_module("exact")
    x = _cell_centers(64)
    np.testing.assert_allclose(
        exact.rhs_exact(x),
        (TWO_PI**2) * exact.phi_exact(x),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_discretization_error_is_proportional_to_h_squared():
    exact = _load_case_module("exact")
    errors = [exact.discretization_error(n_cells) for n_cells in RESOLUTIONS]
    spacings = [1.0 / float(n_cells) for n_cells in RESOLUTIONS]
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0), rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(
        exact.discretization_error(DOUBLED_N),
        exact.discretization_error(FIXED_N) / 4.0,
        rtol=0.0,
        atol=1.0e-16,
    )


def test_algebraic_error_equals_tolerance():
    exact = _load_case_module("exact")
    for tol in TOLERANCE_SWEEP:
        np.testing.assert_allclose(
            exact.algebraic_error(tol),
            tol,
            rtol=0.0,
            atol=0.0,
        )


def test_combined_error_is_max_of_discretization_and_algebraic():
    exact = _load_case_module("exact")
    disc = exact.discretization_error(FIXED_N)
    loose = 10.0 * disc
    tight = 0.1 * disc
    np.testing.assert_allclose(
        exact.combined_error(FIXED_N, loose),
        loose,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        exact.combined_error(FIXED_N, tight),
        disc,
        rtol=0.0,
        atol=0.0,
    )
    for tol in TOLERANCE_SWEEP:
        np.testing.assert_allclose(
            exact.combined_error(FIXED_N, tol),
            max(disc, float(tol)),
            rtol=0.0,
            atol=0.0,
        )


def test_error_plateaus_versus_tolerance_at_fixed_n():
    exact = _load_case_module("exact")
    disc = exact.discretization_error(FIXED_N)
    errors = [exact.combined_error(FIXED_N, tol) for tol in TOLERANCE_SWEEP]
    assert errors[0] > errors[1] > disc
    plateau = [value for value in errors if np.isclose(value, disc, rtol=0.0, atol=1.0e-16)]
    assert len(plateau) >= 3
    np.testing.assert_allclose(plateau, np.full(len(plateau), disc), rtol=0.0, atol=1.0e-16)
    tight = [exact.combined_error(FIXED_N, tol) for tol in TOLERANCE_SWEEP if tol < disc]
    np.testing.assert_allclose(tight, np.full(len(tight), disc), rtol=0.0, atol=1.0e-16)


def test_discretization_plateau_drops_when_n_doubles():
    exact = _load_case_module("exact")
    coarse = exact.discretization_error(FIXED_N)
    fine = exact.discretization_error(DOUBLED_N)
    assert fine < coarse
    np.testing.assert_allclose(fine, coarse / 4.0, rtol=0.0, atol=1.0e-16)
    tight = 1.0e-12
    np.testing.assert_allclose(
        exact.combined_error(FIXED_N, tight),
        coarse,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        exact.combined_error(DOUBLED_N, tight),
        fine,
        rtol=0.0,
        atol=1.0e-16,
    )
    assert exact.combined_error(DOUBLED_N, tight) < exact.combined_error(FIXED_N, tight)


def test_write_po07_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_po07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_po07_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_po07_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    from verification.pops_verify.native_evidence import REDUCED_NOT_SUPPORTED

    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 0
    assert loaded["coverage"]["cases_not_supported"] == 1
    assert loaded["not_applicable_reason"]["orders"] == REDUCED_NOT_SUPPORTED["PO-07"]


def test_case_modules_do_not_mention_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_po07_report(
        tmp_path,
        native_series={"linf": linf, "spacings": spacings},
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1 or loaded["coverage"]["cases_not_supported"] == 1


def test_run_native_accepts_fail_closed_campaign_request():
    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest.from_job(
        CampaignJob(case_id="PO-07", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
