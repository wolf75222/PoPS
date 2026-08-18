"""PO-01 periodic trigonometric Poisson (1-d reduction; no solver required)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement

from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "periodic_trig"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
TWO_PI = 2.0 * np.pi
MANUFACTURED_RESOLUTIONS = (16, 32, 64, 128)
MANUFACTURED_ERROR_SCALE = 0.04


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"po01_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    phi = exact.phi_exact(x)
    rhs = exact.rhs_exact(x)
    np.testing.assert_allclose(rhs, (TWO_PI**2) * phi, rtol=0.0, atol=1.0e-14)


def test_minus_d2_phi_matches_rhs_exact_to_spectral_or_identity():
    exact = _load_case_module("exact")
    x = _cell_centers(256)
    phi = exact.phi_exact(x)
    rhs = exact.rhs_exact(x)
    identity = np.allclose(rhs, (TWO_PI**2) * phi, rtol=0.0, atol=1.0e-14)
    h = 1.0 / x.size
    wave = TWO_PI * np.fft.fftfreq(x.size, d=h)
    minus_d2 = np.fft.ifft((wave**2) * np.fft.fft(phi)).real
    spectral = np.allclose(minus_d2, rhs, rtol=0.0, atol=1.0e-10)
    assert identity or spectral


def test_e_exact_is_minus_two_pi_cos():
    exact = _load_case_module("exact")
    x = _cell_centers(64)
    np.testing.assert_allclose(
        exact.e_exact(x),
        -TWO_PI * np.cos(TWO_PI * x),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_observed_order_utility_recovers_quadratic_spacing():
    exact = _load_case_module("exact")
    errors = []
    spacings = []
    for n_cells in MANUFACTURED_RESOLUTIONS:
        x = _cell_centers(n_cells)
        volumes = np.full(n_cells, 1.0 / n_cells, dtype=np.float64)
        phi = exact.phi_exact(x)
        spacing = 1.0 / n_cells
        manufactured = phi + MANUFACTURED_ERROR_SCALE * spacing**2
        errors.append(reference_errors(manufactured, phi, volumes).linf)
        spacings.append(spacing)
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0), rtol=1.0e-9, atol=1.0e-9)


def test_write_po01_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_po01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_po01_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_po01_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    reasons = " ".join(item["reason"] for item in loaded["failures"]).lower()
    assert "native" in reasons


def test_case_modules_do_not_mention_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_potential_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        phi = np.asarray(run.run_native(16), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert phi.shape == (16,)
    assert np.isfinite(phi).all()

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_po01_report(
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
        CampaignJob(case_id="PO-01", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
