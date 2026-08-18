"""PO-06 CF gradient placement (1-d PO-01 oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.interface_error import (
    band_max_abs_error,
    interface_band_mask,
)
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "cf_gradient"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
TWO_PI = 2.0 * np.pi
MANUFACTURED_RESOLUTIONS = (16, 32, 64, 128)
MANUFACTURED_ERROR_SCALE = 0.04
EXPECTED_PLACEMENTS = {
    "max_phi": 0.25,
    "zero": 0.0,
    "max_abs_dphi": 0.0,
    "max_abs_d2phi": 0.25,
}


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


def test_interface_locations_are_max_phi_zero_max_grad_max_laplace():
    exact = _load_case_module("exact")
    assert exact.interface_location("max_phi") == EXPECTED_PLACEMENTS["max_phi"]
    assert exact.interface_location("zero") == EXPECTED_PLACEMENTS["zero"]
    assert exact.interface_location("max_abs_dphi") == EXPECTED_PLACEMENTS["max_abs_dphi"]
    assert (
        exact.interface_location("max_abs_d2phi")
        == EXPECTED_PLACEMENTS["max_abs_d2phi"]
    )
    x = np.linspace(0.0, 1.0, 4000, endpoint=False)
    phi = exact.phi_exact(x)
    dphi = exact.dphi_exact(x)
    d2phi = exact.d2phi_exact(x)
    np.testing.assert_allclose(exact.phi_exact(0.25), np.max(phi), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(exact.phi_exact(0.0), 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.abs(exact.dphi_exact(0.0)),
        np.max(np.abs(dphi)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.abs(exact.d2phi_exact(0.25)),
        np.max(np.abs(d2phi)),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_e_exact_is_minus_two_pi_cos():
    exact = _load_case_module("exact")
    x = _cell_centers(64)
    np.testing.assert_allclose(
        exact.e_exact(x),
        -TWO_PI * np.cos(TWO_PI * x),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_cf_error_defined_at_each_interface_location():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    x = _cell_centers(64)
    h_fine = 1.0 / 64.0
    e = exact.e_exact(x)
    manufactured = e + MANUFACTURED_ERROR_SCALE * h_fine**2
    for name, x_interface in EXPECTED_PLACEMENTS.items():
        sample = run.placement_sample(64, name)
        np.testing.assert_allclose(sample["x_interface"], x_interface, rtol=0.0, atol=0.0)
        mask = interface_band_mask(sample["distance"], h_fine=h_fine)
        e_cf = band_max_abs_error(manufactured, e, mask)
        assert np.isfinite(e_cf)
        assert e_cf > 0.0


def test_manufactured_e_order_two_at_every_placement():
    run = _load_case_module("run")
    for name in EXPECTED_PLACEMENTS:
        errors = []
        spacings = []
        for n_cells in MANUFACTURED_RESOLUTIONS:
            sample = run.manufactured_gradient(n_cells, name)
            errors.append(sample["field_error"])
            spacings.append(1.0 / float(n_cells))
        orders = observed_order(errors, spacings)
        np.testing.assert_allclose(
            orders, np.full(orders.shape, 2.0), rtol=1.0e-9, atol=1.0e-9
        )
        assert float(orders[-1]) > 1.8


def test_write_po06_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_po06_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_po06_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_po06_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"


def test_case_modules_do_not_mention_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text
        assert "from exact import" not in text


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
    assert phi.size > 0
    assert np.isfinite(phi).all()
