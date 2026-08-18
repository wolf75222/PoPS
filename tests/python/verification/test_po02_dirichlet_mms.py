"""PO-02 Dirichlet MMS Poisson (2-d oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "dirichlet_mms"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
TWO_PI = 2.0 * np.pi
MANUFACTURED_RESOLUTIONS = (16, 32, 64, 128)
MANUFACTURED_ERROR_SCALE = 0.04


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers_2d(n_cells: int):
    count = int(n_cells)
    width = 1.0 / count
    axis = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(axis, axis, indexing="ij")
    volumes = np.full((count, count), width * width, dtype=np.float64)
    return x, y, volumes


def _cell_centers_1d(n_cells: int) -> np.ndarray:
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


def test_minus_laplacian_matches_rhs_exact():
    exact = _load_case_module("exact")
    x, y, _ = _cell_centers_2d(64)
    phi = exact.phi_exact(x, y)
    d2x = np.exp(x) * np.sin(TWO_PI * y) + 2.0 * y
    d2y = -(TWO_PI**2) * np.exp(x) * np.sin(TWO_PI * y)
    np.testing.assert_allclose(
        exact.rhs_exact(x, y),
        -(d2x + d2y),
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        phi,
        np.exp(x) * np.sin(TWO_PI * y) + x**2 * y,
        rtol=0.0,
        atol=1.0e-14,
    )
    x1 = _cell_centers_1d(64)
    np.testing.assert_allclose(
        exact.rhs_exact_1d(x1),
        -np.exp(x1),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        exact.phi_exact_1d(x1),
        np.exp(x1),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_manufactured_h2_series_has_observed_order_two():
    exact = _load_case_module("exact")
    errors = []
    spacings = []
    for n_cells in MANUFACTURED_RESOLUTIONS:
        x, y, volumes = _cell_centers_2d(n_cells)
        phi = exact.phi_exact(x, y)
        spacing = 1.0 / n_cells
        manufactured = phi + MANUFACTURED_ERROR_SCALE * spacing**2
        errors.append(reference_errors(manufactured, phi, volumes).linf)
        spacings.append(spacing)
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0), rtol=1.0e-9, atol=1.0e-9)


def test_write_po02_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_po02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_po02_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_po02_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"


def test_case_modules_do_not_mention_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text
        assert "from exact import" not in text
