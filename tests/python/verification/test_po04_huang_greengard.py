"""PO-04 Huang–Greengard multi-blob Poisson (1-d stand-in; no solver required)."""
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
import inspect
from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.provenance import RUN_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "poisson" / "huang_greengard"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
CENTRES = (0.25, 0.5, 0.75)
AMPLITUDES = (1.0, 1.0, 1.0)
SIGMA = 0.04
PERIOD = 1.0
MANUFACTURED_RESOLUTIONS = (16, 32, 64, 128)
MANUFACTURED_ERROR_SCALE = 0.04
FD_RESOLUTIONS = (128, 256, 512)
LOCALIZATION_SIGMA = 5.0
LOCALIZATION_FLOOR = 1.0e-6


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_centers(n_cells: int) -> np.ndarray:
    width = 1.0 / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def _nearest_image(x, centre, period: float = PERIOD) -> np.ndarray:
    displacement = np.asarray(x, dtype=np.float64) - float(centre)
    width = float(period)
    return displacement - width * np.round(displacement / width)


def _blob_phi(x, amplitude, centre, sigma: float = SIGMA) -> np.ndarray:
    radius = _nearest_image(x, centre)
    return float(amplitude) * np.exp(-np.square(radius / float(sigma)))


def _independent_phi(x) -> np.ndarray:
    samples = np.asarray(x, dtype=np.float64)
    total = np.zeros_like(samples, dtype=np.float64)
    for amplitude, centre in zip(AMPLITUDES, CENTRES, strict=True):
        total = total + _blob_phi(samples, amplitude, centre)
    return total


def _independent_rhs(x) -> np.ndarray:
    """Independent −φ'' of the nearest-image multi-blob potential."""
    samples = np.asarray(x, dtype=np.float64)
    total = np.zeros_like(samples, dtype=np.float64)
    sigma = float(SIGMA)
    for amplitude, centre in zip(AMPLITUDES, CENTRES, strict=True):
        radius = _nearest_image(samples, centre)
        scaled = radius / sigma
        total = total + (2.0 * float(amplitude) / sigma**2) * (
            1.0 - 2.0 * np.square(scaled)
        ) * np.exp(-np.square(scaled))
    return total


def _periodic_minus_d2(phi, spacing: float) -> np.ndarray:
    field = np.asarray(phi, dtype=np.float64)
    width = float(spacing)
    return -(np.roll(field, 1) - 2.0 * field + np.roll(field, -1)) / (width * width)


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


def test_phi_is_sum_of_three_nearest_image_gaussians():
    exact = _load_case_module("exact")
    assert tuple(exact.CENTRES) == CENTRES
    assert tuple(exact.AMPLITUDES) == AMPLITUDES
    np.testing.assert_allclose(exact.SIGMA, SIGMA, rtol=0.0, atol=0.0)
    x = _cell_centers(64)
    np.testing.assert_allclose(
        exact.phi_exact(x),
        _independent_phi(x),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        exact.phi_exact(-0.01),
        exact.phi_exact(0.99),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        exact.nearest_image_displacement(0.99, 0.25),
        0.99 - 1.25,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_minus_d2_phi_matches_rhs_exact():
    exact = _load_case_module("exact")
    x = _cell_centers(128)
    rhs = exact.rhs_exact(x)
    np.testing.assert_allclose(rhs, _independent_rhs(x), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(rhs, -exact.d2phi_exact(x), rtol=0.0, atol=1.0e-14)


def test_finite_diff_residual_decreases_as_n_increases():
    exact = _load_case_module("exact")
    residuals = []
    spacings = []
    for n_cells in FD_RESOLUTIONS:
        x = _cell_centers(n_cells)
        spacing = 1.0 / float(n_cells)
        residual = _periodic_minus_d2(exact.phi_exact(x), spacing) - exact.rhs_exact(x)
        residuals.append(float(np.max(np.abs(residual))))
        spacings.append(spacing)
    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[-1] < residuals[0]
    orders = observed_order(residuals, spacings)
    assert np.all(np.isfinite(orders))
    assert float(orders[-1]) > 1.5


def test_blobs_are_localized():
    exact = _load_case_module("exact")
    for centre, amplitude in zip(CENTRES, AMPLITUDES, strict=True):
        np.testing.assert_allclose(
            exact.phi_exact(centre),
            amplitude,
            rtol=0.0,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(exact.phi_exact(0.0), 0.0, rtol=0.0, atol=1.0e-12)
    midpoints = (0.375, 0.625)
    for midpoint in midpoints:
        assert abs(float(exact.phi_exact(midpoint))) < 1.0e-3
    samples = np.linspace(0.0, 1.0, 2000, endpoint=False)
    significant = np.abs(exact.phi_exact(samples)) > LOCALIZATION_FLOOR
    assert np.any(significant)
    for location in samples[significant]:
        distances = [
            abs(float(exact.nearest_image_displacement(location, centre)))
            for centre in CENTRES
        ]
        assert min(distances) < LOCALIZATION_SIGMA * SIGMA


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


def test_write_po04_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_po04_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_po04_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_po04_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    from verification.pops_verify.native_evidence import REDUCED_NOT_SUPPORTED

    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["coverage"]["cases_failed"] == 1
    assert loaded["coverage"]["cases_not_supported"] == 0
    assert loaded["not_applicable_reason"]["orders"] == REDUCED_NOT_SUPPORTED["PO-04"]
    assert loaded["poisson"]["potential_error"] is None


def test_case_modules_do_not_mention_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text
        assert "from exact import" not in text

def test_report_orders_come_from_supplied_native_series(tmp_path: Path):
    analyze = _load_case_module("analyze")
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    linf = [0.08, 0.03, 0.011]
    analyze.write_po04_report(
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
        CampaignJob(case_id="PO-04", pops_native_dim=1, min_resolution=16)
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert "result" in result
