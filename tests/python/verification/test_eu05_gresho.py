"""EU-05 Gresho vortex (in-memory stationary oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler" / "gresho"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32
SAMPLE_RADII = (0.1, 0.3, 0.6)
RESIDUAL_ATOL = 1.0e-12


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_mesh(exact, n_cells: int = N_CELLS):
    length = float(exact.PERIOD)
    width = length / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y


def test_radial_velocity_of_exact_field_is_zero():
    exact = _load_case_module("exact")
    x, y = _cell_mesh(exact)
    state = exact.exact_gresho(x, y, 0.0)
    later = exact.exact_gresho(x, y, 1.0)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_array_equal(state[key], later[key])
    radial = exact.radial_velocity(x, y, 0.0)
    np.testing.assert_allclose(radial, 0.0, rtol=0.0, atol=1.0e-14)
    dx = np.asarray(x, dtype=np.float64) - float(exact.X0)
    dy = np.asarray(y, dtype=np.float64) - float(exact.Y0)
    radius = np.hypot(dx, dy)
    reconstructed = np.zeros_like(radius)
    nonzero = radius > 0.0
    reconstructed[nonzero] = (
        np.asarray(state["u"])[nonzero] * dx[nonzero] / radius[nonzero]
        + np.asarray(state["v"])[nonzero] * dy[nonzero] / radius[nonzero]
    )
    np.testing.assert_allclose(reconstructed, 0.0, rtol=0.0, atol=1.0e-14)


def test_pressure_continuous_at_r_kinks():
    exact = _load_case_module("exact")
    assert exact.R1 == 0.2
    assert exact.R2 == 0.4
    r1 = float(exact.R1)
    r2 = float(exact.R2)
    np.testing.assert_allclose(
        exact.pressure_inner(r1),
        exact.pressure_middle(r1),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        exact.pressure_middle(r2),
        exact.pressure_outer(r2),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        exact.pressure(r1),
        exact.pressure_inner(r1),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        exact.pressure(r2),
        exact.pressure_outer(r2),
        rtol=0.0,
        atol=0.0,
    )


def test_centrifugal_balance_residual_near_zero_in_each_piece():
    exact = _load_case_module("exact")
    assert 0.0 < SAMPLE_RADII[0] < float(exact.R1)
    assert float(exact.R1) < SAMPLE_RADII[1] < float(exact.R2)
    assert SAMPLE_RADII[2] > float(exact.R2)
    residuals = exact.centrifugal_residual(SAMPLE_RADII)
    np.testing.assert_allclose(residuals, 0.0, rtol=0.0, atol=RESIDUAL_ATOL)


def test_write_eu05_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_eu05_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert "stationary vortex" in loaded["not_applicable_reason"]["orders"]


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "from exact import" not in text
        if name != "exact.py":
            assert "load_sibling_module" in text
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)
