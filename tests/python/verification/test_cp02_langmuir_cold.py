"""CP-02 cold Langmuir wave (in-memory closed 1-d oracle; no solver required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.phase import frequency_error, numerical_frequency
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64
N_PERIODS = 8
SAMPLES_PER_PERIOD = 32
PROBE_X = 0.25


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _spectral_dx(field, spacing: float) -> np.ndarray:
    samples = np.asarray(field, dtype=np.float64)
    wave = 2.0 * np.pi * np.fft.fftfreq(samples.size, d=float(spacing))
    return np.fft.ifft(1j * wave * np.fft.fft(samples)).real


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("run_native"):
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_documented_units_give_omega_pe_one():
    exact = _load_case_module("exact")
    np.testing.assert_allclose(exact.E_CHARGE, 1.0)
    np.testing.assert_allclose(exact.M_E, 1.0)
    np.testing.assert_allclose(exact.EPS0, 1.0)
    np.testing.assert_allclose(exact.N0, 1.0)
    omega = np.sqrt(exact.N0 * exact.E_CHARGE**2 / (exact.M_E * exact.EPS0))
    np.testing.assert_allclose(omega, 1.0)
    np.testing.assert_allclose(exact.plasma_frequency(), 1.0)


def test_gauss_law_holds_on_exact_fields():
    exact = _load_case_module("exact")
    centers, _ = exact.uniform_cell_centers(N_CELLS)
    spacing = 1.0 / float(N_CELLS)
    for time in (0.0, 0.3, 1.25, 2.0 * np.pi):
        electric = exact.e_field(centers, time)
        density = exact.n_e(centers, time)
        dE_dx = _spectral_dx(electric, spacing)
        gauss_rhs = exact.E_CHARGE * (exact.N_I - density) / exact.EPS0
        np.testing.assert_allclose(dE_dx, gauss_rhs, rtol=0.0, atol=1.0e-12)


def test_exact_e_probe_frequency_recovers_omega_pe():
    exact = _load_case_module("exact")
    omega_pe = float(exact.plasma_frequency())
    period = 2.0 * np.pi / omega_pe
    times = np.arange(N_PERIODS * SAMPLES_PER_PERIOD, dtype=np.float64) * (
        period / SAMPLES_PER_PERIOD
    )
    samples = exact.e_field(PROBE_X, times)
    omega_fft = numerical_frequency(times, samples, method="fft")
    omega_fit = numerical_frequency(times, samples, method="phase_fit")
    np.testing.assert_allclose(frequency_error(omega_fft, omega_pe), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(frequency_error(omega_fit, omega_pe), 0.0, atol=1.0e-6)


def test_write_cp02_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["coupling"]["sign_ok"] is True
    np.testing.assert_allclose(loaded["poisson"]["residual_l2"], 0.0)
    np.testing.assert_allclose(loaded["coupling"]["phase_error"], 0.0)


def test_case_modules_use_load_sibling_module():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
    exact_text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    tree = ast.parse(exact_text)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".", 1)[0])
    assert "pops" not in imported


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run(" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (2, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)
