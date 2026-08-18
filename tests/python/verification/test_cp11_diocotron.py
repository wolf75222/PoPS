"""CP-11 linear diocotron mode (in-memory toy growth oracle; no solver)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "diocotron"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_THETA = 64
GROWTH_TIME = 1.0
ORDERS_REASON = "linear growth / not a published reproduction"


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def _mid_ring_radius(exact) -> float:
    return 0.5 * (float(exact.R1) + float(exact.R2))


def test_unperturbed_ring_is_independent_of_theta():
    exact = _load_case_module("exact")
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    radius = _mid_ring_radius(exact)
    ring = exact.polar_density(radius, theta, 0.0, eps=0.0)
    np.testing.assert_allclose(ring, exact.N0, rtol=0.0, atol=0.0)
    assert float(np.max(ring) - np.min(ring)) == 0.0
    outside = exact.polar_density(float(exact.R2) + 0.05, theta, 1.0, eps=0.0)
    inside = exact.polar_density(0.5 * float(exact.R1), theta, 1.0, eps=0.0)
    np.testing.assert_array_equal(outside, 0.0)
    np.testing.assert_array_equal(inside, 0.0)
    later = exact.polar_density(radius, theta, GROWTH_TIME, eps=0.0)
    np.testing.assert_array_equal(ring, later)


def test_mode_m2_angular_fft_peaks_at_bin_2():
    exact = _load_case_module("exact")
    assert int(exact.M) == 2
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    radius = _mid_ring_radius(exact)
    density = exact.polar_density(radius, theta, GROWTH_TIME)
    perturbation = np.asarray(density, dtype=np.float64) - float(exact.ring_density(radius))
    spectrum = np.abs(np.fft.rfft(perturbation))
    peak_bin = int(np.argmax(spectrum[1:])) + 1
    assert peak_bin == 2
    assert spectrum[2] > spectrum[1]
    assert spectrum[2] > spectrum[3]


def test_amplitude_is_eps_exp_gamma_t():
    exact = _load_case_module("exact")
    np.testing.assert_allclose(exact.GROWTH_RATE, 0.1, rtol=0.0, atol=0.0)
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    radius = _mid_ring_radius(exact)
    for time in (0.0, GROWTH_TIME, 2.5):
        expected = float(exact.EPS) * math.exp(float(exact.GROWTH_RATE) * float(time))
        np.testing.assert_allclose(exact.amplitude(time), expected, rtol=0.0, atol=1.0e-14)
        density = exact.polar_density(radius, theta, time)
        perturbation = np.asarray(density, dtype=np.float64) - float(exact.ring_density(radius))
        np.testing.assert_allclose(
            perturbation,
            expected * np.cos(float(exact.M) * theta),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_write_cp11_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp11_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [2]
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_siblings_use_load_sibling_module():
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
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text
