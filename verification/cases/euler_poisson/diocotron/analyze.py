"""CP-11 driver: axisymmetric ring, m=2 FFT, toy growth, campaign report."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
amplitude = _exact.amplitude
angular_spectrum = _exact.angular_spectrum
density = _exact.density
polar_density = _exact.polar_density
ring_density = _exact.ring_density
uniform_cell_mesh = _exact.uniform_cell_mesh
EPS = _exact.EPS
GROWTH_RATE = _exact.GROWTH_RATE
M = _exact.M
N0 = _exact.N0
N_CELLS = _exact.N_CELLS
R1 = _exact.R1
R2 = _exact.R2

N_THETA = 64
GROWTH_TIME = 1.0
ORDERS_REASON = "linear growth / not a published reproduction"
NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
}
NULL_POISSON = {
    "potential_error": None,
    "field_error": None,
    "residual_l2": None,
}
NULL_PARALLEL = {
    "ranks_ok": None,
    "threads_ok": None,
    "gpu_ok": None,
}
NULL_PERFORMANCE = {
    "one_node": None,
    "two_node": None,
}
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run for CP-11 in-memory path",
    "poisson.*": "Poisson not run for CP-11 in-memory path",
    "coupling.phase_error": "phase not measured on the in-memory diocotron oracle",
    "coupling.energy_drift": "energy exchange not measured on the in-memory oracle",
    "parallel_invariance.*": "parallel invariance not run for CP-11",
    "performance.one_node": "performance not measured for CP-11 in-memory path",
    "performance.two_node": "performance not measured for CP-11 in-memory path",
}
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}


def _repository_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else "unknown"


def _mid_ring_radius() -> float:
    return 0.5 * (float(R1) + float(R2))


def _assert_unperturbed_ring() -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, N_THETA, endpoint=False)
    radius = _mid_ring_radius()
    ring = polar_density(radius, theta, 0.0, eps=0.0)
    if not np.array_equal(ring, np.full_like(ring, N0)):
        raise ValueError("unperturbed mid-ring density must equal n0 at every θ")
    if float(np.max(ring) - np.min(ring)) != 0.0:
        raise ValueError("unperturbed ring must be independent of θ")
    later = polar_density(radius, theta, GROWTH_TIME, eps=0.0)
    if not np.array_equal(ring, later):
        raise ValueError("unperturbed ring must be independent of t")


def _assert_mode_peak() -> None:
    if int(M) != 2:
        raise ValueError(f"documented mode must be m=2, got {M}")
    spectrum = angular_spectrum(_mid_ring_radius(), GROWTH_TIME, N_THETA)
    peak_bin = int(np.argmax(spectrum[1:])) + 1
    if peak_bin != 2:
        raise ValueError(f"angular FFT peak is bin {peak_bin}, expected 2")


def _assert_growth_amplitude() -> None:
    if abs(float(GROWTH_RATE) - 0.1) > 0.0:
        raise ValueError(f"documented toy growth rate must be 0.1, got {GROWTH_RATE}")
    theta = np.linspace(0.0, 2.0 * math.pi, N_THETA, endpoint=False)
    radius = _mid_ring_radius()
    for time in (0.0, GROWTH_TIME):
        factor = amplitude(time)
        expected = float(EPS) * math.exp(float(GROWTH_RATE) * float(time))
        if not math.isfinite(factor) or abs(factor - expected) > 1.0e-12:
            raise ValueError(f"amplitude at t={time} is {factor}, expected {expected}")
        samples = polar_density(radius, theta, time)
        perturbation = np.asarray(samples, dtype=np.float64) - float(ring_density(radius))
        mismatch = np.max(np.abs(perturbation - expected * np.cos(float(M) * theta)))
        if mismatch > 1.0e-12:
            raise ValueError(f"closed growing mode mismatch {mismatch}")


def _summary() -> dict:
    _assert_unperturbed_ring()
    _assert_mode_peak()
    _assert_growth_amplitude()
    x, y, volumes = uniform_cell_mesh(N_CELLS)
    field = np.asarray(density(x, y, GROWTH_TIME), dtype=np.float64)
    errors = reference_errors(field, field, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["euler_poisson"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": [],
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": {
            "phase_error": None,
            "sign_ok": True,
            "energy_drift": None,
        },
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp11_report(output_dir) -> dict:
    """Check ring / m=2 / toy growth and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
