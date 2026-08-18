"""CP-06 driver: ω(k), eigenvector identity, time advance, campaign report."""
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
system_matrix = _exact.system_matrix
eigenvalue = _exact.eigenvalue
right_eigenvector = _exact.right_eigenvector
angular_frequency = _exact.angular_frequency
dispersion_residual = _exact.dispersion_residual
exact_state = _exact.exact_state
advance_fourier = _exact.advance_fourier
uniform_cell_centers = _exact.uniform_cell_centers
wavenumber = _exact.wavenumber
MODES = _exact.MODES
EPS = _exact.EPS
CANONICAL_K = _exact.CANONICAL_K
BACKGROUND = _exact.BACKGROUND
WAVE_NUMBERS_OVER_2PI = _exact.WAVE_NUMBERS_OVER_2PI

N_CELLS = 32
ADVANCE_TIME = 0.125
CASE_ID = "CP-06"
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
NULL_COUPLING = {
    "phase_error": None,
    "sign_ok": None,
    "energy_drift": None,
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
    "orders": "ion-acoustic eigenmode identity, not a resolution series",
    "amr.*": "AMR not run for CP-06 in-memory path",
    "poisson.*": "Poisson not run for CP-06 in-memory path",
    "coupling.*": "native coupling not run for CP-06 in-memory path",
    "parallel_invariance.*": "parallel invariance not run for CP-06",
    "performance.one_node": "performance not measured for CP-06 in-memory path",
    "performance.two_node": "performance not measured for CP-06 in-memory path",
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


def _assert_dispersion_identity() -> None:
    for cycles in WAVE_NUMBERS_OVER_2PI:
        residual = dispersion_residual(wavenumber(cycles))
        if not math.isfinite(residual) or abs(residual) > 1.0e-12:
            raise ValueError(f"dispersion residual at k/2π={cycles} is {residual}")


def _assert_eigenvector_identity() -> None:
    for cycles in WAVE_NUMBERS_OVER_2PI:
        k = wavenumber(cycles)
        matrix = system_matrix(k)
        omega = angular_frequency(k)
        plus = right_eigenvector("plus", k)
        plus_residual = matrix @ plus + 1.0j * omega * plus
        if not np.all(np.isfinite(plus_residual)) or np.max(np.abs(plus_residual)) > 1.0e-12:
            raise ValueError(f"M r + iω r residual at k/2π={cycles} is {plus_residual}")
        for mode in MODES:
            lam = eigenvalue(mode, k)
            vector = right_eigenvector(mode, k)
            residual = matrix @ vector - lam * vector
            if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1.0e-12:
                raise ValueError(f"eigenvector residual for mode {mode!r} at k/2π={cycles} is {residual}")


def _assert_time_advance() -> None:
    centers, volumes = uniform_cell_centers(N_CELLS)
    background = np.asarray(BACKGROUND, dtype=np.float64)
    omega = angular_frequency(CANONICAL_K)
    for mode in MODES:
        lam = eigenvalue(mode, CANONICAL_K)
        vector = right_eigenvector(mode, CANONICAL_K)
        evolved = exact_state(
            centers, ADVANCE_TIME, mode=mode, k=CANONICAL_K, eps=EPS
        )
        phase = np.exp(1.0j * CANONICAL_K * centers + lam * ADVANCE_TIME)
        expected = background[:, None] + float(EPS) * np.real(
            vector[:, None] * phase[None, :]
        )
        if np.max(np.abs(evolved - expected)) > 1.0e-12:
            raise ValueError(f"time-advance mismatch for mode {mode!r}")
        if mode == "plus":
            traveling = np.exp(1.0j * CANONICAL_K * centers - 1.0j * omega * ADVANCE_TIME)
            if np.max(np.abs(phase - traveling)) > 1.0e-12:
                raise ValueError("plus branch is not exp(ikx - iωt)")
        hat_t = advance_fourier(float(EPS) * vector, ADVANCE_TIME, k=CANONICAL_K)
        expected_hat = float(EPS) * vector * np.exp(lam * ADVANCE_TIME)
        if np.max(np.abs(hat_t - expected_hat)) > 1.0e-12:
            raise ValueError(f"Fourier time-advance mismatch for mode {mode!r}")
        density = np.asarray(evolved[0], dtype=np.float64)
        errors = reference_errors(density, density, volumes)
        if errors.linf != 0.0 or not (
            math.isfinite(errors.l1)
            and math.isfinite(errors.l2)
            and math.isfinite(errors.linf)
        ):
            raise ValueError("exact-vs-exact density L∞ must be 0")


def _summary() -> dict:
    _assert_dispersion_identity()
    _assert_eigenvector_identity()
    _assert_time_advance()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "nightly",
        "max_nodes": 2,
        "native_dimensions": [1],
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
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp06_report(output_dir) -> dict:
    """Check ω(k) / M r = -iω r / time advance and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
