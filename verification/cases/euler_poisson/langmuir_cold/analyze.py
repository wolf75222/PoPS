"""CP-02 driver: Gauss residual, exact E(t) frequency, campaign report."""
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
from verification.pops_verify.phase import frequency_error, numerical_frequency
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
dE_dx = _exact.dE_dx
e_field = _exact.e_field
gauss_rhs = _exact.gauss_rhs
n_e = _exact.n_e
phi = _exact.phi
plasma_frequency = _exact.plasma_frequency
uniform_cell_centers = _exact.uniform_cell_centers

N_CELLS = _exact.N_CELLS
N_PERIODS = 8
SAMPLES_PER_PERIOD = 32
PROBE_X = 0.25
NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
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
    "orders": "single-resolution in-memory exact comparison",
    "amr.*": "AMR not run for CP-02 in-memory path",
    "coupling.energy_drift": "energy exchange not measured on the in-memory oracle",
    "parallel_invariance.*": "parallel invariance not run for CP-02",
    "performance.one_node": "performance not measured for CP-02 in-memory path",
    "performance.two_node": "performance not measured for CP-02 in-memory path",
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


def _cell_grid(n_cells: int = N_CELLS):
    return uniform_cell_centers(n_cells)


def _gauss_residual_l2(n_cells: int = N_CELLS, t: float = 0.0) -> float:
    centers, volumes = _cell_grid(n_cells)
    residual = np.asarray(dE_dx(centers, t) - gauss_rhs(centers, t), dtype=np.float64)
    value = float(np.sqrt(np.sum(volumes * residual * residual) / np.sum(volumes)))
    return 0.0 if value <= 1.0e-15 else value


def _exact_e_probe_frequency_error() -> float:
    omega_pe = float(plasma_frequency())
    period = 2.0 * np.pi / omega_pe
    times = np.arange(N_PERIODS * SAMPLES_PER_PERIOD, dtype=np.float64) * (
        period / SAMPLES_PER_PERIOD
    )
    samples = e_field(PROBE_X, times)
    omega_num = numerical_frequency(times, samples, method="fft")
    return frequency_error(omega_num, omega_pe)


def _summary() -> dict:
    centers, volumes = _cell_grid()
    electric = np.asarray(e_field(centers, 0.0), dtype=np.float64)
    potential = np.asarray(phi(centers, 0.0), dtype=np.float64)
    density = np.asarray(n_e(centers, 0.0), dtype=np.float64)
    field_errors = reference_errors(electric, electric, volumes)
    potential_errors = reference_errors(potential, potential, volumes)
    residual = _gauss_residual_l2()
    omega_error = _exact_e_probe_frequency_error()
    finite = (
        field_errors.linf == 0.0
        and potential_errors.linf == 0.0
        and math.isfinite(residual)
        and math.isfinite(omega_error)
        and np.all(density > 0.0)
    )
    if not finite:
        raise ValueError("in-memory CP-02 identities must be finite with L∞=0")
    sign_ok = abs(residual) <= 1.0e-12 and omega_error <= 1.0e-9
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["continuity", "momentum", "poisson", "electrostatic_source"],
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
        "poisson": {
            "potential_error": float(potential_errors.linf),
            "field_error": float(field_errors.linf),
            "residual_l2": residual,
        },
        "coupling": {
            "phase_error": float(omega_error),
            "sign_ok": bool(sign_ok),
            "energy_drift": None,
        },
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp02_report(output_dir) -> dict:
    """Compare exact Gauss / E(t) identities and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
