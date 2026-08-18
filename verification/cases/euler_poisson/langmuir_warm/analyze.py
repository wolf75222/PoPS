"""CP-03 driver: dispersion identity, exact vs exact, campaign report."""
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
angular_frequency = _exact.angular_frequency
dispersion_residual = _exact.dispersion_residual
exact_fields = _exact.exact_fields
uniform_cell_centers = _exact.uniform_cell_centers
wavenumber = _exact.wavenumber
WAVE_NUMBERS_OVER_2PI = _exact.WAVE_NUMBERS_OVER_2PI

N_CELLS = 32
CASE_ID = "CP-03"
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
    "orders": "dispersion identity, not a resolution series",
    "amr.*": "AMR not run for CP-03 in-memory path",
    "poisson.*": "Poisson not run for CP-03 in-memory path",
    "coupling.*": "native coupling not run for CP-03 in-memory path",
    "parallel_invariance.*": "parallel invariance not run for CP-03",
    "performance.one_node": "performance not measured for CP-03 in-memory path",
    "performance.two_node": "performance not measured for CP-03 in-memory path",
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


def _summary() -> dict:
    _assert_dispersion_identity()
    centers, volumes = uniform_cell_centers(N_CELLS)
    fields = exact_fields(centers, 0.0, k=wavenumber(1))
    density = np.asarray(fields["n_e"], dtype=np.float64)
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
    omega = angular_frequency(wavenumber(1))
    if not math.isfinite(omega):
        raise ValueError("non-finite warm Langmuir frequency")
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


def write_cp03_report(output_dir) -> dict:
    """Check the dispersion identity and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
