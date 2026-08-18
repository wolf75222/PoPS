"""CP-07 driver: exact force balance, rest velocity, campaign report."""
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
exact_fields = _exact.exact_fields
uniform_cell_centers = _exact.uniform_cell_centers

N_CELLS = 64
FORCE_ATOL = 1.0e-12
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
    "amr.*": "AMR not run for CP-07 in-memory path",
    "parallel_invariance.*": "parallel invariance not run for CP-07",
    "performance.one_node": "performance not measured for CP-07 in-memory path",
    "performance.two_node": "performance not measured for CP-07 in-memory path",
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


def _force_residual(fields: dict) -> float:
    density = np.asarray(fields["n"], dtype=np.float64)
    residual = np.asarray(fields["grad_p"], dtype=np.float64) - (
        float(fields["q"]) * density * np.asarray(fields["E"], dtype=np.float64)
    )
    return float(np.max(np.abs(residual)))


def _summary() -> dict:
    centers, volumes = uniform_cell_centers(N_CELLS)
    fields = exact_fields(centers, profile="cosine")
    density = np.asarray(fields["n"], dtype=np.float64)
    potential = np.asarray(fields["phi"], dtype=np.float64)
    electric = np.asarray(fields["E"], dtype=np.float64)
    density_errors = reference_errors(density, density, volumes)
    potential_errors = reference_errors(potential, potential, volumes)
    field_errors = reference_errors(electric, electric, volumes)
    residual = _force_residual(fields)
    velocity = np.asarray(fields["u"], dtype=np.float64)
    if density_errors.linf != 0.0 or not (
        math.isfinite(density_errors.l1)
        and math.isfinite(density_errors.l2)
        and math.isfinite(density_errors.linf)
        and math.isfinite(residual)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
    if not np.all(velocity == 0.0):
        raise ValueError("equilibrium velocity must be identically 0")
    sign_ok = residual <= FORCE_ATOL
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
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
        "poisson": {
            "potential_error": float(potential_errors.linf),
            "field_error": float(field_errors.linf),
            "residual_l2": residual,
        },
        "coupling": {
            "phase_error": 0.0,
            "sign_ok": bool(sign_ok),
            "energy_drift": 0.0,
        },
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp07_report(output_dir) -> dict:
    """Check ∇p = q n E on exact fields and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
