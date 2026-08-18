"""CP-01 driver: in-memory exact vs exact, documented Poisson sign, campaign report."""
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
fields_1d = _exact.fields_1d
poisson_residual_1d = _exact.poisson_residual_1d

N_CELLS = 32
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
    "amr.*": "AMR not run for CP-01 in-memory path",
    "coupling.phase_error": "phase not measured on the in-memory CP-01 path",
    "coupling.energy_drift": "energy drift not measured on the in-memory CP-01 path",
    "parallel_invariance.*": "parallel invariance not run for CP-01",
    "performance.one_node": "performance not measured for CP-01 in-memory path",
    "performance.two_node": "performance not measured for CP-01 in-memory path",
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
    width = 1.0 / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    volumes = np.full(n_cells, width, dtype=np.float64)
    return centers, volumes


def _summary() -> dict:
    centers, volumes = _cell_grid()
    fields = fields_1d(centers, 0.0)
    density = np.asarray(fields["n_e"], dtype=np.float64)
    potential = np.asarray(fields["phi"], dtype=np.float64)
    electric = np.asarray(fields["E"], dtype=np.float64)
    density_errors = reference_errors(density, density, volumes)
    potential_errors = reference_errors(potential, potential, volumes)
    field_errors = reference_errors(electric, electric, volumes)
    residual = poisson_residual_1d(centers, 0.0)
    residual_errors = reference_errors(
        residual, np.zeros_like(residual), volumes
    )
    charge = _exact.N_I - density
    predicted = (_exact.EPS0 * _exact.K**2 / _exact.E_CHARGE) * potential
    sign_ok = bool(np.max(np.abs(charge - predicted)) <= 1.0e-14)
    if density_errors.linf != 0.0 or potential_errors.linf != 0.0:
        raise ValueError("exact-vs-exact L∞ must be 0")
    if residual_errors.l2 != 0.0 or not sign_ok:
        raise ValueError("documented Poisson residual must be 0 (no sign flip)")
    if not (
        math.isfinite(density_errors.l1)
        and math.isfinite(density_errors.l2)
        and math.isfinite(density_errors.linf)
        and math.isfinite(residual_errors.l2)
    ):
        raise ValueError("non-finite CP-01 diagnostics")
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
            "residual_l2": float(residual_errors.l2),
        },
        "coupling": {
            "phase_error": None,
            "sign_ok": sign_ok,
            "energy_drift": None,
        },
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp01_report(output_dir) -> dict:
    """Compare exact vs exact and write the four Task 20 artifacts.

    Poisson residual uses ``-eps0 phi_xx - e (n_i - n_e)``. The sign is not
    flipped to force a zero residual.
    """
    return write_verification_report(_summary(), output_dir)
