"""GE-03 driver: radial independence, ω = c k, campaign report."""
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

N_CELLS = int(_exact.N_CELLS)
ANGULAR_STD_ATOL = 1.0e-12
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
    "orders": "single-resolution in-memory radial-acoustic contract; no spatial-order campaign",
    "amr.*": "AMR not run for GE-03 in-memory path",
    "poisson.*": "Poisson not run for GE-03",
    "coupling.*": "coupling not run for GE-03",
    "parallel_invariance.*": "parallel invariance not run for GE-03",
    "performance.one_node": "performance not measured for GE-03 in-memory path",
    "performance.two_node": "performance not measured for GE-03 in-memory path",
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
    count = int(n_cells)
    lower_x, lower_y = (float(value) for value in _exact.DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in _exact.DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / float(count)
    width_y = (upper_y - lower_y) / float(count)
    x_centers = lower_x + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    volumes = np.full((count, count), width_x * width_y, dtype=np.float64)
    return x, y, volumes


def _assert_radial_contract() -> None:
    if not math.isclose(_exact.C, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("documented sound speed must be c=1")
    if not math.isclose(_exact.omega(), _exact.C * _exact.K, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("ω must equal c k")
    spread = _exact.angular_std(_exact.RING_RADIUS, 0.0, _exact.N_THETA)
    if not math.isfinite(spread) or spread > ANGULAR_STD_ATOL:
        raise ValueError("φ at t=0 must depend only on r (angular std ~ 0)")
    x, y, volumes = _cell_grid()
    field = np.asarray(_exact.phi(x, y, 0.0), dtype=np.float64)
    errors = reference_errors(field, field, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact φ L∞ must be 0")


def _summary() -> dict:
    _assert_radial_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["geometry"],
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


def write_ge03_report(output_dir) -> dict:
    """Check the in-memory radial-acoustic contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
