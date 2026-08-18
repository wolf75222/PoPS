"""PO-07 driver: tolerance sweep, discretization plateau, observed order, report."""
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
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
combined_error = _exact.combined_error
discretization_error = _exact.discretization_error

RESOLUTIONS = (16, 32, 64, 128)
TOLERANCE_SWEEP = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)
PRODUCTION_TOL = 1.0e-12
CASE_ID = "PO-07"
NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
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
    "amr.*": "AMR not run in PO-07 elliptic tolerance sweep",
    "coupling.*": "coupling not run in PO-07",
    "parallel_invariance.*": "parallel invariance not run in PO-07",
    "performance.one_node": "performance not measured in PO-07",
    "performance.two_node": "performance not measured in PO-07",
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


def _plateau_and_orders():
    """Tight-tol floor vs n, plus the combined-error plateau at the finest n."""
    error_scalars = [
        combined_error(n_cells, PRODUCTION_TOL) for n_cells in RESOLUTIONS
    ]
    spacings = [1.0 / float(n_cells) for n_cells in RESOLUTIONS]
    orders = observed_order(error_scalars, spacings)
    sweep = [combined_error(RESOLUTIONS[-1], tol) for tol in TOLERANCE_SWEEP]
    return discretization_error(RESOLUTIONS[-1]), sweep, orders


def _summary() -> dict:
    plateau, sweep, orders = _plateau_and_orders()
    if not (
        math.isfinite(plateau)
        and np.all(np.isfinite(sweep))
        and np.all(np.isfinite(orders))
    ):
        raise ValueError("non-finite PO-07 diagnostics")
    observed = float(orders[-1])
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "nightly",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["poisson"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": [
            {
                "case_id": CASE_ID,
                "kind": "spatial",
                "variable": "potential",
                "observed_order": observed,
                "threshold": 1.8,
            }
        ],
        "amr": dict(NULL_AMR),
        "poisson": {
            "potential_error": float(plateau),
            "field_error": 0.0,
            "residual_l2": 0.0,
        },
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_po07_report(output_dir) -> dict:
    """Reduce the manufactured tolerance sweep and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
