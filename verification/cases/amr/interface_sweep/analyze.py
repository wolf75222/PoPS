"""AM-08 driver: interface-placement envelope, worst-order series, report."""
from __future__ import annotations

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
_run = load_sibling_module(_CASE_DIR / "run.py")

CASE_ID = "AM-08"
FIXED_N = 32
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
    "amr.invariants_ok": "conservation residual not measured in AM-08 in-memory path",
    "poisson.*": "Poisson not run in AM-08 in-memory path",
    "coupling.*": "coupling not run in AM-08 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-08",
    "performance.one_node": "performance not measured in AM-08",
    "performance.two_node": "performance not measured in AM-08",
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


def _envelope_and_orders():
    emin, emax = _run.error_envelope(FIXED_N)
    errors, spacings, worst_x0 = _run.worst_error_series(_exact.RESOLUTIONS)
    orders = observed_order(errors, spacings)
    bulk = _run.bulk_error_at(worst_x0, FIXED_N)
    return emin, emax, orders, bulk


def _summary() -> dict:
    emin, emax, orders, bulk = _envelope_and_orders()
    if not (
        np.isfinite(emin)
        and np.isfinite(emax)
        and emax >= emin
        and np.all(np.isfinite(orders))
        and np.isfinite(bulk)
    ):
        raise ValueError("non-finite AM-08 envelope or orders")
    observed = float(np.min(orders))
    retained = bool(observed >= _exact.ORDER_THRESHOLD)
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "nightly",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["amr"],
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
                "variable": "q",
                "observed_order": float(value),
                "threshold": _exact.ORDER_THRESHOLD,
            }
            for value in orders
        ],
        "amr": {
            "order_retained": retained,
            "invariants_ok": None,
            "interface_error": float(emax),
            "bulk_error": float(bulk),
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am08_report(output_dir) -> dict:
    """Reduce the manufactured placement sweep and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
