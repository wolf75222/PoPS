"""PF-08 driver: closed reflux residual, open control, timed observation, report."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

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
ORDERS_REASON = "reflux/AMR sync stand-in, time is an observation"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.order_retained": "conservation residual contract; no AMR order series",
    "amr.interface_error": "interface-band error is AM-01; PF-08 records the reflux residual",
    "amr.bulk_error": "bulk vs uniform-fine comparison is later native work",
    "poisson.*": "Poisson not run in PF-08 in-memory path",
    "coupling.*": "coupling not run in PF-08 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-08",
    "performance.one_node": "elapsed wall time is an observation, not a timed PF run",
    "performance.two_node": "two-node performance not measured in PF-08",
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


def _assert_reflux_contract() -> dict:
    if _exact.RATIO != 2:
        raise ValueError("PF-08 documents a ratio-2 coarse-fine face")
    observed = _run.timed_sync(reflux=True)
    opened = _run.timed_sync(reflux=False)
    closed = float(observed["residual"])
    open_residual = float(opened["residual"])
    if closed != 0.0:
        raise ValueError("closed statement with reflux must have residual 0")
    if open_residual == 0.0:
        raise ValueError("open statement without reflux must have a nonzero residual")
    terms = _exact.closed_balance_terms()
    if float(terms["reflux"]) == 0.0:
        raise ValueError("manufactured reflux correction must be nonzero")
    if float(_exact.open_balance_terms()["reflux"]) != 0.0:
        raise ValueError("open negative control must omit reflux")
    return observed


def _summary() -> dict:
    observed = _assert_reflux_contract()
    elapsed_s = float(observed["elapsed_s"])
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["performance"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": [],
        "amr": {
            "order_retained": None,
            "invariants_ok": True,
            "interface_error": None,
            "bulk_error": None,
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": {
            "one_node": {
                "cells_per_second": None,
                "notes": f"elapsed_s={elapsed_s:.6e} observation only",
            },
            "two_node": None,
        },
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_pf08_report(output_dir) -> dict:
    """Check the reflux residual, record elapsed time, write four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
