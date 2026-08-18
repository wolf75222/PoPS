"""PF-04 driver: free-stream Rusanov identity, campaign report writer."""
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
ORDERS_REASON = "kernel stand-in, not a timed PF run"
PYTHON_LOOP_NOTES = "python loop observation; not a timed native PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in PF-04 in-memory path",
    "poisson.*": "Poisson not run in PF-04 in-memory path",
    "coupling.*": "coupling not run in PF-04 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-04",
    "performance.two_node": "two-node not measured; python loop is a one-node observation",
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


def _assert_free_stream_identity() -> dict:
    result = _run.one_rusanov_step()
    residuals = _exact.free_stream_residuals(result["rho"], result["u"], result["p"])
    limit = float(_exact.FREE_STREAM_ATOL)
    for key in ("rho", "u", "p"):
        if residuals[key] > limit:
            raise ValueError(
                f"uniform {key} must stay within {limit}, got {residuals[key]}"
            )
    return result


def _summary() -> dict:
    _assert_free_stream_identity()
    timed = _run.time_python_loop()
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
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": {
            "one_node": {
                "cells_per_second": timed["cells_per_second"],
                "notes": PYTHON_LOOP_NOTES,
            },
            "two_node": None,
        },
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_pf04_report(output_dir) -> dict:
    """Check free-stream identity, time the python loop, write four artifacts."""
    return write_verification_report(_summary(), output_dir)
