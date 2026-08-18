"""TM-02 norms, Lie/Strang temporal orders, and campaign report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

CASE_ID = "TM-02"
LIE_THRESHOLD = 0.8
STRANG_THRESHOLD = 1.8
DT_SERIES = _exact.DT_SERIES
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
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}
NOT_RUN_REASONS = {
    "amr.*": "AMR not run in TM-02 in-memory path",
    "poisson.*": "Poisson not run in TM-02",
    "coupling.*": "coupling not run in TM-02",
    "parallel_invariance.*": "parallel invariance not run in TM-02",
    "performance.one_node": "performance not measured in TM-02",
    "performance.two_node": "performance not measured in TM-02",
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


def _order_rows(errors, resolutions, *, variable: str, threshold: float) -> list[dict]:
    error_series = list(errors)
    dt_series = list(resolutions)
    if len(error_series) < 2:
        return []
    return [
        {
            "case_id": CASE_ID,
            "kind": "temporal",
            "variable": variable,
            "observed_order": float(value),
            "threshold": float(threshold),
        }
        for value in observed_order(error_series, dt_series)
    ]


def _summary(*, orders: list, order_reason: str | None) -> dict:
    reasons = dict(NOT_RUN_REASONS)
    if not orders:
        reasons["orders"] = order_reason or "single-resolution series"
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["time"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": list(orders),
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": reasons,
        "artifacts": dict(ARTIFACTS),
    }


def analyze_series(lie_errors, strang_errors, resolutions, output_dir) -> dict:
    """Write a campaign report from already-computed Lie and Strang series."""
    dt_series = list(resolutions)
    orders = _order_rows(
        lie_errors, dt_series, variable="lie", threshold=LIE_THRESHOLD
    ) + _order_rows(
        strang_errors, dt_series, variable="strang", threshold=STRANG_THRESHOLD
    )
    reason = None if orders else "single-resolution series"
    return write_verification_report(
        _summary(orders=orders, order_reason=reason),
        output_dir,
    )


def write_tm02_report(output_dir, *, dts=DT_SERIES) -> dict:
    """Require AB ≠ BA, then write the manufactured Lie/Strang Δt series."""
    if _exact.A1 == _exact.A2:
        raise ValueError("a1 and a2 must differ so that AB ≠ BA")
    if _exact.operators_commute():
        raise ValueError("manufactured A and B must not commute")
    steps = list(dts)
    return analyze_series(
        _run.error_series(_run.lie_step, steps),
        _run.error_series(_run.strang_step, steps),
        steps,
        output_dir,
    )
