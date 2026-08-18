"""PF-06 driver: seven-segment EP pipeline timings and campaign report writer."""
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
NULL_PERFORMANCE = {
    "one_node": None,
    "two_node": None,
}
ORDERS_REASON = "pipeline segment stand-in, not a timed PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in PF-06 in-memory path",
    "poisson.*": "Poisson not run in PF-06 in-memory path",
    "coupling.*": "coupling not run in PF-06 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-06",
    "performance.one_node": ORDERS_REASON,
    "performance.two_node": ORDERS_REASON,
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


def _assert_segment_partition() -> None:
    if tuple(_exact.PIPELINE_SEGMENTS) != (
        "halo",
        "hyperbolic",
        "charge",
        "poisson",
        "gradient",
        "source",
    ):
        raise ValueError("PF-06 pipeline must be halo/hyperbolic/charge/poisson/gradient/source")
    if len(_exact.SEGMENT_NAMES) != 7:
        raise ValueError("PF-06 must expose seven segments including total")
    result = _run.run_ep_step()
    timings = result["timings"]
    if list(result["segments"]) != list(_exact.SEGMENT_NAMES):
        raise ValueError("run_ep_step segments must match SEGMENT_NAMES")
    for name in _exact.PIPELINE_SEGMENTS:
        if float(timings[name]) <= 0.0:
            raise ValueError(f"segment {name!r} timing must be > 0")
    expected = _exact.pipeline_total(timings)
    if timings["total"] != expected:
        raise ValueError("total must equal the sum of pipeline segment timings")


def _summary() -> dict:
    _assert_segment_partition()
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
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_pf06_report(output_dir) -> dict:
    """Check the seven-segment partition, then write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
