"""PF-07 driver: pulse-tag coverage, patch-count contract, campaign report."""
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
from verification.pops_verify.report import write_verification_report

_TR02_DIR = _CASE_DIR.parents[1] / "transport" / "gaussian_pulse"
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
_tr02 = load_sibling_module(_TR02_DIR / "exact.py")

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
ORDERS_REASON = "kernel microbench stand-in, not a timed PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in PF-07 in-memory path",
    "poisson.*": "Poisson not run in PF-07 in-memory path",
    "coupling.*": "coupling not run in PF-07 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-07",
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


def _assert_cluster_contract() -> None:
    if int(_exact.MIN_PATCH_WIDTH) != 4:
        raise ValueError("PF-07 min patch width must be 4")
    if float(_exact.TAG_THRESHOLD) <= 0.0:
        raise ValueError("documented tag threshold must be positive")
    result = _run.cluster_tagged_pulse()
    field = np.asarray(result["field"], dtype=np.float64)
    centers = np.asarray(result["centers"], dtype=np.float64)
    expected = _tr02.exact_gaussian(centers, 0.0)
    if not np.allclose(field, expected):
        raise ValueError("sampled field must be the TR-02 exact Gaussian")
    tags = np.asarray(result["tags"], dtype=bool)
    raw = int(result["raw_tag_count"])
    patches = int(result["patch_count"])
    if raw <= 0 or not np.any(tags):
        raise ValueError("documented pulse tag set must be non-empty")
    if raw != int(np.count_nonzero(tags)):
        raise ValueError("raw_tag_count must match the tagged cell count")
    if patches > raw:
        raise ValueError("clustered patch count must be <= raw tag count")
    covered = _exact.coverage_mask(result["patches"], tags.size)
    if not np.array_equal(covered, np.asarray(result["covered"], dtype=bool)):
        raise ValueError("reported coverage must match the clustered patches")
    if not np.all(covered[tags]):
        raise ValueError("clustered patches must cover every tagged cell")
    for _start, width in result["patches"]:
        if int(width) < int(_exact.MIN_PATCH_WIDTH):
            raise ValueError("every clustered patch must have width >= 4")


def _summary() -> dict:
    _assert_cluster_contract()
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


def write_pf07_report(output_dir) -> dict:
    """Check tag coverage and patch-count, then write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
