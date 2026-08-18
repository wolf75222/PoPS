"""PF-12 driver: 15-wide saxpy identity, bytes/cell, campaign report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
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
ORDERS_REASON = "kernel microbench stand-in, not a timed PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in PF-12 in-memory path",
    "poisson.*": "Poisson not run in PF-12 in-memory path",
    "coupling.*": "coupling not run in PF-12 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-12",
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


def _assert_wide_state_identities() -> None:
    if int(_exact.N_COMPONENTS) != 15:
        raise ValueError("PF-12 HyQMOM15 width must be 15 components")
    if len(_exact.COMPONENT_NAMES) != 15:
        raise ValueError("PF-12 component names must have length 15")
    if int(_exact.BYTES_PER_CELL) != 15 * 8:
        raise ValueError("PF-12 bytes/cell must be 15 * 8")
    if int(_exact.bytes_per_cell()) != 120:
        raise ValueError("PF-12 bytes/cell must be 120")
    comparison = _run.width_comparison()
    if comparison["hyqmom15_bytes_per_cell"] != 120:
        raise ValueError("HyQMOM15 bytes/cell must be 120")
    if comparison["euler_components"] != 5:
        raise ValueError("Euler comparison width must be 5 components")
    if comparison["euler_bytes_per_cell"] != 40:
        raise ValueError("Euler bytes/cell must be 40")
    if comparison["ratio"] != 3.0:
        raise ValueError("HyQMOM15 / Euler bytes/cell ratio must be 3")
    state = _run.wide_state()["state"]
    if tuple(state.shape) != (int(_exact.N_CELLS), 15):
        raise ValueError("wide state must have shape (n, 15)")
    saxpy = _run.saxpy_fields()
    saxpy_err = reference_errors(saxpy["a"], 2.0 * saxpy["b"], saxpy["volumes"])
    if saxpy_err.l1 != 0.0 or saxpy_err.l2 != 0.0 or saxpy_err.linf != 0.0:
        raise ValueError("a = 2*b must be exact on the (n, 15) state")


def _summary() -> dict:
    _assert_wide_state_identities()
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


def write_pf12_report(output_dir) -> dict:
    """Check 15-wide saxpy and bytes/cell, then write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
