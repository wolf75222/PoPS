"""PF-09 driver: uniform CV=0, hotspot max>mean, campaign report writer."""
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
ORDERS_REASON = "load-balance stand-in, not a timed PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in PF-09 in-memory path",
    "poisson.*": "Poisson not run in PF-09 in-memory path",
    "coupling.*": "coupling not run in PF-09 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-09",
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


def _assert_load_identities() -> None:
    if int(_exact.N_CELLS) != 64:
        raise ValueError("PF-09 cell count must be 64")
    if int(_exact.N_RANKS) != 4:
        raise ValueError("PF-09 rank count must be 4")
    if int(_exact.HOTSPOT_RANK0) != 40:
        raise ValueError("PF-09 hotspot must place 40 cells on rank 0")
    uniform = _run.uniform_stats()
    if uniform["cv"] != 0.0:
        raise ValueError("uniform load CV must be 0")
    if uniform["max"] != uniform["mean"]:
        raise ValueError("uniform load max must equal mean")
    hotspot = _run.hotspot_stats()
    if not (hotspot["max"] > hotspot["mean"]):
        raise ValueError("hotspot max must exceed mean")


def _summary() -> dict:
    _assert_load_identities()
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


def write_pf09_report(output_dir) -> dict:
    """Check uniform CV=0 and hotspot max>mean, then write the four artifacts."""
    return write_verification_report(_summary(), output_dir)
