"""AM-01 driver: static CF interface-band observation and campaign report."""
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
NULL_PERFORMANCE = {
    "one_node": None,
    "two_node": None,
}
NOT_APPLICABLE = {
    "orders": "static CF interface-band contract; no spatial-order campaign in this increment",
    "amr.order_retained": "no AMR order series in AM-01 in-memory path",
    "amr.invariants_ok": "invariants not measured in AM-01 in-memory path",
    "poisson.*": "Poisson not run in AM-01 in-memory path",
    "coupling.*": "coupling residuals not measured in AM-01 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-01",
    "performance.one_node": "performance not measured in AM-01",
    "performance.two_node": "performance not measured in AM-01",
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


def _summary() -> dict:
    sample = _run.interface_bulk_errors()
    e_cf = float(sample["e_cf"])
    e_bulk = float(sample["e_bulk"])
    if sample["ratio"] is None:
        raise ValueError("AM-01 leftover E_cf/E_bulk must be a recorded observation")
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
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
        "orders": [],
        "amr": {
            "order_retained": None,
            "invariants_ok": None,
            "interface_error": e_cf,
            "bulk_error": e_bulk,
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am01_report(output_dir) -> dict:
    """Record the leftover interface-band observation and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
