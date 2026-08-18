"""AM-05 driver: regrid-frequency leftover contract and campaign report."""
from __future__ import annotations

import math
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
NULL_PERFORMANCE = {
    "one_node": None,
    "two_node": None,
}
NOT_APPLICABLE = {
    "orders": "exact-field leftover is identically 0 vs 1/k; no fitted order campaign",
    "amr.order_retained": "regrid cadence leftover is a contract, not a fitted AMR order series",
    "amr.interface_error": "static coarse-fine band is AM-01; AM-05 records rebuild cadence",
    "amr.bulk_error": "bulk vs uniform-fine comparison is later native work",
    "poisson.*": "Poisson not run in AM-05 in-memory path",
    "coupling.*": "coupling not run in AM-05 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-05",
    "performance.one_node": "performance not measured in AM-05",
    "performance.two_node": "performance not measured in AM-05",
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


def _assert_regrid_frequency_contract() -> None:
    if tuple(_exact.K_VALUES) != (1, 2, 4, 8, 16):
        raise ValueError("documented regrid frequencies must be (1, 2, 4, 8, 16)")
    if _exact.N_STEPS != 16:
        raise ValueError("documented step count must be 16")
    for k in _exact.K_VALUES:
        expected = _exact.N_STEPS // int(k)
        if _exact.expected_rebuilds(k) != expected:
            raise ValueError(f"expected rebuilds for k={k} must be N/k")
        if _run.count_rebuilds(k) != expected:
            raise ValueError(f"rebuild count for k={k} must be N/k")
        leftover = _run.interval_leftover(k)
        if not math.isclose(leftover, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("leftover |k_regrid - k_requested| must be 0")
    _, leftovers, slope = _run.field_leftover_vs_inv_k()
    if any(float(value) != 0.0 for value in leftovers):
        raise ValueError("exact-field leftover must be 0 for every k")
    if not math.isclose(slope, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("exact field must have no linear leftover vs 1/k")


def _summary() -> dict:
    _assert_regrid_frequency_contract()
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
            "invariants_ok": True,
            "interface_error": None,
            "bulk_error": None,
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am05_report(output_dir) -> dict:
    """Check the in-memory regrid-frequency contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
