"""PF-11 driver: rebuild cadence, warmup exclusion, campaign report writer."""
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
ORDERS_REASON = "dynamic AMR e2e stand-in, not a timed PF run"
THROUGHPUT_NOTES = "leaf-cell throughput from fake time; not a timed PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR order/invariants not measured in PF-11 in-memory path",
    "poisson.*": "Poisson not run in PF-11 in-memory path",
    "coupling.*": "coupling not run in PF-11 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in PF-11",
    "performance.two_node": "two-node throughput not measured in PF-11 stand-in",
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


def _assert_rebuild_contract() -> dict:
    if int(_exact.N_WARMUP) != 2:
        raise ValueError("PF-11 warmup must be 2 steps")
    if int(_exact.N_STEPS) != 50:
        raise ValueError("PF-11 measured step count must be 50")
    if int(_exact.REGRID_EVERY) != 8:
        raise ValueError("PF-11 regrid interval must be 8")
    expected = int(_exact.N_STEPS) // int(_exact.REGRID_EVERY)
    if _exact.expected_rebuilds() != expected:
        raise ValueError("expected rebuilds must be n_steps / regrid_every")
    sample = _run.advance_fake_amr()
    if int(sample["rebuilds"]) != expected:
        raise ValueError("measured rebuild count must be n_steps / regrid_every")
    if int(sample["rebuilds"]) != (
        int(sample["rebuilds_including_warmup"]) - int(sample["warmup_rebuilds"])
    ):
        raise ValueError("warmup rebuilds must be excluded from the measured count")
    if any(int(step) < int(_exact.N_WARMUP) for step in sample["rebuild_steps"]):
        raise ValueError("measured rebuilds must not include warmup steps")
    if float(sample["throughput"]) <= 0.0:
        raise ValueError("leaf-cell throughput observation must be positive")
    return sample


def _summary() -> dict:
    sample = _assert_rebuild_contract()
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
                "cells_per_second": float(sample["throughput"]),
                "notes": THROUGHPUT_NOTES,
            },
            "two_node": None,
        },
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_pf11_report(output_dir) -> dict:
    """Check rebuild cadence and warmup exclusion, then write four artifacts."""
    return write_verification_report(_summary(), output_dir)
