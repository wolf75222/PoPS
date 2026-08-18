"""AM-12 driver: reflected leftover E_cf invariance and campaign report."""
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
    "orders": "reflection-symmetry contract; no spatial-order campaign in this increment",
    "amr.order_retained": "no AMR order series in AM-12 in-memory path",
    "poisson.*": "Poisson not run in AM-12 in-memory path",
    "coupling.*": "coupling residuals not measured in AM-12 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-12",
    "performance.one_node": "performance not measured in AM-12",
    "performance.two_node": "performance not measured in AM-12",
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


def _assert_reflection_invariance() -> tuple[float, float]:
    original = _run.patch_sample()
    rebuilt = _run.patch_sample(reflected=True)
    reflected = _run.reflect_leftover_and_oracle(original)
    if not np.array_equal(rebuilt["interface"], original["interface"][::-1]):
        raise ValueError("reflected interface mask must equal the reversed original mask")
    if not np.isclose(reflected["e_cf"], original["e_cf"], rtol=0.0, atol=0.0):
        raise ValueError("E_cf must be unchanged under reflection of leftover+oracle")
    e_cf = float(original["e_cf"])
    e_bulk = float(original["e_bulk"])
    if not (np.isfinite(e_cf) and np.isfinite(e_bulk)):
        raise ValueError("non-finite AM-12 interface or bulk error")
    return e_cf, e_bulk


def _summary() -> dict:
    e_cf, e_bulk = _assert_reflection_invariance()
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


def write_am12_report(output_dir) -> dict:
    """Check leftover reflection invariance and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
