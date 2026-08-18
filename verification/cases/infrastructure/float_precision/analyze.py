"""IF-09 driver: float32 vs float64 plateau and campaign report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

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
ORDERS_REASON = "float32 vs float64 plateau / no live compile"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in IF-09 in-memory path",
    "poisson.*": "Poisson not run in IF-09 in-memory path",
    "coupling.*": "coupling not run in IF-09 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in IF-09",
    "performance.one_node": "performance not measured in IF-09",
    "performance.two_node": "performance not measured in IF-09",
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


def precision_linf(n_cells: int = _exact.DEFAULT_N_CELLS, t=_exact.T) -> float:
    """Return L∞(float32, float64); must be O(1e-7) and both fields finite."""
    fields = _run.evaluate_precisions(n_cells, t)
    f32 = fields["float32"]
    f64 = fields["float64"]
    if f32.dtype != _exact.DTYPES[0] or f64.dtype != _exact.DTYPES[1]:
        raise ValueError("fields must be float32 and float64")
    if not _exact.fields_are_finite(f32, f64):
        raise ValueError("float32 and float64 fields must remain finite")
    worst = _run.max_precision_difference(n_cells, t)
    if worst > float(_exact.LINF_BOUND):
        raise ValueError("Linf(float32, float64) must be O(1e-7)")
    return worst


def _assert_precision_plateau() -> None:
    if tuple(_exact.DTYPES) != (np.float32, np.float64):
        raise ValueError("canonical dtypes must be float32 and float64")
    if float(_exact.LINF_BOUND) != 1.0e-6:
        raise ValueError("Linf bound must be 1e-6 (O(1e-7))")
    if precision_linf(_exact.DEFAULT_N_CELLS, t=_exact.T) > float(_exact.LINF_BOUND):
        raise ValueError("TR-01 float32 vs float64 Linf exceeds the plateau bound")


def _summary() -> dict:
    _assert_precision_plateau()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["infrastructure"],
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


def write_if09_report(output_dir) -> dict:
    """Check the dtype plateau and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
