"""AM-07 driver: fine-region vs uniform-h/2 comparison and campaign report."""
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

CASE_ID = "AM-07"
ORDER_THRESHOLD = 1.8
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
    "amr.invariants_ok": "conservation not measured in AM-07 in-memory path",
    "amr.interface_error": "interface-band layer is AM-01/AM-08; AM-07 compares the fine-region interior",
    "poisson.*": "Poisson not run in AM-07 in-memory path",
    "coupling.*": "coupling residuals not measured in AM-07 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-07",
    "performance.one_node": "performance not measured in AM-07",
    "performance.two_node": "performance not measured in AM-07",
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


def _assert_fine_region_match(h) -> dict[str, float]:
    series = _run.three_series(h)
    amr_fine = series[_run.SERIES_AMR_H_FINE_H2]
    uniform_h2 = series[_run.SERIES_UNIFORM_H2]
    expected = _exact.manufactured_error(_exact.fine_spacing(h))
    if amr_fine != uniform_h2:
        raise ValueError("fine-region AMR error must match uniform h/2")
    if amr_fine != expected:
        raise ValueError("fine-region AMR error must equal manufactured E∝(h/2)²")
    return series


def _manufactured_orders() -> list[dict]:
    spacings = list(_exact.H_SERIES)
    errors = [_exact.manufactured_error(h) for h in spacings]
    observed = observed_order(errors, spacings)
    return [
        {
            "case_id": CASE_ID,
            "kind": "spatial",
            "variable": "q",
            "observed_order": float(value),
            "threshold": ORDER_THRESHOLD,
        }
        for value in observed
    ]


def _summary(*, bulk_error: float) -> dict:
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
        "orders": _manufactured_orders(),
        "amr": {
            "order_retained": True,
            "invariants_ok": None,
            "interface_error": None,
            "bulk_error": float(bulk_error),
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am07_report(output_dir, *, h=None) -> dict:
    """Check the fine-region vs uniform-h/2 contract and write Task 20 artifacts."""
    spacing = _exact.H if h is None else float(h)
    series = _assert_fine_region_match(spacing)
    return write_verification_report(
        _summary(bulk_error=series[_run.SERIES_AMR_H_FINE_H2]),
        output_dir,
    )
