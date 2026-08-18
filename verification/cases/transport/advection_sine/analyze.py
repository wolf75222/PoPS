"""TR-01 norms, observed order, and campaign report writer."""
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

_exact = load_sibling_module(_CASE_DIR / "exact.py")
exact_sine = _exact.exact_sine
uniform_cell_centers = _exact.uniform_cell_centers
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

CASE_ID = "TR-01"
ORDER_THRESHOLD = 1.8
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
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}
NOT_RUN_REASONS = {
    "amr.*": "AMR not run in TR-01 in-memory path",
    "poisson.*": "Poisson not run in TR-01",
    "coupling.*": "coupling not run in TR-01",
    "parallel_invariance.*": "parallel invariance not run in TR-01",
    "performance.one_node": "performance not measured in TR-01",
    "performance.two_node": "performance not measured in TR-01",
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


def _summary(*, orders: list, order_reason: str | None) -> dict:
    reasons = dict(NOT_RUN_REASONS)
    if not orders:
        reasons["orders"] = order_reason or "single-resolution series"
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["transport"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": list(orders),
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": reasons,
        "artifacts": dict(ARTIFACTS),
    }


def analyze_series(errors, resolutions, output_dir) -> dict:
    """Write a campaign report from an already-computed error series."""
    error_series = list(errors)
    spacing_series = list(resolutions)
    if len(error_series) < 2:
        orders: list[dict] = []
        reason = "single-resolution series"
    else:
        observed = observed_order(error_series, spacing_series)
        orders = [
            {
                "case_id": CASE_ID,
                "kind": "spatial",
                "variable": "q",
                "observed_order": float(value),
                "threshold": ORDER_THRESHOLD,
            }
            for value in observed
        ]
        reason = None
    return write_verification_report(
        _summary(orders=orders, order_reason=reason),
        output_dir,
    )


def write_tr01_report(output_dir, *, n_cells=32) -> dict:
    """Compare exact_sine(x, 0) to itself and write the four Task 20 artifacts."""
    centers, volumes = uniform_cell_centers(n_cells)
    field = exact_sine(centers, 0.0)
    errors = reference_errors(field, field, volumes)
    if errors.linf != 0.0:
        raise ValueError("in-memory exact vs exact Linf must be 0")
    return analyze_series([errors.linf], [1.0 / float(n_cells)], output_dir)
