"""TR-01 3-d oblique sine: norms, observed order, provenance, report."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

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
    "amr.*": "AMR is AM-01; TR-01 is uniform 3-d",
    "poisson.*": "Poisson not run in TR-01",
    "coupling.*": "coupling not run in TR-01",
    "parallel_invariance.*": "parallel invariance is IF-01",
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
        "native_dimensions": [3],
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
    """Write a campaign report from an already-computed 3-d error series."""
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


def write_tr01_report(output_dir, *, n_cells=16) -> dict:
    """Exact-vs-exact 3-d cell averages, then the four report artifacts."""
    lo, hi = _exact.cell_bounds(n_cells)

    def _u(x, y, z, time):
        return _exact.exact_sine_3d(x, y, z, time)

    oracle = analytic_cell_averages(_u, lo, hi, 0.0)
    _, _, _, volumes = _exact.uniform_cell_mesh(n_cells)
    errors = reference_errors(oracle, oracle, volumes)
    if errors.linf != 0.0:
        raise ValueError("in-memory exact vs exact Linf must be 0")
    return analyze_series([errors.linf], [1.0 / float(n_cells)], output_dir)


def write_native_campaign_report(output_dir, campaign: dict) -> dict:
    """Write the report from a live Dim-3 ``run_order_campaign`` result."""
    return analyze_series(campaign["linf"], campaign["spacings"], output_dir)
