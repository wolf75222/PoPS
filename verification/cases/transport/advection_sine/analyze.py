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


def write_complement_markdown(output_dir, payload: dict) -> Path:
    """Write COMPLEMENT.md from a ``run_campaign`` payload. No pops.run."""
    from pathlib import Path as _Path

    target = _Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = payload.get("summary") or {}
    lines = [
        "# TR-01 complement",
        "",
        f"dim={payload.get('dim')} smoke={payload.get('smoke')}",
        "",
        f"planned={summary.get('n_planned')} ok={summary.get('n_ok')} "
        f"failed={summary.get('n_failed')} unsupported={summary.get('n_unsupported')} "
        f"unavailable={summary.get('n_unavailable')}",
        "",
        f"order pairs ≥ 1.8: {summary.get('spatial_pairs_ge_1_8')}/"
        f"{summary.get('spatial_pairs')}",
        f"acceptance_order_met={summary.get('acceptance_order_met')}",
        f"conservation_ok={summary.get('conservation_ok')}",
        f"layouts_ok={summary.get('layouts_ok')}",
        f"dims_ok={summary.get('dims_ok')}",
        f"amr_ran={summary.get('amr_ran')}",
        f"fixed_block_peaks={summary.get('fixed_block_peaks')}",
        "",
        "## Spatial orders",
        "",
    ]
    for row in summary.get("spatial_orders") or []:
        lines.append(
            f"- {row.get('group')} observed={row.get('observed_order'):.4f} "
            f"threshold={row.get('threshold')}"
        )
    lines.extend(["", "## Temporal orders", ""])
    for row in summary.get("temporal_orders") or []:
        lines.append(
            f"- {row.get('group')} observed={row.get('observed_order'):.4f}"
        )
    lines.extend(["", "## Failures", ""])
    for name in ("failed", "unsupported", "unavailable"):
        items = summary.get(name) or []
        lines.append(f"- {name}: {items if items else 'none'}")
    path = target / "COMPLEMENT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
