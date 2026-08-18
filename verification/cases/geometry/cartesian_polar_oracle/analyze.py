"""GE-04 driver: peak radius, interpolated L∞, capability-gated report."""
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

N_CELLS = int(_exact.N_CELLS)
LINF_BOUND = float(_run.LINF_BOUND)
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
NOT_APPLICABLE = {
    "orders": "capability-gated polar runtime",
    "amr.*": "AMR not run for GE-04 in-memory path",
    "poisson.*": "Poisson not run for GE-04",
    "coupling.*": "coupling not run for GE-04",
    "parallel_invariance.*": "parallel invariance not run for GE-04",
    "performance.one_node": "performance not measured for GE-04 in-memory path",
    "performance.two_node": "performance not measured for GE-04 in-memory path",
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


def _assert_oracle_contract() -> None:
    cartesian_peak = _exact.peak_radius_cartesian()
    polar_peak = _exact.peak_radius_polar()
    cartesian_tol = 2.0 / float(N_CELLS)
    polar_tol = float(_exact.R_OUTER) / float(_exact.N_R)
    if not math.isfinite(cartesian_peak) or abs(cartesian_peak - _exact.RING_RADIUS) > cartesian_tol:
        raise ValueError("Cartesian sampling of φ must peak at r=0.5")
    if not math.isfinite(polar_peak) or abs(polar_peak - _exact.RING_RADIUS) > polar_tol:
        raise ValueError("polar sampling of φ must peak at r=0.5")
    errors = _run.field_to_field_errors()
    if not (
        math.isfinite(errors.l1)
        and math.isfinite(errors.l2)
        and math.isfinite(errors.linf)
    ):
        raise ValueError("interpolated field-to-field norms must be finite")
    if errors.linf >= LINF_BOUND:
        raise ValueError(
            f"polar→Cartesian L∞={errors.linf} must be < {LINF_BOUND} on {N_CELLS}×{N_CELLS}"
        )
    reason = _run.refuse_public_polar_runtime()
    if not reason:
        raise ValueError("polar runtime refusal reason must be non-empty")


def _summary() -> dict:
    _assert_oracle_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["geometry"],
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


def write_ge04_report(output_dir) -> dict:
    """Check the in-memory Cartesian/polar oracle contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
