"""GE-05 driver: polar volume identity and campaign report."""
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

VOLUME_ATOL = 1.0e-14
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
    "amr.*": "AMR not run for GE-05 in-memory path",
    "poisson.*": "Poisson not run for GE-05",
    "coupling.*": "coupling not run for GE-05",
    "parallel_invariance.*": "parallel invariance not run for GE-05",
    "performance.one_node": "performance not measured for GE-05 in-memory path",
    "performance.two_node": "performance not measured for GE-05 in-memory path",
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


def _assert_volume_contract() -> None:
    r_in = float(_exact.R_IN)
    r_out = float(_exact.R_OUT)
    n_r = int(_exact.N_R)
    n_theta = int(_exact.N_THETA)
    area = _exact.annulus_area(r_in, r_out)
    expected = math.pi * (r_out * r_out - r_in * r_in)
    if not math.isclose(area, expected, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("analytic annulus area must be π(r_out² − r_in²)")
    discrete = _run.annulus_volume(r_in, r_out, n_r, n_theta)
    if not math.isfinite(discrete) or abs(discrete - area) > VOLUME_ATOL:
        raise ValueError("discrete annulus volume must match π(r_out² − r_in²)")
    integral = _run.constant_state_integral(1.0, r_in, r_out, n_r, n_theta)
    if not math.isfinite(integral) or abs(integral - area) > VOLUME_ATOL:
        raise ValueError("constant-state integral must equal the annulus area")
    width = (r_out - r_in) / float(n_r)
    angle = 2.0 * math.pi / float(n_theta)
    axis = _run.axis_cell_volume(width, angle)
    documented = 0.5 * width * width * angle
    if not math.isfinite(axis) or abs(axis - documented) > 0.0:
        raise ValueError("axis helper must return ½ (Δr)² Δθ without dividing by r=0")
    reason = _run.refuse_public_polar_runtime()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("refuse_public_polar_runtime must return a documented reason")


def _summary() -> dict:
    _assert_volume_contract()
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


def write_ge05_report(output_dir) -> dict:
    """Check the in-memory polar-volume contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
