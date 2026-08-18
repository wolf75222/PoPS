"""GE-01 driver: harmonic Δφ = 0, polar-runtime refusal, campaign report."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

LAPLACIAN_ATOL = 1.0e-12
ORDERS_REASON = "capability-gated polar runtime"
NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
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
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run for GE-01 in-memory path",
    "poisson.field_error": "no discrete field solve on the in-memory path",
    "coupling.*": "coupling not run for GE-01",
    "parallel_invariance.*": "parallel invariance not run for GE-01",
    "performance.one_node": "performance not measured for GE-01 in-memory path",
    "performance.two_node": "performance not measured for GE-01 in-memory path",
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


def _assert_harmonic_contract() -> dict:
    if int(_exact.M) != 2:
        raise ValueError("documented azimuthal mode must be m=2")
    if float(_exact.R_MIN) != 0.2 or float(_exact.R_MAX) != 1.0:
        raise ValueError("documented annulus must be r in [0.2, 1]")
    if _exact.R_MIN <= 0.0 or _exact.in_annulus(0.0):
        raise ValueError("r=0 must be excluded from the annulus")
    reason = _run.refuse_public_polar_runtime()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("refuse_public_polar_runtime must return a non-empty reason")
    radius, angle, volumes = _exact.polar_cell_grid()
    if not _exact.in_annulus(radius) or np.any(radius <= 0.0):
        raise ValueError("polar cell grid must stay inside the annulus")
    laplacian = np.asarray(_exact.polar_laplacian(radius, angle), dtype=np.float64)
    residual = reference_errors(laplacian, np.zeros_like(laplacian), volumes)
    if not math.isfinite(residual.l2) or residual.linf > LAPLACIAN_ATOL:
        raise ValueError("analytic polar Laplacian of the harmonic must vanish")
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    polar_field = np.asarray(_exact.phi(radius, angle), dtype=np.float64)
    cartesian_field = np.asarray(_exact.cartesian_equivalent(x, y), dtype=np.float64)
    potential = reference_errors(cartesian_field, polar_field, volumes)
    if potential.linf > LAPLACIAN_ATOL or not (
        math.isfinite(potential.l1)
        and math.isfinite(potential.l2)
        and math.isfinite(potential.linf)
    ):
        raise ValueError("cartesian_equivalent must match φ(r, θ)")
    return {
        "potential_error": float(potential.linf),
        "field_error": None,
        "residual_l2": float(residual.l2),
    }


def _summary() -> dict:
    poisson = _assert_harmonic_contract()
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
        "poisson": dict(poisson),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_ge01_report(output_dir) -> dict:
    """Check the in-memory harmonic contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
