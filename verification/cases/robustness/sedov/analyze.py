"""RB-05 driver: 2-d R(t)∝t^{2/5} scaling, zero anisotropy, campaign report."""
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
from verification.pops_verify.report import write_verification_report
from verification.pops_verify.symmetry import radial_anisotropy

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
NOT_APPLICABLE = {
    "orders": "self-similar Sedov radius; no spatial-order campaign in this increment",
    "amr.*": "AMR not run in RB-05 in-memory path",
    "poisson.*": "Poisson not run in RB-05 in-memory path",
    "coupling.*": "coupling residuals not measured in RB-05 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in RB-05",
    "performance.one_node": "performance not measured in RB-05",
    "performance.two_node": "performance not measured in RB-05",
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


def _assert_sedov_contract() -> None:
    exponent = _exact.self_similar_time_exponent()
    if not math.isclose(exponent, 2.0 / 5.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("2-d documented Sedov exponent must be 2/5")
    if _exact.blast_center() == _exact.domain_center():
        raise ValueError("blast centre must be off-centre")
    radius1 = _exact.shock_radius(1.0)
    radius2 = _exact.shock_radius(32.0)
    if not math.isclose(radius2 / radius1, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("R(32)/R(1) must be 32^{2/5} = 4")
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = _exact.polar_shock_radius(theta, 1.0)
    if not np.allclose(radii, radii[0]):
        raise ValueError("self-similar R(θ) must be constant")
    if not math.isclose(radial_anisotropy(radii), 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("constant R(θ) must have Task 18 anisotropy 0")
    if not np.allclose(_run.polar_radii(theta, 1.0), radii):
        raise ValueError("run.polar_radii must match the circular front")
    if not math.isclose(_run.front_anisotropy(theta, 1.0), 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("run.front_anisotropy must be 0 for a circular front")


def _summary() -> dict:
    _assert_sedov_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "nightly",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["euler"],
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


def write_rb05_report(output_dir) -> dict:
    """Check the in-memory Sedov contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
