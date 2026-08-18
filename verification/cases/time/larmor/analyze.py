"""TM-04 driver: in-memory Larmor rotation and campaign report writer."""
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
    "orders": "Larmor speed/period contract; no temporal-order campaign in this increment",
    "amr.*": "AMR not run in TM-04 in-memory path",
    "poisson.*": "Poisson not run in TM-04 in-memory path",
    "coupling.*": "coupling residuals not measured in TM-04 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TM-04",
    "performance.one_node": "performance not measured in TM-04",
    "performance.two_node": "performance not measured in TM-04",
}
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}
SPEED_TOL = 1.0e-12


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


def _assert_larmor_contract() -> None:
    omega_c = float(_exact.OMEGA_C)
    if omega_c <= 0.0:
        raise ValueError("cyclotron frequency OMEGA_C must be positive")
    u0 = np.asarray(_exact.U0, dtype=np.float64)
    if u0.shape != (2,) or not np.allclose(u0, (1.0, 0.0)):
        raise ValueError("canonical u0 must be (1, 0)")
    initial_speed = _exact.speed(u0)
    if not math.isclose(initial_speed, 1.0, rel_tol=0.0, abs_tol=SPEED_TOL):
        raise ValueError("canonical |u0| must be 1")
    for time in (0.0, 0.1, 0.5, math.pi / omega_c):
        advanced = _exact.exact_advance(u0, time, omega_c=omega_c)
        if not math.isclose(_exact.speed(advanced), initial_speed, rel_tol=0.0, abs_tol=SPEED_TOL):
            raise ValueError("exact rotation must conserve |u|")
        if not np.allclose(advanced, _run.exact_advance(u0, time, omega_c=omega_c)):
            raise ValueError("run.exact_advance must match the closed-form rotation")
    period = 2.0 * math.pi / omega_c
    if not np.allclose(_exact.exact_advance(u0, period, omega_c=omega_c), u0, atol=SPEED_TOL):
        raise ValueError("exact rotation must return to u0 at t=2π/ωc")
    quarter = _exact.exact_advance(u0, 0.5 * math.pi / omega_c, omega_c=omega_c)
    if not np.allclose(quarter, (0.0, -1.0), atol=SPEED_TOL):
        raise ValueError("exact rotation of (1, 0) at t=π/(2ωc) must be (0, -1)")
    midpoint = _run.implicit_midpoint(u0, _exact.DT, omega_c=omega_c)
    if not math.isclose(_exact.speed(midpoint), initial_speed, rel_tol=0.0, abs_tol=SPEED_TOL):
        raise ValueError("implicit midpoint must conserve |u| to 1e-12 on one step")
    euler = _run.explicit_euler(u0, _exact.DT, omega_c=omega_c)
    if not (_exact.speed(euler) > initial_speed):
        raise ValueError("explicit Euler |u| must grow on one step")


def _summary() -> dict:
    _assert_larmor_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["time"],
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


def write_tm04_report(output_dir) -> dict:
    """Check the in-memory Larmor map and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
