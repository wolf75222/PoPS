"""TM-06 driver: multirate BE contract and campaign report writer."""
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
    "orders": "multirate BE substep contract; no temporal-order campaign in this increment",
    "amr.*": "AMR not run in TM-06 in-memory path",
    "poisson.*": "Poisson not run in TM-06 in-memory path",
    "coupling.*": "coupling residuals not measured in TM-06 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TM-06",
    "performance.one_node": "performance not measured in TM-06",
    "performance.two_node": "performance not measured in TM-06",
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


def _assert_multirate_contract() -> None:
    if float(_exact.LAMBDA_F) != 8.0 or float(_exact.LAMBDA_S) != 1.0:
        raise ValueError("documented rates must be λ_f=8 and λ_s=1")
    if tuple(_exact.RATIOS) != (1, 2, 4, 8):
        raise ValueError("documented substep ratios must be (1, 2, 4, 8)")
    if float(_exact.DT) <= 0.0:
        raise ValueError("macro-step DT must be positive")
    y0 = float(_exact.Y0)
    z0 = float(_exact.Z0)
    dt = float(_exact.DT)
    if not math.isclose(_exact.exact_y(0.0, y0), y0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("exact y at t=0 must recover y0")
    if not math.isclose(_exact.exact_z(0.0, z0), z0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("exact z at t=0 must recover z0")
    for time in (0.0, 0.125, 0.25, 0.5, 1.0):
        expected_y = y0 * math.exp(-float(_exact.LAMBDA_F) * time)
        expected_z = z0 * math.exp(-float(_exact.LAMBDA_S) * time)
        if not math.isclose(_exact.exact_y(time, y0), expected_y, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("exact y must be y0 exp(-λ_f t)")
        if not math.isclose(_exact.exact_z(time, z0), expected_z, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("exact z must be z0 exp(-λ_s t)")
        state = _exact.exact_state(time, y0, z0)
        if not math.isclose(state[0], expected_y, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("exact_state y must match exact_y")
        if not math.isclose(state[1], expected_z, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("exact_state z must match exact_z")
    single = _run.single_rate_step(y0, z0, dt)
    multi_r1 = _run.multirate_step(y0, z0, dt, 1)
    if not math.isclose(single[0], multi_r1[0], rel_tol=0.0, abs_tol=0.0):
        raise ValueError("r=1 fast component must match single-rate BE")
    if not math.isclose(single[1], multi_r1[1], rel_tol=0.0, abs_tol=0.0):
        raise ValueError("r=1 slow component must match single-rate BE")
    errors = [_run.fast_error(ratio, dt, y0=y0, z0=z0) for ratio in _exact.RATIOS]
    if errors[0] <= 0.0:
        raise ValueError("r=1 fast-component error must be positive at finite Δt")
    for earlier, later in zip(errors, errors[1:]):
        if not later < earlier:
            raise ValueError("larger r must reduce the fast-component error")


def _summary() -> dict:
    _assert_multirate_contract()
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


def write_tm06_report(output_dir) -> dict:
    """Check the in-memory multirate BE contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
