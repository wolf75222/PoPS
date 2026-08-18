"""AM-02 driver: prescribed barycenter, 256-cycle mass drift, campaign report."""
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

_TR02_DIR = _CASE_DIR.parents[1] / "transport" / "gaussian_pulse"
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
_tr02 = load_sibling_module(_TR02_DIR / "exact.py")

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
    "orders": "manufactured regrid jump; no fitted order campaign in this increment",
    "amr.order_retained": "manufactured jump ∝ h² is a contract, not a fitted AMR order series",
    "amr.interface_error": "static coarse-fine band is AM-01; AM-02 records regrid before/after scalars",
    "amr.bulk_error": "bulk vs uniform-fine comparison is later native work",
    "poisson.*": "Poisson not run in AM-02 in-memory path",
    "coupling.*": "coupling not run in AM-02 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-02",
    "performance.one_node": "performance not measured in AM-02",
    "performance.two_node": "performance not measured in AM-02",
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


def _assert_prescribed_patch_contract() -> None:
    if _exact.STRESS_CYCLES != 256:
        raise ValueError("AM-02 stress campaign is 256 refine/coarsen cycles")
    time = 0.2
    x0 = _tr02.X0
    speed = _tr02.A
    expected = (x0 + speed * time) % 1.0
    closed = _exact.patch_center(time, x0=x0, a=speed)
    center = _run.prescribed_patch_center(time)
    if not math.isclose(closed, expected, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("closed-form patch center must follow x0 + a t")
    if not math.isclose(center, expected, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("prescribed patch center must equal the exact barycenter")
    drift = _run.stress_mass_drift(n_cycles=_exact.STRESS_CYCLES)
    if not math.isclose(drift, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("exact-field 256-cycle mass drift must be 0")
    coarse = 1.0 / 32.0
    fine = 1.0 / 64.0
    before_c, after_c = _exact.manufactured_regrid_errors(coarse)
    before_f, after_f = _exact.manufactured_regrid_errors(fine)
    jump_c = after_c - before_c
    jump_f = after_f - before_f
    if jump_c <= 0.0 or jump_f <= 0.0:
        raise ValueError("manufactured regrid jump must be positive")
    ratio = jump_c / jump_f
    expected_ratio = (coarse / fine) ** 2
    if not math.isclose(ratio, expected_ratio, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError("manufactured regrid jump must be proportional to h²")


def _summary() -> dict:
    _assert_prescribed_patch_contract()
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
        "orders": [],
        "amr": {
            "order_retained": None,
            "invariants_ok": True,
            "interface_error": None,
            "bulk_error": None,
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am02_report(output_dir) -> dict:
    """Check the in-memory prescribed-patch contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
