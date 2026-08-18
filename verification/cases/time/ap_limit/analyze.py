"""TM-05 driver: AP implicit sweep, explicit blow-up, campaign report writer."""
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
    "orders": "AP stiffness sweep at fixed Δt; no temporal-order campaign in this increment",
    "amr.*": "AMR not run in TM-05 in-memory path",
    "poisson.*": "Poisson not run in TM-05 in-memory path",
    "coupling.*": "coupling residuals not measured in TM-05 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TM-05",
    "performance.one_node": "performance not measured in TM-05",
    "performance.two_node": "performance not measured in TM-05",
}
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}
STIFF_EPS = 1.0e-4
EXPLICIT_DIVERGE = 1.0e2
REDUCED_ATOL = 2.0e-3


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


def _assert_ap_contract() -> None:
    if float(_exact.DT) <= 0.0:
        raise ValueError("macro step DT must be positive")
    if float(_exact.Y0) != 1.0 or float(_exact.G) != 0.0 or float(_exact.F) != 0.0:
        raise ValueError("canonical data are y(0)=1, g=0, f=0")
    if float(_exact.reduced_limit()) != 0.0:
        raise ValueError("reduced limit ε→0 must be y=0")
    if not math.isclose(_exact.exact_y(0.0, 1.0), _exact.Y0):
        raise ValueError("exact relaxation at t=0 must recover y0")
    magnitudes: list[float] = []
    for eps in _exact.EPS_SWEEP:
        implicit = float(_run.implicit_step(_exact.Y0, _exact.DT, eps))
        if not math.isfinite(implicit) or abs(implicit) > 1.0:
            raise ValueError("implicit |y| must stay ≤ 1 after one macro step")
        magnitudes.append(abs(implicit))
    explicit = float(_run.explicit_step(_exact.Y0, _exact.DT, STIFF_EPS))
    if math.isfinite(explicit) and abs(explicit) <= EXPLICIT_DIVERGE:
        raise ValueError("explicit |y| must diverge for ε=1e-4")
    if magnitudes[-1] >= magnitudes[0]:
        raise ValueError("implicit |y| must decrease toward the reduced limit")
    for earlier, later in zip(magnitudes, magnitudes[1:]):
        if later > earlier:
            raise ValueError("implicit |y| must be monotone as ε decreases")
    if abs(magnitudes[-1] - float(_exact.reduced_limit())) > REDUCED_ATOL:
        raise ValueError("implicit solution must approach y=0 as ε→0")


def _summary() -> dict:
    _assert_ap_contract()
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


def write_tm05_report(output_dir) -> dict:
    """Check the in-memory AP contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
