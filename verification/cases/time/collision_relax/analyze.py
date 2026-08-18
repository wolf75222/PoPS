"""TM-03 driver: in-memory exponential relaxation and campaign report writer."""
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
    "orders": "exact collision relaxation; no temporal-order campaign in this increment",
    "amr.*": "AMR not run in TM-03 in-memory path",
    "poisson.*": "Poisson not run in TM-03 in-memory path",
    "coupling.*": "coupling residuals not measured in TM-03 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TM-03",
    "performance.one_node": "performance not measured in TM-03",
    "performance.two_node": "performance not measured in TM-03",
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


def _assert_relaxation_contract() -> None:
    if float(_exact.NU) <= 0.0:
        raise ValueError("collision frequency NU must be positive")
    centers, volumes = _exact.uniform_cell_centers()
    u0 = _exact.initial_field(centers)
    u_bar = _exact.barycenter(u0, volumes)
    if not math.isclose(u_bar, 1.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("manufactured cosine barycenter must be 1")
    if not np.allclose(_exact.exact_relax(u0, 0.0, nu=_exact.NU, u_bar=u_bar), u0):
        raise ValueError("exact relaxation at t=0 must recover u0")
    half_life = math.log(2.0) / float(_exact.NU)
    half = _exact.exact_relax(u0, half_life, nu=_exact.NU, u_bar=u_bar)
    if not np.allclose(half, u_bar + 0.5 * (u0 - u_bar)):
        raise ValueError("deviation must halve at t=ln(2)/ν")
    for time in (0.0, 0.1, 0.5, 1.0, 4.0):
        relaxed = _exact.exact_relax(u0, time, nu=_exact.NU, u_bar=u_bar)
        advanced = _run.relax(u0, time, nu=_exact.NU, volumes=volumes)
        if not math.isclose(
            _exact.barycenter(relaxed, volumes), u_bar, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError("barycenter moment must stay constant")
        if not np.allclose(advanced, relaxed):
            raise ValueError("run.relax must match the closed-form map")


def _summary() -> dict:
    _assert_relaxation_contract()
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


def write_tm03_report(output_dir) -> dict:
    """Check the in-memory collision map and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
