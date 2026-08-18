"""RB-08 driver: RH/geometry IC contract, campaign report with empty orders."""
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

N_CELLS = int(_exact.N_CELLS)
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
    "orders": "DMR morphology / no closed form",
    "amr.*": "AMR not run for RB-08 in-memory path",
    "poisson.*": "Poisson not run for RB-08",
    "coupling.*": "coupling not run for RB-08",
    "parallel_invariance.*": "parallel invariance not run for RB-08",
    "performance.one_node": "performance not measured for RB-08 in-memory path",
    "performance.two_node": "performance not measured for RB-08 in-memory path",
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


def _cell_grid(n_cells: int = N_CELLS):
    count = int(n_cells)
    lower_x, lower_y = (float(value) for value in _exact.DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in _exact.DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / float(count)
    width_y = (upper_y - lower_y) / float(count)
    x_centers = lower_x + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    volumes = np.full((count, count), width_x * width_y, dtype=np.float64)
    return x, y, volumes


def _assert_dmr_contract() -> None:
    if not math.isclose(_exact.SHOCK_ANGLE_DEG, 30.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("documented shock angle must be 30 degrees")
    if not math.isclose(_exact.WEDGE_ANGLE_DEG, 30.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("documented wedge angle must be 30 degrees")
    jump = _exact.rankine_hugoniot(mach=10.0, gamma=1.4, rho0=1.4, u0=0.0, p0=1.0)
    post = _exact.post_shock_state()
    if not math.isclose(post["rho"], jump["rho"], rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("post-shock density must match Rankine–Hugoniot")
    if not math.isclose(post["p"], jump["p"], rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("post-shock pressure must match Rankine–Hugoniot")
    speed = math.hypot(post["u"], post["v"])
    if not math.isclose(speed, jump["u"], rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("post-shock speed must match Rankine–Hugoniot")
    x, y, volumes = _cell_grid()
    primitives = _exact.primitives(x, y, 0.0)
    density = np.asarray(primitives["rho"], dtype=np.float64)
    pressure = np.asarray(primitives["p"], dtype=np.float64)
    if not (np.all(density > 0.0) and np.all(pressure > 0.0)):
        raise ValueError("DMR IC density and pressure must be positive")
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("IC-vs-IC density L∞ must be 0")


def _summary() -> dict:
    _assert_dmr_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
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


def write_rb08_report(output_dir) -> dict:
    """Check the in-memory DMR contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
