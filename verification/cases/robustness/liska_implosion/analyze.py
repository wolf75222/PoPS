"""RB-07 driver: leftover residual 0 on the exact IC, empty-order report."""
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
from verification.pops_verify.symmetry import xy_symmetry_error

_exact = load_sibling_module(_CASE_DIR / "exact.py")
primitives = _exact.primitives
leftover_residual = _exact.leftover_residual

N_CELLS = 32
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
    "orders": "implosion / no analytic late-time: observed spatial order is not applicable",
    "amr.*": "AMR not run for RB-07 in-memory path",
    "poisson.*": "Poisson not run for RB-07",
    "coupling.*": "coupling not run for RB-07",
    "parallel_invariance.*": "parallel invariance not run for RB-07",
    "performance.one_node": "performance not measured for RB-07 in-memory path",
    "performance.two_node": "performance not measured for RB-07 in-memory path",
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
    length = float(_exact.DOMAIN_UPPER[0] - _exact.DOMAIN_LOWER[0])
    width = length / float(count)
    origin = float(_exact.DOMAIN_LOWER[0])
    centers = origin + (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    volumes = np.full((count, count), width * width, dtype=np.float64)
    return x, y, volumes


def _summary() -> dict:
    x, y, volumes = _cell_grid()
    field = primitives(x, y, 0.0)
    density = np.asarray(field["rho"], dtype=np.float64)
    residual = leftover_residual(field)
    for values in residual.values():
        peak = float(np.max(np.abs(values)))
        if peak != 0.0 or not math.isfinite(peak):
            raise ValueError("leftover residual of the exact IC must be 0")
    if xy_symmetry_error(density) != 0.0:
        raise ValueError("exact density must have Task 18 E_xy = 0")
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
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


def write_rb07_report(output_dir) -> dict:
    """Check leftover residual 0 and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
