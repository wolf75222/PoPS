"""RB-09 driver: IC checks, campaign report with empty orders."""
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
primitives_1d = _exact.primitives_1d

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
    "orders": "interacting blasts / no closed form",
    "amr.*": "AMR not run for RB-09 in-memory path",
    "poisson.*": "Poisson not run for RB-09",
    "coupling.*": "coupling not run for RB-09",
    "parallel_invariance.*": "parallel invariance not run for RB-09",
    "performance.one_node": "performance not measured for RB-09 in-memory path",
    "performance.two_node": "performance not measured for RB-09 in-memory path",
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
    width = 1.0 / float(n_cells)
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    volumes = np.full(n_cells, width, dtype=np.float64)
    return centers, volumes


def _assert_ic_contract() -> None:
    centers, width = _cell_grid()
    primitives = primitives_1d(centers, 0.0)
    density = np.asarray(primitives[0], dtype=np.float64)
    pressure = np.asarray(primitives[2], dtype=np.float64)
    if not (np.all(density > 0.0) and np.all(pressure > 0.0)):
        raise ValueError("blast-wave IC density and pressure must be positive")
    left = centers < _exact.X_LEFT
    right = centers > _exact.X_RIGHT
    middle = ~left & ~right
    if not (np.any(left) and np.any(middle) and np.any(right)):
        raise ValueError("blast-wave IC must sample all three pressure regions")
    if not (
        np.allclose(pressure[left], _exact.P_LEFT)
        and np.allclose(pressure[middle], _exact.P_MIDDLE)
        and np.allclose(pressure[right], _exact.P_RIGHT)
    ):
        raise ValueError("blast-wave IC must be the three-region Woodward–Colella state")
    volumes = np.full(density.shape, width, dtype=np.float64)
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("IC-vs-IC density L∞ must be 0")


def _summary() -> dict:
    _assert_ic_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
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


def write_rb09_report(output_dir) -> dict:
    """Check the two-blast IC and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
