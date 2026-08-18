"""CP-12 driver: cancelled charge, E=0, constant φ, campaign report."""
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
phi_exact = _exact.phi_exact
e_exact = _exact.e_exact
poisson_rhs = _exact.poisson_rhs
build_oracle = _run.build_oracle

N_CELLS = 32
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
    "orders": "single-resolution in-memory exact comparison",
    "amr.*": "AMR not run for CP-12 in-memory path",
    "coupling.*": "coupling dynamics not run for CP-12 in-memory path",
    "parallel_invariance.*": "parallel invariance not run for CP-12",
    "performance.one_node": "performance not measured for CP-12 in-memory path",
    "performance.two_node": "performance not measured for CP-12 in-memory path",
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


def _summary() -> dict:
    sample = build_oracle(N_CELLS, delta=0.1)
    charge = np.asarray(sample["net_charge"], dtype=np.float64)
    zeros = np.zeros_like(charge)
    residual = reference_errors(sample["rhs"], poisson_rhs(sample["x"], delta=0.1), sample["volumes"])
    potential = reference_errors(sample["phi"], phi_exact(sample["x"]), sample["volumes"])
    field = reference_errors(sample["e"], e_exact(sample["x"]), sample["volumes"])
    charge_errors = reference_errors(charge, zeros, sample["volumes"])
    if (
        charge_errors.linf != 0.0
        or field.linf != 0.0
        or potential.linf != 0.0
        or residual.l2 != 0.0
        or not (
            math.isfinite(charge_errors.l1)
            and math.isfinite(field.linf)
            and math.isfinite(potential.linf)
            and math.isfinite(residual.l2)
        )
    ):
        raise ValueError("cancelled charge must give E=0, constant φ, and zero Poisson residual")
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["euler_poisson"],
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
        "poisson": {
            "potential_error": float(potential.linf),
            "field_error": float(field.linf),
            "residual_l2": float(residual.l2),
        },
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp12_report(output_dir) -> dict:
    """Compare cancelled charge / E=0 / constant φ and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
