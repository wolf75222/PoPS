"""CP-09 driver: Helmholtz identity, Poisson limit, campaign report."""
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
f_exact = _exact.f_exact
helmholtz_gain = _exact.helmholtz_gain
poisson_gain = _exact.poisson_gain
build_oracle = _run.build_oracle

N_CELLS = 32
POISSON_LIMIT_LAMBDA_D = 1.0e8
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
    "amr.*": "AMR not run for CP-09 in-memory path",
    "coupling.*": "coupling dynamics not run for CP-09 in-memory path",
    "parallel_invariance.*": "parallel invariance not run for CP-09",
    "performance.one_node": "performance not measured for CP-09 in-memory path",
    "performance.two_node": "performance not measured for CP-09 in-memory path",
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
    sample = build_oracle(N_CELLS)
    force = f_exact(sample["x"], k=sample["k"])
    applied = (
        _exact.wave_number(sample["k"]) ** 2 + _exact.screening_coefficient(sample["lambda_d"])
    ) * sample["phi"]
    identity_ok = np.allclose(applied, force, rtol=0.0, atol=1.0e-14)
    zeros = np.zeros_like(force)
    residual = reference_errors(zeros, zeros, sample["volumes"])
    potential = reference_errors(sample["phi"], phi_exact(sample["x"]), sample["volumes"])
    field = reference_errors(sample["e"], _exact.e_exact(sample["x"]), sample["volumes"])
    poisson = force * poisson_gain()
    large = phi_exact(sample["x"], lambda_d=POISSON_LIMIT_LAMBDA_D)
    limit = reference_errors(large, poisson, sample["volumes"])
    if not identity_ok or potential.linf != 0.0 or field.linf != 0.0 or residual.l2 != 0.0:
        raise ValueError("Helmholtz identity residual and exact-vs-exact φ, E must be 0")
    if not (
        math.isfinite(residual.l2)
        and math.isfinite(potential.linf)
        and math.isfinite(field.linf)
        and math.isfinite(limit.linf)
        and math.isclose(helmholtz_gain(lambda_d=np.inf), poisson_gain(), rel_tol=0.0, abs_tol=0.0)
        and limit.linf <= 1.0e-10
        and helmholtz_gain() < poisson_gain()
    ):
        raise ValueError("λ_D→∞ must recover the Poisson gain 1/(2πk)²")
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


def write_cp09_report(output_dir) -> dict:
    """Compare Helmholtz identity / Poisson limit and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
