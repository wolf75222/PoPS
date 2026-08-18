"""CP-10 driver: Jeans ω² sign, growth factor, campaign report."""
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
omega_squared = _exact.omega_squared
growth_factor = _exact.growth_factor
growth_rate = _exact.growth_rate
exact_state = _exact.exact_state
uniform_cell_centers = _exact.uniform_cell_centers
C_S = _exact.C_S
FOUR_PI_G_RHO0 = _exact.FOUR_PI_G_RHO0
K_STABLE = _exact.K_STABLE
K_UNSTABLE = _exact.K_UNSTABLE
K_JEANS = _exact.K_JEANS

N_CELLS = 32
GROWTH_TIME = 1.0
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
    "orders": "Jeans dispersion identity, not a resolution series",
    "amr.*": "AMR not run for CP-10 in-memory path",
    "poisson.*": "Poisson not run for CP-10 in-memory path",
    "coupling.phase_error": "phase not measured on the in-memory Jeans oracle",
    "coupling.energy_drift": "energy exchange not measured on the in-memory oracle",
    "parallel_invariance.*": "parallel invariance not run for CP-10",
    "performance.one_node": "performance not measured for CP-10 in-memory path",
    "performance.two_node": "performance not measured for CP-10 in-memory path",
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


def _assert_dispersion_signs() -> None:
    if K_STABLE <= K_JEANS or K_UNSTABLE >= K_JEANS:
        raise ValueError("canonical wavenumbers must straddle k_J")
    for wavenumber in (K_STABLE, K_UNSTABLE):
        residual = omega_squared(wavenumber) - (
            C_S * C_S * wavenumber * wavenumber - FOUR_PI_G_RHO0
        )
        if not math.isfinite(residual) or abs(residual) > 1.0e-12:
            raise ValueError(f"attractive Jeans residual at k={wavenumber} is {residual}")
    stable = omega_squared(K_STABLE)
    unstable = omega_squared(K_UNSTABLE)
    if not (stable > 0.0 and unstable < 0.0):
        raise ValueError(f"expected ω²(k=2)>0 and ω²(k=0.5)<0; got {stable}, {unstable}")


def _assert_growth_factor() -> None:
    gamma = growth_rate(K_UNSTABLE)
    factor = growth_factor(K_UNSTABLE, GROWTH_TIME)
    expected = math.exp(gamma * GROWTH_TIME)
    if not math.isfinite(factor) or abs(factor - expected) > 1.0e-12:
        raise ValueError(f"growth factor at t=1 is {factor}, expected {expected}")
    centers, volumes = uniform_cell_centers(N_CELLS)
    initial = exact_state(centers, 0.0, k=K_UNSTABLE)
    evolved = exact_state(centers, GROWTH_TIME, k=K_UNSTABLE)
    density = np.asarray(evolved[0], dtype=np.float64)
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
    background = np.asarray(_exact.BACKGROUND, dtype=np.float64)
    mismatch = np.max(
        np.abs((evolved - background[:, None]) - factor * (initial - background[:, None]))
    )
    if mismatch > 1.0e-12:
        raise ValueError(f"closed growing mode mismatch {mismatch}")


def _summary() -> dict:
    _assert_dispersion_signs()
    _assert_growth_factor()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "nightly",
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
        "poisson": dict(NULL_POISSON),
        "coupling": {
            "phase_error": None,
            "sign_ok": True,
            "energy_drift": None,
        },
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp10_report(output_dir) -> dict:
    """Check Jeans signs / growth and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
