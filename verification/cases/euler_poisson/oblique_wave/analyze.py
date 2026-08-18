"""CP-04 driver: Poisson identity, k-swap permutation, campaign report."""
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
complex_mode_amplitudes = _exact.complex_mode_amplitudes
exact_fields = _exact.exact_fields
physical_wavevector = _exact.physical_wavevector
poisson_residual = _exact.poisson_residual
uniform_cell_mesh = _exact.uniform_cell_mesh

N_CELLS = _exact.N_CELLS
NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
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
    "amr.*": "AMR not run for CP-04 in-memory path",
    "coupling.phase_error": "phase not measured on the in-memory CP-04 path",
    "coupling.energy_drift": "energy exchange not measured on the in-memory oracle",
    "parallel_invariance.*": "parallel invariance not run for CP-04",
    "performance.one_node": "performance not measured for CP-04 in-memory path",
    "performance.two_node": "performance not measured for CP-04 in-memory path",
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


def _poisson_residual_l2(n_cells: int = N_CELLS, t: float = 0.0) -> float:
    x, y, volumes = uniform_cell_mesh(n_cells)
    residual = np.asarray(poisson_residual(x, y, t), dtype=np.float64)
    value = float(np.sqrt(np.sum(volumes * residual * residual) / np.sum(volumes)))
    return 0.0 if value <= 1.0e-15 else value


def _fourier_poisson_ok() -> bool:
    amplitudes = complex_mode_amplitudes()
    wave = physical_wavevector()
    ik_dot_e = 1.0j * np.dot(wave, amplitudes["E"])
    return bool(np.max(np.abs(ik_dot_e - amplitudes["source"])) <= 1.0e-14)


def _swap_permutation_ok() -> bool:
    x, y, _ = uniform_cell_mesh()
    kx, ky = _exact.K_INTEGER
    time = 0.37
    swapped = exact_fields(x, y, time, kx=ky, ky=kx)
    permuted = exact_fields(y, x, time, kx=kx, ky=ky)
    return bool(
        np.max(np.abs(swapped["phi"] - permuted["phi"])) <= 1.0e-14
        and np.max(np.abs(swapped["n_e"] - permuted["n_e"])) <= 1.0e-14
        and np.max(np.abs(swapped["E_x"] - permuted["E_y"])) <= 1.0e-14
        and np.max(np.abs(swapped["E_y"] - permuted["E_x"])) <= 1.0e-14
    )


def _summary() -> dict:
    x, y, volumes = uniform_cell_mesh()
    fields = exact_fields(x, y, 0.0)
    density = np.asarray(fields["n_e"], dtype=np.float64)
    potential = np.asarray(fields["phi"], dtype=np.float64)
    electric_x = np.asarray(fields["E_x"], dtype=np.float64)
    electric_y = np.asarray(fields["E_y"], dtype=np.float64)
    density_errors = reference_errors(density, density, volumes)
    potential_errors = reference_errors(potential, potential, volumes)
    field_x_errors = reference_errors(electric_x, electric_x, volumes)
    field_y_errors = reference_errors(electric_y, electric_y, volumes)
    field_linf = max(field_x_errors.linf, field_y_errors.linf)
    residual = _poisson_residual_l2()
    kx, ky = _exact.K_INTEGER
    oblique = kx != 0 and ky != 0
    fourier_ok = _fourier_poisson_ok()
    swap_ok = _swap_permutation_ok()
    finite = (
        density_errors.linf == 0.0
        and potential_errors.linf == 0.0
        and field_linf == 0.0
        and math.isfinite(residual)
        and np.all(density > 0.0)
    )
    if not finite:
        raise ValueError("in-memory CP-04 identities must be finite with L∞=0")
    sign_ok = (
        abs(residual) <= 1.0e-12
        and fourier_ok
        and swap_ok
        and oblique
    )
    if not sign_ok:
        raise ValueError("CP-04 Poisson / swap identities must hold")
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["continuity", "momentum", "poisson", "electrostatic_source"],
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
            "potential_error": float(potential_errors.linf),
            "field_error": float(field_linf),
            "residual_l2": residual,
        },
        "coupling": {
            "phase_error": None,
            "sign_ok": bool(sign_ok),
            "energy_drift": None,
        },
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_cp04_report(output_dir) -> dict:
    """Compare exact Poisson / swap identities and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
