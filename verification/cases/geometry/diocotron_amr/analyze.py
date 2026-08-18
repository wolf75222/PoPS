"""GE-06 driver: ring tag coverage, unused-mode FFT, two-level envelope, report.

The in-memory ``|n-n_bg|`` envelope is the AMR oracle. Native smoke is the
uniform Cartesian Euler–Poisson path in ``run.py``; this driver does not
call ``pops.run``.
"""
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

N_CELLS = int(_exact.N_CELLS)
FFT_ATOL = 1.0e-12
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
    "orders": "in-memory diocotron tagging envelope; no spatial-order campaign",
    "amr.order_retained": "no AMR order series; public AM-01/AM-11 are 1-d",
    "amr.interface_error": "static coarse-fine band is AM-01; GE-06 records the tag envelope",
    "amr.bulk_error": "bulk vs uniform-fine comparison is later native work",
    "poisson.*": "Poisson is authored on the uniform native Case; in-memory report does not sample it",
    "coupling.*": "coupling not run for GE-06",
    "parallel_invariance.*": "parallel invariance not run for GE-06",
    "performance.one_node": "performance not measured for GE-06 in-memory path",
    "performance.two_node": "performance not measured for GE-06 in-memory path",
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


def _cell_volumes(n_cells: int = N_CELLS) -> np.ndarray:
    count = int(n_cells)
    _, _, width_x, width_y = _run.cell_centers(count)
    return np.full((count, count), width_x * width_y, dtype=np.float64)


def _chebyshev_halo(mask, width: int) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    added = np.zeros(selected.shape, dtype=bool)
    span = int(width)
    for shift_i in range(-span, span + 1):
        for shift_j in range(-span, span + 1):
            if shift_i == 0 and shift_j == 0:
                continue
            added |= np.roll(np.roll(selected, shift_i, axis=0), shift_j, axis=1)
    return added & ~selected


def _assert_tagging_contract() -> None:
    if _exact.THETA <= 0.0 or not (_exact.THETA < (_exact.N0 - _exact.N_BG)):
        raise ValueError("documented θ must sit strictly between 0 and n0-n_bg")
    if int(_exact.BUFFER_CELLS) != 2:
        raise ValueError("documented two-level envelope buffer must be 2")
    if int(_exact.UNUSED_MODE) != 3:
        raise ValueError("unused azimuthal mode must be m=3")
    field = _run.sample_field()
    ring = _run.ring_mask()
    if not np.any(ring):
        raise ValueError("documented ring must be non-empty on the mesh")
    raw = _run.raw_tag_mask()
    if not np.all(raw[ring]):
        raise ValueError("tagged set must cover the ring for documented θ")
    envelope = _run.envelope_mask()
    added = envelope & ~raw
    halo = _chebyshev_halo(raw, 2)
    if not np.array_equal(added, halo):
        raise ValueError("two-level envelope must be tagged cells plus buffer 2")
    amplitude = _run.unused_mode_amplitude()
    if not math.isfinite(amplitude) or amplitude > FFT_ATOL:
        raise ValueError("unused-mode (m=3) FFT of the unperturbed ring must be ~0")
    errors = reference_errors(field, field, _cell_volumes())
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact ring L∞ must be 0")


def _summary() -> dict:
    _assert_tagging_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["geometry", "amr"],
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


def write_ge06_report(output_dir) -> dict:
    """Check the in-memory ring/tag/FFT contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
