"""AM-03 driver: pulse-core tag, buffer-2 halo, static hysteresis, report."""
from __future__ import annotations

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

_TR02_DIR = _CASE_DIR.parents[1] / "transport" / "gaussian_pulse"
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
_tr02 = load_sibling_module(_TR02_DIR / "exact.py")

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
    "orders": "tagging contract; no spatial-order campaign in this increment",
    "amr.order_retained": "no AMR order series in AM-03 in-memory path",
    "amr.interface_error": "static coarse-fine band is AM-01; AM-03 records the tag mask",
    "amr.bulk_error": "bulk vs uniform-fine comparison is later native work",
    "poisson.*": "Poisson not run in AM-03 in-memory path",
    "coupling.*": "coupling not run in AM-03 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-03",
    "performance.one_node": "performance not measured in AM-03",
    "performance.two_node": "performance not measured in AM-03",
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


def _periodic_halo(mask, width: int) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    added = np.zeros(selected.shape, dtype=bool)
    for shift in range(1, int(width) + 1):
        added |= np.roll(selected, shift)
        added |= np.roll(selected, -shift)
    return added & ~selected


def _assert_tagging_contract() -> None:
    if _exact.THETA <= 0.0 or _exact.THETA2 <= 0.0:
        raise ValueError("documented θ and θ2 must be positive")
    if tuple(_exact.BUFFER_WIDTHS) != (1, 2, 4):
        raise ValueError("documented buffer widths must be (1, 2, 4)")
    if not np.isclose(_exact.X0, _tr02.X0) or not np.isclose(_exact.SIGMA, _tr02.SIGMA):
        raise ValueError("AM-03 pulse geometry must match TR-02")
    centers, field = _run.sample_field()
    expected = _tr02.exact_gaussian(centers, 0.0)
    if not np.allclose(field, expected):
        raise ValueError("sampled field must be the TR-02 exact Gaussian")
    raw = _run.raw_tag_mask()
    core = _run.pulse_core_mask()
    if not np.any(core):
        raise ValueError("documented pulse core must be non-empty")
    if not np.all(raw[core]):
        raise ValueError("tagged set must contain the pulse core for documented θ")
    buffered = _run.buffered_tag_mask(2)
    added = buffered & ~raw
    halo = _periodic_halo(raw, 2)
    if not np.array_equal(added, halo) or int(np.count_nonzero(added)) != 4:
        raise ValueError("buffer of 2 must add exactly 2 cells on each side")
    empty = np.zeros_like(raw)
    first = _run.hysteresis_update(empty)
    second = _run.hysteresis_update(first)
    if not np.array_equal(first, raw) or not np.array_equal(second, first):
        raise ValueError("hysteresis must not oscillate on the static field")
    from_full = _run.hysteresis_update(np.ones_like(raw))
    if not np.array_equal(_run.hysteresis_update(from_full), from_full):
        raise ValueError("hysteresis from a fully tagged start must be stationary")


def _summary() -> dict:
    _assert_tagging_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["amr"],
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


def write_am03_report(output_dir) -> dict:
    """Check the in-memory tagging contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
