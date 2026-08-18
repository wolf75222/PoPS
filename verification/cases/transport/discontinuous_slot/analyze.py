"""TR-07 driver: translation identity, TV/overshoot pair, campaign report."""
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
    "orders": "discontinuous / limiter, not order-2",
    "amr.*": "AMR not run in TR-07 in-memory path",
    "poisson.*": "Poisson not run in TR-07 in-memory path",
    "coupling.*": "coupling not run in TR-07 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TR-07",
    "performance.one_node": "performance not measured in TR-07",
    "performance.two_node": "performance not measured in TR-07",
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


def _assert_slot_contract() -> None:
    if _exact.X0 != 0.5 or _exact.WIDTH != 0.25 or _exact.A != 1.0:
        raise ValueError("canonical slot is x0=0.5, w=0.25, a=1")
    n_cells = int(_exact.DEFAULT_N_CELLS)
    centers, _volumes = _exact.cell_centers(n_cells)
    time = 0.25
    translated = _exact.exact_slot(np.mod(centers - _exact.A * time, 1.0), 0.0)
    if not np.array_equal(_exact.exact_slot(centers, time), translated):
        raise ValueError("exact slot must translate as q(x,t)=q0(x-a t)")
    reference = _exact.exact_slot(centers, 0.0)
    if _run.total_variation(reference) != 2.0:
        raise ValueError("exact slot total variation must be 2")
    _centers, field, exact, _cell_volumes = _run.manufactured_smeared_pair(n_cells)
    if _run.overshoot(field, exact) <= 0.0:
        raise ValueError("manufactured 1.1 overshoot must report overshoot>0")
    if _run.undershoot(field, exact) <= 0.0:
        raise ValueError("manufactured undershoot must report undershoot>0")
    spiked = np.asarray(reference, dtype=np.float64).copy()
    spiked[int(np.argmax(spiked))] = 1.1
    if _run.overshoot(spiked, reference) <= 0.0:
        raise ValueError("a field with overshoot 1.1 must report overshoot>0")


def _summary() -> dict:
    _assert_slot_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["transport"],
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


def write_tr07_report(output_dir) -> dict:
    """Check the in-memory slot contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
