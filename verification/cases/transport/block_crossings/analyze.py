"""TR-04 driver: placement agreement and campaign report writer."""
from __future__ import annotations

import subprocess
import sys
from itertools import combinations
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
    "orders": "single-resolution in-memory exact translation",
    "amr.*": "AMR not run in TR-04 in-memory path",
    "poisson.*": "Poisson not run in TR-04 in-memory path",
    "coupling.*": "coupling not run in TR-04 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TR-04",
    "performance.one_node": "performance not measured in TR-04",
    "performance.two_node": "performance not measured in TR-04",
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


def field_to_field_linf(u, v) -> float:
    """Return max |u - v| on a shared sample."""
    left = np.asarray(u, dtype=np.float64)
    right = np.asarray(v, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("shape mismatch")
    if left.size == 0:
        raise ValueError("empty input")
    return float(np.max(np.abs(left - right)))


def placements_agree(n_cells=None) -> float:
    """Return field-to-field L∞ across the three placements (must be 0)."""
    count = _exact.N_CELLS if n_cells is None else int(n_cells)
    fields = [_run.sample_placement(name, count)[1] for name in _exact.PLACEMENTS]
    worst = 0.0
    for left, right in combinations(fields, 2):
        worst = max(worst, field_to_field_linf(left, right))
    if worst != 0.0:
        raise ValueError("exact placements must agree with field-to-field Linf = 0")
    return worst


def _assert_placement_contract() -> None:
    if tuple(_exact.PLACEMENTS) != ("face", "edge", "corner"):
        raise ValueError("documented placements must be face, edge, corner")
    if _exact.FACE != 0.5:
        raise ValueError("face must be at 0.5")
    if _exact.N_BLOCKS != 2 or tuple(_exact.BLOCK_EDGES) != (0.0, 0.5, 1.0):
        raise ValueError("1-d placements must use a two-block join at 0.5")
    if tuple(_run.two_block_join()) != (0.0, 0.5, 1.0):
        raise ValueError("run policy must record the two-block join at 0.5")
    centers, field, volumes = _run.sample_placement("face", _exact.N_CELLS)
    errors = reference_errors(field, field, volumes)
    if errors.linf != 0.0:
        raise ValueError("in-memory exact vs exact Linf must be 0")
    if placements_agree(_exact.N_CELLS) != 0.0:
        raise ValueError("exact translation must give field-to-field Linf = 0")
    if int(np.argmax(field)) >= 0 and abs(float(centers[int(np.argmax(field))]) - 0.5) > (
        1.0 / float(centers.size)
    ):
        raise ValueError("face placement peak must sit on the two-block join")


def _summary() -> dict:
    _assert_placement_contract()
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


def write_tr04_report(output_dir) -> dict:
    """Check exact placement agreement and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
