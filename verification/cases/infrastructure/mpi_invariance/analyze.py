"""IF-01 driver: exact-field identity across MPI-style splits and report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    "orders": "exact-field identity / no live MPI",
    "amr.*": "AMR not run in IF-01 in-memory path",
    "poisson.*": "Poisson not run in IF-01 in-memory path",
    "coupling.*": "coupling not run in IF-01 in-memory path",
    "parallel_invariance.*": "no live MPI; exact-field identity only",
    "performance.one_node": "performance not measured in IF-01",
    "performance.two_node": "performance not measured in IF-01",
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


def placements_agree(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return field-to-field L∞ across placements (must be 0)."""
    worst = _run.max_decomposition_difference(n_cells, t)
    if worst != 0.0:
        raise ValueError("exact placements must agree with field-to-field Linf = 0")
    return worst


def _assert_decomposition_identity() -> None:
    if tuple(_exact.PLACEMENTS) != ("one_block", "two_block", "1x4", "4x1"):
        raise ValueError("canonical placements must be one_block, two_block, 1x4, 4x1")
    if tuple(_exact.placement_edges("one_block")) != (0.0, 1.0):
        raise ValueError("one-block placement must cover [0, 1]")
    if tuple(_exact.placement_edges("two_block")) != (0.0, 0.5, 1.0):
        raise ValueError("two-block placement must be [0, 0.5] ∪ [0.5, 1]")
    if tuple(_exact.FOUR_BLOCK_SPLITS) != (0.25, 0.5, 0.75):
        raise ValueError("1×4 / 4×1 splits must be 0.25, 0.5, 0.75")
    if tuple(_exact.placement_edges("1x4")) != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("1×4 placement must split at 0.25 / 0.5 / 0.75")
    if tuple(_exact.placement_edges("4x1")) != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("4×1 placement must split at 0.25 / 0.5 / 0.75")
    n_cells = int(_exact.DEFAULT_N_CELLS)
    if placements_agree(n_cells, t=0.25) != 0.0:
        raise ValueError("exact fields must be identical across MPI-style decompositions")


def _summary() -> dict:
    _assert_decomposition_identity()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["infrastructure"],
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


def write_if01_report(output_dir) -> dict:
    """Check pairwise exact identity and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
