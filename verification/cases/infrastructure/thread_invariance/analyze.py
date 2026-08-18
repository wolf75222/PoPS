"""IF-02 driver: exact-field identity across thread labels and report writer."""
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
    "orders": "exact-field identity / no live OpenMP",
    "amr.*": "AMR not run in IF-02 in-memory path",
    "poisson.*": "Poisson not run in IF-02 in-memory path",
    "coupling.*": "coupling not run in IF-02 in-memory path",
    "parallel_invariance.*": "no live OpenMP; exact-field identity only; live Kokkos thread sweep is ROMEO-only",
    "performance.one_node": "performance not measured in IF-02",
    "performance.two_node": "performance not measured in IF-02",
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


def threads_agree(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return field-to-field L∞ across thread labels (must be 0)."""
    worst = _run.max_thread_difference(n_cells, t)
    if worst != 0.0:
        raise ValueError("exact thread labels must agree with field-to-field Linf = 0")
    return worst


def _assert_thread_identity() -> None:
    if tuple(_exact.THREAD_COUNTS) != (1, 2, 4, 8):
        raise ValueError("canonical thread counts must be 1, 2, 4, 8")
    n_cells = int(_exact.DEFAULT_N_CELLS)
    for n_threads in _exact.THREAD_COUNTS:
        slices = _exact.thread_slices(n_cells, n_threads)
        if len(slices) != n_threads:
            raise ValueError(f"thread count {n_threads} must yield {n_threads} slices")
        if slices[0][0] != 0 or slices[-1][1] != n_cells:
            raise ValueError(f"thread count {n_threads} must cover the full grid")
    if threads_agree(n_cells, t=0.25) != 0.0:
        raise ValueError("exact fields must be identical across OpenMP thread labels")


def _summary() -> dict:
    _assert_thread_identity()
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


def write_if02_report(output_dir) -> dict:
    """Check pairwise exact identity and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
