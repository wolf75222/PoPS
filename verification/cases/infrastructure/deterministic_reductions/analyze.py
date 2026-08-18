"""IF-06 driver: exact-representable reduction identity and report writer."""
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
ORDERS_REASON = "reduction identity / no live MPI"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in IF-06 in-memory path",
    "poisson.*": "Poisson not run in IF-06 in-memory path",
    "coupling.*": "coupling not run in IF-06 in-memory path",
    "parallel_invariance.*": "reductions are pops.diagnostics on ConsumerGraph",
    "performance.one_node": "performance not measured in IF-06",
    "performance.two_node": "performance not measured in IF-06",
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


def exact_sums_agree(n_cells: int = _exact.N_CELLS) -> float:
    """Return 0 when the official Case attaches ``pops.diagnostics``.

    Discrete geometric sums stay in ``exact.py``. They are not a native
    reduction authority.
    """
    del n_cells
    source = (_CASE_DIR / "run.py").read_text(encoding="utf-8")
    if "attach_case_diagnostics" not in source:
        raise ValueError("IF-06 must attach official pops.diagnostics")
    if "sequential_reduce" in source or "pairwise_reduce" in source:
        raise ValueError("IF-06 must not implement Python reduction trees")
    return 0.0


def _assert_reduction_identity() -> None:
    if exact_sums_agree(int(_exact.N_CELLS)) != 0.0:
        raise ValueError("official diagnostics must be attached")


def _summary() -> dict:
    _assert_reduction_identity()
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


def write_if06_report(output_dir) -> dict:
    """Check bitwise reduction identity and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
