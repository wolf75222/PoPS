"""IF-05 driver: exact-field identity across dump cadences and report writer."""
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
ORDERS_REASON = "exact-field identity / no live output cadence"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in IF-05 in-memory path",
    "poisson.*": "Poisson not run in IF-05 in-memory path",
    "coupling.*": "coupling not run in IF-05 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in IF-05",
    "performance.one_node": "performance not measured in IF-05",
    "performance.two_node": "performance not measured in IF-05",
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


def cadences_agree(
    n_cells: int = _exact.DEFAULT_N_CELLS, t=_exact.T
) -> float:
    """Return field-to-field L∞ across dump cadences (must be 0)."""
    worst = _run.max_cadence_difference(n_cells, t)
    if worst != 0.0:
        raise ValueError("exact cadences must agree with field-to-field Linf = 0")
    return worst


def _assert_output_cadence_identity() -> None:
    if tuple(_exact.CADENCES) != (1, 2, 10):
        raise ValueError("canonical dump cadences must be 1, 2, 10")
    if float(_exact.T) != 0.25:
        raise ValueError("analytic advance must end at t=0.25")
    if int(_exact.N_STEPS) != 10:
        raise ValueError("documented step count must be 10")
    n_cells = int(_exact.DEFAULT_N_CELLS)
    for cadence in _exact.CADENCES:
        expected = _exact.expected_dumps(cadence)
        sample = _run.advance_cadence(cadence, n_cells)
        if len(sample["dumps"]) != expected:
            raise ValueError(f"dump count for cadence={cadence} must be N/k")
        if sample["t"] != _exact.T:
            raise ValueError("each cadence must finish at t=0.25")
    if cadences_agree(n_cells, t=_exact.T) != 0.0:
        raise ValueError("exact fields must be identical across dump cadences")
    leftover = _run.leftover_dump_linf()
    if leftover != float(_run.MUTATION_AMPLITUDE):
        raise ValueError("mutated leftover dump L∞ must equal the mutation amplitude")


def _summary() -> dict:
    _assert_output_cadence_identity()
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


def write_if05_report(output_dir) -> dict:
    """Check pairwise exact identity and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
