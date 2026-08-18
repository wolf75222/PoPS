"""IF-04 driver: JSON checkpoint identity (TR-01 state) and report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    "orders": "in-memory checkpoint identity",
    "amr.*": "AMR not run for IF-04 in-memory path",
    "poisson.*": "Poisson not run for IF-04",
    "coupling.*": "coupling not run for IF-04",
    "parallel_invariance.*": "parallel invariance not run for IF-04",
    "performance.one_node": "performance not measured for IF-04 in-memory path",
    "performance.two_node": "performance not measured for IF-04 in-memory path",
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


def _assert_checkpoint_identity() -> None:
    state = _exact.manufactured_state()
    restored = _run.round_trip(state)
    if restored["t"] != state["t"]:
        raise ValueError("checkpoint time t must be preserved")
    volumes = _exact.cell_volumes()
    for key in ("centers", "q"):
        errors = reference_errors(restored[key], state[key], volumes)
        if errors.l1 != 0.0 or errors.l2 != 0.0 or errors.linf != 0.0:
            raise ValueError(f"JSON round-trip {key} L∞ must be 0")


def _summary() -> dict:
    _assert_checkpoint_identity()
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


def write_if04_report(output_dir) -> dict:
    """Check JSON checkpoint identity and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
