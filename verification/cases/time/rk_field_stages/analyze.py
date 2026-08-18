"""TM-07 driver: in-memory stage-count contract and campaign report writer."""
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
    "orders": "field-solve count contract; no temporal-order campaign in this increment",
    "amr.*": "AMR not run in TM-07 in-memory path",
    "poisson.*": "Poisson not run in TM-07 in-memory path",
    "coupling.*": "coupling residuals not measured in TM-07 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TM-07",
    "performance.one_node": "performance not measured in TM-07",
    "performance.two_node": "performance not measured in TM-07",
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


def _assert_field_solve_contract() -> None:
    if _exact.SSPRK2_STAGES != 2 or _exact.SSPRK3_STAGES != 3:
        raise ValueError("documented SSPRK2/SSPRK3 stage counts must be 2 and 3")
    if _exact.required_field_solves(_exact.SSPRK2_STAGES) != 2:
        raise ValueError("SSPRK2 must require 2 field solves per step")
    if _exact.required_field_solves(_exact.SSPRK3_STAGES) != 3:
        raise ValueError("SSPRK3 must require 3 field solves per step")
    if _run.field_solves_per_step("SSPRK2") != 2:
        raise ValueError("SSPRK2 per-stage policy must record 2 field solves")
    if _run.field_solves_per_step("SSPRK3") != 3:
        raise ValueError("SSPRK3 per-stage policy must record 3 field solves")
    if _exact.FROZEN_FIELD_SOLVES_PER_STEP != 1:
        raise ValueError("frozen-field negative control must be 1 solve per step")
    if _run.field_solves_per_step("SSPRK2", frozen_field=True) != 1:
        raise ValueError("frozen SSPRK2 must record 1 field solve per step")
    if _run.field_solves_per_step("SSPRK3", frozen_field=True) != 1:
        raise ValueError("frozen SSPRK3 must record 1 field solve per step")


def _summary() -> dict:
    _assert_field_solve_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["time"],
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


def write_tm07_report(output_dir) -> dict:
    """Check the in-memory field-solve contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
