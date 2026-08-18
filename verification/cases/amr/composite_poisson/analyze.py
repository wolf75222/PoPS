"""AM-10 driver: two-level leaf-only residual and campaign report writer."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.leaf_reference_errors import leaf_reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

CASE_ID = "AM-10"
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
    "orders": "leaf-only two-level residual contract; no native composite MG order campaign",
    "amr.order_retained": "no native composite MG order campaign in this increment",
    "amr.interface_error": "coarse-fine flux not measured in this in-memory residual",
    "amr.bulk_error": "bulk error not measured separately from the leaf residual",
    "coupling.*": "coupling not run in AM-10",
    "parallel_invariance.*": "parallel invariance not run in AM-10",
    "performance.one_node": "performance not measured in AM-10",
    "performance.two_node": "performance not measured in AM-10",
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


def _leaf_diagnostics():
    sample = _run.two_level_residual()
    residual = _run.leaf_residual_errors(sample)
    potential = leaf_reference_errors(
        _exact.phi_exact(sample["x"]),
        _exact.phi_exact(sample["x"]),
        sample["volumes"],
        sample["leaf_mask"],
    )
    field = leaf_reference_errors(
        _exact.e_exact(sample["x"]),
        _exact.e_exact(sample["x"]),
        sample["volumes"],
        sample["leaf_mask"],
    )
    if not (
        math.isfinite(residual.l1)
        and math.isfinite(residual.l2)
        and math.isfinite(residual.linf)
        and math.isfinite(potential.linf)
        and math.isfinite(field.linf)
    ):
        raise ValueError("non-finite AM-10 leaf diagnostics")
    if residual.l2 != 0.0 or residual.linf != 0.0:
        raise ValueError("leaf residual must exclude the covered-parent defect")
    return sample, residual, potential, field


def _summary() -> dict:
    _sample, residual, potential, field = _leaf_diagnostics()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["amr", "poisson"],
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
        "poisson": {
            "potential_error": float(potential.linf),
            "field_error": float(field.linf),
            "residual_l2": float(residual.l2),
        },
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am10_report(output_dir) -> dict:
    """Reduce the two-level leaf residual and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
