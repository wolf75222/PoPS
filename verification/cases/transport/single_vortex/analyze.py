"""TR-03 driver: return-to-IC identity and campaign report writer."""
from __future__ import annotations

import math
import subprocess
import sys
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

DIV_TOL = 1.0e-2
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
    "orders": "return-to-IC / no spatial-order campaign",
    "amr.*": "AMR not run in TR-03 in-memory path",
    "poisson.*": "Poisson not run in TR-03 in-memory path",
    "coupling.*": "coupling not run in TR-03 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TR-03",
    "performance.one_node": "performance not measured in TR-03",
    "performance.two_node": "performance not measured in TR-03",
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


def _assert_return_and_divergence() -> None:
    fields = _run.return_fields()
    errors = reference_errors(fields["returned"], fields["initial"], fields["volumes"])
    if errors.l1 != 0.0 or errors.l2 != 0.0 or errors.linf != 0.0:
        raise ValueError("exact return must equal the IC with L1/L2/Linf = 0")
    if not (
        fields["returned"].shape == fields["initial"].shape
        and (fields["returned"] == fields["initial"]).all()
    ):
        raise ValueError("exact return field must be identical to the IC")
    divergence = np.asarray(_run.manufactured_divergence(), dtype=np.float64)
    max_abs = float(np.max(np.abs(divergence)))
    if not math.isfinite(max_abs) or max_abs >= DIV_TOL:
        raise ValueError("manufactured discrete divergence must be ~0 at cell centres")


def _summary() -> dict:
    _assert_return_and_divergence()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
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


def write_tr03_report(output_dir) -> dict:
    """Check return-to-IC and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
