"""Phase 0 dummy driver: in-memory cosine, norms, and a campaign report."""
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
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
exact_sample = _exact.exact_sample
numerical_sample = _run.numerical_sample
from verification.pops_verify.report import write_verification_report

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
    "orders": "single-resolution dummy",
    "amr.*": "AMR not run in Phase 0 dummy",
    "poisson.*": "Poisson not run in Phase 0 dummy",
    "coupling.*": "coupling not run in Phase 0 dummy",
    "parallel_invariance.*": "parallel invariance not run in Phase 0 dummy",
    "performance.one_node": "performance not measured in Phase 0 dummy",
    "performance.two_node": "performance not measured in Phase 0 dummy",
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


def _summary() -> dict:
    u_exact, volumes = exact_sample()
    u_num, _ = numerical_sample()
    errors = reference_errors(u_num, u_exact, volumes)
    if not (
        math.isfinite(errors.l1)
        and math.isfinite(errors.l2)
        and math.isfinite(errors.linf)
    ):
        raise ValueError("non-finite reference errors")
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


def write_dummy_report(output_dir) -> dict:
    """Reduce the manufactured pair and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
