"""Native-order campaign helper: manufactured L2 series and report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

CASE_ID = "NO-01"
ORDER_THRESHOLD = float(_exact.ORDER_THRESHOLD)
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
    "amr.*": "AMR not run in native-order in-memory path",
    "poisson.*": "Poisson not run in native-order in-memory path",
    "coupling.*": "coupling not run in native-order in-memory path",
    "parallel_invariance.*": "parallel invariance not run in native-order in-memory path",
    "performance.one_node": "performance not measured in native-order in-memory path",
    "performance.two_node": "performance not measured in native-order in-memory path",
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


def _order_rows(errors, resolutions) -> list[dict]:
    observed = observed_order(errors, resolutions)
    if any(float(value) < ORDER_THRESHOLD for value in observed):
        raise ValueError(
            f"observed_order must be >= {ORDER_THRESHOLD}, got {list(observed)}"
        )
    return [
        {
            "case_id": CASE_ID,
            "kind": "spatial",
            "variable": "l2",
            "observed_order": float(value),
            "threshold": ORDER_THRESHOLD,
        }
        for value in observed
    ]


def _summary(*, orders: list) -> dict:
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1, 2],
        "execution_spaces": ["KokkosSerial", "KokkosOpenMP"],
        "coverage": {
            "components": ["infrastructure"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [
                "live Serial/OpenMP × Dim1/Dim2 native compile is ROMEO-only",
            ],
        },
        "failures": [],
        "orders": list(orders),
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def analyze_series(errors, resolutions, output_dir) -> dict:
    """Write a campaign report from an already-computed L2 series."""
    return write_verification_report(
        _summary(orders=_order_rows(errors, resolutions)),
        output_dir,
    )


def write_native_order_report(output_dir) -> dict:
    """Gate manufactured L2 ∝ h² and write the four Task 20 artifacts."""
    series = _run.manufactured_series()
    return analyze_series(series["l2"], series["spacings"], output_dir)
