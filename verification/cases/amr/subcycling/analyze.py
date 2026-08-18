"""AM-04 driver: subcycling clock contract, manufactured temporal order, report."""
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

CASE_ID = "AM-04"
ORDER_THRESHOLD = 1.8
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
    "amr.*": "spatial AMR interface/invariants not run in AM-04 in-memory path",
    "poisson.*": "Poisson not run in AM-04 in-memory path",
    "coupling.*": "coupling not run in AM-04 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-04",
    "performance.one_node": "performance not measured in AM-04",
    "performance.two_node": "performance not measured in AM-04",
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


def _assert_subcycling_contract() -> None:
    if tuple(_exact.RATIOS) != (1, 2, 4):
        raise ValueError("documented subcycling ratios must be (1, 2, 4)")
    if _exact.fine_steps_per_coarse(2) != 2:
        raise ValueError("ratio 2 must have 2 fine steps per coarse step")
    if _run.fine_steps_per_coarse(2) != 2:
        raise ValueError("run policy must record 2 fine steps per coarse at ratio 2")
    for ratio in _exact.RATIOS:
        expected = float(_exact.COARSE_DT) / float(ratio)
        if _exact.fine_dt(_exact.COARSE_DT, ratio) != expected:
            raise ValueError(f"fine dt must be coarse_dt / {ratio}")
        if _run.fine_dt(_exact.COARSE_DT, ratio) != expected:
            raise ValueError(f"run policy must record fine dt = coarse_dt / {ratio}")
        if _exact.fine_steps_per_coarse(ratio) != int(ratio):
            raise ValueError(f"ratio {ratio} must have {ratio} fine steps per coarse")


def _temporal_orders() -> list[dict]:
    dts, errors = _run.manufactured_error_series(
        coarse_dt=_exact.COARSE_DT,
        ratios=_exact.RATIOS,
    )
    observed = observed_order(errors, dts)
    return [
        {
            "case_id": CASE_ID,
            "kind": "temporal",
            "variable": "q",
            "observed_order": float(value),
            "threshold": ORDER_THRESHOLD,
        }
        for value in observed
    ]


def _summary() -> dict:
    _assert_subcycling_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["amr"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": _temporal_orders(),
        "amr": {
            "order_retained": None,
            "invariants_ok": None,
            "interface_error": None,
            "bulk_error": None,
        },
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_am04_report(output_dir) -> dict:
    """Check the in-memory subcycling contract and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
