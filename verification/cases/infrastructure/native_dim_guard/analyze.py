"""IF-08 driver: native-dim refuse/emit contract and campaign report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.campaign import CampaignError, CampaignJob
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
ORDERS_REASON = "exact native-dim specialization / no live native artifact"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run for IF-08 in-memory path",
    "poisson.*": "Poisson not run for IF-08",
    "coupling.*": "coupling not run for IF-08",
    "parallel_invariance.*": "parallel invariance not run for IF-08",
    "performance.one_node": "performance not measured for IF-08 in-memory path",
    "performance.two_node": "performance not measured for IF-08 in-memory path",
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


def _assert_native_dim_guard() -> None:
    expected = [CampaignJob(case_id="TR-01", pops_native_dim=1)]
    if _exact.tr01_case()["id"] != "TR-01":
        raise ValueError("IF-08 planner case must be TR-01")
    if list(_exact.tr01_case()["native_dimensions"]) != [1]:
        raise ValueError("TR-01 native_dimensions must be [1]")
    matching = _run.plan_tr01_jobs([1], artifact_dim=1)
    if matching != expected:
        raise ValueError("matching dim=1 must emit TR-01")
    from_env = _run.plan_tr01_jobs(
        [1], artifact_dim=None, environ={"POPS_NATIVE_DIM": "1"}
    )
    if from_env != expected:
        raise ValueError("POPS_NATIVE_DIM=1 must emit TR-01")
    try:
        _run.plan_tr01_jobs([2], artifact_dim=1)
    except CampaignError as exc:
        message = str(exc)
        if "POPS_NATIVE_DIM" not in message or "fallback" not in message.lower():
            raise ValueError(
                "mismatch must name POPS_NATIVE_DIM and refuse fallback"
            ) from exc
    else:
        raise ValueError("requested dim 2 with artifact_dim 1 must be refused")
    _run.reset_fake_runs()
    try:
        _run.fake_run_tr01([2], artifact_dim=1)
    except CampaignError:
        if _run.fake_run_count() != 0:
            raise ValueError("mismatch must be refused before a fake run")
    else:
        raise ValueError("mismatch must be refused before a fake run")
    accepted = _run.fake_run_tr01([1], artifact_dim=1)
    if accepted["jobs"] != expected or accepted["ran"] is not True:
        raise ValueError("matching dim=1 fake run must emit TR-01")
    if _run.fake_run_count() != 1:
        raise ValueError("exactly one fake run may start after a matching plan")


def _summary() -> dict:
    _assert_native_dim_guard()
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


def write_if08_report(output_dir) -> dict:
    """Check the native-dim guard and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
