"""PF-10 driver: npz identity, no-output silence, campaign report writer."""
from __future__ import annotations

import subprocess
import sys
import tempfile
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
ORDERS_REASON = "npz checkpoint stand-in, not a timed PF run"
NOT_APPLICABLE = {
    "orders": ORDERS_REASON,
    "amr.*": "AMR not run in PF-10 npz stand-in path",
    "poisson.*": "Poisson not run in PF-10 npz stand-in path",
    "coupling.*": "coupling not run in PF-10 npz stand-in path",
    "parallel_invariance.*": "parallel invariance not run in PF-10",
    "performance.one_node": ORDERS_REASON,
    "performance.two_node": ORDERS_REASON,
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


def _assert_checkpoint_io() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dest = root / _exact.ARTIFACT_NAME
        result = _run.round_trip(dest)
        if not dest.is_file():
            raise ValueError("output path must write the npz artifact")
        if int(result["bytes"]) != dest.stat().st_size or int(result["bytes"]) <= 0:
            raise ValueError("output path must record a positive byte count")
        if float(result["write_time_s"]) != float(_exact.FAKE_WRITE_TIME_S):
            raise ValueError("output path must record the fake write-time observation")
        field = _exact.manufactured_field()
        errors = reference_errors(
            result["restored"], field, _exact.cell_volumes()
        )
        if errors.l1 != 0.0 or errors.l2 != 0.0 or errors.linf != 0.0:
            raise ValueError("npz round-trip L∞ must be 0")
        silent = root / "no_output" / _exact.ARTIFACT_NAME
        skipped = _run.run_no_output(silent)
        if silent.exists() or (silent.parent.exists() and any(silent.parent.iterdir())):
            raise ValueError("no-output path must not write the artifact")
        if skipped["wrote"] or int(skipped["bytes"]) != 0:
            raise ValueError("no-output path must record a skipped write")


def _summary() -> dict:
    _assert_checkpoint_io()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["performance"],
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


def write_pf10_report(output_dir) -> dict:
    """Check npz identity and no-output silence, then write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
