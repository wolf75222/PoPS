"""GE-02 driver: return-to-IC, quarter-turn peak, campaign report."""
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

N_CELLS = int(_exact.N_CELLS)
QUARTER_TURN_PEAK = (0.0, 0.5)
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
    "orders": "capability-gated polar runtime",
    "amr.*": "AMR not run for GE-02 in-memory path",
    "poisson.*": "Poisson not run for GE-02",
    "coupling.*": "coupling not run for GE-02",
    "parallel_invariance.*": "parallel invariance not run for GE-02",
    "performance.one_node": "performance not measured for GE-02 in-memory path",
    "performance.two_node": "performance not measured for GE-02 in-memory path",
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


def _assert_rotation_contract() -> None:
    if not math.isclose(_exact.PERIOD, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("documented period must be T=1")
    if not math.isclose(_exact.OMEGA, 2.0 * math.pi, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("documented angular speed must be Ω=2π")
    fields = _run.return_fields(N_CELLS)
    errors = reference_errors(fields["returned"], fields["initial"], fields["volumes"])
    if errors.l1 != 0.0 or errors.l2 != 0.0 or errors.linf != 0.0:
        raise ValueError("exact return must equal the IC with L1/L2/Linf = 0")
    if not (
        fields["returned"].shape == fields["initial"].shape
        and (fields["returned"] == fields["initial"]).all()
    ):
        raise ValueError("exact return field must be identical to the IC")
    peak = _exact.peak_location(_exact.PERIOD / 4.0)
    if not (
        math.isclose(float(peak[0]), QUARTER_TURN_PEAK[0], rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(float(peak[1]), QUARTER_TURN_PEAK[1], rel_tol=0.0, abs_tol=1.0e-15)
    ):
        raise ValueError("t=T/4 peak must sit at (0, 0.5)")
    quarter = _run.quarter_turn_fields(N_CELLS)
    field = np.asarray(quarter["field"], dtype=np.float64)
    index = np.unravel_index(int(np.argmax(field)), field.shape)
    width = float(quarter["width"])
    if not (
        math.isclose(float(quarter["x"][index]), QUARTER_TURN_PEAK[0], abs_tol=width)
        and math.isclose(float(quarter["y"][index]), QUARTER_TURN_PEAK[1], abs_tol=width)
    ):
        raise ValueError("t=T/4 grid peak must be near (0, 0.5)")
    reason = _run.refuse_public_polar_runtime()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("refuse_public_polar_runtime must return a documented reason")


def _summary() -> dict:
    _assert_rotation_contract()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["geometry"],
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


def write_ge02_report(output_dir) -> dict:
    """Check return-to-IC / quarter-turn and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
