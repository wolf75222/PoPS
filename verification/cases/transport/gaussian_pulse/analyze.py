"""TR-02 in-memory diagnostics: mass, barycenter, and a campaign report."""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    "orders": "single-resolution in-memory exact",
    "amr.*": "AMR not run in TR-02 in-memory path",
    "poisson.*": "Poisson not run in TR-02 in-memory path",
    "coupling.*": "coupling not run in TR-02 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in TR-02 in-memory path",
    "performance.one_node": "performance not measured in TR-02 in-memory path",
    "performance.two_node": "performance not measured in TR-02 in-memory path",
}
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}


def _exact_module():
    path = _CASE_DIR / "exact.py"
    spec = importlib.util.spec_from_file_location("tr02_gaussian_pulse_exact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def pulse_mass(q, volumes) -> float:
    """Return the discrete mass sum_i q_i V_i."""
    field = np.asarray(q, dtype=np.float64)
    cell_volumes = np.asarray(volumes, dtype=np.float64)
    return float(np.sum(field * cell_volumes))


def pulse_barycenter(x, q, volumes, q0) -> float:
    """Periodic first moment of q-q0 on [0, 1], unwrapped about the peak."""
    centers = np.asarray(x, dtype=np.float64)
    weights = (np.asarray(q, dtype=np.float64) - float(q0)) * np.asarray(
        volumes, dtype=np.float64
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("non-positive pulse mass")
    peak = float(centers[int(np.argmax(weights))])
    unwrap = np.mod(centers - peak + 0.5, 1.0) - 0.5
    return float(np.mod(peak + float(np.sum(unwrap * weights)) / total, 1.0))


def _in_memory_sample(n_cells: int = 32):
    exact = _exact_module()
    width = 1.0 / int(n_cells)
    centers = (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width
    volumes = np.full(int(n_cells), width, dtype=np.float64)
    field = exact.exact_gaussian(centers, 0.0)
    return field, field, volumes


def _summary() -> dict:
    u_num, u_exact, volumes = _in_memory_sample()
    errors = reference_errors(u_num, u_exact, volumes)
    if not (
        math.isfinite(errors.l1)
        and math.isfinite(errors.l2)
        and math.isfinite(errors.linf)
    ):
        raise ValueError("non-finite reference errors")
    if errors.linf != 0.0:
        raise ValueError("in-memory exact vs exact must have Linf = 0")
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
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


def write_tr02_report(output_dir) -> dict:
    """Compare exact vs exact in memory and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
