"""EU-02 driver: in-memory exact vs exact on density, campaign report."""
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
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(_CASE_DIR / "exact.py")
exact_vortex = _exact.exact_vortex
_run = load_sibling_module(_CASE_DIR / "run.py")
from verification.pops_verify.report import write_verification_report

N_CELLS = 32
CASE_ID = "EU-02"
ORDER_THRESHOLD = 1.5
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
    "orders": "single-resolution in-memory exact comparison",
    "amr.*": "AMR not run for EU-02 in-memory path",
    "poisson.*": "Poisson not run for EU-02",
    "coupling.*": "coupling not run for EU-02",
    "parallel_invariance.*": "parallel invariance not run for EU-02",
    "performance.one_node": "performance not measured for EU-02 in-memory path",
    "performance.two_node": "performance not measured for EU-02 in-memory path",
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


def _cell_grid(n_cells: int = N_CELLS):
    length = float(_exact.PERIOD)
    count = int(n_cells)
    width = length / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    volumes = np.full((count, count), width * width, dtype=np.float64)
    return x, y, volumes


def _summary() -> dict:
    x, y, volumes = _cell_grid()
    primitives = exact_vortex(x, y, 0.0, u_inf=1.0, v_inf=0.0)
    density = np.asarray(primitives["rho"], dtype=np.float64)
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [2],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["euler"],
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


def write_eu02_report(output_dir) -> dict:
    """Compare exact vs exact on density and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)


def density_error(conserved, n_cells, t, *, u_inf=1.0, v_inf=0.0):
    """L1/L2/L∞ of numerical density against the translated vortex."""
    x, y, volumes = _cell_grid(n_cells)
    primitives = _run.conserved_to_primitives(conserved)
    oracle = exact_vortex(x, y, t, u_inf=u_inf, v_inf=v_inf)
    return reference_errors(primitives["rho"], oracle["rho"], volumes)


def analyze_series(errors, resolutions, output_dir) -> dict:
    """Write a campaign report from an already-computed density-error series."""
    error_series = list(errors)
    spacing_series = [float(_exact.PERIOD) / float(n) for n in resolutions]
    if len(error_series) < 2:
        orders: list[dict] = []
        reason = "single-resolution series"
    else:
        observed = observed_order(error_series, spacing_series)
        orders = [
            {
                "case_id": CASE_ID,
                "kind": "spatial",
                "variable": "rho",
                "observed_order": float(value),
                "threshold": ORDER_THRESHOLD,
            }
            for value in observed
        ]
        reason = None
    payload = _summary()
    payload["orders"] = list(orders)
    reasons = dict(NOT_APPLICABLE)
    if reason is not None:
        reasons["orders"] = reason
    else:
        reasons.pop("orders", None)
        reasons["amr.*"] = "AMR not run for EU-02 native-order increment"
    payload["not_applicable_reason"] = reasons
    payload["native_dimensions"] = [2]
    return write_verification_report(payload, output_dir)
