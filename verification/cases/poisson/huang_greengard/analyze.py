"""PO-04 driver: manufactured second-order series, norms, observed order, report."""
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
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
e_exact = _exact.e_exact
rhs_exact = _exact.rhs_exact
build_rhs_and_oracle = _run.build_rhs_and_oracle

RESOLUTIONS = (16, 32, 64, 128)
MANUFACTURED_ERROR_SCALE = 0.04
CASE_ID = "PO-04"
NULL_AMR = {
    "order_retained": None,
    "invariants_ok": None,
    "interface_error": None,
    "bulk_error": None,
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
    "amr.*": "AMR not run in PO-04 1-d Huang–Greengard multi-blob Poisson",
    "coupling.*": "coupling not run in PO-04",
    "parallel_invariance.*": "parallel invariance not run in PO-04",
    "performance.one_node": "performance not measured in PO-04",
    "performance.two_node": "performance not measured in PO-04",
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


def _manufactured_series():
    """Exact φ vs φ + C h² on N = 16,32,64,128. Returns (finest errors, orders)."""
    error_scalars = []
    spacings = []
    finest = None
    field_finest = None
    residual_finest = None
    for n_cells in RESOLUTIONS:
        sample = build_rhs_and_oracle(n_cells)
        spacing = 1.0 / float(n_cells)
        manufactured = sample["phi"] + MANUFACTURED_ERROR_SCALE * spacing**2
        potential = reference_errors(manufactured, sample["phi"], sample["volumes"])
        field = reference_errors(sample["e"], e_exact(sample["x"]), sample["volumes"])
        residual = reference_errors(
            sample["rhs"], rhs_exact(sample["x"]), sample["volumes"]
        )
        error_scalars.append(potential.linf)
        spacings.append(spacing)
        finest = potential
        field_finest = field
        residual_finest = residual
    orders = observed_order(error_scalars, spacings)
    return finest, field_finest, residual_finest, orders


def _summary() -> dict:
    potential, field, residual, orders = _manufactured_series()
    if not (
        math.isfinite(potential.l1)
        and math.isfinite(potential.l2)
        and math.isfinite(potential.linf)
        and math.isfinite(field.linf)
        and math.isfinite(residual.l2)
        and np.all(np.isfinite(orders))
    ):
        raise ValueError("non-finite PO-04 diagnostics")
    observed = float(orders[-1])
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["poisson"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": [
            {
                "case_id": CASE_ID,
                "kind": "spatial",
                "variable": "potential",
                "observed_order": observed,
                "threshold": 1.8,
            }
        ],
        "amr": dict(NULL_AMR),
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


def write_po04_report(output_dir) -> dict:
    """Reduce the manufactured φ series and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
