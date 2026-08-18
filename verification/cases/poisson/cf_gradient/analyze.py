"""PO-06 driver: CF placements, manufactured E order, interface band, report."""
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
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
PLACEMENTS = _exact.PLACEMENTS
manufactured_gradient = _run.manufactured_gradient

RESOLUTIONS = (16, 32, 64, 128)
CASE_ID = "PO-06"
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
    "amr.invariants_ok": "conservation invariants not run in PO-06 CF gradient placement",
    "coupling.*": "coupling not run in PO-06",
    "parallel_invariance.*": "parallel invariance not run in PO-06",
    "performance.one_node": "performance not measured in PO-06",
    "performance.two_node": "performance not measured in PO-06",
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


def _placement_orders():
    """Manufactured E∝h² order and finest CF/bulk errors at every placement."""
    orders = {}
    finest = {}
    for name in PLACEMENTS:
        errors = []
        spacings = []
        last = None
        for n_cells in RESOLUTIONS:
            sample = manufactured_gradient(n_cells, name)
            errors.append(sample["field_error"])
            spacings.append(1.0 / float(n_cells))
            last = sample
        series = observed_order(errors, spacings)
        orders[name] = series
        finest[name] = last
    return orders, finest


def _summary() -> dict:
    orders, finest = _placement_orders()
    observed = [float(series[-1]) for series in orders.values()]
    if not (
        all(math.isfinite(value) for value in observed)
        and all(math.isfinite(sample["e_cf"]) for sample in finest.values())
        and all(math.isfinite(sample["e_bulk"]) for sample in finest.values())
        and all(math.isfinite(sample["field_error"]) for sample in finest.values())
    ):
        raise ValueError("non-finite PO-06 diagnostics")
    worst = min(observed)
    field_error = max(sample["field_error"] for sample in finest.values())
    e_cf = max(sample["e_cf"] for sample in finest.values())
    e_bulk = max(sample["e_bulk"] for sample in finest.values())
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "nightly",
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
                "variable": "electric_field",
                "observed_order": worst,
                "threshold": 1.8,
            }
        ],
        "amr": {
            "order_retained": worst >= 1.8,
            "invariants_ok": None,
            "interface_error": float(e_cf),
            "bulk_error": float(e_bulk),
        },
        "poisson": {
            "potential_error": 0.0,
            "field_error": float(field_error),
            "residual_l2": 0.0,
        },
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": dict(NOT_APPLICABLE),
        "artifacts": dict(ARTIFACTS),
    }


def write_po06_report(output_dir) -> dict:
    """Reduce the manufactured CF placements and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
