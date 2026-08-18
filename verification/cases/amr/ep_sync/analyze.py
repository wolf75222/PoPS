"""AM-11 driver: leaf-only Euler–Poisson charge and campaign report writer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

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
    "orders": "leaf-only charge contract; no spatial-order campaign in this increment",
    "amr.order_retained": "no order campaign in AM-11 in-memory path",
    "amr.interface_error": "interface error not measured in leaf-charge contract",
    "amr.bulk_error": "bulk error not measured in leaf-charge contract",
    "poisson.*": "elliptic solve not run in AM-11 in-memory path",
    "coupling.*": "coupling residuals not measured in AM-11 in-memory path",
    "parallel_invariance.*": "parallel invariance not run in AM-11",
    "performance.one_node": "performance not measured in AM-11",
    "performance.two_node": "performance not measured in AM-11",
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


def _assert_leaf_only_charge() -> None:
    hierarchy = _exact.two_level_hierarchy()
    rho = np.asarray(hierarchy["charge_density"], dtype=np.float64)
    volumes = np.asarray(hierarchy["volumes"], dtype=np.float64)
    leaf_mask = np.asarray(hierarchy["leaf_mask"])
    leaf = _exact.leaf_net_charge(rho, volumes, leaf_mask)
    parent = _exact.covered_parent_charge(rho, volumes, leaf_mask)
    naive = _exact.naive_net_charge(rho, volumes)
    composed = _run.compose_charge(rho, volumes, leaf_mask)
    restricted = _exact.restricted_parent_charge(hierarchy)
    if parent == 0.0:
        raise ValueError("covered parent charge must be nonzero so double-count is observable")
    if not np.isclose(naive, leaf + parent, rtol=0.0, atol=1e-15):
        raise ValueError("naive net charge must be leaf plus covered parent")
    if not np.isclose(composed, leaf, rtol=0.0, atol=0.0):
        raise ValueError("compose_charge must be leaf-only")
    if np.isclose(composed, naive, rtol=0.0, atol=0.0):
        raise ValueError("leaf-only charge must not equal the double-counted sum")
    if not np.isclose(parent, restricted, rtol=0.0, atol=1e-15):
        raise ValueError("covered parent must equal the restricted fine-patch charge")


def _summary() -> dict:
    _assert_leaf_only_charge()
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": [1],
        "execution_spaces": ["KokkosSerial"],
        "coverage": {
            "components": ["amr", "euler_poisson"],
            "cases_planned": 1,
            "cases_run": 1,
            "cases_passed": 1,
            "cases_failed": 0,
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": [],
        "orders": [],
        "amr": {
            "order_retained": None,
            "invariants_ok": True,
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


def write_am11_report(output_dir) -> dict:
    """Check leaf-only charge and write the four Task 20 artifacts."""
    return write_verification_report(_summary(), output_dir)
