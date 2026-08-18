"""EU-06 driver: uniform-flow leftover contract and campaign report."""
from __future__ import annotations

import math
import subprocess
import sys
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

N_CELLS = int(_exact.N_CELLS)
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
    "orders": "machine-zero free-stream; no spatial-order campaign in this increment",
    "amr.*": "AMR not run for EU-06 in-memory path",
    "poisson.*": "Poisson not run for EU-06",
    "coupling.*": "coupling not run for EU-06",
    "parallel_invariance.*": "parallel invariance not run for EU-06",
    "performance.one_node": "performance not measured for EU-06 in-memory path",
    "performance.two_node": "performance not measured for EU-06 in-memory path",
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


def _assert_uniform_flow_contract() -> None:
    x, y = _exact.cell_centers(N_CELLS)
    volumes = _exact.cell_volumes(N_CELLS)
    primitives = _exact.exact_primitives(x, y, 0.0)
    later = _exact.exact_primitives(x, y, 1.0)
    for key in ("rho", "u", "v", "p"):
        if not _exact.is_spatially_constant(primitives[key]):
            raise ValueError(f"exact {key} must be spatially constant")
        if not (later[key] == primitives[key]).all():
            raise ValueError(f"exact {key} must be invariant in t")
    density = primitives["rho"]
    errors = reference_errors(density, density, volumes)
    if errors.linf != 0.0 or not (
        math.isfinite(errors.l1) and math.isfinite(errors.l2) and math.isfinite(errors.linf)
    ):
        raise ValueError("exact-vs-exact density L∞ must be 0")
    leftover = _run.one_cell_bump_leftover_linf(
        n_cells=N_CELLS, amplitude=_run.BUMP_AMPLITUDE
    )
    if leftover != float(_run.BUMP_AMPLITUDE):
        raise ValueError("1-cell leftover L∞ must equal the bump amplitude")


def _summary() -> dict:
    _assert_uniform_flow_contract()
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


def write_eu06_report(output_dir) -> dict:
    """Check the uniform-flow leftover contract and write four artifacts."""
    return write_verification_report(_summary(), output_dir)
