"""TR-01 analysis: native-only orders, schema artifacts, Phase 8 visual_data."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.metrics import write_metrics
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")

CASE_ID = "TR-01"
ORDER_THRESHOLD = 1.8
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}
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


class NativeSeriesError(ValueError):
    """Raised when an order claim lacks a real native series."""


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


def analyze_series(errors, resolutions, output_dir) -> dict:
    """Refuse injected error lists. Order claims need native fields."""
    raise NativeSeriesError(
        "refusing synthetic or injected error series; native fields required"
    )


def evaluate_order_claim(campaign: dict) -> dict[str, Any]:
    """Accept an order claim only from native fields at four or more resolutions."""
    if not isinstance(campaign, dict) or not campaign:
        raise NativeSeriesError("native series absent")
    source = campaign.get("source")
    if source != "native":
        raise NativeSeriesError(
            "refusing synthetic or injected series; native fields required"
        )
    resolutions = tuple(int(n) for n in (campaign.get("resolutions") or ()))
    fields = campaign.get("fields") or {}
    oracles = campaign.get("oracles") or {}
    volumes = campaign.get("volumes") or {}
    if not resolutions or not fields:
        raise NativeSeriesError("native series absent")
    missing = [
        n
        for n in resolutions
        if n not in fields or n not in oracles or n not in volumes
    ]
    if missing:
        raise NativeSeriesError("native series absent")
    family = str(campaign.get("family") or "")
    dt_scaling = str(campaign.get("dt_scaling") or "")
    if family == "spatial" and dt_scaling == "cfl":
        raise NativeSeriesError(
            "constant-CFL series is global, not isolated spatial; "
            "refuse spatial + CFL as an order claim"
        )
    l1: list[float] = []
    l2: list[float] = []
    linf: list[float] = []
    for n_cells in resolutions:
        errors = reference_errors(fields[n_cells], oracles[n_cells], volumes[n_cells])
        l1.append(float(errors.l1))
        l2.append(float(errors.l2))
        linf.append(float(errors.linf))
    if len(resolutions) < 4:
        return {
            "verdict": "smoke",
            "order_pass": False,
            "orders": [],
            "l1": tuple(l1),
            "l2": tuple(l2),
            "linf": tuple(linf),
            "reason": "fewer than four resolutions; smoke/not-run, never an order pass",
        }
    if any(value <= 0.0 for value in linf):
        raise NativeSeriesError(
            "refusing exact-vs-exact or non-positive native errors as an order pass"
        )
    if family == "temporal":
        spacings = tuple(float(value) for value in (campaign.get("dts") or campaign.get("spacings") or ()))
        if len(spacings) != len(resolutions):
            raise NativeSeriesError("temporal series requires dts matching the native fields")
    else:
        spacings = tuple(1.0 / float(n) for n in resolutions)
    orders = [float(value) for value in observed_order(linf, spacings)]
    order_pass = all(value >= ORDER_THRESHOLD for value in orders)
    reason = None
    if not order_pass:
        reason = (
            f"observed L∞ orders {orders} below spatial_order_min {ORDER_THRESHOLD}"
        )
    return {
        "verdict": "pass" if order_pass else "fail",
        "order_pass": order_pass,
        "orders": orders,
        "l1": tuple(l1),
        "l2": tuple(l2),
        "linf": tuple(linf),
        "spacings": spacings,
        "family": family or "global",
        "reason": reason,
    }


def _summary(
    *,
    orders: list,
    order_reason: str | None,
    dimensions: list[int],
    spaces: list[str],
    cases_run: int,
    cases_passed: int,
    cases_failed: int,
    failures: list | None = None,
) -> dict:
    reasons = {
        "amr.*": "AMR variants are authoring_ok or required_failure; leaf order is not claimed here",
        "poisson.*": "Poisson not run in TR-01",
        "coupling.*": "coupling not run in TR-01",
        "parallel_invariance.*": "parallel invariance is IF-01",
        "performance.one_node": "performance not measured in TR-01",
        "performance.two_node": "performance not measured in TR-01",
    }
    if not orders:
        reasons["orders"] = order_reason or "native series absent"
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": list(dimensions),
        "execution_spaces": list(spaces),
        "coverage": {
            "components": ["transport"],
            "cases_planned": 1,
            "cases_run": int(cases_run),
            "cases_passed": int(cases_passed),
            "cases_failed": int(cases_failed),
            "cases_not_supported": 0,
            "not_tested": [],
        },
        "failures": list(failures or []),
        "orders": list(orders),
        "amr": dict(NULL_AMR),
        "poisson": dict(NULL_POISSON),
        "coupling": dict(NULL_COUPLING),
        "parallel_invariance": dict(NULL_PARALLEL),
        "performance": dict(NULL_PERFORMANCE),
        "not_applicable_reason": reasons,
        "artifacts": dict(ARTIFACTS),
    }


def write_tr01_report(output_dir, *, n_cells=16) -> dict:
    """Schema-valid not-run report. Exact-vs-exact is not an order pass."""
    del n_cells
    return write_verification_report(
        _summary(
            orders=[],
            order_reason="native series absent; exact-vs-exact is not a campaign pass",
            dimensions=[1, 2, 3],
            spaces=["KokkosSerial"],
            cases_run=0,
            cases_passed=0,
            cases_failed=0,
        ),
        output_dir,
    )


def _linecut(field, oracle) -> dict[str, list]:
    array = np.asarray(field, dtype=np.float64)
    exact = np.asarray(oracle, dtype=np.float64)
    if array.ndim == 1:
        sample = array
        reference = exact
    elif array.ndim == 2:
        mid = array.shape[0] // 2
        sample = array[mid, :]
        reference = exact[mid, :]
    else:
        mid_y = array.shape[1] // 2
        mid_z = array.shape[0] // 2
        sample = array[mid_z, mid_y, :]
        reference = exact[mid_z, mid_y, :]
    return {
        "numerical": [float(v) for v in np.ravel(sample)],
        "oracle": [float(v) for v in np.ravel(reference)],
        "error": [float(v) for v in np.ravel(sample - reference)],
    }


def _write_visual_data(output_dir: Path, campaign: dict, claim: dict) -> None:
    visual_dir = output_dir / "visual_data"
    visual_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [int(n) for n in campaign["resolutions"]]
    family = str(campaign.get("family") or "global")
    figure_kind = {
        "temporal": "temporal_convergence",
        "spatial": "spatial_convergence",
        "global": "global_convergence",
    }.get(family, "global_convergence")
    convergence = {
        "figure_id": "spatial_convergence",
        "kind": figure_kind,
        "family": family,
        "reconstruction": campaign.get("reconstruction"),
        "dt_scaling": campaign.get("dt_scaling"),
        "case_id": CASE_ID,
        "dimension": int(campaign.get("dimension") or 3),
        "label": campaign.get("label"),
        "resolutions": resolutions,
        "spacings": [1.0 / float(n) for n in resolutions],
        "l1": list(claim["l1"]),
        "l2": list(claim["l2"]),
        "linf": list(claim["linf"]),
        "orders": list(claim["orders"]),
        "threshold": ORDER_THRESHOLD,
    }
    (visual_dir / "spatial_convergence.json").write_text(
        json.dumps(convergence, indent=2) + "\n", encoding="utf-8"
    )
    with (visual_dir / "spatial_convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "h", "l1", "l2", "linf"])
        for n_cells, l1, l2, linf in zip(
            resolutions, claim["l1"], claim["l2"], claim["linf"], strict=True
        ):
            writer.writerow([n_cells, 1.0 / float(n_cells), l1, l2, linf])
    finest = resolutions[-1]
    comparison = {
        "figure_id": "reference_comparison",
        "kind": "reference_comparison",
        "n_cells": finest,
        "linecut": _linecut(campaign["fields"][finest], campaign["oracles"][finest]),
    }
    (visual_dir / "reference_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    diagnostics = campaign.get("diagnostics") or {}
    phase_rows = []
    for n_cells in resolutions:
        row = diagnostics.get(n_cells) or {}
        phase_rows.append(
            {
                "n_cells": int(n_cells),
                "phase_error": row.get("phase_error"),
                "amplitude_loss": row.get("amplitude_loss"),
                "mass_error": row.get("mass_error"),
            }
        )
    (visual_dir / "phase_amplitude.json").write_text(
        json.dumps(
            {
                "figure_id": "phase_amplitude",
                "kind": "phase_amplitude",
                "rows": phase_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "pops.verification.tr01_visual_data.v1",
        "case_id": CASE_ID,
        "run_id": output_dir.name,
        "repository_sha": _repository_sha(),
        "dimension": int(campaign.get("dimension") or 3),
        "label": campaign.get("label"),
        "figures": [
            {
                "figure_id": "spatial_convergence",
                "kind": "spatial_convergence",
                "data_file": "visual_data/spatial_convergence.json",
                "proves": (
                    "temporal L1/L2/Linf vs dt at fixed grid"
                    if family == "temporal"
                    else "L1/L2/Linf vs h at constant CFL (global, not isolated spatial)"
                    if family != "spatial"
                    else "L1/L2/Linf vs h with dt ∝ h² (isolated spatial)"
                ),
                "does_not_prove": "AMR leaf retention or a different family",
            },
            {
                "figure_id": "reference_comparison",
                "kind": "reference_comparison",
                "data_file": "visual_data/reference_comparison.json",
                "proves": "native vs cell-average oracle along one linecut",
                "does_not_prove": "full-volume 3-d publication figures",
            },
            {
                "figure_id": "phase_amplitude",
                "kind": "phase_amplitude",
                "data_file": "visual_data/phase_amplitude.json",
                "proves": "phase, amplitude, and mass diagnostics from native fields",
                "does_not_prove": "a time history unless multiple periods were run",
            },
        ],
    }
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _write_metrics(output_dir: Path, campaign: dict, claim: dict) -> None:
    finest_l1 = float(claim["l1"][-1]) if claim["l1"] else None
    finest_l2 = float(claim["l2"][-1]) if claim["l2"] else None
    finest_linf = float(claim["linf"][-1]) if claim["linf"] else None
    observed = float(claim["orders"][-1]) if claim["orders"] else None
    diagnostics = campaign.get("diagnostics") or {}
    finest = None
    if campaign.get("resolutions"):
        finest = int(campaign["resolutions"][-1])
    mass = None
    if finest is not None:
        mass = (diagnostics.get(finest) or {}).get("mass_error")
    reasons = {
        "conservation.momentum_total.*": "scalar advection has no momentum diagnostic",
        "conservation.energy_total.*": "scalar advection has no energy diagnostic",
        "conservation.charge_total.*": "no charge in TR-01",
        "conservation.electrostatic_energy.*": "no Poisson equation",
        "poisson.*": "no Poisson equation",
        "extrema.*": "no density or pressure in TR-01",
        "symmetry.error": "symmetry is not a TR-01 acceptance metric",
        "amr.*": "leaf-only AMR order is not claimed from this series",
        "timings_seconds.ghost_fill": "ghost fill not timed in TR-01",
        "timings_seconds.reflux": "no reflux timing in TR-01",
    }
    if observed is None:
        reasons["errors.q.observed_order"] = claim.get("reason") or "no order claim"
    if mass is None:
        reasons["conservation.mass_total.*"] = "mass diagnostic absent from this series"
    document = {
        "schema": "pops.verification.metrics.v1",
        "case_id": CASE_ID,
        "errors": {
            "q": {
                "l1": finest_l1,
                "l2": finest_l2,
                "linf": finest_linf,
                "observed_order": observed,
            }
        },
        "conservation": {
            "mass_total": {
                "initial": None if mass is None else float(_exact.Q0),
                "final": None if mass is None else float(_exact.Q0) + float(mass),
                "max_relative_drift": None if mass is None else abs(float(mass)),
            },
            "momentum_total": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
            "energy_total": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
            "charge_total": {
                "initial": None,
                "final": None,
                "max_absolute_drift": None,
            },
            "electrostatic_energy": {
                "initial": None,
                "final": None,
                "max_relative_drift": None,
            },
        },
        "poisson": {"residual_l2": None, "gauss_defect_l2": None},
        "extrema": {"rho_min": None, "p_min": None},
        "symmetry": {"error": None},
        "amr": {
            "interface_error": None,
            "bulk_error": None,
            "leaf_cells": None,
            "patch_count": None,
            "regrid_count": None,
        },
        "not_applicable_reason": reasons,
        "timings_seconds": {"ghost_fill": None, "poisson": 0, "reflux": None},
        "csv_files": ["visual_data/spatial_convergence.csv"],
    }
    write_metrics(output_dir / "metrics.json", document)


def write_native_campaign_report(output_dir, campaign: dict) -> dict:
    """Write report, metrics, and Phase 8 visual_data from a native campaign."""
    claim = evaluate_order_claim(campaign)
    family = str(campaign.get("family") or claim.get("family") or "global")
    kind = family if family in {"spatial", "temporal", "global"} else "global"
    orders = [
        {
            "case_id": CASE_ID,
            "kind": kind,
            "variable": "q",
            "observed_order": float(value),
            "threshold": ORDER_THRESHOLD,
        }
        for value in claim["orders"]
    ]
    failures = []
    if claim["verdict"] == "fail":
        failures = [
            {
                "case_id": CASE_ID,
                "reason": claim.get("reason")
                or f"observed order below spatial_order_min {ORDER_THRESHOLD}",
                "metrics_ref": "metrics.json",
                "provenance_ref": "provenance.json",
            }
        ]
    output = Path(output_dir)
    written = write_verification_report(
        _summary(
            orders=orders,
            order_reason=claim.get("reason"),
            dimensions=[int(campaign.get("dimension") or 3)],
            spaces=["KokkosSerial"],
            cases_run=1,
            cases_passed=1 if claim["order_pass"] else 0,
            cases_failed=0 if claim["order_pass"] else 1,
            failures=failures,
        ),
        output,
    )
    _write_metrics(output, campaign, claim)
    _write_visual_data(output, campaign, claim)
    return written
