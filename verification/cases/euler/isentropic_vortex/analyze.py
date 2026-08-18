"""EU-02 analysis: cell-average oracles, honest order families, Phase 8 visuals."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.conservation import conservation_residual, conservation_tolerance
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.metrics import write_metrics
from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report
from verification.pops_verify.symmetry import radial_anisotropy, xy_symmetry_error

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")

CASE_ID = "EU-02"
ORDER_THRESHOLD = 1.8
ROUNDING_FLOOR = 1.0e-14
PRIMITIVE_VARS = ("rho", "u", "v", "p")
CONSERVED_VARS = ("rho", "rho_u", "rho_v", "E")
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


def _volumes(n_cells: int) -> np.ndarray:
    width = float(_exact.PERIOD) / float(n_cells)
    return np.full((int(n_cells), int(n_cells)), width * width, dtype=np.float64)


def primitive_errors(conserved, n_cells, t, *, u_inf=1.0, v_inf=0.0):
    """L1/L2/L∞ of primitives against analytic primitive cell averages."""
    primitives = _run.conserved_to_primitives(conserved)
    oracle = _run.average_primitives(n_cells, t, u_inf=u_inf, v_inf=v_inf)
    volumes = _volumes(n_cells)
    return {
        name: reference_errors(primitives[name], oracle[name], volumes)
        for name in PRIMITIVE_VARS
    }


def conserved_errors(conserved, n_cells, t, *, u_inf=1.0, v_inf=0.0):
    """L1/L2/L∞ of conserved fields against analytic conserved cell averages."""
    named = (
        _run.unpack_conserved(conserved, n_cells)
        if isinstance(conserved, np.ndarray)
        else conserved
    )
    oracle = _run.average_conserved(n_cells, t, u_inf=u_inf, v_inf=v_inf)
    volumes = _volumes(n_cells)
    return {
        name: reference_errors(named[name], oracle[name], volumes)
        for name in CONSERVED_VARS
    }


def density_error(conserved, n_cells, t, *, u_inf=1.0, v_inf=0.0):
    """L1/L2/L∞ of numerical density against cell-average vortex density.

    Oracle path: ``analytic_cell_averages`` of exact density, never point samples.
    """
    return primitive_errors(conserved, n_cells, t, u_inf=u_inf, v_inf=v_inf)["rho"]


def vorticity_from_velocity(velocity_x, velocity_y, width: float):
    """Periodic second-order vorticity ω = ∂v/∂x − ∂u/∂y on an xy mesh."""
    dv_dx = (
        np.roll(velocity_y, -1, axis=1) - np.roll(velocity_y, 1, axis=1)
    ) / (2.0 * float(width))
    du_dy = (
        np.roll(velocity_x, -1, axis=0) - np.roll(velocity_x, 1, axis=0)
    ) / (2.0 * float(width))
    return dv_dx - du_dy


def vortex_center_from_density(density, n_cells, *, expected=None):
    """Periodic centroid of the density deficit (ρ∞ − ρ)+."""
    rho = np.asarray(density, dtype=np.float64)
    x, y, _ = _run.cell_centers(n_cells)
    if expected is None:
        expected = (float(_exact.X0), float(_exact.Y0))
    dx = _exact.minimum_image(x - float(expected[0]))
    dy = _exact.minimum_image(y - float(expected[1]))
    weight = np.maximum(float(_exact.RHO_INF) - rho, 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        index = np.unravel_index(int(np.argmin(rho)), rho.shape)
        return (float(x[index]), float(y[index]))
    center_x = float(expected[0]) + float(np.sum(weight * dx) / total)
    center_y = float(expected[1]) + float(np.sum(weight * dy) / total)
    return (
        float(_exact.wrap_periodic(center_x)),
        float(_exact.wrap_periodic(center_y)),
    )


def center_error(numerical, analytic):
    """Minimum-image displacement of the numerical centre from the analytic one."""
    dx = float(_exact.minimum_image(float(numerical[0]) - float(analytic[0])))
    dy = float(_exact.minimum_image(float(numerical[1]) - float(analytic[1])))
    return {"dx": dx, "dy": dy, "distance": float(np.hypot(dx, dy))}


def conservation_integrals(conserved, n_cells):
    """Discrete volume integrals of native conserved fields."""
    named = (
        _run.unpack_conserved(conserved, n_cells)
        if isinstance(conserved, np.ndarray)
        else conserved
    )
    volumes = _volumes(n_cells)
    mass = float(np.sum(named["rho"] * volumes))
    momentum = [
        float(np.sum(named["rho_u"] * volumes)),
        float(np.sum(named["rho_v"] * volumes)),
    ]
    energy = float(np.sum(named["E"] * volumes))
    return {"mass": mass, "momentum": momentum, "energy": energy}


def conservation_drifts(initial, final, *, n_updates: int):
    """Periodic Euler residuals Q(t)−Q(0) with §8.4 tolerances."""
    residuals = {
        "mass": float(
            conservation_residual(final["mass"] - initial["mass"], 0.0, 0.0)
        ),
        "momentum_x": float(
            conservation_residual(
                final["momentum"][0] - initial["momentum"][0], 0.0, 0.0
            )
        ),
        "momentum_y": float(
            conservation_residual(
                final["momentum"][1] - initial["momentum"][1], 0.0, 0.0
            )
        ),
        "energy": float(
            conservation_residual(final["energy"] - initial["energy"], 0.0, 0.0)
        ),
    }
    scales = {
        "mass": abs(initial["mass"]),
        "momentum_x": max(abs(initial["momentum"][0]), 1.0),
        "momentum_y": max(abs(initial["momentum"][1]), 1.0),
        "energy": abs(initial["energy"]),
    }
    tolerances = {
        name: conservation_tolerance(
            scales[name], abs_tol=1.0e-12, rel_tol=1.0e-12, n_updates=n_updates
        )
        for name in residuals
    }
    relative = {
        name: abs(residuals[name]) / scales[name] if scales[name] > 0.0 else abs(residuals[name])
        for name in residuals
    }
    return {
        "residual": residuals,
        "relative": relative,
        "tolerance": tolerances,
        "ok": {name: abs(residuals[name]) <= tolerances[name] for name in residuals},
    }


def radial_symmetry(density, n_cells, center):
    """Radial anisotropy of a mid-deficit isocontour in the vortex frame."""
    rho = np.asarray(density, dtype=np.float64)
    x, y, _ = _run.cell_centers(n_cells)
    dx = _exact.minimum_image(x - float(center[0]))
    dy = _exact.minimum_image(y - float(center[1]))
    radius = np.sqrt(dx * dx + dy * dy)
    deficit = float(_exact.RHO_INF) - rho
    peak = float(np.max(deficit))
    if peak <= 0.0:
        raise NativeSeriesError("density deficit vanished; no vortex core")
    target = 0.5 * peak
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    radii = []
    for angle in angles:
        ray_x = float(center[0]) + np.cos(angle) * np.linspace(0.0, 0.5 * float(_exact.PERIOD), 64)
        ray_y = float(center[1]) + np.sin(angle) * np.linspace(0.0, 0.5 * float(_exact.PERIOD), 64)
        sample = _bilinear_periodic(rho, n_cells, ray_x, ray_y)
        sample_deficit = float(_exact.RHO_INF) - sample
        crossed = np.where(sample_deficit <= target)[0]
        if crossed.size == 0:
            radii.append(float(radius.max()))
        else:
            radii.append(float(np.hypot(
                _exact.minimum_image(ray_x[crossed[0]] - center[0]),
                _exact.minimum_image(ray_y[crossed[0]] - center[1]),
            )))
    shifted = _shift_to_center(deficit, n_cells, center)
    return {
        "radial_anisotropy": float(radial_anisotropy(radii)),
        "xy_symmetry": float(xy_symmetry_error(shifted)),
    }


def _bilinear_periodic(field, n_cells, query_x, query_y):
    width = float(_exact.PERIOD) / float(n_cells)
    fx = np.mod(np.asarray(query_x, dtype=np.float64) / width - 0.5, float(n_cells))
    fy = np.mod(np.asarray(query_y, dtype=np.float64) / width - 0.5, float(n_cells))
    i0 = np.floor(fx).astype(int) % int(n_cells)
    j0 = np.floor(fy).astype(int) % int(n_cells)
    i1 = (i0 + 1) % int(n_cells)
    j1 = (j0 + 1) % int(n_cells)
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    return (
        (1.0 - ty) * ((1.0 - tx) * field[j0, i0] + tx * field[j0, i1])
        + ty * ((1.0 - tx) * field[j1, i0] + tx * field[j1, i1])
    )


def _shift_to_center(field, n_cells, center):
    x, y, width = _run.cell_centers(n_cells)
    mid = 0.5 * float(_exact.PERIOD)
    shift_i = int(round((float(center[0]) - mid) / width))
    shift_j = int(round((float(center[1]) - mid) / width))
    return np.roll(np.roll(np.asarray(field, dtype=np.float64), -shift_i, axis=1), -shift_j, axis=0)


def evaluate_order_claim(campaign: dict) -> dict[str, Any]:
    """Accept an order claim only from native fields at four or more resolutions."""
    if not isinstance(campaign, dict) or not campaign:
        raise NativeSeriesError("native series absent")
    if campaign.get("source") != "native":
        raise NativeSeriesError(
            "refusing synthetic or injected series; native fields required"
        )
    resolutions = tuple(int(n) for n in (campaign.get("resolutions") or ()))
    fields = campaign.get("fields") or {}
    if not resolutions or not fields:
        raise NativeSeriesError("native series absent")
    missing = [n for n in resolutions if n not in fields]
    if missing:
        raise NativeSeriesError("native series absent")
    family = str(campaign.get("family") or "global")
    dt_scaling = str(campaign.get("dt_scaling") or "")
    if family == "spatial" and dt_scaling == "cfl":
        raise NativeSeriesError(
            "constant-CFL series is global, not isolated spatial; "
            "refuse spatial + CFL as an order claim"
        )
    t_end = float(campaign.get("t_end") or _run.T_END_CANONICAL)
    u_inf = float(campaign.get("u_inf") or 1.0)
    v_inf = float(campaign.get("v_inf") or 0.0)
    variable = str(campaign.get("variable") or "rho")
    runs = list(campaign.get("runs") or [])
    if family == "temporal":
        if not runs:
            raise NativeSeriesError("temporal series requires dts matching the native fields")
        samples = [(int(run["n_cells"]), run["field"]) for run in runs]
        spacings = tuple(float(run["dt"]) for run in runs)
        if any(not np.isfinite(value) or value <= 0.0 for value in spacings):
            raise NativeSeriesError("temporal series requires positive finite dts")
    else:
        samples = [(int(n_cells), fields[n_cells]) for n_cells in resolutions]
        spacings = tuple(float(_exact.PERIOD) / float(n) for n in resolutions)
        if family != "spatial":
            family = "global"
    l1: list[float] = []
    l2: list[float] = []
    linf: list[float] = []
    for n_cells, packed in samples:
        errors = primitive_errors(packed, n_cells, t_end, u_inf=u_inf, v_inf=v_inf)
        if variable not in errors:
            raise NativeSeriesError(f"unknown primitive variable {variable}")
        l1.append(float(errors[variable].l1))
        l2.append(float(errors[variable].l2))
        linf.append(float(errors[variable].linf))
    if len(samples) < 4:
        return {
            "verdict": "smoke",
            "order_pass": False,
            "orders": [],
            "gated_orders": [],
            "l1": tuple(l1),
            "l2": tuple(l2),
            "linf": tuple(linf),
            "family": family,
            "reason": "fewer than four resolutions; smoke/not-run, never an order pass",
        }
    if any(value <= 0.0 for value in linf):
        raise NativeSeriesError(
            "refusing exact-vs-exact or non-positive native errors as an order pass"
        )
    return _claim_from_recorded_errors(
        l1=l1, l2=l2, linf=linf, spacings=spacings, family=family
    )


def _claim_from_recorded_errors(*, l1, l2, linf, spacings, family: str = "global"):
    """§9.3: keep every interval as evidence; gate 1.8 on the last two usable L∞ pairs."""
    orders = [float(value) for value in observed_order(linf, spacings)]
    usable = [index for index, error in enumerate(linf) if float(error) > ROUNDING_FLOOR]
    usable_set = set(usable)
    interval_ids = [
        index
        for index in range(len(orders))
        if index in usable_set and (index + 1) in usable_set
    ]
    if len(interval_ids) < 2:
        return {
            "verdict": "fail",
            "order_pass": False,
            "orders": orders,
            "gated_orders": [],
            "l1": tuple(float(value) for value in l1),
            "l2": tuple(float(value) for value in l2),
            "linf": tuple(float(value) for value in linf),
            "spacings": tuple(float(value) for value in spacings),
            "family": family,
            "reason": (
                "insufficient finest L∞ intervals above rounding floor "
                f"{ROUNDING_FLOOR} to apply spatial_order_min {ORDER_THRESHOLD}"
            ),
        }
    gated_ids = interval_ids[-2:]
    gated = [orders[index] for index in gated_ids]
    order_pass = all(value >= ORDER_THRESHOLD for value in gated)
    reason = None
    if not order_pass:
        reason = (
            f"observed L∞ gated orders {gated} (last two intervals above rounding) "
            f"below spatial_order_min {ORDER_THRESHOLD}; all intervals {orders}"
        )
    return {
        "verdict": "pass" if order_pass else "fail",
        "order_pass": order_pass,
        "orders": orders,
        "gated_orders": gated,
        "l1": tuple(float(value) for value in l1),
        "l2": tuple(float(value) for value in l2),
        "linf": tuple(float(value) for value in linf),
        "spacings": tuple(float(value) for value in spacings),
        "family": family,
        "reason": reason,
    }


def campaign_from_evidence(series_dir) -> dict[str, Any]:
    """Build a native campaign mapping from an authenticated EvidenceBundle."""
    from verification.pops_verify.evidence_bundle import EvidenceBundle

    bundle = EvidenceBundle(series_dir)
    if bundle.case_id != CASE_ID:
        raise NativeSeriesError(f"EvidenceBundle case_id {bundle.case_id} is not EU-02")
    runs = []
    for record in bundle.records:
        result = np.asarray(record["result"], dtype=np.float64)
        n_cells = int(result.shape[-1])
        resolved = record.get("resolved_case") or {}
        job = resolved.get("job") if isinstance(resolved.get("job"), Mapping) else {}
        width = float(_exact.PERIOD) / float(n_cells)
        dt = job.get("dt")
        if dt is None:
            cfl = record["provenance"].get("cfl")
            dt = float(cfl) * width if cfl not in (None, "") else None
        runs.append(
            {
                "n_cells": n_cells,
                "field": result,
                "dt": float(dt) if dt is not None else None,
                "t": float(record["provenance"].get("final_time") or _run.T_END_CANONICAL),
            }
        )
    family_file = Path(series_dir) / "family.json"
    declared = {}
    if family_file.is_file():
        declared = json.loads(family_file.read_text(encoding="utf-8"))
    times = [run["t"] for run in runs]
    n_cells_set = {run["n_cells"] for run in runs}
    first = bundle.records[0]
    family = "global"
    dt_scaling = "cfl"
    time_program = str(first["provenance"].get("time_program") or "")
    if str(declared.get("family") or "") in {"temporal", "spatial", "global"}:
        family = str(declared["family"])
        dt_scaling = "fixed" if family == "temporal" else ("h2" if family == "spatial" else "cfl")
    elif "FixedDt" in time_program and len({round(t, 12) for t in times}) == 1:
        dts = [run["dt"] for run in runs]
        if len(n_cells_set) == 1 and all(value is not None for value in dts):
            family = "temporal"
            dt_scaling = "fixed"
        elif all(value not in (None, 0.0) for value in dts) and all(
            abs(dts[index] / dts[0] - (runs[0]["n_cells"] / runs[index]["n_cells"]) ** 2) < 0.15
            for index in range(len(dts))
        ):
            family = "spatial"
            dt_scaling = "h2"
    if family == "temporal":
        runs.sort(key=lambda run: float(run["dt"]))
    else:
        runs.sort(key=lambda run: int(run["n_cells"]))
    resolutions = [int(run["n_cells"]) for run in runs]
    fields = {int(run["n_cells"]): run["field"] for run in runs}
    return {
        "source": "native",
        "family": family,
        "dt_scaling": dt_scaling,
        "resolutions": resolutions,
        "fields": fields,
        "runs": runs,
        "dts": [run["dt"] for run in runs],
        "t_end": float(times[0]) if times else _run.T_END_CANONICAL,
        "u_inf": 1.0,
        "v_inf": 0.0,
        "variable": "rho",
        "dimension": 2,
        "label": first["provenance"].get("kokkos_execution_space"),
        "bundle_path": str(Path(series_dir).resolve()),
        "repository_sha": first["provenance"].get("repository_sha") or _repository_sha(),
        "provenance": dict(first["provenance"]),
    }


def _field_grid(n_cells: int):
    x, y, width = _run.cell_centers(n_cells)
    return (
        [float(value) for value in x[0, :]],
        [float(value) for value in y[:, 0]],
        width,
    )


def _tolist(array) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


def _write_visual_data(output_dir: Path, campaign: dict, claim: dict, extras: dict) -> None:
    visual_dir = output_dir / "analysis" / "visual_data"
    visual_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [int(n) for n in campaign["resolutions"]]
    family = str(claim.get("family") or campaign.get("family") or "global")
    t_end = float(campaign.get("t_end") or _run.T_END_CANONICAL)
    finest = resolutions[-1]
    packed = campaign["fields"][finest]
    primitives = _run.conserved_to_primitives(packed)
    oracle_p = _run.average_primitives(finest, t_end)
    x_axis, y_axis, width = _field_grid(finest)
    vorticity = vorticity_from_velocity(primitives["u"], primitives["v"], width)
    x, y, _ = _run.cell_centers(finest)
    vorticity_exact = _exact.exact_vorticity(x, y, t_end)
    shared_rho = [float(min(oracle_p["rho"].min(), primitives["rho"].min())), float(max(oracle_p["rho"].max(), primitives["rho"].max()))]
    shared_p = [float(min(oracle_p["p"].min(), primitives["p"].min())), float(max(oracle_p["p"].max(), primitives["p"].max()))]
    peak_w = float(max(abs(vorticity).max(), abs(vorticity_exact).max(), 1.0e-16))

    def _field_payload(kind: str, field, limits, units_field: str, title: str):
        return {
            "figure_id": kind,
            "kind": kind,
            "data_kind": "campaign",
            "x": x_axis,
            "y": y_axis,
            "field": _tolist(field),
            "units": {"x": "x", "y": "y", "field": units_field},
            "color_limits": limits,
            "times": [t_end],
            "step_numbers": [0],
            "title": title,
        }

    (visual_dir / "exact_field.json").write_text(
        json.dumps(_field_payload("exact_field", oracle_p["rho"], shared_rho, "rho", "Exact cell-average density"), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (visual_dir / "numerical_field.json").write_text(
        json.dumps(_field_payload("numerical_field", primitives["rho"], shared_rho, "rho", "Numerical density"), indent=2)
        + "\n",
        encoding="utf-8",
    )
    error_rho = primitives["rho"] - oracle_p["rho"]
    peak_e = float(max(abs(error_rho).max(), 1.0e-16))
    (visual_dir / "signed_error_field.json").write_text(
        json.dumps(
            _field_payload(
                "signed_error_field",
                error_rho,
                [-peak_e, peak_e],
                "rho error",
                "Signed density error",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (visual_dir / "exact_pressure.json").write_text(
        json.dumps(_field_payload("exact_field", oracle_p["p"], shared_p, "p", "Exact cell-average pressure"), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (visual_dir / "numerical_pressure.json").write_text(
        json.dumps(_field_payload("numerical_field", primitives["p"], shared_p, "p", "Numerical pressure"), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (visual_dir / "exact_vorticity.json").write_text(
        json.dumps(_field_payload("exact_field", vorticity_exact, [-peak_w, peak_w], "vorticity", "Exact vorticity"), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (visual_dir / "numerical_vorticity.json").write_text(
        json.dumps(_field_payload("numerical_field", vorticity, [-peak_w, peak_w], "vorticity", "Numerical vorticity"), indent=2)
        + "\n",
        encoding="utf-8",
    )
    mid = finest // 2
    r_axis = [float(value) for value in x[mid, :] - float(x[mid, mid])]
    linecuts = {
        "figure_id": "linecuts",
        "kind": "linecuts",
        "data_kind": "campaign",
        "units": {"x": "x - x_mid", "y": "rho"},
        "times": [t_end],
        "step_numbers": [0],
        "series": [
            {"name": "exact", "x": r_axis, "y": [float(v) for v in oracle_p["rho"][mid, :]]},
            {"name": "numerical", "x": r_axis, "y": [float(v) for v in primitives["rho"][mid, :]]},
        ],
        "title": "Density mid-line",
    }
    (visual_dir / "linecuts.json").write_text(json.dumps(linecuts, indent=2) + "\n", encoding="utf-8")
    radial = extras.get("radial_cut") or {}
    if radial:
        (visual_dir / "radial_cut.json").write_text(json.dumps(radial, indent=2) + "\n", encoding="utf-8")
    figure_kind = {
        "temporal": "temporal_convergence",
        "spatial": "spatial_convergence",
        "global": "spatial_convergence",
    }.get(family, "spatial_convergence")
    spacings = list(claim.get("spacings") or [float(_exact.PERIOD) / float(n) for n in resolutions])
    convergence = {
        "figure_id": "spatial_convergence",
        "kind": figure_kind,
        "data_kind": "campaign",
        "family": family,
        "case_id": CASE_ID,
        "units": {"x": "h", "y": "Linf(rho)"},
        "series": [
            {"name": "L1", "x": spacings, "y": list(claim["l1"])},
            {"name": "L2", "x": spacings, "y": list(claim["l2"])},
            {"name": "Linf", "x": spacings, "y": list(claim["linf"])},
        ],
        "reference_slopes": [{"order": 2.0, "name": "order 2"}],
        "resolutions": resolutions,
        "spacings": spacings,
        "l1": list(claim["l1"]),
        "l2": list(claim["l2"]),
        "linf": list(claim["linf"]),
        "orders": list(claim["orders"]),
        "threshold": ORDER_THRESHOLD,
        "title": f"{family} convergence (rho)",
        "times": [t_end],
        "step_numbers": [0],
    }
    (visual_dir / "spatial_convergence.json").write_text(
        json.dumps(convergence, indent=2) + "\n", encoding="utf-8"
    )
    with (visual_dir / "spatial_convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "h", "l1", "l2", "linf"])
        for n_cells, l1, l2, linf, spacing in zip(
            resolutions, claim["l1"], claim["l2"], claim["linf"], spacings, strict=True
        ):
            writer.writerow([n_cells, spacing, l1, l2, linf])
    invariants = extras.get("invariants") or {
        "figure_id": "invariants_vs_time",
        "kind": "invariants_vs_time",
        "data_kind": "campaign",
        "units": {"x": "t", "y": "relative drift"},
        "series": [
            {"name": "mass", "x": [0.0, t_end], "y": [0.0, float((extras.get("conservation") or {}).get("relative", {}).get("mass") or 0.0)]},
            {"name": "energy", "x": [0.0, t_end], "y": [0.0, float((extras.get("conservation") or {}).get("relative", {}).get("energy") or 0.0)]},
        ],
        "times": [0.0, t_end],
        "step_numbers": [0, 0],
        "title": "Conservation residuals",
    }
    (visual_dir / "invariants_vs_time.json").write_text(
        json.dumps(invariants, indent=2) + "\n", encoding="utf-8"
    )
    trajectory = extras.get("trajectory")
    if trajectory:
        (visual_dir / "vortex_center_trajectory.json").write_text(
            json.dumps(trajectory, indent=2) + "\n", encoding="utf-8"
        )
    storyboard = extras.get("storyboard") or {
        "figure_id": "storyboard",
        "kind": "storyboard",
        "data_kind": "campaign",
        "units": {"x": "x - x_mid", "y": "rho"},
        "times": [t_end],
        "step_numbers": [0],
        "frames": [
            {
                "event": "final",
                "time": t_end,
                "step": 0,
                "source": "accepted_state",
                "series": linecuts["series"],
            }
        ],
    }
    (visual_dir / "storyboard.json").write_text(json.dumps(storyboard, indent=2) + "\n", encoding="utf-8")
    animation = extras.get("animation") or {
        "figure_id": "animation",
        "kind": "animation",
        "data_kind": "campaign",
        "units": {"x": "x", "y": "y", "field": "rho"},
        "color_limits": shared_rho,
        "periodic": True,
        "times": [t_end],
        "step_numbers": [0],
        "frames": [
            {
                "time": t_end,
                "step": 0,
                "source": "accepted_state",
                "x": x_axis,
                "y": y_axis,
                "field": _tolist(primitives["rho"]),
            }
        ],
    }
    (visual_dir / "animation.json").write_text(json.dumps(animation, indent=2) + "\n", encoding="utf-8")
    hero = _field_payload("hero_figure", vorticity, [-peak_w, peak_w], "vorticity", "EU-02 vorticity")
    (visual_dir / "hero_figure.json").write_text(json.dumps(hero, indent=2) + "\n", encoding="utf-8")
    report_figure = {
        "figure_id": "report_figure",
        "kind": "report_figure",
        "data_kind": "campaign",
        "units": {"x": "h", "y": "Linf(rho)"},
        "times": [t_end],
        "step_numbers": [0],
        "title": "EU-02 vorticity and convergence",
        "panels": [
            {
                "type": "field",
                "name": "vorticity",
                "field": _tolist(vorticity),
            },
            {
                "type": "convergence",
                "name": "Linf",
                "x": spacings,
                "y": list(claim["linf"]),
                "xlabel": "h",
                "ylabel": "Linf(rho)",
            },
        ],
        "series": [{"name": "Linf", "x": spacings, "y": list(claim["linf"])}],
    }
    (visual_dir / "report_figure.json").write_text(
        json.dumps(report_figure, indent=2) + "\n", encoding="utf-8"
    )
    extras["shared_scales"] = {
        "rho": shared_rho,
        "p": shared_p,
        "vorticity": [-peak_w, peak_w],
        "density_error": [-peak_e, peak_e],
    }


def _write_status_and_program(output_dir: Path, campaign: dict, claim: dict) -> None:
    output = Path(output_dir)
    verdict = "pass" if claim.get("order_pass") else "fail"
    if claim.get("verdict") == "smoke":
        verdict = "not-run"
    (output / "status.json").write_text(
        json.dumps(
            {
                "case_id": CASE_ID,
                "run_id": output.name,
                "verdict": verdict,
                "data_kind": "campaign",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bundle_path = campaign.get("bundle_path")
    digest = ""
    provenance_src = None
    if bundle_path:
        finest = Path(bundle_path) / f"n{int(campaign['resolutions'][-1]):03d}" / "program.sha256"
        if finest.is_file():
            digest = finest.read_text(encoding="utf-8").strip()
            provenance_src = finest.parent / "provenance.json"
        if provenance_src is None or not provenance_src.is_file():
            for child in Path(bundle_path).rglob("provenance.json"):
                provenance_src = child
                break
        if provenance_src is not None and provenance_src.is_file():
            shutil.copy2(provenance_src, output / "provenance.json")
    (output / "program.json").write_text(
        json.dumps({"sha256": digest or "absent", "source": "program.bin"}, indent=2) + "\n",
        encoding="utf-8",
    )


def _finest_sample(campaign: dict) -> tuple[int, Any]:
    runs = list(campaign.get("runs") or [])
    if campaign.get("family") == "temporal" and runs:
        finest = min(runs, key=lambda run: float(run["dt"]))
        return int(finest["n_cells"]), finest["field"]
    finest = int(campaign["resolutions"][-1])
    return finest, campaign["fields"][finest]


def _write_metrics(output_dir: Path, campaign: dict, claim: dict, extras: dict) -> None:
    finest, packed = _finest_sample(campaign)
    primitives = _run.conserved_to_primitives(packed)
    primitive = extras.get("primitive_errors") or primitive_errors(
        packed, finest, float(campaign["t_end"])
    )
    conservation = extras.get("conservation") or {}
    symmetry = extras.get("symmetry") or {}
    observed = float(claim["orders"][-1]) if claim.get("orders") else None
    reasons = {
        "conservation.charge_total.*": "no charge in EU-02",
        "conservation.electrostatic_energy.*": "no Poisson equation",
        "poisson.*": "no Poisson equation",
        "amr.*": "this EU-02 series is uniform; AMR variants are not claimed",
        "timings_seconds.ghost_fill": "ghost fill not timed in EU-02",
        "timings_seconds.reflux": "no reflux timing in EU-02",
    }
    if observed is None:
        reasons["errors.*.observed_order"] = claim.get("reason") or "no order claim"
    relative = conservation.get("relative") or {}
    residual = conservation.get("residual") or {}
    initial = extras.get("initial_integrals") or {}
    final = extras.get("final_integrals") or {}
    document = {
        "schema": "pops.verification.metrics.v1",
        "case_id": CASE_ID,
        "errors": {
            name: {
                "l1": float(primitive[name].l1),
                "l2": float(primitive[name].l2),
                "linf": float(primitive[name].linf),
                "observed_order": observed if name == "rho" else None,
            }
            for name in PRIMITIVE_VARS
        },
        "conservation": {
            "mass_total": {
                "initial": initial.get("mass"),
                "final": final.get("mass"),
                "max_relative_drift": relative.get("mass"),
            },
            "momentum_total": {
                "initial": initial.get("momentum"),
                "final": final.get("momentum"),
                "max_relative_drift": max(
                    relative.get("momentum_x") or 0.0,
                    relative.get("momentum_y") or 0.0,
                )
                if relative
                else None,
            },
            "energy_total": {
                "initial": initial.get("energy"),
                "final": final.get("energy"),
                "max_relative_drift": relative.get("energy"),
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
        "extrema": {
            "rho_min": float(np.min(primitives["rho"])),
            "p_min": float(np.min(primitives["p"])),
        },
        "symmetry": {"error": symmetry.get("radial_anisotropy")},
        "amr": {
            "interface_error": None,
            "bulk_error": None,
            "leaf_cells": None,
            "patch_count": None,
            "regrid_count": None,
        },
        "not_applicable_reason": reasons,
        "timings_seconds": {"ghost_fill": None, "poisson": 0, "reflux": None},
        "csv_files": ["analysis/visual_data/spatial_convergence.csv"],
    }
    if document["errors"]["u"]["observed_order"] is None:
        reasons["errors.u.observed_order"] = "order gate is reported on rho; u/v/p orders live in the report tables"
        reasons["errors.v.observed_order"] = "order gate is reported on rho; u/v/p orders live in the report tables"
        reasons["errors.p.observed_order"] = "order gate is reported on rho; u/v/p orders live in the report tables"
    if residual:
        document["time_series"] = {"conservation_residual": residual}
    write_metrics(output_dir / "metrics.json", document)


def _summary(*, orders, order_reason, dimensions, spaces, cases_run, cases_passed, cases_failed, failures=None, repository_sha=None):
    reasons = {
        "amr.*": "this EU-02 series is uniform; AMR variants are not claimed",
        "poisson.*": "Poisson not run in EU-02",
        "coupling.*": "coupling not run in EU-02",
        "parallel_invariance.*": "OpenMP/MPI smokes are reported separately; not an invariance proof",
        "performance.one_node": "performance not measured in EU-02",
        "performance.two_node": "performance not measured in EU-02",
    }
    if not orders:
        reasons["orders"] = order_reason or "native series absent"
    return {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": repository_sha or _repository_sha(),
        "suite": "pr",
        "max_nodes": 2,
        "native_dimensions": list(dimensions),
        "execution_spaces": list(spaces),
        "coverage": {
            "components": ["euler"],
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
        "artifacts": {
            "report_md": "REPORT.md",
            "summary_json": "summary.json",
            "coverage_csv": "coverage.csv",
            "failures_csv": "failures.csv",
        },
    }


def diagnose_resolution(conserved, n_cells, t, *, u_inf=1.0, v_inf=0.0, n_updates=1):
    """Centre, vorticity, symmetry, and conservation at one native resolution."""
    primitives = _run.conserved_to_primitives(conserved)
    x, y, width = _run.cell_centers(n_cells)
    analytic = _exact.analytic_center(t, u_inf=u_inf, v_inf=v_inf)
    numerical = vortex_center_from_density(primitives["rho"], n_cells, expected=analytic)
    vorticity = vorticity_from_velocity(primitives["u"], primitives["v"], width)
    vorticity_exact = _exact.exact_vorticity(x, y, t, u_inf=u_inf, v_inf=v_inf)
    initial = conservation_integrals(
        _run.initial_conserved(n_cells, u_inf=u_inf, v_inf=v_inf), n_cells
    )
    final = conservation_integrals(conserved, n_cells)
    return {
        "primitive_errors": primitive_errors(conserved, n_cells, t, u_inf=u_inf, v_inf=v_inf),
        "conserved_errors": conserved_errors(conserved, n_cells, t, u_inf=u_inf, v_inf=v_inf),
        "center": {
            "analytic": analytic,
            "numerical": numerical,
            "error": center_error(numerical, analytic),
        },
        "vorticity_max": {
            "numerical": float(np.max(vorticity)),
            "exact": float(np.max(vorticity_exact)),
        },
        "symmetry": radial_symmetry(primitives["rho"], n_cells, numerical),
        "conservation": conservation_drifts(initial, final, n_updates=n_updates),
        "initial_integrals": initial,
        "final_integrals": final,
        "extrema": {
            "rho_min": float(np.min(primitives["rho"])),
            "p_min": float(np.min(primitives["p"])),
        },
    }


def write_native_campaign_report(output_dir, campaign: dict, extras: dict | None = None) -> dict:
    """Write report, metrics, and Phase 8 visual_data from a native campaign."""
    extras = dict(extras or {})
    claim = evaluate_order_claim(campaign)
    family = str(claim.get("family") or campaign.get("family") or "global")
    kind = family if family in {"spatial", "temporal", "global"} else "global"
    orders = [
        {
            "case_id": CASE_ID,
            "kind": kind,
            "variable": str(campaign.get("variable") or "rho"),
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
    finest, packed = _finest_sample(campaign)
    if "conservation" not in extras:
        extras.update(
            diagnose_resolution(
                packed,
                finest,
                float(campaign["t_end"]),
                u_inf=float(campaign.get("u_inf") or 1.0),
                v_inf=float(campaign.get("v_inf") or 0.0),
            )
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = write_verification_report(
        _summary(
            orders=orders,
            order_reason=claim.get("reason"),
            dimensions=[int(campaign.get("dimension") or 2)],
            spaces=[str(campaign.get("label") or "KokkosSerial")],
            cases_run=1,
            cases_passed=1 if claim["order_pass"] else 0,
            cases_failed=0 if claim["order_pass"] else 1,
            failures=failures,
            repository_sha=str(campaign.get("repository_sha") or _repository_sha()),
        ),
        output,
    )
    _write_metrics(output, campaign, claim, extras)
    _write_visual_data(output, campaign, claim, extras)
    _write_status_and_program(output, campaign, claim)
    report_path = output / "REPORT.md"
    extra = (
        "\n## Claims\n"
        f"family: {family} (constant-CFL AdaptiveCFL is global, never isolated spatial)\n"
        f"gated_orders: {claim.get('gated_orders')}\n"
        f"threshold: {ORDER_THRESHOLD} (never lowered)\n"
        f"centre error: {(extras.get('center') or {}).get('error')}\n"
        f"conservation relative: {(extras.get('conservation') or {}).get('relative')}\n"
        "Proves: native 2-d Euler vortex vs analytic cell-average oracles when the series exists.\n"
        "Does not prove: AMR, 3-d, isolated spatial unless family=spatial, temporal unless family=temporal.\n"
    )
    report_path.write_text(report_path.read_text(encoding="utf-8") + extra, encoding="utf-8")
    return written


def write_eu02_report(output_dir, native_series=None) -> dict:
    """Write artifacts from an EvidenceBundle series, or fail closed."""
    from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError

    if native_series is None or isinstance(native_series, dict):
        return write_verification_report(
            report_from_native_series(
                CASE_ID,
                None,
                native_dimensions=[2],
                components=["euler"],
                variable="rho",
                threshold=ORDER_THRESHOLD,
            ),
            output_dir,
        )
    try:
        campaign = campaign_from_evidence(native_series)
        written = write_native_campaign_report(output_dir, campaign)
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[2],
            components=["euler"],
            variable="rho",
            threshold=ORDER_THRESHOLD,
        )
        return written
    except (EvidenceError, NativeSeriesError, TypeError, ValueError):
        return write_verification_report(
            report_from_native_series(
                CASE_ID,
                None,
                native_dimensions=[2],
                components=["euler"],
                variable="rho",
                threshold=ORDER_THRESHOLD,
            ),
            output_dir,
        )
