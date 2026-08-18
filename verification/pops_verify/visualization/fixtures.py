"""Labeled deterministic fixture runs for Phase 8 tests.

These numbers are not live PoPS campaign results. Every written status
document sets data_kind=deterministic_fixture. The provenance SHA uses the
fixture: prefix so it cannot be mistaken for a git commit.
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
import math
from typing import Any

from verification.pops_verify.metrics import write_metrics
from verification.pops_verify.provenance import write_provenance
from verification.pops_verify.visualization.catalog import catalog_entry, visual_contract_for

FIXTURE_LABEL = "DETERMINISTIC FIXTURE — not a PoPS campaign result"
FIXTURE_SHA = "fixture:pops-visuals-v1"
AM01_EVENTS = (
    "before_entry",
    "entry",
    "inside_fine",
    "exit",
    "periodic_crossing",
    "final",
)

_PROVENANCE = {
    "schema": "pops.verification.provenance.v1",
    "case_id": "TR-01",
    "repository": "wolf75222/PoPS",
    "repository_sha": FIXTURE_SHA,
    "repository_dirty": False,
    "pops_version": "1.0.0",
    "component_catalog_digest": (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "native_header_signature": (
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "native_variant_manifest_digest": (
        "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "doctor_ok": True,
    "date_utc": "2026-08-17T00:00:00Z",
    "compiler": "GCC 13.x",
    "build_type": "Release",
    "precision": "float64",
    "pops_native_dim": 1,
    "kokkos_execution_space": "Serial",
    "mpi_enabled": False,
    "mpi_library": "none",
    "mpi_thread_level_requested": "MPI_THREAD_SINGLE",
    "mpi_thread_level_provided": "MPI_THREAD_SINGLE",
    "hdf5_collective_enabled": False,
    "nodes": 1,
    "mpi_ranks": 1,
    "omp_threads_per_rank": 1,
    "gpus": 0,
    "hostname": "fixture-host",
    "slurm_job_id": "fixture",
    "dimension": 1,
    "resolution": [32],
    "block_size": [16],
    "amr_total_levels": 1,
    "refinement_ratio": 2,
    "subcycling": False,
    "time_program": "SSPRK2",
    "cfl": 0.4,
    "final_time": 1.0,
}

_METRICS = {
    "schema": "pops.verification.metrics.v1",
    "case_id": "TR-01",
    "errors": {
        "scalar": {
            "l1": 0.002,
            "l2": 0.001,
            "linf": 0.003,
            "observed_order": 2.0,
        }
    },
    "conservation": {
        "mass_total": {
            "initial": 1.0,
            "final": 1.0,
            "max_relative_drift": 1.0e-12,
        },
        "momentum_total": {
            "initial": [0.0],
            "final": [0.0],
            "max_relative_drift": 0.0,
        },
        "energy_total": {
            "initial": 1.0,
            "final": 1.0,
            "max_relative_drift": 1.0e-12,
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
        "bulk_error": 0.001,
        "leaf_cells": 32,
        "patch_count": 1,
        "regrid_count": 0,
    },
    "not_applicable_reason": {
        "conservation.charge_total.*": "fixture transport/no-charge model",
        "conservation.electrostatic_energy.*": "fixture has no Poisson solve",
        "poisson.*": "fixture has no Poisson solve",
        "extrema.*": "scalar fixture has no Euler extrema",
        "symmetry.error": "fixture does not claim a symmetry metric",
        "amr.interface_error": "uniform fixture grid",
    },
    "timings_seconds": {"ghost_fill": 0.0, "poisson": 0, "reflux": 0.0},
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _linspace(count: int, start: float = 0.0, stop: float = 1.0) -> list[float]:
    if count == 1:
        return [start]
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


def _order2_errors(scale: float) -> list[float]:
    return [scale * (16.0 / resolution) ** 2 for resolution in (16, 32, 64, 128)]


def _convergence_payload(figure_id: str, kind: str) -> dict[str, Any]:
    xs = [16, 32, 64, 128]
    return {
        "figure_id": figure_id,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "units": {"x": "1/h" if "temporal" not in kind else "1/dt", "y": "L2 error"},
        "variables": ["scalar"],
        "series": [
            {"name": "L1", "x": xs, "y": _order2_errors(0.032), "unit": "1"},
            {"name": "L2", "x": xs, "y": _order2_errors(0.016), "unit": "1"},
            {"name": "Linf", "x": xs, "y": _order2_errors(0.048), "unit": "1"},
        ],
        "reference_slopes": [{"order": 2, "anchor": [16, 0.016]}],
        "title": f"{kind} (fixture)",
    }


def _signed_error_series() -> dict[str, Any]:
    xs = _linspace(9)
    return {
        "name": "error",
        "x": xs,
        "y": [0.02 * math.sin(4.0 * math.pi * x) for x in xs],
        "unit": "1",
    }


def _profile_payload(figure_id: str, kind: str) -> dict[str, Any]:
    xs = _linspace(9)
    exact = {
        "name": "exact",
        "x": xs,
        "y": [math.sin(2.0 * math.pi * x) for x in xs],
        "unit": "1",
    }
    error = _signed_error_series()
    numerical = {
        "name": "numerical",
        "x": xs,
        "y": [base + delta for base, delta in zip(exact["y"], error["y"], strict=True)],
        "unit": "1",
    }
    series = [error] if kind == "signed_error_profile" else [exact, numerical, error]
    ylabel = "signed error" if kind == "signed_error_profile" else "scalar"
    return {
        "figure_id": figure_id,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "units": {"x": "x / L", "y": ylabel},
        "variables": ["scalar"],
        "series": series,
        "title": f"{kind} (fixture)",
    }


def _time_payload(figure_id: str, kind: str) -> dict[str, Any]:
    times = _linspace(8, 0.0, 1.0)
    if kind == "phase_amplitude":
        series = [
            {"name": "phase", "x": times, "y": [0.1 * t for t in times], "unit": "rad"},
            {
                "name": "amplitude",
                "x": times,
                "y": [1.0 - 0.05 * t for t in times],
                "unit": "1",
            },
        ]
        ylabel = "phase / amplitude"
    elif kind == "frequency_spectrum":
        freqs = [1.0, 2.0, 3.0, 4.0]
        series = [{"name": "power", "x": freqs, "y": [1.0, 0.2, 0.05, 0.01], "unit": "1"}]
        ylabel = "power"
    else:
        series = [
            {
                "name": kind,
                "x": times,
                "y": [0.01 * math.exp(-2.0 * t) for t in times],
                "unit": "1",
            }
        ]
        ylabel = kind
    return {
        "figure_id": figure_id,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "units": {"x": "t / T" if kind != "frequency_spectrum" else "mode", "y": ylabel},
        "variables": [kind],
        "series": series,
        "title": f"{kind} (fixture)",
    }


def _exact_field(nx: int = 7) -> tuple[list[float], list[float], list[list[float]]]:
    xs = _linspace(nx)
    ys = _linspace(nx)
    field = [
        [math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y) for x in xs] for y in ys
    ]
    return xs, ys, field


def _numerical_and_error(
    exact: list[list[float]], xs: list[float], ys: list[float]
) -> tuple[list[list[float]], list[list[float]]]:
    numerical = []
    error = []
    for iy, y in enumerate(ys):
        nrow = []
        erow = []
        for ix, x in enumerate(xs):
            delta = 0.03 * math.sin(4.0 * math.pi * x) * math.sin(2.0 * math.pi * y)
            numerical_value = exact[iy][ix] + delta
            nrow.append(numerical_value)
            erow.append(numerical_value - exact[iy][ix])
        numerical.append(nrow)
        error.append(erow)
    return numerical, error


def _field_payload(kind: str, field, xs, ys) -> dict[str, Any]:
    return {
        "figure_id": kind,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "units": {"x": "x / L", "y": "y / L", "field": kind},
        "variables": [kind],
        "x": xs,
        "y": ys,
        "field": field,
        "title": f"{kind} (fixture)",
    }


def _volume(n: int = 8) -> tuple[list[float], list[list[list[float]]]]:
    xs = _linspace(n)
    field = []
    for z in xs:
        plane = []
        for y in xs:
            row = []
            for x in xs:
                row.append(
                    math.sin(2.0 * math.pi * x)
                    + 0.5 * math.sin(4.0 * math.pi * y)
                    + 0.25 * math.sin(6.0 * math.pi * z)
                )
            plane.append(row)
        field.append(plane)
    return xs, field


def _slice_from_volume(kind: str, xs: list[float], volume: list[list[list[float]]]) -> dict[str, Any]:
    mid = len(xs) // 2
    if kind == "slice_xy":
        field = volume[mid]
        xlabel, ylabel = "x / L", "y / L"
    elif kind == "slice_xz":
        field = [[volume[iz][mid][ix] for ix in range(len(xs))] for iz in range(len(xs))]
        xlabel, ylabel = "x / L", "z / L"
    else:
        field = [[volume[iz][iy][mid] for iy in range(len(xs))] for iz in range(len(xs))]
        xlabel, ylabel = "y / L", "z / L"
    return {
        "figure_id": kind,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "units": {"x": xlabel, "y": ylabel, "field": kind},
        "variables": [kind],
        "x": xs,
        "y": xs,
        "field": field,
        "plane": kind[-2:],
        "title": f"{kind} (fixture)",
    }


def _pulse(center: float) -> dict[str, Any]:
    xs = _linspace(17)
    return {
        "name": "scalar",
        "x": xs,
        "y": [math.exp(-(((x - center) / 0.08) ** 2)) for x in xs],
        "unit": "1",
    }


def _storyboard_payload(case_id: str) -> dict[str, Any]:
    events = AM01_EVENTS
    if case_id == "AM-01":
        events = tuple(visual_contract_for(case_id)["animation"]["key_events"])
    frames = []
    for index, event in enumerate(events):
        time = index / max(len(events) - 1, 1)
        frames.append(
            {
                "event": event,
                "time": time,
                "step": 10 * index,
                "series": [_pulse(0.1 + 0.8 * time)],
            }
        )
    return {
        "figure_id": "storyboard",
        "kind": "storyboard",
        "data_kind": "deterministic_fixture",
        "units": {"x": "x / L", "y": "scalar"},
        "variables": ["scalar"],
        "frames": frames,
        "title": "storyboard (fixture)",
    }


def _animation_payload() -> dict[str, Any]:
    xs = _linspace(7)
    ys = _linspace(7)
    frames = []
    for index in range(6):
        time = index / 5.0
        phase = index * math.pi / 3.0
        field = [
            [math.sin(2.0 * math.pi * x + phase) * math.cos(2.0 * math.pi * y) for x in xs]
            for y in ys
        ]
        frames.append({"time": time, "step": index, "x": xs, "y": ys, "field": field})
    return {
        "figure_id": "animation",
        "kind": "animation",
        "data_kind": "deterministic_fixture",
        "units": {"x": "x / L", "y": "y / L", "field": "scalar"},
        "variables": ["scalar"],
        "frames": frames,
        "color_limits": [-1.0, 1.0],
        "periodic": True,
        "title": "animation (fixture)",
    }


def _report_figure_payload() -> dict[str, Any]:
    conv = _convergence_payload("report_figure", "spatial_convergence")
    profile = _profile_payload("report_figure", "reference_profile")
    return {
        "figure_id": "report_figure",
        "kind": "report_figure",
        "data_kind": "deterministic_fixture",
        "units": {"x": "1/h", "y": "L2 error"},
        "variables": ["scalar"],
        "panels": [
            {
                "type": "convergence",
                "name": "L2",
                "x": conv["series"][1]["x"],
                "y": conv["series"][1]["y"],
                "xlabel": "1/h",
                "ylabel": "L2 error",
                "title": "convergence",
            },
            {
                "type": "profile",
                "name": "exact",
                "x": profile["series"][0]["x"],
                "y": profile["series"][0]["y"],
                "xlabel": "x / L",
                "ylabel": "scalar",
                "title": "profile",
            },
        ],
        "reference_slopes": conv["reference_slopes"],
        "series": conv["series"],
        "title": "report figure (fixture)",
    }


def _payload_for(kind: str, *, case_id: str, dimension: int) -> dict[str, Any] | None:
    if kind in {"spatial_convergence", "temporal_convergence"}:
        return _convergence_payload(kind, kind)
    if kind == "coarse_fine_error":
        xs = [16, 32, 64, 128]
        return {
            "figure_id": kind,
            "kind": kind,
            "data_kind": "deterministic_fixture",
            "units": {"x": "1/h", "y": "L2 error"},
            "series": [
                {"name": "interface", "x": xs, "y": _order2_errors(0.04), "unit": "1"},
                {"name": "bulk", "x": xs, "y": _order2_errors(0.01), "unit": "1"},
            ],
            "reference_slopes": [{"order": 2, "anchor": [16, 0.04]}],
            "title": "coarse-fine error (fixture)",
        }
    if kind == "backend_parity":
        return {
            "figure_id": kind,
            "kind": kind,
            "data_kind": "deterministic_fixture",
            "units": {"x": "backend", "y": "L2 error"},
            "backends": ["KokkosSerial", "KokkosOpenMP", "MPI"],
            "metrics": ["L2", "mass_drift"],
            "values": [[1.0e-3, 1.0e-12], [1.01e-3, 1.1e-12], [1.02e-3, 0.9e-12]],
            "title": "backend parity (fixture)",
        }
    if kind == "performance_breakdown":
        return {
            "figure_id": kind,
            "kind": kind,
            "data_kind": "deterministic_fixture",
            "units": {"x": "stage", "y": "s"},
            "stages": ["ghost_fill", "poisson", "reflux"],
            "seconds": [0.12, 0.44, 0.05],
            "title": "performance breakdown (fixture)",
        }
    if kind == "report_figure":
        return _report_figure_payload()
    if kind == "signed_error_profile":
        return _profile_payload(kind, kind)
    if kind in {
        "reference_profile",
        "reference_comparison",
        "linecuts",
        "linecut",
    }:
        if kind in {"linecut", "linecuts"} and dimension == 3:
            xs, volume = _volume()
            mid = len(xs) // 2
            return {
                "figure_id": kind,
                "kind": kind,
                "data_kind": "deterministic_fixture",
                "units": {"x": "x / L", "y": "scalar"},
                "series": [
                    {
                        "name": "linecut",
                        "x": xs,
                        "y": [volume[mid][mid][ix] for ix in range(len(xs))],
                        "unit": "1",
                    }
                ],
                "title": f"{kind} (fixture)",
            }
        return _profile_payload(kind, kind)
    if kind in {"invariants_vs_time", "phase_amplitude", "frequency_spectrum", "symmetry_metric"}:
        return _time_payload(kind, kind)
    if kind == "storyboard":
        return _storyboard_payload(case_id)
    if kind == "animation":
        return _animation_payload()
    if kind in {"slice_xy", "slice_xz", "slice_yz"}:
        if dimension != 3:
            return None
        xs, volume = _volume()
        return _slice_from_volume(kind, xs, volume)
    if kind == "amr_boxes":
        if dimension != 3:
            return None
        return {
            "figure_id": kind,
            "kind": kind,
            "data_kind": "deterministic_fixture",
            "units": {"x": "x / L", "y": "y / L", "z": "z / L"},
            "boxes": [
                {"level": 0, "lo": [0.0, 0.0, 0.0], "hi": [1.0, 1.0, 1.0]},
                {"level": 1, "lo": [0.25, 0.25, 0.35], "hi": [0.7, 0.65, 0.8]},
            ],
            "title": "AMR boxes (fixture)",
        }
    if kind == "isosurface":
        return None
    if kind in {
        "exact_field",
        "numerical_field",
        "signed_error_field",
        "absolute_error_field",
        "field_snapshot",
        "amr_patch_map",
        "hero_figure",
    }:
        if dimension == 3 and kind in {"amr_patch_map", "hero_figure"}:
            xs, volume = _volume()
            mid = _slice_from_volume("slice_xy", xs, volume)
            mid["figure_id"] = kind
            mid["kind"] = kind
            mid["title"] = f"{kind} xy projection (fixture)"
            return mid
        xs, ys, exact = _exact_field()
        numerical, error = _numerical_and_error(exact, xs, ys)
        if kind == "exact_field":
            return _field_payload(kind, exact, xs, ys)
        if kind == "numerical_field":
            return _field_payload(kind, numerical, xs, ys)
        if kind == "signed_error_field":
            return _field_payload(kind, error, xs, ys)
        if kind == "absolute_error_field":
            absolute = [[abs(value) for value in row] for row in error]
            return _field_payload(kind, absolute, xs, ys)
        if kind == "amr_patch_map":
            payload = _field_payload(kind, exact, xs, ys)
            payload["boxes"] = [
                {"level": 0, "lo": [0.0, 0.0], "hi": [1.0, 1.0]},
                {"level": 1, "lo": [0.3, 0.3], "hi": [0.7, 0.7]},
            ]
            return payload
        return _field_payload(kind, numerical, xs, ys)
    return None


def write_fixture_run(
    output_root: str | Path,
    case_id: str,
    *,
    dimension: int,
    verdict: str = "pass",
    run_id: str | None = None,
) -> Path:
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be 1, 2, or 3")
    entry = catalog_entry(case_id)
    run_name = run_id or f"fixture-{dimension}d"
    run_dir = Path(output_root) / case_id / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = copy.deepcopy(_METRICS)
    metrics["case_id"] = case_id
    provenance = copy.deepcopy(_PROVENANCE)
    provenance["case_id"] = case_id
    provenance["pops_native_dim"] = dimension
    provenance["dimension"] = dimension
    provenance["resolution"] = [32] * dimension
    provenance["block_size"] = [16] * dimension
    write_metrics(run_dir / "metrics.json", metrics)
    write_provenance(run_dir / "provenance.json", provenance)
    _write_json(
        run_dir / "status.json",
        {
            "case_id": case_id,
            "run_id": run_name,
            "verdict": verdict,
            "data_kind": "deterministic_fixture",
            "label": FIXTURE_LABEL,
            "title": entry.title,
        },
    )
    axis = {1: "1d", 2: "2d", 3: "3d"}[dimension]
    visual_dir = run_dir / "analysis" / "visual_data"
    if verdict in {"pass", "fail"}:
        for kind in entry.artifacts[axis]:
            payload = _payload_for(kind, case_id=case_id, dimension=dimension)
            if payload is not None:
                _write_json(visual_dir / f"{kind}.json", payload)
    return run_dir
