"""Labeled deterministic fixture runs for Phase 8 tests and examples.

These numbers are not live PoPS campaign results. Every written status
document sets data_kind=deterministic_fixture.
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
import math
from typing import Any

from verification.pops_verify.metrics import write_metrics
from verification.pops_verify.provenance import write_provenance
from verification.pops_verify.visualization.catalog import catalog_entry

FIXTURE_LABEL = "DETERMINISTIC FIXTURE — not a PoPS campaign result"

_PROVENANCE = {
    "schema": "pops.verification.provenance.v1",
    "case_id": "TR-01",
    "repository": "wolf75222/PoPS",
    "repository_sha": "0123456789abcdef0123456789abcdef01234567",
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


def _sine_series(name: str, phase: float = 0.0, noise: float = 0.0) -> dict[str, Any]:
    xs = _linspace(9)
    return {
        "name": name,
        "x": xs,
        "y": [math.sin(2.0 * math.pi * x + phase) + noise for x in xs],
        "unit": "1",
    }


def _convergence_payload(figure_id: str, kind: str) -> dict[str, Any]:
    xs = [16, 32, 64, 128]
    return {
        "figure_id": figure_id,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "verdict": "pass",
        "units": {"x": "1/h" if "spatial" in kind else "dt", "y": "L2 error"},
        "variables": ["scalar"],
        "series": [
            {
                "name": "L1",
                "x": xs,
                "y": [0.032 / (2**i) for i in range(4)],
                "unit": "1",
            },
            {
                "name": "L2",
                "x": xs,
                "y": [0.016 / (2**i) for i in range(4)],
                "unit": "1",
            },
            {
                "name": "Linf",
                "x": xs,
                "y": [0.048 / (2**i) for i in range(4)],
                "unit": "1",
            },
        ],
        "reference_slopes": [{"order": 2, "anchor": [16, 0.016]}],
        "title": f"{kind} (fixture)",
    }


def _profile_payload(figure_id: str, kind: str) -> dict[str, Any]:
    exact = _sine_series("exact")
    numerical = _sine_series("numerical", noise=0.01)
    error = {
        "name": "error",
        "x": exact["x"],
        "y": [n - e for n, e in zip(numerical["y"], exact["y"], strict=True)],
        "unit": "1",
    }
    return {
        "figure_id": figure_id,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "verdict": "pass",
        "units": {"x": "x / L", "y": "scalar"},
        "variables": ["scalar"],
        "series": [exact, numerical, error],
        "title": f"{kind} (fixture)",
    }


def _time_payload(figure_id: str, kind: str) -> dict[str, Any]:
    times = _linspace(8, 0.0, 1.0)
    return {
        "figure_id": figure_id,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "verdict": "pass",
        "units": {"x": "t / T", "y": kind},
        "variables": [kind],
        "series": [
            {
                "name": kind,
                "x": times,
                "y": [0.01 * math.exp(-2.0 * t) for t in times],
                "unit": "1",
            }
        ],
        "title": f"{kind} (fixture)",
    }


def _field_grid(kind: str) -> dict[str, Any]:
    xs = _linspace(5)
    ys = _linspace(5)
    field = []
    for y in ys:
        row = []
        for x in xs:
            value = math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y)
            if "error" in kind:
                value *= 0.05
            row.append(value)
        field.append(row)
    return {
        "figure_id": kind,
        "kind": kind,
        "data_kind": "deterministic_fixture",
        "verdict": "pass",
        "units": {"x": "x / L", "y": "y / L", "field": kind},
        "variables": [kind],
        "x": xs,
        "y": ys,
        "field": field,
        "title": f"{kind} (fixture)",
    }


def _storyboard_payload() -> dict[str, Any]:
    frames = []
    for index, event in enumerate(("initial", "mid", "final")):
        payload = _profile_payload(f"storyboard_{event}", "reference_profile")
        frames.append(
            {
                "event": event,
                "time": 0.5 * index,
                "step": 10 * index,
                "series": payload["series"],
            }
        )
    return {
        "figure_id": "storyboard",
        "kind": "storyboard",
        "data_kind": "deterministic_fixture",
        "verdict": "pass",
        "units": {"x": "x / L", "y": "scalar"},
        "variables": ["scalar"],
        "frames": frames,
        "title": "storyboard (fixture)",
        "series": frames[0]["series"],
    }


def _animation_payload() -> dict[str, Any]:
    frames = []
    for index in range(6):
        field = _field_grid("field_snapshot")
        phase = index * math.pi / 3.0
        xs = field["x"]
        ys = field["y"]
        values = [
            [
                math.sin(2.0 * math.pi * x + phase) * math.cos(2.0 * math.pi * y)
                for x in xs
            ]
            for y in ys
        ]
        frames.append(
            {
                "time": index / 5.0,
                "step": index,
                "x": xs,
                "y": ys,
                "field": values,
            }
        )
    return {
        "figure_id": "animation",
        "kind": "animation",
        "data_kind": "deterministic_fixture",
        "verdict": "pass",
        "units": {"x": "x / L", "y": "y / L", "field": "scalar"},
        "variables": ["scalar"],
        "frames": frames,
        "color_limits": [-1.0, 1.0],
        "periodic": True,
        "title": "animation (fixture)",
        "series": _sine_series("frame0"),
    }


def _payload_for(kind: str) -> dict[str, Any]:
    if kind in {
        "spatial_convergence",
        "temporal_convergence",
        "report_figure",
        "backend_parity",
        "performance_breakdown",
        "coarse_fine_error",
    }:
        mapped = "temporal_convergence" if kind == "temporal_convergence" else "spatial_convergence"
        payload = _convergence_payload(kind, mapped)
        payload["kind"] = kind
        return payload
    if kind in {
        "reference_profile",
        "signed_error_profile",
        "reference_comparison",
        "linecuts",
        "linecut",
        "phase_amplitude",
        "invariants_vs_time",
        "frequency_spectrum",
        "symmetry_metric",
    }:
        if kind in {"invariants_vs_time", "phase_amplitude", "frequency_spectrum", "symmetry_metric"}:
            return _time_payload(kind, kind)
        return _profile_payload(kind, kind)
    if kind == "storyboard":
        return _storyboard_payload()
    if kind in {"animation", "hero_figure"}:
        payload = _animation_payload() if kind == "animation" else _field_grid("signed_error_field")
        payload["figure_id"] = kind
        payload["kind"] = kind
        if kind == "hero_figure":
            payload["series"] = [_sine_series("companion")]
        return payload
    return _field_grid(kind)


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
            _write_json(visual_dir / f"{kind}.json", _payload_for(kind))
    return run_dir
