#!/usr/bin/env python3
"""EU-02 v1.5 driver: official CampaignRequest / EvidenceBundle path.

Stages:
  series   — scripts/run_verification.py Serial/OpenMP four-resolution global
  temporal — FixedDt n=64, dt, dt/2, dt/4, dt/8
  spatial  — FixedDt Δt ∝ h² at 16/32/64/128
  smoke    — single n=16 CampaignRequest
  dump     — finest global run with accepted-state snapshots
  analyze  — EvidenceBundle → report / metrics / visual_data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verification.pops_verify.campaign import (  # noqa: E402
    CampaignJob,
    CampaignRequest,
    CampaignResources,
)
from verification.pops_verify.case_authoring import load_sibling_module  # noqa: E402
from verification.pops_verify.capabilities import authenticate_installed_artifact  # noqa: E402
from verification.pops_verify.evidence_contract import (  # noqa: E402
    emit_job_directory,
    write_series_json,
)
from verification.pops_verify.metrics import collect_metrics  # noqa: E402
from verification.pops_verify.native_evidence import (  # noqa: E402
    emission_from_payload,
    run_fields_from_payload,
)
from verification.pops_verify.provenance import collect_provenance  # noqa: E402

RUN = load_sibling_module(
    ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "run.py"
)
ANALYZE = load_sibling_module(
    ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "analyze.py"
)
TEMPORAL_DTS = (0.04, 0.02, 0.01, 0.005)
TEMPORAL_T_END = 0.32
DUMP_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _artifact(dimension: int = 2):
    return authenticate_installed_artifact(dimension=dimension)


def _job(space: str, mpi: str, n_cells: int, resolutions=(), ranks: int = 1, threads: int = 1):
    return CampaignJob(
        case_id="EU-02",
        pops_native_dim=2,
        suite="pr",
        execution_space=space,
        mpi_mode=mpi,
        min_resolution=n_cells,
        resources=CampaignResources(
            nodes=1,
            mpi_ranks=ranks,
            omp_threads=threads,
            resolutions=tuple(resolutions) if resolutions else (n_cells,),
        ),
    )


def _emit(job, request, payload, artifact, job_dir: Path, case: dict):
    fields = run_fields_from_payload(payload)
    emission = emission_from_payload(payload)
    import pops

    resolved = {
        "case": case,
        "job": {
            "case_id": job.case_id,
            "pops_native_dim": job.pops_native_dim,
            "suite": job.suite,
            "execution_space": job.execution_space,
            "mpi_mode": job.mpi_mode,
            "min_resolution": request.min_resolution,
            "evidence_status": job.evidence_status,
            "resources": {
                "nodes": job.resources.nodes,
                "mpi_ranks": job.resources.mpi_ranks,
                "omp_threads": job.resources.omp_threads,
                "resolutions": list(job.resources.resolutions),
            },
        },
        "status": "pass",
        "reason": "native run completed; scientific pass is not minted here",
    }
    if emission.get("dt") is not None:
        resolved["job"]["dt"] = float(emission["dt"])
    provenance = collect_provenance(
        job.case_id,
        pops_native_dim=2,
        dimension=2,
        nodes=1,
        pops_version=str(pops.__version__),
        doctor_ok=artifact.doctor_ok,
        component_catalog_digest=artifact.component_catalog_digest,
        native_header_signature=artifact.native_header_signature,
        native_variant_manifest_digest=artifact.native_variant_manifest_digest,
        **fields,
    )
    emit_job_directory(
        job_dir,
        resolved_case=resolved,
        provenance=provenance,
        metrics=collect_metrics(job.case_id, reason="native field recorded; analyze separately"),
        result=emission["result"],
        program_bytes=emission["program_bytes"],
        native_artifact={
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "dimension": int(artifact.dimension),
        },
    )


def _run_one(job, n_cells: int, output: Path, *, dt=None, family="global", dump_times=None, t_end=None):
    artifact = _artifact()
    job_dir = output / f"n{int(n_cells):03d}"
    request = CampaignRequest.from_job(job, output_dir=job_dir)
    kwargs = {
        "request": request,
        "n_cells": n_cells,
        "family": family,
    }
    if dt is not None:
        kwargs["dt"] = float(dt)
    if dump_times is not None:
        kwargs["dump_times"] = dump_times
    if t_end is not None:
        kwargs["t_end"] = float(t_end)
    payload = RUN.run_native(**kwargs)
    case = {"id": "EU-02", "path": "verification/cases/euler/isentropic_vortex/run.py"}
    _emit(job, request, payload, artifact, job_dir, case)
    return job_dir, payload


def stage_series(output: Path, space: str, mpi: str) -> int:
    from run_verification import main as verification_main

    argv = [
        "--suite",
        "pr",
        "--dimensions",
        "2",
        "--max-nodes",
        "1",
        "--output",
        str(output),
        "--pops-native-dim",
        "2",
        "--cases",
        "EU-02",
        "--execution-space",
        space,
        "--mpi-mode",
        mpi,
        "--execute",
    ]
    return int(verification_main(argv))


def stage_temporal(output: Path, space: str, mpi: str) -> int:
    job = _job(space, mpi, 64, resolutions=(64, 64, 64, 64))
    names = []
    for dt in TEMPORAL_DTS:
        name = f"dt{dt:g}".replace(".", "p")
        job_dir = output / name
        request = CampaignRequest.from_job(job, output_dir=job_dir)
        payload = RUN.run_native(
            request=request,
            n_cells=64,
            t_end=TEMPORAL_T_END,
            dt=float(dt),
            family="temporal",
        )
        artifact = _artifact()
        _emit(
            job,
            request,
            payload,
            artifact,
            job_dir,
            {"id": "EU-02", "path": "verification/cases/euler/isentropic_vortex/run.py"},
        )
        names.append(name)
    write_series_json(output, "EU-02", names)
    (output / "family.json").write_text(
        json.dumps({"family": "temporal", "dts": list(TEMPORAL_DTS), "t_end": TEMPORAL_T_END}, indent=2)
        + "\n"
    )
    return 0


def stage_spatial(output: Path, space: str, mpi: str) -> int:
    resolutions = (16, 32, 64, 128)
    job = _job(space, mpi, 16, resolutions=resolutions)
    names = []
    for n_cells in resolutions:
        _run_one(
            job,
            n_cells,
            output,
            family="spatial",
            dt=RUN.spatial_fixed_dt(n_cells),
            t_end=1.0,
        )
        names.append(f"n{int(n_cells):03d}")
    write_series_json(output, "EU-02", names)
    (output / "family.json").write_text(json.dumps({"family": "spatial", "dt_scaling": "h2"}, indent=2) + "\n")
    return 0


def stage_smoke(output: Path, space: str, mpi: str) -> int:
    ranks = 2 if mpi == "on" else 1
    threads = 4 if space == "KokkosOpenMP" else 1
    job = _job(space, mpi, 16, resolutions=(16,), ranks=ranks, threads=threads)
    _run_one(job, 16, output, family="global", t_end=0.25)
    write_series_json(output, "EU-02", ["n016"])
    return 0


def stage_dump(output: Path, space: str, mpi: str) -> int:
    job = _job(space, mpi, 64, resolutions=(64,))
    _run_one(job, 64, output, family="global", t_end=1.0, dump_times=DUMP_TIMES)
    write_series_json(output, "EU-02", ["n064"])
    return 0


def stage_analyze(output: Path, space: str, mpi: str) -> int:
    series = output
    if not (series / "series.json").is_file():
        raise SystemExit(f"missing series.json under {series}")
    ANALYZE.write_eu02_report(output, native_series=series)
    campaign = ANALYZE.campaign_from_evidence(series)
    extras = {}
    finest = int(campaign["resolutions"][-1])
    extras.update(
        ANALYZE.diagnose_resolution(
            campaign["fields"][finest],
            finest,
            float(campaign["t_end"]),
        )
    )
    snap = series / f"n{finest:03d}" / "snapshots.npz"
    if snap.is_file():
        extras.update(_trajectory_from_snapshots(snap, finest))
    ANALYZE.write_native_campaign_report(output, campaign, extras)
    try:
        from verification.pops_verify.visualization.render import render_run

        render_run(output, suite="release", formats=("svg", "png", "pdf", "gif"))
    except Exception as exc:
        (output / "visual_render_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    return 0


def _trajectory_from_snapshots(path: Path, n_cells: int) -> dict:
    data = np.load(path)
    times = [float(value) for value in data["times"]]
    analytic_x = []
    analytic_y = []
    numerical_x = []
    numerical_y = []
    frames = []
    x, y, width = RUN.cell_centers(n_cells)
    x_axis = [float(value) for value in x[0, :]]
    y_axis = [float(value) for value in y[:, 0]]
    rho_limits = None
    for index, instant in enumerate(times):
        packed = data[f"t_{index}"]
        primitives = RUN.conserved_to_primitives(packed)
        analytic = ANALYZE._exact.analytic_center(instant) if hasattr(ANALYZE, "_exact") else None
        from verification.pops_verify.case_authoring import load_sibling_module

        exact = load_sibling_module(
            ROOT / "verification" / "cases" / "euler" / "isentropic_vortex" / "exact.py"
        )
        analytic = exact.analytic_center(instant)
        numerical = ANALYZE.vortex_center_from_density(
            primitives["rho"], n_cells, expected=analytic
        )
        analytic_x.append(analytic[0])
        analytic_y.append(analytic[1])
        numerical_x.append(numerical[0])
        numerical_y.append(numerical[1])
        if rho_limits is None:
            rho_limits = [float(primitives["rho"].min()), float(primitives["rho"].max())]
        else:
            rho_limits = [
                min(rho_limits[0], float(primitives["rho"].min())),
                max(rho_limits[1], float(primitives["rho"].max())),
            ]
        frames.append(
            {
                "time": instant,
                "step": index,
                "source": "accepted_state",
                "x": x_axis,
                "y": y_axis,
                "field": np.asarray(primitives["rho"], dtype=np.float64).tolist(),
            }
        )
    return {
        "trajectory": {
            "figure_id": "vortex_center_trajectory",
            "kind": "linecuts",
            "data_kind": "campaign",
            "units": {"x": "t", "y": "x_c"},
            "times": times,
            "step_numbers": list(range(len(times))),
            "series": [
                {"name": "analytic_x", "x": times, "y": analytic_x},
                {"name": "numerical_x", "x": times, "y": numerical_x},
                {"name": "analytic_y", "x": times, "y": analytic_y},
                {"name": "numerical_y", "x": times, "y": numerical_y},
            ],
            "title": "Vortex-centre trajectory",
        },
        "animation": {
            "figure_id": "animation",
            "kind": "animation",
            "data_kind": "campaign",
            "units": {"x": "x", "y": "y", "field": "rho"},
            "color_limits": rho_limits,
            "periodic": True,
            "times": times,
            "step_numbers": list(range(len(times))),
            "frames": frames,
        },
        "storyboard": {
            "figure_id": "storyboard",
            "kind": "storyboard",
            "data_kind": "campaign",
            "units": {"x": "x", "y": "rho"},
            "times": times,
            "step_numbers": list(range(len(times))),
            "frames": [
                {
                    "event": "t",
                    "time": frame["time"],
                    "step": frame["step"],
                    "source": "accepted_state",
                    "series": [
                        {
                            "name": "rho mid",
                            "x": frame["x"],
                            "y": [float(row) for row in np.asarray(frame["field"])[n_cells // 2]],
                        }
                    ],
                }
                for frame in frames
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--space", default="KokkosSerial")
    parser.add_argument("--mpi", default="off")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("POPS_NATIVE_DIM", "2")
    stages = {
        "series": stage_series,
        "temporal": stage_temporal,
        "spatial": stage_spatial,
        "smoke": stage_smoke,
        "dump": stage_dump,
        "analyze": stage_analyze,
    }
    return stages[args.stage](args.output, args.space, args.mpi)


if __name__ == "__main__":
    raise SystemExit(main())
