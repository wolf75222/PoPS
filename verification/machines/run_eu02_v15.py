#!/usr/bin/env python3
"""EU-02 v1.5 driver: official CampaignRequest / EvidenceBundle path.

Stages:
  series   — official WENO5-Z four-resolution global (acceptance)
  temporal — WENO FixedDt at n=128 (or 256) only when isolated
  spatial  — WENO FixedDt dt ∝ h² at 16/32/64/128
  tvd      — labeled VanLeer fail control at 16/32/64/128 (no n=256)
  smoke    — n=16 t=0.25 CampaignRequest
  dump     — finest (n=128) global snapshots
  parity   — Serial vs OpenMP4 vs MPI2 bit/norm compare
  analyze  — EvidenceBundle → report / Phase 8 visuals in finest job
"""
from __future__ import annotations

import argparse
import dataclasses
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
TEMPORAL_N = 128
TEMPORAL_FINER_N = 256
TEMPORAL_DTS = (0.008, 0.004, 0.002, 0.001)
TEMPORAL_T_END = 0.32
DUMP_N = 128
DUMP_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
ACCEPTANCE_RESOLUTIONS = (16, 32, 64, 128)


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


def _run_one(
    job,
    n_cells: int,
    output: Path,
    *,
    dt=None,
    family="global",
    dump_times=None,
    t_end=None,
    reconstruction="weno5z",
    job_name=None,
):
    artifact = _artifact()
    job_dir = output / (job_name or f"n{int(n_cells):03d}")
    leaf_job = dataclasses.replace(job, min_resolution=int(n_cells))
    request = CampaignRequest.from_job(leaf_job, output_dir=job_dir)
    kwargs = {
        "request": request,
        "n_cells": n_cells,
        "family": family,
        "reconstruction": reconstruction,
    }
    if dt is not None:
        kwargs["dt"] = float(dt)
    if dump_times is not None:
        kwargs["dump_times"] = dump_times
    if t_end is not None:
        kwargs["t_end"] = float(t_end)
    payload = RUN.run_native(**kwargs)
    case = {"id": "EU-02", "path": "verification/cases/euler/isentropic_vortex/run.py"}
    _emit(leaf_job, request, payload, artifact, job_dir, case)
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
    status = int(verification_main(argv))
    try:
        series = resolve_series_dir(output, space, mpi)
        (series / "family.json").write_text(
            json.dumps(
                {
                    "family": "global",
                    "dt_scaling": "cfl",
                    "reconstruction": "weno5z",
                    "reconstruction_role": "acceptance",
                },
                indent=2,
            )
            + "\n"
        )
    except SystemExit:
        pass
    return status


def _spatial_linf_at(space: str, mpi: str, n_cells: int, sibling: Path) -> float | None:
    spatial_root = sibling.parent / f"EU02_D2_{space}_{mpi}_spatial"
    try:
        series = resolve_series_dir(spatial_root, space, mpi)
        campaign = ANALYZE.campaign_from_evidence(series)
        packed = campaign["fields"][int(n_cells)]
        errors = ANALYZE.primitive_errors(packed, int(n_cells), float(campaign["t_end"]))
        return float(errors["rho"].linf)
    except Exception:
        return None


def _dt_job_name(dt: float) -> str:
    return f"dt{dt:g}".replace(".", "p")


def _temporal_prefix(n_cells: int) -> str:
    return f"n{int(n_cells):03d}-"


def _run_temporal_dt(output: Path, space: str, mpi: str, n_cells: int, dt: float) -> str:
    name = f"{_temporal_prefix(n_cells)}{_dt_job_name(float(dt))}"
    job = _job(space, mpi, n_cells, resolutions=(n_cells, n_cells, n_cells, n_cells))
    _run_one(
        job,
        n_cells,
        output,
        dt=float(dt),
        family="temporal",
        t_end=TEMPORAL_T_END,
        reconstruction="weno5z",
        job_name=name,
    )
    return name


def _coarsest_dt_linf(output: Path, name: str, n_cells: int) -> float:
    packed = np.load(output / name / "result.npy")
    return float(ANALYZE.primitive_errors(packed, n_cells, TEMPORAL_T_END)["rho"].linf)


def _probe_spatial_linf(output: Path, space: str, mpi: str, n_cells: int) -> float:
    known = _spatial_linf_at(space, mpi, n_cells, output)
    if known is not None:
        return known
    name = f"spatial-probe-n{int(n_cells):03d}"
    job = _job(space, mpi, n_cells, resolutions=(n_cells,))
    _run_one(
        job,
        n_cells,
        output,
        family="spatial",
        dt=RUN.spatial_fixed_dt(n_cells),
        t_end=TEMPORAL_T_END,
        reconstruction="weno5z",
        job_name=name,
    )
    packed = np.load(output / name / "result.npy")
    return float(ANALYZE.primitive_errors(packed, n_cells, TEMPORAL_T_END)["rho"].linf)


def stage_temporal(output: Path, space: str, mpi: str) -> int:
    chosen_n = None
    spatial_linf = None
    claim_names: list[str] = []
    extra_names: list[str] = []
    for n_cells in (TEMPORAL_N, TEMPORAL_FINER_N):
        spatial_linf = _probe_spatial_linf(output, space, mpi, n_cells)
        coarsest = float(TEMPORAL_DTS[0])
        first = _run_temporal_dt(output, space, mpi, n_cells, coarsest)
        extra_names.append(first)
        coarsest_err = _coarsest_dt_linf(output, first, n_cells)
        if ANALYZE.temporal_is_isolated(spatial_linf, coarsest_err):
            chosen_n = n_cells
            claim_names = [first]
            for dt in TEMPORAL_DTS[1:]:
                claim_names.append(_run_temporal_dt(output, space, mpi, n_cells, float(dt)))
            break
    family = {
        "family": "temporal",
        "t_end": TEMPORAL_T_END,
        "reconstruction": "weno5z",
        "reconstruction_role": "acceptance",
        "spatial_linf": spatial_linf,
        "isolated": chosen_n is not None,
    }
    if chosen_n is None:
        write_series_json(output, "EU-02", extra_names)
        family.update(
            {
                "dts": [float(TEMPORAL_DTS[0])],
                "n_cells": TEMPORAL_FINER_N,
                "reason": "spatial L∞ not 10× below coarsest-dt error; temporal not claimed",
            }
        )
        (output / "family.json").write_text(json.dumps(family, indent=2) + "\n")
        return 0
    if chosen_n == TEMPORAL_N:
        finer_check = _run_temporal_dt(
            output, space, mpi, TEMPORAL_FINER_N, float(TEMPORAL_DTS[-1])
        )
        family["finer_grid_check"] = finer_check
    write_series_json(output, "EU-02", claim_names)
    family.update({"dts": list(TEMPORAL_DTS), "n_cells": chosen_n})
    (output / "family.json").write_text(json.dumps(family, indent=2) + "\n")
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
    (output / "family.json").write_text(
        json.dumps(
            {
                "family": "spatial",
                "dt_scaling": "h2",
                "reconstruction": "weno5z",
                "reconstruction_role": "acceptance",
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def stage_smoke(output: Path, space: str, mpi: str) -> int:
    ranks = 2 if mpi == "on" else 1
    threads = 4 if space == "KokkosOpenMP" else 1
    job = _job(space, mpi, 16, resolutions=(16,), ranks=ranks, threads=threads)
    _run_one(job, 16, output, family="global", t_end=0.25)
    write_series_json(output, "EU-02", ["n016"])
    return 0


def stage_dump(output: Path, space: str, mpi: str) -> int:
    job = _job(space, mpi, DUMP_N, resolutions=(DUMP_N,))
    _run_one(
        job,
        DUMP_N,
        output,
        family="global",
        t_end=1.0,
        dump_times=DUMP_TIMES,
        reconstruction="weno5z",
    )
    write_series_json(output, "EU-02", [f"n{DUMP_N:03d}"])
    return 0


def stage_tvd(output: Path, space: str, mpi: str) -> int:
    resolutions = ACCEPTANCE_RESOLUTIONS
    job = _job(space, mpi, 16, resolutions=resolutions)
    names = []
    for n_cells in resolutions:
        _run_one(
            job,
            n_cells,
            output,
            family="global",
            t_end=1.0,
            reconstruction="vanleer",
        )
        names.append(f"n{int(n_cells):03d}")
    write_series_json(output, "EU-02", names)
    (output / "family.json").write_text(
        json.dumps(
            {
                "family": "global",
                "reconstruction": "vanleer",
                "reconstruction_role": "tvd_fail_control",
                "dt_scaling": "cfl",
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def stage_parity(output: Path, space: str, mpi: str) -> int:
    del space, mpi
    root = output.parent
    serial = Path(os.environ.get("POPS_EU02_SERIAL_SMOKE", root / "EU02_D2_KokkosSerial_off_smoke"))
    openmp = Path(os.environ.get("POPS_EU02_OPENMP_SMOKE", root / "EU02_D2_KokkosOpenMP_off_smoke"))
    mpi_dir = Path(os.environ.get("POPS_EU02_MPI_SMOKE", root / "EU02_D2_KokkosSerial_on_smoke"))
    report = ANALYZE.compare_smokes_from_dirs(serial, openmp, mpi_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


def resolve_series_dir(output: Path, space: str | None = None, mpi: str | None = None) -> Path:
    """Official runner writes series.json under EU-02/dimN-space-mpi."""
    root = Path(output)
    if (root / "series.json").is_file():
        return root
    if space and mpi:
        official = root / "EU-02" / f"dim2-{space}-{mpi}"
        if (official / "series.json").is_file():
            return official
    for space_name in ("KokkosSerial", "KokkosOpenMP"):
        for mpi_mode in ("off", "on"):
            candidate = root / "EU-02" / f"dim2-{space_name}-{mpi_mode}"
            if (candidate / "series.json").is_file():
                return candidate
    matches = [path.parent for path in root.rglob("series.json")]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"missing series.json under {root}")


def stage_analyze(output: Path, space: str, mpi: str) -> int:
    series = resolve_series_dir(output, space, mpi)
    (output / "resolved_series.txt").write_text(str(series) + "\n", encoding="utf-8")
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
    try:
        ANALYZE.write_native_campaign_report(output, campaign, extras)
    except ANALYZE.NativeSeriesError as exc:
        (output / "order_claim_error.txt").write_text(f"{exc}\n")
        ANALYZE.write_eu02_report(output)
        return 0
    visual_job = ANALYZE.finest_visual_job_dir(series)
    try:
        from verification.pops_verify.visualization.render import render_run

        render_run(visual_job, suite="release", formats=("svg", "png", "pdf", "gif"))
    except Exception as exc:
        (visual_job / "visual_render_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        (output / "visual_render_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\njob={visual_job}\n"
        )
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
        "tvd": stage_tvd,
        "smoke": stage_smoke,
        "dump": stage_dump,
        "parity": stage_parity,
        "analyze": stage_analyze,
    }
    return stages[args.stage](args.output, args.space, args.mpi)


if __name__ == "__main__":
    raise SystemExit(main())
