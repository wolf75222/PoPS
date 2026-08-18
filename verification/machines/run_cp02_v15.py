"""ROMEO/local CP-02 campaign: Serial order plus bounded OpenMP/MPI smokes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np

from verification.pops_verify.campaign import CampaignRequest, CampaignResources
from verification.pops_verify.case_authoring import load_sibling_module

ROOT = Path(__file__).resolve().parents[2]
RUN = load_sibling_module(
    ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "run.py"
)
ANALYZE = load_sibling_module(
    ROOT / "verification" / "cases" / "euler_poisson" / "langmuir_cold" / "analyze.py"
)


def _request(*, space: str, mpi_mode: str, n: int, threads: int, ranks: int) -> CampaignRequest:
    return CampaignRequest(
        case_id="CP-02",
        pops_native_dim=1,
        suite="pr",
        execution_space=space,
        mpi_mode=mpi_mode,
        min_resolution=n,
        resources=CampaignResources(
            nodes=1,
            mpi_ranks=ranks,
            omp_threads=threads,
            resolutions=(n,) if mpi_mode == "on" or space != "KokkosSerial" else (16, 32, 64, 128),
        ),
        evidence_status="required",
    )


def write_temporal_step_audit(series: Path) -> dict:
    """Prove temporal jobs took distinct FixedDt paths and distinct result hashes."""
    jobs = list(json.loads((series / "series.json").read_text(encoding="utf-8")).get("jobs") or [])
    rows = []
    for name in jobs:
        resolved = json.loads((series / name / "resolved_case.json").read_text(encoding="utf-8"))
        job = resolved.get("job") or {}
        result = series / name / "result.npy"
        digest = hashlib.sha256(result.read_bytes()).hexdigest() if result.is_file() else None
        rows.append(
            {
                "job": name,
                "dt": job.get("dt"),
                "accepted_steps": job.get("accepted_steps"),
                "expected_accepted_steps": job.get("expected_accepted_steps"),
                "result_digest": digest,
            }
        )
    payload = {
        "jobs": rows,
        "first_two_steps_differ": bool(
            rows
            and rows[0].get("accepted_steps") is not None
            and rows[0].get("accepted_steps") != rows[1].get("accepted_steps")
        ),
        "first_two_hashes_differ": bool(
            rows
            and rows[0].get("result_digest")
            and rows[0].get("result_digest") != rows[1].get("result_digest")
        ),
    }
    dest = series / "temporal_step_audit.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def compare_mpi_serial(serial_job: Path, mpi_job: Path, output: Path) -> dict:
    """Bit and L2/L∞ compare of MPI n=16 result.npy against Serial n=16."""
    serial = np.load(serial_job / "result.npy")
    mpi = np.load(mpi_job / "result.npy")
    if serial.shape != mpi.shape:
        raise ValueError(f"MPI/Serial shape mismatch {mpi.shape} vs {serial.shape}")
    delta = np.asarray(mpi, dtype=np.float64) - np.asarray(serial, dtype=np.float64)
    payload = {
        "serial": str(serial_job),
        "mpi": str(mpi_job),
        "shape": list(serial.shape),
        "bit_identical": bool(np.array_equal(serial, mpi)),
        "linf": float(np.max(np.abs(delta))),
        "l2": float(np.sqrt(np.mean(delta * delta))),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _write_failure(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "required_failure", "reason": reason}, indent=2) + "\n")


def _analyze_and_render(output: Path, series: Path) -> None:
    """Score and render on the EvidenceBundle series directory, then mirror reports."""
    report_dir = output / "native" / "dim1"
    ANALYZE.write_cp02_report(series, series_dir=series)
    try:
        ANALYZE.render_campaign_figures(series)
    except Exception as exc:
        (series / "render_error.txt").write_text(str(exc))
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "render_error.txt").write_text(str(exc))
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "REPORT.md",
        "summary.json",
        "coverage.csv",
        "failures.csv",
        "metrics.json",
        "provenance.json",
        "status.json",
        "program.json",
        "resolved_case.json",
        "native_artifact.json",
        "visual_manifest.json",
        "render_error.txt",
    ):
        source = series / name
        if source.is_file():
            shutil.copy2(source, report_dir / name)
    analysis = series / "analysis"
    if analysis.is_dir():
        dest = report_dir / "analysis"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(analysis, dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gate",
        choices=("serial", "temporal", "spatial", "openmp", "mpi", "analyze", "compare", "parity"),
        default="serial",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.gate == "analyze":
        _analyze_and_render(output, output / "series")
        return 0
    if args.gate == "serial":
        request = _request(space="KokkosSerial", mpi_mode="off", n=16, threads=1, ranks=1)
        series = RUN.run_order_campaign(
            output / "series", request=request, family="global"
        )
        _analyze_and_render(output, series)
        return 0
    if args.gate == "temporal":
        request = _request(space="KokkosSerial", mpi_mode="off", n=256, threads=1, ranks=1)
        series = RUN.run_temporal_campaign(
            output / "series", request=request, n_cells=256
        )
        write_temporal_step_audit(series)
        _analyze_and_render(output, series)
        return 0
    if args.gate == "spatial":
        request = _request(space="KokkosSerial", mpi_mode="off", n=16, threads=1, ranks=1)
        series = RUN.run_spatial_campaign(output / "series", request=request)
        _analyze_and_render(output, series)
        return 0
    if args.gate in {"compare", "parity"}:
        serial = Path(
            os.environ.get(
                "POPS_CP02_SERIAL_N16",
                output / "horizon" / "n16",
            )
        )
        if args.gate == "parity" or not serial.is_dir():
            request = _request(space="KokkosSerial", mpi_mode="off", n=16, threads=1, ranks=1)
            RUN.run_order_campaign(
                output / "horizon",
                request=request,
                resolutions=(16,),
                family="global",
                t_end=RUN.period(),
            )
            serial = output / "horizon" / "n16"
        openmp = Path(
            os.environ.get("POPS_CP02_OPENMP_N16", output.parent / "openmp" / "openmp" / "n16")
        )
        mpi = Path(os.environ.get("POPS_CP02_MPI_N16", output.parent / "mpi" / "mpi" / "n16"))
        if openmp.is_dir():
            compare_mpi_serial(serial, openmp, output / "openmp_serial_compare.json")
        if mpi.is_dir():
            compare_mpi_serial(serial, mpi, output / "mpi_serial_compare.json")
        return 0
    if args.gate == "openmp":
        request = _request(space="KokkosOpenMP", mpi_mode="off", n=16, threads=4, ranks=1)
        RUN.run_order_campaign(
            output / "openmp",
            request=request,
            resolutions=(16,),
            family="global",
            t_end=RUN.period(),
        )
        return 0
    request = _request(space="KokkosSerial", mpi_mode="on", n=16, threads=1, ranks=2)
    RUN.run_order_campaign(
        output / "mpi",
        request=request,
        resolutions=(16,),
        family="global",
        t_end=RUN.period(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
