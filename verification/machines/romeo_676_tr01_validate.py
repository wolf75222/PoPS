#!/usr/bin/env python3
"""Fail-closed checks for one ROMEO TR-01 stage directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess

from jsonschema import Draft202012Validator


def _validate(document: dict, schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(document)


def _git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=("smoke", "pair", "series", "temporal"))
    parser.add_argument("--space", required=True)
    parser.add_argument("--mpi-mode", required=True, choices=("off", "on"))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    output = args.output
    sha = _git_sha(repo)
    native_root = output / "native" / f"dim{args.dim}"
    if args.stage in {"series", "temporal"}:
        summary = json.loads((native_root / "summary.json").read_text(encoding="utf-8"))
        metrics = json.loads((native_root / "metrics.json").read_text(encoding="utf-8"))
        visual = json.loads((native_root / "visual_manifest.json").read_text(encoding="utf-8"))
        _validate(summary, repo / "schemas" / "verification_report.v1.json")
        _validate(metrics, repo / "schemas" / "verification_metrics.v1.json")
        if not (native_root / "visual_data" / "spatial_convergence.json").is_file():
            raise SystemExit(f"{args.gate}: missing visual_data")
        if visual.get("case_id") != "TR-01":
            raise SystemExit(f"{args.gate}: visual_manifest case_id {visual.get('case_id')}")
        if not summary.get("orders"):
            raise SystemExit(f"{args.gate}: series produced no orders")
        expected_kind = "temporal" if args.stage == "temporal" else "global"
        kinds = {row.get("kind") for row in summary["orders"]}
        if expected_kind not in kinds:
            raise SystemExit(
                f"{args.gate}: expected order kind {expected_kind}, got {kinds}"
            )
        if "spatial" in kinds and args.stage != "series":
            raise SystemExit(f"{args.gate}: constant-CFL/temporal must not be labeled spatial")
        if args.stage == "series" and kinds != {"global"}:
            raise SystemExit(
                f"{args.gate}: four-resolution CFL series must be labeled global, got {kinds}"
            )
        if summary["coverage"]["cases_failed"] and not summary.get("failures"):
            raise SystemExit(
                f"{args.gate}: order gate failed but failures[] is empty"
            )
        failures_csv = (native_root / "failures.csv").read_text(encoding="utf-8")
        if summary["coverage"]["cases_failed"] and "TR-01" not in failures_csv:
            raise SystemExit(f"{args.gate}: failures.csv missing TR-01")
        print(
            f"{args.gate} {args.stage} sha={sha} dim={args.dim} "
            f"orders={[row.get('observed_order') for row in summary['orders']]} "
            f"kind={sorted(kinds)} passed={summary['coverage']['cases_passed']}"
        )
        return 0

    smoke = json.loads((native_root / "smoke.json").read_text(encoding="utf-8"))
    if smoke.get("order_pass") is True:
        raise SystemExit(f"{args.gate}: smoke/pair claimed an order pass")
    if smoke.get("verdict") not in {"smoke", "not-run"}:
        raise SystemExit(f"{args.gate}: expected smoke/not-run, got {smoke}")
    if args.dim != 3 and smoke.get("label") == "canonical":
        raise SystemExit(f"{args.gate}: dim {args.dim} must not be labeled canonical 3-d")
    if args.stage == "smoke" and (output / "campaign" / "results.json").is_file():
        results = json.loads((output / "campaign" / "results.json").read_text(encoding="utf-8"))
        if args.mpi_mode == "on":
            # MPI campaign invoke is smoke/non-scientific; not TR-01 acceptance.
            # A runner-foundation pass or fail here must not become the TR-01 gate.
            print(
                f"{args.gate} mpi campaign invoke is smoke/non-scientific; "
                f"not TR-01 acceptance (status={results[0].get('status') if results else None})"
            )
            print(
                f"{args.gate} {args.stage} sha={sha} dim={args.dim} "
                f"label={smoke.get('label')} linf={smoke.get('linf')}"
            )
            return 0
        if len(results) != 1 or results[0].get("status") != "pass":
            raise SystemExit(f"{args.gate}: campaign smoke did not pass: {results}")
        job_dir = (
            output
            / "campaign"
            / "TR-01"
            / f"dim{args.dim}-{args.space}-{args.mpi_mode}"
        )
        provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
        metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
        _validate(provenance, repo / "schemas" / "verification_provenance.v1.json")
        _validate(metrics, repo / "schemas" / "verification_metrics.v1.json")
        if provenance.get("repository_sha") != sha:
            raise SystemExit(
                f"{args.gate}: provenance sha {provenance.get('repository_sha')} != {sha}"
            )
        if provenance.get("pops_native_dim") != args.dim:
            raise SystemExit(f"{args.gate}: provenance dim mismatch")
        if args.mpi_mode == "off" and provenance.get("mpi_enabled"):
            raise SystemExit(f"{args.gate}: serial smoke recorded mpi_enabled")
        live_ranks = provenance.get("mpi_ranks")
        live_threads = provenance.get("omp_threads_per_rank")
        if args.mpi_mode == "on" and live_ranks != 2:
            raise SystemExit(
                f"{args.gate}: mpi_ranks={live_ranks} is not the native world size 2"
            )
        if args.mpi_mode == "off" and live_ranks not in {1, None}:
            raise SystemExit(f"{args.gate}: serial smoke recorded mpi_ranks={live_ranks}")
        if args.space == "KokkosSerial":
            if provenance.get("kokkos_execution_space") != "KokkosSerial":
                raise SystemExit(
                    f"{args.gate}: Serial request recorded "
                    f"{provenance.get('kokkos_execution_space')}; Serial not-run"
                )
            if live_threads not in {1, None}:
                raise SystemExit(
                    f"{args.gate}: Serial smoke recorded omp_threads_per_rank={live_threads}"
                )
        if args.space == "KokkosOpenMP":
            if provenance.get("kokkos_execution_space") != "KokkosOpenMP":
                raise SystemExit(
                    f"{args.gate}: OpenMP smoke recorded "
                    f"{provenance.get('kokkos_execution_space')}"
                )
            if not isinstance(live_threads, int) or live_threads <= 1:
                raise SystemExit(
                    f"{args.gate}: OpenMP smoke recorded omp_threads_per_rank={live_threads}"
                )
    print(
        f"{args.gate} {args.stage} sha={sha} dim={args.dim} "
        f"label={smoke.get('label')} linf={smoke.get('linf')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
