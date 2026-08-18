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
    parser.add_argument("--stage", required=True, choices=("smoke", "pair", "series"))
    parser.add_argument("--space", required=True)
    parser.add_argument("--mpi-mode", required=True, choices=("off", "on"))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    output = args.output
    sha = _git_sha(repo)
    native_root = output / "native" / f"dim{args.dim}"
    if args.stage == "series":
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
        print(
            f"{args.gate} series sha={sha} dim={args.dim} "
            f"orders={[row.get('observed_order') for row in summary['orders']]} "
            f"passed={summary['coverage']['cases_passed']}"
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
        if args.space == "KokkosSerial" and provenance.get("kokkos_execution_space") != "KokkosSerial":
            raise SystemExit(f"{args.gate}: serial smoke recorded {provenance.get('kokkos_execution_space')}")
    print(
        f"{args.gate} {args.stage} sha={sha} dim={args.dim} "
        f"label={smoke.get('label')} linf={smoke.get('linf')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
