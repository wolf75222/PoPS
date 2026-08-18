"""Fail-closed checks for one ROMEO foundation gate directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

from jsonschema import Draft202012Validator

from verification.pops_verify.capabilities import authenticate_installed_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--case", required=True)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--mpi-mode", required=True, choices=("off", "on"))
    parser.add_argument("--expect-ranks", required=True, type=int)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    output = args.output
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    jobs = plan.get("jobs") or []
    if not jobs:
        raise SystemExit(f"{args.gate}: planned zero jobs")
    job = jobs[0]
    if job.get("case_id") != args.case or int(job.get("pops_native_dim")) != args.dim:
        raise SystemExit(f"{args.gate}: unexpected planned job {job}")
    if job.get("mpi_mode") != args.mpi_mode:
        raise SystemExit(f"{args.gate}: planned mpi_mode {job.get('mpi_mode')!r}")

    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if len(results) != 1:
        raise SystemExit(f"{args.gate}: expected one rank-0 ledger row, got {results}")
    if results[0].get("status") != "pass":
        raise SystemExit(f"{args.gate}: campaign did not pass: {results}")

    job_dir = output / args.case / f"dim{args.dim}-KokkosSerial-{args.mpi_mode}"
    resolved = json.loads((job_dir / "resolved_case.json").read_text(encoding="utf-8"))
    metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    _validate(metrics, repo / "schemas" / "verification_metrics.v1.json")
    _validate(provenance, repo / "schemas" / "verification_provenance.v1.json")
    _validate(report, repo / "schemas" / "verification_report.v1.json")
    if resolved.get("status") != "pass":
        raise SystemExit(f"{args.gate}: resolved_case status {resolved.get('status')}")

    sha = _git_sha(repo)
    if provenance.get("repository_sha") != sha:
        raise SystemExit(
            f"{args.gate}: provenance sha {provenance.get('repository_sha')} != {sha}"
        )
    if provenance.get("repository_dirty") is not False:
        raise SystemExit(f"{args.gate}: provenance is dirty")
    if int(provenance.get("mpi_ranks")) != args.expect_ranks:
        raise SystemExit(
            f"{args.gate}: mpi_ranks={provenance.get('mpi_ranks')} expected {args.expect_ranks}"
        )
    if bool(provenance.get("mpi_enabled")) != (args.mpi_mode == "on"):
        raise SystemExit(f"{args.gate}: mpi_enabled={provenance.get('mpi_enabled')}")

    artifact = authenticate_installed_artifact(dimension=args.dim)
    if artifact.sha256 != _sha256(artifact.path):
        raise SystemExit(f"{args.gate}: leaf digest drifted")
    if artifact.native_variant_manifest_digest != provenance.get(
        "native_variant_manifest_digest"
    ):
        raise SystemExit(f"{args.gate}: variants digest disagrees")
    if artifact.native_header_signature != provenance.get("native_header_signature"):
        raise SystemExit(f"{args.gate}: header signature disagrees")
    if artifact.component_catalog_digest != provenance.get("component_catalog_digest"):
        raise SystemExit(f"{args.gate}: catalog digest disagrees")
    if artifact.doctor_ok is not True or provenance.get("doctor_ok") is not True:
        raise SystemExit(
            f"{args.gate}: doctor_ok artifact={artifact.doctor_ok} "
            f"provenance={provenance.get('doctor_ok')}"
        )
    if args.mpi_mode == "on":
        if artifact.has_mpi is not True or artifact.hdf5_collective is not True:
            raise SystemExit(
                f"{args.gate}: MPI artifact facts has_mpi={artifact.has_mpi} "
                f"hdf5_collective={artifact.hdf5_collective}"
            )
        if provenance.get("hdf5_collective_enabled") is not True:
            raise SystemExit(f"{args.gate}: provenance hdf5_collective_enabled is not true")
        if int(os.environ.get("SLURM_NTASKS", "0") or 0) != 2:
            print(
                f"{args.gate}: warning SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}",
                file=sys.stderr,
            )

    print(
        json.dumps(
            {
                "gate": args.gate,
                "case": args.case,
                "dim": args.dim,
                "status": "pass",
                "repository_sha": sha,
                "leaf_sha256": artifact.sha256,
                "doctor_ok": True,
                "mpi_enabled": provenance.get("mpi_enabled"),
                "mpi_ranks": provenance.get("mpi_ranks"),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
