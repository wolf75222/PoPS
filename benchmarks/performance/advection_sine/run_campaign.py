#!/usr/bin/env python3
"""Launch one validated campaign against the public Python PoPS case."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import CampaignError, load_campaign, route_uses_gpu


_SLURM_REQUIRED_ENVIRONMENT = (
    "PATH",
    "PYTHONPATH",
    "POPS_NATIVE_VARIANTS_ROOT",
    "POPS_INCLUDE",
    "POPS_CACHE_DIR",
    "XDG_CACHE_HOME",
    "OMP_NUM_THREADS",
    "KOKKOS_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_DYNAMIC",
)
_SLURM_OPTIONAL_ENVIRONMENT = ("LD_LIBRARY_PATH",)


def _write_new(path: Path, text: str) -> None:
    """Publish launch evidence once; a retry must use a fresh raw directory."""
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError as error:
        raise CampaignError(f"refusing to overwrite launch evidence: {path}") from error


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="authenticated Python interpreter importing the selected PoPS build",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", choices=("local", "slurm"), default="local")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _command(
    campaign: dict,
    point: dict,
    python: Path,
    output: Path,
    launcher: str,
    environment: dict[str, str],
) -> list[str]:
    command: list[str] = []
    if launcher == "slurm":
        command.extend(
            [
                "srun",
                "--exclusive",
                "--kill-on-bad-exit=1",
                f"--nodes={point['nodes']}",
                f"--ntasks={point['ranks']}",
                f"--cpus-per-task={point['threads']}",
                "--cpu-bind=cores",
            ]
        )
        if route_uses_gpu(campaign["route"]):
            command.append("--gpus-per-task=1")
        missing = [name for name in _SLURM_REQUIRED_ENVIRONMENT if not environment.get(name)]
        if missing:
            raise CampaignError(
                "Slurm launcher requires explicit environment values: " + ", ".join(missing)
            )
        forwarded = _SLURM_REQUIRED_ENVIRONMENT + tuple(
            name for name in _SLURM_OPTIONAL_ENVIRONMENT if environment.get(name)
        )
        command.extend(["/usr/bin/env", *(f"{name}={environment[name]}" for name in forwarded)])
    elif point["nodes"] != 1 or point["ranks"] != 1:
        raise CampaignError("local launcher supports only one node and one rank")
    command.extend(
        [
            str(python),
            str(Path(__file__).with_name("advection_sine.py")),
            "--resolution=" + ",".join(str(value) for value in point["resolution"]),
            f"--mode={campaign['mode']}",
            f"--route={campaign['route']}",
            f"--campaign={campaign['id']}",
            f"--point={point['id']}",
            f"--expected-ranks={point['ranks']}",
            f"--nodes={point['nodes']}",
            f"--threads={point['threads']}",
            f"--block-size={campaign['block_size']}",
            f"--steps={campaign['steps']}",
            f"--cfl={campaign['cfl']}",
            f"--warmups={campaign['warmups']}",
            f"--repetitions={campaign['repetitions']}",
            f"--output-dir={output}",
        ]
    )
    return command


def main() -> int:
    args = _arguments()
    try:
        campaign = load_campaign(args.campaign.resolve())
        python = args.python.resolve(strict=True)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise CampaignError(f"Python interpreter is not runnable: {python}")
    except (CampaignError, OSError) as error:
        print(f"campaign launch refused: {error}", file=sys.stderr)
        return 2

    output_root = args.output.resolve()
    permitted_prior_evidence = {"source.manifest.json", "build.receipt.json"}
    if output_root.exists():
        present = {path.name for path in output_root.iterdir()}
        if present != permitted_prior_evidence:
            print(
                f"campaign launch refused: output already contains evidence: {output_root}",
                file=sys.stderr,
            )
            return 2
    else:
        output_root.mkdir(parents=True)
    normalized = output_root / "campaign.normalized.json"
    try:
        _write_new(normalized, json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    except CampaignError as error:
        print(f"campaign launch refused: {error}", file=sys.stderr)
        return 2
    manifest = {
        "schema": "pops.performance.advection-sine.launch.v1",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign["id"],
        "launcher": args.launcher,
        "python": str(python),
        "dry_run": args.dry_run,
        "runs": [],
    }
    for point in campaign["points"]:
        result_path = output_root / point["id"]
        log_path = output_root / f"{point['id']}.log"
        environment = os.environ.copy()
        thread_count = str(point["threads"])
        environment["OMP_NUM_THREADS"] = thread_count
        environment["KOKKOS_NUM_THREADS"] = thread_count
        environment["OMP_PROC_BIND"] = "spread"
        environment["OMP_PLACES"] = "cores"
        environment["OMP_DYNAMIC"] = "false"
        try:
            command = _command(
                campaign, point, python, result_path, args.launcher, environment
            )
        except CampaignError as error:
            print(f"campaign launch refused: {error}", file=sys.stderr)
            return 2
        run = {
            "point": point["id"],
            "command": command,
            "result": str(result_path),
            "log": str(log_path),
        }
        manifest["runs"].append(run)
        print(" ".join(command))
        if args.dry_run:
            run["status"] = "planned"
            continue
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        run["launcher_wall_seconds"] = time.perf_counter() - started
        run["returncode"] = completed.returncode
        run["status"] = "passed" if completed.returncode == 0 else "failed"
        try:
            _write_new(
                log_path,
                "COMMAND\n"
                + " ".join(command)
                + "\n\nSTDOUT\n"
                + completed.stdout
                + "\nSTDERR\n"
                + completed.stderr,
            )
        except CampaignError as error:
            print(f"campaign launch refused: {error}", file=sys.stderr)
            return 2
        if completed.returncode != 0:
            manifest["status"] = "failed"
            try:
                _write_new(
                    output_root / "launch.json",
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
            except CampaignError as error:
                print(f"campaign launch refused: {error}", file=sys.stderr)
                return 2
            print(f"point {point['id']} failed; inspect {log_path}", file=sys.stderr)
            return completed.returncode or 1
    manifest["status"] = "planned" if args.dry_run else "passed"
    try:
        _write_new(
            output_root / "launch.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except CampaignError as error:
        print(f"campaign launch refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
