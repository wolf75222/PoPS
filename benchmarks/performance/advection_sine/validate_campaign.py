#!/usr/bin/env python3
"""Validate a campaign and print its execution plan without running PoPS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import CampaignError, load_campaign, route_requires_mpi, route_uses_gpu, slurm_arguments


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--slurm-args",
        action="store_true",
        help="print one safe sbatch argument per line for the submit wrappers",
    )
    parser.add_argument(
        "--slurm-partition",
        choices=("short",),
        help="submit-time escalation only; it never edits the canonical campaign JSON",
    )
    parser.add_argument(
        "--slurm-time",
        help="walltime used only with --slurm-partition short, in [D-]HH:MM:SS form",
    )
    parser.add_argument("--json", action="store_true", help="print the normalized plan as JSON")
    parser.add_argument(
        "--field",
        choices=("id", "route", "platform", "dimension"),
        help="print one validated scalar for batch scripts",
    )
    return parser.parse_args()


def _plan(campaign: dict) -> dict:
    rows = []
    for point in campaign["points"]:
        launcher = [
            "srun",
            "--kill-on-bad-exit=1",
            f"--nodes={point['nodes']}",
            f"--ntasks={point['ranks']}",
            f"--cpus-per-task={point['threads']}",
            "--cpu-bind=cores",
        ]
        if route_uses_gpu(campaign["route"]):
            launcher.append("--gpus-per-task=1")
        rows.append(
            {
                **point,
                "workers": point["gpus"] or point["ranks"] * point["threads"],
                "cells_per_rank": point["cells"] / point["ranks"],
                "launcher": launcher,
            }
        )
    warnings = []
    if campaign["scaling"] == "weak_spatial":
        warnings.append(
            "weak_spatial fixes cells/rank and the SSPRK2 step count; dt and physical final time "
            "therefore vary with resolution, so compare computational work rather than one common T"
        )
    if route_uses_gpu(campaign["route"]):
        warnings.append("--gpus-per-task=1 is an allocation contract, not physical GPU UUID proof")
    if route_requires_mpi(campaign["route"]):
        warnings.append("the installed PoPS artifact must authenticate MPI_COMM_WORLD")
    return {
        "campaign": campaign["id"],
        "route": campaign["route"],
        "platform": campaign["platform"],
        "scaling": campaign["scaling"],
        "allocation": campaign["allocation"],
        "slurm_arguments_without_account": slurm_arguments(campaign),
        "warmups_per_point": campaign["warmups"],
        "repetitions_per_point": campaign["repetitions"],
        "points": rows,
        "warnings": warnings,
    }


def main() -> int:
    args = _arguments()
    try:
        campaign = load_campaign(args.campaign.resolve())
    except CampaignError as error:
        print(f"campaign invalid: {error}", file=sys.stderr)
        return 2
    if args.slurm_args:
        print(
            "\n".join(
                slurm_arguments(
                    campaign,
                    partition_override=args.slurm_partition,
                    time_override=args.slurm_time,
                )
            )
        )
        return 0
    if args.slurm_partition or args.slurm_time:
        print("campaign invalid: SLURM overrides require --slurm-args", file=sys.stderr)
        return 2
    if args.field:
        print(campaign[args.field])
        return 0
    plan = _plan(campaign)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    print(f"campaign: {plan['campaign']}")
    print(f"route: {plan['route']} on {plan['platform']} ({plan['scaling']})")
    print(
        "allocation: nodes={nodes} ranks={ranks} threads/task={threads} gpus/node={gpus_per_node}".format(
            **plan["allocation"]
        )
    )
    for point in plan["points"]:
        print(
            "  {id}: n={resolution} nodes={nodes} ranks={ranks} threads={threads} workers={workers}".format(
                **point
            )
        )
    for warning in plan["warnings"]:
        print(f"warning: {warning}")
    print("dry-run only: no process or SLURM job was launched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
