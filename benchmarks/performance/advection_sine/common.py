#!/usr/bin/env python3
"""Shared, dependency-free contracts for the sine-advection performance campaign."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any


CAMPAIGN_SCHEMA = "pops.performance.advection-sine.campaign.v2"
RANK_SAMPLE_SCHEMA = "pops.performance.advection-sine.rank-sample.v3"
POINT_SAMPLE_SCHEMA = "pops.performance.advection-sine.point-sample.v2"
SUMMARY_SCHEMA = "pops.performance.advection-sine.summary.v2"

# ROMEO's CUDA native loader is intentionally compiled below the generic
# Release default. NVCC 12.6/CICC has crashed while compiling the real public
# Python DSL loader at -O3, -O2, and -O1; this changes only loader compilation, never the
# declared scientific workload or its acquisition matrix.
ROMEO_CUDA_DSL_OPTFLAGS = "-O0 -DNDEBUG"

# The performance suite is deliberately a closed scientific inventory.  These
# digests are of the source JSON after canonical JSON encoding (sorted keys and
# compact separators), not of a mutable run-time normalisation.  Consequently a
# convenient edit such as reducing 256^3 to 128^3, removing a scaling point, or
# changing a five-repeat acquisition is rejected before a job can be submitted.
CANONICAL_CAMPAIGN_DIGESTS = {
    "cuda_reference.json": "b2e489574591162c21fde4d3c6d703fdf08ec5446fad00a08ab5fdaf68317b7f",
    "serial_reference.json": "521c553e6c8219ede114c4634a142e31234e345522c7e6dd1fe5877c2feb8b26",
    "strong_cuda_mpi.json": "e45774e5e4a8d697bf9ca29ba40a43df06bfc10c3db7af005f87e7cf60f0953e",
    "strong_mpi_openmp.json": "fc3f5d8139d3c6bc87da0819e15b9486a6196ea119f9e88cd5c487aec00d86c9",
    "strong_openmp.json": "0d9cfe90fae3731d54d5f0b5495f32cadf3568fb24439897c0f75b4809217897",
    "weak_cuda_mpi.json": "7f8142bb45d7a4a0320177dd923c33e7f76ca669f5d480c3b21de33c803c57c8",
    "weak_mpi_openmp.json": "d945f25f35539553633c7cbe8ea0283b4a4b4704fd2255116830bd5ce0a1bc09",
}
CANONICAL_CAMPAIGN_FILENAMES = frozenset(CANONICAL_CAMPAIGN_DIGESTS)

CPU_CORES_PER_NODE = {"x64cpu": 192, "armgpu": 288}
GPU_PER_NODE = {"x64cpu": 0, "armgpu": 4}
ROUTES = {
    "kokkos_serial": {
        "platform": "x64cpu",
        "backend": "serial",
        "mpi": False,
        "gpu": False,
    },
    "kokkos_openmp": {
        "platform": "x64cpu",
        "backend": "openmp",
        "mpi": False,
        "gpu": False,
    },
    "kokkos_openmp_mpi": {
        "platform": "x64cpu",
        "backend": "openmp",
        "mpi": True,
        "gpu": False,
    },
    "kokkos_cuda": {
        "platform": "armgpu",
        "backend": "cuda",
        "mpi": False,
        "gpu": True,
    },
    "kokkos_cuda_mpi": {
        "platform": "armgpu",
        "backend": "cuda",
        "mpi": True,
        "gpu": True,
    },
}


class CampaignError(ValueError):
    """The campaign is incomplete, unsafe, or analytically incomparable."""


def _campaign_digest(data: object) -> str:
    """Return the stable content identity used by the closed campaign inventory."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_canonical_campaign_inventory(campaigns_dir: Path) -> None:
    """Require exactly the seven reviewed full-workload campaign files."""
    if campaigns_dir.is_symlink() or not campaigns_dir.is_dir():
        raise CampaignError("canonical campaign directory is unavailable")
    observed = {path.name for path in campaigns_dir.iterdir() if path.is_file() and not path.is_symlink()}
    if observed != CANONICAL_CAMPAIGN_FILENAMES:
        missing = sorted(CANONICAL_CAMPAIGN_FILENAMES - observed)
        extra = sorted(observed - CANONICAL_CAMPAIGN_FILENAMES)
        raise CampaignError(
            "canonical campaign inventory differs: "
            f"missing={missing or 'none'} extra={extra or 'none'}"
        )
    for filename, expected in CANONICAL_CAMPAIGN_DIGESTS.items():
        path = campaigns_dir / filename
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(f"cannot read canonical campaign {filename}: {error}") from error
        if _campaign_digest(raw) != expected:
            raise CampaignError(
                f"canonical campaign {filename} differs from the reviewed full matrix"
            )


def _enforce_canonical_campaign_file(path: Path, data: object) -> None:
    """Authenticate a named file when it belongs to a versioned campaigns directory."""
    if path.parent.name != "campaigns":
        return
    expected = CANONICAL_CAMPAIGN_DIGESTS.get(path.name)
    if expected is None:
        raise CampaignError(f"{path.name} is not in the seven-campaign canonical inventory")
    if _campaign_digest(data) != expected:
        raise CampaignError(
            f"canonical campaign {path.name} differs from the reviewed full matrix"
        )


def _exact_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise CampaignError(f"{label} must be a non-empty string")
    return value


def _duration_seconds(raw: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", raw)
    if match is None:
        raise CampaignError("slurm.time must use [D-]HH:MM:SS")
    days, hours, minutes, seconds = (int(token or 0) for token in match.groups())
    if (days and hours > 23) or minutes > 59 or seconds > 59:
        raise CampaignError("slurm.time has an invalid clock component")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _resolution(value: Any, dimension: int, label: str) -> list[int]:
    if type(value) is not list or len(value) != dimension:
        raise CampaignError(f"{label} must contain exactly {dimension} integers")
    return [_exact_int(cell, f"{label}[{axis}]", minimum=4) for axis, cell in enumerate(value)]


def _product(values: list[int]) -> int:
    return math.prod(values)


def load_campaign(path: Path) -> dict[str, Any]:
    """Load and fail-closed validate one campaign without importing PoPS."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign {path}: {error}") from error
    if type(data) is not dict or data.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignError(f"{path}: unexpected campaign schema")
    _enforce_canonical_campaign_file(path, data)

    campaign_id = _text(data.get("id"), "id")
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", campaign_id) is None:
        raise CampaignError("id must use lowercase letters, digits, '.', '_' or '-'")
    scaling = data.get("scaling")
    if scaling not in {"reference", "strong", "weak_spatial"}:
        raise CampaignError("scaling must be reference, strong, or weak_spatial")
    route = data.get("route")
    if route not in ROUTES:
        raise CampaignError(f"route must be one of {sorted(ROUTES)}")
    route_contract = ROUTES[route]
    platform_name = data.get("platform")
    if platform_name != route_contract["platform"]:
        raise CampaignError(
            f"route {route} requires platform {route_contract['platform']}, got {platform_name}"
        )
    dimension = _exact_int(data.get("dimension"), "dimension")
    if dimension not in {1, 2, 3}:
        raise CampaignError("dimension must be exactly 1, 2, or 3")
    if data.get("mode") not in {"x", "y", "z", "diagonal"}:
        raise CampaignError("mode must be x, y, z, or diagonal")
    if dimension == 1 and data["mode"] != "x":
        raise CampaignError("Dim1 accepts only mode=x")
    if dimension == 2 and data["mode"] == "z":
        raise CampaignError("Dim2 has no z propagation direction")
    if data.get("layout") != "uniform" or data.get("subcycling") != "synchronous":
        raise CampaignError(
            "the standalone performance workload currently supports only uniform/synchronous; "
            "AMR performance requires a qualified mixed-level hierarchy first"
        )
    _exact_int(data.get("block_size"), "block_size", minimum=2)
    _exact_int(data.get("steps"), "steps")
    cfl = data.get("cfl")
    if type(cfl) not in {float, int} or isinstance(cfl, bool) or not 0.0 < float(cfl) < 1.0:
        raise CampaignError("cfl must be a finite scalar strictly between 0 and 1")
    _exact_int(data.get("warmups"), "warmups", minimum=0)
    _exact_int(data.get("repetitions"), "repetitions", minimum=3)

    slurm = data.get("slurm")
    if type(slurm) is not dict:
        raise CampaignError("slurm must be an object")
    duration = _duration_seconds(_text(slurm.get("time"), "slurm.time"))
    memory = _text(slurm.get("mem"), "slurm.mem")
    if re.fullmatch(r"[1-9]\d*(?:[KMGTP])", memory, flags=re.IGNORECASE) is None:
        raise CampaignError("slurm.mem must be a positive SLURM size such as 64G")
    partition = slurm.get("partition")
    limits = {"instant": 3600, "short": 86400, "long": 30 * 86400}
    if partition not in limits or duration > limits[partition]:
        raise CampaignError("slurm.partition is inconsistent with slurm.time")
    if "account" in slurm:
        raise CampaignError("do not store an account in a campaign; submit with POPS_SLURM_ACCOUNT")

    points = data.get("points")
    if type(points) is not list or not points:
        raise CampaignError("points must be a non-empty list")
    common_resolution = data.get("resolution")
    if common_resolution is not None:
        common_resolution = _resolution(common_resolution, dimension, "resolution")
    seen_ids: set[str] = set()
    normalized_points: list[dict[str, Any]] = []
    for index, raw_point in enumerate(points):
        if type(raw_point) is not dict:
            raise CampaignError(f"points[{index}] must be an object")
        point_id = _text(raw_point.get("id"), f"points[{index}].id")
        if point_id in seen_ids:
            raise CampaignError(f"duplicate point id {point_id!r}")
        seen_ids.add(point_id)
        nodes = _exact_int(raw_point.get("nodes"), f"points[{index}].nodes")
        ranks = _exact_int(raw_point.get("ranks"), f"points[{index}].ranks")
        threads = _exact_int(raw_point.get("threads"), f"points[{index}].threads")
        resolution = _resolution(
            raw_point.get("resolution", common_resolution),
            dimension,
            f"points[{index}].resolution",
        )
        if not route_contract["mpi"] and ranks != 1:
            raise CampaignError(f"route {route} requires exactly one rank")
        if route == "kokkos_serial" and threads != 1:
            raise CampaignError("Kokkos Serial requires threads=1")
        if route_contract["gpu"] and threads < 1:
            raise CampaignError("GPU host thread count must remain positive")
        gpu_count = ranks if route_contract["gpu"] else 0
        if gpu_count > nodes * GPU_PER_NODE[platform_name]:
            raise CampaignError(
                f"point {point_id}: {gpu_count} GPUs exceed {GPU_PER_NODE[platform_name]}/node"
            )
        core_count = ranks * threads
        if core_count > nodes * CPU_CORES_PER_NODE[platform_name]:
            raise CampaignError(
                f"point {point_id}: {core_count} CPU workers exceed node hardware capacity"
            )
        normalized_points.append(
            {
                "id": point_id,
                "nodes": nodes,
                "ranks": ranks,
                "threads": threads,
                "gpus": gpu_count,
                "resolution": resolution,
                "cells": _product(resolution),
            }
        )

    if scaling in {"reference", "strong"}:
        resolutions = {tuple(point["resolution"]) for point in normalized_points}
        if len(resolutions) != 1:
            raise CampaignError(f"{scaling} scaling requires one fixed global resolution")
    if scaling == "reference" and len(normalized_points) != 1:
        raise CampaignError("reference campaigns contain exactly one point")
    if scaling == "strong" and len(normalized_points) < 2:
        raise CampaignError("strong scaling requires at least two points")
    worker_counts = {
        point["ranks"] if route_contract["gpu"] else point["ranks"] * point["threads"]
        for point in normalized_points
    }
    if len(worker_counts) != len(normalized_points):
        raise CampaignError("campaign points must have unique CPU-worker or GPU counts")
    if scaling == "weak_spatial":
        if any(point["cells"] % point["ranks"] for point in normalized_points):
            raise CampaignError("weak_spatial requires an integer cell count per MPI rank")
        cells_per_rank = {point["cells"] // point["ranks"] for point in normalized_points}
        if len(cells_per_rank) != 1:
            raise CampaignError("weak_spatial requires exactly constant global cells per MPI rank")
        if len({point["threads"] for point in normalized_points}) != 1:
            raise CampaignError("weak_spatial requires a fixed thread count per MPI rank")

    allocation_nodes = max(point["nodes"] for point in normalized_points)
    allocation_ranks = max(point["ranks"] for point in normalized_points)
    allocation_threads = max(point["threads"] for point in normalized_points)
    if allocation_ranks * allocation_threads > (
        allocation_nodes * CPU_CORES_PER_NODE[platform_name]
    ):
        raise CampaignError(
            "the enclosing SLURM allocation (max ranks x max threads) exceeds node capacity; "
            "split the campaign"
        )

    data["points"] = normalized_points
    data["allocation"] = {
        "nodes": allocation_nodes,
        "ranks": allocation_ranks,
        "threads": allocation_threads,
        "gpus_per_node": (
            max(math.ceil(point["gpus"] / point["nodes"]) for point in normalized_points)
            if route_contract["gpu"]
            else 0
        ),
    }
    return data


def slurm_arguments(
    campaign: dict[str, Any], *, partition_override: str | None = None, time_override: str | None = None
) -> list[str]:
    """Return one argument per line so Bash can read an array without eval."""
    allocation = campaign["allocation"]
    slurm = campaign["slurm"]
    partition = slurm["partition"]
    duration = slurm["time"]
    if partition_override is not None or time_override is not None:
        if partition_override != "short" or partition != "instant":
            raise CampaignError(
                "the only submit-time escalation is instant -> short; campaign JSON remains unchanged"
            )
        partition = partition_override
        if time_override is not None:
            if _duration_seconds(time_override) > 86400:
                raise CampaignError("short walltime override exceeds one day")
            duration = time_override
    arguments = [
        f"--constraint={campaign['platform']}",
        f"--partition={partition}",
        f"--time={duration}",
        f"--mem={slurm['mem']}",
        f"--nodes={allocation['nodes']}",
        f"--ntasks={allocation['ranks']}",
        f"--cpus-per-task={allocation['threads']}",
    ]
    if allocation["gpus_per_node"]:
        arguments.append(f"--gpus-per-node={allocation['gpus_per_node']}")
    return arguments


def expected_backend(route: str, observed: Any) -> bool:
    token = str(observed).lower()
    return token == ROUTES[route]["backend"]


def route_requires_mpi(route: str) -> bool:
    return bool(ROUTES[route]["mpi"])


def route_uses_gpu(route: str) -> bool:
    return bool(ROUTES[route]["gpu"])
