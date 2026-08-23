#!/usr/bin/env python3
"""Collect complete rank-owned public-Python measurements into scaling summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    CampaignError,
    ROMEO_CUDA_DSL_OPTFLAGS,
    ROUTES,
    SUMMARY_SCHEMA,
    expected_backend,
    load_campaign,
    route_uses_gpu,
)

MEASUREMENT_SCHEMA = "pops.performance.advection-sine.measurement.v3"
BUILD_RECEIPT_SCHEMA = "pops.performance.advection-sine.build-receipt.v4"
LAUNCH_SCHEMA = "pops.performance.advection-sine.launch.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"missing or invalid {label}: {path}: {error}") from error
    if type(value) is not dict:
        raise CampaignError(f"{label} must be one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_campaign_sha256(campaign: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(campaign, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _safe_relative_path(value: Any, label: str) -> str:
    if type(value) is not str:
        raise CampaignError(f"{label} must be a relative path")
    from pathlib import PurePosixPath

    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or path.as_posix() != value
        or ".." in path.parts
        or "." in path.parts
    ):
        raise CampaignError(f"{label} is not a safe relative path")
    return value


def _is_openmpi_launcher_receipt(value: Any) -> bool:
    """Accept OpenMPI's public mpirun name or its authenticated orterun target."""
    if type(value) is not dict or type(value.get("path")) is not str:
        return False
    path = Path(value["path"])
    return (
        path.is_absolute()
        and ".." not in path.parts
        and path.name in {"mpirun", "orterun"}
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is not None
        and isinstance(value.get("version"), str)
        and value["version"].startswith("mpirun (Open MPI)")
    )


def _lower_hex(value: Any, length: int, label: str) -> str:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise CampaignError(f"{label} must be lowercase {length}-hex")
    return value


def _expected_kokkos_configuration(route: str) -> dict[str, str]:
    expected = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "Kokkos_ENABLE_SERIAL": "ON",
    }
    if route in {"kokkos_serial", "kokkos_openmp", "kokkos_openmp_mpi"}:
        expected["Kokkos_ENABLE_CUDA"] = "OFF"
        expected["Kokkos_ENABLE_OPENMP"] = "ON" if "openmp" in route else "OFF"
    elif route in {"kokkos_cuda", "kokkos_cuda_mpi"}:
        expected.update(
            {
                "Kokkos_ENABLE_CUDA": "ON",
                "Kokkos_ENABLE_CUDA_LAMBDA": "ON",
                "Kokkos_ARCH_HOPPER90": "ON",
                "Kokkos_ENABLE_OPENMP": "OFF",
            }
        )
    else:
        raise CampaignError(f"unsupported Kokkos route {route}")
    return expected


def _require_cuda_loader_policy(cuda: dict[str, Any]) -> None:
    """Reject CUDA evidence not built with the canonical ROMEO DSL policy."""
    if cuda.get("native_loader") != {"pops_dsl_optflags": ROMEO_CUDA_DSL_OPTFLAGS}:
        raise CampaignError("CUDA build receipt lacks the canonical DSL optimization policy")


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(value):
        raise CampaignError(f"{label} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise CampaignError(f"{label} must be positive")
    return result


def _box_key(
    box: dict[str, Any], dimension: int, label: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if set(box) != {"lower", "upper_exclusive"}:
        raise CampaignError(f"{label}: local box has unsupported shape")
    lower, upper = tuple(box["lower"]), tuple(box["upper_exclusive"])
    if (
        len(lower) != dimension
        or len(upper) != dimension
        or any(type(v) is not int for v in lower + upper)
    ):
        raise CampaignError(f"{label}: local box is not exact-rank integer bounds")
    if any(hi <= lo for lo, hi in zip(lower, upper, strict=True)):
        raise CampaignError(f"{label}: local box is empty")
    return lower, upper


def _covers_exactly(rows: list[dict[str, Any]], cells: list[int], label: str) -> None:
    raw_boxes: list[dict[str, Any]] = []
    for row in rows:
        local_boxes = row.get("local_boxes")
        if type(local_boxes) is not list or any(type(box) is not dict for box in local_boxes):
            raise CampaignError(f"{label}: local_boxes must be a list of box objects")
        raw_boxes.extend(local_boxes)
    boxes = [_box_key(box, len(cells), label) for box in raw_boxes]
    if not boxes:
        raise CampaignError(f"{label}: no rank-owned local boxes")
    for low, high in boxes:
        if any(lo < 0 or hi > extent for lo, hi, extent in zip(low, high, cells, strict=True)):
            raise CampaignError(f"{label}: local box lies outside global cells")
    for index, (low, high) in enumerate(boxes):
        for other_low, other_high in boxes[:index]:
            if all(
                max(a, b) < min(c, d)
                for a, c, b, d in zip(low, high, other_low, other_high, strict=True)
            ):
                raise CampaignError(f"{label}: rank-owned boxes overlap")
    covered = sum(
        math.prod(hi - lo for lo, hi in zip(low, high, strict=True)) for low, high in boxes
    )
    if covered != math.prod(cells):
        raise CampaignError(f"{label}: rank-owned boxes do not cover the global grid")


def _require_receipts(
    root: Path, campaign: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _object(root / "source.manifest.json", "source manifest")
    build = _object(root / "build.receipt.json", "Release build receipt")
    if (
        source.get("schema") != "pops.performance.source-export.v1"
        or type(source.get("files")) is not list
    ):
        raise CampaignError("source manifest has an unsupported schema or inventory")
    if source.get("source_dirty") is not False:
        raise CampaignError(
            "source manifest is dirty; performance collection accepts only committed source evidence"
        )
    tree = source.get("tree_sha256")
    if type(tree) is not str or len(tree) != 64 or any(ch not in "0123456789abcdef" for ch in tree):
        raise CampaignError("source manifest lacks authenticated tree SHA-256")
    if build.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise CampaignError("build receipt has an unsupported schema")
    if build.get("build_type") != "Release" or build.get("workload") != "public-python":
        raise CampaignError("build receipt is not a Release public-Python build")
    source_facts = build.get("source")
    if type(source_facts) is not dict or source_facts.get("tree_sha256") != tree:
        raise CampaignError("build receipt is not linked to this authenticated source tree")
    if source_facts.get("manifest_sha256") != _sha256_file(root / "source.manifest.json"):
        raise CampaignError("build receipt is not linked to this source-manifest file")
    campaign_facts = build.get("campaign")
    if type(campaign_facts) is not dict:
        raise CampaignError("build receipt lacks campaign provenance")
    if any(campaign_facts.get(key) != campaign[key] for key in ("id", "route", "dimension")):
        raise CampaignError("build receipt campaign differs from collection request")
    path = _safe_relative_path(campaign_facts.get("path"), "build receipt campaign path")
    matching = [entry for entry in source.get("files", []) if entry.get("path") == path]
    if len(matching) != 1 or matching[0].get("sha256") != campaign_facts.get("sha256"):
        raise CampaignError("build receipt campaign is not authenticated by the source manifest")
    if campaign_facts.get("normalized_sha256") != _canonical_campaign_sha256(campaign):
        raise CampaignError("build receipt campaign normalization differs from collection request")
    cmake = build.get("cmake")
    expected_mpi = "ON" if ROUTES[campaign["route"]]["mpi"] else "OFF"
    expected_configuration = {
        "CMAKE_BUILD_TYPE": "Release",
        "POPS_BUILD_PYTHON": "ON",
        "POPS_USE_KOKKOS": "ON",
        "POPS_USE_MPI": expected_mpi,
        "POPS_USE_HDF5": "OFF",
        "POPS_NATIVE_DIM": str(campaign["dimension"]),
    }
    if type(cmake) is not dict or cmake.get("configured") != expected_configuration:
        raise CampaignError("build receipt CMake contract differs from the requested route")
    native = build.get("native_import")
    if type(native) is not dict or type(native.get("extension")) is not dict:
        raise CampaignError("build receipt lacks imported native-extension provenance")
    extension = native["extension"]
    extension_path = _safe_relative_path(extension.get("path"), "native extension path")
    if (
        extension_path
        != _safe_relative_path(extension.get("imported_path"), "imported native extension path")
        or type(extension.get("sha256")) is not str
        or extension.get("dimension") != campaign["dimension"]
    ):
        raise CampaignError("build receipt native extension was not the imported build-tree module")
    build_fingerprint = extension.get("build_fingerprint")
    if (
        type(build_fingerprint) is not str
        or len(build_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in build_fingerprint)
    ):
        raise CampaignError("build receipt native extension lacks one exact build fingerprint")
    if (
        extension.get("has_kokkos") is not True
        or extension.get("has_mpi") is not ROUTES[campaign["route"]]["mpi"]
    ):
        raise CampaignError("imported native extension backend facts differ from route")
    compiler = build.get("compiler")
    kokkos = build.get("kokkos")
    if (
        type(compiler) is not dict
        or not isinstance(compiler.get("version"), str)
        or type(kokkos) is not dict
        or not isinstance(kokkos.get("version"), str)
    ):
        raise CampaignError("build receipt lacks compiler or Kokkos version provenance")
    for field in ("config", "header", "version_file"):
        entry = kokkos.get(field)
        if (
            type(entry) is not dict
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is None
            or not _safe_relative_path(entry.get("path"), f"Kokkos {field} path")
        ):
            raise CampaignError(f"build receipt lacks Kokkos {field} hash provenance")
    core = kokkos.get("libkokkoscore")
    cmake_dir = kokkos.get("cmake_dir")
    authority = kokkos.get("source_authority")
    kokkos_build = kokkos.get("build")
    if (
        type(core) is not dict
        or core.get("kind") != "static-archive"
        or re.fullmatch(r"[0-9a-f]{64}", str(core.get("sha256", ""))) is None
        or not str(_safe_relative_path(core.get("path"), "Kokkos core-library path")).endswith(
            "/libkokkoscore.a"
        )
        or type(cmake_dir) is not dict
        or not _safe_relative_path(cmake_dir.get("path"), "PoPS Kokkos_DIR path")
    ):
        raise CampaignError("build receipt lacks installed Kokkos library/CMake authority")
    if (
        type(authority) is not dict
        or authority.get("kind") != "git-export"
        or authority.get("schema") != "pops.performance.kokkos-source-export.v1"
    ):
        raise CampaignError("ROMEO campaign requires authenticated Git-export Kokkos source authority")
    _lower_hex(authority.get("commit"), 40, "Kokkos source commit")
    _lower_hex(authority.get("archive_sha256"), 64, "Kokkos source archive SHA-256")
    _lower_hex(authority.get("archive_tree_sha256"), 64, "Kokkos extracted source tree SHA-256")
    if (
        type(kokkos_build) is not dict
        or type(kokkos_build.get("cache")) is not dict
        or not _safe_relative_path(kokkos_build["cache"].get("path"), "Kokkos build cache path")
        or re.fullmatch(r"[0-9a-f]{64}", str(kokkos_build["cache"].get("sha256", "")))
        is None
        or kokkos_build.get("source_tree_sha256") != authority.get("archive_tree_sha256")
        or kokkos_build.get("configured") != _expected_kokkos_configuration(campaign["route"])
    ):
        raise CampaignError("build receipt Kokkos build semantics differ from the requested route")
    mpi = build.get("mpi")
    cuda = build.get("cuda")
    if type(mpi) is not dict or mpi.get("enabled") is not ROUTES[campaign["route"]]["mpi"]:
        raise CampaignError("build receipt MPI state differs from route")
    if mpi["enabled"] and (
        type(mpi.get("launcher")) is not dict
        or not isinstance(mpi["launcher"].get("version"), str)
        or Path(str(mpi["launcher"].get("path", ""))).name != "srun"
        or not _is_openmpi_launcher_receipt(mpi.get("openmpi_launcher"))
    ):
        raise CampaignError("MPI route lacks SLURM/OpenMPI launcher provenance")
    if type(cuda) is not dict or cuda.get("enabled") is not route_uses_gpu(campaign["route"]):
        raise CampaignError("build receipt CUDA state differs from route")
    if cuda["enabled"] and (
        type(cuda.get("compiler")) is not dict
        or not isinstance(cuda["compiler"].get("version"), str)
    ):
        raise CampaignError("CUDA route lacks compiler provenance")
    if cuda["enabled"]:
        _require_cuda_loader_policy(cuda)
    return source, build


def _require_launch(root: Path, campaign: dict[str, Any]) -> None:
    launch = _object(root / "launch.json", "launch receipt")
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("campaign") != campaign["id"]
        or launch.get("launcher") != "slurm"
        or launch.get("dry_run") is not False
        or launch.get("status") != "passed"
    ):
        raise CampaignError("launch receipt is not a completed SLURM campaign")
    normalized = _object(root / "campaign.normalized.json", "normalized campaign")
    if normalized != campaign:
        raise CampaignError("normalized campaign differs from collection request")
    runs = launch.get("runs")
    if type(runs) is not list or len(runs) != len(campaign["points"]):
        raise CampaignError("launch receipt does not contain every campaign point")
    if any(type(run) is not dict or type(run.get("point")) is not str for run in runs):
        raise CampaignError("launch receipt contains an invalid point record")
    by_point = {run["point"]: run for run in runs}
    if set(by_point) != {point["id"] for point in campaign["points"]}:
        raise CampaignError("launch receipt has missing or duplicate point identifiers")
    for point in campaign["points"]:
        run = by_point[point["id"]]
        command = run.get("command")
        if (
            run.get("status") != "passed"
            or run.get("returncode") != 0
            or type(command) is not list
            or any(type(token) is not str for token in command)
        ):
            raise CampaignError(f"{point['id']}: launch receipt records a failed point")
        required = {
            "srun",
            f"--route={campaign['route']}",
            f"--campaign={campaign['id']}",
            f"--point={point['id']}",
            f"--nodes={point['nodes']}",
            f"--ntasks={point['ranks']}",
            f"--cpus-per-task={point['threads']}",
            f"--expected-ranks={point['ranks']}",
            f"--nodes={point['nodes']}",
            f"--threads={point['threads']}",
        }
        if not required.issubset(set(command)):
            raise CampaignError(f"{point['id']}: launch resources differ from campaign")


def _read_rank_rows(root: Path, point: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted((root / point["id"]).glob("rank-*.json"))
    if len(paths) != point["ranks"]:
        raise CampaignError(f"{point['id']}: expected exactly {point['ranks']} rank files")
    rows = [_object(path, "rank measurement") for path in paths]
    if any(row.get("schema") != MEASUREMENT_SCHEMA for row in rows):
        raise CampaignError(f"{point['id']}: unexpected rank measurement schema")
    if sorted(row.get("rank") for row in rows) != list(range(point["ranks"])):
        raise CampaignError(f"{point['id']}: rank files are missing or duplicated")
    for path, row in zip(paths, rows, strict=True):
        match = re.fullmatch(r"rank-([0-9]{5})", path.stem)
        if match is None or row["rank"] != int(match.group(1)):
            raise CampaignError(f"{point['id']}: filename rank differs from row.rank")
    return rows


def _expected_dt(campaign: dict[str, Any], point: dict[str, Any]) -> float:
    dimension = campaign["dimension"]
    speeds = {
        "x": (1.0,) + (0.0,) * (dimension - 1),
        "y": (0.0, 1.0) + (0.0,) * (dimension - 2),
        "z": (0.0, 0.0, 1.0),
        "diagonal": (1.0,) * dimension,
    }[campaign["mode"]]
    inverse_stable_dt = (
        sum(abs(speed) * cells for speed, cells in zip(speeds, point["resolution"], strict=True))
        / campaign["cfl"]
    )
    return math.ldexp(1.0, -math.ceil(math.log2(inverse_stable_dt)))


def _program_artifact(value: Any, label: str) -> dict[str, Any]:
    """Validate the Program binaries compiled for every timed public lifecycle."""
    if type(value) is not dict or set(value) != {
        "artifact_identity",
        "abi_key",
        "cache_key",
        "programs",
    }:
        raise CampaignError(f"{label}: program artifact receipt has an unsupported schema")
    for key in ("artifact_identity", "abi_key", "cache_key"):
        if type(value[key]) is not str or not value[key]:
            raise CampaignError(f"{label}: program artifact {key} is absent")
    programs = value["programs"]
    if type(programs) is not list or not programs:
        raise CampaignError(f"{label}: program artifact has no compiled Programs")
    canonical: list[dict[str, Any]] = []
    layouts: set[str] = set()
    for program in programs:
        if type(program) is not dict or set(program) != {"layout", "sha256", "size"}:
            raise CampaignError(f"{label}: Program receipt has an unsupported schema")
        layout, digest, size = program["layout"], program["sha256"], program["size"]
        if (
            type(layout) is not str
            or not layout
            or layout in layouts
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or isinstance(size, bool)
            or size <= 0
        ):
            raise CampaignError(f"{label}: Program binary authority is invalid")
        layouts.add(layout)
        canonical.append({"layout": layout, "sha256": digest, "size": size})
    if canonical != sorted(canonical, key=lambda program: program["layout"]):
        raise CampaignError(f"{label}: Program receipt order is not canonical")
    return {
        "artifact_identity": value["artifact_identity"],
        "abi_key": value["abi_key"],
        "cache_key": value["cache_key"],
        "programs": canonical,
    }


def _validate_point(
    campaign: dict[str, Any], point: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    label = f"{campaign['id']}/{point['id']}"
    expected_backend_name = {"serial": "Serial", "openmp": "OpenMP", "cuda": "Cuda"}[
        ROUTES[campaign["route"]]["backend"]
    ]
    samples_by_rank: list[list[float]] = []
    gpu_uuids: list[str] = []
    mass_drifts: list[float] = []
    l2_errors: list[float] = []
    program_artifacts: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("campaign") != campaign["id"]
            or row.get("point") != point["id"]
            or row.get("route") != campaign["route"]
        ):
            raise CampaignError(f"{label}: measurement identity differs from campaign")
        metadata, resources, problem, validation, timing = (
            row.get(name) for name in ("metadata", "resources", "problem", "validation", "timing")
        )
        if any(
            type(item) is not dict for item in (metadata, resources, problem, validation, timing)
        ):
            raise CampaignError(f"{label}: measurement sections must be JSON objects")
        program_artifacts.append(_program_artifact(row.get("program_artifact"), label))
        if metadata.get("execution_space") != expected_backend_name or not expected_backend(
            campaign["route"], metadata.get("execution_space")
        ):
            raise CampaignError(f"{label}: Kokkos backend differs from campaign")
        if metadata.get("mpi_ranks") != point["ranks"]:
            raise CampaignError(f"{label}: MPI rank count differs from campaign")
        if (
            ROUTES[campaign["route"]]["backend"] == "openmp"
            and metadata.get("execution_concurrency") != point["threads"]
        ):
            raise CampaignError(f"{label}: OpenMP concurrency differs from campaign")
        if resources != {
            "nodes": point["nodes"],
            "ranks": point["ranks"],
            "threads_per_rank": point["threads"],
        }:
            raise CampaignError(f"{label}: measurement resources differ from campaign")
        expected_problem = {
            "dimension": campaign["dimension"],
            "resolution": point["resolution"],
            "mode": campaign["mode"],
            "layout": "uniform",
            "block_size": campaign["block_size"],
            "cfl": campaign["cfl"],
            "steps": campaign["steps"],
        }
        if any(problem.get(key) != value for key, value in expected_problem.items()):
            raise CampaignError(f"{label}: problem differs from campaign")
        if _finite(problem.get("dt"), f"{label}.dt", positive=True) != _expected_dt(
            campaign, point
        ):
            raise CampaignError(f"{label}: dyadic fixed dt differs from campaign")
        if validation.get("timed") is not False or validation.get("passed") is not True:
            raise CampaignError(f"{label}: outside-timing validation failed")
        for name in ("initial_exact_errors", "final_exact_errors", "stationary_initial_errors"):
            norms = validation.get(name)
            if type(norms) is not dict or set(norms) != {"l1", "l2", "linf"}:
                raise CampaignError(f"{label}: {name} is not complete")
            for metric, value in norms.items():
                _finite(value, f"{label}.{name}.{metric}")
        if max(float(value) for value in validation["initial_exact_errors"].values()) > 5e-12:
            raise CampaignError(f"{label}: initial native state differs from analytic projection")
        stationary_l2 = _finite(
            validation["stationary_initial_errors"]["l2"], f"{label}.stationary_l2"
        )
        final_l2 = _finite(validation["final_exact_errors"]["l2"], f"{label}.final_l2")
        ratio = _finite(
            validation.get("final_to_stationary_l2_ratio"), f"{label}.dissipation_ratio"
        )
        if (
            stationary_l2 <= 1e-12
            or abs(ratio - final_l2 / stationary_l2) > 5e-14
            or ratio > 0.25
            or validation["final_exact_errors"]["linf"] >= 0.2
        ):
            raise CampaignError(f"{label}: final exact-error guards failed")
        if (
            _finite(validation.get("nonfinite_final_cells"), f"{label}.nonfinite_final_cells")
            != 0.0
        ):
            raise CampaignError(f"{label}: final field has nonfinite cells")
        if (
            abs(
                _finite(validation.get("native_integral"), f"{label}.native_integral")
                - _finite(validation.get("host_integral"), f"{label}.host_integral")
            )
            > 5e-12
        ):
            raise CampaignError(f"{label}: native and host integrals disagree")
        if (
            abs(
                _finite(
                    validation.get("initial_native_integral"),
                    f"{label}.initial_native_integral",
                )
                - _finite(
                    validation.get("initial_host_integral"),
                    f"{label}.initial_host_integral",
                )
            )
            > 5e-12
        ):
            raise CampaignError(f"{label}: initial native and host integrals disagree")
        mass_drift = _finite(validation.get("mass_drift"), f"{label}.mass_drift")
        if mass_drift > 5e-12:
            raise CampaignError(f"{label}: mass drift exceeds the campaign guard")
        mass_drifts.append(mass_drift)
        l2_errors.append(final_l2)
        if timing.get("metric") != "public_lifecycle_wall_seconds":
            raise CampaignError(f"{label}: not the public lifecycle metric")
        samples = timing.get("samples")
        if type(samples) is not list or len(samples) != campaign["repetitions"]:
            raise CampaignError(f"{label}: missing complete repetition samples")
        samples_by_rank.append(
            [_finite(value, f"{label}.samples", positive=True) for value in samples]
        )
        if route_uses_gpu(campaign["route"]):
            uuid = metadata.get("gpu_uuid")
            if (
                type(uuid) is not str
                or re.fullmatch(
                    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid
                )
                is None
            ):
                raise CampaignError(f"{label}: CUDA route requires one canonical GPU UUID per rank")
            if (
                type(metadata.get("gpu_device_ordinal")) is not int
                or metadata["gpu_device_ordinal"] < 0
            ):
                raise CampaignError(f"{label}: CUDA route requires one nonnegative device ordinal")
            if metadata.get("gpu_uuid_method") != "cudaGetDeviceProperties.uuid":
                raise CampaignError(f"{label}: CUDA UUID method is unauthenticated")
            if metadata.get("gpu_uuid_diagnostic") != "":
                raise CampaignError(f"{label}: CUDA UUID diagnostic must be empty")
            gpu_uuids.append(uuid)
        elif (
            metadata.get("gpu_device_ordinal") != -1
            or metadata.get("gpu_uuid") != ""
            or metadata.get("gpu_uuid_method") != "none"
            or metadata.get("gpu_uuid_diagnostic") != ""
        ):
            raise CampaignError(
                f"{label}: CPU route GPU identity must be the exact empty authority"
            )
        if (
            metadata.get("omp_proc_bind") != "spread"
            or metadata.get("omp_places") != "cores"
            or metadata.get("omp_dynamic") != "false"
        ):
            raise CampaignError(f"{label}: OpenMP placement authority differs from campaign")
    _covers_exactly(rows, point["resolution"], label)
    if route_uses_gpu(campaign["route"]) and len(set(gpu_uuids)) != point["ranks"]:
        raise CampaignError(f"{label}: CUDA UUIDs are not distinct across ranks")
    if max(mass_drifts) - min(mass_drifts) > 5e-14 or max(l2_errors) - min(l2_errors) > 5e-14:
        raise CampaignError(f"{label}: rank validation values are not globally consistent")
    if any(receipt != program_artifacts[0] for receipt in program_artifacts[1:]):
        raise CampaignError(f"{label}: Program binary authority differs between ranks")
    max_samples = [max(values) for values in zip(*samples_by_rank, strict=True)]
    median = statistics.median(max_samples)
    mad = statistics.median(abs(value - median) for value in max_samples)
    updates = point["cells"] * campaign["steps"] * 2
    workers = (
        point["ranks"] if route_uses_gpu(campaign["route"]) else point["ranks"] * point["threads"]
    )
    return {
        "point": point["id"],
        "route": campaign["route"],
        "execution_space": expected_backend_name,
        "nodes": point["nodes"],
        "ranks": point["ranks"],
        "threads_per_rank": point["threads"],
        "workers": workers,
        "resolution": point["resolution"],
        "global_cells": point["cells"],
        "block_size": campaign["block_size"],
        "cfl": campaign["cfl"],
        "steps": campaign["steps"],
        "cell_updates": updates,
        "median_seconds": median,
        "mad_seconds": mad,
        "min_seconds": min(max_samples),
        "max_seconds": max(max_samples),
        "throughput_cell_updates_per_second": updates / median,
        "mass_drift": mass_drifts[0],
        "l2_error": l2_errors[0],
        "program_artifact": program_artifacts[0],
    }


def _derive_scaling(campaign: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    baseline = min(rows, key=lambda row: row["workers"])
    for row in rows:
        if campaign["scaling"] == "strong":
            row["speedup"] = baseline["median_seconds"] / row["median_seconds"]
            row["ideal_speedup"] = row["workers"] / baseline["workers"]
            row["parallel_efficiency"] = row["speedup"] / row["ideal_speedup"]
        elif campaign["scaling"] == "weak_spatial":
            row["speedup"] = row["ideal_speedup"] = None
            row["parallel_efficiency"] = baseline["median_seconds"] / row["median_seconds"]
        else:
            row["speedup"] = row["ideal_speedup"] = row["parallel_efficiency"] = None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["resolution"] = "x".join(map(str, row["resolution"]))
            writer.writerow(payload)


def main() -> int:
    args = _arguments()
    try:
        campaign = load_campaign(args.campaign.resolve())
        root = args.input.resolve()
        source, build = _require_receipts(root, campaign)
        _require_launch(root, campaign)
        rows = [
            _validate_point(campaign, point, _read_rank_rows(root, point))
            for point in campaign["points"]
        ]
        _derive_scaling(campaign, rows)
    except CampaignError as error:
        print(f"collection refused: {error}", file=sys.stderr)
        return 2
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign["id"],
        "route": campaign["route"],
        "scaling": campaign["scaling"],
        "source_manifest": source,
        "build_receipt": build,
        "metric": {
            "primary": "public_lifecycle_wall_seconds",
            "rank_aggregation": "max",
            "statistic": "median_and_MAD",
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    if args.json.exists() or args.csv.exists():
        print("collection refused: summary destination already contains evidence", file=sys.stderr)
        return 2
    try:
        with args.json.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _write_csv(args.csv, rows)
    except FileExistsError:
        print("collection refused: summary destination already contains evidence", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
