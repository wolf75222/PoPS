#!/usr/bin/env python3
"""Fail-closed collector for five macOS sample and Time Profiler leaves."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from profile_contract import (
    PROFILE_SCHEMA,
    ProfileContractError,
    command_sha256,
    profile_command,
    read_json,
    sha256,
    source_manifest_receipt,
    tree_digest,
    write_json_new,
)

REPETITIONS = range(1, 6)
_FRAME = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _leaf(root: Path, tool: str, repetition: int) -> Path:
    leaf = root / tool / ("rep%02d" % repetition)
    if not leaf.is_dir():
        raise ProfileContractError("missing %s leaf %s" % (tool, leaf))
    return leaf


def _ready_receipt(leaf: Path) -> dict[str, Any]:
    ready = read_json(leaf / "ready.json", "READY receipt")
    required = {
        "campaign",
        "source",
        "python_package",
        "artifact",
        "program_artifact",
        "native",
        "build",
        "runtime",
        "host",
        "command",
    }
    if (
        ready.get("schema") != PROFILE_SCHEMA
        or ready.get("phase") != "ready_after_bind_warmup"
        or type(ready.get("nonce")) is not str
        or type(ready.get("pid")) is not int
        or type(ready.get("provenance")) is not dict
        or set(ready["provenance"]) != required
        or any(type(ready["provenance"][name]) is not dict for name in required)
    ):
        raise ProfileContractError("%s lacks authenticated post-warmup READY provenance" % leaf)
    native = ready["provenance"]["native"]
    native_required = {
        "manifest_sha256",
        "path",
        "sha256",
        "dimension",
        "version",
        "abi_key",
        "build_fingerprint",
        "has_mpi",
        "has_kokkos",
    }
    if (
        set(native) != native_required
        or native.get("dimension") != 3
        or native.get("has_mpi") is not False
        or native.get("has_kokkos") is not True
        or any(
            type(native[key]) is not str or not native[key]
            for key in native_required - {"dimension", "has_mpi", "has_kokkos"}
        )
        or len(native["build_fingerprint"]) != 64
        or any(character not in "0123456789abcdef" for character in native["build_fingerprint"])
    ):
        raise ProfileContractError("%s native variant receipt is incomplete" % leaf)
    build = ready["provenance"]["build"]
    if (
        set(build)
        != {
            "filename",
            "sha256",
            "source_tree_sha256",
            "campaign",
            "native_sha256",
            "native_build_fingerprint",
            "cmake_cache_sha256",
        }
        or build.get("filename") != "build.receipt.json"
        or build.get("source_tree_sha256") != ready["provenance"]["source"].get("tree_sha256")
        or build.get("campaign") != ready["provenance"]["campaign"].get("id")
        or build.get("native_sha256") != native.get("sha256")
        or build.get("native_build_fingerprint") != native.get("build_fingerprint")
        or any(
            type(build[key]) is not str or len(build[key]) != 64
            for key in ("sha256", "cmake_cache_sha256")
        )
    ):
        raise ProfileContractError("%s build receipt does not bind source and native extension" % leaf)
    runtime = ready["provenance"]["runtime"]
    if (
        runtime.get("kokkos_backend") != "OpenMP"
        or runtime.get("kokkos_concurrency") != 8
        or runtime.get("mpi_active") is not False
        or runtime.get("mpi_ranks") != 1
    ):
        raise ProfileContractError("%s runtime differs from canonical OpenMP t8" % leaf)
    return ready


def _command_and_rank(
    leaf: Path, ready: dict[str, Any], source_root: Path
) -> tuple[dict[str, Any], Path]:
    command = read_json(leaf / "command.json", "canonical command")
    required = {"schema", "phase", "argv", "sha256", "output_dir"}
    if set(command) != required or command.get("schema") != PROFILE_SCHEMA:
        raise ProfileContractError("%s command receipt has an unsupported schema" % leaf)
    if command.get("phase") != "canonical_command" or type(command.get("argv")) is not list:
        raise ProfileContractError("%s lacks a canonical public command" % leaf)
    argv = command["argv"]
    if command.get("sha256") != command_sha256(argv):
        raise ProfileContractError("%s canonical command digest differs" % leaf)
    if type(command.get("output_dir")) is not str:
        raise ProfileContractError("%s canonical output directory is invalid" % leaf)
    expected_output = (leaf / "rank-output").resolve()
    if Path(command["output_dir"]).resolve() != expected_output:
        raise ProfileContractError("%s command targets a non-owned rank output directory" % leaf)
    campaign_value = ready["provenance"]["campaign"].get("path")
    if type(campaign_value) is not str:
        raise ProfileContractError("%s READY campaign path is invalid" % leaf)
    try:
        expected_argv = profile_command(
            campaign_path=Path(campaign_value),
            python=Path(argv[0]),
            output_dir=expected_output,
            source_root=source_root,
        )
    except (OSError, ValueError) as error:
        raise ProfileContractError("%s cannot reconstruct its canonical command" % leaf) from error
    if argv != expected_argv or ready["provenance"]["command"] != {
        "argv": argv,
        "sha256": command["sha256"],
        "output_dir": str(expected_output),
    }:
        raise ProfileContractError("%s READY is not bound to its canonical command" % leaf)
    rank_path = expected_output / "rank-00000.json"
    rank = read_json(rank_path, "rank-owned performance result")
    timing = rank.get("timing")
    validation = rank.get("validation")
    if (
        rank.get("schema") != "pops.performance.advection-sine.measurement.v3"
        or rank.get("rank") != 0
        or rank.get("campaign") != ready["provenance"]["campaign"]["id"]
        or rank.get("point") != ready["provenance"]["campaign"]["point"]
        or type(timing) is not dict
        or timing.get("metric") != "public_lifecycle_wall_seconds"
        or type(validation) is not dict
        or validation.get("passed") is not True
    ):
        raise ProfileContractError("%s has no accepted canonical rank result" % leaf)
    return command, rank_path


def _completed_receipt(leaf: Path, ready: dict[str, Any]) -> dict[str, Any]:
    receipt = read_json(leaf / "worker.receipt.json", "worker receipt")
    if (
        receipt.get("schema") != PROFILE_SCHEMA
        or receipt.get("phase") != "completed_public_lifecycle"
    ):
        raise ProfileContractError("%s has no completed public-lifecycle receipt" % leaf)
    if receipt.get("returncode") != 0 or receipt.get("nonce") != ready["nonce"]:
        raise ProfileContractError("%s public lifecycle failed or has no nonce" % leaf)
    completed = receipt.get("completed_unix_seconds")
    if type(completed) not in {int, float}:
        raise ProfileContractError("%s completion timestamp is absent" % leaf)
    return receipt


def _tool_receipt(
    leaf: Path, tool: str, ready: dict[str, Any], completed: dict[str, Any]
) -> dict[str, Any]:
    receipt = read_json(leaf / "tool.receipt.json", "acquisition receipt")
    required = {
        "schema",
        "phase",
        "tool",
        "nonce",
        "started_unix_seconds",
        "ended_unix_seconds",
        "target_completed_unix_seconds",
        "target_reaped_unix_seconds",
        "profiler_reaped_unix_seconds",
        "target_completed_during_acquisition",
        "attachment_proof",
    }
    if set(receipt) != required:
        raise ProfileContractError("%s acquisition receipt has an unsupported schema" % leaf)
    if (
        receipt["schema"] != PROFILE_SCHEMA
        or receipt["phase"] != "acquisition_complete"
        or receipt["tool"] != tool
        or receipt["nonce"] != ready["nonce"]
        or receipt["target_completed_during_acquisition"] is not True
        or receipt["attachment_proof"]
        != {"sample": "sample_header_before_go_after_stop", "xctrace": "xctrace_notify_before_go"}[
            tool
        ]
    ):
        raise ProfileContractError("%s did not authenticate its complete acquisition" % leaf)
    observed = [
        receipt["started_unix_seconds"],
        receipt["ended_unix_seconds"],
        receipt["target_completed_unix_seconds"],
        receipt["target_reaped_unix_seconds"],
        receipt["profiler_reaped_unix_seconds"],
        completed["completed_unix_seconds"],
    ]
    if any(type(value) not in {int, float} for value in observed):
        raise ProfileContractError("%s acquisition timestamps are invalid" % leaf)
    started, ended, target_completed, target_reaped, profiler_reaped, worker_completed = observed
    if not (
        started <= target_completed <= target_reaped <= profiler_reaped == ended
        and target_completed == worker_completed
    ):
        raise ProfileContractError("%s target did not finish during acquisition" % leaf)
    return receipt


def _profiler_exit(leaf: Path, tool: str, completed: dict[str, Any]) -> dict[str, Any]:
    receipt = read_json(leaf / "profiler.exit.json", "profiler exit")
    if (
        set(receipt) != {"schema", "phase", "tool", "returncode", "exited_unix_seconds"}
        or receipt.get("schema") != PROFILE_SCHEMA
        or receipt.get("phase") != "profiler_exit"
        or receipt.get("tool") != tool
        or receipt.get("returncode") != 0
        or type(receipt.get("exited_unix_seconds")) not in {int, float}
        or receipt["exited_unix_seconds"] < completed["completed_unix_seconds"]
    ):
        raise ProfileContractError("%s profiler exited before the complete public lifecycle" % leaf)
    return receipt


def _sample_frames_and_stacks(path: Path) -> tuple[Counter[str], Counter[tuple[str, ...]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ProfileContractError("sample report is absent or empty: %s" % path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "Call graph:") + 1
        end = next(
            index
            for index, line in enumerate(lines[start:], start)
            if line.strip() == "Binary Images:"
        )
    except StopIteration as error:
        raise ProfileContractError(
            "sample report lacks its Call graph/Binary Images boundaries"
        ) from error
    parsed: list[tuple[int, int, str]] = []
    for line in lines[start:end]:
        match = _FRAME.match(line)
        if match is not None:
            parsed.append((len(line) - len(line.lstrip()), int(match.group(1)), match.group(2)))
    frames: Counter[str] = Counter()
    stacks: Counter[tuple[str, ...]] = Counter()
    ancestors: list[tuple[int, str]] = []
    # `sample` call graphs repeat inclusive counts at every ancestor.  Count
    # only collapsed leaves, whose next parsed frame is not more indented.
    for index, (indent, weight, frame) in enumerate(parsed):
        while ancestors and indent <= ancestors[-1][0]:
            ancestors.pop()
        stack = [name for _, name in ancestors] + [frame]
        next_indent = parsed[index + 1][0] if index + 1 < len(parsed) else -1
        if next_indent <= indent:
            frames[frame] += weight
            stacks[tuple(stack)] += weight
        ancestors.append((indent, frame))
    if not frames:
        raise ProfileContractError("sample report has no parseable weighted frames: %s" % path)
    return frames, stacks


def _time_profiler_toc(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ProfileContractError("invalid Time Profiler TOC: %s" % error) from error
    if root.tag.rsplit("}", 1)[-1] != "trace-toc":
        raise ProfileContractError("TOC root must be trace-toc")
    schemas = [
        element.attrib.get("schema")
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "table"
    ]
    if schemas != ["time-profile"]:
        raise ProfileContractError("TOC must contain exactly the time-profile table schema")
    return schemas


def _image_name(frame: str) -> str:
    marker = " (in "
    if marker in frame and frame.endswith(")"):
        return frame.split(marker, 1)[1][:-1]
    return "unattributed"


def collect(root: Path) -> dict[str, Any]:
    aggregate: Counter[str] = Counter()
    stack_aggregate: Counter[tuple[str, ...]] = Counter()
    leaves: dict[str, list[dict[str, Any]]] = {"sample": [], "xctrace": []}
    observed_nonces: set[str] = set()
    provenance: dict[str, Any] | None = None
    rank_outputs: set[Path] = set()
    source_root = root / "source-tree"
    source = source_manifest_receipt(
        manifest_path=root / "source.manifest.json", source_root=source_root
    )
    for repetition in REPETITIONS:
        sample_leaf = _leaf(root, "sample", repetition)
        sample_ready = _ready_receipt(sample_leaf)
        sample_command, sample_rank_path = _command_and_rank(sample_leaf, sample_ready, source_root)
        sample_receipt = _completed_receipt(sample_leaf, sample_ready)
        sample_profiler_exit = _profiler_exit(sample_leaf, "sample", sample_receipt)
        sample_tool_receipt = _tool_receipt(sample_leaf, "sample", sample_ready, sample_receipt)
        sample_path = sample_leaf / "sample.txt"
        frames, stacks = _sample_frames_and_stacks(sample_path)
        aggregate.update(frames)
        stack_aggregate.update(stacks)
        leaves["sample"].append(
            {
                "rep": repetition,
                "receipt_sha256": sha256(sample_leaf / "worker.receipt.json"),
                "ready_sha256": sha256(sample_leaf / "ready.json"),
                "command_sha256": sha256(sample_leaf / "command.json"),
                "acquisition_receipt_sha256": sha256(sample_leaf / "tool.receipt.json"),
                "rank_result_sha256": sha256(sample_rank_path),
                "sample_sha256": sha256(sample_path),
                "command": sample_command,
                "acquisition": sample_tool_receipt,
                "profiler_exit": sample_profiler_exit,
                "nonce": sample_ready["nonce"],
            }
        )
        trace_leaf = _leaf(root, "xctrace", repetition)
        trace_ready = _ready_receipt(trace_leaf)
        trace_command, trace_rank_path = _command_and_rank(trace_leaf, trace_ready, source_root)
        trace_receipt = _completed_receipt(trace_leaf, trace_ready)
        trace_profiler_exit = _profiler_exit(trace_leaf, "xctrace", trace_receipt)
        trace_tool_receipt = _tool_receipt(trace_leaf, "xctrace", trace_ready, trace_receipt)
        trace = trace_leaf / "time-profiler.trace"
        toc = trace_leaf / "toc.txt"
        if not trace.is_dir() or not toc.is_file() or toc.stat().st_size == 0:
            raise ProfileContractError("%s lacks a Time Profiler trace or TOC" % trace_leaf)
        toc_schemas = _time_profiler_toc(toc)
        trace_tree_sha256, trace_file_count = tree_digest(trace)
        leaves["xctrace"].append(
            {
                "rep": repetition,
                "receipt_sha256": sha256(trace_leaf / "worker.receipt.json"),
                "ready_sha256": sha256(trace_leaf / "ready.json"),
                "command_sha256": sha256(trace_leaf / "command.json"),
                "acquisition_receipt_sha256": sha256(trace_leaf / "tool.receipt.json"),
                "rank_result_sha256": sha256(trace_rank_path),
                "trace_toc_sha256": sha256(toc),
                "trace_tree_sha256": trace_tree_sha256,
                "trace_file_count": trace_file_count,
                "toc_schemas": toc_schemas,
                "command": trace_command,
                "acquisition": trace_tool_receipt,
                "profiler_exit": trace_profiler_exit,
                "nonce": trace_ready["nonce"],
            }
        )
        observed_nonces.update((sample_ready["nonce"], trace_ready["nonce"]))
        rank_outputs.update((sample_rank_path.resolve(), trace_rank_path.resolve()))
        for candidate in (sample_ready["provenance"], trace_ready["provenance"]):
            if candidate["source"] != source:
                raise ProfileContractError("READY source tree differs from the exported manifest")
            python_package = candidate["python_package"]
            expected_package = source_root / "python" / "pops" / "__init__.py"
            if python_package != {
                "path": "python/pops/__init__.py",
                "sha256": sha256(expected_package),
            }:
                raise ProfileContractError(
                    "READY Python package differs from the exported authority"
                )
            common_candidate = {key: value for key, value in candidate.items() if key != "command"}
            if provenance is None:
                provenance = common_candidate
            elif common_candidate != provenance:
                raise ProfileContractError("READY provenance differs between profile processes")
    if len(observed_nonces) != 10:
        raise ProfileContractError("every profiling process must have one unique READY/GO nonce")
    if len(rank_outputs) != 10:
        raise ProfileContractError("every profiling process must publish one distinct rank result")
    images: Counter[str] = Counter()
    for frame, weight in aggregate.items():
        images[_image_name(frame)] += weight
    return {
        "schema": PROFILE_SCHEMA + ".summary",
        "tools": {"sample": 5, "xctrace_time_profiler": 5},
        "leaves": leaves,
        "sample_top15": [
            {"frame": frame, "weight": weight} for frame, weight in aggregate.most_common(15)
        ],
        "sample_image_composition": [
            {"image": image, "weight": weight} for image, weight in images.most_common()
        ],
        # The stack strings come solely from indentation in the sample Call graph;
        # their weights are the exclusive leaf counts, never inclusive ancestors.
        "sample_leaf_stacks": [
            {"frames": list(stack), "weight": weight}
            for stack, weight in stack_aggregate.most_common()
        ],
        "provenance": provenance,
        "scaling_excluded": True,
    }


def main() -> int:
    args = _arguments()
    try:
        write_json_new(args.output, collect(args.input))
    except ProfileContractError as error:
        print("macOS profile collection refused: %s" % error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
