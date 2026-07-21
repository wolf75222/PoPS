#!/usr/bin/env python3
"""Partition the production Python runtime objects for parallel CI prewarming.

The final ``_pops`` build remains the single build and link authority.  These lanes only
populate ccache with object files compiled from the same configured Ninja graph, compiler,
flags, and source revision.  Splitting the large System and AMR template seams across runners
shortens a genuinely cold build without weakening optimisation or reusing linked binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


LANES = ("system", "amr-block", "amr-compiled")


def _runtime_objects(ninja_targets: str) -> list[str]:
    objects: list[str] = []
    for raw_line in ninja_targets.splitlines():
        target, separator, rule = raw_line.partition(": ")
        if not separator or not target.endswith(".o") or not rule.startswith("CXX_COMPILER_"):
            continue
        if target.startswith(
            (
                "src/CMakeFiles/pops_runtime_core_objects.dir/",
                "src/CMakeFiles/pops_runtime_system.dir/",
                "src/CMakeFiles/pops_runtime_amr.dir/",
                "src/CMakeFiles/pops_runtime_output.dir/",
            )
        ):
            objects.append(target)
    if not objects:
        raise SystemExit("configured Ninja graph exposes no production runtime objects")
    return sorted(set(objects))


def partition_runtime_objects(ninja_targets: str) -> dict[str, list[str]]:
    """Return a deterministic, disjoint, exact cover of production runtime objects."""
    lanes = {lane: [] for lane in LANES}
    for target in _runtime_objects(ninja_targets):
        if target.startswith("src/CMakeFiles/pops_runtime_amr.dir/"):
            if "/generated_seams/amr/compiled/" in target:
                lanes["amr-compiled"].append(target)
            else:
                lanes["amr-block"].append(target)
        else:
            lanes["system"].append(target)

    empty = [lane for lane, targets in lanes.items() if not targets]
    if empty:
        raise SystemExit("empty Python module prewarm lanes: " + ", ".join(empty))

    expected = _runtime_objects(ninja_targets)
    flattened = [target for lane in LANES for target in lanes[lane]]
    if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
        raise SystemExit("Python module prewarm lanes are not an exact disjoint object cover")
    return lanes


def configured_ninja_targets(build_dir: Path) -> str:
    if not (build_dir / "build.ninja").is_file():
        raise SystemExit(f"missing configured Ninja graph: {build_dir / 'build.ninja'}")
    try:
        return subprocess.check_output(
            ["ninja", "-C", str(build_dir), "-t", "targets", "all"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot inspect configured Ninja graph {build_dir}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lane_contract(build_dir: Path, lane: str, targets: list[str]) -> dict:
    commands_path = build_dir / "compile_commands.json"
    try:
        commands = json.loads(commands_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {commands_path}: {exc}") from exc
    by_output: dict[str, dict] = {}
    for entry in commands:
        if not isinstance(entry, dict) or not isinstance(entry.get("output"), str):
            continue
        output = Path(entry["output"])
        if not output.is_absolute():
            output = build_dir / output
        try:
            relative_output = output.resolve().relative_to(build_dir.resolve())
        except ValueError as exc:
            raise SystemExit(f"compile output escapes configured build directory: {output}") from exc
        by_output[str(relative_output)] = entry
    entries: dict[str, dict[str, str]] = {}
    compiler_identities: set[tuple[str, str]] = set()
    for target in targets:
        entry = by_output.get(target)
        if entry is None:
            raise SystemExit(f"compile_commands.json misses prewarm object {target}")
        command = entry.get("command")
        source_text = entry.get("file")
        if not isinstance(command, str) or not isinstance(source_text, str):
            raise SystemExit(f"invalid compile command for prewarm object {target}")
        source = Path(source_text)
        if not source.is_absolute():
            directory = entry.get("directory")
            if not isinstance(directory, str):
                raise SystemExit(f"relative prewarm source lacks a compile directory: {source}")
            source = Path(directory) / source
        if not source.is_file():
            raise SystemExit(f"missing prewarm source {source}")
        tokens = shlex.split(command)
        if not tokens:
            raise SystemExit(f"empty compile command for prewarm object {target}")
        compiler_index = 1 if Path(tokens[0]).name == "ccache" and len(tokens) > 1 else 0
        compiler_token = tokens[compiler_index]
        compiler_path = shutil.which(compiler_token) or compiler_token
        compiler = str(Path(compiler_path).resolve())
        try:
            compiler_version = subprocess.check_output(
                [compiler, "--version"], text=True, stderr=subprocess.STDOUT
            ).splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError) as exc:
            raise SystemExit(f"cannot identify prewarm compiler {compiler}: {exc}") from exc
        compiler_identities.add((compiler, compiler_version))
        entries[target] = {
            "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "source_sha256": _sha256(source),
        }
    if len(compiler_identities) != 1:
        raise SystemExit(f"prewarm lane {lane} uses multiple compiler identities")
    compiler, compiler_version = compiler_identities.pop()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot identify prewarm source revision: {exc}") from exc
    return {
        "schema": 1,
        "lane": lane,
        "revision": revision,
        "compiler": compiler,
        "compiler_version": compiler_version,
        "entries": entries,
    }


def verify_contracts(build_dir: Path, contract_paths: list[Path]) -> None:
    if len(contract_paths) != len(LANES):
        raise SystemExit(f"expected {len(LANES)} prewarm contracts, got {len(contract_paths)}")
    lanes = partition_runtime_objects(configured_ninja_targets(build_dir))
    seen: set[str] = set()
    for path in sorted(contract_paths):
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read prewarm contract {path}: {exc}") from exc
        lane = recorded.get("lane") if isinstance(recorded, dict) else None
        if lane not in LANES or lane in seen:
            raise SystemExit(f"invalid or duplicate prewarm lane in {path}: {lane!r}")
        seen.add(lane)
        current = lane_contract(build_dir, lane, lanes[lane])
        if recorded != current:
            raise SystemExit(
                f"prewarm contract {path} differs from the final compiler/flags/sources for {lane}"
            )
    if seen != set(LANES):
        raise SystemExit("prewarm contracts do not cover every declared lane")
    print("verified identical compiler, flags, revision and sources for all prewarm caches")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--lane", choices=LANES)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--contract-file", type=Path)
    parser.add_argument("--verify-contracts", type=Path, nargs="+")
    args = parser.parse_args()
    selected_modes = int(bool(args.lane)) + int(args.verify) + int(bool(args.verify_contracts))
    if selected_modes != 1:
        parser.error("select exactly one of --lane, --verify, or --verify-contracts")

    lanes = partition_runtime_objects(configured_ninja_targets(args.build_dir))
    if args.verify_contracts:
        verify_contracts(args.build_dir, args.verify_contracts)
        return 0
    if args.verify:
        counts = ", ".join(f"{lane}={len(lanes[lane])}" for lane in LANES)
        print(f"verified exact Python runtime object partition: {counts}")
        return 0
    if args.contract_file is None:
        parser.error("--lane requires --contract-file")
    contract = lane_contract(args.build_dir, args.lane, lanes[args.lane])
    args.contract_file.parent.mkdir(parents=True, exist_ok=True)
    args.contract_file.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for target in lanes[args.lane]:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
