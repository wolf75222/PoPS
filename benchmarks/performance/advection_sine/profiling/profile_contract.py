"""Fail-closed macOS profiling contracts for the canonical public Python case."""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import platform
import re
import secrets
import socket
from pathlib import Path, PurePosixPath
from typing import Any

PROFILE_SCHEMA = "pops.performance.advection-sine.macos-profile.v1"
PROFILE_COMPLETE_SCHEMA = PROFILE_SCHEMA + ".complete.v1"
CANONICAL_CAMPAIGN = "strong_openmp.json"
CANONICAL_POINT = "t8"
CANONICAL = {
    "dimension": 3,
    "resolution": [128, 128, 128],
    "block_size": 32,
    "cfl": 0.40,
    "steps": 32,
    "warmups": 1,
    "repetitions": 5,
    "threads": 8,
    "ranks": 1,
}


class ProfileContractError(ValueError):
    """The profile cannot become evidence."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def source_manifest_receipt(*, manifest_path: Path, source_root: Path) -> dict[str, object]:
    """Authenticate the extracted Git-visible tree used by the public case."""
    manifest_path = manifest_path.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileContractError("cannot read macOS source manifest") from error
    required = {
        "schema",
        "base_sha",
        "source_dirty",
        "tree_sha256",
        "archive_sha256",
        "archive_format",
        "file_count",
        "files",
    }
    if type(manifest) is not dict or set(manifest) != required:
        raise ProfileContractError("macOS source manifest schema is unsupported")
    entries = manifest["files"]
    if (
        manifest["schema"] != "pops.performance.source-export.v1"
        or type(entries) is not list
        or manifest["file_count"] != len(entries)
        or type(manifest["tree_sha256"]) is not str
        or _canonical_sha256(entries) != manifest["tree_sha256"]
    ):
        raise ProfileContractError("macOS source manifest is internally inconsistent")
    if manifest["source_dirty"] is not False:
        raise ProfileContractError(
            "macOS profiling requires a clean source manifest (source_dirty=false)"
        )
    required_paths = {
        "benchmarks/performance/advection_sine/advection_sine.py",
        "benchmarks/performance/advection_sine/support.py",
        "benchmarks/performance/advection_sine/profiling/profile_contract.py",
        "benchmarks/performance/advection_sine/profiling/ready_go.py",
        "helpers/verification/sine_wave.py",
        "python/pops/__init__.py",
    }
    if any(type(entry) is not dict or type(entry.get("path")) is not str for entry in entries):
        raise ProfileContractError("macOS source manifest file inventory is invalid")
    indexed = {entry["path"]: entry for entry in entries}
    if len(indexed) != len(entries):
        raise ProfileContractError("macOS source manifest repeats a file path")
    for relative in required_paths:
        entry = indexed.get(relative)
        path = source_root / relative
        if (
            type(entry) is not dict
            or entry.get("type") != "file"
            or type(entry.get("sha256")) is not str
            or path.is_symlink()
            or not path.is_file()
            or sha256(path) != entry["sha256"]
        ):
            raise ProfileContractError("macOS source export lacks authenticated %s" % relative)
    return {
        "base_sha": manifest["base_sha"],
        "source_dirty": manifest["source_dirty"],
        "tree_sha256": manifest["tree_sha256"],
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "file_count": manifest["file_count"],
    }


def native_variant_receipt(module: object) -> dict[str, object]:
    """Bind the loaded extension bytes and runtime facts to variants.json."""
    origin = getattr(module, "__file__", None)
    if type(origin) is not str:
        raise ProfileContractError("selected native module has no origin")
    extension = Path(origin)
    if (
        extension.is_symlink()
        or not extension.is_file()
        or extension.parent.name not in {"dim1", "dim2", "dim3"}
    ):
        raise ProfileContractError("selected native extension path is not canonical")
    native_root = extension.parent.parent
    manifest_path = native_root / "variants.json"
    if native_root.name != "_native" or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ProfileContractError("selected native variants manifest is unavailable")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = (
            extension.resolve(strict=True).relative_to(native_root.resolve(strict=True)).as_posix()
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ProfileContractError("cannot read selected native variant authority") from error
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "variants"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 2
        or type(manifest["variants"]) is not list
    ):
        raise ProfileContractError("selected native variants manifest schema is unsupported")
    rows = manifest["variants"]
    required = {
        "dimension",
        "path",
        "sha256",
        "version",
        "abi_key",
        "build_fingerprint",
        "has_mpi",
        "has_kokkos",
    }
    if not rows or any(
        type(candidate) is not dict or set(candidate) != required for candidate in rows
    ):
        raise ProfileContractError("selected native variants manifest rows are malformed")
    fingerprints = {candidate["build_fingerprint"] for candidate in rows}
    if len(fingerprints) != 1:
        raise ProfileContractError("selected native variants disagree on their build fingerprint")
    matching = [candidate for candidate in rows if candidate["path"] == relative]
    if len(matching) != 1:
        raise ProfileContractError("selected native extension is absent from variants manifest")
    row = matching[0]
    if set(row) != required or row["path"] != relative or sha256(extension) != row["sha256"]:
        raise ProfileContractError("selected native extension bytes differ from variants manifest")
    fingerprint = row["build_fingerprint"]
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ProfileContractError("selected native build fingerprint is malformed")
    if (
        getattr(module, "__native_dimension__", None) != row["dimension"]
        or getattr(module, "__version__", None) != row["version"]
        or not callable(getattr(module, "abi_key", None))
        or module.abi_key() != row["abi_key"]
        or getattr(module, "__has_mpi__", None) is not row["has_mpi"]
        or getattr(module, "__has_kokkos__", None) is not row["has_kokkos"]
    ):
        raise ProfileContractError("selected native runtime facts differ from variants manifest")
    if getattr(module, "__build_fingerprint__", None) != fingerprint:
        raise ProfileContractError(
            "selected native build fingerprint differs from variants manifest"
        )
    expected_leafs = {"_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES}
    parts = PurePosixPath(relative).parts
    if len(parts) != 2 or parts[1] not in expected_leafs:
        raise ProfileContractError("selected native extension file name is unsupported")
    return {
        "manifest_sha256": sha256(manifest_path),
        "path": relative,
        "sha256": row["sha256"],
        "dimension": row["dimension"],
        "version": row["version"],
        "abi_key": row["abi_key"],
        "build_fingerprint": fingerprint,
        "has_mpi": row["has_mpi"],
        "has_kokkos": row["has_kokkos"],
    }


def program_artifact_receipt(artifact: object) -> dict[str, object]:
    """Seal every compiled Program binary used by the public artifact."""
    identity = getattr(getattr(artifact, "artifact_identity", None), "token", None)
    abi_key = getattr(artifact, "abi_key", None)
    cache_key = getattr(artifact, "cache_key", None)
    paths = getattr(artifact, "layout_program_paths", None)
    if (
        type(identity) is not str
        or not identity
        or type(abi_key) is not str
        or not abi_key
        or type(cache_key) is not str
        or not cache_key
        or type(paths) is not dict
        or not paths
    ):
        raise ProfileContractError("compiled artifact lacks identity/ABI/cache/program authorities")
    programs = []
    for layout, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if type(layout) is not str or not layout or path.is_symlink() or not path.is_file():
            raise ProfileContractError("compiled Program path is not a regular file")
        programs.append({"layout": layout, "sha256": sha256(path), "size": path.stat().st_size})
    return {
        "artifact_identity": identity,
        "abi_key": abi_key,
        "cache_key": cache_key,
        "programs": programs,
    }


def exported_build_receipt(
    *, receipt_path: Path, source: dict[str, object], campaign_id: str, native: dict[str, object]
) -> dict[str, object]:
    """Bind a macOS extension to the immutable exported tree that built it.

    A variants manifest proves extension bytes, but cannot by itself prove which
    headers produced those bytes.  The build receipt supplies that missing
    source/build relationship and is deliberately mandatory for profiling.
    """
    receipt_path = receipt_path.resolve(strict=True)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileContractError("cannot read exported-source build receipt") from error
    build_source = receipt.get("source") if type(receipt) is dict else None
    build_campaign = receipt.get("campaign") if type(receipt) is dict else None
    native_import = receipt.get("native_import") if type(receipt) is dict else None
    extension = native_import.get("extension") if type(native_import) is dict else None
    cmake = receipt.get("cmake") if type(receipt) is dict else None
    cache = cmake.get("cache") if type(cmake) is dict else None
    kokkos = receipt.get("kokkos") if type(receipt) is dict else None
    kokkos_authority = kokkos.get("source_authority") if type(kokkos) is dict else None
    kokkos_core = kokkos.get("libkokkoscore") if type(kokkos) is dict else None
    kokkos_cmake_dir = kokkos.get("cmake_dir") if type(kokkos) is dict else None
    kokkos_core_kind = kokkos_core.get("kind") if type(kokkos_core) is dict else None
    kokkos_core_path = kokkos_core.get("path") if type(kokkos_core) is dict else None
    kokkos_core_relative = (
        PurePosixPath(kokkos_core_path) if isinstance(kokkos_core_path, str) else None
    )
    safe_core_path = (
        kokkos_core_relative is not None
        and not kokkos_core_relative.is_absolute()
        and bool(kokkos_core_relative.parts)
        and ".." not in kokkos_core_relative.parts
    )
    kokkos_core_name = kokkos_core_relative.name if safe_core_path else ""
    kokkos_cmake_path = kokkos_cmake_dir.get("path") if type(kokkos_cmake_dir) is dict else None
    kokkos_cmake_relative = (
        PurePosixPath(kokkos_cmake_path) if isinstance(kokkos_cmake_path, str) else None
    )
    safe_cmake_path = (
        kokkos_cmake_relative is not None
        and not kokkos_cmake_relative.is_absolute()
        and bool(kokkos_cmake_relative.parts)
        and ".." not in kokkos_cmake_relative.parts
    )
    canonical_core = (
        kokkos_core_kind == "static-archive" and kokkos_core_name == "libkokkoscore.a"
    ) or (
        kokkos_core_kind == "shared-library"
        and (
            re.fullmatch(r"libkokkoscore(?:\.[0-9]+)*\.dylib", kokkos_core_name) is not None
            or re.fullmatch(r"libkokkoscore\.so(?:\.[0-9]+)*", kokkos_core_name) is not None
        )
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "pops.performance.advection-sine.build-receipt.v3"
        or not isinstance(build_source, dict)
        or build_source.get("tree_sha256") != source.get("tree_sha256")
        or not isinstance(build_campaign, dict)
        or build_campaign.get("id") != campaign_id
        or not isinstance(extension, dict)
        or extension.get("sha256") != native.get("sha256")
        or extension.get("dimension") != native.get("dimension")
        or extension.get("build_fingerprint") != native.get("build_fingerprint")
        or extension.get("has_mpi") != native.get("has_mpi")
        or extension.get("has_kokkos") is not True
        or not isinstance(cache, dict)
        or not isinstance(cache.get("sha256"), str)
        or not isinstance(kokkos_authority, dict)
        or kokkos_authority.get("kind") != "installed-distribution"
        or not isinstance(kokkos_core, dict)
        or not canonical_core
        or re.fullmatch(r"[0-9a-f]{64}", str(kokkos_core.get("sha256", ""))) is None
        or not isinstance(kokkos_cmake_dir, dict)
        or not safe_cmake_path
    ):
        raise ProfileContractError(
            "native extension cannot be proven to have been built from the exported source"
        )
    return {
        "filename": receipt_path.name,
        "sha256": sha256(receipt_path),
        "source_tree_sha256": build_source["tree_sha256"],
        "campaign": campaign_id,
        "native_sha256": extension["sha256"],
        "native_build_fingerprint": extension["build_fingerprint"],
        "cmake_cache_sha256": cache["sha256"],
    }


def authenticated_profile_provenance(
    *,
    artifact: object,
    args: object,
    case_path: Path,
    campaign_value: str | None,
    expected_command_sha256: str | None,
    expected_source_tree_sha256: str | None,
    native_module: object,
    pops_module: object,
    python_executable: Path,
    runtime_report: object,
    source_manifest_value: str | None,
    source_root_value: str | None,
    build_receipt_value: str | None,
) -> dict[str, object]:
    """Authenticate one post-bind macOS profiling target without case mechanics.

    The public case supplies all dynamic facts explicitly.  Keeping the checks
    here makes the scientific script linear while preserving the exact READY
    provenance schema consumed by the runner and collector.
    """
    if not campaign_value:
        raise ProfileContractError("macOS profiling requires a canonical campaign path")
    if (
        type(expected_command_sha256) is not str
        or len(expected_command_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_command_sha256)
    ):
        raise ProfileContractError("macOS profiling requires an exact canonical command digest")
    if not source_manifest_value or not source_root_value:
        raise ProfileContractError("macOS profiling requires an authenticated exported source tree")

    campaign_path = Path(campaign_value).resolve(strict=True)
    source_root = Path(source_root_value).resolve(strict=True)
    plan = canonical_plan(campaign_path)
    expected = plan["canonical"]
    dimension = len(args.resolution)
    observed = {
        "dimension": dimension,
        "resolution": list(args.resolution),
        "block_size": args.block_size,
        "cfl": args.cfl,
        "steps": args.steps,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "threads": args.threads,
        "ranks": args.expected_ranks,
    }
    if (
        observed != expected
        or args.campaign != plan["campaign"]["id"]
        or args.point != plan["point"]["id"]
        or args.route != plan["campaign"]["route"]
        or args.nodes != plan["point"]["nodes"]
        or args.mode != plan["campaign"]["mode"]
    ):
        raise ProfileContractError("profiled command differs from the canonical campaign point")
    canonical_argv = profile_command(
        campaign_path=campaign_path,
        python=python_executable,
        output_dir=args.output_dir,
        source_root=source_root,
    )
    if command_sha256(canonical_argv) != expected_command_sha256:
        raise ProfileContractError("profiled command does not match the runner canonical command")

    source = source_manifest_receipt(
        manifest_path=Path(source_manifest_value), source_root=source_root
    )
    expected_case = (
        source_root / "benchmarks" / "performance" / "advection_sine" / "advection_sine.py"
    )
    if case_path.resolve() != expected_case.resolve():
        raise ProfileContractError("profiled case is not executing from the authenticated export")
    package_path = Path(getattr(pops_module, "__file__", "")).resolve()
    expected_package = source_root / "python" / "pops" / "__init__.py"
    if package_path != expected_package.resolve():
        raise ProfileContractError(
            "profiled Python package is not loaded from the authenticated export"
        )
    if expected_source_tree_sha256 and source["tree_sha256"] != expected_source_tree_sha256:
        raise ProfileContractError("profiled source tree differs from the runner export authority")
    semantic_identity = getattr(artifact, "semantic_identity", None)
    artifact_token = getattr(semantic_identity, "token", None)
    if type(artifact_token) is not str or not artifact_token:
        raise ProfileContractError("profiled artifact does not expose a semantic identity token")
    if type(runtime_report) is not dict:
        raise ProfileContractError("runtime environment report must be a dictionary")
    native = native_variant_receipt(native_module)
    if not build_receipt_value:
        raise ProfileContractError(
            "macOS profiling requires a build receipt made from the exported source tree"
        )
    build = exported_build_receipt(
        receipt_path=Path(build_receipt_value),
        source=source,
        campaign_id=plan["campaign"]["id"],
        native=native,
    )
    return {
        "campaign": {
            "path": str(campaign_path),
            "sha256": sha256(campaign_path),
            "id": plan["campaign"]["id"],
            "point": plan["point"]["id"],
        },
        "source": source,
        "python_package": {
            "path": "python/pops/__init__.py",
            "sha256": sha256(package_path),
        },
        "artifact": {"semantic_identity": artifact_token},
        "program_artifact": program_artifact_receipt(artifact),
        "native": native,
        "build": build,
        "runtime": runtime_report,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "command": {
            "argv": canonical_argv,
            "sha256": expected_command_sha256,
            "output_dir": str(args.output_dir.resolve()),
        },
    }


def command_sha256(argv: list[str]) -> str:
    """Return the canonical digest of one exact public-process command line."""
    if not argv or any(type(argument) is not str or not argument for argument in argv):
        raise ProfileContractError("profile command must be a non-empty string vector")
    encoded = json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tree_digest(path: Path) -> tuple[str, int]:
    """Hash every regular file in one `.trace` package, with canonical paths."""
    if not path.is_dir() or path.is_symlink():
        raise ProfileContractError("trace package must be one real directory")
    digest = hashlib.sha256()
    count = 0
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise ProfileContractError("trace package must not contain symlinks")
        if not item.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        count += 1
    if count == 0:
        raise ProfileContractError("trace package contains no regular files")
    return digest.hexdigest(), count


def _profile_inventory(root: Path) -> list[dict[str, object]]:
    """Hash every regular evidence file except the self-referential seal."""
    root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.name == "COMPLETE.json" and path.parent == root:
            continue
        if path.is_symlink():
            raise ProfileContractError("profile evidence must not contain symlinks")
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    if not entries:
        raise ProfileContractError("profile evidence inventory is empty")
    return entries


def _inventory_sha256(entries: list[dict[str, object]]) -> str:
    return _canonical_sha256(entries)


def profile_figure_publication_path(profile_root: Path) -> Path:
    """Choose the immutable profile's figure directory as a sibling.

    ``COMPLETE.json`` inventories the complete evidence root.  A figure
    directory beneath that root would therefore alter the inventory after the
    receipt has been written.  Keeping the publication beside the evidence
    lets the figures cite the receipt without becoming unsealed evidence.
    """
    root = profile_root.absolute()
    if root.name in {"", ".", ".."}:
        raise ProfileContractError("profile root has no safe sibling figure path")
    return root.parent / (root.name + ".figures")


def external_profile_publication_path(*, profile_root: Path, publication_root: Path) -> Path:
    """Reject a publication target inside a sealed profile evidence root."""
    root = profile_root.resolve(strict=True)
    publication = publication_root.resolve(strict=False)
    try:
        publication.relative_to(root)
    except ValueError:
        return publication
    raise ProfileContractError("profile figures must be published outside the sealed evidence root")


def create_profile_complete_receipt(root: Path) -> dict[str, object]:
    """Seal ten acquired public lifecycles, their source and native identities."""
    root = root.resolve(strict=True)
    summary_path = root / "summary.json"
    summary = read_json(summary_path, "profile summary")
    tools = summary.get("tools")
    provenance = summary.get("provenance")
    leaves = summary.get("leaves")
    if (
        summary.get("schema") != PROFILE_SCHEMA + ".summary"
        or tools != {"sample": 5, "xctrace_time_profiler": 5}
        or not isinstance(provenance, dict)
        or not isinstance(leaves, dict)
        or set(leaves) != {"sample", "xctrace"}
        or any(not isinstance(leaves[name], list) or len(leaves[name]) != 5 for name in leaves)
    ):
        raise ProfileContractError("profile summary does not prove ten complete acquisitions")
    source = provenance.get("source")
    native = provenance.get("native")
    build = provenance.get("build")
    if (
        not isinstance(source, dict)
        or not isinstance(native, dict)
        or not isinstance(build, dict)
        or source.get("source_dirty") is not False
        or build.get("source_tree_sha256") != source.get("tree_sha256")
        or build.get("native_sha256") != native.get("sha256")
        or build.get("native_build_fingerprint") != native.get("build_fingerprint")
    ):
        raise ProfileContractError("profile summary lacks a linked source/native build identity")
    entries = _profile_inventory(root)
    receipt = {
        "schema": PROFILE_COMPLETE_SCHEMA,
        "runs": {"sample": 5, "xctrace_time_profiler": 5},
        "summary": {"filename": "summary.json", "sha256": sha256(summary_path)},
        "source": source,
        "native": native,
        "build": build,
        "evidence": {"sha256": _inventory_sha256(entries), "files": entries},
    }
    write_json_new(root / "COMPLETE.json", receipt)
    return receipt


def verify_profile_complete_receipt(root: Path) -> dict[str, object]:
    """Fail closed when any profile evidence was changed after publication."""
    root = root.resolve(strict=True)
    receipt = read_json(root / "COMPLETE.json", "profile COMPLETE receipt")
    summary_path = root / "summary.json"
    summary = read_json(summary_path, "profile summary")
    expected = receipt.get("evidence")
    if (
        receipt.get("schema") != PROFILE_COMPLETE_SCHEMA
        or receipt.get("runs") != {"sample": 5, "xctrace_time_profiler": 5}
        or not isinstance(expected, dict)
        or not isinstance(expected.get("files"), list)
        or not isinstance(expected.get("sha256"), str)
    ):
        raise ProfileContractError("profile COMPLETE receipt has an unsupported schema")
    source = source_manifest_receipt(
        manifest_path=root / "source.manifest.json", source_root=root / "source-tree"
    )
    provenance = summary.get("provenance") if isinstance(summary, dict) else None
    summary_source = provenance.get("source") if isinstance(provenance, dict) else None
    if (
        summary.get("schema") != PROFILE_SCHEMA + ".summary"
        or not isinstance(summary_source, dict)
        or summary_source.get("source_dirty") is not False
        or summary_source != source
        or receipt.get("source") != source
        or receipt.get("source") != summary_source
        or receipt.get("summary") != {"filename": "summary.json", "sha256": sha256(summary_path)}
    ):
        raise ProfileContractError(
            "profile COMPLETE summary is not bound to the clean source manifest"
        )
    observed = _profile_inventory(root)
    if observed != expected["files"] or _inventory_sha256(observed) != expected["sha256"]:
        raise ProfileContractError("profile COMPLETE inventory differs after publication")
    return receipt


def canonical_plan(campaign_path: Path) -> dict[str, Any]:
    """Load the one auditable macOS profiling point from the versioned campaign."""
    import sys

    harness = Path(__file__).resolve().parents[1]
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    from common import load_campaign

    campaign = load_campaign(campaign_path)
    if campaign_path.name != CANONICAL_CAMPAIGN or campaign["route"] != "kokkos_openmp":
        raise ProfileContractError("macOS profiling accepts only the canonical OpenMP campaign")
    point = next((row for row in campaign["points"] if row["id"] == CANONICAL_POINT), None)
    if point is None:
        raise ProfileContractError("canonical t8 point is absent")
    observed = {
        "dimension": campaign["dimension"],
        "resolution": campaign["resolution"],
        "block_size": campaign["block_size"],
        "cfl": campaign["cfl"],
        "steps": campaign["steps"],
        "warmups": campaign["warmups"],
        "repetitions": campaign["repetitions"],
        "threads": point["threads"],
        "ranks": point["ranks"],
    }
    if observed != CANONICAL:
        raise ProfileContractError("campaign no longer matches the canonical macOS point")
    return {"campaign": campaign, "point": point, "canonical": observed}


def profile_command(
    *, campaign_path: Path, python: Path, output_dir: Path, source_root: Path | None = None
) -> list[str]:
    """Build the only allowed macOS profile command, from versioned JSON."""
    plan = canonical_plan(campaign_path.resolve())
    campaign = plan["campaign"]
    point = plan["point"]
    root = (
        source_root.resolve(strict=True)
        if source_root is not None
        else Path(__file__).resolve().parents[4]
    )
    case = root / "benchmarks" / "performance" / "advection_sine" / "advection_sine.py"
    if not case.is_file():
        raise ProfileContractError("authenticated source tree lacks the public advection case")
    interpreter = python.resolve(strict=True)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ProfileContractError("profile Python interpreter is not executable")
    return [
        str(interpreter),
        "-B",
        str(case),
        "--resolution=" + ",".join(str(value) for value in point["resolution"]),
        "--mode=" + campaign["mode"],
        "--route=" + campaign["route"],
        "--campaign=" + campaign["id"],
        "--point=" + point["id"],
        "--expected-ranks=" + str(point["ranks"]),
        "--nodes=" + str(point["nodes"]),
        "--threads=" + str(point["threads"]),
        "--block-size=" + str(campaign["block_size"]),
        "--steps=" + str(campaign["steps"]),
        "--cfl=" + str(campaign["cfl"]),
        "--warmups=" + str(campaign["warmups"]),
        "--repetitions=" + str(campaign["repetitions"]),
        "--output-dir=" + str(output_dir.resolve()),
    ]


def fresh_nonce() -> str:
    """Return a non-guessable READY/GO capability for one process only."""
    return secrets.token_hex(32)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Publish one receipt without replacing previous evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = None
    temporary = path.parent / (".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise ProfileContractError("refusing to overwrite %s" % path) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileContractError("invalid %s: %s" % (label, error)) from error
    if type(row) is not dict:
        raise ProfileContractError("%s must be a JSON object" % label)
    return row
