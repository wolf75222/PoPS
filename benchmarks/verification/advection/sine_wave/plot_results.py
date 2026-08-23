#!/usr/bin/env python3
"""Publish figures only from one sealed 37-case sine-wave COMPLETE.json manifest."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from helpers.verification import convergence_orders  # noqa: E402

SCHEMA_VERSION = "pops.sine-wave.v3"
PLOT_PUBLICATION_SCHEMA = "pops.sine-wave.plot-publication.v1"
COMPLETE_SCHEMA = "pops.sine-wave.matrix-complete.v1"
MATRIX_SOURCE_AUTHORITY_SCHEMA = "pops.sine-wave.matrix-source-authority.v1"
CONVERGENCE_NORMS = ("l1", "l2", "linf")
# Recomputing a JSON-recorded reduction from an authenticated NPZ can cross a
# platform boundary.  These bounds cover only round-off in reductions/logs;
# they remain far below any scientifically meaningful convergence difference.
ERROR_RECEIPT_TOLERANCE = 1024.0 * np.finfo(np.float64).eps
ORDER_RECEIPT_TOLERANCE = 4096.0 * np.finfo(np.float64).eps
BUILD_SOURCE_AUTHORITY_ROOTS = (
    "CMakeLists.txt",
    "cmake",
    "include",
    "src",
    "python",
    "pyproject.toml",
    "schemas",
    "scripts",
)
NATIVE_ABI_VARIANT_FIELDS = frozenset({"dim", "mpi", "mpi_abi"})
NATIVE_ABI_REQUIRED_FIELDS = frozenset(
    {"compiler", "std", "headers", "kokkos", "stdlib", "dim", "mpi"}
)
MATRIX_PATH = Path(__file__).with_name("matrix.v1.json")
VISUAL_CASE_IDS = ("d1-face", "d2-cf-subcycled", "d3-cf-subcycled")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete",
        type=Path,
        action="append",
        required=True,
        metavar="COMPLETE.json",
        help="the one sealed COMPLETE.json produced by the complete matrix driver",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        help="output directory (default: one isolated directory per sealed publication)",
    )
    parser.add_argument("--fps", type=int, default=5, help="GIF frame rate (1 to 30, default: 5)")
    args = parser.parse_args()
    if len(args.complete) != 1:
        parser.error("provide exactly one --complete COMPLETE.json manifest")
    args.complete = args.complete[0]
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_exists(path: Path) -> bool:
    """Include dangling symlinks in the no-clobber check."""
    return os.path.lexists(path)


def _require_fresh_path(path: Path) -> None:
    if path.is_symlink() or _path_exists(path):
        raise FileExistsError("refusing to replace an existing plot artifact: %s" % path)


def _save_figure(figure, target: Path, *, dpi: int) -> Path:
    """Save only inside a newly-created publication staging directory."""
    _require_fresh_path(target)
    figure.savefig(target, dpi=dpi)
    return target


def _input_manifest(
    paths: list[Path],
    metadata_paths: list[Path],
    metadata: list[dict[str, object]],
    *,
    fps: int,
) -> dict[str, object]:
    """Describe exactly the authenticated files from which figures are derived."""
    if not (len(paths) == len(metadata_paths) == len(metadata)):
        raise ValueError("plot publication inputs have inconsistent cardinality")
    inputs: list[dict[str, object]] = []
    for data_path, metadata_path, item in zip(paths, metadata_paths, metadata, strict=True):
        result_identity = item.get("result_identity")
        source_fingerprint = item.get("source_fingerprint")
        if not isinstance(result_identity, str) or not isinstance(source_fingerprint, str):
            raise ValueError("plot publication input lacks an authenticated result identity")
        inputs.append(
            {
                "data_filename": data_path.name,
                "data_sha256": _sha256(data_path),
                "metadata_filename": metadata_path.name,
                "metadata_sha256": _sha256(metadata_path),
                "result_identity": result_identity,
                "source_fingerprint": source_fingerprint,
            }
        )
    payload: dict[str, object] = {
        "schema": PLOT_PUBLICATION_SCHEMA,
        "fps": fps,
        "renderer": {
            "filename": Path(__file__).name,
            "sha256": _sha256(Path(__file__)),
        },
        "inputs": inputs,
    }
    payload["publication_identity"] = _canonical_sha256(payload)
    return payload


def _sealed_publication_manifest(
    paths: list[Path],
    metadata_paths: list[Path],
    metadata: list[dict[str, object]],
    *,
    complete_path: Path,
    complete: dict[str, object],
    fps: int,
) -> dict[str, object]:
    """Extend an input manifest with COMPLETE without hashing its provisional identity."""
    manifest = _input_manifest(paths, metadata_paths, metadata, fps=fps)
    manifest.pop("publication_identity")
    manifest["complete"] = {
        "filename": complete_path.name,
        "sha256": _sha256(complete_path),
        "case_count": complete["case_count"],
        "matrix_sha256": complete["matrix_sha256"],
    }
    manifest["publication_identity"] = _canonical_sha256(manifest)
    return manifest


def _canonical_publication_target(target: Path) -> Path:
    """Refuse a lexical target first; only its parent may be canonicalised."""
    lexical_target = target.absolute()
    _require_fresh_path(lexical_target)
    parent = lexical_target.parent
    if _path_exists(parent) and not parent.is_dir():
        raise NotADirectoryError(parent)
    parent.mkdir(parents=True, exist_ok=True)
    canonical_target = parent.resolve(strict=True) / lexical_target.name
    _require_fresh_path(canonical_target)
    return canonical_target


def _create_staging_directory(target: Path) -> Path:
    """Allocate a unique sibling directory without touching the final target."""
    target = _canonical_publication_target(target)
    return Path(tempfile.mkdtemp(prefix=".%s.staging-" % target.name, dir=target.parent))


def _write_publication_manifest(staging: Path, manifest: dict[str, object]) -> Path:
    """Add media hashes before the one-shot directory publication."""
    target = staging / "plot_manifest.json"
    _require_fresh_path(target)
    files = [
        {
            "path": path.relative_to(staging).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path != target
    ]
    if not files:
        raise ValueError("plot publication produced no media artifacts")
    published = dict(manifest)
    published["media"] = files
    target.write_text(json.dumps(published, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def _atomic_rename_noreplace(staging: Path, target: Path) -> None:
    """Use the platform no-replace rename primitive; never emulate it with os.rename."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(target)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication requires libc renameat2 on Linux",
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, target_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication requires renamex_np on macOS",
            ) from error
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, 0x0004)  # RENAME_EXCL
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace plot publication is unavailable on this platform",
        )
    if result == 0:
        return
    failure = ctypes.get_errno()
    if failure in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            failure, "refusing to replace an existing plot publication", str(target)
        )
    raise OSError(failure, os.strerror(failure), str(target))


def _publish_staging_directory(staging: Path, target: Path) -> Path:
    """Publish atomically, refusing even a target created after the preliminary check."""
    target = _canonical_publication_target(target)
    _atomic_rename_noreplace(staging, target)
    return target


_RESULT_CONFIGURATION_KEYS = (
    "case",
    "dimension",
    "resolution",
    "mode",
    "wave_numbers",
    "velocity",
    "epsilon",
    "probe_time",
    "period",
    "cycles",
    "final_time",
    "layout",
    "subcycling",
    "block_size",
    "patch_marker",
    "coverage",
    "mpi",
    "mpi_ranks",
    "mpi_topology",
    "timeline",
    "amr_diagnostics",
)


def _validate_result_identity(
    path: Path,
    data: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    result_identity = metadata.get("result_identity")
    archive_identity = data.get("result_identity")
    inputs = metadata.get("result_identity_inputs")
    if (
        not isinstance(result_identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", result_identity) is None
        or archive_identity is None
        or archive_identity.shape != ()
        or str(archive_identity.item()) != result_identity
        or not path.stem.endswith("_rid" + result_identity[:16])
        or not isinstance(inputs, dict)
        or set(inputs)
        != {
            "schema_version",
            "configuration",
            "method",
            "execution",
            "source_fingerprint",
            "artifact",
        }
        or not hmac.compare_digest(result_identity, _canonical_sha256(inputs))
    ):
        raise ValueError("JSON/NPZ result identity is missing, stale, or inconsistent")
    configuration = inputs.get("configuration")
    metrics = metadata.get("metrics")
    provenance = metadata.get("provenance")
    if (
        inputs.get("schema_version") != SCHEMA_VERSION
        or not isinstance(configuration, dict)
        or set(configuration) != set(_RESULT_CONFIGURATION_KEYS)
        or configuration != {key: metadata.get(key) for key in _RESULT_CONFIGURATION_KEYS}
        or not isinstance(metrics, dict)
        or inputs.get("method") != metrics.get("method")
        or not isinstance(provenance, dict)
        or inputs.get("execution") != provenance.get("execution")
        or inputs.get("artifact") != provenance.get("artifact")
    ):
        raise ValueError("result identity inputs do not match the authenticated metadata")
    source_fingerprint = inputs.get("source_fingerprint")
    source = provenance.get("source")
    campaign = provenance.get("campaign")
    timeline = metadata.get("timeline")
    execution = provenance.get("execution")
    if (
        not isinstance(source_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_fingerprint) is None
        or metadata.get("source_fingerprint") != source_fingerprint
        or not isinstance(source, dict)
        or source.get("fingerprint") != source_fingerprint
        or not isinstance(execution, dict)
        or not isinstance(execution.get("runtime"), dict)
        or not isinstance(execution.get("environment"), dict)
        or not isinstance(execution.get("host"), dict)
        or not isinstance(campaign, dict)
        or not isinstance(timeline, dict)
        or campaign.get("mpi_ranks") != metadata.get("mpi_ranks")
        or campaign.get("time_snapshots") != metadata.get("time_snapshots")
        or campaign.get("time_snapshots") != timeline.get("frames")
        or campaign.get("timeline_times") != timeline.get("times")
    ):
        raise ValueError("execution/source/snapshot provenance is incomplete or inconsistent")
    runtime = execution["runtime"]
    environment = execution["environment"]
    if any(
        key not in runtime
        for key in (
            "has_kokkos",
            "kokkos_backend",
            "kokkos_device",
            "kokkos_shared_space",
            "field_memory_space",
            "kokkos_concurrency",
            "mpi_ranks",
        )
    ) or any(key not in environment for key in ("OMP_NUM_THREADS", "KOKKOS_NUM_THREADS")):
        raise ValueError("backend, execution-space, concurrency, or thread provenance is missing")
    amr_diagnostics = metadata.get("amr_diagnostics")
    if (
        not isinstance(amr_diagnostics, dict)
        or set(amr_diagnostics) != {"interface_mask", "regrid_events"}
        or not isinstance(amr_diagnostics["interface_mask"], dict)
        or amr_diagnostics["interface_mask"].get("connectivity") != "face"
        or amr_diagnostics["interface_mask"].get("periodic") is not True
        or not isinstance(amr_diagnostics["regrid_events"], dict)
        or amr_diagnostics["regrid_events"].get("source") != "simulation.amr.explain_regrid()"
        or not isinstance(amr_diagnostics["regrid_events"].get("time_semantics"), str)
    ):
        raise ValueError("AMR diagnostic labels are missing or unsupported")


def _load(
    path: Path, metadata_path: Path | None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    candidate = metadata_path or path.with_suffix(".json")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    loaded = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("metadata JSON must contain one object")
    metadata: dict[str, object] = loaded
    if metadata.get("data") != path.name:
        raise ValueError("metadata JSON does not authenticate this NPZ filename")
    archive_schema = data.get("schema_version")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or archive_schema is None
        or archive_schema.shape != ()
        or str(archive_schema.item()) != SCHEMA_VERSION
    ):
        raise ValueError("JSON/NPZ schema_version is missing, stale, or inconsistent")
    _validate_result_identity(path, data, metadata)
    recorded_digest = metadata.get("data_sha256")
    if (
        not isinstance(recorded_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_digest)
        or not hmac.compare_digest(recorded_digest, _sha256(path))
    ):
        raise ValueError("metadata JSON does not authenticate this NPZ content digest")
    dimension = metadata.get("dimension")
    resolution = metadata.get("resolution")
    if type(dimension) is not int or dimension not in (1, 2, 3):
        raise ValueError("metadata dimension must be 1, 2, or 3")
    if (
        not isinstance(resolution, list)
        or len(resolution) != dimension
        or any(type(value) is not int or value < 1 for value in resolution)
    ):
        raise ValueError("metadata resolution must match its dimension")
    base_key = "numeric" if "numeric" in data else "numeric_level_0"
    if base_key not in data or data[base_key].shape != tuple(reversed(resolution)):
        raise ValueError("metadata resolution does not match the NPZ base-grid shape")
    return data, metadata


def _read_regular_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("%s must be one regular file: %s" % (label, path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("%s must contain one JSON object" % label)
    return value


def _complete_relative_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("COMPLETE %s path must be non-empty relative text" % label)
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file() or root not in candidate.resolve().parents:
        raise ValueError("COMPLETE %s is not a regular file below its result root" % label)
    return candidate


def _matrix_source_authority(metadata: dict[str, object]) -> dict[str, object]:
    """Rebuild the common non-native source receipt recorded by the matrix driver."""
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("COMPLETE metadata has no provenance")
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise ValueError("COMPLETE metadata has no source provenance")
    repository_sha = source.get("repository_sha")
    repository_dirty = source.get("repository_dirty")
    tracked_diff_sha256 = source.get("tracked_diff_sha256")
    files = source.get("files")
    build_tree = source.get("build_tree")
    if (
        not isinstance(repository_sha, str)
        or not repository_sha
        or type(repository_dirty) is not bool
        or not isinstance(tracked_diff_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", tracked_diff_sha256) is None
        or not isinstance(files, dict)
        or not files
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for path, digest in files.items()
        )
        or not isinstance(build_tree, dict)
        or build_tree.get("schema_version") != "pops.sine-wave.build-source-tree.v1"
        or build_tree.get("roots") != list(BUILD_SOURCE_AUTHORITY_ROOTS)
        or not isinstance(build_tree.get("files"), dict)
        or not build_tree["files"]
        or not isinstance(build_tree.get("fingerprint"), str)
        or re.fullmatch(r"[0-9a-f]{64}", build_tree["fingerprint"]) is None
        or not hmac.compare_digest(
            build_tree["fingerprint"],
            _canonical_sha256(
                {key: value for key, value in build_tree.items() if key != "fingerprint"}
            ),
        )
    ):
        raise ValueError("COMPLETE metadata has an invalid non-native source authority")
    authority: dict[str, object] = {
        "schema_version": MATRIX_SOURCE_AUTHORITY_SCHEMA,
        "repository_sha": repository_sha,
        "repository_dirty": repository_dirty,
        "tracked_diff_sha256": tracked_diff_sha256,
        "files": files,
        "build_tree": build_tree,
    }
    authority["fingerprint"] = _canonical_sha256(authority)
    return authority


def _validate_complete_provenance_entry(
    entry: dict[str, object],
    metadata: dict[str, object],
    source_authority: dict[str, object],
    *,
    case_id: str,
) -> None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("COMPLETE metadata has no provenance for %s" % case_id)
    execution = provenance.get("execution")
    source = provenance.get("source")
    if (
        not isinstance(execution, dict)
        or entry.get("native_artifact") != provenance.get("artifact")
        or entry.get("runtime") != execution.get("runtime")
    ):
        raise ValueError("COMPLETE native artifact/runtime differs for %s" % case_id)
    if not isinstance(source, dict) or _matrix_source_authority(metadata) != source_authority:
        raise ValueError("COMPLETE source authority differs for %s" % case_id)
    if entry.get("source_authority_fingerprint") != source_authority.get("fingerprint"):
        raise ValueError("COMPLETE source authority receipt differs for %s" % case_id)


def _sealed_complete_inputs(
    complete_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    list[tuple[str, Path, Path, dict[str, object]]],
]:
    """Authenticate a complete campaign before any figure or report is rendered.

    This deliberately reads only JSON/NPZ evidence.  It neither imports PoPS nor
    invokes the scientific generator, so plotting cannot accidentally run a case.
    """
    complete_path = complete_path.resolve()
    complete = _read_regular_json(complete_path, label="COMPLETE manifest")
    required = {
        "schema_version",
        "matrix",
        "matrix_sha256",
        "generator",
        "generator_sha256",
        "driver",
        "driver_sha256",
        "support",
        "support_sha256",
        "build_script",
        "build_script_sha256",
        "source_authority",
        "case_count",
        "pairs",
        "native_compatibility",
        "convergence",
    }
    if set(complete) != required or complete.get("schema_version") != COMPLETE_SCHEMA:
        raise ValueError("COMPLETE manifest has an unsupported schema")
    expected_files = {
        "matrix": MATRIX_PATH,
        "generator": Path(__file__).with_name("generate_data.py"),
        "driver": Path(__file__).with_name("run_matrix.py"),
        "support": Path(__file__).with_name("_case_support.py"),
        "build_script": REPOSITORY_ROOT / "scripts/build_python.sh",
    }
    for name, expected_path in expected_files.items():
        if complete.get(name) != expected_path.name and name != "build_script":
            raise ValueError("COMPLETE does not name the expected %s" % name)
        if name == "build_script" and complete.get(name) != "scripts/build_python.sh":
            raise ValueError("COMPLETE does not name the expected repository build script")
        expected_hash = complete.get(name + "_sha256")
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or not hmac.compare_digest(expected_hash, _sha256(expected_path))
        ):
            raise ValueError("COMPLETE %s hash does not match the current authenticated file" % name)
    matrix = _read_regular_json(MATRIX_PATH, label="matrix")
    if not hmac.compare_digest(str(complete.get("matrix_sha256")), _sha256(MATRIX_PATH)):
        raise ValueError("COMPLETE matrix hash does not match matrix.v1.json")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 37:
        raise ValueError("matrix does not retain the complete 37-case inventory")
    by_id = {
        row.get("id"): row for row in cases if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    if len(by_id) != len(cases):
        raise ValueError("matrix case identities are not unique")
    source_authority = complete.get("source_authority")
    if (
        not isinstance(source_authority, dict)
        or source_authority.get("schema_version") != MATRIX_SOURCE_AUTHORITY_SCHEMA
        or not isinstance(source_authority.get("fingerprint"), str)
        or re.fullmatch(r"[0-9a-f]{64}", source_authority["fingerprint"]) is None
    ):
        raise ValueError("COMPLETE source authority is missing or unsupported")
    authority_without_fingerprint = {
        key: value for key, value in source_authority.items() if key != "fingerprint"
    }
    if (
        set(authority_without_fingerprint)
        != {
            "schema_version",
            "repository_sha",
            "repository_dirty",
            "tracked_diff_sha256",
            "files",
            "build_tree",
        }
        or not hmac.compare_digest(
            source_authority["fingerprint"], _canonical_sha256(authority_without_fingerprint)
        )
    ):
        raise ValueError("COMPLETE source authority receipt is malformed")
    pairs = complete.get("pairs")
    if type(complete.get("case_count")) is not int or complete["case_count"] != len(by_id):
        raise ValueError("COMPLETE case count does not match its matrix")
    if not isinstance(pairs, list) or len(pairs) != len(by_id):
        raise ValueError("COMPLETE must retain one pair for every matrix case")
    root = complete_path.parent
    loaded: list[tuple[str, Path, Path, dict[str, object]]] = []
    identities: set[str] = set()
    for entry in pairs:
        if not isinstance(entry, dict) or set(entry) != {
            "case_id",
            "data",
            "data_sha256",
            "metadata",
            "metadata_sha256",
            "result_identity",
            "source_fingerprint",
            "source_authority_fingerprint",
            "native_artifact",
            "runtime",
        }:
            raise ValueError("COMPLETE pair has an unsupported shape")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or case_id not in by_id or case_id in identities:
            raise ValueError("COMPLETE pair has an unknown or duplicate case identity")
        identities.add(case_id)
        data_path = _complete_relative_file(root, entry.get("data"), label="data")
        metadata_path = _complete_relative_file(root, entry.get("metadata"), label="metadata")
        for path, name in ((data_path, "data"), (metadata_path, "metadata")):
            digest = entry.get(name + "_sha256")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not hmac.compare_digest(digest, _sha256(path))
            ):
                raise ValueError("COMPLETE %s hash mismatch for %s" % (name, case_id))
        _, metadata = _load(data_path, metadata_path)
        if (
            metadata.get("result_identity") != entry.get("result_identity")
            or metadata.get("source_fingerprint") != entry.get("source_fingerprint")
        ):
            raise ValueError("COMPLETE identity does not match authenticated metadata for %s" % case_id)
        _validate_complete_provenance_entry(
            entry, metadata, source_authority, case_id=case_id
        )
        case = by_id[case_id]
        for name in (
            "dimension",
            "mode",
            "layout",
            "subcycling",
            "block_size",
            "mpi",
            "mpi_ranks",
            "mpi_topology",
            "cycles",
            "time_snapshots",
        ):
            expected = bool(case.get("mpi", False)) if name == "mpi" else case.get(name)
            if metadata.get(name) != expected:
                raise ValueError("COMPLETE case %s disagrees with matrix %s" % (case_id, name))
        resolution = case.get("resolution")
        expected_resolution = [int(resolution)] * int(case["dimension"])
        if metadata.get("resolution") != expected_resolution:
            raise ValueError("COMPLETE case %s disagrees with matrix resolution" % case_id)
        _validate_coverage_obligations(case_id, case, metadata)
        loaded.append((case_id, data_path, metadata_path, metadata))
    if identities != set(by_id):
        raise ValueError("COMPLETE does not retain the exact matrix case inventory")
    _validate_native_compatibility(
        complete.get("native_compatibility"),
        by_id,
        {case_id: metadata for case_id, _, _, metadata in loaded},
    )
    return complete, by_id, loaded


def _native_abi_tokens(value: object, *, case_id: str) -> dict[str, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("sealed native ABI key is absent for %s" % case_id)
    tokens: dict[str, str] = {}
    for field in value.split(";"):
        key, separator, item = field.partition("=")
        if not separator or not key or not item or key in tokens:
            raise ValueError("sealed native ABI key is malformed for %s" % case_id)
        tokens[key] = item
    if NATIVE_ABI_REQUIRED_FIELDS.difference(tokens):
        raise ValueError("sealed native ABI key lacks compatibility fields for %s" % case_id)
    return tokens


def _validate_mpi_topology_receipt(case_id: str, case: dict[str, object], metadata: dict[str, object]) -> None:
    """Check the observed Cartesian ownership receipt, including the np=1 MPI case."""
    expected = case.get("mpi_topology")
    coverage = metadata.get("coverage")
    receipt = coverage.get("mpi_topology") if isinstance(coverage, dict) else None
    ranks = case.get("mpi_ranks")
    if (
        not isinstance(expected, list)
        or not expected
        or any(type(value) is not int or value < 1 for value in expected)
        or type(ranks) is not int
        or int(np.prod(np.asarray(expected, dtype=np.int64))) != ranks
        or not isinstance(receipt, dict)
        or receipt.get("requested_ranks") != ranks
        or receipt.get("observed_ranks") != ranks
        or receipt.get("expected_spatial_decomposition") != expected
        or receipt.get("ownership_active") is not True
    ):
        raise ValueError("sealed MPI topology receipt disagrees with matrix case %s" % case_id)
    owners = receipt.get("rank_ownership")
    coordinates = receipt.get("rank_coordinates")
    if (
        not isinstance(owners, list)
        or not isinstance(coordinates, list)
        or len(owners) != ranks
        or len(coordinates) != ranks
        or {row.get("rank") for row in owners if isinstance(row, dict)} != set(range(ranks))
        or {row.get("rank") for row in coordinates if isinstance(row, dict)} != set(range(ranks))
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("local_boxes"), list)
            or not row["local_boxes"]
            for row in owners
        )
    ):
        raise ValueError("sealed MPI topology receipt lacks one ownership witness per rank")
    observed_coordinates = {
        tuple(row.get("coordinate", ())) for row in coordinates if isinstance(row, dict)
    }
    expected_coordinates = set(np.ndindex(*expected))
    if observed_coordinates != expected_coordinates:
        raise ValueError("sealed MPI topology receipt has the wrong Cartesian rank coordinates")
    corner = receipt.get("inter_rank_corner_crossing")
    if case_id != "d3-mpi-np8-corner":
        if corner is not None:
            raise ValueError("sealed MPI topology receipt has an unexpected inter-rank corner witness")
        return
    expected_start = [0.137, 0.137, 0.137]
    if (
        not isinstance(corner, dict)
        or corner.get("observed") is not True
        or corner.get("corner_index") != [16, 16, 16]
        or corner.get("corner_coordinate") != [0.5, 0.5, 0.5]
        or corner.get("participating_ranks") != list(range(8))
        or corner.get("characteristic_start") != expected_start
        or corner.get("velocity") != [1.0, 1.0, 1.0]
        or type(corner.get("arrival_time")) not in (int, float)
        or not 0.0 < float(corner["arrival_time"]) < 1.0
        or not np.isclose(float(corner["arrival_time"]), 0.363, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("sealed MPI topology receipt lacks the exact np8 inter-rank corner witness")


def _validate_coverage_obligations(
    case_id: str, case: dict[str, object], metadata: dict[str, object]
) -> None:
    """Require the sealed sidecar to retain every matrix obligation and observed witness."""
    expected = case.get("obligations")
    coverage = metadata.get("coverage")
    if not isinstance(expected, list) or not isinstance(coverage, dict):
        raise ValueError("sealed coverage is absent for %s" % case_id)
    requested = coverage.get("requested_obligations")
    witnesses = coverage.get("witnesses")
    if requested != expected or not isinstance(witnesses, dict):
        raise ValueError("sealed coverage obligations disagree with matrix case %s" % case_id)
    for obligation in expected:
        witness = witnesses.get(obligation)
        if (
            not isinstance(witness, dict)
            or witness.get("applicable") is not True
            or witness.get("observed") is not True
        ):
            raise ValueError("sealed coverage witness is absent for %s:%s" % (case_id, obligation))


def _recomputed_native_compatibility(
    by_id: dict[str, dict[str, object]], metadata_by_id: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Reconstruct the build compatibility receipt and MPI proof from every pair."""
    common_identities: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    if set(metadata_by_id) != set(by_id):
        raise ValueError("sealed native compatibility has an incomplete case inventory")
    for case_id, case in by_id.items():
        metadata = metadata_by_id[case_id]
        provenance = metadata.get("provenance")
        execution = provenance.get("execution") if isinstance(provenance, dict) else None
        source = provenance.get("source") if isinstance(provenance, dict) else None
        runtime = execution.get("runtime") if isinstance(execution, dict) else None
        native = source.get("native") if isinstance(source, dict) else None
        if not isinstance(runtime, dict) or not isinstance(native, dict):
            raise ValueError("sealed native compatibility has no runtime/native receipt for %s" % case_id)
        expected_mpi = case.get("mpi") is True
        if (
            type(metadata.get("mpi")) is not bool
            or metadata["mpi"] is not expected_mpi
            or type(native.get("has_mpi")) is not bool
            or native["has_mpi"] is not expected_mpi
            or type(runtime.get("mpi_compiled")) is not bool
            or runtime["mpi_compiled"] is not expected_mpi
            or type(runtime.get("mpi_active")) is not bool
            or runtime["mpi_active"] is not expected_mpi
            or type(runtime.get("mpi_ranks")) is not int
            or runtime["mpi_ranks"] != case.get("mpi_ranks")
        ):
            raise ValueError("sealed MPI receipt disagrees with matrix case %s" % case_id)
        if expected_mpi:
            _validate_mpi_topology_receipt(case_id, case, metadata)
        elif isinstance(metadata.get("coverage"), dict) and metadata["coverage"].get("mpi_topology") is not None:
            raise ValueError("sealed non-MPI case has an unexpected MPI topology receipt")
        if (
            type(native.get("dimension")) is not int
            or native["dimension"] != case.get("dimension")
            or native.get("has_kokkos") is not True
            or type(native.get("version")) is not str
            or not native["version"]
            or not isinstance(native.get("build_fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", native["build_fingerprint"]) is None
        ):
            raise ValueError("sealed native receipt is malformed for %s" % case_id)
        tokens = _native_abi_tokens(native.get("abi_key"), case_id=case_id)
        if (
            tokens["dim"] != str(case["dimension"])
            or tokens["mpi"] != ("1" if expected_mpi else "0")
            or tokens["kokkos"] != ("1" if native["has_kokkos"] else "0")
        ):
            raise ValueError("sealed native ABI key disagrees with matrix case %s" % case_id)
        common = tuple(
            sorted((name, value) for name, value in tokens.items() if name not in NATIVE_ABI_VARIANT_FIELDS)
        )
        common_identities.add((native["version"], native["build_fingerprint"], common))
    if len(common_identities) != 1:
        raise ValueError("sealed native compatibility differs across matrix build phases")
    version, build_fingerprint, abi_common = common_identities.pop()
    return {
        "version": version,
        "build_fingerprint": build_fingerprint,
        "abi_common": [list(item) for item in abi_common],
    }


def _validate_native_compatibility(
    recorded: object,
    by_id: dict[str, dict[str, object]],
    metadata_by_id: dict[str, dict[str, object]],
) -> None:
    """Require COMPLETE to seal the independently reconstructed native compatibility."""
    recomputed = _recomputed_native_compatibility(by_id, metadata_by_id)
    if not isinstance(recorded, dict) or set(recorded) != set(recomputed):
        raise ValueError("COMPLETE has no supported native compatibility receipt")
    if (
        not isinstance(recorded.get("version"), str)
        or not hmac.compare_digest(recorded["version"], str(recomputed["version"]))
        or not isinstance(recorded.get("build_fingerprint"), str)
        or not hmac.compare_digest(
            recorded["build_fingerprint"], str(recomputed["build_fingerprint"])
        )
        or recorded.get("abi_common") != recomputed["abi_common"]
    ):
        raise ValueError("COMPLETE native compatibility receipt disagrees with authenticated pairs")


def _finite_scalar(value: object, *, label: str) -> float:
    """Accept one finite JSON number, never a Boolean masquerading as one."""
    if type(value) not in (int, float) or not np.isfinite(float(value)):
        raise ValueError("%s must be one finite number" % label)
    return float(value)


def _receipt_close(expected: float, received: float, *, tolerance: float, label: str) -> None:
    """Compare independently rounded scalar receipts with an ULP-scale bound."""
    scale = max(1.0, abs(expected), abs(received))
    if abs(expected - received) > tolerance * scale:
        raise ValueError("sealed convergence receipt disagrees with recomputation at %s" % label)


def _uniform_final_error_norms(
    data: dict[str, np.ndarray], metadata: dict[str, object]
) -> dict[str, float]:
    """Recompute final norms from the authenticated uniform NPZ leaf field.

    The matrix deliberately declares its three qualifying series as uniform.
    Rebuilding the norms here avoids treating JSON metric text as an authority
    for either the plotted values or the qualifying orders.
    """
    if metadata.get("layout") != "uniform":
        raise ValueError("sealed convergence case must be uniform")
    resolution = metadata.get("resolution")
    if (
        not isinstance(resolution, list)
        or not resolution
        or any(type(value) is not int or value < 1 for value in resolution)
    ):
        raise ValueError("sealed convergence case has an invalid resolution")
    numerical, exact, mask = _validated_fields(
        data.get("numeric"), data.get("exact"), data.get("mask")
    )
    if numerical.shape != tuple(reversed(resolution)):
        raise ValueError("sealed convergence field shape disagrees with its resolution")
    if not mask.all():
        raise ValueError("uniform sealed convergence field must retain every leaf")
    volume = 1.0 / float(np.prod(np.asarray(resolution, dtype=np.float64)))
    represented_volume = float(np.sum(mask, dtype=np.float64) * volume)
    if abs(represented_volume - 1.0) > ERROR_RECEIPT_TOLERANCE:
        raise ValueError("uniform sealed convergence field does not represent unit volume")
    difference = np.asarray(numerical, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    absolute = np.abs(difference)
    l1 = float(np.sum(absolute, dtype=np.float64) * volume / represented_volume)
    l2 = float(
        np.sqrt(np.sum(difference * difference, dtype=np.float64) * volume / represented_volume)
    )
    linf = float(np.max(absolute))
    recomputed = {"l1": l1, "l2": l2, "linf": linf}
    if any(not np.isfinite(value) or value <= 0.0 for value in recomputed.values()):
        raise ValueError("sealed convergence field has non-positive or non-finite final errors")
    return recomputed


def _metadata_error_receipt(metadata: dict[str, object], *, case_id: str) -> dict[str, float]:
    """Require the sidecar's metric receipt to agree with its NPZ, too."""
    metrics = metadata.get("metrics")
    errors = metrics.get("errors") if isinstance(metrics, dict) else None
    if not isinstance(errors, dict) or set(errors) != set(CONVERGENCE_NORMS):
        raise ValueError("sealed convergence metadata has no exact final-norm receipt for %s" % case_id)
    return {
        norm: _finite_scalar(errors[norm], label="metadata %s %s" % (case_id, norm))
        for norm in CONVERGENCE_NORMS
    }


def _convergence_declarations(
    matrix: dict[str, object], by_id: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    """Read the fixed three-series convergence contract directly from matrix.v1.json."""
    declarations = matrix.get("convergence_series")
    if not isinstance(declarations, dict) or set(declarations) != {"dim1", "dim2", "dim3"}:
        raise ValueError("matrix has no exact Dim1/Dim2/Dim3 convergence declaration")
    result: dict[str, dict[str, object]] = {}
    for dimension in (1, 2, 3):
        name = "dim%d" % dimension
        declaration = declarations[name]
        if not isinstance(declaration, dict) or set(declaration) != {
            "case_ids",
            "qualified_norm",
            "reported_norms",
            "minimum_order",
        }:
            raise ValueError("matrix convergence declaration %s has an unsupported shape" % name)
        identifiers = declaration["case_ids"]
        reported = declaration["reported_norms"]
        minimum = _finite_scalar(declaration["minimum_order"], label="%s minimum_order" % name)
        if (
            not isinstance(identifiers, list)
            or len(identifiers) != 3
            or len(set(identifiers)) != 3
            or any(not isinstance(identifier, str) for identifier in identifiers)
            or declaration["qualified_norm"] not in CONVERGENCE_NORMS
            or not isinstance(reported, list)
            or tuple(reported) != CONVERGENCE_NORMS
            or not 1.5 <= minimum < 2.0
        ):
            raise ValueError("matrix convergence declaration %s is invalid" % name)
        cases = []
        for identifier in identifiers:
            case = by_id.get(identifier)
            if not isinstance(case, dict):
                raise ValueError("matrix convergence declaration %s names an unknown case" % name)
            if (
                case.get("dimension") != dimension
                or case.get("layout") != "uniform"
                or case.get("mode") != "x"
                or case.get("mpi") is True
                or case.get("mpi_ranks") != 1
                or case.get("cycles") != 1
            ):
                raise ValueError("matrix convergence declaration %s is not a serial uniform final-time series" % name)
            cases.append(case)
        resolutions = [int(case["resolution"]) for case in cases]
        controls = {
            tuple(
                (key, json.dumps(case.get(key), sort_keys=True))
                for key in (
                    "dimension",
                    "mode",
                    "layout",
                    "subcycling",
                    "block_size",
                    "mpi_ranks",
                    "cycles",
                    "time_snapshots",
                    "mpi",
                )
            )
            for case in cases
        }
        if resolutions != sorted(resolutions) or len(set(resolutions)) != 3 or len(controls) != 1:
            raise ValueError("matrix convergence declaration %s lacks a controlled resolution series" % name)
        result[name] = {
            "case_ids": identifiers,
            "resolutions": resolutions,
            "qualified_norm": declaration["qualified_norm"],
            "minimum_order": minimum,
            "reported_norms": tuple(reported),
        }
    return result


def _recomputed_convergence_receipt(
    matrix: dict[str, object],
    by_id: dict[str, dict[str, object]],
    metadata_by_id: dict[str, dict[str, object]],
    data_by_id: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Derive every normative convergence value from authenticated NPZ leaves."""
    receipt: dict[str, object] = {}
    for name, declaration in _convergence_declarations(matrix, by_id).items():
        identifiers = declaration["case_ids"]
        rows = []
        for identifier in identifiers:
            metadata = metadata_by_id.get(identifier)
            data = data_by_id.get(identifier)
            if not isinstance(metadata, dict) or not isinstance(data, dict):
                raise ValueError("sealed convergence %s lacks authenticated case %s" % (name, identifier))
            rows.append((identifier, metadata, data))
        source_fingerprints = {row[1].get("source_fingerprint") for row in rows}
        runtimes = {
            _canonical_sha256(row[1].get("provenance", {}).get("execution", {}).get("runtime"))
            if isinstance(row[1].get("provenance"), dict)
            and isinstance(row[1]["provenance"].get("execution"), dict)
            and isinstance(row[1]["provenance"]["execution"].get("runtime"), dict)
            else None
            for row in rows
        }
        methods = {
            _canonical_sha256(row[1].get("metrics", {}).get("method"))
            if isinstance(row[1].get("metrics"), dict)
            and isinstance(row[1]["metrics"].get("method"), dict)
            else None
            for row in rows
        }
        if len(source_fingerprints) != 1 or None in source_fingerprints or len(runtimes) != 1 or None in runtimes or len(methods) != 1 or None in methods:
            raise ValueError("sealed convergence %s mixes source, backend, or method" % name)
        errors = {norm: [] for norm in declaration["reported_norms"]}
        for identifier, metadata, data in rows:
            recomputed = _uniform_final_error_norms(data, metadata)
            metadata_receipt = _metadata_error_receipt(metadata, case_id=identifier)
            for norm in declaration["reported_norms"]:
                _receipt_close(
                    recomputed[norm],
                    metadata_receipt[norm],
                    tolerance=ERROR_RECEIPT_TOLERANCE,
                    label="metadata %s %s" % (identifier, norm),
                )
                errors[norm].append(recomputed[norm])
        orders: dict[str, list[float]] = {}
        for norm, values in errors.items():
            observed = convergence_orders(declaration["resolutions"], values)
            if len(observed) != 3 or observed[0] is not None or any(value is None for value in observed[1:]):
                raise ValueError("sealed convergence %s has invalid %s orders" % (name, norm))
            orders[norm] = [float(value) for value in observed[1:]]
        qualified = orders[str(declaration["qualified_norm"])]
        if any(order < float(declaration["minimum_order"]) for order in qualified):
            raise ValueError("sealed convergence %s is below its declared final-order threshold" % name)
        receipt[name] = {
            "case_ids": list(identifiers),
            "resolutions": list(declaration["resolutions"]),
            "qualified_norm": declaration["qualified_norm"],
            "minimum_order": declaration["minimum_order"],
            "errors": errors,
            "orders": orders,
        }
    return receipt


def _validate_convergence_receipt(
    recorded: object, recomputed: dict[str, object]
) -> None:
    """Fail closed unless COMPLETE retains the same canonical convergence evidence."""
    if not isinstance(recorded, dict) or set(recorded) != set(recomputed):
        raise ValueError("COMPLETE has an unsupported convergence receipt")
    for series, expected in recomputed.items():
        received = recorded.get(series)
        if not isinstance(expected, dict) or not isinstance(received, dict) or set(received) != set(expected):
            raise ValueError("COMPLETE convergence receipt has an unsupported shape for %s" % series)
        for key in ("case_ids", "resolutions", "qualified_norm"):
            if received[key] != expected[key]:
                raise ValueError("COMPLETE convergence receipt differs at %s.%s" % (series, key))
        _receipt_close(
            _finite_scalar(expected["minimum_order"], label="%s minimum_order" % series),
            _finite_scalar(received["minimum_order"], label="COMPLETE %s minimum_order" % series),
            tolerance=ERROR_RECEIPT_TOLERANCE,
            label="%s.minimum_order" % series,
        )
        for container in ("errors", "orders"):
            expected_values = expected[container]
            received_values = received[container]
            if not isinstance(expected_values, dict) or not isinstance(received_values, dict) or set(received_values) != set(expected_values):
                raise ValueError("COMPLETE convergence receipt differs at %s.%s" % (series, container))
            tolerance = ERROR_RECEIPT_TOLERANCE if container == "errors" else ORDER_RECEIPT_TOLERANCE
            for norm, values in expected_values.items():
                candidate = received_values.get(norm)
                if not isinstance(values, list) or not isinstance(candidate, list) or len(candidate) != len(values):
                    raise ValueError("COMPLETE convergence receipt differs at %s.%s.%s" % (series, container, norm))
                for index, (expected_value, received_value) in enumerate(zip(values, candidate, strict=True)):
                    _receipt_close(
                        _finite_scalar(expected_value, label="%s.%s.%s[%d]" % (series, container, norm, index)),
                        _finite_scalar(received_value, label="COMPLETE %s.%s.%s[%d]" % (series, container, norm, index)),
                        tolerance=tolerance,
                        label="%s.%s.%s[%d]" % (series, container, norm, index),
                    )


def _validated_fields(
    numerical: np.ndarray, exact: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerical_array = np.asarray(numerical)
    exact_array = np.asarray(exact)
    mask_array = np.asarray(mask)
    if not (
        numerical_array.shape == exact_array.shape == mask_array.shape
        and numerical_array.ndim in (1, 2, 3)
        and numerical_array.size > 0
    ):
        raise ValueError(
            "numeric, exact, and mask arrays must share one non-empty rank-1/2/3 shape"
        )
    if mask_array.dtype != np.bool_:
        raise ValueError("leaf mask must have boolean dtype")
    if not (
        np.isfinite(numerical_array[mask_array]).all()
        and np.isfinite(exact_array[mask_array]).all()
    ):
        raise ValueError("active numeric and exact cells must be finite")
    return numerical_array, exact_array, mask_array


def _field_pair(
    data: dict[str, np.ndarray], *, prefix: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if prefix + "numeric" in data:
        numerical, exact, mask = _validated_fields(
            data[prefix + "numeric"], data[prefix + "exact"], data[prefix + "mask"]
        )
        if not mask.any():
            raise ValueError("leaf mask has no active cell")
        return numerical, exact, mask, 0
    level_prefix = prefix + "numeric_level_"
    levels = [int(name.rsplit("_", 1)[1]) for name in data if name.startswith(level_prefix)]
    if not levels:
        raise ValueError("NPZ has neither uniform nor AMR sine-wave fields")
    finest = max(levels)
    target_shape = data["%snumeric_level_%d" % (prefix, finest)].shape
    numerical = np.full(target_shape, np.nan, dtype=np.float64)
    exact = np.full(target_shape, np.nan, dtype=np.float64)
    covered = np.zeros(target_shape, dtype=bool)

    def expand(array: np.ndarray) -> np.ndarray:
        result = np.asarray(array)
        if result.ndim != len(target_shape):
            raise ValueError("AMR levels do not have a common rank")
        for axis, (source, target) in enumerate(zip(result.shape, target_shape, strict=True)):
            factor, remainder = divmod(target, source)
            if remainder or factor < 1:
                raise ValueError("AMR level shapes are not nested integer refinements")
            result = np.repeat(result, factor, axis=axis)
        return result

    # Coarse leaf averages are repeated for display; fine leaves fill their own footprint.
    for level in sorted(levels):
        level_numerical, level_exact, level_mask = _validated_fields(
            data["%snumeric_level_%d" % (prefix, level)],
            data["%sexact_level_%d" % (prefix, level)],
            data["%smask_level_%d" % (prefix, level)],
        )
        mask = expand(level_mask).astype(bool, copy=False)
        level_numerical = expand(level_numerical)
        level_exact = expand(level_exact)
        numerical[mask] = level_numerical[mask]
        exact[mask] = level_exact[mask]
        covered[mask] = True
    if not covered.all():
        raise ValueError("AMR leaf masks do not cover the complete periodic domain")
    return numerical, exact, covered, finest


def _interface_mask(
    data: dict[str, np.ndarray], *, prefix: str, target_shape: tuple[int, ...]
) -> np.ndarray:
    """Compose authenticated coarse/fine interface leaves on the display mesh."""

    def validate(interface: np.ndarray, active: np.ndarray, *, where: str) -> np.ndarray:
        interface_array = np.asarray(interface)
        active_array = np.asarray(active)
        if (
            interface_array.dtype != np.bool_
            or active_array.dtype != np.bool_
            or interface_array.shape != active_array.shape
            or np.any(interface_array & ~active_array)
        ):
            raise ValueError(
                "%s interface mask must be boolean and a subset of active leaves" % where
            )
        return interface_array

    def expand(array: np.ndarray) -> np.ndarray:
        result = np.asarray(array)
        if result.ndim != len(target_shape):
            raise ValueError("interface mask rank does not match the composed field")
        for axis, (source, target) in enumerate(zip(result.shape, target_shape, strict=True)):
            factor, remainder = divmod(target, source)
            if remainder or factor < 1:
                raise ValueError("interface-mask shapes are not nested integer refinements")
            result = np.repeat(result, factor, axis=axis)
        return result

    direct_key = prefix + "interface_mask"
    if prefix + "numeric" in data:
        if direct_key not in data:
            raise ValueError("uniform snapshot is missing its explicit interface mask")
        result = validate(data[direct_key], data[prefix + "mask"], where="uniform")
        if result.shape != target_shape:
            raise ValueError("uniform interface mask does not match the field shape")
        return result
    levels = sorted(
        int(name.rsplit("_", 1)[1]) for name in data if name.startswith(prefix + "numeric_level_")
    )
    if not levels or levels != list(range(max(levels) + 1)):
        raise ValueError("AMR interface masks require contiguous field levels")
    result = np.zeros(target_shape, dtype=bool)
    for level in levels:
        interface_key = "%sinterface_mask_level_%d" % (prefix, level)
        active_key = "%smask_level_%d" % (prefix, level)
        if interface_key not in data or active_key not in data:
            raise ValueError("AMR snapshot is missing a level interface mask")
        level_interface = validate(
            data[interface_key], data[active_key], where="AMR level %d" % level
        )
        result |= expand(level_interface)
    return result


def _coordinates(
    data: dict[str, np.ndarray], *, prefix: str, level: int, dimension: int
) -> tuple[np.ndarray, ...]:
    direct = tuple(
        data[prefix + name] for name in ("x", "y", "z")[:dimension] if prefix + name in data
    )
    if len(direct) != dimension:
        direct = tuple(
            data["%s%s_level_%d" % (prefix, name, level)]
            for name in ("x", "y", "z")[:dimension]
            if "%s%s_level_%d" % (prefix, name, level) in data
        )
    if len(direct) != dimension:
        raise ValueError("every timeline frame must contain all coordinate axes")
    if any(np.asarray(axis).ndim != 1 or not np.isfinite(axis).all() for axis in direct):
        raise ValueError("timeline coordinates must be finite one-dimensional arrays")
    return tuple(np.asarray(axis, dtype=np.float64) for axis in direct)


def _timeline_frames(
    data: dict[str, np.ndarray], metadata: dict[str, object]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Validate and compose every authenticated frame before plotting any artifact."""
    timeline = metadata.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("metadata timeline is required by this schema")
    frame_count = timeline.get("frames")
    metadata_times = timeline.get("times")
    if type(frame_count) is not int or frame_count < 9:
        raise ValueError("timeline must describe at least nine frames")
    if timeline.get("storage_prefix") != "timeline_{index:04d}_":
        raise ValueError("timeline storage_prefix is unsupported")
    if (
        not isinstance(metadata_times, list)
        or len(metadata_times) != frame_count
        or any(type(value) not in (int, float) for value in metadata_times)
    ):
        raise ValueError("timeline metadata times must match the declared frame count")
    times = np.asarray(data.get("timeline_time"), dtype=np.float64)
    expected_times = np.asarray(metadata_times, dtype=np.float64)
    if (
        times.shape != (frame_count,)
        or not np.isfinite(times).all()
        or not np.array_equal(times, expected_times)
        or times[0] != 0.0
        or np.any(np.diff(times) <= 0.0)
        or not np.isclose(
            times[-1], float(metadata.get("final_time", np.nan)), rtol=0.0, atol=1.0e-14
        )
    ):
        raise ValueError("JSON/NPZ timeline times must agree and increase from zero to final_time")
    discovered_indices: set[int] = set()
    malformed_timeline_key = False
    for name in data:
        if not name.startswith("timeline_") or name == "timeline_time":
            continue
        match = re.match(r"timeline_(\d{4})_", name)
        if match is None:
            malformed_timeline_key = True
        else:
            discovered_indices.add(int(match.group(1)))
    if malformed_timeline_key or discovered_indices != set(range(frame_count)):
        raise ValueError("timeline frame prefixes must be contiguous and zero-padded")

    dimension = int(metadata["dimension"])
    layout = metadata.get("layout")
    if layout not in {"uniform", "amr-frozen", "amr-mobile"}:
        raise ValueError("timeline layout label is missing or unsupported")
    frames: list[dict[str, object]] = []
    for index, time in enumerate(times):
        prefix = "timeline_%04d_" % index
        numerical, exact, mask, level = _field_pair(data, prefix=prefix)
        interface = _interface_mask(data, prefix=prefix, target_shape=numerical.shape)
        if numerical.ndim != dimension:
            raise ValueError("timeline field rank does not match metadata dimension")
        coordinates = _coordinates(data, prefix=prefix, level=level, dimension=dimension)
        expected_lengths = tuple(reversed(numerical.shape))
        if tuple(axis.size for axis in coordinates) != expected_lengths:
            raise ValueError("timeline coordinates do not match composed field shape")
        patches = np.asarray(data.get(prefix + "patch_boxes", np.empty((0,), dtype=int)))
        if patches.size and (
            patches.ndim != 2
            or patches.shape[1] != 2 * dimension + 1
            or not np.issubdtype(patches.dtype, np.integer)
        ):
            raise ValueError("timeline patch boxes have an invalid shape or dtype")

        def nonnegative_counter(name: str, snapshot_prefix: str = prefix) -> int:
            value = data.get(snapshot_prefix + name)
            if (
                value is None
                or np.asarray(value).shape != ()
                or not np.issubdtype(np.asarray(value).dtype, np.integer)
                or int(np.asarray(value).item()) < 0
            ):
                raise ValueError("timeline %s must be a non-negative integer scalar" % name)
            return int(np.asarray(value).item())

        frames.append(
            {
                "time": float(time),
                "numeric": np.where(mask, numerical, np.nan),
                "exact": np.where(mask, exact, np.nan),
                "interface": interface,
                "coordinates": coordinates,
                "patches": patches,
                "level": level,
                "native_shape": numerical.shape,
                "regrid_count": nonnegative_counter("regrid_count"),
                "topology_epoch": nonnegative_counter("topology_epoch"),
            }
        )
    target_shape = tuple(
        max(frame["numeric"].shape[axis] for frame in frames) for axis in range(dimension)
    )
    for frame in frames:
        native_shape = frame["numeric"].shape
        numerical = frame["numeric"]
        exact = frame["exact"]
        interface = frame["interface"]
        for axis, (source, target) in enumerate(zip(native_shape, target_shape, strict=True)):
            factor, remainder = divmod(target, source)
            if remainder or factor < 1:
                raise ValueError("timeline field shapes are not nested integer refinements")
            numerical = np.repeat(numerical, factor, axis=axis)
            exact = np.repeat(exact, factor, axis=axis)
            interface = np.repeat(interface, factor, axis=axis)
        frame["numeric"] = numerical
        frame["exact"] = exact
        frame["interface"] = interface
        frame["coordinates"] = tuple(
            (np.arange(count, dtype=np.float64) + 0.5) / count for count in reversed(target_shape)
        )
    regrid_counts = np.asarray([frame["regrid_count"] for frame in frames], dtype=np.int64)
    topology_epochs = np.asarray([frame["topology_epoch"] for frame in frames], dtype=np.int64)
    if np.any(np.diff(regrid_counts) < 0) or np.any(np.diff(topology_epochs) < 0):
        raise ValueError("AMR regrid/topology counters must be monotone")
    if layout == "uniform" and (np.any(regrid_counts) or np.any(topology_epochs)):
        raise ValueError("uniform timelines cannot claim AMR regrid events")
    return times, frames


def _stable_scales(frames: list[dict[str, object]]) -> tuple[float, float, float]:
    field_min = min(
        float(np.nanmin(frame[name])) for frame in frames for name in ("exact", "numeric")
    )
    field_max = max(
        float(np.nanmax(frame[name])) for frame in frames for name in ("exact", "numeric")
    )
    if field_min == field_max:
        field_min -= np.finfo(float).eps
        field_max += np.finfo(float).eps
    error_limit = max(
        max(float(np.nanmax(np.abs(frame["numeric"] - frame["exact"]))) for frame in frames),
        np.finfo(float).eps,
    )
    return field_min, field_max, error_limit


def _patch_overlay(
    axis, patches: np.ndarray, *, level: int, shape: tuple[int, int]
) -> list[object]:
    if patches.size == 0 or patches.shape[1] != 5:
        return []
    from matplotlib.patches import Rectangle

    artists = []
    ny, nx = shape
    for x_lower, y_lower, x_upper, y_upper, patch_level in patches:
        if int(patch_level) != level:
            continue
        artist = Rectangle(
            (x_lower / nx, y_lower / ny),
            (x_upper - x_lower + 1) / nx,
            (y_upper - y_lower + 1) / ny,
            fill=False,
            edgecolor="white",
            linewidth=0.7,
        )
        axis.add_patch(artist)
        artists.append(artist)
    return artists


def _plot_1d(
    plt,
    numeric: np.ndarray,
    exact: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    target: Path,
) -> Path:
    x = coordinates[0]
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].plot(x, exact, label="moyenne exacte", linewidth=2)
    axes[0].plot(x, numeric, "o", label="PoPS", markersize=3)
    axes[0].set_ylabel("q")
    axes[0].legend()
    axes[1].plot(x, numeric - exact)
    axes[1].set(xlabel="x", ylabel="erreur")
    figure.tight_layout()
    _save_figure(figure, target, dpi=160)
    plt.close(figure)
    return target


def _plot_2d(
    plt,
    numeric: np.ndarray,
    exact: np.ndarray,
    patches: np.ndarray,
    level: int,
    target: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    field_min = float(np.nanmin((exact, numeric)))
    field_max = float(np.nanmax((exact, numeric)))
    error = numeric - exact
    error_limit = max(float(np.nanmax(np.abs(error))), np.finfo(float).eps)
    for axis, field, title in zip(
        axes[:2], (exact, numeric), ("moyenne exacte", "PoPS"), strict=True
    ):
        image = axis.imshow(
            field,
            origin="lower",
            aspect="equal",
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=field_min,
            vmax=field_max,
        )
        axis.set_title(title)
        axis.set(xlabel="x", ylabel="y")
        figure.colorbar(image, ax=axis, shrink=0.8)
    error_image = axes[2].imshow(
        error,
        origin="lower",
        aspect="equal",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[2].set_title("erreur")
    axes[2].set(xlabel="x", ylabel="y")
    figure.colorbar(error_image, ax=axes[2], shrink=0.8)
    _patch_overlay(axes[1], patches, level=level, shape=numeric.shape)
    _save_figure(figure, target, dpi=160)
    plt.close(figure)
    return target


def _cut_axes(label: str) -> tuple[str, str, int]:
    return {
        "xy": ("x", "y", 2),
        "xz": ("x", "z", 1),
        "yz": ("y", "z", 0),
    }[label]


def _patch_overlay_3d_cut(
    axis,
    patches: np.ndarray,
    *,
    label: str,
    finest_level: int,
    finest_shape: tuple[int, int, int],
) -> list[object]:
    """Project authenticated AMR boxes onto the labelled central 3D slice."""
    if patches.size == 0 or patches.ndim != 2 or patches.shape[1] != 7:
        return []
    from matplotlib.patches import Rectangle

    x_name, y_name, normal = _cut_axes(label)
    physical_shape = np.asarray(tuple(reversed(finest_shape)), dtype=np.float64)
    center = 0.5
    artists: list[object] = []
    for row in patches:
        patch_level = int(row[-1])
        scale = 2 ** (finest_level - patch_level)
        level_shape = physical_shape / scale
        lower = np.asarray(row[:3], dtype=np.float64) / level_shape
        upper = (np.asarray(row[3:6], dtype=np.float64) + 1.0) / level_shape
        if not lower[normal] <= center <= upper[normal]:
            continue
        names = ("x", "y", "z")
        x_index = names.index(x_name)
        y_index = names.index(y_name)
        artist = Rectangle(
            (lower[x_index], lower[y_index]),
            upper[x_index] - lower[x_index],
            upper[y_index] - lower[y_index],
            fill=False,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        axis.add_patch(artist)
        artists.append(artist)
    return artists


def _plot_3d(
    plt,
    numeric: np.ndarray,
    exact: np.ndarray,
    patches: np.ndarray,
    level: int,
    target: Path,
) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    centre = tuple(extent // 2 for extent in numeric.shape)
    cuts = (
        (numeric[centre[0]], exact[centre[0]], "xy"),
        (numeric[:, centre[1], :], exact[:, centre[1], :], "xz"),
        (numeric[:, :, centre[2]], exact[:, :, centre[2]], "yz"),
    )
    field_min = min(float(np.nanmin(reference)) for _, reference, _ in cuts)
    field_min = min(field_min, *(float(np.nanmin(computed)) for computed, _, _ in cuts))
    field_max = max(float(np.nanmax(reference)) for _, reference, _ in cuts)
    field_max = max(field_max, *(float(np.nanmax(computed)) for computed, _, _ in cuts))
    error_limit = max(
        *(float(np.nanmax(np.abs(computed - reference))) for computed, reference, _ in cuts),
        np.finfo(float).eps,
    )
    field_image = None
    error_image = None
    for row, (computed, reference, label) in enumerate(cuts):
        field_image = axes[row, 0].imshow(
            reference,
            origin="lower",
            aspect="equal",
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=field_min,
            vmax=field_max,
        )
        axes[row, 1].imshow(
            computed,
            origin="lower",
            aspect="equal",
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=field_min,
            vmax=field_max,
        )
        error_image = axes[row, 2].imshow(
            computed - reference,
            origin="lower",
            aspect="equal",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
        )
        x_label, y_label, _ = _cut_axes(label)
        for axis in axes[row]:
            axis.set(xlabel=x_label, ylabel=y_label)
            _patch_overlay_3d_cut(
                axis,
                patches,
                label=label,
                finest_level=level,
                finest_shape=numeric.shape,
            )
    for column, title in enumerate(("moyenne exacte", "PoPS", "erreur")):
        axes[0, column].set_title(title)
    figure.colorbar(field_image, ax=axes[:, :2], shrink=0.75)
    figure.colorbar(error_image, ax=axes[:, 2], shrink=0.75)
    _save_figure(figure, target, dpi=160)
    plt.close(figure)
    return target


def _save_gif(animation, target: Path, *, fps: int) -> Path:
    """Write into the fresh publication staging directory without replacement."""
    from matplotlib.animation import PillowWriter

    _require_fresh_path(target)
    animation.save(target, writer=PillowWriter(fps=fps))
    return target


def _animate_1d(plt, frames: list[dict[str, object]], target: Path, *, fps: int) -> Path:
    from matplotlib.animation import FuncAnimation

    field_min, field_max, error_limit = _stable_scales(frames)
    padding = 0.05 * (field_max - field_min)
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), constrained_layout=True)
    first = frames[0]
    x = first["coordinates"][0]
    (exact_line,) = axes[0].plot(x, first["exact"], color="#174A5B", linewidth=2, label="exacte")
    (numeric_line,) = axes[0].plot(
        x,
        first["numeric"],
        color="#D47A2C",
        marker="o",
        linewidth=1,
        markersize=3,
        label="PoPS",
    )
    (error_line,) = axes[1].plot(
        x, first["numeric"] - first["exact"], color="#A33E3E", linewidth=1.5
    )
    axes[0].set(ylabel="q", ylim=(field_min - padding, field_max + padding))
    axes[0].legend(loc="upper right")
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set(xlabel="x", ylabel="PoPS - exacte", ylim=(-error_limit, error_limit))
    time_label = figure.suptitle("")

    def update(index: int):
        frame = frames[index]
        frame_x = frame["coordinates"][0]
        exact_line.set_data(frame_x, frame["exact"])
        numeric_line.set_data(frame_x, frame["numeric"])
        error_line.set_data(frame_x, frame["numeric"] - frame["exact"])
        time_label.set_text("Advection sinusoïdale 1D — t = %.4f" % frame["time"])
        return exact_line, numeric_line, error_line, time_label

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    update(0)
    try:
        return _save_gif(animation, target, fps=fps)
    finally:
        plt.close(figure)


def _animate_2d(plt, frames: list[dict[str, object]], target: Path, *, fps: int) -> Path:
    from matplotlib.animation import FuncAnimation

    field_min, field_max, error_limit = _stable_scales(frames)
    contour_levels = np.linspace(field_min, field_max, 9)[1:-1]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    first = frames[0]
    exact_image = axes[0, 0].imshow(
        first["exact"],
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    numeric_image = axes[0, 1].imshow(
        first["numeric"],
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    error_image = axes[1, 0].imshow(
        first["numeric"] - first["exact"],
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    figure.colorbar(exact_image, ax=axes[0, :], shrink=0.75, label="q")
    figure.colorbar(error_image, ax=axes[1, 0], shrink=0.75, label="PoPS - exacte")
    axes[0, 0].set_title("moyenne exacte")
    axes[0, 1].set_title("PoPS + contours exacts + patches")
    axes[1, 0].set_title("erreur")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set(xlabel="x", ylabel="y")
    contour_holder = [
        axes[0, 1].contour(
            first["coordinates"][0],
            first["coordinates"][1],
            first["exact"],
            levels=contour_levels,
            colors="white",
            linewidths=0.6,
        )
    ]
    patch_artists = _patch_overlay(
        axes[0, 1],
        first["patches"],
        level=int(first["level"]),
        shape=first["native_shape"],
    )
    cut_axis = axes[1, 1]
    cut_error_axis = cut_axis.twinx()
    center = first["numeric"].shape[0] // 2
    x = first["coordinates"][0]
    (exact_cut,) = cut_axis.plot(
        x, first["exact"][center], color="#174A5B", linewidth=2, label="exacte"
    )
    (numeric_cut,) = cut_axis.plot(
        x, first["numeric"][center], color="#D47A2C", linewidth=1.5, label="PoPS"
    )
    (error_cut,) = cut_error_axis.plot(
        x,
        first["numeric"][center] - first["exact"][center],
        color="#A33E3E",
        linestyle="--",
        linewidth=1,
        label="erreur",
    )
    padding = 0.05 * (field_max - field_min)
    cut_axis.set(xlabel="x à y≈0,5", ylabel="q", ylim=(field_min - padding, field_max + padding))
    cut_error_axis.set(ylabel="erreur", ylim=(-error_limit, error_limit))
    handles = (exact_cut, numeric_cut, error_cut)
    cut_axis.legend(handles, tuple(item.get_label() for item in handles), loc="upper right")
    time_label = figure.suptitle("")

    def update(index: int):
        nonlocal patch_artists
        frame = frames[index]
        exact_image.set_data(frame["exact"])
        numeric_image.set_data(frame["numeric"])
        error_image.set_data(frame["numeric"] - frame["exact"])
        contour_holder[0].remove()
        contour_holder[0] = axes[0, 1].contour(
            frame["coordinates"][0],
            frame["coordinates"][1],
            frame["exact"],
            levels=contour_levels,
            colors="white",
            linewidths=0.6,
        )
        for artist in patch_artists:
            artist.remove()
        patch_artists = _patch_overlay(
            axes[0, 1],
            frame["patches"],
            level=int(frame["level"]),
            shape=frame["native_shape"],
        )
        row = frame["numeric"].shape[0] // 2
        frame_x = frame["coordinates"][0]
        exact_cut.set_data(frame_x, frame["exact"][row])
        numeric_cut.set_data(frame_x, frame["numeric"][row])
        error_cut.set_data(frame_x, frame["numeric"][row] - frame["exact"][row])
        time_label.set_text("Advection sinusoïdale 2D — t = %.4f" % frame["time"])
        return exact_image, numeric_image, error_image, exact_cut, numeric_cut, error_cut

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    update(0)
    try:
        return _save_gif(animation, target, fps=fps)
    finally:
        plt.close(figure)


def _central_cuts(field: np.ndarray) -> tuple[tuple[np.ndarray, str], ...]:
    center = tuple(extent // 2 for extent in field.shape)
    return (
        (field[center[0], :, :], "xy"),
        (field[:, center[1], :], "xz"),
        (field[:, :, center[2]], "yz"),
    )


def _cell_centers(count: int) -> np.ndarray:
    if type(count) is not int or count < 1:
        raise ValueError("cell count must be a positive integer")
    return (np.arange(count, dtype=np.float64) + 0.5) / count


def _storyboard_3d(plt, frames: list[dict[str, object]], target: Path) -> Path:
    field_min, field_max, _ = _stable_scales(frames)
    selected = np.unique(np.linspace(0, len(frames) - 1, 6, dtype=int))
    figure, axes = plt.subplots(
        3, len(selected), figsize=(3.3 * len(selected), 9), constrained_layout=True
    )
    contour_levels = np.linspace(field_min, field_max, 9)[1:-1]
    image = None
    for column, frame_index in enumerate(selected):
        frame = frames[int(frame_index)]
        numeric_cuts = _central_cuts(frame["numeric"])
        exact_cuts = _central_cuts(frame["exact"])
        for row, ((numeric_cut, label), (exact_cut, _)) in enumerate(
            zip(numeric_cuts, exact_cuts, strict=True)
        ):
            image = axes[row, column].imshow(
                numeric_cut,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="viridis",
                vmin=field_min,
                vmax=field_max,
            )
            axes[row, column].contour(
                _cell_centers(exact_cut.shape[1]),
                _cell_centers(exact_cut.shape[0]),
                exact_cut,
                levels=contour_levels,
                colors="white",
                linewidths=0.5,
            )
            x_label, y_label, _ = _cut_axes(label)
            axes[row, column].set(xlabel=x_label, ylabel=y_label)
            _patch_overlay_3d_cut(
                axes[row, column],
                frame["patches"],
                label=label,
                finest_level=int(frame["level"]),
                finest_shape=frame["native_shape"],
            )
            if column == 0:
                axes[row, column].set_ylabel("coupe %s\n%s" % (label, y_label))
            if row == 0:
                axes[row, column].set_title("t = %.3f" % frame["time"])
    figure.suptitle("Coupes 3D PoPS — contours blancs = solution exacte")
    figure.colorbar(image, ax=axes, shrink=0.7, label="q")
    _save_figure(figure, target, dpi=180)
    plt.close(figure)
    return target


def _animate_3d_cuts(plt, frames: list[dict[str, object]], target: Path, *, fps: int) -> Path:
    from matplotlib.animation import FuncAnimation

    field_min, field_max, error_limit = _stable_scales(frames)
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 9.5), constrained_layout=True)
    first = frames[0]
    images: list[object] = []
    field_image = None
    error_image = None
    for row, ((numeric_cut, label), (exact_cut, _)) in enumerate(
        zip(_central_cuts(first["numeric"]), _central_cuts(first["exact"]), strict=True)
    ):
        field_image = axes[row, 0].imshow(
            exact_cut,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
        )
        numeric_image = axes[row, 1].imshow(
            numeric_cut,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
        )
        error_image = axes[row, 2].imshow(
            numeric_cut - exact_cut,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
        )
        images.extend((field_image, numeric_image, error_image))
        x_label, y_label, _ = _cut_axes(label)
        for axis in axes[row]:
            axis.set(xlabel=x_label, ylabel=y_label)
        axes[row, 0].set_ylabel("coupe %s\n%s" % (label, y_label))
    for column, title in enumerate(("moyenne exacte", "PoPS", "erreur")):
        axes[0, column].set_title(title)
    figure.colorbar(field_image, ax=axes[:, :2], shrink=0.7, label="q")
    figure.colorbar(error_image, ax=axes[:, 2], shrink=0.7, label="PoPS - exacte")
    time_label = figure.suptitle("")
    patch_artists: list[object] = []

    def update(index: int):
        nonlocal patch_artists
        frame = frames[index]
        image_index = 0
        for (numeric_cut, _), (exact_cut, _) in zip(
            _central_cuts(frame["numeric"]),
            _central_cuts(frame["exact"]),
            strict=True,
        ):
            images[image_index].set_data(exact_cut)
            images[image_index + 1].set_data(numeric_cut)
            images[image_index + 2].set_data(numeric_cut - exact_cut)
            image_index += 3
        for artist in patch_artists:
            artist.remove()
        patch_artists = []
        for row, (_cut, label) in enumerate(_central_cuts(frame["numeric"])):
            for axis in axes[row]:
                patch_artists.extend(
                    _patch_overlay_3d_cut(
                        axis,
                        frame["patches"],
                        label=label,
                        finest_level=int(frame["level"]),
                        finest_shape=frame["native_shape"],
                    )
                )
        time_label.set_text("Advection sinusoïdale 3D — t = %.4f" % frame["time"])
        return (*images, *patch_artists, time_label)

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    update(0)
    try:
        return _save_gif(animation, target, fps=fps)
    finally:
        plt.close(figure)


def _maximum_error_frame(frames: list[dict[str, object]]) -> dict[str, object]:
    return max(
        frames,
        key=lambda frame: float(np.nanmean((frame["numeric"] - frame["exact"]) ** 2)),
    )


def _plot_maximum_spatial_error_3d(plt, frames: list[dict[str, object]], target: Path) -> Path:
    """Show where the largest sampled 3D error resides, with AMR projections."""
    frame = _maximum_error_frame(frames)
    error = frame["numeric"] - frame["exact"]
    cuts = _central_cuts(error)
    limit = max(float(np.nanmax(np.abs(error))), np.finfo(float).eps)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    image = None
    for axis, (cut, label) in zip(axes, cuts, strict=True):
        image = axis.imshow(
            cut,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        x_label, y_label, _ = _cut_axes(label)
        axis.set(title="erreur %s" % label, xlabel=x_label, ylabel=y_label)
        _patch_overlay_3d_cut(
            axis,
            frame["patches"],
            label=label,
            finest_level=int(frame["level"]),
            finest_shape=frame["native_shape"],
        )
    figure.colorbar(image, ax=axes, shrink=0.8, label="PoPS − exacte")
    figure.suptitle("Erreur spatiale 3D au snapshot L2 maximal — t = %.4f" % frame["time"])
    _save_figure(figure, target, dpi=180)
    plt.close(figure)
    return target


def _periodic_trilinear(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample a cell-centred [z,y,x] field at periodic physical [x,y,z] points."""
    nz, ny, nx = field.shape
    sizes = np.asarray((nx, ny, nz), dtype=np.float64)
    position = points * sizes - 0.5
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    values = np.zeros(points.shape[0], dtype=np.float64)
    for z_offset in (0, 1):
        z_weight = (1.0 - fraction[:, 2]) if z_offset == 0 else fraction[:, 2]
        z_index = (lower[:, 2] + z_offset) % nz
        for y_offset in (0, 1):
            y_weight = (1.0 - fraction[:, 1]) if y_offset == 0 else fraction[:, 1]
            y_index = (lower[:, 1] + y_offset) % ny
            for x_offset in (0, 1):
                x_weight = (1.0 - fraction[:, 0]) if x_offset == 0 else fraction[:, 0]
                x_index = (lower[:, 0] + x_offset) % nx
                values += x_weight * y_weight * z_weight * field[z_index, y_index, x_index]
    return values


def _plot_oblique_cut_3d(
    plt,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
    target: Path,
) -> Path:
    """Compare exact/numerical data along the periodic line directed by velocity a."""
    velocity = np.asarray(metadata.get("velocity"), dtype=np.float64)
    if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.any(velocity):
        raise ValueError("3D oblique cut requires one finite nonzero velocity vector")
    frame = _maximum_error_frame(frames)
    parameter = np.linspace(0.0, 1.0, 768, endpoint=False)
    points = (0.5 + parameter[:, None] * velocity[None, :]) % 1.0
    exact = _periodic_trilinear(frame["exact"], points)
    numeric = _periodic_trilinear(frame["numeric"], points)
    error = numeric - exact
    error_limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), constrained_layout=True)
    axes[0].plot(parameter, exact, color="#174A5B", linewidth=2, label="exacte")
    axes[0].plot(parameter, numeric, color="#D47A2C", linewidth=1.5, label="PoPS")
    axes[0].set(ylabel="q", title="Coupe périodique suivant a")
    axes[0].legend()
    axes[1].plot(parameter, error, color="#A33E3E", linewidth=1.3)
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set(
        xlabel="paramètre s : x(s)=(0.5, 0.5, 0.5)+s a modulo 1",
        ylabel="PoPS - exacte",
        ylim=(-error_limit, error_limit),
    )
    figure.suptitle("t = %.4f — instant d'erreur L2 maximale" % frame["time"])
    _save_figure(figure, target, dpi=180)
    plt.close(figure)
    return target


def _vtk_isosurface(field: np.ndarray, level: float):
    """Extract a true triangle isosurface when optional VTK is installed."""
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except ImportError:
        return None
    nz, ny, nx = field.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(1.0 / nx, 1.0 / ny, 1.0 / nz)
    image.SetOrigin(0.5 / nx, 0.5 / ny, 0.5 / nz)
    scalar = numpy_to_vtk(np.ascontiguousarray(field).ravel(order="C"), deep=True)
    scalar.SetName("q")
    image.GetPointData().SetScalars(scalar)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, level)
    contour.ComputeNormalsOn()
    contour.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(contour.GetOutputPort())
    triangles.Update()
    surface = triangles.GetOutput()
    if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfPolys() == 0:
        return None
    points = vtk_to_numpy(surface.GetPoints().GetData())
    polygons = surface.GetPolys()
    if hasattr(polygons, "GetConnectivityArray"):
        faces = vtk_to_numpy(polygons.GetConnectivityArray()).reshape(-1, 3)
    else:
        cells = vtk_to_numpy(polygons.GetData()).reshape(-1, 4)
        if not np.all(cells[:, 0] == 3):
            return None
        faces = cells[:, 1:]
    return points, faces


def _slice_contour_surface(axis, field: np.ndarray, level: float, *, color: str) -> None:
    """Dependency-free, explicitly slice-based fallback when VTK is unavailable."""
    nz, ny, nx = field.shape
    x = _cell_centers(nx)
    y = _cell_centers(ny)
    for index in np.unique(np.linspace(0, nz - 1, min(9, nz), dtype=int)):
        plane = field[int(index)]
        if float(np.nanmin(plane)) <= level <= float(np.nanmax(plane)):
            axis.contour(
                x,
                y,
                plane,
                levels=(level,),
                zdir="z",
                offset=(float(index) + 0.5) / nz,
                colors=(color,),
                linewidths=1.0,
            )


def _wireframe_patches_3d(
    axis,
    patches: np.ndarray,
    *,
    finest_level: int,
    finest_shape: tuple[int, int, int],
) -> None:
    if patches.size == 0 or patches.shape[1] != 7:
        return
    edges = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    finest_xyz = np.asarray(tuple(reversed(finest_shape)), dtype=np.float64)
    for row in patches:
        patch_level = int(row[-1])
        scale = 2 ** (finest_level - patch_level)
        level_shape = finest_xyz / scale
        lower = np.asarray(row[:3], dtype=np.float64) / level_shape
        upper = (np.asarray(row[3:6], dtype=np.float64) + 1.0) / level_shape
        corners = np.asarray(
            [
                (x, y, z)
                for z in (lower[2], upper[2])
                for y in (lower[1], upper[1])
                for x in (lower[0], upper[0])
            ]
        )
        for start, end in edges:
            axis.plot(
                corners[(start, end), 0],
                corners[(start, end), 1],
                corners[(start, end), 2],
                color="#303030",
                linewidth=0.45,
                alpha=0.65,
            )


def _plot_isosurface_3d(
    plt, frames: list[dict[str, object]], target: Path, *, epsilon: float
) -> Path:
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frame = _maximum_error_frame(frames)
    levels = (1.0 - 0.5 * epsilon, 1.0 + 0.5 * epsilon)
    figure = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    axes = (
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    )
    try:
        import vtk  # noqa: F401

        vtk_available = True
    except ImportError:
        vtk_available = False
    for axis, field, title, colors in zip(
        axes,
        (frame["exact"], frame["numeric"]),
        ("solution exacte", "PoPS"),
        (("#6BA3B5", "#174A5B"), ("#F0B27A", "#D47A2C")),
        strict=True,
    ):
        surfaces_drawn = 0
        for level, color in zip(levels, colors, strict=True):
            surface = _vtk_isosurface(field, level) if vtk_available else None
            if surface is not None:
                points, faces = surface
                collection = Poly3DCollection(
                    points[faces],
                    facecolor=color,
                    edgecolor="#202020",
                    linewidth=0.06,
                    alpha=0.66,
                )
                axis.add_collection3d(collection)
                surfaces_drawn += 1
            elif not vtk_available:
                _slice_contour_surface(axis, field, level, color=color)
        if vtk_available and surfaces_drawn < len(levels):
            axis.text2D(
                0.5,
                0.5,
                "%d niveau(x) absent(s)\nplage q : [%.4f, %.4f]"
                % (
                    len(levels) - surfaces_drawn,
                    float(np.nanmin(field)),
                    float(np.nanmax(field)),
                ),
                transform=axis.transAxes,
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
            )
        _wireframe_patches_3d(
            axis,
            frame["patches"],
            finest_level=int(frame["level"]),
            finest_shape=frame["native_shape"],
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            zlim=(0.0, 1.0),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=title,
        )
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=-58)
    method = (
        "surfaces triangulées VTK" if vtk_available else "contours de tranches (VTK indisponible)"
    )
    figure.suptitle(
        "Niveaux q=1±ε/2 à t=%.4f (instant d'erreur L2 maximale) — %s" % (frame["time"], method)
    )
    figure.legend(
        handles=(
            Patch(facecolor="#6BA3B5", label="q=1-ε/2"),
            Patch(facecolor="#174A5B", label="q=1+ε/2"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
    )
    _save_figure(figure, target, dpi=180)
    plt.close(figure)
    return target


def _plot_diagnostics(plt, metadata: dict[str, object], target: Path) -> Path | None:
    history = (
        metadata.get("metrics", {}).get("time_history", {})
        if isinstance(metadata.get("metrics"), dict)
        else {}
    )
    times = history.get("time") if isinstance(history, dict) else None
    masses = history.get("mass") if isinstance(history, dict) else None
    if not times or not masses:
        return None
    time_values = np.asarray(times, dtype=float)
    mass_values = np.asarray(masses, dtype=float)
    amplitude_values = np.asarray(
        [np.nan if value is None else value for value in history.get("amplitude_rms", ())],
        dtype=float,
    )
    exact_amplitude_values = np.asarray(
        [np.nan if value is None else value for value in history.get("exact_amplitude_rms", ())],
        dtype=float,
    )
    phase_values = np.asarray(
        [np.nan if value is None else value for value in history.get("phase_cosine", ())],
        dtype=float,
    )
    signed_phase_values = np.asarray(
        [np.nan if value is None else value for value in history.get("phase_error_cycles", ())],
        dtype=float,
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(time_values, mass_values / mass_values[0] - 1.0, marker="o")
    axes[0].set(xlabel="temps", ylabel="dérive relative de masse")
    if amplitude_values.size == time_values.size:
        axes[1].plot(time_values, amplitude_values, marker="o", label="PoPS")
    if exact_amplitude_values.size == time_values.size:
        axes[1].plot(time_values, exact_amplitude_values, "--", label="exacte")
    if axes[1].lines:
        axes[1].legend()
    axes[1].set(xlabel="temps", ylabel="amplitude RMS")
    if signed_phase_values.size == time_values.size and np.isfinite(signed_phase_values).any():
        axes[2].plot(time_values, signed_phase_values, marker="o")
        axes[2].axhline(0.0, color="#444444", linewidth=0.8)
        axes[2].set(xlabel="temps", ylabel="retard de phase signé [cycles]")
    elif phase_values.size == time_values.size:
        axes[2].plot(time_values, phase_values, marker="o")
        axes[2].set(xlabel="temps", ylabel="corrélation de phase", ylim=(-1.05, 1.05))
    for name, style in zip(("l1", "l2", "linf"), ("-", "--", ":"), strict=True):
        values = np.asarray(
            [np.nan if value is None else value for value in history.get(name, ())],
            dtype=float,
        )
        if values.size == time_values.size:
            axes[3].plot(time_values, values, linestyle=style, marker="o", label=name.upper())
    axes[3].set(xlabel="temps", ylabel="erreur pondérée")
    if axes[3].lines:
        axes[3].legend()
    _save_figure(figure, target, dpi=160)
    plt.close(figure)
    return target


def _observed_regrid_events(
    frames: list[dict[str, object]],
) -> list[tuple[float, int]]:
    counts = np.asarray([frame["regrid_count"] for frame in frames], dtype=np.int64)
    return [
        (float(frames[index]["time"]), int(delta))
        for index, delta in enumerate(np.diff(counts), start=1)
        if delta > 0
    ]


def _draw_regrid_lines(axes, events: list[tuple[float, int]]) -> None:
    for event_index, (time, increment) in enumerate(events):
        for axis in np.atleast_1d(axes).flat:
            axis.axvline(
                time,
                color="#A33E3E",
                linestyle=":",
                linewidth=0.9,
                alpha=0.75,
                label=("regrid observé (Δ=%d)" % increment if event_index == 0 else None),
            )


def _plot_interface_vs_bulk_error(
    plt,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
    target: Path,
) -> Path | None:
    """Compare errors on explicit interface leaves and bulk leaves, never inferred labels."""
    if metadata.get("layout") == "uniform":
        return None
    times = np.asarray([frame["time"] for frame in frames], dtype=np.float64)
    interface_values = {name: [] for name in ("l1", "l2", "linf")}
    bulk_values = {name: [] for name in ("l1", "l2", "linf")}
    valid_frames = 0
    for frame in frames:
        error = np.asarray(frame["numeric"]) - np.asarray(frame["exact"])
        active = np.isfinite(error)
        interface = np.asarray(frame["interface"])
        if interface.dtype != np.bool_ or interface.shape != error.shape:
            raise ValueError("frame interface labels must be a boolean field mask")
        interface = interface & active
        bulk = ~interface & active
        if not interface.any() or not bulk.any():
            for values in (interface_values, bulk_values):
                for name in values:
                    values[name].append(np.nan)
            continue
        valid_frames += 1
        for selection, values in ((interface, interface_values), (bulk, bulk_values)):
            selected = error[selection]
            values["l1"].append(float(np.mean(np.abs(selected))))
            values["l2"].append(float(np.sqrt(np.mean(selected * selected))))
            values["linf"].append(float(np.max(np.abs(selected))))
    if valid_frames < 2:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, name in zip(axes, ("l1", "l2", "linf"), strict=True):
        axis.plot(times, interface_values[name], "o-", label="interface coarse–fine")
        axis.plot(times, bulk_values[name], "s--", label="bulk")
        axis.set(xlabel="temps", ylabel="erreur %s" % name.upper())
    events = _observed_regrid_events(frames)
    _draw_regrid_lines(axes, events)
    axes[0].legend()
    figure.suptitle(
        "Interface vs bulk — feuilles actives adjacentes par face, topologie périodique"
    )
    _save_figure(figure, target, dpi=170)
    plt.close(figure)
    return target


def _plot_regrid_events(
    plt,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
    target: Path,
) -> Path | None:
    """Plot sampled public AMR counters; vertical lines are right-end observations."""
    if metadata.get("layout") == "uniform":
        return None
    events = _observed_regrid_events(frames)
    if not events:
        return None
    times = np.asarray([frame["time"] for frame in frames], dtype=np.float64)
    regrid_counts = np.asarray([frame["regrid_count"] for frame in frames], dtype=np.int64)
    topology_epochs = np.asarray([frame["topology_epoch"] for frame in frames], dtype=np.int64)
    patch_counts = np.asarray(
        [len(np.asarray(frame["patches"])) for frame in frames], dtype=np.int64
    )
    l2 = np.asarray(
        [
            np.sqrt(np.nanmean((np.asarray(frame["numeric"]) - np.asarray(frame["exact"])) ** 2))
            for frame in frames
        ],
        dtype=np.float64,
    )
    if not np.isfinite(l2).all():
        raise ValueError("regrid diagnostic requires finite composed L2 errors")
    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    axes[0].step(times, regrid_counts, where="post", label="regrid_count")
    axes[0].step(times, topology_epochs, where="post", label="topology_epoch")
    axes[0].set(ylabel="compteur public")
    axes[0].legend()
    axes[1].step(times, patch_counts, where="post", color="#3A7D44")
    axes[1].set(ylabel="patches AMR")
    axes[1].set_ylim(float(np.min(patch_counts)) - 0.5, float(np.max(patch_counts)) + 0.5)
    axes[2].plot(times, l2, "o-", color="#355C9A")
    axes[2].set(xlabel="temps", ylabel="erreur L2")
    _draw_regrid_lines(axes, events)
    figure.suptitle(
        "Événements de regrid observés aux snapshots\n"
        "traits = borne droite d'observation; instant interne exact non revendiqué",
        fontsize=11,
    )
    _save_figure(figure, target, dpi=170)
    plt.close(figure)
    return target


def _validated_history(metadata: dict[str, object]) -> dict[str, np.ndarray]:
    metrics = metadata.get("metrics")
    timeline = metadata.get("timeline")
    if not isinstance(metrics, dict) or not isinstance(timeline, dict):
        raise ValueError("comparison requires metrics and timeline metadata")
    history = metrics.get("time_history")
    if not isinstance(history, dict):
        raise ValueError("comparison requires a time_history object")
    expected_time = np.asarray(timeline.get("times"), dtype=np.float64)
    result: dict[str, np.ndarray] = {}
    for name in ("time", "mass_relative_drift", "l1", "l2", "linf"):
        values = np.asarray(history.get(name), dtype=np.float64)
        if values.shape != expected_time.shape or not np.isfinite(values).all():
            raise ValueError("comparison time_history.%s is missing or non-finite" % name)
        result[name] = values
    if not np.array_equal(result["time"], expected_time):
        raise ValueError("comparison history times do not match authenticated timeline times")
    return result


def _run_label(metadata: dict[str, object]) -> str:
    execution = metadata["provenance"]["execution"]
    runtime = execution["runtime"]
    backend = runtime.get("kokkos_backend")
    concurrency = runtime.get("kokkos_concurrency")
    metrics = metadata.get("metrics")
    qualification = metrics.get("qualification") if isinstance(metrics, dict) else None
    integrity = (
        qualification.get("run_integrity_passed", qualification.get("passed"))
        if isinstance(qualification, dict)
        else None
    )
    if (
        not isinstance(backend, str)
        or not backend
        or type(concurrency) is not int
        or type(integrity) is not bool
    ):
        raise ValueError("comparison labels require backend, Kokkos concurrency, and run integrity")
    return "%s/%s — %s c=%d, np=%d, intégrité=%s" % (
        metadata["layout"],
        metadata["subcycling"],
        backend,
        concurrency,
        metadata["mpi_ranks"],
        "PASS" if integrity else "FAIL",
    )


_COMPARISON_KEYS = (
    "schema_version",
    "case",
    "dimension",
    "resolution",
    "mode",
    "wave_numbers",
    "velocity",
    "epsilon",
    "probe_time",
    "period",
    "cycles",
    "final_time",
    "layout",
    "subcycling",
    "block_size",
    "patch_marker",
    "mpi",
    "mpi_ranks",
    "timeline",
    "amr_diagnostics",
)


def _comparison_compatible(
    left: dict[str, object],
    right: dict[str, object],
    *,
    ignored: set[str],
) -> bool:
    if any(
        key not in left or key not in right or left[key] != right[key]
        for key in _COMPARISON_KEYS
        if key not in ignored
    ):
        return False
    left_metrics = left.get("metrics")
    right_metrics = right.get("metrics")
    return bool(
        isinstance(left_metrics, dict)
        and isinstance(right_metrics, dict)
        and left_metrics.get("method") == right_metrics.get("method")
        and _convergence_provenance_signature(left)
        == _convergence_provenance_signature(right)
        is not None
    )


def _plot_run_comparison(
    plt,
    left: tuple[dict[str, np.ndarray], dict[str, object]],
    right: tuple[dict[str, np.ndarray], dict[str, object]],
    *,
    title: str,
    target: Path,
) -> Path:
    left_data, left_metadata = left
    right_data, right_metadata = right
    left_history = _validated_history(left_metadata)
    right_history = _validated_history(right_metadata)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for metadata, history, style in (
        (left_metadata, left_history, "o-"),
        (right_metadata, right_history, "s--"),
    ):
        label = _run_label(metadata)
        axes[0].plot(history["time"], history["l2"], style, label=label)
        axes[1].plot(
            history["time"],
            np.abs(history["mass_relative_drift"]),
            style,
            label=label,
        )
    axes[0].set(xlabel="temps", ylabel="erreur L2")
    axes[1].set(xlabel="temps", ylabel="|dérive de masse relative|")
    axes[0].legend(fontsize="small")
    axes[1].legend(fontsize="small")
    for data, metadata in (left, right):
        _, frames = _timeline_frames(data, metadata)
        _draw_regrid_lines(axes, _observed_regrid_events(frames))
    figure.suptitle(title + " — traits = hausse de regrid_count observée au snapshot")
    _save_figure(figure, target, dpi=170)
    plt.close(figure)
    return target


def _plot_amr_comparisons(
    plt,
    runs: list[tuple[dict[str, np.ndarray], dict[str, object]]],
    output: Path,
) -> list[Path]:
    """Emit only uniquely matched comparisons; an absent/ambiguous pair makes no figure."""
    generated: list[Path] = []
    uniform_pairs = [
        (uniform, amr)
        for uniform in runs
        for amr in runs
        if uniform[1].get("layout") == "uniform"
        and uniform[1].get("subcycling") == "synchronous"
        and amr[1].get("layout") in {"amr-frozen", "amr-mobile"}
        and amr[1].get("subcycling") == "synchronous"
        and _comparison_compatible(
            uniform[1],
            amr[1],
            # `block_size` is an AMR patch-layout control.  Uniform runs are required to keep
            # the CLI default, where it is deliberately inert, so equality here would reject
            # the scientifically valid comparison whenever the AMR campaign exercises another
            # patch size.
            ignored={"layout", "block_size", "patch_marker"},
        )
    ]
    uniform_target = output / "uniform_vs_amr.png"
    if len(uniform_pairs) == 1:
        generated.append(
            _plot_run_comparison(
                plt,
                *uniform_pairs[0],
                title="Uniforme vs AMR synchrone",
                target=uniform_target,
            )
        )
    subcycling_pairs = [
        (synchronous, subcycled)
        for synchronous in runs
        for subcycled in runs
        if synchronous[1].get("layout") in {"amr-frozen", "amr-mobile"}
        and synchronous[1].get("subcycling") == "synchronous"
        and subcycled[1].get("layout") == synchronous[1].get("layout")
        and subcycled[1].get("subcycling") == "subcycled"
        and _comparison_compatible(synchronous[1], subcycled[1], ignored={"subcycling"})
    ]
    subcycling_target = output / "subcycling_vs_nosubcycling.png"
    if len(subcycling_pairs) == 1:
        generated.append(
            _plot_run_comparison(
                plt,
                *subcycling_pairs[0],
                title="AMR subcyclé vs non subcyclé",
                target=subcycling_target,
            )
        )
    return generated


def _comparison_controls_match(
    left: dict[str, object], right: dict[str, object], *, ignored: set[str]
) -> bool:
    controls = (
        "schema_version",
        "case",
        "dimension",
        "resolution",
        "mode",
        "wave_numbers",
        "velocity",
        "epsilon",
        "probe_time",
        "final_time",
        "cycles",
        "layout",
        "subcycling",
        "block_size",
        "patch_marker",
        "mpi",
        "time_snapshots",
        "timeline",
        "amr_diagnostics",
    )
    if any(
        name not in left
        or name not in right
        or left[name] != right[name]
        for name in controls
        if name not in ignored
    ):
        return False
    left_metrics = left.get("metrics")
    right_metrics = right.get("metrics")
    if not isinstance(left_metrics, dict) or not isinstance(right_metrics, dict):
        return False
    if left_metrics.get("method") != right_metrics.get("method"):
        return False
    left_source = left.get("source_fingerprint")
    right_source = right.get("source_fingerprint")
    return isinstance(left_source, str) and left_source == right_source


def _final_error(metadata: dict[str, object], norm: str = "l2") -> float | None:
    metrics = metadata.get("metrics")
    errors = metrics.get("errors") if isinstance(metrics, dict) else None
    value = errors.get(norm) if isinstance(errors, dict) else None
    if type(value) not in (int, float) or not np.isfinite(float(value)) or float(value) <= 0.0:
        return None
    return float(value)


def _plot_mpi_invariance(plt, metadata_by_id: dict[str, dict[str, object]], output: Path) -> Path | None:
    identifiers = ("d2-mpi-np1", "d2-mpi-np2", "d2-mpi-np4")
    if any(identifier not in metadata_by_id for identifier in identifiers):
        return None
    rows = [metadata_by_id[identifier] for identifier in identifiers]
    if (
        [row.get("mpi_ranks") for row in rows] != [1, 2, 4]
        or any(not row.get("mpi") for row in rows)
        or any(
            not _comparison_controls_match(rows[0], row, ignored={"mpi_ranks"})
            for row in rows[1:]
        )
    ):
        return None
    errors = [_final_error(row) for row in rows]
    if any(value is None for value in errors):
        return None
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ranks = np.asarray([1, 2, 4], dtype=int)
    axis.semilogy(ranks, np.asarray(errors, dtype=float), "o-", color="#174A5B")
    axis.set(
        xlabel="rangs MPI",
        ylabel="erreur L2 finale à T=1",
        title="Invariance MPI 2D — même cas, décomposition différente",
        xticks=ranks,
    )
    axis.grid(True, which="both", color="#d8d8d8", linewidth=0.6)
    _save_figure(figure, output / "mpi_invariance_dim2.png", dpi=170)
    plt.close(figure)
    return output / "mpi_invariance_dim2.png"


def _plot_block_size_comparison(
    plt, metadata_by_id: dict[str, dict[str, object]], output: Path
) -> Path | None:
    """Plot the explicitly declared 2D y-advection 8/16/32 block study only."""
    identifiers = ("d2-y-block8", "d2-y-block16", "d2-y")
    if any(identifier not in metadata_by_id for identifier in identifiers):
        return None
    rows = [metadata_by_id[identifier] for identifier in identifiers]
    if [row.get("block_size") for row in rows] != [8, 16, 32] or any(
        not _comparison_controls_match(rows[0], row, ignored={"block_size"})
        for row in rows[1:]
    ):
        return None
    errors = [_final_error(row) for row in rows]
    if any(value is None for value in errors):
        return None
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    sizes = np.asarray([int(row["block_size"]) for row in rows], dtype=int)
    axis.semilogy(sizes, np.asarray(errors, dtype=float), "o-", color="#D47A2C")
    axis.set(
        xlabel="taille de bloc",
        ylabel="erreur L2 finale à T=1",
        title="Sensibilité à la taille de bloc — contrôles identiques",
        xticks=sizes,
    )
    axis.grid(True, which="both", color="#d8d8d8", linewidth=0.6)
    _save_figure(figure, output / "block_size_comparison.png", dpi=170)
    plt.close(figure)
    return output / "block_size_comparison.png"


def _convergence_provenance_signature(
    metadata: dict[str, object],
) -> dict[str, object] | None:
    provenance = metadata.get("provenance")
    identity_inputs = metadata.get("result_identity_inputs")
    if not isinstance(provenance, dict) or not isinstance(identity_inputs, dict):
        return None
    source = provenance.get("source")
    execution = provenance.get("execution")
    if (
        not isinstance(source, dict)
        or not isinstance(execution, dict)
        or identity_inputs.get("source_fingerprint") != source.get("fingerprint")
        or identity_inputs.get("execution") != execution
    ):
        return None
    source_fingerprint = source.get("fingerprint")
    repository_sha = source.get("repository_sha")
    runtime = execution.get("runtime")
    environment = execution.get("environment")
    if (
        not isinstance(source_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_fingerprint) is None
        or not isinstance(repository_sha, str)
        or not repository_sha
        or not isinstance(runtime, dict)
        or not isinstance(environment, dict)
    ):
        return None
    runtime_keys = (
        "has_kokkos",
        "kokkos_backend",
        "kokkos_device",
        "kokkos_shared_space",
        "field_memory_space",
        "kokkos_concurrency",
        "mpi_compiled",
        "mpi_active",
        "mpi_ranks",
        "communicator",
    )
    environment_keys = (
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "KOKKOS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    )
    if any(key not in runtime for key in runtime_keys) or any(
        key not in environment for key in environment_keys
    ):
        return None
    return {
        "repository_sha": repository_sha,
        "source_fingerprint": source_fingerprint,
        "pops_version": provenance.get("pops_version"),
        "runtime": {key: runtime[key] for key in runtime_keys},
        "environment": {key: environment[key] for key in environment_keys},
    }


def _plot_convergence(
    plt,
    metadata: list[dict[str, object]],
    output: Path,
    *,
    error_key: str = "probe_errors",
    target_name: str = "convergence.png",
    title: str | None = None,
) -> Path | None:
    """Plot only one compatible isotropic series; otherwise make no claim."""
    target = output / target_name
    required_equal = (
        "schema_version",
        "case",
        "dimension",
        "mode",
        "wave_numbers",
        "velocity",
        "epsilon",
        "probe_time",
        "final_time",
        "cycles",
        "layout",
        "subcycling",
        "block_size",
        "patch_marker",
        "mpi",
        "mpi_ranks",
        "timeline",
        "amr_diagnostics",
    )
    required_present = (*required_equal, "provenance", "result_identity_inputs")
    if len(metadata) < 2 or any(
        any(key not in item for key in required_present) for item in metadata
    ):
        return None
    first = metadata[0]
    if any(any(item[key] != first[key] for key in required_equal) for item in metadata[1:]):
        return None
    first_metrics = first.get("metrics")
    if not isinstance(first_metrics, dict):
        return None
    first_method = first_metrics.get("method")
    if not isinstance(first_method, dict):
        return None
    first_provenance = _convergence_provenance_signature(first)
    if first_provenance is None:
        return None
    rows: list[tuple[int, dict[str, object]]] = []
    for item in metadata:
        resolution = item.get("resolution")
        metrics = item.get("metrics")
        if (
            not isinstance(resolution, list)
            or not resolution
            or len(set(resolution)) != 1
            or not isinstance(metrics, dict)
        ):
            return None
        qualification = metrics.get("qualification")
        if (
            not isinstance(qualification, dict)
            or qualification.get("run_integrity_passed", qualification.get("passed")) is not True
            or metrics.get("method") != first_method
            or _convergence_provenance_signature(item) != first_provenance
        ):
            return None
        errors = metrics.get(error_key)
        if not isinstance(errors, dict) or any(name not in errors for name in ("l1", "l2", "linf")):
            return None
        rows.append((int(resolution[0]), errors))
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        return None
    counts = np.asarray([row[0] for row in rows], dtype=float)
    figure, axis = plt.subplots(figsize=(6, 4))
    for name in ("l1", "l2", "linf"):
        values = np.asarray([float(row[1][name]) for row in rows], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            plt.close(figure)
            return None
        orders = [
            order for order in convergence_orders(counts.astype(int), values) if order is not None
        ]
        label = "%s (p=%s)" % (name.upper(), ", ".join("%.2f" % order for order in orders))
        axis.loglog(counts, values, "o-", label=label)
    guide = float(rows[0][1]["l2"]) * (counts / counts[0]) ** -2
    axis.loglog(counts, guide, "k--", label="guide ordre 2")
    axis.set(
        xlabel="résolution isotrope N",
        ylabel="erreur",
        title=title
        or ("Convergence mesurée à t=%g" % float(first["probe_time"])),
    )
    axis.text(
        0.02,
        0.02,
        "Intégrité du run validée; la précision reste quantifiée par les erreurs.",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
    )
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, target, dpi=160)
    plt.close(figure)
    return target


def _plot_final_time_convergence(
    plt,
    metadata: list[dict[str, object]],
    output: Path,
    *,
    dimension: int,
) -> Path | None:
    return _plot_convergence(
        plt,
        metadata,
        output,
        error_key="errors",
        target_name="convergence_dim%d_final_time.png" % dimension,
        title="Convergence normative à T=1 (Dim%d)" % dimension,
    )


def _plot_snapshot(
    plt,
    data: dict[str, np.ndarray],
    *,
    figures: Path,
    prefix: str,
    suffix: str,
) -> Path:
    numeric, exact, mask, level = _field_pair(data, prefix=prefix)
    numeric = np.where(mask, numeric, np.nan)
    exact = np.where(mask, exact, np.nan)
    coordinates = tuple(data[prefix + name] for name in ("x", "y", "z") if prefix + name in data)
    if not coordinates:
        coordinates = tuple(
            data["%s%s_level_%d" % (prefix, name, level)]
            for name in ("x", "y", "z")
            if "%s%s_level_%d" % (prefix, name, level) in data
        )
    if numeric.ndim == 1:
        return _plot_1d(plt, numeric, exact, coordinates, figures / ("profile_1d%s.png" % suffix))
    if numeric.ndim == 2:
        return _plot_2d(
            plt,
            numeric,
            exact,
            data.get(prefix + "patch_boxes", np.empty((0, 5), dtype=int)),
            level,
            figures / ("fields_2d%s.png" % suffix),
        )
    if numeric.ndim == 3:
        return _plot_3d(
            plt,
            numeric,
            exact,
            np.asarray(data.get(prefix + "patch_boxes", np.empty((0,), dtype=int))),
            level,
            figures / ("cuts_3d%s.png" % suffix),
        )
    raise ValueError("sine-wave fields must have rank 1, 2, or 3")


def _render_run(
    plt,
    data: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    figures: Path,
    fps: int,
) -> list[Path]:
    """Render one validated result into an empty publication staging directory."""
    _, frames = _timeline_frames(data, metadata)
    generated = [_plot_snapshot(plt, data, figures=figures, prefix="", suffix="")]
    if any(name.startswith("probe_numeric") for name in data):
        generated.append(
            _plot_snapshot(plt, data, figures=figures, prefix="probe_", suffix="_probe")
        )
    diagnostics = _plot_diagnostics(plt, metadata, figures / "diagnostics_time.png")
    if diagnostics is not None:
        generated.append(diagnostics)
    interface_diagnostic = _plot_interface_vs_bulk_error(
        plt, frames, metadata, figures / "interface_vs_bulk_error.png"
    )
    if interface_diagnostic is not None:
        generated.append(interface_diagnostic)
    regrid_diagnostic = _plot_regrid_events(plt, frames, metadata, figures / "regrid_events.png")
    if regrid_diagnostic is not None:
        generated.append(regrid_diagnostic)
    dimension = int(metadata["dimension"])
    if dimension == 1:
        generated.append(_animate_1d(plt, frames, figures / "evolution_1d.gif", fps=fps))
    elif dimension == 2:
        generated.append(_animate_2d(plt, frames, figures / "evolution_2d.gif", fps=fps))
    else:
        generated.extend(
            (
                _storyboard_3d(plt, frames, figures / "storyboard_3d_cuts.png"),
                _plot_oblique_cut_3d(plt, frames, metadata, figures / "oblique_cut_3d.png"),
                _plot_isosurface_3d(
                    plt,
                    frames,
                    figures / "isosurface_3d.png",
                    epsilon=float(metadata.get("epsilon", 0.1)),
                ),
                _plot_maximum_spatial_error_3d(
                    plt, frames, figures / "maximum_spatial_error_3d.png"
                ),
                _animate_3d_cuts(plt, frames, figures / "evolution_3d.gif", fps=fps),
            )
        )
    return sorted(generated)


def _mobile_trajectory_rows(metadata: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the authenticated mobile-window witness, never inferred patch motion."""
    dimension = metadata.get("dimension")
    coverage = metadata.get("coverage")
    marker = metadata.get("patch_marker")
    witnesses = coverage.get("witnesses") if isinstance(coverage, dict) else None
    witness = witnesses.get("prescribed_mobile_regrid") if isinstance(witnesses, dict) else None
    if type(dimension) is not int or dimension not in (1, 2, 3) or not isinstance(witness, dict):
        raise ValueError("mobile trajectory figure requires one dimensional mobile witness")
    trajectory = witness.get("expected_trajectory")
    snapshots = witness.get("snapshots")
    if (
        witness.get("observed") is not True
        or not isinstance(trajectory, dict)
        or trajectory.get("periodic_axes") != list(range(dimension))
        or trajectory.get("formula") != "(center + velocity * time) mod 1"
        or not isinstance(snapshots, list)
        or not snapshots
        or not isinstance(marker, dict)
        or marker.get("kind") != "prescribed_window"
    ):
        raise ValueError("mobile trajectory figure requires an observed prescribed-window receipt")
    center = np.asarray(trajectory.get("center"), dtype=np.float64)
    velocity = np.asarray(trajectory.get("velocity"), dtype=np.float64)
    if (
        center.shape != (dimension,)
        or velocity.shape != (dimension,)
        or not np.isfinite(center).all()
        or not np.isfinite(velocity).all()
        or marker.get("center") != center.tolist()
        or marker.get("velocity") != velocity.tolist()
    ):
        raise ValueError("mobile trajectory receipt disagrees with the prescribed patch marker")
    times: list[float] = []
    expected: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for row in snapshots:
        if not isinstance(row, dict) or type(row.get("index")) is not int:
            raise ValueError("mobile trajectory snapshot has an invalid index")
        time = row.get("time")
        box = row.get("box")
        expected_center = np.asarray(row.get("expected_center"), dtype=np.float64)
        box_center = np.asarray(row.get("box_center"), dtype=np.float64)
        if (
            type(time) not in (int, float)
            or not np.isfinite(float(time))
            or expected_center.shape != (dimension,)
            or box_center.shape != (dimension,)
            or not np.isfinite(expected_center).all()
            or not np.isfinite(box_center).all()
            or type(row.get("window_probe_count")) is not int
            or row["window_probe_count"] != 1 + 2 * dimension
            or not isinstance(box, dict)
            or type(box.get("level")) is not int
            or box["level"] < 1
            or not isinstance(box.get("lower"), list)
            or not isinstance(box.get("upper"), list)
            or len(box["lower"]) != dimension
            or len(box["upper"]) != dimension
        ):
            raise ValueError("mobile trajectory snapshot is malformed")
        predicted = np.mod(center + velocity * float(time), 1.0)
        if not np.allclose(expected_center, predicted, rtol=0.0, atol=1.0e-12):
            raise ValueError("mobile trajectory expected center violates its sealed formula")
        times.append(float(time))
        expected.append(expected_center)
        observed.append(box_center)
    if any(right <= left for left, right in zip(times[:-1], times[1:], strict=True)):
        raise ValueError("mobile trajectory snapshots are not strictly chronological")
    return np.asarray(times), np.vstack(expected), np.vstack(observed)


def _plot_mobile_patch_trajectory(plt, metadata: dict[str, object], target: Path) -> Path:
    """Show prescribed and observed clustering-box centres at sealed regrid snapshots."""
    times, expected, observed = _mobile_trajectory_rows(metadata)
    dimension = expected.shape[1]
    figure, axes = plt.subplots(dimension, 1, figsize=(8.5, 2.5 * dimension), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, index, label in zip(axes, range(dimension), ("x", "y", "z"), strict=False):
        axis.plot(times, expected[:, index], "o-", color="#174A5B", label="centre prescrit")
        axis.plot(times, observed[:, index], "s--", color="#D47A2C", label="centre de boîte fine")
        axis.set(ylabel=label, ylim=(-0.03, 1.03))
        axis.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].legend(loc="best")
    axes[-1].set(xlabel="temps (snapshots de regrid observés)")
    figure.suptitle("Patch AMR mobile prescrit — reçu scellé, pas une trajectoire reconstruite")
    _save_figure(figure, target, dpi=170)
    plt.close(figure)
    return target


def _plot_mpi_corner_dim3_np8(
    plt, case: dict[str, object], metadata: dict[str, object], target: Path
) -> Path:
    """Render the exact 2x2x2 ownership-corner receipt as an explanatory 3D figure."""
    _validate_mpi_topology_receipt("d3-mpi-np8-corner", case, metadata)
    receipt = metadata["coverage"]["mpi_topology"]
    corner = receipt["inter_rank_corner_crossing"]
    coordinates = sorted(receipt["rank_coordinates"], key=lambda row: row["rank"])
    figure = plt.figure(figsize=(7.5, 6.5), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    for row in coordinates:
        coordinate = np.asarray(row["coordinate"], dtype=np.float64)
        point = 0.25 + 0.5 * coordinate
        axis.scatter(*point, s=65, color="#174A5B")
        axis.text(*point, "r%d" % row["rank"], color="#111111")
    for value in (0.0, 0.5, 1.0):
        axis.plot((value, value), (0.0, 1.0), (0.5, 0.5), color="#aaaaaa", linewidth=0.7)
        axis.plot((0.0, 1.0), (value, value), (0.5, 0.5), color="#aaaaaa", linewidth=0.7)
        axis.plot((0.5, 0.5), (0.0, 1.0), (value, value), color="#aaaaaa", linewidth=0.7)
    start = np.asarray(corner["characteristic_start"], dtype=np.float64)
    end = np.asarray(corner["corner_coordinate"], dtype=np.float64)
    axis.quiver(*start, *(end - start), color="#A33E3E", arrow_length_ratio=0.08, linewidth=1.8)
    axis.scatter(*end, s=95, color="#A33E3E", marker="*")
    axis.set(xlabel="x", ylabel="y", zlabel="z", xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
    axis.set_title("Reçu MPI 3D : coin (0.5, 0.5, 0.5), 8 rangs participants")
    _save_figure(figure, target, dpi=180)
    plt.close(figure)
    return target


def _sealed_analysis_facts(
    metadata_by_id: dict[str, dict[str, object]]
) -> tuple[dict[str, list[str]], float, list[str]]:
    """Summarize only obligatory witnesses, conservation, and recorded execution environments."""
    obligations: dict[str, list[str]] = {}
    maximum_drift = 0.0
    environments: dict[str, list[str]] = {}
    for case_id, metadata in sorted(metadata_by_id.items()):
        coverage = metadata.get("coverage")
        metrics = metadata.get("metrics")
        provenance = metadata.get("provenance")
        execution = provenance.get("execution") if isinstance(provenance, dict) else None
        conservation = metrics.get("conservation") if isinstance(metrics, dict) else None
        runtime = execution.get("runtime") if isinstance(execution, dict) else None
        environment = execution.get("environment") if isinstance(execution, dict) else None
        requested = coverage.get("requested_obligations") if isinstance(coverage, dict) else None
        if (
            not isinstance(requested, list)
            or not isinstance(conservation, dict)
            or type(conservation.get("max_relative_drift")) not in (int, float)
            or not np.isfinite(float(conservation["max_relative_drift"]))
            or float(conservation["max_relative_drift"]) < 0.0
            or type(conservation.get("timeline_samples")) is not int
            or conservation["timeline_samples"] < 9
            or not isinstance(runtime, dict)
            or not isinstance(environment, dict)
        ):
            raise ValueError("sealed analysis facts are malformed for %s" % case_id)
        maximum_drift = max(maximum_drift, float(conservation["max_relative_drift"]))
        for obligation in requested:
            if not isinstance(obligation, str):
                raise ValueError("sealed analysis has a non-text coverage obligation")
            obligations.setdefault(obligation, []).append(case_id)
        signature = {
            "backend": runtime.get("kokkos_backend"),
            "device": runtime.get("kokkos_device"),
            "concurrency": runtime.get("kokkos_concurrency"),
            "mpi": runtime.get("mpi_ranks"),
            "omp": environment.get("OMP_NUM_THREADS"),
            "kokkos_threads": environment.get("KOKKOS_NUM_THREADS"),
        }
        if any(value is None for value in signature.values()):
            raise ValueError("sealed analysis environment is incomplete for %s" % case_id)
        environments.setdefault(json.dumps(signature, sort_keys=True), []).append(case_id)
    labels = ["%s: %s" % (signature, ", ".join(cases)) for signature, cases in environments.items()]
    return obligations, maximum_drift, labels


def _media_interpretation(name: str) -> str:
    """Keep every emitted visual adjacent to its useful, bounded interpretation."""
    if name.endswith("mobile_patch_trajectory.png"):
        return "Le reçu compare le centre prescrit au centre d'une boîte fine aux seuls snapshots de regrid observés; ce centre de boîte n'est pas un nouvel oracle géométrique."
    if name.endswith("mpi_corner_dim3_np8.png"):
        return "Le cube représente uniquement le reçu de décomposition 2×2×2 et le coin inter-rang authentifié; il n'ajoute pas de preuve au témoin natif."
    if name.endswith(".gif"):
        return "L'animation conserve des échelles temporelles cohérentes pour voir transport, diffusion et interfaces; elle ne remplace ni les normes ni la conservation."
    if "convergence" in name and "final_time" in name:
        return "Cette courbe utilise exclusivement les erreurs finales à T=1 recalculées depuis les champs NPZ scellés."
    if "convergence" in name:
        return "Cette courbe au probe t=0.37 est diagnostique et ne qualifie pas l'ordre final."
    if "mpi_invariance" in name:
        return "La comparaison isole np=1/2/4 pour le cas 2D déclaré; elle ne mesure pas le coût MPI."
    if "block_size" in name:
        return "La comparaison isole la taille de bloc pour le triplet 2D y-advection; elle n'est pas un scaling."
    if "trajectory" in name or "regrid" in name or "interface" in name:
        return "Ce diagnostic est borné par les snapshots et les masques publiés; aucun événement interne non observé n'est inféré."
    if "3d" in name or "isosurface" in name or "oblique" in name:
        return "Cette vue 3D rend visibles coupes, patches ou erreur; sa lecture reste complémentaire aux reçus et aux normes."
    return "Cette figure dérive exclusivement des données scellées et sert de diagnostic visuel, pas de qualification isolée."


def _write_sealed_analysis(
    staging: Path,
    *,
    complete_path: Path,
    complete: dict[str, object],
    metadata_by_id: dict[str, dict[str, object]],
    generated: list[Path],
    recomputed_convergence: dict[str, object] | None = None,
) -> Path:
    """Write a report that describes only the evidence sealed by COMPLETE.json."""
    target = staging / "ANALYSIS.md"
    _require_fresh_path(target)
    media = [path.relative_to(staging).as_posix() for path in sorted(generated)]
    obligations, maximum_drift, environments = _sealed_analysis_facts(metadata_by_id)
    final_convergence = [name for name in media if name.endswith("_final_time.png")]
    diagnostic_convergence = [name for name in media if name.endswith("_probe_time.png")]
    lines = [
        "# Advection sinusoïdale périodique — rapport de données scellées",
        "",
        "## Résultat",
        "",
        "Cette publication dérive exclusivement du manifeste `COMPLETE.json` indiqué ci-dessous. "
        "Chaque paire NPZ/JSON, son identité de résultat et son SHA-256 ont été vérifiés avant le rendu. "
        "Ce rapport ne lance ni PoPS ni le générateur.",
        "",
        "- Manifeste scellé : `%s`" % complete_path.name,
        "- SHA-256 du manifeste : `%s`" % _sha256(complete_path),
        "- Matrice : `%s` (%d cas)" % (complete["matrix"], complete["case_count"]),
        "- Empreinte de la matrice : `%s`" % complete["matrix_sha256"],
        "",
        "## Couverture, conservation et environnement",
        "",
        "Toutes les obligations demandées dans la matrice ont un témoin applicable et observé dans les métadonnées scellées.",
        "",
        *[
            "- `%s` : %s" % (obligation, ", ".join(case_ids))
            for obligation, case_ids in sorted(obligations.items())
        ],
        "",
        "La dérive relative de masse maximale rapportée par les 37 historiques est `%.6g`. Cette conservation est rapportée séparément des normes d'exactitude et de l'ordre." % maximum_drift,
        "",
        "Environnements réellement enregistrés :",
        "",
        *["- %s" % label for label in environments],
        "",
        "## Convergence",
        "",
        "Les figures `convergence_dim*_final_time.png` sont les seules figures normatives : "
        "elles emploient les erreurs finales à `T=1` et les trois séries déclarées dans la matrice. "
        "Toute figure nommée `convergence_dim*_probe_time.png` utilise au contraire le probe à `t=0.37` : elle est "
        "diagnostique, utile pour la propagation, mais ne qualifie pas l'ordre final.",
        "",
    ]
    if final_convergence:
        convergence = recomputed_convergence
        for name in final_convergence:
            dimension = name.removeprefix("convergence_dim").removesuffix("_final_time.png")
            receipt = convergence.get("dim%s" % dimension) if isinstance(convergence, dict) else None
            orders = receipt.get("orders", {}).get("l1") if isinstance(receipt, dict) else None
            if not isinstance(orders, list) or any(type(value) not in (int, float) for value in orders):
                lines.extend(
                    (
                        "![Convergence finale Dim%s](%s)" % (dimension, name),
                        "",
                        "Les ordres ne sont pas affichés sans recalcul indépendant depuis les champs NPZ scellés.",
                        "",
                    )
                )
                continue
            lines.extend(
                (
                    "![Convergence finale Dim%s](%s)" % (dimension, name),
                    "",
                    "Dim%s : ordres L1 finaux observés `%s`, recalculés depuis les champs NPZ scellés puis comparés au reçu `COMPLETE.json`. Cette figure qualifie uniquement la série uniforme déclarée à `T=1`."
                    % (dimension, ", ".join("%.4g" % float(value) for value in orders)),
                    "",
                )
            )
    else:
        lines.append(
            "Aucune figure de convergence finale n'a été publiée : les données scellées ne formaient pas une série compatible."
        )
    if diagnostic_convergence:
        lines.extend(("", "## Diagnostic au probe", ""))
        for name in diagnostic_convergence:
            lines.extend(
                (
                    "![Convergence probe](%s)" % name,
                    "",
                    "Cette lecture à `t=0.37` visualise la propagation avant le retour périodique ; elle ne remplace pas la qualification finale.",
                    "",
                )
            )
    mpi_figure = "mpi_invariance_dim2.png"
    if mpi_figure in media:
        mpi_errors = [_final_error(metadata_by_id[identifier]) for identifier in ("d2-mpi-np1", "d2-mpi-np2", "d2-mpi-np4")]
        if any(value is None for value in mpi_errors):
            raise ValueError("sealed MPI comparison has no finite final errors")
        spread = max(mpi_errors) - min(mpi_errors)
        lines.extend(
            (
                "## Invariance MPI",
                "",
                "![Invariance MPI](%s)" % mpi_figure,
                "",
                "Les trois exécutions 2D ne diffèrent que par `np=1/2/4`; la plage observée des erreurs L2 finales est `%.6g`. Elle mesure une invariance de résultat, pas un coût de communication."
                % spread,
                "",
            )
        )
    block_figure = "block_size_comparison.png"
    if block_figure in media:
        block_errors = [_final_error(metadata_by_id[identifier]) for identifier in ("d2-y-block8", "d2-y-block16", "d2-y")]
        if any(value is None for value in block_errors):
            raise ValueError("sealed block-size comparison has no finite final errors")
        spread = max(block_errors) - min(block_errors)
        lines.extend(
            (
                "## Taille de bloc",
                "",
                "![Taille de bloc](%s)" % block_figure,
                "",
                "Les cas 2D y-advection 8/16/32 ne diffèrent que par `block_size`; la plage des erreurs L2 finales est `%.6g`. Cette figure isole l'effet de la décomposition par blocs, pas le scaling de performance."
                % spread,
                "",
            )
        )
    lines.extend(
        (
            "",
            "## Figures observées",
            "",
            "Les cartes 2D/3D, storyboards et GIFs représentent les snapshots authentifiés. Les boîtes blanches "
            "sont les projections des patches AMR publiés, pas une reconstruction graphique de la topologie. "
            "Les tracés MPI et taille de bloc ne sont émis que lorsqu'ils constituent une expérience à un seul paramètre.",
            "",
        )
    )
    three_dimensional = [
        name
        for name in media
        if name.startswith("d3-cf-subcycled/")
        and name.endswith(("storyboard_3d_cuts.png", "maximum_spatial_error_3d.png", "evolution_3d.gif"))
    ]
    for name in three_dimensional:
        lines.extend(
            (
                "![](%s)" % name,
                "",
                "Les coupes/GIF 3D montrent les champs et les projections de patches publiés au snapshot; ils rendent visibles interfaces, arêtes et évolution. Ils ne prouvent pas seuls les témoins AMR : ceux-ci restent les enregistrements scellés du cas.",
                "",
            )
        )
    amr_diagnostics = [
        name
        for name in media
        if name.startswith("d2-cf-subcycled/")
        and name.endswith(("interface_vs_bulk_error.png", "regrid_events.png", "evolution_2d.gif"))
    ]
    for name in amr_diagnostics:
        lines.extend(
            (
                "![](%s)" % name,
                "",
                "Ce diagnostic AMR utilise les masques interface/bulk et compteurs de regridding authentifiés. Il localise une erreur ou un événement observé; il ne permet pas d'inférer un événement entre deux snapshots.",
                "",
            )
        )
    lines.extend(
        (
            "",
            "## Limites",
            "",
            "L'absence d'une figure est une absence de comparaison contrôlée dans cette publication, pas une preuve de réussite. "
            "Les résultats partiels, les anciens rapports et tout répertoire sans `COMPLETE.json` sont volontairement exclus.",
            "",
        )
    )
    lines.extend(("", "## Inventaire des médias et interprétation", ""))
    for name in media:
        lines.extend(
            (
                "![Média scellé](%s)" % name,
                "",
                _media_interpretation(name),
                "",
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _publish_complete_plot_set(*, complete_path: Path, figures: Path, fps: int) -> list[Path]:
    """Publish immutable figures/reports from a complete matrix, never from partial output."""
    complete, matrix_cases, entries = _sealed_complete_inputs(complete_path)
    metadata_by_id = {case_id: metadata for case_id, _, _, metadata in entries}
    data_by_id: dict[str, dict[str, np.ndarray]] = {}
    for case_id, data_path, metadata_path, authenticated_metadata in entries:
        data, reloaded_metadata = _load(data_path, metadata_path)
        if reloaded_metadata != authenticated_metadata:
            raise ValueError("sealed convergence metadata changed between authentication and recomputation")
        data_by_id[case_id] = data
    matrix = _read_regular_json(MATRIX_PATH, label="matrix")
    recomputed_convergence = _recomputed_convergence_receipt(
        matrix, matrix_cases, metadata_by_id, data_by_id
    )
    _validate_convergence_receipt(complete.get("convergence"), recomputed_convergence)
    paths = [data_path for _, data_path, _, _ in entries]
    metadata_paths = [metadata_path for _, _, metadata_path, _ in entries]
    manifest = _sealed_publication_manifest(
        paths,
        metadata_paths,
        list(metadata_by_id.values()),
        complete_path=complete_path,
        complete=complete,
        fps=fps,
    )
    staging = _create_staging_directory(figures)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        generated: list[Path] = []
        by_id = {case_id: (data_path, metadata_path) for case_id, data_path, metadata_path, _ in entries}
        for case_id in VISUAL_CASE_IDS:
            if case_id not in by_id:
                raise ValueError("sealed complete matrix lacks required visual case %s" % case_id)
            data_path, metadata_path = by_id[case_id]
            data, metadata = _load(data_path, metadata_path)
            destination = staging / case_id
            destination.mkdir()
            generated.extend(_render_run(plt, data, metadata, figures=destination, fps=fps))
        mobile_case = "d3-mobile-sync"
        mpi_corner_case = "d3-mpi-np8-corner"
        if mobile_case not in metadata_by_id or mpi_corner_case not in metadata_by_id:
            raise ValueError("sealed complete matrix lacks required mobile/MPI visual receipts")
        generated.append(
            _plot_mobile_patch_trajectory(
                plt, metadata_by_id[mobile_case], staging / "mobile_patch_trajectory.png"
            )
        )
        generated.append(
            _plot_mpi_corner_dim3_np8(
                plt,
                matrix_cases[mpi_corner_case],
                metadata_by_id[mpi_corner_case],
                staging / "mpi_corner_dim3_np8.png",
            )
        )
        convergence_series = matrix.get("convergence_series")
        if not isinstance(convergence_series, dict):
            raise ValueError("matrix has no convergence declarations")
        for dimension in (1, 2, 3):
            declaration = convergence_series.get("dim%d" % dimension)
            identifiers = declaration.get("case_ids") if isinstance(declaration, dict) else None
            if not isinstance(identifiers, list):
                raise ValueError("matrix convergence declaration is malformed")
            series = [metadata_by_id[identifier] for identifier in identifiers]
            final = _plot_final_time_convergence(plt, series, staging, dimension=dimension)
            if final is not None:
                generated.append(final)
            probe = _plot_convergence(
                plt,
                series,
                staging,
                target_name="convergence_dim%d_probe_time.png" % dimension,
                title="Convergence diagnostique au probe t=0.37 (Dim%d)" % dimension,
            )
            if probe is not None:
                generated.append(probe)
        mpi = _plot_mpi_invariance(plt, metadata_by_id, staging)
        if mpi is not None:
            generated.append(mpi)
        block = _plot_block_size_comparison(plt, metadata_by_id, staging)
        if block is not None:
            generated.append(block)
        comparison_pairs = (
            ("d1-cf-sync", "d1-cf-subcycled"),
            ("d2-cf-sync", "d2-cf-subcycled"),
            ("d3-cf-sync", "d3-cf-subcycled"),
            ("d1-mobile-sync", "d1-mobile-subcycled"),
            ("d2-mobile-sync", "d2-mobile-subcycled"),
            ("d3-mobile-sync", "d3-mobile-subcycled"),
        )
        for left_id, right_id in comparison_pairs:
            left_data, left_metadata = _load(*by_id[left_id])
            right_data, right_metadata = _load(*by_id[right_id])
            if not _comparison_compatible(left_metadata, right_metadata, ignored={"subcycling"}):
                raise ValueError("sealed comparison %s/%s is not controlled" % (left_id, right_id))
            target = staging / ("subcycling_%s_vs_%s.png" % (left_id, right_id))
            generated.append(
                _plot_run_comparison(
                    plt,
                    (left_data, left_metadata),
                    (right_data, right_metadata),
                    title="Subcycling : %s vs %s" % (left_id, right_id),
                    target=target,
                )
            )
        generated.append(
            _write_sealed_analysis(
                staging,
                complete_path=complete_path,
                complete=complete,
                metadata_by_id=metadata_by_id,
                generated=generated,
                recomputed_convergence=recomputed_convergence,
            )
        )
        _write_publication_manifest(staging, manifest)
    except Exception as error:
        raise RuntimeError(
            "sealed plot rendering failed; retained incomplete staging directory for inspection: %s"
            % staging
        ) from error
    published = _publish_staging_directory(staging, figures)
    return [published / path.relative_to(staging) for path in sorted(generated)]


def main() -> None:
    args = _arguments()
    complete = args.complete.resolve()
    figures = args.figures or (
        Path(__file__).with_name("figures") / ("complete_" + _sha256(complete)[:16])
    )
    generated = _publish_complete_plot_set(
        complete_path=complete, figures=figures, fps=args.fps
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
