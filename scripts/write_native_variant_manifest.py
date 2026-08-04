#!/usr/bin/env python3
"""Write one authenticated PoPS native-variant row without importing ``pops``.

The extension is loaded directly from the exact path supplied by CMake under the logical
``pops._pops`` name.  Its compile-time facts, path and final bytes become one row in the manifest
next to ``dimN/``.  Repeated invocations merge disjoint dimensions and replace only the requested
dimension.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from types import ModuleType
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_DIMENSIONS = (1, 2, 3)
MANIFEST_NAME = "variants.json"
_ROW_KEYS = {
    "dimension",
    "path",
    "sha256",
    "version",
    "abi_key",
    "has_mpi",
    "has_kokkos",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")


class NativeVariantManifestError(RuntimeError):
    """A native variant or its manifest is malformed or unauthenticated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_dimensions(values: Iterable[Any], *, where: str) -> tuple[int, ...]:
    """Return one sorted, non-empty set of exact dimensions."""
    dimensions = tuple(values)
    if not dimensions:
        raise NativeVariantManifestError("%s must name at least one dimension" % where)
    if any(type(value) is not int or value not in SUPPORTED_DIMENSIONS for value in dimensions):
        raise NativeVariantManifestError("%s must contain only exact dimensions 1, 2, or 3" % where)
    if len(set(dimensions)) != len(dimensions):
        raise NativeVariantManifestError("%s contains a duplicate dimension" % where)
    return tuple(sorted(dimensions))


def _has_extension_suffix(filename: str) -> bool:
    return any(filename == "_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _validate_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != _ROW_KEYS:
        raise NativeVariantManifestError("native variant manifest row has an unknown schema")
    dimension = row["dimension"]
    if type(dimension) is not int or dimension not in SUPPORTED_DIMENSIONS:
        raise NativeVariantManifestError("native variant dimension must be exactly 1, 2, or 3")
    relative_value = row["path"]
    if not isinstance(relative_value, str) or not relative_value:
        raise NativeVariantManifestError("native variant path must be a non-empty relative path")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or str(relative) != relative_value \
            or any(part in {"", ".", ".."} for part in relative.parts):
        raise NativeVariantManifestError("native variant path is not canonical and relative")
    if len(relative.parts) != 2 or relative.parts[0] != "dim%d" % dimension \
            or not _has_extension_suffix(relative.parts[1]):
        raise NativeVariantManifestError(
            "native variant path must be dimN/_pops<EXT_SUFFIX> for its dimension"
        )
    if not isinstance(row["sha256"], str) or _DIGEST.fullmatch(row["sha256"]) is None:
        raise NativeVariantManifestError("native variant sha256 is malformed")
    for name in ("version", "abi_key"):
        if not isinstance(row[name], str) or not row[name]:
            raise NativeVariantManifestError("native variant %s is empty" % name)
    for name in ("has_mpi", "has_kokkos"):
        if type(row[name]) is not bool:
            raise NativeVariantManifestError("native variant %s must be a boolean" % name)
    return dict(row)


def validate_manifest_payload(
    payload: Any,
    *,
    expected_dimensions: Iterable[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate the closed schema, canonical order and optional exact dimension set."""
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "variants"}:
        raise NativeVariantManifestError("native variant manifest has an unknown schema")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise NativeVariantManifestError("native variant manifest schema version is unsupported")
    variants = payload["variants"]
    if not isinstance(variants, list) or not variants:
        raise NativeVariantManifestError("native variant manifest must contain at least one row")
    rows = tuple(_validate_row(row) for row in variants)
    dimensions = tuple(row["dimension"] for row in rows)
    if dimensions != tuple(sorted(dimensions)) or len(set(dimensions)) != len(dimensions):
        raise NativeVariantManifestError(
            "native variant manifest dimensions must be unique and sorted"
        )
    paths = tuple(row["path"] for row in rows)
    if len(set(paths)) != len(paths):
        raise NativeVariantManifestError("native variant manifest contains a duplicate path")
    if expected_dimensions is not None:
        expected = exact_dimensions(expected_dimensions, where="expected native variants")
        if dimensions != expected:
            raise NativeVariantManifestError(
                "native variant manifest dimensions %s do not equal the explicit set %s"
                % (dimensions, expected)
            )
    return rows


def load_manifest(
    manifest: Path,
    *,
    expected_dimensions: Iterable[int] | None = None,
    verify_files: bool = True,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Read and authenticate a manifest against its sibling ``dimN`` leaves."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeVariantManifestError(
            "native variant manifest is missing or invalid JSON: %s" % manifest
        ) from exc
    rows = validate_manifest_payload(payload, expected_dimensions=expected_dimensions)
    if not verify_files:
        return rows
    root = manifest.parent.resolve()
    for row in rows:
        extension = root.joinpath(*PurePosixPath(row["path"]).parts)
        if extension.is_symlink() or not extension.is_file():
            raise NativeVariantManifestError(
                "native variant is missing or is a symlink: %s" % extension
            )
        resolved = extension.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise NativeVariantManifestError(
                "native variant escapes its manifest root: %s" % extension
            ) from exc
        if verify_hashes and sha256_file(resolved) != row["sha256"]:
            raise NativeVariantManifestError(
                "native variant bytes disagree with variants.json: Dim=%d" % row["dimension"]
            )
    return rows


def manifest_payload(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    validated = [_validate_row(dict(row)) for row in rows]
    validated.sort(key=lambda row: row["dimension"])
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "variants": validated}
    validate_manifest_payload(payload)
    return payload


def write_manifest_atomic(manifest: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Replace ``manifest`` atomically with one canonical JSON document."""
    payload = manifest_payload(rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=manifest.parent, prefix=".variants.", delete=False
    ) as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        temporary.chmod(0o644)
        temporary.replace(manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_exact_extension(extension: Path) -> ModuleType:
    name = "pops._pops"

    def load() -> ModuleType:
        specification = importlib.util.spec_from_file_location(name, extension)
        if specification is None or specification.loader is None:
            raise NativeVariantManifestError("cannot create a loader for %s" % extension)
        module = importlib.util.module_from_spec(specification)
        previous = sys.modules.get(name)
        sys.modules[name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
            raise
        return module

    if not (hasattr(sys, "setdlopenflags") and hasattr(sys, "getdlopenflags")):
        return load()
    previous_flags = sys.getdlopenflags()
    selected_flags = previous_flags
    if hasattr(os, "RTLD_NOW"):
        selected_flags |= os.RTLD_NOW
    if hasattr(os, "RTLD_GLOBAL"):
        selected_flags |= os.RTLD_GLOBAL
    sys.setdlopenflags(selected_flags)
    try:
        return load()
    finally:
        sys.setdlopenflags(previous_flags)


def native_variant_row(
    extension: Path,
    *,
    manifest: Path,
    dimension: int,
    version: str,
) -> dict[str, Any]:
    """Extract one row from the exact compiled leaf and its final bytes."""
    expected_dimension = exact_dimensions((dimension,), where="native variant writer")[0]
    if not isinstance(version, str) or not version:
        raise NativeVariantManifestError("expected package version must not be empty")
    native_root = manifest.parent.resolve()
    binary = extension.resolve()
    if extension.is_symlink() or not binary.is_file():
        raise NativeVariantManifestError("native extension is missing or is a symlink: %s" % extension)
    try:
        relative = binary.relative_to(native_root)
    except ValueError as exc:
        raise NativeVariantManifestError(
            "native extension is outside the manifest root: %s" % binary
        ) from exc
    relative_value = relative.as_posix()
    placeholder = {
        "dimension": expected_dimension,
        "path": relative_value,
        "sha256": "0" * 64,
        "version": version,
        "abi_key": "pending",
        "has_mpi": False,
        "has_kokkos": False,
    }
    _validate_row(placeholder)

    module = _load_exact_extension(binary)
    actual_dimension = getattr(module, "__native_dimension__", None)
    if actual_dimension != expected_dimension or type(actual_dimension) is not int:
        raise NativeVariantManifestError(
            "native extension authenticates Dim=%r, expected Dim=%d"
            % (actual_dimension, expected_dimension)
        )
    actual_version = getattr(module, "__version__", None)
    if actual_version != version:
        raise NativeVariantManifestError(
            "native extension version %r disagrees with expected %r" % (actual_version, version)
        )
    abi_provider = getattr(module, "abi_key", None)
    if not callable(abi_provider):
        raise NativeVariantManifestError("native extension has no callable abi_key")
    abi_key = abi_provider()
    if not isinstance(abi_key, str) or not abi_key:
        raise NativeVariantManifestError("native extension abi_key is empty")
    facts: dict[str, bool] = {}
    for attribute, field in (("__has_mpi__", "has_mpi"), ("__has_kokkos__", "has_kokkos")):
        value = getattr(module, attribute, None)
        if type(value) is not bool:
            raise NativeVariantManifestError("native extension %s fact is not boolean" % attribute)
        facts[field] = value
    return {
        "dimension": expected_dimension,
        "path": relative_value,
        "sha256": sha256_file(binary),
        "version": version,
        "abi_key": abi_key,
        "has_mpi": facts["has_mpi"],
        "has_kokkos": facts["has_kokkos"],
    }


def update_manifest(
    manifest: Path,
    extension: Path,
    *,
    dimension: int,
    version: str,
) -> tuple[dict[str, Any], ...]:
    """Merge one freshly authenticated dimension and atomically publish the result."""
    current: tuple[dict[str, Any], ...] = ()
    if manifest.exists():
        current = load_manifest(manifest, verify_files=False)
    row = native_variant_row(
        extension, manifest=manifest, dimension=dimension, version=version
    )
    retained = [item for item in current if item["dimension"] != dimension]
    for item in retained:
        candidate = manifest.parent.joinpath(*PurePosixPath(item["path"]).parts)
        if candidate.is_symlink() or not candidate.is_file() \
                or sha256_file(candidate.resolve()) != item["sha256"]:
            raise NativeVariantManifestError(
                "refusing to preserve an unauthenticated existing Dim=%d row" % item["dimension"]
            )
    write_manifest_atomic(manifest, [*retained, row])
    return load_manifest(manifest, verify_files=True, verify_hashes=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dimension", required=True, type=int, choices=SUPPORTED_DIMENSIONS)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    try:
        rows = update_manifest(
            args.manifest,
            args.extension,
            dimension=args.dimension,
            version=args.version,
        )
    except (NativeVariantManifestError, OSError, ValueError) as exc:
        print("native variant manifest generation failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({"manifest": str(args.manifest.resolve()), "variants": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
