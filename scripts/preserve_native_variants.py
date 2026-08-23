#!/usr/bin/env python3
"""Snapshot, isolate, and restore ``pops/_native/dimN`` leaves across a pip replace.

Installing one ``--dim N`` wheel replaces the whole ``pops`` package.  This helper copies
already-authenticated sibling dimensions out of the way, removes only authenticated leftovers
which pip failed to replace, and merges the siblings back after the new leaf is authenticated.
Dimensions are never merged into one ``.so``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_native_variant_manifest import (
    LEGACY_MANIFEST_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    NativeVariantManifestError,
    load_manifest,
    manifest_schema_version,
    sha256_file,
    validate_manifest_payload,
    write_manifest_atomic,
)


_PER_VARIANT_ABI_FIELDS = frozenset({"dim", "mpi", "mpi_abi"})
_LEGACY_ROW_KEYS = frozenset(
    {
        "dimension",
        "path",
        "sha256",
        "version",
        "abi_key",
        "has_mpi",
        "has_kokkos",
    }
)


def _compatibility_identity(
    row: Mapping[str, Any],
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Return the build/API identity shared by compatible dimension and MPI variants."""
    tokens: dict[str, str] = {}
    for field in row["abi_key"].split(";"):
        name, separator, value = field.partition("=")
        if not separator or not name or not value or name in tokens:
            raise NativeVariantManifestError(
                "native Dim=%d ABI key is not a unique key=value sequence" % row["dimension"]
            )
        tokens[name] = value
    required = {"compiler", "std", "headers", "kokkos", "stdlib", *_PER_VARIANT_ABI_FIELDS}
    missing = sorted(required.difference(tokens))
    if missing:
        raise NativeVariantManifestError(
            "native Dim=%d ABI key lacks fields %s" % (row["dimension"], missing)
        )
    if tokens["dim"] != str(row["dimension"]):
        raise NativeVariantManifestError(
            "native Dim=%d ABI key names Dim=%s" % (row["dimension"], tokens["dim"])
        )
    expected_mpi = "1" if row["has_mpi"] else "0"
    if tokens["mpi"] != expected_mpi:
        raise NativeVariantManifestError(
            "native Dim=%d ABI key disagrees with has_mpi" % row["dimension"]
        )
    expected_kokkos = "1" if row["has_kokkos"] else "0"
    if tokens["kokkos"] != expected_kokkos:
        raise NativeVariantManifestError(
            "native Dim=%d ABI key disagrees with has_kokkos" % row["dimension"]
        )
    common = tuple(
        sorted(
            (name, value) for name, value in tokens.items() if name not in _PER_VARIANT_ABI_FIELDS
        )
    )
    build_fingerprint = row.get("build_fingerprint")
    if (
        not isinstance(build_fingerprint, str)
        or len(build_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in build_fingerprint)
    ):
        raise NativeVariantManifestError(
            "native Dim=%d has no authenticated build_fingerprint" % row["dimension"]
        )
    return row["version"], build_fingerprint, common


def _load_snapshot_manifest(
    manifest: Path,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Authenticate a v2 snapshot or a v1 quarantine inventory.

    Schema v1 has no build fingerprint.  Its exact files may be removed during isolation, but its
    rows are never eligible for restoration into a schema-v2 installation.
    """
    schema_version = manifest_schema_version(manifest)
    if schema_version == MANIFEST_SCHEMA_VERSION:
        return load_manifest(manifest, verify_files=True, verify_hashes=True), True
    if schema_version != LEGACY_MANIFEST_SCHEMA_VERSION:
        raise NativeVariantManifestError(
            "snapshot native variant manifest schema version is unsupported"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeVariantManifestError(
            "legacy native variant manifest is missing or invalid JSON"
        ) from exc
    legacy_rows = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(legacy_rows, list) or not legacy_rows:
        raise NativeVariantManifestError("legacy native variant manifest has no rows")
    upgraded = []
    for row in legacy_rows:
        if not isinstance(row, dict) or set(row) != _LEGACY_ROW_KEYS:
            raise NativeVariantManifestError(
                "legacy native variant manifest row has an unknown schema"
            )
        upgraded.append({**row, "build_fingerprint": "0" * 64})
    rows = validate_manifest_payload(
        {"schema_version": MANIFEST_SCHEMA_VERSION, "variants": upgraded}
    )
    root = manifest.parent.resolve()
    for row in rows:
        extension = root.joinpath(*PurePosixPath(row["path"]).parts)
        if extension.is_symlink() or not extension.is_file():
            raise NativeVariantManifestError(
                "legacy native variant is missing or is a symlink: %s" % extension
            )
        resolved = extension.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise NativeVariantManifestError(
                "legacy native variant escapes its snapshot root: %s" % extension
            ) from exc
        if sha256_file(resolved) != row["sha256"]:
            raise NativeVariantManifestError(
                "legacy native variant bytes disagree with variants.json: Dim=%d" % row["dimension"]
            )
    return rows, False


def _installed_native_root() -> Path | None:
    try:
        import pops
    except ImportError:
        return None
    roots = [Path(item).resolve() / "_native" for item in getattr(pops, "__path__", ())]
    existing = [root for root in roots if root.is_dir()]
    if not existing:
        return None
    return existing[0]


def snapshot(dest: Path) -> int:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    root = _installed_native_root()
    if root is None:
        print("preserve-native-variants: no installed pops/_native to snapshot")
        return 0
    manifest = root / "variants.json"
    if not manifest.is_file():
        print("preserve-native-variants: installed pops/_native has no variants.json")
        return 0
    rows, restorable = _load_snapshot_manifest(manifest)
    shutil.copy2(manifest, dest / "variants.json")
    for row in rows:
        relative = PurePosixPath(row["path"])
        source = root.joinpath(*relative.parts)
        target = dest.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(
        "preserve-native-variants: snapshotted dimensions %s" % [row["dimension"] for row in rows]
    )
    if not restorable:
        print("preserve-native-variants: schema-v1 snapshot is quarantine-only; rebuild required")
    return 0


def isolate(src: Path, expect_dim: int) -> int:
    """Leave only pip's declared Dim=N extension before post-install codesign.

    Some pip installs leave a previous sibling extension on disk while replacing ``variants.json``
    with the new wheel's one-row manifest.  Such a leaf must not be silently tolerated by the
    codesign locator.  Remove it here only when its canonical path and bytes are authenticated by
    the pre-install snapshot; validate every candidate before deleting any of them.
    """
    src = src.resolve()
    root = _installed_native_root()
    if root is None:
        raise NativeVariantManifestError("installed pops/_native is missing after pip")
    installed_manifest = root / "variants.json"
    if not installed_manifest.is_file():
        raise NativeVariantManifestError("installed variants.json is missing after pip")
    installed = load_manifest(
        installed_manifest,
        expected_dimensions=(expect_dim,),
        verify_files=True,
        verify_hashes=False,
    )
    declared_paths = {row["path"] for row in installed}

    snapshot_manifest = src / "variants.json"
    if snapshot_manifest.is_file():
        snapshotted, _snapshot_restorable = _load_snapshot_manifest(snapshot_manifest)
    else:
        snapshotted = ()
    snapshot_by_path = {row["path"]: row for row in snapshotted}

    removable: list[tuple[Path, int]] = []
    for candidate in sorted(root.glob("**/_pops*")):
        if candidate.is_symlink():
            raise NativeVariantManifestError(
                "unmanifested native extension is a symlink: %s" % candidate
            )
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in declared_paths:
            continue
        authenticated = snapshot_by_path.get(relative)
        if authenticated is None:
            raise NativeVariantManifestError(
                "unmanifested native extension is not authenticated by the snapshot: %s" % candidate
            )
        if sha256_file(candidate) != authenticated["sha256"]:
            raise NativeVariantManifestError(
                "unmanifested Dim=%d bytes disagree with the snapshot manifest"
                % authenticated["dimension"]
            )
        removable.append((candidate, authenticated["dimension"]))

    isolated = []
    for candidate, dimension in removable:
        candidate.unlink()
        try:
            candidate.parent.rmdir()
        except OSError:
            pass
        isolated.append(dimension)
    remaining = {
        candidate.relative_to(root).as_posix()
        for candidate in root.glob("**/_pops*")
        if candidate.is_file()
    }
    if remaining != declared_paths:
        raise NativeVariantManifestError(
            "native isolation did not leave exactly the declared Dim=%d extension" % expect_dim
        )
    print("preserve-native-variants: isolated sibling dimensions %s" % isolated)
    return 0


def restore(src: Path, expect_dim: int) -> int:
    src = src.resolve()
    snapshot_manifest = src / "variants.json"
    if not snapshot_manifest.is_file():
        print("preserve-native-variants: no snapshot; nothing to restore")
        return 0
    root = _installed_native_root()
    if root is None:
        raise NativeVariantManifestError("installed pops/_native is missing after pip")
    installed_manifest = root / "variants.json"
    if not installed_manifest.is_file():
        raise NativeVariantManifestError("installed variants.json is missing after pip")
    installed = list(load_manifest(installed_manifest, verify_files=True, verify_hashes=True))
    installed_dims = {row["dimension"] for row in installed}
    if expect_dim not in installed_dims:
        raise NativeVariantManifestError(
            "installed variants.json does not contain the just-built Dim=%d" % expect_dim
        )
    snapshotted, snapshot_restorable = _load_snapshot_manifest(snapshot_manifest)
    target_row = next(row for row in installed if row["dimension"] == expect_dim)
    target_identity = _compatibility_identity(target_row)
    for row in installed:
        if _compatibility_identity(row) != target_identity:
            raise NativeVariantManifestError(
                "installed Dim=%d is incompatible with just-built Dim=%d"
                % (row["dimension"], expect_dim)
            )

    merged = list(installed)
    restored = []
    skipped = []
    restore_candidates = []
    for row in snapshotted:
        if row["dimension"] in installed_dims:
            continue
        if not snapshot_restorable:
            skipped.append(row["dimension"])
            continue
        if _compatibility_identity(row) != target_identity:
            skipped.append(row["dimension"])
            continue
        relative = PurePosixPath(row["path"])
        source = src.joinpath(*relative.parts)
        target = root.joinpath(*relative.parts)
        if source.is_symlink() or not source.is_file():
            raise NativeVariantManifestError(
                "snapshotted Dim=%d leaf is missing: %s" % (row["dimension"], source)
            )
        if sha256_file(source) != row["sha256"]:
            raise NativeVariantManifestError(
                "snapshotted Dim=%d bytes disagree with the snapshot manifest" % row["dimension"]
            )
        if target.is_symlink() or target.exists():
            raise NativeVariantManifestError(
                "restore target for Dim=%d already exists: %s" % (row["dimension"], target)
            )
        restore_candidates.append((row, source, target))

    for row, source, target in restore_candidates:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != row["sha256"]:
            raise NativeVariantManifestError(
                "restored Dim=%d bytes disagree with the snapshot manifest" % row["dimension"]
            )
        merged.append(row)
        restored.append(row["dimension"])
    write_manifest_atomic(installed_manifest, merged)
    load_manifest(installed_manifest, verify_files=True, verify_hashes=True)
    print("preserve-native-variants: restored sibling dimensions %s" % restored)
    if skipped:
        print(
            "preserve-native-variants: skipped incompatible sibling dimensions %s; rebuild required"
            % skipped
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--dest", required=True, type=Path)
    iso = sub.add_parser("isolate")
    iso.add_argument("--src", required=True, type=Path)
    iso.add_argument("--expect-dim", required=True, type=int, choices=(1, 2, 3))
    rest = sub.add_parser("restore")
    rest.add_argument("--src", required=True, type=Path)
    rest.add_argument("--expect-dim", required=True, type=int, choices=(1, 2, 3))
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            return snapshot(args.dest)
        if args.command == "isolate":
            return isolate(args.src, args.expect_dim)
        return restore(args.src, args.expect_dim)
    except (NativeVariantManifestError, OSError, ValueError) as exc:
        print("preserve-native-variants failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
