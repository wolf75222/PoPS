#!/usr/bin/env python3
"""Snapshot and restore sibling ``pops/_native/dimN`` leaves across a pip replace.

Installing one ``--dim N`` wheel replaces the whole ``pops`` package.  This helper copies
already-authenticated sibling dimensions out of the way and merges them back so
``select_native_dimension`` can still pick Dim=2 after a Dim=3 install.  Dimensions are never
merged into one ``.so``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_native_variant_manifest import (
    NativeVariantManifestError,
    load_manifest,
    sha256_file,
    write_manifest_atomic,
)


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
    rows = load_manifest(manifest, verify_files=True, verify_hashes=True)
    shutil.copy2(manifest, dest / "variants.json")
    for row in rows:
        relative = PurePosixPath(row["path"])
        source = root.joinpath(*relative.parts)
        target = dest.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print("preserve-native-variants: snapshotted dimensions %s" %
          [row["dimension"] for row in rows])
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
    snapshotted = load_manifest(snapshot_manifest, verify_files=True, verify_hashes=True)
    merged = list(installed)
    restored = []
    for row in snapshotted:
        if row["dimension"] in installed_dims:
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
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--dest", required=True, type=Path)
    rest = sub.add_parser("restore")
    rest.add_argument("--src", required=True, type=Path)
    rest.add_argument("--expect-dim", required=True, type=int, choices=(1, 2, 3))
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            return snapshot(args.dest)
        return restore(args.src, args.expect_dim)
    except (NativeVariantManifestError, OSError, ValueError) as exc:
        print("preserve-native-variants failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
