#!/usr/bin/env python3
"""Fail closed before packaging files from the working tree.

Only Git-tracked Python sources may enter the wheel, and every tracked C++ header must have one
classification in the shared installed-header manifest.  This rejects sync/editor copies before a
directory glob can silently package names such as ``module 2.py`` or ``header 3.hpp``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = PurePosixPath("include/pops_public_headers.manifest")
HEADER_SUFFIXES = {".h", ".hpp"}
PYTHON_SOURCE_SUFFIXES = {".py", ".pyi", ".typed"}
PACKAGING_ROOTS = (PurePosixPath("include/pops"), PurePosixPath("python/pops"))


class PackagingManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeaderManifest:
    public: tuple[PurePosixPath, ...]
    test_only: tuple[PurePosixPath, ...]

    @property
    def all_headers(self) -> frozenset[PurePosixPath]:
        return frozenset((*self.public, *self.test_only))


def read_manifest(root: Path = ROOT) -> HeaderManifest:
    path = root / MANIFEST_REL
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PackagingManifestError(f"cannot read {MANIFEST_REL}: {exc}") from exc

    classified: dict[PurePosixPath, str] = {}
    for line_number, raw in enumerate(rows, 1):
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        try:
            category, value = row.split(maxsplit=1)
        except ValueError as exc:
            raise PackagingManifestError(
                f"{MANIFEST_REL}:{line_number}: expected '<public|test-only> pops/...'") from exc
        relative = PurePosixPath(value)
        if category not in {"public", "test-only"}:
            raise PackagingManifestError(
                f"{MANIFEST_REL}:{line_number}: unknown category {category!r}")
        if relative.is_absolute() or ".." in relative.parts or not relative.parts \
                or relative.parts[0] != "pops" or relative.suffix not in HEADER_SUFFIXES:
            raise PackagingManifestError(
                f"{MANIFEST_REL}:{line_number}: invalid header path {value!r}")
        if relative in classified:
            raise PackagingManifestError(
                f"{MANIFEST_REL}:{line_number}: duplicate header {relative.as_posix()}")
        classified[relative] = category

    public = tuple(sorted(path for path, kind in classified.items() if kind == "public"))
    test_only = tuple(sorted(path for path, kind in classified.items() if kind == "test-only"))
    if not public:
        raise PackagingManifestError(f"{MANIFEST_REL} declares no public headers")
    return HeaderManifest(public=public, test_only=test_only)


def git_tracked_files(root: Path = ROOT) -> frozenset[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *(path.as_posix() for path in PACKAGING_ROOTS)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PackagingManifestError(f"git ls-files failed: {detail or 'unknown error'}")
    return frozenset(
        PurePosixPath(value.decode("utf-8", errors="strict"))
        for value in result.stdout.split(b"\0")
        if value
    )


def physical_packaging_files(root: Path = ROOT) -> frozenset[PurePosixPath]:
    files: set[PurePosixPath] = set()
    for path in (root / "include" / "pops").rglob("*"):
        if path.is_file() and path.suffix in HEADER_SUFFIXES:
            files.add(PurePosixPath(path.relative_to(root).as_posix()))
    for path in (root / "python" / "pops").rglob("*"):
        if path.is_file() and path.suffix in PYTHON_SOURCE_SUFFIXES:
            files.add(PurePosixPath(path.relative_to(root).as_posix()))
    return frozenset(files)


def validate_packaging_inputs(
    root: Path = ROOT,
    *,
    tracked: Iterable[PurePosixPath] | None = None,
    physical: Iterable[PurePosixPath] | None = None,
) -> HeaderManifest:
    manifest = read_manifest(root)
    tracked_set = frozenset(tracked) if tracked is not None else git_tracked_files(root)
    physical_set = frozenset(physical) if physical is not None else physical_packaging_files(root)

    tracked_headers = frozenset(
        path.relative_to("include")
        for path in tracked_set
        if path.parts[:2] == ("include", "pops") and path.suffix in HEADER_SUFFIXES
    )
    missing_rows = sorted(tracked_headers - manifest.all_headers)
    non_tracked_rows = sorted(manifest.all_headers - tracked_headers)
    if missing_rows or non_tracked_rows:
        details = []
        if missing_rows:
            details.append("tracked headers outside the manifest: " + ", ".join(map(str, missing_rows)))
        if non_tracked_rows:
            details.append("manifest rows not backed by tracked headers: "
                           + ", ".join(map(str, non_tracked_rows)))
        raise PackagingManifestError("; ".join(details))

    tracked_sources = frozenset(
        path for path in tracked_set
        if (path.parts[:2] == ("include", "pops") and path.suffix in HEADER_SUFFIXES)
        or (path.parts[:2] == ("python", "pops") and path.suffix in PYTHON_SOURCE_SUFFIXES)
    )
    untracked = sorted(physical_set - tracked_sources)
    absent = sorted(tracked_sources - physical_set)
    if untracked or absent:
        details = []
        if untracked:
            details.append("untracked packaging inputs: " + ", ".join(map(str, untracked)))
        if absent:
            details.append("tracked packaging inputs absent from the working tree: "
                           + ", ".join(map(str, absent)))
        raise PackagingManifestError("; ".join(details))

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = validate_packaging_inputs(args.root.resolve())
    except PackagingManifestError as exc:
        print(f"PACKAGING-MANIFEST: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PACKAGING-MANIFEST: OK: "
        f"{len(manifest.public)} public, {len(manifest.test_only)} test-only headers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
