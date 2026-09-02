#!/usr/bin/env python3
"""Create and verify an immutable ADC-700 extracted-archive tree receipt.

The ROMEO harness builds this receipt only after ``git archive`` extraction, the candidate wheel
build, and the recursive read-only transition.  A receipt is useful only together with the exact
tree it names: verification rescans every regular file, rejects symlinks and writable entries, and
recomputes the canonical tree digest before every measurement run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


SCHEMA = "pops.adc700.archive_receipt.v1"
ROLES = ("baseline", "candidate")


class ArchiveReceiptError(ValueError):
    """The extracted tree or its receipt is incomplete, mutable, or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ArchiveReceiptError("cannot hash %s: %s" % (path, error)) from error
    return digest.hexdigest()


def _require_root(value: Path, *, role: str) -> Path:
    if role not in ROLES:
        raise ArchiveReceiptError("archive role must be baseline or candidate")
    if not value.is_absolute():
        raise ArchiveReceiptError("archive root must be absolute")
    if value.is_symlink():
        raise ArchiveReceiptError("archive root must not be a symlink")
    root = value.resolve()
    if not root.is_dir() or root.name != role:
        raise ArchiveReceiptError(
            "archive root must be the extracted immutable %s tree" % role
        )
    return root


def _is_read_only(path: Path) -> bool:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as error:
        raise ArchiveReceiptError("cannot stat archive entry %s: %s" % (path, error)) from error
    return not bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _entry(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ArchiveReceiptError("archive contains a symlink: %s" % path)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ArchiveReceiptError("archive entry escaped its root: %s" % path) from error
    if not path.is_file():
        raise ArchiveReceiptError("archive entry is not a regular file: %s" % path)
    if not _is_read_only(path):
        raise ArchiveReceiptError("archive entry is writable: %s" % path)
    relative = path.relative_to(root).as_posix()
    if not relative or relative.startswith("../") or "/../" in "/%s" % relative:
        raise ArchiveReceiptError("archive entry has an unsafe relative path: %s" % relative)
    mode = stat.S_IMODE(path.stat().st_mode)
    return {
        "path": relative,
        "mode": "%04o" % mode,
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _scan(root: Path) -> tuple[list[dict[str, Any]], str]:
    if not _is_read_only(root):
        raise ArchiveReceiptError("archive root is writable: %s" % root)
    files: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        if current.is_symlink() or not _is_read_only(current):
            raise ArchiveReceiptError("archive directory is symlinked or writable: %s" % current)
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ArchiveReceiptError("archive directory is not a regular directory: %s" % child)
            if not _is_read_only(child):
                raise ArchiveReceiptError("archive directory is writable: %s" % child)
        for name in sorted(file_names):
            files.append(_entry(root, current / name))
    files.sort(key=lambda item: item["path"])
    material = "".join(
        "%s\t%d\t%s\t%s\n" % (item["mode"], item["size"], item["sha256"], item["path"])
        for item in files
    ).encode("utf-8")
    return files, hashlib.sha256(material).hexdigest()


def _revision(value: str) -> str:
    if not isinstance(value, str) or len(value) != 40 \
            or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ArchiveReceiptError("archive revision must be a 40-character commit SHA")
    return value.lower()


def _receipt_path(value: Path) -> Path:
    if not value.is_absolute():
        raise ArchiveReceiptError("receipt path must be absolute")
    if value.is_symlink() or (value.exists() and not value.is_file()):
        raise ArchiveReceiptError("receipt path must be a regular non-symlink file")
    return value.resolve()


def _script_provenance(root: Path, helper: Path | None) -> dict[str, str]:
    """Authenticate the external receipt helper used to scan ``root``.

    The pinned baseline predates the ADC-700 helper, so the verifier must not require
    ``benchmarks/adc700/archive_receipt.py`` to exist inside every archived tree.  The batch
    harness copies the candidate helper to a separate immutable work-root path and passes that
    path explicitly for both trees.  Keeping the helper outside ``root`` also prevents a tree from
    authenticating a modified helper that it supplies itself.
    """
    script = Path(__file__).resolve() if helper is None else helper.expanduser()
    if not script.is_absolute() or script.is_symlink() or not script.is_file():
        raise ArchiveReceiptError("external receipt helper is missing or not a regular file")
    script = script.resolve()
    try:
        script.relative_to(root)
    except ValueError:
        pass
    else:
        raise ArchiveReceiptError("external receipt helper must be outside the archived tree")
    if not _is_read_only(script):
        raise ArchiveReceiptError("external receipt helper must be immutable")
    return {"path": str(script), "sha256": _sha256(script)}


def build(
    root: Path, *, role: str, revision: str, output: Path, helper: Path | None = None
) -> dict[str, Any]:
    root = _require_root(root, role=role)
    revision = _revision(revision)
    output = _receipt_path(output)
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ArchiveReceiptError("archive receipt must be outside the extracted tree")
    files, tree_sha256 = _scan(root)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "role": role,
        "revision": revision,
        "root_name": root.name,
        "root_path": str(root),
        "tree_sha256": tree_sha256,
        "entry_count": len(files),
        "entries": files,
        "immutable": True,
        "receipt_script": _script_provenance(root, helper),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.chmod(output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return payload


def verify(
    root: Path, *, role: str, revision: str, receipt: Path, helper: Path | None = None
) -> dict[str, Any]:
    root = _require_root(root, role=role)
    revision = _revision(revision)
    receipt = _receipt_path(receipt)
    if not receipt.is_file() or receipt.is_symlink() or not _is_read_only(receipt):
        raise ArchiveReceiptError("archive receipt must be an immutable regular file")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchiveReceiptError("cannot read archive receipt %s: %s" % (receipt, error)) from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "schema_version", "role", "revision", "root_name", "root_path",
        "tree_sha256", "entry_count", "entries", "immutable", "receipt_script",
    }:
        raise ArchiveReceiptError("archive receipt has an invalid schema")
    if payload["schema"] != SCHEMA or payload["schema_version"] != 1:
        raise ArchiveReceiptError("archive receipt schema is unsupported")
    if payload["role"] != role or payload["revision"] != revision \
            or payload["root_name"] != root.name or payload["root_path"] != str(root):
        raise ArchiveReceiptError("archive receipt is not linked to this role/revision/tree")
    if payload["immutable"] is not True:
        raise ArchiveReceiptError("archive receipt does not assert immutability")
    script = _script_provenance(root, helper)
    if payload["receipt_script"] != script:
        raise ArchiveReceiptError("archive receipt script changed or is not archived")
    files, tree_sha256 = _scan(root)
    if payload["entry_count"] != len(files) or payload["entries"] != files \
            or payload["tree_sha256"] != tree_sha256:
        raise ArchiveReceiptError("archive tree changed after receipt creation")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--helper",
        type=Path,
        default=None,
        help="immutable helper path outside the archived root (defaults to this script)",
    )
    parser.add_argument("--build", action="store_true")
    return parser


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep subprocess output bounded while the on-disk receipt retains every entry."""
    return {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "role": payload["role"],
        "revision": payload["revision"],
        "root_name": payload["root_name"],
        "root_path": payload["root_path"],
        "tree_sha256": payload["tree_sha256"],
        "entry_count": payload["entry_count"],
        "immutable": payload["immutable"],
        "receipt_script": payload["receipt_script"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(
            args.root,
            role=args.role,
            revision=args.revision,
            output=args.receipt,
            helper=args.helper,
        ) if args.build else verify(
            args.root,
            role=args.role,
            revision=args.revision,
            receipt=args.receipt,
            helper=args.helper,
        )
        print(json.dumps(_summary(result), sort_keys=True, separators=(",", ":")))
        return 0
    except (ArchiveReceiptError, OSError, TypeError, ValueError) as error:
        print("ADC-700 archive receipt rejected: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
