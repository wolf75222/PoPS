"""Build/runtime shared, dependency-free fingerprint of compiler include trees.

The build declares the roots. Observed generated depfiles never grant authority.
Symlinks contribute their exact target and content; only enumerated regular files
are returned, never a directory prefix. A visited-directory set bounds aliases
and cycles without recursively walking Python frames.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def file_digest(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(data)
            digest.update(data)
    return size, digest.hexdigest()


def snapshot_include_tree(root: str) -> dict:
    if type(root) is not str or not os.path.isabs(root) or os.path.realpath(root) != root:
        raise ValueError("compiler include root must be an absolute canonical path")
    if not Path(root).is_dir():
        raise ValueError("compiler include root is unavailable: " + root)
    records = []
    files = set()
    visited = set()
    stack = [(Path(root), "")]
    total_bytes = 0
    while stack:
        path, relative = stack.pop()
        canonical = str(path.resolve(strict=True))
        if path.is_dir():
            records.append(("directory", relative, canonical))
            if canonical in visited:
                continue
            visited.add(canonical)
            stack.extend((child, child.name if not relative else relative + "/" + child.name)
                         for child in sorted(path.iterdir(), reverse=True))
        elif path.is_file():
            size, digest = file_digest(canonical)
            records.append(("file", relative, canonical, size, digest))
            files.add(canonical)
            total_bytes += size
        else:
            raise ValueError("compiler include tree contains a non-regular entry: " + str(path))
    material = json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode()
    return {"root": root, "sha256": hashlib.sha256(material).hexdigest(),
            "file_count": len(files), "files": frozenset(files), "bytes": total_bytes}
