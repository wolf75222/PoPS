"""Exact file authority for the implicit headers of the native build compiler."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from pops.codegen._compiler_include_tree import file_digest, snapshot_include_tree


def _digest(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True, slots=True)
class CapturedCompilerIncludes:
    compiler_selector: str
    compiler_path: str
    compiler_size: int
    compiler_sha256: str
    trees: tuple[tuple[str, str, int], ...]

    def verify(self) -> frozenset[str]:
        if (not os.path.isfile(self.compiler_path)
                or os.path.realpath(self.compiler_selector) != self.compiler_path
                or os.path.realpath(self.compiler_path) != self.compiler_path
                or file_digest(self.compiler_path) != (self.compiler_size, self.compiler_sha256)):
            raise RuntimeError("native build compiler changed or is unavailable")
        files = set()
        for root, digest, count in self.trees:
            try:
                snapshot = snapshot_include_tree(root)
            except (OSError, ValueError) as exc:
                raise RuntimeError("native compiler include tree changed or is unavailable") from exc
            if snapshot["sha256"] != digest or snapshot["file_count"] != count:
                raise RuntimeError("native compiler include tree changed after its build-time capture: " + root)
            files.update(snapshot["files"])
        return frozenset(files)

    def to_data(self) -> dict[str, Any]:
        self.verify()
        return {"contract": "pops.native-compiler-includes@1",
                "compiler": {"selector": self.compiler_selector,
                             "path": self.compiler_path, "size": self.compiler_size,
                             "sha256": self.compiler_sha256},
                "include_trees": [{"root": root, "sha256": digest, "file_count": count}
                                  for root, digest, count in self.trees]}


def capture_compiler_includes() -> CapturedCompilerIncludes | None:
    from pops.codegen.toolchain import _pops_module

    module = _pops_module()
    raw = getattr(module, "__compiler_include_contract__", None)
    # Older modules grant no additional authority. In particular their CUDA
    # implicit dependencies continue to fail the ordinary closed-depfile check.
    if raw is None:
        return None
    if type(raw) is not str:
        raise TypeError("native compiler include contract must be exact JSON text")
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ValueError("native compiler include contract is malformed") from exc
    if json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")) != raw:
        raise ValueError("native compiler include contract must use canonical JSON")
    if (type(data) is not dict or set(data) != {"schema_version", "compiler", "include_trees"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1):
        raise ValueError("native compiler include contract has an unsupported schema")
    compiler = data["compiler"]
    if (type(compiler) is not dict or set(compiler) != {"selector", "path", "size", "sha256"}
            or type(compiler["selector"]) is not str or not os.path.isabs(compiler["selector"])
            or type(compiler["path"]) is not str or not os.path.isabs(compiler["path"])
            or type(compiler["size"]) is not int or compiler["size"] < 0
            or not _digest(compiler["sha256"])):
        raise ValueError("native compiler identity is malformed")
    baked = getattr(module, "__cxx_compiler__", None)
    if (type(baked) is not str or os.path.abspath(baked) != compiler["selector"]
            or os.path.realpath(baked) != compiler["path"]):
        raise ValueError("native compiler include contract differs from its native compiler")
    trees = data["include_trees"]
    if type(trees) is not list:
        raise ValueError("native compiler include trees must be an exact sequence")
    records = []
    for tree in trees:
        if (type(tree) is not dict or set(tree) != {"root", "sha256", "file_count"}
                or type(tree["root"]) is not str or not os.path.isabs(tree["root"])
                or not _digest(tree["sha256"])
                or type(tree["file_count"]) is not int or tree["file_count"] < 0):
            raise ValueError("native compiler include tree descriptor is malformed")
        records.append((tree["root"], tree["sha256"], tree["file_count"]))
    if len({row[0] for row in records}) != len(records):
        raise ValueError("native compiler include roots must be unique")
    capture = CapturedCompilerIncludes(compiler["selector"], compiler["path"], compiler["size"],
                                       compiler["sha256"], tuple(records))
    capture.verify()
    return capture
