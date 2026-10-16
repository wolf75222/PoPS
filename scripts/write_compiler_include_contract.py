#!/usr/bin/env python3
"""Bake CMake's declared implicit include trees into the native module."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", action="append", default=[])
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "python/pops/codegen/_compiler_include_tree.py"
    spec = importlib.util.spec_from_file_location("pops_compiler_include_tree", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("compiler include fingerprint module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.perf_counter()
    compiler = os.path.realpath(args.compiler)
    size, digest = module.file_digest(compiler)
    roots = tuple(dict.fromkeys(os.path.realpath(root) for root in args.root))
    snapshots = [module.snapshot_include_tree(root) for root in roots]
    contract = {"schema_version": 1,
                "compiler": {"selector": os.path.abspath(args.compiler), "path": compiler,
                             "size": size, "sha256": digest},
                "include_trees": [{key: snapshot[key]
                                   for key in ("root", "sha256", "file_count")}
                                  for snapshot in snapshots]}
    payload = json.dumps(contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    assert ')POPS_TC"' not in payload
    text = ('// Generated from the configured compiler; do not edit.\n#pragma once\n'
            'namespace pops::build {\n'
            'inline constexpr char compiler_include_contract_json[] = R"POPS_TC('
            + payload + ')POPS_TC";\n}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)
    print(json.dumps({"compiler_include_roots": len(roots),
                      "files": sum(row["file_count"] for row in snapshots),
                      "bytes_hashed": sum(row["bytes"] for row in snapshots),
                      "elapsed_s": time.perf_counter() - started}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
