#!/usr/bin/env python3
"""Key native CI caches by the compiler and its post-install include trees.

This is a cache discriminator, never a substitute for the native build-time
contract. It uses that contract's exact tree fingerprint implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def compiler_search_roots(compiler: str) -> tuple[str, ...]:
    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-v", "-"], input="", text=True,
        capture_output=True, check=True, env={**os.environ, "LC_ALL": "C"},
    )
    begin = "#include <...> search starts here:"
    end = "End of search list."
    if result.stderr.count(begin) != 1 or result.stderr.count(end) != 1:
        raise RuntimeError("compiler did not report one unambiguous include search list")
    paths = result.stderr.split(begin, 1)[1].split(end, 1)[0].splitlines()
    roots = []
    for value in paths:
        value = value.strip()
        if not value:
            continue
        value = value.removesuffix(" (framework directory)")
        if not os.path.isabs(value) or not Path(value).is_dir():
            raise RuntimeError("compiler reported an invalid include search root: " + value)
        root = os.path.realpath(value)
        if root not in roots:
            roots.append(root)
    if not roots:
        raise RuntimeError("compiler include search roots are empty")
    return tuple(roots)


def capture(compiler: str, roots: tuple[str, ...]) -> dict:
    source = Path(__file__).resolve().parents[1] / "python/pops/codegen/_compiler_include_tree.py"
    spec = importlib.util.spec_from_file_location("pops_ci_compiler_include_tree", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("compiler include fingerprint implementation is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.perf_counter()
    binary = os.path.realpath(compiler)
    size, digest = module.file_digest(binary)
    snapshots = [module.snapshot_include_tree(root) for root in roots]
    contract = {
        "schema_version": 1,
        "compiler": {"selector": os.path.abspath(compiler), "path": binary,
                     "size": size, "sha256": digest},
        "include_trees": [{key: row[key] for key in ("root", "sha256", "file_count")}
                          for row in snapshots],
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {"identity": hashlib.sha256(payload).hexdigest(), "contract": contract,
            "bytes_hashed": sum(row["bytes"] for row in snapshots),
            "elapsed_s": time.perf_counter() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise RuntimeError("compiler is unavailable: " + args.compiler)
    receipt = capture(compiler, compiler_search_roots(compiler))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.github_output is not None:
        with args.github_output.open("a") as stream:
            stream.write("identity=" + receipt["identity"] + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
