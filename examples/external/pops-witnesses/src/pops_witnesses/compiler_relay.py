#!/usr/bin/env python3
"""Transparent compiler relay. Retain the bytes read by the real JIT compiler.

The wrapper passes argv, stdout/stderr and return code through unchanged. It is
selected through POPS_CXX; it does not rewrite source, flags or the PoPS core.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = sys.argv[1:]
    compiler = os.environ["POPS_WITNESS_CXX_REAL"]
    sources = [
        Path(x).resolve() for x in args if x.endswith((".cpp", ".cc", ".cxx")) and Path(x).is_file()
    ]
    folder = None
    if sources:
        folder = Path(os.environ["POPS_WITNESS_COMPILER_EVIDENCE"]) / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        inputs = []
        # Generated helper/component headers can hold the arithmetic, not just the TU.
        for i, source in enumerate(sources):
            for path in sorted(source.parent.rglob("*")):
                if path.is_file() and path.suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                    dest = Path("inputs") / str(i) / path.relative_to(source.parent)
                    (folder / dest).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, folder / dest)
                    inputs.append(
                        {"original": str(path), "retained": str(dest), "sha256": sha(path)}
                    )
        record = {
            "argv": [compiler, *args],
            "cwd": os.getcwd(),
            "compiler_sha256": sha(compiler),
            "inputs": inputs,
            "pid": os.getpid(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "compiler_to_binary_graph_proof": False,
        }
    result = subprocess.run([compiler, *args], check=False)
    if folder is not None:
        record["returncode"] = result.returncode
        if "-o" in args:
            output = Path(args[args.index("-o") + 1])
            if result.returncode == 0 and output.is_file():
                shutil.copyfile(output, folder / "compiler-output.so")
                record["output_sha256"] = sha(output)
        if "-MF" in args:
            dependencies = Path(args[args.index("-MF") + 1])
            if dependencies.is_file():
                shutil.copyfile(dependencies, folder / "dependencies.d")
        (folder / "invocation.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
