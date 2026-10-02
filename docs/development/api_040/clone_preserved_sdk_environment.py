"""Copy existing dependencies offline, preserving a measured SDK installation.

The destination is an unqualified build environment until a repository build
and its installed-package identity checks have succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def package_at(prefix):
    text = subprocess.check_output(
        [str(prefix / "bin/python"), "-I", "-c",
         "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    )
    package = Path(text.strip()).resolve() / "pops"
    if not package.is_relative_to(prefix) or not package.is_dir():
        raise RuntimeError("the measured PoPS installation is outside its prefix")
    return package


def installation(package, expected):
    actual = {}
    for name, sha in expected.items():
        relative = name.removeprefix("python/pops/")
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or digest(path) != sha:
            raise RuntimeError(f"measured installed source differs: {name}")
        actual[name] = sha
    native = sorted((package / "_native").glob("dim*/*.so"))
    if not native:
        raise RuntimeError("the measured installation has no native extension")
    for path in native:
        actual[str(path.relative_to(package))] = digest(path)
    return actual


def dependencies(prefix):
    rows = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        meta = json.loads(path.read_text())
        rows.append({key: meta.get(key) for key in
                     ("name", "version", "build", "build_number", "subdir")})
    if not rows:
        raise RuntimeError("the dependency prefix has no Conda package records")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--conda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "PYTHONPATH" in os.environ:
        parser.error("run with env -u PYTHONPATH")
    source, target, output = args.source.resolve(), args.target.resolve(), args.output.resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        parser.error("source and target prefixes must be separate")
    if target.exists() or output.exists():
        parser.error("target and output must be fresh")
    expected = json.loads(args.source_receipt.read_text())
    if not expected or not all(type(k) is str and type(v) is str and len(v) == 64
                               for k, v in expected.items()):
        parser.error("source receipt must contain a closed source hash inventory")
    package = package_at(source)
    before = installation(package, expected)
    source_dependencies = dependencies(source)
    output.mkdir(parents=True)
    command = [str(args.conda.resolve()), "create", "--yes", "--offline", "--copy",
               "--prefix", str(target), "--clone", str(source)]
    write(output / "command.json", {"command": command,
          "driver": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
          "source_receipt": {"path": str(args.source_receipt.resolve()),
                             "sha256": digest(args.source_receipt)}})
    write(output / "source-installation-before.json", before)
    write(output / "source-dependencies.json", source_dependencies)
    with (output / "clone.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    after = installation(package, expected)
    target_dependencies = dependencies(target) if result.returncode == 0 else None
    write(output / "source-installation-after.json", after)
    receipt = dict(schema="root.preserved-sdk-environment-clone@2",
                   source_prefix=str(source), target_prefix=str(target),
                   clone_returncode=result.returncode,
                   source_runtime_untouched=before == after,
                   verified_source_files=len(expected),
                   conda_packages_equal=source_dependencies == target_dependencies,
                   conda_package_count=len(source_dependencies),
                   destination_native_scientific_acceptance=False,
                   destination_sdk_built=False,
                   log_sha256=digest(output / "clone.log"))
    write(output / "result.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return result.returncode or (0 if receipt["source_runtime_untouched"] and
                                receipt["conda_packages_equal"] else 3)


if __name__ == "__main__":
    sys.exit(main())
