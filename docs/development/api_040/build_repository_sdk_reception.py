"""Record an actual repository-script build; scientific acceptance is separate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(checkout, *args):
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def production(checkout):
    paths = git(checkout, "ls-files", "include", "src", "python", "cmake", "scripts",
                "CMakeLists.txt", "CMakePresets.json", "pyproject.toml", "packaging_manifest.json")
    return {name: digest(checkout / name) for name in paths.splitlines()}


def old_installation(package, expected, copy_receipt):
    prefix = Path(copy_receipt["source_prefix"]).resolve()
    if not package.resolve().is_relative_to(prefix):
        raise RuntimeError("preserved package differs from pinned source environment")
    for item in copy_receipt["bytes_verified"]:
        path = (prefix / item["relative_path"]).resolve()
        if not path.is_relative_to(prefix) or digest(path) != item["sha256"]:
            raise RuntimeError("preserved native/Python/header artifact differs from pinned receipt")
    actual = {name: digest(package / name) for name in expected}
    if actual != expected:
        raise RuntimeError("preserved SDK2e4 Python/header installation differs from closed receipt")
    actual.update({str(path.relative_to(package)): digest(path)
                   for path in sorted((package / "_native").glob("dim*/*.so"))})
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dimension", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preserved-package", required=True, type=Path)
    parser.add_argument("--preserved-source-receipt", required=True, type=Path)
    parser.add_argument("--preserved-copy-receipt", required=True, type=Path)
    args = parser.parse_args()
    checkout, environment, output = (p.resolve() for p in
                                     (args.checkout, args.environment, args.output))
    if "PYTHONPATH" in os.environ:
        parser.error("run with env -u PYTHONPATH")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be fresh")
    if git(checkout, "rev-parse", "HEAD") != args.source_commit:
        parser.error("checkout HEAD differs from the requested source freeze")
    if git(checkout, "status", "--porcelain", "--untracked-files=no"):
        parser.error("tracked source must be clean before compilation")
    if not (environment / "bin/python").is_file():
        parser.error("environment is not prepared")
    output.mkdir(parents=True, exist_ok=True)
    source = production(checkout)
    write(output / "production-source-before.json", source)
    preserved_expected = json.loads(args.preserved_source_receipt.read_text())
    preserved_copy = json.loads(args.preserved_copy_receipt.read_text())
    preserved = old_installation(args.preserved_package, preserved_expected, preserved_copy)
    write(output / "preserved-sdk2e4-before.json", preserved)
    env = dict(os.environ)
    env.update(POPS_ENV_NAME=environment.name, CONDA_PREFIX=str(environment),
               Kokkos_ROOT=str(environment), POPS_KOKKOS_ROOT=str(environment),
               CMAKE_PREFIX_PATH=str(environment),
               POPS_INCLUDE=str(environment / "lib/python3.12/site-packages/pops/include"),
               POPS_NATIVE_DIM=str(args.dimension), PYTHONNOUSERSITE="1",
               POPS_HEAVY_MODULE_TU_POOL="1", CMAKE_BUILD_PARALLEL_LEVEL="1",
               OMP_NUM_THREADS="1", POPS_THREADS="1", OMP_PROC_BIND="false", FI_PROVIDER="tcp")
    env["PATH"] = str(environment.parent.parent / "bin") + os.pathsep + str(environment / "bin") + os.pathsep + env["PATH"]
    command = ["bash", "scripts/build_python.sh", "--dim", str(args.dimension), "--mpi",
               "--wheel-dir", str(output / "retained-wheel"), "--",
               "-C", "cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    write(output / "command.json", {"command": command, "source_commit": args.source_commit,
                                    "environment": {k: env[k] for k in
                                      ("POPS_ENV_NAME", "CONDA_PREFIX", "Kokkos_ROOT", "POPS_KOKKOS_ROOT",
                                       "CMAKE_PREFIX_PATH", "POPS_INCLUDE", "POPS_NATIVE_DIM",
                                       "POPS_HEAVY_MODULE_TU_POOL", "CMAKE_BUILD_PARALLEL_LEVEL",
                                       "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "POPS_THREADS",
                                       "OMP_PROC_BIND", "FI_PROVIDER")},
                                    "driver": record(__file__)})
    print("actual build starts", args.dimension, args.source_commit, flush=True)
    with (output / "build.log").open("w") as log:
        build = subprocess.run(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT)
    after = production(checkout)
    write(output / "production-source-after.json", after)
    old_after = old_installation(args.preserved_package, preserved_expected, preserved_copy)
    write(output / "preserved-sdk2e4-after.json", old_after)
    result = {"schema": "root.api040.repository-native-build@3", "dimension": args.dimension,
              "source_commit": args.source_commit, "build_returncode": build.returncode,
              "source_file_count": len(source), "production_bytes_unchanged": source == after,
              "source_head_unchanged": git(checkout, "rev-parse", "HEAD") == args.source_commit,
              "preserved_sdk2e4_unchanged": preserved == old_after,
              "environment": str(environment), "build_log": record(output / "build.log"),
              "command": record(output / "command.json"),
              "preserved_copy_receipt": record(args.preserved_copy_receipt),
              "native_scientific_acceptance": False,
              "cpp_to_dso_cryptographic_graph_proof": False}
    if build.returncode == 0:
        identity_dir = output / "installed-identity"
        with (output / "installed-identity.log").open("w") as log:
            identity = subprocess.run([str(environment / "bin/python"),
                                      "docs/development/api_040/run_installed_checks.py",
                                      "--output", str(identity_dir), "--identity-only"], cwd=checkout,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        result["installed_identity_returncode"] = identity.returncode
        if identity.returncode == 0:
            proof = json.loads((identity_dir / "identity.json").read_text())
            native = Path(proof["native_file"])
            if not native.resolve().is_relative_to(environment):
                raise RuntimeError("native artifact is outside the requested environment")
            result.update(installed_identity=record(identity_dir / "identity.json"),
                          verified_installed_source_files=proof["verified_source_files"],
                          native=record(native), abi_key=proof["abi_key"])
            wheels = tuple((output / "retained-wheel").glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("actual build did not retain exactly one wheel")
            result["wheel"] = record(wheels[0])
            member = "pops/_native/dim%d/%s" % (args.dimension, native.name)
            with zipfile.ZipFile(wheels[0]) as archive:
                result["wheel_native_member"] = member
                result["wheel_native_member_sha256"] = hashlib.sha256(archive.read(member)).hexdigest()
            result["wheel_and_installed_native_bytes_equal"] = result["wheel_native_member_sha256"] == digest(native)
            result["sdk_manifest_file"] = record(environment / "lib/python3.12/site-packages/pops/include/pops_headers.manifest")
        caches = tuple((checkout / "build").glob("cp3*-dim%d" % args.dimension))
        if len(caches) == 1:
            for name in ("compile_commands.json", "CMakeCache.txt", "build.ninja"):
                path = caches[0] / name
                if path.is_file():
                    shutil.copyfile(path, output / name)
                    result[name] = record(output / name)
            if (caches[0] / "build.ninja").is_file():
                with (output / "ninja-commands.txt").open("w") as log:
                    subprocess.run([str(environment / "bin/ninja"), "-t", "commands"], cwd=caches[0],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                result["ninja_commands"] = record(output / "ninja-commands.txt")
    result["closed_at"] = datetime.now(timezone.utc).isoformat()
    write(output / "build-reception.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not all((result["production_bytes_unchanged"], result["source_head_unchanged"],
                result["preserved_sdk2e4_unchanged"])):
        return 2
    if build.returncode:
        return build.returncode
    if result.get("installed_identity_returncode") != 0:
        return result.get("installed_identity_returncode", 3)
    if not result.get("wheel_and_installed_native_bytes_equal", False):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
