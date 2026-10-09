"""Assemble an internal mono-Dim CI wheel from an authenticated prebuilt native leaf; no CXX build."""

from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys, sysconfig, tomllib
from pathlib import Path
from packaging.tags import sys_tags
from assemble_native_variant_wheel import _record_bytes, _write_deterministic_wheel
from check_packaging_manifest import read_manifest, PYTHON_SOURCE_SUFFIXES
from write_native_variant_manifest import validate_manifest_payload

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def assemble(
    package: Path, out: Path, dimension: int, *, source: Path = ROOT, backend: str = "serial"
) -> Path:
    package = package.resolve()
    source = source.resolve()
    out = out.resolve()
    if dimension not in (1, 2, 3) or backend not in ("serial", "mpi"):
        raise ValueError("Explicit dimension/backend required")
    manifest_path = package / "pops/_native/variants.json"
    raw = manifest_path.read_bytes()
    rows = validate_manifest_payload(json.loads(raw), expected_dimensions=(dimension,))
    row = rows[0]
    if row["has_mpi"] != (backend == "mpi") or not row["has_kokkos"]:
        raise ValueError("Native backend does not match explicit CI contract")
    version = re.search(
        r"(?m)^\s*VERSION (\d+\.\d+\.\d+)", (source / "CMakeLists.txt").read_text()
    ).group(1)
    if row["version"] != version:
        raise ValueError("Built native and project version differ")
    headers = read_manifest(source)
    signature = sha(
        "".join(
            "%s %s\n%s\n" % (category, path, sha((source / "include" / path).read_bytes()))
            for category, path in sorted(
                ((category, str(path)) for category, path in headers.signed_rows)
            )
        ).encode()
    )
    if "headers=" + signature not in row["abi_key"]:
        raise ValueError("Native SDK does not match exact installed header manifest")
    payload = {}
    names = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--", "python/pops"], text=True
    ).splitlines()
    for name in names:
        if Path(name).suffix in PYTHON_SOURCE_SUFFIXES:
            payload[name.removeprefix("python/")] = (source / name).read_bytes()
    for path in headers.installed_headers:
        payload["pops/include/" + str(path)] = (source / "include" / path).read_bytes()
    payload["pops/include/pops_headers.manifest"] = (
        source / "include/pops_headers.manifest"
    ).read_bytes()
    payload["pops/_native/variants.json"] = raw
    native_member = "pops/_native/" + row["path"]
    binary = package / native_member
    if binary.is_symlink() or sha(binary.read_bytes()) != row["sha256"]:
        raise ValueError("Native leaf identity mismatch")
    payload[native_member] = binary.read_bytes()
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    tag = next(sys_tags())
    info = "pops-" + version + ".dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        "Name: " + project["name"],
        "Version: " + version,
        "Summary: " + project["description"],
        "Requires-Python: " + project["requires-python"],
    ]
    metadata += ["Requires-Dist: " + dependency for dependency in project.get("dependencies", [])]
    for extra, dependencies in project.get("optional-dependencies", {}).items():
        metadata.append("Provides-Extra: " + extra)
        metadata += [
            "Requires-Dist: " + dependency + "; extra == " + repr(extra)
            for dependency in dependencies
        ]
    if (
        "license" in project
        and isinstance(project["license"], dict)
        and "text" in project["license"]
    ):
        metadata.append("License: " + project["license"]["text"])
    metadata += ["Classifier: " + value for value in project.get("classifiers", [])]
    payload[info + "/METADATA"] = ("\n".join(metadata) + "\n\n").encode()
    payload[info + "/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: pops-ci-prebuilt-wheel\nRoot-Is-Purelib: false\nTag: "
        + str(tag)
        + "\n"
    ).encode()
    payload[info + "/RECORD"] = _record_bytes(payload, info + "/RECORD")
    out.mkdir(parents=True, exist_ok=True)
    wheel = out / ("pops-" + version + "-" + str(tag) + ".whl")
    if wheel.exists():
        raise FileExistsError("Refuse an existing wheel output")
    _write_deterministic_wheel(wheel, payload)
    command = (
        ["readelf", "-d", str(binary)]
        if sys.platform.startswith("linux")
        else ["otool", "-L", str(binary)]
        if sys.platform == "darwin"
        else None
    )
    deps = None
    if command:
        run = subprocess.run(command, text=True, capture_output=True)
        deps = {"argv": command, "exit": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
        if run.returncode:
            raise ValueError("Cannot retain native dependency/RPATH provenance")
        if sys.platform == "darwin":
            detail = subprocess.run(["otool", "-l", str(binary)], text=True, capture_output=True)
            if detail.returncode:
                raise ValueError("Cannot retain MachO load-command/RPATH provenance")
            deps["load_commands"] = {
                "argv": detail.args,
                "exit": detail.returncode,
                "stdout": detail.stdout,
                "stderr": detail.stderr,
            }
    receipt = {
        "scope": "Assembly of actual prebuilt native bytes plus tracked Python/header manifest; no CXX compilation or auditwheel/fixups",
        "dimension": dimension,
        "backend": backend,
        "wheel": str(wheel),
        "wheel_sha256": sha(wheel.read_bytes()),
        "native_sha256": row["sha256"],
        "SDK": signature,
        "members": len(payload),
        "native_dependency_RPATH_provenance": deps,
        "relocation_or_auditwheel_performed": False,
    }
    (out / "assembly-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return wheel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--backend", choices=("serial", "mpi"), default="serial")
    a = p.parse_args()
    print(assemble(a.package, a.output_dir, a.dimension, source=a.source_root, backend=a.backend))


if __name__ == "__main__":
    main()
