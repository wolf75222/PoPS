#!/usr/bin/env python3
"""Assemble exactly three repaired mono-Dim PoPS wheels into one deterministic fat wheel."""
from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
import csv
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    manifest_payload,
    sha256_file,
    validate_manifest_payload,
)


FAT_WHEEL_DIMENSIONS = (1, 2, 3)
ASSEMBLY_SCHEMA_VERSION = 1
_MANIFEST_MEMBER = "pops/_native/variants.json"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ABI_DIMENSION = re.compile(r"^(?P<common>.+);dim=(?P<dimension>[123])$")


class FatWheelAssemblyError(RuntimeError):
    """The three inputs cannot authenticate one homogeneous fat wheel."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _safe_member(name: str) -> None:
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or str(relative) != name \
            or any(part in {"", ".", ".."} for part in relative.parts):
        raise FatWheelAssemblyError("wheel member path is unsafe or non-canonical: %r" % name)
    if ".data" in relative.parts:
        raise FatWheelAssemblyError("wheel .data installation schemes are unsupported: %s" % name)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _native_member(name: str) -> bool:
    filename = PurePosixPath(name).name
    return any(filename == "_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _read_record(archive: zipfile.ZipFile, record_member: str) -> None:
    """Authenticate the input wheel RECORD before reusing any repaired bytes."""
    try:
        text = archive.read(record_member).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise FatWheelAssemblyError("input wheel RECORD is unreadable") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or any(len(row) != 3 for row in rows):
        raise FatWheelAssemblyError("input wheel RECORD is malformed")
    by_name: dict[str, tuple[str, str]] = {}
    for name, digest, size in rows:
        _safe_member(name)
        if name in by_name:
            raise FatWheelAssemblyError("input wheel RECORD repeats %s" % name)
        by_name[name] = (digest, size)
    files = {
        info.filename for info in archive.infolist()
        if not info.is_dir()
    }
    if set(by_name) != files:
        raise FatWheelAssemblyError("input wheel RECORD does not cover its exact file set")
    for name in sorted(files):
        digest, size = by_name[name]
        if name == record_member:
            if digest or size:
                raise FatWheelAssemblyError("input wheel RECORD must not hash itself")
            continue
        payload = archive.read(name)
        if digest != _record_digest(payload) or size != str(len(payload)):
            raise FatWheelAssemblyError("input wheel RECORD digest drifted: %s" % name)


def _manifest_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        json.dumps(manifest_payload(rows), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _record_bytes(payloads: Mapping[str, bytes], record_member: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(payloads):
        payload = payloads[name]
        writer.writerow((name, _record_digest(payload), str(len(payload))))
    writer.writerow((record_member, "", ""))
    return stream.getvalue().encode("utf-8")


def _write_deterministic_wheel(path: Path, payloads: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _normalized_abi(row: Mapping[str, Any]) -> str:
    match = _ABI_DIMENSION.fullmatch(row["abi_key"])
    if match is None or int(match.group("dimension")) != row["dimension"]:
        raise FatWheelAssemblyError(
            "Dim=%d ABI key does not end in its exact dimension token" % row["dimension"]
        )
    return match.group("common")


def _archive_payload(
    wheel: Path,
    *,
    expected_dimension: int,
) -> tuple[dict[str, bytes], dict[str, Any], str]:
    """Read one repaired mono-variant wheel into authenticated immutable bytes."""
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise FatWheelAssemblyError("Dim=%d input is not one readable wheel" % expected_dimension)
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise FatWheelAssemblyError("input wheel contains duplicate members")
            for info in infos:
                _safe_member(info.filename.rstrip("/"))
                if _is_symlink(info):
                    raise FatWheelAssemblyError(
                        "input wheel contains a symlink: %s" % info.filename
                    )
            files = {name: archive.read(name) for name in names if not name.endswith("/")}
            manifests = [name for name in files if name == _MANIFEST_MEMBER]
            records = [name for name in files if name.endswith(".dist-info/RECORD")]
            if manifests != [_MANIFEST_MEMBER] or len(records) != 1:
                raise FatWheelAssemblyError(
                    "input wheel lacks one manifest or one RECORD"
                )
            _read_record(archive, records[0])
    except (OSError, zipfile.BadZipFile) as exc:
        raise FatWheelAssemblyError("input wheel is unreadable: %s" % wheel) from exc
    try:
        manifest = json.loads(files[_MANIFEST_MEMBER])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FatWheelAssemblyError("input native manifest is invalid JSON") from exc
    try:
        rows = validate_manifest_payload(
            manifest, expected_dimensions=(expected_dimension,)
        )
    except NativeVariantManifestError as exc:
        raise FatWheelAssemblyError(str(exc)) from exc
    row = rows[0]
    member = "pops/_native/" + row["path"]
    native_members = {name for name in files if _native_member(name)}
    if native_members != {member}:
        raise FatWheelAssemblyError(
            "Dim=%d wheel native members do not equal its manifest leaf"
            % expected_dimension
        )
    # The mono-Dim manifest is emitted at link time, before cibuildwheel's repair step.  Repairing
    # and ad-hoc signing a Mach-O intentionally changes its bytes, so its source digest is only a
    # pre-repair identity.  The final digest is recomputed below from the RECORD-authenticated wheel
    # member, and the complete row is then reproved by loading those exact final bytes in a fresh
    # process.  Treating the pre-repair digest as final would make a legitimate repair impossible.
    return files, row, records[0]


def _checked_signature(extension: Path) -> None:
    codesign = shutil.which("codesign")
    if codesign is None:
        raise FatWheelAssemblyError("ad-hoc signature policy requires codesign on PATH")
    verification = subprocess.run(
        (codesign, "--verify", "--strict", "--verbose=2", str(extension)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    inspection = subprocess.run(
        (codesign, "--display", "--verbose=4", str(extension)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if verification.returncode or inspection.returncode \
            or "Signature=adhoc" not in inspection.stdout:
        raise FatWheelAssemblyError(
            "repaired native leaf lacks its final ad-hoc signature: %s" % extension
        )


def _subprocess_native_proof(
    python_executable: Path,
    extracted: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one final leaf in a fresh process and return its authenticated row."""
    extension = extracted / "pops" / "_native" / row["path"]
    manifest = extracted / _MANIFEST_MEMBER
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        (
            str(python_executable),
            str(SCRIPTS / "write_native_variant_manifest.py"),
            "--extension", str(extension),
            "--manifest", str(manifest),
            "--dimension", str(row["dimension"]),
            "--version", row["version"],
        ),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise FatWheelAssemblyError(
            "fresh Dim=%d native proof failed: %s"
            % (row["dimension"], completed.stdout[-2000:])
        )
    try:
        evidence = json.loads(completed.stdout)
        rows = evidence["variants"]
        actual = next(item for item in rows if item["dimension"] == row["dimension"])
    except (json.JSONDecodeError, KeyError, StopIteration, TypeError) as exc:
        raise FatWheelAssemblyError(
            "fresh Dim=%d native proof was malformed" % row["dimension"]
        ) from exc
    return actual


def _prove_final_wheel(
    wheel: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    python_executable: Path,
    signature_policy: str,
    proof_runner: Callable[[Path, Path, Mapping[str, Any]], dict[str, Any]],
) -> None:
    with tempfile.TemporaryDirectory(prefix="pops-fat-wheel-proof-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                destination = extracted.joinpath(*PurePosixPath(info.filename).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info.filename))
        for row in rows:
            extension = extracted / "pops" / "_native" / row["path"]
            if signature_policy == "adhoc":
                _checked_signature(extension)
            actual = proof_runner(python_executable, extracted, row)
            if actual != dict(row):
                raise FatWheelAssemblyError(
                    "fresh Dim=%d proof disagrees with the final manifest"
                    % row["dimension"]
                )


def assemble_native_variant_wheel(
    variant_wheels: Mapping[int, Path],
    *,
    output: Path,
    python_executable: Path,
    expect_mpi: bool,
    expect_kokkos: bool,
    signature_policy: str,
    proof_runner: Callable[
        [Path, Path, Mapping[str, Any]], dict[str, Any]
    ] = _subprocess_native_proof,
) -> dict[str, Any]:
    """Assemble and atomically publish one exact Dim=1/2/3 wheel."""
    if set(variant_wheels) != set(FAT_WHEEL_DIMENSIONS):
        raise FatWheelAssemblyError("fat wheel inputs must be exactly dimensions 1, 2, and 3")
    if signature_policy not in {"adhoc", "none"}:
        raise FatWheelAssemblyError("signature policy must be explicitly adhoc or none")
    interpreter = python_executable.resolve()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise FatWheelAssemblyError("proof interpreter is not executable: %s" % interpreter)
    destination = output.resolve()
    if destination.exists() or destination.suffix != ".whl":
        raise FatWheelAssemblyError("output must name one absent .whl file")
    destination.parent.mkdir(parents=True, exist_ok=True)

    input_names = {Path(path).name for path in variant_wheels.values()}
    if len(input_names) != 1 or input_names != {destination.name}:
        raise FatWheelAssemblyError(
            "all inputs and output must carry one identical wheel filename"
        )
    source_payloads: dict[int, dict[str, bytes]] = {}
    rows = []
    record_members = set()
    for dimension in FAT_WHEEL_DIMENSIONS:
        payloads, row, record_member = _archive_payload(
            Path(variant_wheels[dimension]).resolve(), expected_dimension=dimension
        )
        if row["has_mpi"] is not expect_mpi or row["has_kokkos"] is not expect_kokkos:
            raise FatWheelAssemblyError(
                "Dim=%d backend facts disagree with the explicit release backend" % dimension
            )
        source_payloads[dimension] = payloads
        rows.append(dict(row))
        record_members.add(record_member)
    if len(record_members) != 1:
        raise FatWheelAssemblyError("input wheels disagree on their RECORD location")
    record_member = next(iter(record_members))

    native_members = {
        "pops/_native/" + row["path"] for row in rows
    }
    excluded = {_MANIFEST_MEMBER, record_member, *native_members}
    common_names = [set(payloads) - excluded for payloads in source_payloads.values()]
    if any(names != common_names[0] for names in common_names[1:]):
        raise FatWheelAssemblyError("mono-variant wheels disagree on their common file set")
    common_payloads = {
        name: source_payloads[1][name] for name in common_names[0]
    }
    for dimension in (2, 3):
        for name, payload in common_payloads.items():
            if source_payloads[dimension][name] != payload:
                raise FatWheelAssemblyError(
                    "mono-variant wheels differ outside native leaves: %s" % name
                )
    versions = {row["version"] for row in rows}
    normalized_abis = {_normalized_abi(row) for row in rows}
    build_fingerprints = {row["build_fingerprint"] for row in rows}
    suffixes = {PurePosixPath(row["path"]).name for row in rows}
    if len(versions) != 1 or len(normalized_abis) != 1 or len(build_fingerprints) != 1 \
            or len(suffixes) != 1:
        raise FatWheelAssemblyError(
            "mono-variant wheels do not share one version, build fingerprint, toolchain ABI "
            "and extension suffix"
        )

    final_payloads = dict(common_payloads)
    final_rows = []
    for dimension, row in zip(FAT_WHEEL_DIMENSIONS, rows, strict=True):
        member = "pops/_native/" + row["path"]
        payload = source_payloads[dimension][member]
        final_payloads[member] = payload
        final = dict(row)
        final["sha256"] = _sha256_bytes(payload)
        final_rows.append(final)
    final_payloads[_MANIFEST_MEMBER] = _manifest_bytes(final_rows)
    final_payloads[record_member] = _record_bytes(final_payloads, record_member)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pops-fat-", suffix=".whl", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_wheel(temporary, final_payloads)
        _prove_final_wheel(
            temporary,
            rows=final_rows,
            python_executable=interpreter,
            signature_policy=signature_policy,
            proof_runner=proof_runner,
        )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "output": str(destination),
        "sha256": sha256_file(destination),
        "dimensions": list(FAT_WHEEL_DIMENSIONS),
        "signature_policy": signature_policy,
        "backend": {"has_mpi": expect_mpi, "has_kokkos": expect_kokkos},
        "source_wheels": [
            {
                "dimension": dimension,
                "path": str(Path(variant_wheels[dimension]).resolve()),
                "sha256": sha256_file(Path(variant_wheels[dimension]).resolve()),
            }
            for dimension in FAT_WHEEL_DIMENSIONS
        ],
        "variants": final_rows,
    }


def _variant_argument(value: str) -> tuple[int, Path]:
    try:
        raw_dimension, raw_path = value.split("=", 1)
        dimension = int(raw_dimension)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("--variant must be N=/path/to/wheel") from exc
    if dimension not in FAT_WHEEL_DIMENSIONS or not raw_path:
        raise argparse.ArgumentTypeError("--variant dimension must be exactly 1, 2, or 3")
    return dimension, Path(raw_path)


def _on_off(value: str) -> bool:
    return value == "on"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, type=_variant_argument)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--expect-mpi", required=True, choices=("on", "off"))
    parser.add_argument("--expect-kokkos", required=True, choices=("on", "off"))
    parser.add_argument("--signature-policy", required=True, choices=("adhoc", "none"))
    args = parser.parse_args(argv)
    variants: dict[int, Path] = {}
    for dimension, path in args.variant:
        if dimension in variants:
            parser.error("--variant repeats Dim=%d" % dimension)
        variants[dimension] = path
    try:
        evidence = assemble_native_variant_wheel(
            variants,
            output=args.output,
            python_executable=args.python,
            expect_mpi=_on_off(args.expect_mpi),
            expect_kokkos=_on_off(args.expect_kokkos),
            signature_policy=args.signature_policy,
        )
    except (FatWheelAssemblyError, NativeVariantManifestError, OSError, ValueError) as exc:
        print("fat wheel assembly failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
