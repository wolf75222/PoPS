#!/usr/bin/env python3
"""Authenticate one completed ADC-700 ABBA campaign.

``compare.py`` performs the numerical/performance comparison.  This verifier is the second,
strict gate: it refuses a report unless the raw JSONL rows, four-rank device inventory, installed
extension/module provenance, and both allocation/dispatch probes are all present and internally
consistent.  It never turns a missing field into a pass and never creates evidence for a campaign
that did not run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import statistics
import stat
from typing import Any


SCHEMA = "pops.adc700.program_cutover.measurement.v1"
REPORT_SCHEMA = "pops.adc700.program_cutover.report.v1"
BASELINE_ROUTE = "pre_cutover_native"
CANDIDATE_ROUTE = "program_only"
BASELINE_REVISION = "db3d390f43dfb14f12e88db31a9b3e631ff50488"
DEVICE_TOKENS = ("cuda", "hip", "sycl")
DISPATCH_COMPLEXITY = "O(operations*levels), never O(cells)"
PROBE_NAMES = ("uniform", "amr_refined_planned")
MINIMUM_ABBA_BLOCKS = 5
MINIMUM_RATIO = 0.98
WHEEL_PROOF_SCHEMA_VERSION = 3
SIGNATURE_FIELDS = ("mass", "checksum", "checksum_square", "maximum")
SIGNATURE_RTOL = 1.0e-11
SIGNATURE_ATOL = 1.0e-12
CAMPAIGN_HARNESSES = (
    "benchmarks/adc700/CMakeLists.txt",
    "benchmarks/adc700/compare.py",
    "benchmarks/adc700/program_cutover.cpp",
    "benchmarks/adc700/program_cutover.py",
    "benchmarks/adc700/archive_receipt.py",
    "benchmarks/adc700/verify.py",
    "benchmarks/adc700/with_gpu_identity.sh",
    "benchmarks/manifest.toml",
    "benchmarks/romeo/adc700_program_cutover.sbatch",
    "benchmarks/romeo/submit_adc700_program_cutover.sh",
    "scripts/build_python.sh",
    "scripts/conda_runtime.sh",
    "scripts/prove_installed_wheel.py",
    "scripts/preserve_native_variants.py",
    "scripts/write_native_variant_manifest.py",
    "scripts/codesign_pops_extensions.py",
    "scripts/verify_installed_native.py",
)


class EvidenceError(ValueError):
    """The supplied campaign evidence is incomplete, forged, or inconsistent."""


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("%s must be a finite number" % where)
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError("%s must be a finite number" % where)
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError("cannot read JSON evidence %s: %s" % (path, error)) from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceError("cannot read raw measurements %s: %s" % (path, error)) from error
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvidenceError("%s:%d: invalid JSON: %s" % (path, line_number, error)) from error
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            raise EvidenceError("%s:%d: unexpected measurement schema" % (path, line_number))
        rows.append(row)
    if not rows:
        raise EvidenceError("raw measurements contain no rows")
    return rows


def _read_inventory(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceError("cannot read GPU inventory %s: %s" % (path, error)) from error
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        columns = raw.split("\t")
        if len(columns) != 2:
            raise EvidenceError("%s:%d: expected '<rank>\\t<device UUID>'" % (path, line_number))
        try:
            rank = int(columns[0])
        except ValueError as error:
            raise EvidenceError("%s:%d: rank is not an integer" % (path, line_number)) from error
        uuid = columns[1].strip()
        if rank < 0 or not uuid:
            raise EvidenceError("%s:%d: invalid rank or empty device UUID" % (path, line_number))
        rows.append({"rank": rank, "uuid": uuid})
    if sorted(row["rank"] for row in rows) != [0, 1, 2, 3]:
        raise EvidenceError("GPU inventory must cover exactly ranks 0,1,2,3")
    if len({row["uuid"] for row in rows}) != 4:
        raise EvidenceError("GPU inventory must contain four distinct device UUIDs")
    return sorted(rows, key=lambda row: row["rank"])


def _inventory_contract(inventory: Any) -> list[dict[str, str | int]]:
    """Normalize an inventory object before it participates in any raw-row comparison."""
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise EvidenceError("GPU inventory must contain exactly four rows")
    rows: list[dict[str, str | int]] = []
    for index, row in enumerate(inventory):
        if not isinstance(row, dict) or set(row) != {"rank", "uuid"}:
            raise EvidenceError("GPU inventory row %d has an invalid schema" % index)
        rank = row["rank"]
        uuid = row["uuid"]
        if type(rank) is not int or rank < 0 or rank >= 4:
            raise EvidenceError("GPU inventory row %d has an invalid rank" % index)
        if not isinstance(uuid, str) or not uuid.strip() or any(
            char in uuid for char in "\r\n\t"
        ):
            raise EvidenceError("GPU inventory row %d has an invalid UUID" % index)
        rows.append({"rank": rank, "uuid": uuid})
    if sorted(row["rank"] for row in rows) != [0, 1, 2, 3]:
        raise EvidenceError("GPU inventory must cover exactly ranks 0,1,2,3")
    if len({row["uuid"] for row in rows}) != 4:
        raise EvidenceError("GPU inventory must contain four distinct device UUIDs")
    return sorted(rows, key=lambda row: int(row["rank"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EvidenceError("cannot hash provenance file %s: %s" % (path, error)) from error
    return digest.hexdigest()


def _regular_path(value: Any, *, field: str, source_root: Path | None = None,
                  outside_source: bool = False, inside_source: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError("%s is missing" % field)
    path = Path(value)
    if not path.is_absolute():
        raise EvidenceError("%s must be absolute" % field)
    if path.is_symlink():
        raise EvidenceError("%s must not be a symlink: %s" % (field, path))
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("%s is not a regular file: %s" % (field, path))
    if source_root is not None:
        root = source_root.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            in_source = False
        else:
            in_source = True
        if outside_source and in_source:
            raise EvidenceError("%s points inside the candidate source tree: %s" % (field, path))
        if inside_source and not in_source:
            raise EvidenceError("%s points outside the candidate source tree: %s" % (field, path))
    return path


def _digest(value: Any, *, path: Path, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceError("%s must be a SHA-256" % field)
    observed = _sha256(path)
    if value.lower() != observed:
        raise EvidenceError("%s does not authenticate %s" % (field, path))
    return observed


def _candidate_source_root(value: Path | None) -> Path:
    if value is None:
        raise EvidenceError("an immutable archived candidate source root is required")
    if not value.is_absolute():
        raise EvidenceError("candidate source root must be absolute")
    if value.is_symlink():
        raise EvidenceError("candidate source root must not be a symlink")
    root = value.resolve()
    if not root.is_dir() or root.name != "candidate":
        raise EvidenceError(
            "candidate source root must be the extracted immutable candidate archive"
        )
    if stat.S_IMODE(root.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise EvidenceError("candidate source root must be immutable (read-only)")
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        if current.is_symlink() or stat.S_IMODE(current.stat().st_mode) & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise EvidenceError("candidate archive contains a writable directory")
        for name in directory_names:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise EvidenceError("candidate archive contains a symlink/non-directory")
            if stat.S_IMODE(child.stat().st_mode) & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise EvidenceError("candidate archive contains a writable directory")
        for name in file_names:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise EvidenceError("candidate archive contains a symlink/non-regular file")
            if stat.S_IMODE(child.stat().st_mode) & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise EvidenceError("candidate archive contains a writable file")
    return root


def _command_version(path: Path, *, field: str) -> str:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError("cannot query %s version: %s" % (field, error)) from error
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or not output:
        raise EvidenceError("%s --version did not return authenticated output" % field)
    return output


def _archive_contract(
    path: Path, *, source_root: Path, candidate_revision: str, helper_path: Path
) -> dict[str, Any]:
    """Recompute the archived-tree receipt with an external immutable helper.

    The pinned baseline does not contain this ADC-700 helper.  The caller therefore supplies a
    helper copied from the candidate archive to a separate immutable work-root path; its bytes are
    linked back to the candidate harness before either receipt is accepted.
    """
    root = _candidate_source_root(source_root)
    receipt = _regular_path(
        str(path), field="candidate archive receipt", source_root=root, outside_source=True
    )
    if stat.S_IMODE(receipt.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise EvidenceError("candidate archive receipt must be immutable")
    helper = _regular_path(
        str(helper_path), field="external archive receipt helper", source_root=root,
        outside_source=True,
    )
    if stat.S_IMODE(helper.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise EvidenceError("external archive receipt helper must be immutable")
    archived_helper = _regular_path(
        str(root / "benchmarks" / "adc700" / "archive_receipt.py"),
        field="candidate archive receipt helper", source_root=root, inside_source=True,
    )
    archived_helper_sha = _sha256(archived_helper)
    if _sha256(helper) != archived_helper_sha:
        raise EvidenceError("external archive receipt helper differs from candidate harness")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--root",
                str(root),
                "--role",
                "candidate",
                "--revision",
                candidate_revision,
                "--receipt",
                str(receipt),
                "--helper",
                str(helper),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError("cannot execute archived archive receipt helper: %s" % error) from error
    if result.returncode != 0:
        raise EvidenceError(
            "candidate archive receipt verification failed: %s" % result.stderr.strip()
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise EvidenceError("archived archive receipt helper did not return JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != "pops.adc700.archive_receipt.v1":
        raise EvidenceError("candidate archive receipt helper returned an invalid schema")
    if payload.get("schema_version") != 1 or payload.get("role") != "candidate" \
            or payload.get("revision") != candidate_revision \
            or payload.get("root_name") != "candidate" \
            or payload.get("root_path") != str(root) \
            or payload.get("immutable") is not True:
        raise EvidenceError("candidate archive receipt is not linked to the immutable SHA/tree")
    if not isinstance(payload.get("tree_sha256"), str) or len(payload["tree_sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in payload["tree_sha256"].lower()):
        raise EvidenceError("candidate archive receipt tree digest is malformed")
    if type(payload.get("entry_count")) is not int or payload["entry_count"] <= 0:
        raise EvidenceError("candidate archive receipt has no file manifest")
    script = payload.get("receipt_script")
    if not isinstance(script, dict) or set(script) != {"path", "sha256"}:
        raise EvidenceError("candidate archive receipt lacks helper provenance")
    if script.get("path") != str(helper) or script.get("sha256") != _sha256(helper) \
            or script.get("sha256") != archived_helper_sha:
        raise EvidenceError("candidate archive receipt helper provenance is not authenticated")
    return {
        "path": str(receipt),
        "sha256": _sha256(receipt),
        "revision": candidate_revision,
        "tree_sha256": payload.get("tree_sha256"),
        "entry_count": payload.get("entry_count"),
        "script_sha256": script["sha256"],
    }


def _wheel_contract(path: Path, *, source_root: Path | None) -> dict[str, Any]:
    root = _candidate_source_root(source_root)
    path = path.expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise EvidenceError("installed wheel proof must be an absolute non-symlink file")
    path = _regular_path(str(path), field="installed wheel proof", source_root=root,
                         outside_source=True)
    proof = _read_json(path)
    if proof.get("schema_version") != WHEEL_PROOF_SCHEMA_VERSION:
        raise EvidenceError("installed wheel proof schema is not version %d" % WHEEL_PROOF_SCHEMA_VERSION)
    if proof.get("expected_dimensions") != [2]:
        raise EvidenceError("installed wheel proof must contain exactly Dim=2")
    wheel = _regular_path(proof.get("wheel_path"), field="wheel proof.wheel_path",
                          source_root=root, outside_source=True)
    if wheel.suffix != ".whl":
        raise EvidenceError("wheel proof.wheel_path must name a .whl archive")
    wheel_sha = _digest(proof.get("wheel_sha256"), path=wheel, field="wheel proof.wheel_sha256")
    variants = proof.get("native_variants")
    if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], dict):
        raise EvidenceError("installed wheel proof must authenticate exactly one native variant")
    variant = variants[0]
    if set(variant) != {
        "dimension", "extension", "member", "sha256", "version", "abi_key", "has_mpi", "has_kokkos"
    }:
        raise EvidenceError("installed wheel native variant has an invalid schema")
    if variant.get("dimension") != 2 or variant.get("has_mpi") is not True \
            or variant.get("has_kokkos") is not True:
        raise EvidenceError("installed wheel proof does not authenticate Dim=2 MPI/Kokkos")
    extension = _regular_path(
        variant.get("extension"), field="wheel proof native extension",
        source_root=root, outside_source=True
    )
    extension_sha = _digest(variant.get("sha256"), path=extension,
                            field="wheel proof native extension sha256")
    abi_key = variant.get("abi_key")
    if not isinstance(abi_key, str) or not abi_key.strip():
        raise EvidenceError("wheel proof native ABI key is empty")
    package = _regular_path(proof.get("package_file"), field="wheel proof.package_file",
                            source_root=root, outside_source=True)
    manifest = _regular_path(proof.get("native_manifest"), field="wheel proof.native_manifest",
                             source_root=root, outside_source=True)
    if manifest != package.parent / "_native" / "variants.json":
        raise EvidenceError("wheel proof native manifest is not under the installed pops package")
    if extension.parent != package.parent / "_native":
        raise EvidenceError("wheel proof native extension is not under the installed pops package")
    manifest_sha = _digest(proof.get("native_manifest_sha256"), path=manifest,
                           field="wheel proof.native_manifest_sha256")
    if package.name != "__init__.py" or package.parent.name != "pops":
        raise EvidenceError("wheel proof package_file must be the installed pops/__init__.py")
    installed_tree = proof.get("installed_tree_sha256")
    if not isinstance(installed_tree, str) or len(installed_tree) != 64 \
            or any(char not in "0123456789abcdef" for char in installed_tree):
        raise EvidenceError("wheel proof installed_tree_sha256 is malformed")
    version = proof.get("version")
    if not isinstance(version, str) or not version:
        raise EvidenceError("wheel proof version is empty")
    if variant.get("version") != version:
        raise EvidenceError("wheel proof native variant version differs from package version")
    proof_script = _regular_path(
        str(root / "scripts" / "prove_installed_wheel.py"),
        field="archived wheel proof script", source_root=root, inside_source=True,
    )
    proof_script_sha = _digest(
        proof.get("proof_script_sha256"), path=proof_script,
        field="wheel proof.proof_script_sha256",
    )
    return {
        "proof_path": str(path),
        "proof_sha256": _sha256(path),
        "wheel_path": str(wheel),
        "wheel_sha256": wheel_sha,
        "installed_tree_sha256": installed_tree,
        "package_file": str(package),
        "package_sha256": _sha256(package),
        "native_manifest": str(manifest),
        "native_manifest_sha256": manifest_sha,
        "native_extension_path": str(extension),
        "native_extension_sha256": extension_sha,
        "dimension": 2,
        "abi_key": abi_key,
        "proof_script_sha256": proof_script_sha,
        "version": version,
    }


def _text_list(value: Any, *, field: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise EvidenceError("%s must be a %s list" % (field, "non-empty" if not allow_empty else ""))
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise EvidenceError("%s contains an empty/non-text value" % field)
    if len(set(value)) != len(value):
        raise EvidenceError("%s contains duplicates" % field)
    return list(value)


def _contract_files(value: Any, *, field: str, source_root: Path | None) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise EvidenceError("%s must be a non-empty file list" % field)
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise EvidenceError("%s[%d] is not a path/hash record" % (field, index))
        path = _regular_path(item["path"], field="%s[%d].path" % (field, index),
                             source_root=source_root, outside_source=True)
        result.append({
            "path": str(path),
            "sha256": _digest(item["sha256"], path=path,
                              field="%s[%d].sha256" % (field, index)),
        })
    return result


def _toolchain_contract(extension: dict[str, Any], *, source_root: Path | None) -> dict[str, Any]:
    toolchain = extension.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != {
            "nvcc_wrapper", "mpi", "kokkos", "native_loader", "requested"}:
        raise EvidenceError("candidate toolchain provenance is incomplete")
    nvcc = toolchain["nvcc_wrapper"]
    if not isinstance(nvcc, dict) or set(nvcc) != {"path", "sha256", "version"}:
        raise EvidenceError("candidate NVCC wrapper provenance is incomplete")
    nvcc_path = _regular_path(nvcc["path"], field="toolchain.nvcc_wrapper.path",
                              source_root=source_root, outside_source=True)
    nvcc_sha = _digest(nvcc["sha256"], path=nvcc_path, field="toolchain.nvcc_wrapper.sha256")
    if "nvcc_wrapper" not in nvcc_path.name.lower() or not isinstance(nvcc["version"], str) \
            or not nvcc["version"].strip():
        raise EvidenceError("candidate NVCC wrapper identity/version is not authenticated")
    if _command_version(nvcc_path, field="candidate NVCC wrapper") != nvcc["version"]:
        raise EvidenceError("candidate NVCC wrapper version changed in place")
    mpi = toolchain["mpi"]
    mpi_required = {
        "schema_version", "compiler", "compiler_version", "compiler_sha256", "abi_sha256",
        "standard", "compile_options", "compile_definitions", "link_options", "link_libraries",
        "include_dirs", "headers", "libraries",
    }
    if not isinstance(mpi, dict) or set(mpi) != mpi_required or mpi.get("schema_version") != 1:
        raise EvidenceError("candidate MPI toolchain provenance is incomplete")
    mpi_compiler = _regular_path(mpi["compiler"], field="toolchain.mpi.compiler",
                                 source_root=source_root, outside_source=True)
    mpi_compiler_sha = _digest(mpi["compiler_sha256"], path=mpi_compiler,
                               field="toolchain.mpi.compiler_sha256")
    for field in ("abi_sha256",):
        if not isinstance(mpi[field], str) or len(mpi[field]) != 64 \
                or any(char not in "0123456789abcdef" for char in mpi[field]):
            raise EvidenceError("toolchain.mpi.%s is malformed" % field)
    if not isinstance(mpi["compiler_version"], str) or not mpi["compiler_version"].strip():
        raise EvidenceError("toolchain.mpi.compiler_version is empty")
    if _command_version(mpi_compiler, field="candidate MPI compiler") != mpi["compiler_version"]:
        raise EvidenceError("candidate MPI compiler version changed in place")
    if not isinstance(mpi["standard"], str) or not mpi["standard"].strip():
        raise EvidenceError("toolchain.mpi.standard is empty")
    mpi_includes = _text_list(
        mpi.get("include_dirs"), field="toolchain.mpi.include_dirs", allow_empty=False
    )
    if any(not Path(item).is_absolute() or not Path(item).is_dir() for item in mpi_includes):
        raise EvidenceError("toolchain.mpi.include_dirs must name absolute directories")
    mpi_headers = _contract_files(mpi["headers"], field="toolchain.mpi.headers", source_root=source_root)
    mpi_libraries = _contract_files(mpi["libraries"], field="toolchain.mpi.libraries", source_root=source_root)
    mpi_options = {
        field: _text_list(mpi[field], field="toolchain.mpi.%s" % field)
        for field in ("compile_options", "compile_definitions", "link_options", "link_libraries")
    }
    if mpi_options["link_libraries"] != [item["path"] for item in mpi_libraries]:
        raise EvidenceError("toolchain.mpi.link_libraries differ from authenticated library files")
    mpi_header_map = {item["path"]: item["sha256"] for item in mpi_headers}
    mpi_material_lines = [
        "compiler=%s" % mpi_compiler,
        "standard=%s" % mpi["standard"],
        *("compile_option=%s" % value for value in mpi_options["compile_options"]),
        *("compile_definition=%s" % value for value in mpi_options["compile_definitions"]),
        *("link_option=%s" % value for value in mpi_options["link_options"]),
    ]
    for include_dir in mpi_includes:
        mpi_material_lines.append("include=%s" % include_dir)
        header_path = str(Path(include_dir) / "mpi.h")
        if header_path in mpi_header_map:
            mpi_material_lines.append(
                "header=%s;sha256=%s" % (header_path, mpi_header_map[header_path])
            )
    mpi_material_lines.extend(
        "library=%s;sha256=%s" % (item["path"], item["sha256"])
        for item in mpi_libraries
    )
    if hashlib.sha256(("\n".join(mpi_material_lines) + "\n").encode("utf-8")).hexdigest() \
            != mpi["abi_sha256"]:
        raise EvidenceError("toolchain.mpi.abi_sha256 does not authenticate its files/flags")
    kokkos = toolchain["kokkos"]
    if not isinstance(kokkos, dict) or set(kokkos) != {
            "schema_version", "abi_sha256", "include_dirs", "headers", "compile_options",
            "compile_definitions", "link_options", "link_libraries"} \
            or kokkos.get("schema_version") != 1:
        raise EvidenceError("candidate Kokkos toolchain provenance is incomplete")
    if not isinstance(kokkos["abi_sha256"], str) or len(kokkos["abi_sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in kokkos["abi_sha256"]):
        raise EvidenceError("toolchain.kokkos.abi_sha256 is malformed")
    kokkos_includes = _text_list(kokkos["include_dirs"], field="toolchain.kokkos.include_dirs", allow_empty=False)
    if any(not Path(item).is_absolute() or not Path(item).is_dir() for item in kokkos_includes):
        raise EvidenceError("toolchain.kokkos.include_dirs must name absolute directories")
    kokkos_headers = _contract_files(kokkos["headers"], field="toolchain.kokkos.headers", source_root=source_root)
    if any(Path(item["path"]).parent not in {Path(directory) for directory in kokkos_includes}
           for item in kokkos_headers):
        raise EvidenceError("toolchain.kokkos.headers must belong to Kokkos include directories")
    if len(kokkos_headers) != 2 or {
        Path(item["path"]).name for item in kokkos_headers
    } != {"Kokkos_Core.hpp", "KokkosCore_config.h"}:
        raise EvidenceError(
            "toolchain.kokkos.headers must authenticate exactly Kokkos_Core.hpp and "
            "KokkosCore_config.h"
        )
    expected_kokkos_headers = [
        str(Path(directory) / name)
        for directory in kokkos_includes
        for name in ("Kokkos_Core.hpp", "KokkosCore_config.h")
        if any(item["path"] == str(Path(directory) / name) for item in kokkos_headers)
    ]
    if [item["path"] for item in kokkos_headers] != expected_kokkos_headers:
        raise EvidenceError("toolchain.kokkos.headers are not in canonical include order")
    kokkos_header_map = {item["path"]: item["sha256"] for item in kokkos_headers}
    kokkos_material = "".join(
        [
            *("include=%s\n" % directory for directory in kokkos_includes),
            *(
                "header=%s;sha256=%s\n" % (path, kokkos_header_map[path])
                for directory in kokkos_includes
                for name in ("Kokkos_Core.hpp", "KokkosCore_config.h")
                for path in (str(Path(directory) / name),)
                if path in kokkos_header_map
            ),
        ]
    ).encode("utf-8")
    if hashlib.sha256(kokkos_material).hexdigest() != kokkos["abi_sha256"]:
        raise EvidenceError("toolchain.kokkos.abi_sha256 does not authenticate its files")
    kokkos_options = {
        field: _text_list(kokkos[field], field="toolchain.kokkos.%s" % field, allow_empty=True)
        for field in ("compile_options", "compile_definitions", "link_options", "link_libraries")
    }
    loader = toolchain["native_loader"]
    if not isinstance(loader, dict) or set(loader) != {"schema_version", "compile_definitions"} \
            or loader.get("schema_version") != 1:
        raise EvidenceError("candidate native-loader contract is incomplete")
    loader_defs = _text_list(loader["compile_definitions"], field="toolchain.native_loader.compile_definitions",
                             allow_empty=False)
    requested = toolchain["requested"]
    if not isinstance(requested, dict) or set(requested) != {
            "cxx", "include", "std", "compile_flags", "link_flags"}:
        raise EvidenceError("candidate requested compile options are incomplete")
    for field in ("cxx", "include", "std"):
        if not isinstance(requested[field], str) or not requested[field].strip():
            raise EvidenceError("toolchain.requested.%s is empty" % field)
    if requested["std"] != "c++20":
        raise EvidenceError("ADC-700 toolchain.requested.std must be c++20 exactly")
    compile_flags = _text_list(
        requested["compile_flags"], field="toolchain.requested.compile_flags", allow_empty=False
    )
    link_flags = _text_list(
        requested["link_flags"], field="toolchain.requested.link_flags", allow_empty=False
    )
    requested_cxx = Path(requested["cxx"])
    if not requested_cxx.is_absolute() or requested_cxx.is_symlink() \
            or requested_cxx.resolve() != nvcc_path:
        raise EvidenceError("requested CXX differs from the authenticated NVCC wrapper")
    include_arg = Path(requested["include"])
    if not include_arg.is_absolute() or include_arg.is_symlink():
        raise EvidenceError("requested include root must be absolute and non-symlink")
    include = include_arg.resolve()
    if not include.is_dir():
        raise EvidenceError("requested include root is unavailable")
    if source_root is not None:
        try:
            include.relative_to(source_root.resolve())
        except ValueError as error:
            raise EvidenceError("requested include root is not the archived candidate include tree") from error
    return {
        "nvcc_wrapper": {"path": str(nvcc_path), "sha256": nvcc_sha, "version": nvcc["version"]},
        "mpi": {"schema_version": 1, **mpi_options, "compiler": str(mpi_compiler),
                "compiler_sha256": mpi_compiler_sha,
                "compiler_version": mpi["compiler_version"], "abi_sha256": mpi["abi_sha256"],
                "standard": mpi["standard"], "include_dirs": mpi_includes,
                "headers": mpi_headers, "libraries": mpi_libraries},
        "kokkos": {"schema_version": 1, "abi_sha256": kokkos["abi_sha256"],
                   "include_dirs": kokkos_includes,
                   "headers": kokkos_headers, **kokkos_options},
        "native_loader": {"schema_version": 1, "compile_definitions": loader_defs},
        "requested": {"cxx": str(Path(requested["cxx"]).resolve()),
                       "include": str(include), "std": requested["std"],
                       "compile_flags": compile_flags, "link_flags": link_flags},
    }


def _toolchain_receipt_contract(
    value: Any,
    *,
    toolchain: dict[str, Any],
    candidate_revision: str,
    source_root: Path,
) -> dict[str, str]:
    """Require the external toolchain receipt to be immutable and byte/object-linked."""
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "revision"}:
        raise EvidenceError("toolchain receipt metadata has an invalid schema")
    receipt = _regular_path(
        value["path"], field="toolchain receipt.path", source_root=source_root, outside_source=True
    )
    if stat.S_IMODE(receipt.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise EvidenceError("toolchain receipt must be immutable")
    digest = _digest(value["sha256"], path=receipt, field="toolchain receipt.sha256")
    if value["revision"] != candidate_revision:
        raise EvidenceError("toolchain receipt is not linked to the candidate revision")
    payload = _read_json(receipt)
    if payload != toolchain:
        raise EvidenceError("toolchain receipt differs from candidate toolchain provenance")
    return {"path": str(receipt), "sha256": digest, "revision": candidate_revision}


def _harness_contract(extension: dict[str, Any], *, source_root: Path | None) -> dict[str, dict[str, str]]:
    root = _candidate_source_root(source_root)
    expected = set(CAMPAIGN_HARNESSES)
    value = extension.get("harnesses")
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError("candidate campaign harness provenance is incomplete")
    result = {}
    for relative, item in value.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise EvidenceError("candidate harness %s is malformed" % relative)
        path = _regular_path(item["path"], field="candidate harness %s" % relative,
                             source_root=root, inside_source=True)
        expected_path = (root / relative).resolve()
        if path != expected_path:
            raise EvidenceError("candidate harness %s does not name its archived path" % relative)
        result[relative] = {"path": str(path), "sha256": _digest(item["sha256"], path=path,
                                                                    field="candidate harness %s sha256" % relative)}
    return result


def _path_from_provenance(value: Any, *, field: str, source_root: Path | None) -> Path:
    try:
        return _regular_path(
            value,
            field="candidate extension.%s" % field,
            source_root=source_root,
            outside_source=True,
        )
    except EvidenceError:
        raise


def _extension(
    row: dict[str, Any], *, source_root: Path | None, wheel: dict[str, Any],
    archive_receipt: dict[str, Any], candidate_revision: str,
) -> dict[str, Any]:
    extension = row.get("extension")
    if not isinstance(extension, dict):
        raise EvidenceError("candidate measurement has no extension provenance")
    required = {
        "extension_path", "extension_sha256", "generated_module_path", "generated_module_sha256",
        "dimension", "mpi", "mpi_compiled", "mpi_active", "mpi_ranks", "kokkos_space",
        "kokkos_backend", "abi_key", "artifact_abi_key", "artifact_key", "program_hash",
        "toolchain", "wheel", "harnesses",
        "toolchain_receipt", "archive_receipt",
    }
    missing = sorted(required - set(extension))
    if missing:
        raise EvidenceError("extension provenance is missing %s" % ", ".join(missing))
    native = _path_from_provenance(
        extension["extension_path"], field="extension_path", source_root=source_root
    )
    module = _path_from_provenance(
        extension["generated_module_path"], field="generated_module_path", source_root=source_root
    )
    for path, digest_field in (
        (native, "extension_sha256"),
        (module, "generated_module_sha256"),
    ):
        digest = extension[digest_field]
        if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != _sha256(path):
            raise EvidenceError("candidate extension digest mismatch for %s" % path)
    if type(extension["dimension"]) is not int or extension["dimension"] != 2:
        raise EvidenceError("ADC-700 candidate MODULE must authenticate native dimension 2")
    mpi = extension["mpi"]
    if not isinstance(mpi, dict) or mpi.get("compiled") is not True \
            or mpi.get("active") is not True or mpi.get("ranks") != 4 \
            or mpi.get("communicator") != "MPI_COMM_WORLD":
        raise EvidenceError("extension provenance does not authenticate active MPI size four")
    if extension["mpi_compiled"] is not True or extension["mpi_active"] is not True \
            or extension["mpi_ranks"] != 4:
        raise EvidenceError("extension top-level MPI provenance is inconsistent")
    space = extension["kokkos_space"]
    if not isinstance(space, str) or not space.strip() or space.lower() == "unknown":
        raise EvidenceError("extension provenance has no authenticated Kokkos space")
    backend = extension.get("kokkos_backend")
    if not isinstance(backend, str) or not any(token in backend.lower() for token in DEVICE_TOKENS):
        raise EvidenceError("extension provenance has no authenticated accelerator backend")
    for field in ("abi_key", "artifact_abi_key", "artifact_key", "program_hash"):
        if not isinstance(extension[field], str) or not extension[field].strip():
            raise EvidenceError("extension.%s is empty" % field)
    toolchain = _toolchain_contract(extension, source_root=source_root)
    if extension["toolchain"] != toolchain:
        raise EvidenceError("candidate toolchain provenance differs from its normalized contract")
    toolchain_receipt = _toolchain_receipt_contract(
        extension["toolchain_receipt"],
        toolchain=toolchain,
        candidate_revision=candidate_revision,
        source_root=_candidate_source_root(source_root),
    )
    if extension["archive_receipt"] != archive_receipt:
        raise EvidenceError("candidate extension archive receipt differs from the retained receipt")
    harnesses = _harness_contract(extension, source_root=source_root)
    wheel_row = extension.get("wheel")
    if not isinstance(wheel_row, dict) or set(wheel_row) != set(wheel):
        raise EvidenceError("candidate wheel provenance is incomplete")
    if wheel_row != wheel:
        raise EvidenceError("candidate wheel provenance differs from installed wheel proof")
    if extension["extension_sha256"].lower() != wheel["native_extension_sha256"] \
            or extension["abi_key"] != wheel["abi_key"] \
            or extension["dimension"] != wheel["dimension"]:
        raise EvidenceError("candidate extension is not linked to the retained wheel ABI/dimension")
    if native != Path(wheel["native_extension_path"]).resolve():
        raise EvidenceError("candidate extension path differs from the retained wheel proof")
    if extension["artifact_abi_key"] != wheel["abi_key"]:
        raise EvidenceError("generated MODULE ABI differs from the retained wheel ABI")
    return {
        "extension_path": str(native),
        "extension_sha256": extension["extension_sha256"].lower(),
        "generated_module_path": str(module),
        "generated_module_sha256": extension["generated_module_sha256"].lower(),
        "dimension": extension["dimension"],
        "mpi": dict(mpi),
        "kokkos_space": space,
        "kokkos_backend": backend,
        "abi_key": extension["abi_key"],
        "artifact_abi_key": extension["artifact_abi_key"],
        "artifact_key": extension["artifact_key"],
        "program_hash": extension["program_hash"],
        "toolchain": toolchain,
        "toolchain_receipt": toolchain_receipt,
        "archive_receipt": dict(archive_receipt),
        "wheel": dict(wheel),
        "harnesses": harnesses,
    }


def _probe_one(probe: dict[str, Any], *, label: str, where: str) -> dict[str, Any]:
    required = {
        "status", "preparation", "allocations_after_prepare", "dispatches", "operations", "levels",
        "measured_steps", "dispatch_bound", "allocation_free", "dispatch_complexity", "cell_count",
        "fixed_operations", "fixed_levels", "cell_independent_dispatches", "profile",
    }
    missing = sorted(required - set(probe))
    if missing:
        raise EvidenceError("%s is missing %s" % (where, ", ".join(missing)))
    if probe["status"] != "passed" or probe["allocation_free"] is not True \
            or type(probe["allocations_after_prepare"]) is not int \
            or probe["allocations_after_prepare"] != 0:
        raise EvidenceError("%s did not prove allocation freedom after preparation" % where)
    if probe["dispatch_complexity"] != DISPATCH_COMPLEXITY:
        raise EvidenceError("%s has an unrecognised dispatch complexity" % where)
    if probe["fixed_operations"] is not True or probe["fixed_levels"] is not True \
            or probe["cell_independent_dispatches"] is not True:
        raise EvidenceError("%s does not declare fixed operations/levels and cell independence" % where)
    for field in ("dispatches", "operations", "levels", "measured_steps", "dispatch_bound", "cell_count"):
        if type(probe[field]) is not int or probe[field] < 0:
            raise EvidenceError("%s.%s is not a non-negative integer" % (where, field))
    if probe["dispatches"] <= 0 or probe["operations"] <= 0 \
            or probe["levels"] <= 0 or probe["measured_steps"] <= 0 \
            or probe["cell_count"] <= 0:
        raise EvidenceError("%s has empty operation/level/step/cell accounting" % where)
    if label == "uniform" and probe["levels"] != 1:
        raise EvidenceError("uniform probe must authenticate exactly one level")
    if label == "amr_refined_planned" and probe["levels"] < 2:
        raise EvidenceError("AMR probe must authenticate a refined hierarchy")
    expected_bound = probe["operations"] * probe["levels"] * probe["measured_steps"] * 4
    if probe["dispatch_bound"] != expected_bound or probe["dispatches"] > expected_bound:
        raise EvidenceError("%s exceeds O(operations*levels), never O(cells)" % where)
    preparation = probe["preparation"]
    if not isinstance(preparation, dict) or preparation.get("bind_complete") is not True \
            or preparation.get("scope") != "bind+warmups" \
            or preparation.get("bind_install_route") not in {
                "System.install_program", "AmrSystem.install_program"
            } \
            or type(preparation.get("warmups")) is not int or preparation["warmups"] <= 0 \
            or not isinstance(preparation.get("profile"), dict) \
            or not isinstance(preparation.get("counters_before_reset"), dict) \
            or preparation.get("reset_after_preparation") is not True:
        raise EvidenceError("%s does not authenticate bind/warmup preparation before counter reset" % where)
    preparation_profile = preparation["profile"]
    if preparation_profile.get("profile") != "advanced" \
            or preparation_profile.get("schema_version") != 1 \
            or preparation_profile.get("source") != "snapshot" \
            or not isinstance(preparation_profile.get("counters"), dict):
        raise EvidenceError("%s preparation profile is not an authenticated Advanced snapshot" % where)
    if label == "uniform" and preparation["bind_install_route"] != "System.install_program":
        raise EvidenceError("uniform probe does not authenticate System.install_program")
    if label == "amr_refined_planned" \
            and preparation["bind_install_route"] != "AmrSystem.install_program":
        raise EvidenceError("AMR probe does not authenticate AmrSystem.install_program")
    profile = probe["profile"]
    if not isinstance(profile, dict):
        raise EvidenceError("%s has no structured Advanced profile" % where)
    if profile.get("profile") != "advanced" or profile.get("schema_version") != 1 \
            or profile.get("source") != "snapshot":
        raise EvidenceError("%s profile is not an authenticated Advanced snapshot" % where)
    counters = profile.get("counters")
    if not isinstance(counters, dict) or counters.get("kernels") != probe["dispatches"]:
        raise EvidenceError("%s profile kernel counter is absent or inconsistent" % where)
    memory_view = profile.get("views", {}).get("by_memory") \
        if isinstance(profile.get("views"), dict) else None
    # PerformanceSummary.to_dict() preserves typed-view availability as
    # {"available": true, "entries": {"scratch_allocs": ...}}.  Requiring that envelope keeps
    # an unavailable view from being mistaken for an observed zero.
    memory = memory_view.get("entries") \
        if isinstance(memory_view, dict) and memory_view.get("available") is True else None
    if not isinstance(memory, dict) or type(memory.get("scratch_allocs")) is not int \
            or memory.get("scratch_allocs") != 0:
        raise EvidenceError("%s Advanced profile does not prove scratch_allocs=0" % where)
    return {
        "status": "passed",
        "allocations_after_prepare": 0,
        "dispatches": probe["dispatches"],
        "operations": probe["operations"],
        "levels": probe["levels"],
        "measured_steps": probe["measured_steps"],
        "dispatch_bound": expected_bound,
        "allocation_free": True,
        "dispatch_complexity": DISPATCH_COMPLEXITY,
        "cell_count": probe["cell_count"],
    }


def _probe(row: dict[str, Any], *, label: str) -> dict[str, Any]:
    probes = row.get("probes")
    if not isinstance(probes, dict) or label not in probes:
        raise EvidenceError("candidate measurement is missing the %s probe" % label)
    probe = probes[label]
    if not isinstance(probe, dict):
        raise EvidenceError("candidate %s probe is not an object" % label)
    normalized = _probe_one(probe, label=label, where="candidate %s probe" % label)
    resolutions = probe.get("resolutions")
    if not isinstance(resolutions, list) or len(resolutions) < 2:
        raise EvidenceError("candidate %s probe requires at least two resolutions" % label)
    nested = []
    seen_resolutions = set()
    for index, resolution_probe in enumerate(resolutions):
        if not isinstance(resolution_probe, dict):
            raise EvidenceError("candidate %s resolution %d is not an object" % (label, index))
        resolution = resolution_probe.get("resolution")
        if type(resolution) is not int or resolution <= 0 or resolution in seen_resolutions:
            raise EvidenceError("candidate %s resolution identity is invalid" % label)
        seen_resolutions.add(resolution)
        nested.append(_probe_one(
            resolution_probe, label=label, where="candidate %s resolution %d" % (label, index)
        ))
    if len({item["operations"] for item in nested}) != 1 \
            or len({item["levels"] for item in nested}) != 1 \
            or len({item["dispatches"] for item in nested}) != 1:
        raise EvidenceError("candidate %s dispatches vary with cell count" % label)
    if normalized["operations"] != nested[0]["operations"] \
            or normalized["levels"] != nested[0]["levels"] \
            or normalized["dispatches"] != nested[0]["dispatches"]:
        raise EvidenceError("candidate %s summary differs from resolution witnesses" % label)
    if probe.get("resolution_count") != len(resolutions) \
            or probe.get("cell_counts") != [item["cell_count"] for item in nested] \
            or probe.get("dispatches_by_resolution") != [item["dispatches"] for item in nested]:
        raise EvidenceError("candidate %s resolution accounting is inconsistent" % label)
    if probe.get("fixed_operations") is not True or probe.get("fixed_levels") is not True \
            or probe.get("cell_independent_dispatches") is not True:
        raise EvidenceError("candidate %s has no independent multi-resolution proof" % label)
    normalized.update({
        "resolution_count": len(resolutions),
        "cell_counts": [item["cell_count"] for item in nested],
        "dispatches_by_resolution": [item["dispatches"] for item in nested],
    })
    return normalized


def _derived_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = []
    for offset in range(0, len(rows), 4):
        a1, b1, b2, a2 = rows[offset:offset + 4]
        baseline_1 = _finite(a1["timing"].get("per_step_seconds"), "baseline timing")
        candidate_1 = _finite(b1["timing"].get("per_step_seconds"), "candidate timing")
        candidate_2 = _finite(b2["timing"].get("per_step_seconds"), "candidate timing")
        baseline_2 = _finite(a2["timing"].get("per_step_seconds"), "baseline timing")
        if min(baseline_1, baseline_2, candidate_1, candidate_2) <= 0.0:
            raise EvidenceError("ABBA timing values must all be positive")
        ratios.append(math.exp(0.5 * (
            math.log(baseline_1) + math.log(baseline_2)
            - math.log(candidate_1) - math.log(candidate_2)
        )))
    median = statistics.median(ratios)
    mad = statistics.median(abs(value - median) for value in ratios)
    return {
        "metric": "candidate_throughput_over_pre_cutover",
        "protocol": "paired_ABBA_geometric_ratio",
        "blocks": len(ratios),
        "ratios": ratios,
        "median": median,
        "mad": mad,
        "threshold": MINIMUM_RATIO,
        "passed": median >= MINIMUM_RATIO,
    }


def _derived_numerical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_rows = [row for row in rows if row.get("route") == BASELINE_ROUTE]
    candidate_rows = [row for row in rows if row.get("route") == CANDIDATE_ROUTE]
    if not baseline_rows or not candidate_rows:
        raise EvidenceError("raw ABBA rows contain no baseline/candidate pair")
    fields: dict[str, Any] = {}
    passed = True
    for field in SIGNATURE_FIELDS:
        baseline = statistics.median(
            _finite(row.get("signature", {}).get(field), "baseline signature.%s" % field)
            for row in baseline_rows
        )
        candidate = statistics.median(
            _finite(row.get("signature", {}).get(field), "candidate signature.%s" % field)
            for row in candidate_rows
        )
        difference = abs(candidate - baseline)
        limit = SIGNATURE_ATOL + SIGNATURE_RTOL * max(abs(baseline), abs(candidate))
        field_passed = difference <= limit
        passed = passed and field_passed
        fields[field] = {
            "baseline_median": baseline,
            "candidate_median": candidate,
            "absolute_difference": difference,
            "limit": limit,
            "passed": field_passed,
        }
    return {
        "relative_tolerance": SIGNATURE_RTOL,
        "absolute_tolerance": SIGNATURE_ATOL,
        "fields": fields,
        "passed": passed,
    }


def _assert_report_aggregates(
    report: dict[str, Any], *, performance: dict[str, Any], numerical: dict[str, Any]
) -> None:
    """Compare every aggregate against independently recomputed JSONL results."""
    observed_performance = report.get("performance")
    if not isinstance(observed_performance, dict):
        raise EvidenceError("comparator report has no performance aggregate")
    for field in ("metric", "protocol", "blocks", "threshold", "passed"):
        if observed_performance.get(field) != performance[field]:
            raise EvidenceError("report performance.%s differs from raw JSONL" % field)
    observed_ratios = observed_performance.get("ratios")
    if not isinstance(observed_ratios, list) or len(observed_ratios) != len(performance["ratios"]):
        raise EvidenceError("report performance.ratios differs from raw JSONL")
    for index, (observed, expected) in enumerate(zip(observed_ratios, performance["ratios"], strict=True)):
        if _finite(observed, "report performance.ratios[%d]" % index) != expected:
            raise EvidenceError("report performance.ratios differs from raw JSONL")
    for field in ("median", "mad"):
        if _finite(observed_performance.get(field), "report performance.%s" % field) != performance[field]:
            raise EvidenceError("report performance.%s differs from raw JSONL" % field)
    observed_numerical = report.get("numerical_parity")
    if not isinstance(observed_numerical, dict):
        raise EvidenceError("comparator report has no numerical parity aggregate")
    if observed_numerical.get("relative_tolerance") != numerical["relative_tolerance"] \
            or observed_numerical.get("absolute_tolerance") != numerical["absolute_tolerance"] \
            or observed_numerical.get("passed") != numerical["passed"]:
        raise EvidenceError("report numerical parity tolerances/status differ from raw JSONL")
    observed_fields = observed_numerical.get("fields")
    if not isinstance(observed_fields, dict) or set(observed_fields) != set(SIGNATURE_FIELDS):
        raise EvidenceError("report numerical parity fields differ from raw JSONL")
    for field in SIGNATURE_FIELDS:
        observed = observed_fields[field]
        expected = numerical["fields"][field]
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise EvidenceError("report numerical parity.%s differs from raw JSONL" % field)
        for key in ("baseline_median", "candidate_median", "absolute_difference", "limit"):
            if _finite(observed.get(key), "report numerical parity.%s.%s" % (field, key)) != expected[key]:
                raise EvidenceError("report numerical parity.%s differs from raw JSONL" % field)
        if observed.get("passed") != expected["passed"]:
            raise EvidenceError("report numerical parity.%s differs from raw JSONL" % field)


def _raw_gpu_contract(
    row: dict[str, Any], *, expected: list[dict[str, str | int]], index: int
) -> None:
    assignments = row.get("gpu_assignments")
    if not isinstance(assignments, list) or len(assignments) != 4:
        raise EvidenceError("raw row %d does not carry four per-run GPU UUID assignments" % index)
    normalized: list[dict[str, str | int]] = []
    for assignment_index, item in enumerate(assignments):
        if not isinstance(item, dict) or set(item) != {"rank", "uuid"}:
            raise EvidenceError(
                "raw row %d GPU assignment %d has an invalid schema" % (index, assignment_index)
            )
        rank = item["rank"]
        uuid = item["uuid"]
        if type(rank) is not int or rank < 0 or rank >= 4 \
                or not isinstance(uuid, str) or not uuid.strip():
            raise EvidenceError(
                "raw row %d GPU assignment %d is malformed" % (index, assignment_index)
            )
        normalized.append({"rank": rank, "uuid": uuid})
    if sorted(normalized, key=lambda item: int(item["rank"])) != expected:
        raise EvidenceError("raw row %d does not carry the authenticated GPU UUID map" % index)
    local = row.get("gpu")
    if not isinstance(local, dict) or set(local) != {"rank", "uuid"} \
            or type(local.get("rank")) is not int or local.get("rank") not in range(4) \
            or not isinstance(local.get("uuid"), str) \
            or local.get("uuid") != row.get("gpu_uuid"):
        raise EvidenceError("raw row %d lacks a rank-local GPU UUID witness" % index)
    rank = int(local["rank"])
    if expected[rank] != {"rank": rank, "uuid": local["uuid"]}:
        raise EvidenceError("raw row %d rank-local GPU UUID differs from inventory" % index)


def _raw_contract(
    rows: list[dict[str, Any]], *, baseline_revision: str, candidate_revision: str,
    source_root: Path | None, inventory: list[dict[str, Any]], wheel: dict[str, Any],
    archive_receipt: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise EvidenceError("raw measurements must be a list of JSON objects")
    if len(rows) < MINIMUM_ABBA_BLOCKS * 4 or len(rows) % 4:
        raise EvidenceError("raw measurements must contain at least five complete ABBA blocks")
    if any(row.get("schema") != SCHEMA for row in rows):
        raise EvidenceError("raw measurements contain an unexpected schema")
    expected = [BASELINE_ROUTE, CANDIDATE_ROUTE, CANDIDATE_ROUTE, BASELINE_ROUTE]
    reference = rows[0]
    if reference.get("mpi_ranks") != 4:
        raise EvidenceError("raw measurements must authenticate exactly four MPI ranks")
    if reference.get("mpi_communicator") != "MPI_COMM_WORLD":
        raise EvidenceError("raw measurements must authenticate MPI_COMM_WORLD")
    toolchain_reference = reference.get("toolchain")
    receipt_reference = reference.get("toolchain_receipt")
    if reference.get("toolchain_build_attested") is not True:
        raise EvidenceError("raw measurements lack the baseline CMake toolchain attestation")
    if not isinstance(toolchain_reference, dict) or not toolchain_reference:
        raise EvidenceError("raw measurements must carry the complete toolchain receipt object")
    if not isinstance(receipt_reference, dict):
        raise EvidenceError("raw measurements must carry toolchain receipt metadata")
    topology_reference = reference.get("topology")
    parameters_reference = reference.get("parameters")
    if not isinstance(parameters_reference, dict) or type(parameters_reference.get("n")) is not int \
            or parameters_reference["n"] < 16 or parameters_reference["n"] % 4:
        raise EvidenceError("raw measurements do not authenticate domain parameters")
    if not isinstance(topology_reference, dict) \
            or topology_reference.get("distribute_coarse") is not True \
            or type(topology_reference.get("coarse_max_grid")) is not int \
            or topology_reference.get("coarse_max_grid") != parameters_reference["n"] // 2:
        raise EvidenceError("raw measurements do not authenticate the coarse PatchLayout contract")
    for field in ("coarse_local_boxes", "coarse_total_boxes"):
        if type(topology_reference.get(field)) is not int or topology_reference[field] <= 0:
            raise EvidenceError("raw measurements do not authenticate coarse box counts")
    execution_space = str(reference.get("execution_space", ""))
    if not any(token in execution_space.lower() for token in DEVICE_TOKENS):
        raise EvidenceError("raw measurements do not authenticate a GPU Kokkos space")
    comparable = (
        "execution_space", "mpi_ranks", "mpi_communicator", "execution_concurrency", "real_bytes",
        "parameters", "topology", "toolchain_build_attested", "toolchain", "toolchain_receipt",
    )
    candidates: list[dict[str, Any]] = []
    candidate_extension: dict[str, Any] | None = None
    expected_gpu = sorted(inventory, key=lambda row: int(row["rank"]))
    for offset in range(0, len(rows), 4):
        block = rows[offset : offset + 4]
        if [row.get("route") for row in block] != expected:
            raise EvidenceError("raw measurements are not ordered as ABBA")
        for index, row in enumerate(block, start=offset):
            route = row.get("route")
            expected_revision = baseline_revision if route == BASELINE_ROUTE else candidate_revision
            if row.get("revision") != expected_revision:
                raise EvidenceError("raw row %d has the wrong revision" % index)
            if any(row.get(field) != reference.get(field) for field in comparable):
                raise EvidenceError("raw row %d differs in comparable topology/toolchain fields" % index)
            validation = row.get("validation")
            if not isinstance(validation, dict) or validation.get("passed") is not True:
                raise EvidenceError("raw row %d has no passed numerical validation" % index)
            timing = row.get("timing")
            if not isinstance(timing, dict) or _finite(timing.get("per_step_seconds"), "raw timing") <= 0:
                raise EvidenceError("raw row %d has no positive timing" % index)
            signature = row.get("signature")
            if not isinstance(signature, dict):
                raise EvidenceError("raw row %d has no numerical signature" % index)
            for field in ("mass", "checksum", "checksum_square", "maximum"):
                _finite(signature.get(field), "raw row %d signature.%s" % (index, field))
            _raw_gpu_contract(row, expected=expected_gpu, index=index)
            if route == CANDIDATE_ROUTE:
                candidates.append(row)
    for row in candidates:
        extension = _extension(
            row,
            source_root=source_root,
            wheel=wheel,
            archive_receipt=archive_receipt,
            candidate_revision=candidate_revision,
        )
        if candidate_extension is None:
            candidate_extension = extension
        elif extension != candidate_extension:
            raise EvidenceError(
                "candidate extension provenance differs between ABBA runs"
            )
        for label in PROBE_NAMES:
            _probe(row, label=label)
    if candidate_extension is None:
        raise EvidenceError("raw measurements contain no candidate extension provenance")
    if candidate_extension["toolchain"] != toolchain_reference:
        raise EvidenceError("baseline C++ toolchain receipt differs from candidate provenance")
    if candidate_extension["toolchain_receipt"] != receipt_reference:
        raise EvidenceError("baseline C++ toolchain receipt metadata differs from candidate")
    _toolchain_receipt_contract(
        receipt_reference,
        toolchain=toolchain_reference,
        candidate_revision=candidate_revision,
        source_root=_candidate_source_root(source_root),
    )
    derived_performance = _derived_performance(rows)
    derived_numerical = _derived_numerical(rows)
    return {
        "blocks": len(rows) // 4,
        "execution_space": execution_space,
        "mpi_ranks": 4,
        "candidate_rows": len(candidates),
        "toolchain": candidate_extension["toolchain"],
        "toolchain_receipt": candidate_extension["toolchain_receipt"],
        "topology": reference["topology"],
        "performance": derived_performance,
        "numerical_parity": derived_numerical,
    }


def _report_contract(
    report: dict[str, Any], *, baseline_revision: str, candidate_revision: str,
) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("schema_version") != 1:
        raise EvidenceError("unexpected comparator report schema")
    if report.get("status") != "passed":
        raise EvidenceError("comparator report is not passed")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) \
            or provenance.get("baseline_revision") != baseline_revision \
            or provenance.get("candidate_revision") != candidate_revision \
            or provenance.get("ordering") != "ABBA":
        raise EvidenceError("report provenance revisions do not match the requested campaign")
    device = report.get("device")
    if not isinstance(device, dict) or device.get("mpi_ranks") != 4 \
            or device.get("one_distinct_device_per_rank") is not True:
        raise EvidenceError("report device evidence is not the exact four-GPU route")
    execution_space = str(device.get("execution_space", ""))
    if not any(token in execution_space.lower() for token in DEVICE_TOKENS):
        raise EvidenceError("report device evidence is not an accelerator")
    performance = report.get("performance")
    if not isinstance(performance, dict) or performance.get("passed") is not True \
            or _finite(performance.get("median"), "report performance median") < MINIMUM_RATIO:
        raise EvidenceError("throughput ratio is below the %.2f acceptance gate" % MINIMUM_RATIO)
    numerical = report.get("numerical_parity")
    if not isinstance(numerical, dict) or numerical.get("passed") is not True:
        raise EvidenceError("numerical parity is not passed")
    return {"execution_space": execution_space, "throughput_ratio": performance["median"]}


def verify(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    *,
    baseline_revision: str = BASELINE_REVISION,
    candidate_revision: str,
    source_root: Path | None = None,
    wheel_proof: dict[str, Any] | None = None,
    archive_receipt: Path | None = None,
    archive_helper: Path | None = None,
) -> dict[str, Any]:
    """Validate all evidence and return a normalized authentication payload."""
    if baseline_revision != BASELINE_REVISION:
        raise EvidenceError("ADC-700 baseline revision is pinned to %s" % BASELINE_REVISION)
    if not candidate_revision or candidate_revision == baseline_revision:
        raise EvidenceError("candidate revision must be non-empty and differ from baseline")
    if wheel_proof is None:
        raise EvidenceError("installed wheel proof is required for ADC-700 authentication")
    normalized_inventory = _inventory_contract(inventory)
    archive_root = _candidate_source_root(source_root)
    if archive_receipt is None:
        raise EvidenceError("immutable candidate archive receipt is required")
    if archive_helper is None:
        raise EvidenceError("external immutable archive receipt helper is required")
    validated_archive = _archive_contract(
        archive_receipt,
        source_root=archive_root,
        candidate_revision=candidate_revision,
        helper_path=archive_helper,
    )
    if not isinstance(wheel_proof, dict) or not isinstance(wheel_proof.get("proof_path"), str):
        raise EvidenceError("installed wheel proof object is not authenticated")
    validated_wheel = _wheel_contract(
        Path(wheel_proof["proof_path"]), source_root=archive_root
    )
    if wheel_proof != validated_wheel:
        raise EvidenceError("installed wheel proof object differs from its retained proof file")
    wheel_proof = validated_wheel
    report_summary = _report_contract(
        report, baseline_revision=baseline_revision, candidate_revision=candidate_revision
    )
    raw_summary = _raw_contract(
        rows,
        baseline_revision=baseline_revision,
        candidate_revision=candidate_revision,
        source_root=archive_root,
        inventory=normalized_inventory,
        wheel=wheel_proof,
        archive_receipt=validated_archive,
    )
    if report["device"].get("execution_space") != raw_summary["execution_space"]:
        raise EvidenceError("report device execution space differs from raw JSONL")
    report_provenance = report.get("provenance")
    if not isinstance(report_provenance, dict) \
            or report_provenance.get("toolchain") != raw_summary["toolchain"] \
            or report_provenance.get("toolchain_receipt") != raw_summary["toolchain_receipt"] \
            or report_provenance.get("topology") != raw_summary["topology"]:
        raise EvidenceError("report provenance toolchain/topology differs from raw JSONL")
    report_device = report["device"].get("assignments")
    if report_device != normalized_inventory:
        raise EvidenceError("report device assignments differ from the authenticated inventory")
    _assert_report_aggregates(
        report,
        performance=raw_summary["performance"],
        numerical=raw_summary["numerical_parity"],
    )
    authenticated = copy.deepcopy(report)
    authenticated["status"] = "passed"
    authenticated["authentication"] = {
        "status": "authenticated",
        "method": "sha256(raw_jsonl + gpu_inventory + comparator_report)",
        "baseline_revision": baseline_revision,
        "candidate_revision": candidate_revision,
        "raw_blocks": raw_summary["blocks"],
        "candidate_rows": raw_summary["candidate_rows"],
        "mpi_ranks": 4,
        "distinct_gpu_count": 4,
        "execution_space": report_summary["execution_space"],
        "throughput_ratio": report_summary["throughput_ratio"],
        "wheel_sha256": wheel_proof["wheel_sha256"],
        "wheel_proof_sha256": wheel_proof["proof_sha256"],
        "wheel_installed_tree_sha256": wheel_proof["installed_tree_sha256"],
        "wheel_native_manifest_sha256": wheel_proof["native_manifest_sha256"],
        "wheel_proof_script_sha256": wheel_proof["proof_script_sha256"],
        "extension_sha256": wheel_proof["extension_sha256"],
        "extension_abi_key": wheel_proof["abi_key"],
        "extension_dimension": wheel_proof["dimension"],
        "candidate_archive_receipt_sha256": validated_archive["sha256"],
        "candidate_archive_tree_sha256": validated_archive["tree_sha256"],
        "candidate_archive_revision": validated_archive["revision"],
        "candidate_archive_receipt_script_sha256": validated_archive["script_sha256"],
        "toolchain_receipt_sha256": raw_summary["toolchain_receipt"]["sha256"],
    }
    return authenticated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="compare.py report JSON")
    parser.add_argument("--raw", type=Path, required=True, help="raw measurement JSONL")
    parser.add_argument("--device-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--wheel-proof", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--archive-helper", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = _read_json(args.input)
        if not isinstance(report, dict):
            raise EvidenceError("comparator report must be a JSON object")
        result = verify(
            report,
            _read_jsonl(args.raw),
            _read_inventory(args.device_inventory),
            baseline_revision=args.baseline_revision,
            candidate_revision=args.candidate_revision,
            source_root=args.source_root,
            archive_receipt=args.archive_receipt,
            archive_helper=args.archive_helper,
            wheel_proof=_wheel_contract(
                args.wheel_proof,
                source_root=None if args.source_root is None else args.source_root.resolve(),
            ),
        )
        # File digests authenticate the exact bytes fed to this verifier.  They are deliberately
        # added after structural validation and do not substitute for it.
        result["authentication"].update({
            "raw_sha256": _sha256(args.raw),
            "device_inventory_sha256": _sha256(args.device_inventory),
            "comparator_report_sha256": _sha256(args.input),
            "wheel_proof_sha256": _sha256(args.wheel_proof.resolve()),
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result["authentication"], sort_keys=True))
        return 0
    except (EvidenceError, OSError, TypeError, ValueError) as error:
        print("ADC-700 evidence rejected: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
