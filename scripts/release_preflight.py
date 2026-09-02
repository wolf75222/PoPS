#!/usr/bin/env python3
"""Fail-closed PoPS release preflight.

Development mode checks every static version/generator contract. ``--release`` additionally requires
an exact tag, installed native package, clean checkout and authenticated evidence for every expensive
build/example/IO gate. The evidence is machine output from the final gate, never a boolean CLI escape.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from final_release_contract import (  # noqa: E402
    FINAL_EXAMPLES,
    FINAL_EXAMPLE_REQUIRED_TESTS,
    FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS,
    INSTALLED_COMPONENT_PACKAGE_NODEID,
    PYTHON_REQUIRED_SELECTION,
    REQUIRED_PROOF_MARKERS,
    REQUIRED_RELEASE_GATES,
    required_python_conformance_nodeids,
    require_release_matrix_source_contract,
    require_source_contract,
)
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    validate_manifest_payload,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "python" / "pops" / "_generated_release_contract.py"
REQUIRED_GATES = REQUIRED_RELEASE_GATES
EVIDENCE_SCHEMA_VERSION = 12
PUBLIC_API_EVIDENCE_SCHEMA_VERSION = 3
RELEASE_NATIVE_DIMENSIONS = (1, 2, 3)


class PreflightError(RuntimeError):
    pass


def _generated() -> Any:
    spec = importlib.util.spec_from_file_location("_pops_release_contract", GENERATED)
    if spec is None or spec.loader is None:
        raise PreflightError("cannot load generated release contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-20:])
        raise PreflightError("command failed (%s):\n%s" % (" ".join(args), tail))
    return result.stdout.strip()


def _project_version() -> str:
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    found = re.findall(r"(?m)^\s*VERSION\s+(\d+\.\d+\.\d+)\b", text)
    if len(found) != 1:
        raise PreflightError("CMake must contain exactly one project VERSION")
    if "PROJECT_VERSION_MAJOR EQUAL 0" not in text or "SameMinorVersion" not in text:
        raise PreflightError("CMake pre-1.0 compatibility policy is not fail-closed")
    return found[0]


def _static_contract(contract: Any) -> list[str]:
    require_source_contract(ROOT)
    require_release_matrix_source_contract(ROOT)
    package_version = _project_version()
    if contract.PACKAGE_VERSION != package_version or package_version == "unknown":
        raise PreflightError("generated/package CMake versions disagree")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"].get("dynamic") != ["version"]:
        raise PreflightError("wheel version must remain dynamic from CMakeLists.txt")
    provider = project["tool"]["scikit-build"]["metadata"]["version"]
    if provider.get("input") != "CMakeLists.txt" or not re.search(
            provider["regex"], (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")):
        raise PreflightError("wheel version provider does not resolve the CMake version")

    source = json.loads((ROOT / "schemas" / "release_contract.v2.json").read_text())
    catalog = json.loads((ROOT / "schemas" / "component_catalog.v2.json").read_text())
    exact = {
        "component_catalog_schema_version": catalog["catalog_schema_version"],
        "component_manifest_schema_version": catalog["component_manifest_schema_version"],
        "component_registry_version": catalog["route_registry_version"],
        "capability_vocabulary_version": catalog["capability_vocabulary_version"],
        "component_interface_abi_version": catalog["native_interface_abi_version"],
    }
    for name, value in exact.items():
        if source[name] != value:
            raise PreflightError("release contract %s drifted from component catalog" % name)
    component_generated = ROOT / "python" / "pops" / "model" / "_generated_component_schema.py"
    component_spec = importlib.util.spec_from_file_location(
        "_release_component_schema", component_generated
    )
    if component_spec is None or component_spec.loader is None:
        raise PreflightError("cannot load generated component schema")
    component_contract = importlib.util.module_from_spec(component_spec)
    component_spec.loader.exec_module(component_contract)
    component_digests = {
        "component_catalog_sha256": component_contract.COMPONENT_CATALOG_SHA256,
        "component_catalog_semantic_sha256": (
            component_contract.COMPONENT_CATALOG_SEMANTIC_SHA256
        ),
    }
    for name, value in component_digests.items():
        if source[name] != value or getattr(contract, name.upper()) != value:
            raise PreflightError("release contract %s drifted from component catalog" % name)
    native = (ROOT / "include" / "pops" / "runtime" / "module_capabilities.hpp").read_text()
    match = re.search(r"kAbiVersion\s*=\s*(\d+)", native)
    if match is None or int(match.group(1)) != source["native_abi_version"]:
        raise PreflightError("release native ABI drifted from module capability ABI")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    if 'POPS_KOKKOS_FETCH_VERSION "' + source["supported_matrix"]["kokkos"]["version"] + '"' \
            not in cmake:
        raise PreflightError("supported Kokkos version drifted from CMake")
    if contract.RELEASE_CONTRACT_SHA256 != hashlib.sha256(json.dumps(
            {"package_version": package_version, **source}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode()).hexdigest():
        raise PreflightError("generated release contract digest is not canonical")
    _run(
        sys.executable,
        "scripts/ci_select_tests.py",
        "verify-cpp-duration-catalogs",
        "--shard-total",
        "7",
    )
    _run(sys.executable, "scripts/generate_release_contract.py", "--check")
    _run(sys.executable, "scripts/generate_component_catalog.py", "--check")
    return ["version", "wheel_metadata", "schemas", "abi", "matrix", "generated",
            "cpp_duration_catalogs", "final_specification", "final_examples"]


def _tag_contract(version: str, tag: str) -> None:
    if tag != "v" + version:
        raise PreflightError("release tag %r must equal v%s" % (tag, version))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(r"(?m)^## \[%s\](?:\s+-\s+\d{4}-\d{2}-\d{2})?\s*$" % re.escape(version),
                 changelog) is None:
        raise PreflightError("CHANGELOG has no exact release section for %s" % version)


def _installed_contract(contract: Any, dimension: int) -> dict[str, Any]:
    import pops
    from pops._native_selector import select_native_dimension

    if type(dimension) is not int or dimension not in RELEASE_NATIVE_DIMENSIONS:
        raise PreflightError("installed native dimension must be exactly 1, 2, or 3")
    select_native_dimension(dimension)
    from pops import _pops

    if pops.__version__ != contract.PACKAGE_VERSION or _pops.__version__ != contract.PACKAGE_VERSION:
        raise PreflightError("installed Python/native/package versions disagree")
    if _pops.__native_dimension__ != dimension:
        raise PreflightError("installed native dimension disagrees with explicit release dimension")
    if _pops.__abi_version__ != contract.NATIVE_ABI_VERSION:
        raise PreflightError("installed native ABI disagrees with release contract")
    if _pops.__release_contract_sha256__ != contract.RELEASE_CONTRACT_SHA256:
        raise PreflightError("installed native release digest disagrees with Python")
    extension = Path(_pops.__file__).resolve()
    if not extension.is_file():
        raise PreflightError("installed native extension has no readable origin")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "pops_file": str(Path(pops.__file__).resolve()),
        "native_dimension": dimension,
        "native_extension": str(extension),
        "native_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }


def _inside(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _command_evidence(directory: Path, rows: Any, *, gate: str) -> list[Path]:
    if not isinstance(rows, list) or not rows:
        raise PreflightError("release evidence %s has no command transcripts" % gate)
    logs: list[Path] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"argv", "log", "sha256"}:
            raise PreflightError("release evidence %s command %d is malformed" % (gate, index))
        if not isinstance(row["argv"], list) or not row["argv"] \
                or not all(isinstance(value, str) and value for value in row["argv"]):
            raise PreflightError("release evidence %s command %d has invalid argv" % (gate, index))
        relative = Path(row["log"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PreflightError("release evidence %s command %d escapes its directory" % (gate, index))
        log = (directory / relative).resolve()
        if not _inside(directory, log) or not log.is_file():
            raise PreflightError("release evidence %s command %d has no transcript" % (gate, index))
        actual = hashlib.sha256(log.read_bytes()).hexdigest()
        if row["sha256"] != actual:
            raise PreflightError("release evidence %s command %d transcript hash drifted" %
                                (gate, index))
        logs.append(log)
    return logs


def _artifact_file(root: Path, relative: Any, digest: Any, *, label: str) -> None:
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise PreflightError("release evidence %s is malformed" % label)
    path = (root / relative).resolve()
    if not _inside(root, path) or not path.is_file():
        raise PreflightError("release evidence %s is absent" % label)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise PreflightError("release evidence %s hash drifted" % label)


def _wheel_lane_contract(path: Path, archive: zipfile.ZipFile, contract: Any) -> None:
    """Require one native wheel whose filename and WHEEL tags match the promised lane."""

    lanes = contract.SUPPORTED_MATRIX["wheels"]
    if len(lanes) != 1:
        raise PreflightError("release contract must promise exactly one wheel lane")
    lane = lanes[0]
    if set(lane) != {"os", "arch", "python", "backend"}:
        raise PreflightError("promised wheel lane is malformed")
    if lane["os"] != "macos" or lane["arch"] != "arm64" \
            or lane["backend"] != "Kokkos Serial":
        raise PreflightError("promised wheel lane has no release tag verifier")

    if path.suffix != ".whl":
        raise PreflightError("release artifact is not a wheel")
    parts = path.name[:-4].split("-")
    if len(parts) != 5:
        raise PreflightError("release wheel filename must not contain a build tag")
    distribution, version, python_tag, abi_tag, platform_tag = parts
    expected_python = lane["python"]
    if distribution.lower().replace("_", "-") != "pops" \
            or version != contract.PACKAGE_VERSION:
        raise PreflightError("release wheel filename name/version disagrees with the contract")
    if python_tag != expected_python or abi_tag != expected_python:
        raise PreflightError("release wheel Python/ABI tags disagree with the promised lane")
    if re.fullmatch(r"macosx_\d+_\d+_arm64", platform_tag) is None:
        raise PreflightError("release wheel platform tag disagrees with the promised lane")

    dist_info = "%s-%s.dist-info" % (distribution, version)
    wheel_names = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
    if wheel_names != [dist_info + "/WHEEL"]:
        raise PreflightError("release wheel has no unique lane-bound WHEEL record")
    try:
        wheel_metadata = archive.read(wheel_names[0]).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreflightError("release wheel WHEEL record is not UTF-8") from exc
    fields: dict[str, list[str]] = {}
    for line in wheel_metadata.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, []).append(value)
    expected_tag = "%s-%s-%s" % (python_tag, abi_tag, platform_tag)
    if fields.get("Wheel-Version") != ["1.0"] \
            or fields.get("Root-Is-Purelib") != ["false"] \
            or fields.get("Tag") != [expected_tag]:
        raise PreflightError("release WHEEL metadata disagrees with the promised native lane")


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _safe_wheel_member(name: str) -> None:
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or str(relative) != name \
            or any(part in {"", ".", ".."} for part in relative.parts):
        raise PreflightError("release wheel contains an unsafe member: %r" % name)
    if ".data" in relative.parts:
        raise PreflightError("release wheel uses an unsupported .data installation scheme")


def _is_native_member(name: str) -> bool:
    filename = PurePosixPath(name).name
    return any(
        filename == "_pops" + suffix
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _wheel_native_contract(
    archive: zipfile.ZipFile,
    contract: Any,
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    """Reauthenticate the exact fat-native manifest, leaves and PEP 427 RECORD."""

    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise PreflightError("release wheel contains duplicate members")
    for info in infos:
        _safe_wheel_member(info.filename.rstrip("/"))
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise PreflightError("release wheel contains a symlink: %s" % info.filename)
    files = {name for name in names if not name.endswith("/")}
    manifest_member = "pops/_native/variants.json"
    manifests = [name for name in files if name == manifest_member]
    records = [name for name in files if name.endswith(".dist-info/RECORD")]
    if manifests != [manifest_member] or len(records) != 1:
        raise PreflightError("release wheel requires one native manifest and one RECORD")
    manifest_bytes = archive.read(manifest_member)
    try:
        payload = json.loads(manifest_bytes)
        variants = validate_manifest_payload(
            payload, expected_dimensions=RELEASE_NATIVE_DIMENSIONS
        )
    except (UnicodeDecodeError, json.JSONDecodeError, NativeVariantManifestError) as exc:
        raise PreflightError("release wheel native manifest is malformed: %s" % exc) from exc
    expected_members = {
        "pops/_native/" + row["path"]: row for row in variants
    }
    discovered = {name for name in files if _is_native_member(name)}
    if discovered != set(expected_members):
        raise PreflightError("release wheel native leaves do not equal its exact manifest")
    common_abi = set()
    for member, row in expected_members.items():
        if hashlib.sha256(archive.read(member)).hexdigest() != row["sha256"]:
            raise PreflightError(
                "release wheel Dim=%d bytes disagree with variants.json"
                % row["dimension"]
            )
        if row["version"] != contract.PACKAGE_VERSION \
                or row["has_mpi"] is not False or row["has_kokkos"] is not True:
            raise PreflightError(
                "release wheel Dim=%d version/backend facts disagree with its lane"
                % row["dimension"]
            )
        # The wheel lane promises Kokkos Serial.  ``has_kokkos`` alone is not enough: an
        # OpenMP (or another Kokkos) leaf can satisfy that boolean while carrying a different
        # execution-space ABI.  The writer obtains this identity from the loaded extension's
        # native runtime report; authenticate the exact value again from the retained manifest.
        if row["kokkos_execution_space"] != "Serial":
            raise PreflightError(
                "release wheel Dim=%d Kokkos execution space is %r, expected 'Serial'"
                % (row["dimension"], row["kokkos_execution_space"])
            )
        match = re.fullmatch(r"(.+);dim=([123])", row["abi_key"])
        if match is None or int(match.group(2)) != row["dimension"]:
            raise PreflightError(
                "release wheel Dim=%d ABI key lacks its exact dimension"
                % row["dimension"]
            )
        common_abi.add(match.group(1))
    if len(common_abi) != 1:
        raise PreflightError("release wheel native variants disagree on their toolchain ABI")

    record_member = records[0]
    try:
        record_rows = list(csv.reader(io.StringIO(
            archive.read(record_member).decode("utf-8"), newline=""
        )))
    except UnicodeDecodeError as exc:
        raise PreflightError("release wheel RECORD is not UTF-8") from exc
    if not record_rows or any(len(row) != 3 for row in record_rows):
        raise PreflightError("release wheel RECORD is malformed")
    by_name: dict[str, tuple[str, str]] = {}
    for name, digest, size in record_rows:
        _safe_wheel_member(name)
        if name in by_name:
            raise PreflightError("release wheel RECORD repeats a member")
        by_name[name] = (digest, size)
    if set(by_name) != files:
        raise PreflightError("release wheel RECORD does not cover its exact file set")
    for name in sorted(files):
        digest, size = by_name[name]
        if name == record_member:
            if digest or size:
                raise PreflightError("release wheel RECORD must not hash itself")
            continue
        member_bytes = archive.read(name)
        if (digest, size) != (_record_digest(member_bytes), str(len(member_bytes))):
            raise PreflightError("release wheel RECORD digest drifted: %s" % name)
    return manifest_bytes, variants


def _checkpoint_tree(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise PreflightError("release evidence checkpoint is empty: %s" % path)
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _wheel_evidence(directory: Path, gates: dict[str, Any], contract: Any) -> None:
    evidence = gates["official_build"]["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"wheel"}:
        raise PreflightError("official build evidence must contain exactly one retained wheel")
    wheel = evidence["wheel"]
    if not isinstance(wheel, dict) or set(wheel) != {"path", "sha256", "size"}:
        raise PreflightError("official build wheel evidence is malformed")
    if not isinstance(wheel["size"], int) or wheel["size"] <= 0:
        raise PreflightError("official build wheel has an invalid size")
    relative = Path(wheel["path"])
    _artifact_file(directory, wheel["path"], wheel["sha256"], label="release wheel")
    path = (directory / relative).resolve()
    if path.stat().st_size != wheel["size"]:
        raise PreflightError("official build wheel size drifted")
    try:
        with zipfile.ZipFile(path) as archive:
            _wheel_lane_contract(path, archive, contract)
            _wheel_native_contract(archive, contract)
            metadata_names = [name for name in archive.namelist()
                              if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise PreflightError("release wheel has no unique METADATA record")
            metadata = archive.read(metadata_names[0]).decode("utf-8")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise PreflightError("release wheel is unreadable: %s" % exc) from exc
    fields = {}
    for line in metadata.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, value)
    if fields.get("Name", "").lower() != "pops" \
            or fields.get("Version") != contract.PACKAGE_VERSION:
        raise PreflightError("release wheel name/version disagrees with the release contract")


def _public_api_evidence(
    path: Path,
    release_evidence: dict[str, Any],
    contract: Any,
) -> None:
    resolved = path.expanduser().resolve()
    if _inside(ROOT, resolved) or not resolved.is_file():
        raise PreflightError(
            "installed public API evidence must be one file outside the checkout")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PreflightError("installed public API evidence is unreadable") from exc
    expected = {
        "schema_version",
        "producer",
        "wheel_path",
        "wheel_sha256",
        "distribution",
        "typed_payload_files",
        "typed_payload_sha256",
        "public_api_sha256",
        "public_names",
        "pure_authoring",
        "qualified_handles",
        "py_typed",
        "installed",
        "installed_distribution",
        "installed_package",
        "installed_typed_payload_sha256",
        "installed_public_api_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected \
            or payload["schema_version"] != PUBLIC_API_EVIDENCE_SCHEMA_VERSION:
        raise PreflightError("installed public API evidence has an unknown schema")
    producer = {
        "script": "scripts/prove_public_api_parity.py",
        "sha256": hashlib.sha256(
            (ROOT / "scripts" / "prove_public_api_parity.py").read_bytes()
        ).hexdigest(),
    }
    if payload["producer"] != producer:
        raise PreflightError("installed public API evidence has another producer")
    wheel = release_evidence["gates"]["official_build"]["evidence"]["wheel"]
    if payload["wheel_sha256"] != wheel["sha256"]:
        raise PreflightError("installed public API evidence belongs to another wheel")
    distribution = payload["distribution"]
    installed_distribution = payload["installed_distribution"]
    if not isinstance(distribution, dict) or set(distribution) != {
            "name", "version", "metadata_sha256"}:
        raise PreflightError("public API wheel distribution identity is malformed")
    if installed_distribution != distribution:
        raise PreflightError("installed distribution identity differs from the release wheel")
    if not isinstance(distribution["name"], str) \
            or not isinstance(distribution["version"], str) \
            or distribution["name"].lower() != "pops" \
            or distribution["version"] != contract.PACKAGE_VERSION:
        raise PreflightError("public API distribution identity disagrees with the release")
    digests = (
        distribution["metadata_sha256"],
        payload["wheel_sha256"],
        payload["typed_payload_sha256"],
        payload["installed_typed_payload_sha256"],
        payload["public_api_sha256"],
        payload["installed_public_api_sha256"],
    )
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in digests):
        raise PreflightError("installed public API evidence contains an invalid digest")
    if payload["installed_typed_payload_sha256"] != payload["typed_payload_sha256"] \
            or payload["installed_public_api_sha256"] != payload["public_api_sha256"]:
        raise PreflightError("installed public API or typing digest differs from source")
    if payload["installed"] is not True or payload["pure_authoring"] is not True \
            or payload["qualified_handles"] is not True or payload["py_typed"] is not True:
        raise PreflightError("installed public API evidence did not prove the final contract")
    if not isinstance(payload["typed_payload_files"], int) \
            or payload["typed_payload_files"] <= 0 \
            or not isinstance(payload["public_names"], list) \
            or not payload["public_names"] \
            or not all(isinstance(name, str) and name for name in payload["public_names"]):
        raise PreflightError("installed public API evidence has an empty public surface")
    if not isinstance(payload["installed_package"], str):
        raise PreflightError("installed public API package path is malformed")
    installed_package = Path(payload["installed_package"]).resolve()
    runtime_package = Path(release_evidence["runtime"]["pops_file"]).resolve().parent
    if installed_package != runtime_package:
        raise PreflightError(
            "public API parity was not proven on the authenticated installed runtime")


def _installed_wheel_evidence(
    directory: Path,
    gates: dict[str, Any],
    contract: Any,
    runtime: dict[str, Any],
) -> None:
    wheel = gates["official_build"]["evidence"]["wheel"]
    retained = (directory / wheel["path"]).resolve()
    row = gates["installed_wheel"]
    evidence = row["evidence"]
    expected = {
        "schema_version",
        "expected_dimensions",
        "python_executable",
        "distribution_root",
        "package_file",
        "native_manifest",
        "native_manifest_member",
        "native_manifest_sha256",
        "native_variants",
        "installed_member_count",
        "installed_tree_sha256",
        "proof_script_sha256",
        "version",
        "wheel_path",
        "wheel_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise PreflightError("installed wheel evidence is malformed")
    expected_dimensions = evidence["expected_dimensions"]
    if evidence["schema_version"] != 3 \
            or expected_dimensions != list(RELEASE_NATIVE_DIMENSIONS) \
            or any(type(value) is not int for value in expected_dimensions):
        raise PreflightError("installed wheel evidence schema is unsupported")
    if evidence["version"] != contract.PACKAGE_VERSION:
        raise PreflightError("installed wheel evidence version disagrees with release contract")
    if Path(evidence["wheel_path"]).resolve() != retained \
            or evidence["wheel_sha256"] != wheel["sha256"]:
        raise PreflightError("installed wheel evidence does not authenticate the retained wheel")
    if evidence["python_executable"] != runtime["python_executable"] \
            or evidence["package_file"] != runtime["pops_file"]:
        raise PreflightError("installed wheel evidence belongs to another runtime")
    expected_manifest = Path(runtime["pops_file"]).resolve().parent / "_native" / "variants.json"
    if Path(evidence["native_manifest"]).resolve() != expected_manifest:
        raise PreflightError("installed wheel evidence authenticates another native manifest")

    commands = row["commands"]
    logs = _command_evidence(directory, commands, gate="installed_wheel")
    if len(logs) != 2:
        raise PreflightError("installed wheel gate requires reinstall and proof transcripts")
    install_suffix = [
        "python",
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(retained),
    ]
    proof_suffix = [
        "python",
        "scripts/prove_installed_wheel.py",
        "--wheel",
        str(retained),
        "--expect-dim",
        "1",
        "--expect-dim",
        "2",
        "--expect-dim",
        "3",
    ]
    if commands[0]["argv"][-len(install_suffix):] != install_suffix \
            or commands[1]["argv"][-len(proof_suffix):] != proof_suffix:
        raise PreflightError("installed wheel gate did not reinstall and prove the retained wheel")
    try:
        with zipfile.ZipFile(retained) as archive:
            manifest_bytes, manifest_rows = _wheel_native_contract(archive, contract)
            tree_rows = []
            for name in sorted(archive.namelist()):
                if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                    continue
                if ".data/" in name:
                    raise PreflightError(
                        "release wheel uses an unsupported .data installation scheme"
                    )
                digest = hashlib.sha256(archive.read(name)).hexdigest()
                tree_rows.append("%s\0%s\n" % (name, digest))
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise PreflightError("installed wheel native payload is unreadable: %s" % exc) from exc
    if evidence["native_manifest_member"] != "pops/_native/variants.json" \
            or evidence["native_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise PreflightError("installed wheel native manifest proof drifted")
    expected_tree = hashlib.sha256("".join(tree_rows).encode("utf-8")).hexdigest()
    if evidence["installed_member_count"] != len(tree_rows) \
            or evidence["installed_tree_sha256"] != expected_tree:
        raise PreflightError("installed wheel payload proof drifted")

    native_variants = evidence["native_variants"]
    if not isinstance(native_variants, list) or len(native_variants) != 3:
        raise PreflightError("installed wheel proof must authenticate exactly three variants")
    dimensions = [row.get("dimension") for row in native_variants if isinstance(row, dict)]
    if dimensions != list(RELEASE_NATIVE_DIMENSIONS) \
            or any(type(value) is not int for value in dimensions):
        raise PreflightError("installed wheel proof variant set/order drifted")
    manifest_by_dimension = {row["dimension"]: row for row in manifest_rows}
    installed_by_dimension: dict[int, dict[str, Any]] = {}
    distribution_root = Path(evidence["distribution_root"]).resolve()
    for variant in native_variants:
        if set(variant) != {
            "dimension", "extension", "member", "sha256", "version", "abi_key",
            "has_mpi", "has_kokkos",
        }:
            raise PreflightError("installed wheel native variant evidence is malformed")
        dimension = variant["dimension"]
        manifest_row = manifest_by_dimension[dimension]
        member = "pops/_native/" + manifest_row["path"]
        expected_extension = (distribution_root / member).resolve()
        if variant != {
            "dimension": dimension,
            "extension": str(expected_extension),
            "member": member,
            "sha256": manifest_row["sha256"],
            "version": manifest_row["version"],
            "abi_key": manifest_row["abi_key"],
            "has_mpi": manifest_row["has_mpi"],
            "has_kokkos": manifest_row["has_kokkos"],
        }:
            raise PreflightError(
                "installed wheel Dim=%d proof disagrees with its manifest" % dimension
            )
        installed_by_dimension[dimension] = variant
    active = installed_by_dimension.get(runtime["native_dimension"])
    if active is None or active["extension"] != runtime["native_extension"] \
            or active["sha256"] != runtime["native_sha256"]:
        raise PreflightError("installed wheel proof does not authenticate the active runtime")
    proof_script = ROOT / "scripts" / "prove_installed_wheel.py"
    if evidence["proof_script_sha256"] != hashlib.sha256(proof_script.read_bytes()).hexdigest():
        raise PreflightError("installed wheel proof script drifted")


def _codesign_evidence(
    directory: Path,
    gates: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    row = gates["codesign"]
    evidence = row["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {
            "schema_version", "platform", "extensions"}:
        raise PreflightError("codesign evidence is malformed")
    if evidence["schema_version"] != 2 or evidence["platform"] != "darwin":
        raise PreflightError("codesign evidence must authenticate the Darwin release lane")
    extensions = evidence["extensions"]
    dimensions = [
        item.get("dimension") for item in extensions if isinstance(item, dict)
    ] if isinstance(extensions, list) else []
    if not isinstance(extensions, list) or len(extensions) != 3 \
            or dimensions != list(RELEASE_NATIVE_DIMENSIONS) \
            or any(type(value) is not int for value in dimensions):
        raise PreflightError("codesign evidence must authenticate exactly Dim=1/2/3")
    installed_rows = gates["installed_wheel"]["evidence"]["native_variants"]
    installed_by_dimension = {item["dimension"]: item for item in installed_rows}
    for extension in extensions:
        if not isinstance(extension, dict) or set(extension) != {
                "dimension", "path", "sha256", "signature"}:
            raise PreflightError("codesign extension evidence is malformed")
        installed = installed_by_dimension.get(extension["dimension"])
        if installed is None or extension != {
            "dimension": extension["dimension"],
            "path": installed["extension"],
            "sha256": installed["sha256"],
            "signature": "adhoc",
        }:
            raise PreflightError(
                "codesign changed a retained-wheel native leaf or authenticated another path"
            )
    active = extensions[runtime["native_dimension"] - 1]
    if active["path"] != runtime["native_extension"] \
            or active["sha256"] != runtime["native_sha256"]:
        raise PreflightError("codesign evidence does not authenticate the active runtime")
    commands = row["commands"]
    logs = _command_evidence(directory, commands, gate="codesign")
    suffix = [
        "python", "scripts/codesign_pops_extensions.py", "--json",
        "--expect-dim", "1", "--expect-dim", "2", "--expect-dim", "3",
    ]
    if len(logs) != 1 or commands[0]["argv"][-len(suffix):] != suffix:
        raise PreflightError("codesign gate did not run the exact structured verifier")


def _examples_evidence(
    directory: Path,
    gates: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    examples = gates["examples"]["evidence"]
    reopen = gates["artifact_reopen"]["evidence"]
    restart = gates["strict_restart"]["evidence"]
    expected = {path.as_posix() for path in FINAL_EXAMPLES}
    if set(examples) != {"examples"} or set(reopen) != {"examples"} or set(restart) != {"examples"}:
        raise PreflightError("final-example evidence has an unknown schema")
    if set(examples["examples"]) != expected or set(reopen["examples"]) != expected \
            or set(restart["examples"]) != expected:
        raise PreflightError("final-example evidence does not cover exactly the final examples")
    command_rows = gates["examples"]["commands"]
    logs = _command_evidence(directory, command_rows, gate="examples")
    if len(logs) != len(FINAL_EXAMPLES):
        raise PreflightError("final examples must have one execution transcript each")
    for index, example in enumerate(FINAL_EXAMPLES):
        key = example.as_posix()
        row = examples["examples"][key]
        if not isinstance(row, dict) or set(row) != {
                "source_sha256", "stdout_sha256", "output_root", "runtime_sha256"}:
            raise PreflightError("release evidence %s is malformed" % key)
        if row["source_sha256"] != hashlib.sha256((ROOT / example).read_bytes()).hexdigest():
            raise PreflightError("release evidence source drifted for %s" % key)
        if not isinstance(row["output_root"], str):
            raise PreflightError("release evidence output root is invalid for %s" % key)
        if row["runtime_sha256"] != runtime["native_sha256"]:
            raise PreflightError("release evidence runtime digest drifted for %s" % key)
        output_root = (directory / row["output_root"]).resolve()
        expected_suffix = [
            "python",
            "scripts/run_installed_example.py",
            "--runtime-sha256",
            runtime["native_sha256"],
            "--example",
            key,
            "--",
            "--output-dir",
            str(output_root),
        ]
        command = command_rows[index]["argv"]
        if command[-len(expected_suffix):] != expected_suffix:
            raise PreflightError("release evidence command drifted for %s" % key)
        transcript = logs[index].read_text(encoding="utf-8")
        if row["stdout_sha256"] != hashlib.sha256(transcript.encode("utf-8")).hexdigest():
            raise PreflightError("release evidence stdout hash drifted for %s" % key)
        if any(marker not in transcript for marker in REQUIRED_PROOF_MARKERS):
            raise PreflightError("release evidence lacks restart/reopen proof output for %s" % key)
        runtime_marker = "PoPS release runtime | native_sha256=" + runtime["native_sha256"]
        if transcript.count(runtime_marker) != 1:
            raise PreflightError("release evidence runtime binding drifted for %s" % key)
        if not _inside(directory, output_root) or not output_root.is_dir():
            raise PreflightError("release evidence output root is absent for %s" % key)
        reopened = reopen["examples"][key]
        if not isinstance(reopened, dict) or set(reopened) != {"hdf5", "npz", "paraview"}:
            raise PreflightError("release evidence reopen record is malformed for %s" % key)
        expected_outputs = FINAL_EXAMPLE_SCIENTIFIC_OUTPUTS[example]
        for format_name in ("hdf5", "npz", "paraview"):
            artifacts = reopened[format_name]
            if not isinstance(artifacts, list):
                raise PreflightError(
                    "release evidence %s output ledger is malformed for %s"
                    % (format_name, key)
                )
            expectations = expected_outputs[format_name]
            if bool(artifacts) != bool(expectations):
                raise PreflightError(
                    "release evidence %s output coverage drifted for %s"
                    % (format_name, key)
                )
            artifact_roots = tuple(
                Path(expectation["artifact_root"]) for expectation in expectations
            )
            covered_roots: dict[Path, set[str]] = {
                artifact_root: set() for artifact_root in artifact_roots
            }
            for artifact in artifacts:
                if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
                    raise PreflightError("release evidence %s output is malformed for %s" %
                                        (format_name, key))
                if not isinstance(artifact["path"], str) or not isinstance(
                    artifact["sha256"], str
                ):
                    raise PreflightError(
                        "release evidence %s output identity is malformed for %s"
                        % (format_name, key)
                    )
                relative_artifact = Path(artifact["path"])
                containing_roots = tuple(
                    artifact_root for artifact_root in artifact_roots
                    if relative_artifact.is_relative_to(artifact_root)
                )
                if len(containing_roots) != 1:
                    raise PreflightError(
                        "release evidence %s output escaped its exact artifact root for %s"
                        % (format_name, key)
                    )
                covered_roots[containing_roots[0]].add(relative_artifact.suffix)
                _artifact_file(output_root, artifact["path"], artifact["sha256"],
                               label="%s %s" % (format_name, key))
            required_suffixes = {
                "hdf5": {".h5"},
                "npz": {".npz"},
                "paraview": {".pvd", ".vtu"},
            }[format_name]
            for artifact_root, suffixes in covered_roots.items():
                if not required_suffixes.issubset(suffixes):
                    raise PreflightError(
                        "release evidence %s output root %s lacks %s for %s"
                        % (format_name, artifact_root,
                           sorted(required_suffixes - suffixes), key)
                    )
        restarted = restart["examples"][key]
        if not isinstance(restarted, dict) or set(restarted) != {
                "checkpoint", "tree_sha256", "proof_markers"}:
            raise PreflightError("release evidence restart record is malformed for %s" % key)
        checkpoint = Path(restarted["checkpoint"]).resolve()
        if not _inside(directory, checkpoint) or not checkpoint.exists():
            raise PreflightError("release evidence checkpoint is absent for %s" % key)
        if restarted["tree_sha256"] != _checkpoint_tree(checkpoint):
            raise PreflightError("release evidence checkpoint hash drifted for %s" % key)
        if restarted["proof_markers"] != list(REQUIRED_PROOF_MARKERS):
            raise PreflightError("release evidence restart proof markers drifted for %s" % key)


def _final_example_test_evidence(evidence: dict[str, Any]) -> None:
    """Require the exact reviewed tests from the authenticated Python lane."""

    if evidence.get("final_example_nodeids") != list(FINAL_EXAMPLE_REQUIRED_TESTS):
        raise PreflightError("release evidence final-example test ledger drifted")


def _junit_evidence(
    report: Path,
    lane: dict[str, Any],
    *,
    required_nodeids: tuple[str, ...] = (),
) -> None:
    """Re-authenticate one retained JUnit report instead of trusting its JSON summary."""

    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PreflightError("release evidence JUnit report is invalid: %s" % exc) from exc
    cases = tuple(root.iter("testcase"))
    failed = tuple(
        case
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    )
    skipped = tuple(case for case in cases if case.find("skipped") is not None)
    actual = {
        "tests": len(cases),
        "failures": len(failed),
        "skips_or_xfails": len(skipped),
    }
    reported = {name: lane[name] for name in actual}
    if actual != reported:
        raise PreflightError(
            "release evidence JUnit summary drifted: reported=%s actual=%s"
            % (reported, actual)
        )
    if not cases or failed or skipped:
        raise PreflightError(
            "release evidence JUnit lane is not all-pass: "
            "tests=%d failures=%d skips_or_xfails=%d"
            % (len(cases), len(failed), len(skipped))
        )
    for nodeid in required_nodeids:
        relative, function_name = nodeid.split("::", 1)
        expected_class = str(Path(relative).with_suffix("")).replace("/", ".")
        matches = [
            case
            for case in cases
            if case.attrib.get("name", "").split("[", 1)[0] == function_name
            and case.attrib.get("classname", "").endswith(expected_class)
        ]
        if len(matches) != 1:
            raise PreflightError(
                "release evidence required final-example test %s appears %d times in JUnit"
                % (nodeid, len(matches))
            )


def _installed_component_package_evidence(
    directory: Path,
    python_conformance: dict[str, Any],
) -> None:
    component = python_conformance["evidence"]["installed_component_package"]
    if not isinstance(component, dict) or set(component) != {"nodeid", "headers", "lane"}:
        raise PreflightError("release evidence installed component package lane is malformed")
    if component["nodeid"] != INSTALLED_COMPONENT_PACKAGE_NODEID \
            or component["headers"] != "installed-wheel":
        raise PreflightError("release evidence installed component package authority drifted")
    lane = component["lane"]
    if not isinstance(lane, dict) or set(lane) != {
            "path", "sha256", "tests", "failures", "skips_or_xfails"}:
        raise PreflightError("release evidence installed component package JUnit is malformed")
    if lane["tests"] != 1 or lane["failures"] != 0 or lane["skips_or_xfails"] != 0:
        raise PreflightError("release evidence installed component package lane is not all-pass")
    component_report = Path(lane["path"]).resolve()
    if not _inside(directory, component_report):
        raise PreflightError(
            "release evidence installed component package JUnit path escapes its directory")
    _artifact_file(
        directory,
        component_report.relative_to(directory).as_posix(),
        lane["sha256"],
        label="installed component package JUnit",
    )
    _junit_evidence(
        component_report,
        lane,
        required_nodeids=(INSTALLED_COMPONENT_PACKAGE_NODEID,),
    )
    component_commands = [
        command for command in python_conformance["commands"]
        if INSTALLED_COMPONENT_PACKAGE_NODEID in command["argv"]
    ]
    if len(component_commands) != 1:
        raise PreflightError(
            "release evidence must execute the installed component package node exactly once")
    component_argv = component_commands[0]["argv"]
    include_assignments = [
        argument for argument in component_argv if argument.startswith("POPS_INCLUDE=")
    ]
    if include_assignments != ["POPS_INCLUDE="] \
            or "POPS_PROVE_INSTALLED_COMPONENT_PACKAGE=1" not in component_argv:
        raise PreflightError(
            "installed component package proof must use only wheel-owned headers")
    expected_suffix = [
        "python",
        "-m",
        "pytest",
        "-q",
        "-s",
        "-o",
        "xfail_strict=true",
        INSTALLED_COMPONENT_PACKAGE_NODEID,
        "--junitxml",
        lane["path"],
    ]
    if component_argv[-len(expected_suffix):] != expected_suffix:
        raise PreflightError("installed component package proof command drifted")


def _evidence(
    path: Path,
    contract: Any,
    commit: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "producer", "commit_sha", "package_version", "contract_sha256",
        "native_variant_set", "artifact_directory", "runtime", "gates",
    }
    if not isinstance(payload, dict) or set(payload) != expected \
            or payload["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise PreflightError("release evidence has an unknown or incomplete schema")
    if payload["commit_sha"] != commit or payload["package_version"] != contract.PACKAGE_VERSION \
            or payload["contract_sha256"] != contract.RELEASE_CONTRACT_SHA256:
        raise PreflightError("release evidence belongs to another build")
    producer = payload["producer"]
    expected_producer = {
        "script": "scripts/run_final_gate.py",
        "sha256": hashlib.sha256((ROOT / "scripts" / "run_final_gate.py").read_bytes()).hexdigest(),
    }
    if producer != expected_producer:
        raise PreflightError("release evidence was not produced by this final gate")
    if payload["runtime"] != runtime:
        raise PreflightError("release evidence belongs to another installed native extension")
    native_variant_set = payload["native_variant_set"]
    if native_variant_set != list(RELEASE_NATIVE_DIMENSIONS) \
            or any(type(value) is not int for value in native_variant_set) \
            or runtime["native_dimension"] not in RELEASE_NATIVE_DIMENSIONS:
        raise PreflightError("release evidence must authenticate exactly native Dim=1/2/3")
    gates = payload["gates"]
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES):
        raise PreflightError("release evidence gate set must be exactly %s" % (REQUIRED_GATES,))
    for name in REQUIRED_GATES:
        row = gates[name]
        if not isinstance(row, dict) or set(row) != {"status", "commands", "evidence"}:
            raise PreflightError("release evidence %s has an invalid row" % name)
        if row["status"] != "passed":
            raise PreflightError("release gate %s did not produce passing evidence" % name)
    artifact_relative = Path(payload["artifact_directory"])
    if artifact_relative.is_absolute() or len(artifact_relative.parts) != 1 \
            or artifact_relative.name in {"", ".", ".."}:
        raise PreflightError("release evidence artifact directory is invalid")
    directory = (path.resolve().parent / artifact_relative).resolve()
    if not _inside(path.resolve().parent, directory) or not directory.is_dir():
        raise PreflightError("release evidence artifact directory is absent")
    for name in REQUIRED_GATES:
        commands = gates[name]["commands"]
        if name == "strict_restart":
            if commands != []:
                raise PreflightError("derived release gate %s must not invent a command" % name)
        else:
            _command_evidence(directory, commands, gate=name)
    _wheel_evidence(directory, gates, contract)
    _installed_wheel_evidence(directory, gates, contract, runtime)
    _codesign_evidence(directory, gates, runtime)
    for name in ("native_conformance", "python_conformance"):
        evidence = gates[name]["evidence"]
        expected = {"required_lane"} if name == "native_conformance" \
            else {
                "required_lane",
                "selection",
                "nodeids",
                "final_example_nodeids",
                "installed_component_package",
            }
        if not isinstance(evidence, dict) or set(evidence) != expected:
            raise PreflightError("release evidence %s lane is malformed" % name)
        lane = evidence["required_lane"]
        if not isinstance(lane, dict) or set(lane) != {
                "path", "sha256", "tests", "failures", "skips_or_xfails"}:
            raise PreflightError("release evidence %s JUnit summary is malformed" % name)
        if not isinstance(lane["tests"], int) or lane["tests"] <= 0 \
                or lane["failures"] != 0 or lane["skips_or_xfails"] != 0:
            raise PreflightError("release evidence %s required lane is not all-pass" % name)
        report = Path(lane["path"]).resolve()
        if not _inside(directory, report):
            raise PreflightError("release evidence %s JUnit path escapes its directory" % name)
        _artifact_file(directory, report.relative_to(directory).as_posix(), lane["sha256"],
                       label="%s JUnit" % name)
        _junit_evidence(
            report,
            lane,
            required_nodeids=(
                required_python_conformance_nodeids(ROOT)
                if name == "python_conformance"
                else ()
            ),
        )
    python_evidence = gates["python_conformance"]["evidence"]
    if python_evidence["selection"] != PYTHON_REQUIRED_SELECTION:
        raise PreflightError("release evidence Python required-lane selection drifted")
    expected_python_nodeids = list(required_python_conformance_nodeids(ROOT))
    if python_evidence["nodeids"] != expected_python_nodeids:
        raise PreflightError("release evidence Python conformance ledger drifted")
    python_lane = python_evidence["required_lane"]
    python_commands = [
        command
        for command in gates["python_conformance"]["commands"]
        if python_lane["path"] in command["argv"]
    ]
    if len(python_commands) != 1:
        raise PreflightError("release evidence must execute one exact Python conformance lane")
    expected_python_suffix = [
        "python",
        "-m",
        "pytest",
        "-q",
        "-s",
        "-o",
        "xfail_strict=true",
        *expected_python_nodeids,
        "--junitxml",
        python_lane["path"],
    ]
    if python_commands[0]["argv"][-len(expected_python_suffix):] != expected_python_suffix:
        raise PreflightError("release evidence Python conformance command drifted")
    _final_example_test_evidence(python_evidence)
    _installed_component_package_evidence(directory, gates["python_conformance"])
    _examples_evidence(directory, gates, runtime)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--dim", type=int, choices=RELEASE_NATIVE_DIMENSIONS)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--public-api-evidence", type=Path)
    args = parser.parse_args()
    try:
        if args.release and (
            not args.tag
            or not args.installed
            or args.dim is None
            or args.evidence is None
            or args.public_api_evidence is None
        ):
            raise PreflightError(
                "--release requires --tag, --installed, --dim, --evidence and "
                "--public-api-evidence")
        contract = _generated()
        checks = _static_contract(contract)
        if args.release:
            _tag_contract(contract.PACKAGE_VERSION, args.tag)
            commit = _run("git", "rev-parse", "HEAD")
            if _run("git", "status", "--porcelain"):
                raise PreflightError("release checkout is dirty")
            runtime = _installed_contract(contract, args.dim)
            release_evidence = _evidence(args.evidence, contract, commit, runtime)
            _public_api_evidence(args.public_api_evidence, release_evidence, contract)
            checks.extend((
                "tag",
                "changelog",
                "installed",
                "evidence",
                "public_api_parity",
                "clean",
            ))
        elif args.tag:
            _tag_contract(contract.PACKAGE_VERSION, args.tag)
            checks.extend(("tag", "changelog"))
        print(json.dumps({"status": "passed", "package_version": contract.PACKAGE_VERSION,
                          "contract_sha256": contract.RELEASE_CONTRACT_SHA256,
                          "checks": checks}, sort_keys=True))
        return 0
    except (PreflightError, NativeVariantManifestError, OSError, ValueError, KeyError, TypeError) as exc:
        print("release preflight failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
