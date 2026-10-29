#!/usr/bin/env python3
"""Source, installed SDK and cohort identities, and the installation lease (ROM-17, CF14, K6).

A *cohort* is one authenticated source commit together with one installed PoPS SDK built from it.
The receipt protocol (``qualification_receipt.py``) names both; this module computes them and owns
the lease that keeps two workers from qualifying two different sources in one installation.

* The **source** is a commit whose tracked files are unmodified and whose SDK inputs carry no
  untracked file.  A dirty tree names no commit, so it never names a source.
* The **SDK** is the installed ``pops`` package directory: Python sources, installed headers, the
  native variant manifest and its leaves.  Its identity is the digest of the exact installed bytes.
  An installation is ``wheel`` only when pip's ``RECORD`` authenticates every installed file
  (``_native`` files may instead be authenticated by ``variants.json``, which the official build
  rewrites when it restores sibling dimensions or re-signs a leaf).
* The **coherence** of an SDK with a source compares the installed Python sources and headers with
  the blobs of the commit, and the header signature baked into each native leaf with the installed
  headers.  Native bytes cannot be compared with C++ sources: the official build *attests* each
  leaf it installs (``attest-build``), keyed by the leaf digest and the SDK-input digest.
* The **cohort record** binds one coherent source, one wheel installation and the attestation of
  every native leaf.  It lives in the installation (``<site>/.pops-sdk/cohort.json``), never
  changes once written, and a later ``verify`` re-hashes the installation against it.
* The **lease**: a qualification holds a shared lease on the installation for one source; a build
  or install holds it exclusively.  A second source, or a mutation, is refused while it is held.

Nothing here imports ``pops``: the identities of the code under qualification are computed from
files, not taken from that code.

Commands::

  identify --source ROOT [--package DIR]          print the source and installed SDK identities
  attest-build --source ROOT --package DIR --dimension N [--wheel W] [-- CMD...]
                                                  attest the Dim=N leaf an official build installed
  publish  --source ROOT --package DIR [--build-receipt FILE] [--seal]
                                                  write the immutable cohort record
  verify   --package DIR [--source ROOT]          re-authenticate an installation and its cohort
  hold     --installation DIR --source ROOT -- CMD   run CMD under a qualification lease
  mutate   --installation DIR -- CMD                 run CMD (a build or install) exclusively
  held     --installation DIR                        exit 0 when an ancestor's ``mutate`` holds DIR
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator, Mapping, Sequence
import contextlib
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    load_manifest,
)


SOURCE_SCHEMA = "pops.source@1"
SDK_SCHEMA = "pops.sdk@1"
COHORT_SCHEMA = "pops.cohort@1"
LEASE_SCHEMA = "pops.installation-lease@1"
NATIVE_BUILD_SCHEMA = "pops.native-build@1"

#: Paths whose content determines the installed SDK: the Python package, the native sources and
#: headers, the build graph and the scripts that the official builds run.  Two commits with equal
#: ``sdk_inputs_sha256`` build the same SDK; tests, examples and documentation are not inputs.
SDK_INPUT_PATHS = (
    "CMakeLists.txt",
    "CMakePresets.json",
    "cmake",
    "include",
    "pyproject.toml",
    "python/CMakeLists.txt",
    "python/bindings",
    "python/pops",
    "scripts/assemble_native_variant_wheel.py",
    "scripts/build_python.sh",
    "scripts/check_packaging_manifest.py",
    "scripts/ci_assemble_installed_wheel.py",
    "scripts/codesign_pops_extensions.py",
    "scripts/preserve_native_variants.py",
    "scripts/write_compiler_include_contract.py",
    "scripts/write_native_variant_manifest.py",
    "src",
)

#: Suffixes of the tracked Python sources that enter the wheel (``check_packaging_manifest.py``).
PYTHON_SOURCE_SUFFIXES = (".py", ".pyi", ".typed")
HEADER_CATEGORIES = ("api", "abi", "sdk-root", "sdk-support", "test-only")
INSTALLED_HEADER_CATEGORIES = HEADER_CATEGORIES[:-1]
STATE_DIRECTORY = ".pops-sdk"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class CohortError(RuntimeError):
    """A source, an installation or a cohort record does not authenticate."""


class InstallationBusy(CohortError):
    """The installation is leased to another source, or is being modified."""


# --- canonical digests -------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(domain: str, value: Any) -> str:
    """Domain-separated SHA-256 of one canonical JSON value."""
    payload = hashlib.sha256()
    payload.update(domain.encode() + b"\0")
    payload.update(canonical_json(value))
    return payload.hexdigest()


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_json_atomic(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".",
            delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.chmod(mode)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# --- source identity ---------------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                                check=False)
    except OSError as exc:
        raise CohortError("git is unavailable: %s" % exc) from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CohortError("git %s failed in %s: %s" % (" ".join(arguments), root, detail))
    return result.stdout


def _object_format(root: Path) -> str:
    try:
        value = _git(root, "rev-parse", "--show-object-format").decode().strip()
    except CohortError:
        return "sha1"
    return value if value in ("sha1", "sha256") else "sha1"


def git_blob_id(data: bytes, object_format: str = "sha1") -> str:
    """The Git object id of ``data`` as a blob, without writing it into any repository."""
    blob = hashlib.new(object_format)
    blob.update(b"blob %d\0" % len(data))
    blob.update(data)
    return blob.hexdigest()


def _tree_rows(root: Path, paths: Sequence[str]) -> list[tuple[str, str, str, str]]:
    """``(mode, type, object id, path)`` of every committed entry under ``paths`` at HEAD."""
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", *paths)
    rows = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, kind, object_id = meta.decode().split(" ")
        rows.append((mode, kind, object_id, path.decode("utf-8")))
    return rows


def _status(root: Path) -> tuple[list[str], list[str]]:
    """Tracked paths that differ from HEAD and untracked, non-ignored paths."""
    raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all",
               "--ignore-submodules=none")
    changed, untracked = [], []
    records = raw.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code, path = record[:2].decode(), record[3:].decode("utf-8")
        if "R" in code or "C" in code:
            index += 1  # a rename or copy carries its origin path as the next record
        (untracked if code == "??" else changed).append(path)
    return sorted(changed), sorted(untracked)


def _under(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def source_identity(root: Path) -> dict[str, Any]:
    """Identify the checkout at ``root``: commit, tree, cleanliness and SDK-input digest."""
    root = Path(root).resolve()
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    commit = _git(top, "rev-parse", "HEAD").decode().strip()
    tree = _git(top, "rev-parse", "HEAD^{tree}").decode().strip()
    if _COMMIT.fullmatch(commit) is None or _COMMIT.fullmatch(tree) is None:
        raise CohortError("HEAD of %s does not name one full commit and tree" % top)
    changed, untracked = _status(top)
    untracked_inputs = [path for path in untracked if _under(path, SDK_INPUT_PATHS)]
    changed_inputs = [path for path in changed if _under(path, SDK_INPUT_PATHS)]
    rows = _tree_rows(top, SDK_INPUT_PATHS)
    inputs = digest("pops.sdk-inputs", [list(row) for row in rows])
    return {
        "schema": SOURCE_SCHEMA,
        "root": str(top),
        "commit": commit,
        "tree": tree,
        "object_format": _object_format(top),
        "clean": not changed and not untracked_inputs,
        "sdk_inputs_clean": not changed_inputs and not untracked_inputs,
        "modified": changed,
        "untracked_sdk_inputs": untracked_inputs,
        "sdk_input_paths": list(SDK_INPUT_PATHS),
        "sdk_inputs_sha256": inputs,
    }


def require_clean_source(source: Mapping[str, Any]) -> None:
    if source.get("schema") != SOURCE_SCHEMA:
        raise CohortError("source identity has an unknown schema")
    if not source["clean"]:
        detail = (source["modified"] + source["untracked_sdk_inputs"])[:8]
        raise CohortError(
            "source %s is not clean (%d modified, %d untracked SDK inputs; first: %s); "
            "a dirty tree names no commit and cannot form a cohort"
            % (source["commit"], len(source["modified"]), len(source["untracked_sdk_inputs"]),
               ", ".join(detail)))


# --- installed SDK identity --------------------------------------------------------------------


def locate_package(package: Path | None = None) -> Path:
    """Return the ``pops`` package directory, explicitly or by import lookup without import."""
    if package is not None:
        resolved = Path(package).resolve()
    else:
        import importlib.util

        spec = importlib.util.find_spec("pops")
        if spec is None or not spec.submodule_search_locations:
            raise CohortError("no pops package is importable from this interpreter")
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1:
            raise CohortError("pops resolves to %d package directories: %s"
                              % (len(locations), locations))
        resolved = Path(locations[0]).resolve()
    if not (resolved / "__init__.py").is_file():
        raise CohortError("%s is not a pops package directory" % resolved)
    return resolved


def installed_files(package_dir: Path) -> tuple[dict[str, str], list[str]]:
    """``{relative path: sha256}`` of every installed file, and the symlinks found.

    Bytecode caches are not part of the installation's identity: ``__pycache__`` directories and
    ``.pyc`` files are skipped.  A symlink is never followed; it is reported instead.
    """
    files: dict[str, str] = {}
    links: list[str] = []
    for directory, subdirectories, names in os.walk(package_dir):
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        here = Path(directory)
        for name in list(subdirectories):
            if (here / name).is_symlink():
                links.append((here / name).relative_to(package_dir).as_posix())
                subdirectories.remove(name)
        for name in sorted(names):
            path = here / name
            relative = path.relative_to(package_dir).as_posix()
            if name.endswith(".pyc"):
                continue
            if path.is_symlink():
                links.append(relative)
                continue
            if path.is_file():
                files[relative] = sha256_file(path)
    return files, sorted(links)


def _record_hash(value: str) -> str | None:
    if not value:
        return None
    algorithm, _, encoded = value.partition("=")
    if algorithm != "sha256" or not encoded:
        return "unsupported:" + value
    padded = encoded + "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(padded.encode()).hex()


def _distribution(package_dir: Path) -> Path | None:
    """The ``*.dist-info`` beside ``package_dir`` whose RECORD owns ``pops/__init__.py``."""
    owners = []
    for info in sorted(package_dir.parent.glob("*.dist-info")):
        record = info / "RECORD"
        if not record.is_file():
            continue
        with open(record, encoding="utf-8", newline="") as stream:
            if any(row and row[0] == "pops/__init__.py" for row in csv.reader(stream)):
                owners.append(info)
    if len(owners) > 1:
        raise CohortError("several distributions claim pops/__init__.py: %s" % owners)
    return owners[0] if owners else None


def installation_record(package_dir: Path, files: Mapping[str, str],
                        native_authenticated: set[str]) -> dict[str, Any]:
    """Authenticate the installed files against pip's RECORD of the owning distribution."""
    info = _distribution(package_dir)
    if info is None:
        return {"kind": "unmanaged", "distribution": None, "version": None,
                "record_verified": False, "mismatched": [], "missing": [], "unrecorded": [],
                "native_outside_record": [], "wheel": None}
    recorded: dict[str, str | None] = {}
    with open(info / "RECORD", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].startswith("pops/"):
                continue
            recorded[row[0][len("pops/"):]] = _record_hash(row[1] if len(row) > 1 else "")
    mismatched, missing, outside = [], [], []
    for relative, expected in sorted(recorded.items()):
        if expected is None:
            continue  # bytecode rows carry no hash; they are not part of the identity
        actual = files.get(relative)
        if actual == expected:
            continue
        if relative in native_authenticated and actual is not None:
            outside.append(relative)
        elif actual is None and relative.endswith(".pyc"):
            continue
        elif actual is None:
            missing.append(relative)
        else:
            mismatched.append(relative)
    unrecorded = []
    for relative in sorted(files):
        if relative in recorded:
            continue
        if relative in native_authenticated:
            outside.append(relative)
        else:
            unrecorded.append(relative)
    metadata = info / "METADATA"
    version = None
    if metadata.is_file():
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version: "):
                version = line.split(": ", 1)[1]
                break
    wheel = None
    direct = info / "direct_url.json"
    if direct.is_file():
        try:
            payload = json.loads(direct.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CohortError("direct_url.json of %s is invalid JSON" % info) from exc
        archive = payload.get("archive_info") if isinstance(payload, dict) else None
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
            wheel = {"url": payload.get("url"), "sha256": hashes["sha256"]}
    verified = not mismatched and not missing and not unrecorded
    return {
        "kind": "wheel" if verified and wheel is not None else "unmanaged",
        "distribution": info.name,
        "version": version,
        "record_verified": verified,
        "mismatched": mismatched,
        "missing": missing,
        "unrecorded": unrecorded,
        "native_outside_record": sorted(set(outside)),
        "wheel": wheel,
    }


def _header_rows(manifest_text: str) -> tuple[tuple[str, str], ...]:
    """Installed ``(category, path)`` rows of ``pops_headers.manifest`` (toolchain grammar)."""
    rows = []
    seen = set()
    for number, raw in enumerate(manifest_text.splitlines(), 1):
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        parts = row.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in HEADER_CATEGORIES:
            raise CohortError("invalid installed-header manifest row %d" % number)
        category, relative = parts
        if relative.startswith("/") or ".." in relative.split("/") \
                or not relative.startswith("pops/") or relative in seen:
            raise CohortError("invalid installed-header manifest path: %s" % relative)
        seen.add(relative)
        if category != "test-only":
            rows.append((category, relative))
    return tuple(sorted(rows))


def header_signature(include: Path) -> str:
    """The installed-header signature that ``abi_key()`` bakes into a native leaf.

    Same normalization as ``pops.codegen.toolchain.pops_header_signature``, recomputed here so the
    check does not trust the package it qualifies.
    """
    rows = _header_rows((include / "pops_headers.manifest").read_text(encoding="utf-8"))
    entries = []
    for category, relative in rows:
        path = include.joinpath(*relative.split("/"))
        if path.is_symlink() or not path.is_file():
            raise CohortError("installed header is missing: %s" % relative)
        entries.append("%s %s\n%s\n" % (category, relative, sha256_file(path)))
    return hashlib.sha256("".join(entries).encode()).hexdigest()


def _abi_header_token(abi_key: str) -> str | None:
    for token in abi_key.split(";"):
        if token.startswith("headers="):
            return token[len("headers="):]
    return None


def native_rows(package_dir: Path) -> tuple[dict[str, Any], ...]:
    """Verified native leaves; none for a package without ``_native``, such as a checkout."""
    manifest = package_dir / "_native" / "variants.json"
    if not manifest.exists() and not manifest.is_symlink():
        return ()
    try:
        return load_manifest(manifest, verify_files=True, verify_hashes=True)
    except NativeVariantManifestError as exc:
        raise CohortError(str(exc)) from exc


def sdk_identity(package: Path | None = None) -> dict[str, Any]:
    """Identify the installed SDK by its exact bytes; paths are recorded, never hashed."""
    package_dir = locate_package(package)
    files, links = installed_files(package_dir)
    rows = native_rows(package_dir)
    native_paths = {"_native/variants.json"} | {"_native/" + row["path"] for row in rows}
    include = package_dir / "include"
    signature = header_signature(include) if (include / "pops_headers.manifest").is_file() \
        else None
    installation = installation_record(package_dir, files, native_paths)
    tree = digest("pops.sdk-tree", sorted(files.items()))
    return {
        "schema": SDK_SCHEMA,
        "sdk_id": digest("pops.sdk", {"tree": tree}),
        "package_dir": str(package_dir),
        "installation_root": str(package_dir.parent),
        "tree_sha256": tree,
        "file_count": len(files),
        "symlinks": links,
        "header_signature": signature,
        "native": [dict(row) for row in rows],
        "native_manifest_sha256": files.get("_native/variants.json"),
        "installation": installation,
    }


def sdk_file_table(package: Path | None = None) -> dict[str, str]:
    """The exact installed file table of an SDK, for receipts that re-check loaded modules."""
    files, _ = installed_files(locate_package(package))
    return files


# --- coherence of an SDK with a source ---------------------------------------------------------


def expected_payload(source_root: Path) -> dict[str, str]:
    """``{installed relative path: git blob id}`` that the commit at HEAD would install.

    Python sources come from ``python/pops`` with the packaged suffixes; headers come from the
    installed categories of ``include/pops_headers.manifest`` at HEAD, plus the manifest itself.
    Native leaves and ``variants.json`` are build outputs and are not part of this payload.
    """
    root = Path(_git(Path(source_root), "rev-parse", "--show-toplevel").decode().strip())
    payload: dict[str, str] = {}
    for _mode, kind, object_id, path in _tree_rows(root, ("python/pops",)):
        if kind != "blob" or not path.endswith(PYTHON_SOURCE_SUFFIXES):
            continue
        payload[path[len("python/pops/"):]] = object_id
    manifest_text = _git(root, "show", "HEAD:include/pops_headers.manifest").decode("utf-8")
    header_objects = {path: object_id for _, kind, object_id, path in _tree_rows(root, ("include",))
                      if kind == "blob"}
    for _category, relative in _header_rows(manifest_text):
        source = "include/" + relative
        if source not in header_objects:
            raise CohortError("manifest header %s is not committed at HEAD" % relative)
        payload["include/" + relative] = header_objects[source]
    payload["include/pops_headers.manifest"] = header_objects["include/pops_headers.manifest"]
    return payload


def coherence(source_root: Path, package: Path | None = None) -> dict[str, Any]:
    """Compare an installed SDK with the commit at ``source_root``; never imports ``pops``."""
    source_root = Path(source_root).resolve()
    package_dir = locate_package(package)
    object_format = _object_format(source_root)
    expected = expected_payload(source_root)
    files, links = installed_files(package_dir)
    rows = native_rows(package_dir)
    native_paths = {"_native/variants.json"} | {"_native/" + row["path"] for row in rows}
    missing, differing = [], []
    for relative, object_id in sorted(expected.items()):
        path = package_dir.joinpath(*relative.split("/"))
        if relative not in files:
            missing.append(relative)
        elif git_blob_id(path.read_bytes(), object_format) != object_id:
            differing.append(relative)
    unexpected = sorted(set(files) - set(expected) - native_paths)
    include = package_dir / "include"
    signature = header_signature(include) if (include / "pops_headers.manifest").is_file() \
        else None
    native_headers = []
    for row in rows:
        token = _abi_header_token(row["abi_key"])
        native_headers.append({"dimension": row["dimension"], "abi_headers": token,
                               "matches_installed_headers": token is not None
                               and token == signature})
    coherent = (not missing and not differing and not unexpected and not links
                and signature is not None
                and all(item["matches_installed_headers"] for item in native_headers))
    return {
        "coherent": coherent,
        "compared_files": len(expected),
        "missing": missing,
        "differing": differing,
        "unexpected": unexpected,
        "symlinks": links,
        "header_signature": signature,
        "native_headers": native_headers,
        "native_provenance": "not comparable with sources; attested by the cohort build record",
    }


# --- native build attestations -----------------------------------------------------------------


def native_build_directory(installation_root: Path) -> Path:
    return state_directory(installation_root) / "native-builds"


def _attestation_path(installation_root: Path, leaf_sha256: str, name: str) -> Path:
    if _DIGEST.fullmatch(leaf_sha256 or "") is None:
        raise CohortError("a native build attestation names one leaf sha256")
    return native_build_directory(installation_root) / leaf_sha256 / ("%s.json" % name)


def attest_native_build(source_root: Path, package: Path, dimension: int, *,
                        wheel: Path | None = None, command: Sequence[str] = ()) -> dict[str, Any]:
    """Record that the installed Dim=``dimension`` leaf was just built from ``source_root``.

    Native bytes cannot be compared with C++ sources, so the official build states it: right after
    installing, it writes ``.pops-sdk/native-builds/<leaf sha256>/<SDK inputs sha256>.json`` under
    the installation's mutation lease.  The key is the digest of the SDK inputs, not the commit:
    two commits with the same SDK inputs build the same SDK.  A build from modified SDK inputs is
    recorded as ``dirty.json`` and attests nothing.  Attestations outlive ``pip`` replacing the
    package, so a sibling dimension restored from an earlier build keeps the source it came from.
    """
    if dimension not in (1, 2, 3):
        raise CohortError("native dimension must be 1, 2 or 3")
    source = source_identity(source_root)
    package_dir = locate_package(package)
    rows = [row for row in native_rows(package_dir) if row["dimension"] == dimension]
    if len(rows) != 1:
        raise CohortError("installed variants.json has no Dim=%d leaf to attest" % dimension)
    row = dict(rows[0])
    leaf = package_dir / "_native" / row["path"]
    record = {
        "schema": NATIVE_BUILD_SCHEMA,
        "attested_utc": utc_now(),
        "host": socket.gethostname(),
        "source": source,
        "dimension": dimension,
        "variant": row,
        "leaf": str(leaf),
        "leaf_sha256": sha256_file(leaf),
        "wheel": None if wheel is None else {
            "path": str(Path(wheel).resolve()), "sha256": sha256_file(Path(wheel))},
        "command": list(command),
        "attester_sha256": sha256_file(Path(__file__).resolve()),
    }
    if record["leaf_sha256"] != row["sha256"]:
        raise CohortError("installed Dim=%d leaf disagrees with variants.json" % dimension)
    name = source["sdk_inputs_sha256"] if source["sdk_inputs_clean"] else "dirty"
    with installation_mutation(package_dir.parent, purpose="attest Dim=%d build" % dimension):
        write_json_atomic(_attestation_path(package_dir.parent, row["sha256"], name), record,
                          mode=0o444)
    return record


def _read_attestation(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("schema") == NATIVE_BUILD_SCHEMA \
        else None


def native_provenance(installation_root: Path, rows: Sequence[Mapping[str, Any]],
                      source: Mapping[str, Any]) -> dict[str, Any]:
    """Which official build of ``source`` attests each installed native leaf."""
    leaves = []
    for row in rows:
        path = _attestation_path(installation_root, row["sha256"], source["sdk_inputs_sha256"])
        attestation = _read_attestation(path)
        problem = None
        if attestation is None:
            dirty = _read_attestation(_attestation_path(installation_root, row["sha256"], "dirty"))
            others = sorted(item.stem for item in path.parent.glob("*.json")
                            if item.stem not in (path.stem, "dirty"))
            problem = ("built from modified SDK inputs at %s" % dirty["source"]["commit"]
                       if dirty is not None else
                       "built from other SDK inputs (%s)" % ", ".join(others) if others else
                       "no official build attests it")
        elif (attestation["source"]["sdk_inputs_sha256"] != source["sdk_inputs_sha256"]
              or not attestation["source"]["sdk_inputs_clean"]
              or attestation["leaf_sha256"] != row["sha256"]
              or attestation["dimension"] != row["dimension"]):
            problem = "attestation %s does not name this leaf and these SDK inputs" % path
        leaves.append({
            "dimension": row["dimension"], "sha256": row["sha256"],
            "attestation": None if attestation is None else str(path),
            "attestation_sha256": None if attestation is None else sha256_file(path),
            "built_from_commit": None if attestation is None
            else attestation["source"]["commit"],
            "problem": problem,
        })
    return {"kind": "build-attestation", "attested": bool(leaves) and not any(
        leaf["problem"] for leaf in leaves), "leaves": leaves}


# --- cohort record -----------------------------------------------------------------------------


def state_directory(installation_root: Path) -> Path:
    return Path(installation_root) / STATE_DIRECTORY


def cohort_path(installation_root: Path) -> Path:
    return state_directory(installation_root) / "cohort.json"


def cohort_id(source: Mapping[str, Any], sdk: Mapping[str, Any]) -> str:
    return digest("pops.cohort", {"commit": source["commit"], "tree": source["tree"],
                                  "sdk_id": sdk["sdk_id"]})


def _sdk_summary(sdk: Mapping[str, Any]) -> dict[str, Any]:
    return {key: sdk[key] for key in (
        "sdk_id", "package_dir", "installation_root", "tree_sha256", "file_count",
        "header_signature", "native", "native_manifest_sha256", "installation")}


_COHORT_KEYS = {"schema", "cohort_id", "published_utc", "host", "source", "sdk", "coherence",
                "native_provenance", "build", "sealed"}


def cohort_record(source: Mapping[str, Any], sdk: Mapping[str, Any],
                  check: Mapping[str, Any], provenance: Mapping[str, Any], *,
                  build: Mapping[str, Any] | None = None, sealed: bool = False) -> dict[str, Any]:
    """The cohort record of one clean source, one wheel installation and its native provenance.

    Pure: it writes nothing.  ``publish_cohort`` stores it in the installation; a test or a CI job
    may build one in memory to admit receipts against the exact installation it runs on.
    """
    require_clean_source(source)
    installation = sdk["installation"]
    if installation["kind"] != "wheel":
        raise CohortError(
            "an official cohort needs a wheel installation authenticated by RECORD; "
            "found %s (mismatched=%s missing=%s unrecorded=%s)"
            % (installation["kind"], installation["mismatched"][:5],
               installation["missing"][:5], installation["unrecorded"][:5]))
    if not check["coherent"]:
        raise CohortError(
            "installed SDK is not coherent with source %s: missing=%s differing=%s "
            "unexpected=%s native headers=%s"
            % (source["commit"], check["missing"][:5], check["differing"][:5],
               check["unexpected"][:5], check["native_headers"]))
    if not provenance.get("attested"):
        raise CohortError(
            "native leaves of source %s are not attested: %s"
            % (source["commit"], [(leaf.get("dimension"), leaf.get("problem"))
                                  for leaf in provenance.get("leaves", ())]))
    return {
        "schema": COHORT_SCHEMA,
        "cohort_id": cohort_id(source, sdk),
        "published_utc": utc_now(),
        "host": socket.gethostname(),
        "source": dict(source),
        "sdk": _sdk_summary(sdk),
        "coherence": dict(check),
        "native_provenance": dict(provenance),
        "build": None if build is None else dict(build),
        "sealed": bool(sealed),
    }


def publish_cohort(source_root: Path, package: Path, *, build_receipt: Path | None = None,
                   seal: bool = False) -> dict[str, Any]:
    """Bind a clean source and a coherent wheel installation into the immutable cohort record.

    The installation is leased exclusively while the record is written.  Every native leaf must be
    attested by an official build of the same SDK inputs (``attest_native_build``).  Publishing the
    same cohort again is a no-op; publishing another cohort into an installation that has one is
    refused: a new source gets a new installation, never a mutated one.
    """
    source = source_identity(source_root)
    require_clean_source(source)
    package_dir = locate_package(package)
    with installation_mutation(package_dir.parent,
                               purpose="publish cohort %s" % source["commit"]):
        sdk = sdk_identity(package_dir)
        check = coherence(source["root"], package_dir)
        provenance = native_provenance(package_dir.parent, sdk["native"], source)
        build = None
        if build_receipt is not None:
            raw = Path(build_receipt).read_bytes()
            build = {"path": str(Path(build_receipt).resolve()),
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "content": json.loads(raw)}
        record = cohort_record(source, sdk, check, provenance, build=build, sealed=seal)
        path = cohort_path(package_dir.parent)
        if path.exists():
            existing = read_cohort(package_dir.parent)
            if existing["cohort_id"] == record["cohort_id"]:
                return existing
            raise CohortError(
                "installation %s already holds cohort %s (source %s); a cohort record never "
                "changes: install source %s into a new installation"
                % (package_dir.parent, existing["cohort_id"], existing["source"]["commit"],
                   source["commit"]))
        write_json_atomic(path, record, mode=0o444)
        if seal:
            _seal(package_dir, sdk["installation"]["distribution"])
        return record


def _seal(package_dir: Path, distribution: str | None) -> None:
    """Remove write permission from the installed package and its distribution metadata."""
    roots = [package_dir]
    if distribution:
        roots.append(package_dir.parent / distribution)
    for root in roots:
        for directory, _subdirectories, names in os.walk(root, topdown=False):
            for name in names:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
            here = Path(directory)
            here.chmod(stat.S_IMODE(here.stat().st_mode) & ~0o222)


def validate_cohort(record: Any, *, where: str = "cohort record") -> dict[str, Any]:
    """Check the closed schema of a cohort record and that it authenticates its own identity."""
    if not isinstance(record, dict) or set(record) != _COHORT_KEYS \
            or record["schema"] != COHORT_SCHEMA:
        raise CohortError("%s has an unknown schema" % where)
    if cohort_id(record["source"], record["sdk"]) != record["cohort_id"]:
        raise CohortError("%s does not authenticate its own identity" % where)
    return record


def read_cohort(installation_root: Path) -> dict[str, Any]:
    path = cohort_path(installation_root)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CohortError("installation %s has no cohort record" % installation_root) from exc
    except json.JSONDecodeError as exc:
        raise CohortError("cohort record %s is invalid JSON" % path) from exc
    return validate_cohort(record, where="cohort record %s" % path)


def refuse_published_installation(installation_root: Path, purpose: str) -> None:
    """A cohort never changes: no build or install mutates an installation that holds one."""
    path = cohort_path(installation_root)
    if not path.exists() and not path.is_symlink():
        return
    try:
        record = read_cohort(installation_root)
        held = "cohort %s of source %s" % (record["cohort_id"], record["source"]["commit"])
    except CohortError as exc:
        held = "an unreadable cohort record (%s)" % exc
    raise CohortError(
        "installation %s holds %s; a cohort never changes, so %s needs a new installation "
        "(to retire this cohort deliberately, remove %s)" % (installation_root, held, purpose, path))


def verify_cohort(package: Path | None = None, *, source_root: Path | None = None
                  ) -> dict[str, Any]:
    """Re-hash an installation against its cohort record; optionally check a checkout too."""
    package_dir = locate_package(package)
    record = read_cohort(package_dir.parent)
    sdk = sdk_identity(package_dir)
    problems = []
    if sdk["sdk_id"] != record["sdk"]["sdk_id"]:
        problems.append("installed bytes changed since publication (sdk %s, cohort %s)"
                        % (sdk["sdk_id"], record["sdk"]["sdk_id"]))
    if sdk["installation"]["kind"] != "wheel":
        problems.append("installation no longer authenticates against RECORD")
    for leaf in record["native_provenance"].get("leaves", ()):
        path = leaf.get("attestation")
        if path and (not Path(path).is_file() or sha256_file(Path(path))
                     != leaf["attestation_sha256"]):
            problems.append("native build attestation of Dim=%s changed or disappeared: %s"
                            % (leaf["dimension"], path))
    if source_root is not None:
        source = source_identity(source_root)
        if not source["clean"]:
            problems.append("checkout %s is not clean" % source["root"])
        if (source["commit"], source["tree"]) != (record["source"]["commit"],
                                                  record["source"]["tree"]):
            problems.append("checkout is at %s, cohort source is %s"
                            % (source["commit"], record["source"]["commit"]))
    return {"cohort_id": record["cohort_id"], "verified": not problems, "problems": problems,
            "sdk_id": sdk["sdk_id"], "source": record["source"]["commit"]}


# --- installation lease ------------------------------------------------------------------------


def _flock(descriptor: int, operation: int) -> None:
    import fcntl

    fcntl.flock(descriptor, operation)


def _read_lease(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class Lease:
    """A held installation lease; released when closed or when its process ends."""

    def __init__(self, descriptor: int, record: dict[str, Any]) -> None:
        self._descriptor: int | None = descriptor
        self.record = record

    @property
    def held(self) -> bool:
        return self._descriptor is not None

    def release(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)  # closing the description drops its flock
            self._descriptor = None

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _lease_files(installation_root: Path) -> tuple[Path, Path, Path]:
    if os.name != "posix":
        raise CohortError("the installation lease requires POSIX flock")
    directory = state_directory(installation_root)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CohortError("cannot create the lease directory %s: %s" % (directory, exc)) from exc
    return directory / "guard.lock", directory / "lease.lock", directory / "lease.json"


def acquire_qualification_lease(installation_root: Path, source_commit: str, *,
                                purpose: str = "qualification") -> Lease:
    """Share the installation with other qualifiers of the same source, refuse any other one."""
    import fcntl

    installation_root = Path(installation_root).resolve()
    if _COMMIT.fullmatch(source_commit or "") is None:
        raise CohortError("a qualification lease names one full source commit")
    guard_path, lease_path, owner_path = _lease_files(installation_root)
    guard = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        _flock(guard, fcntl.LOCK_EX)
        descriptor = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            try:
                _flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                owner = _read_lease(owner_path)
                try:
                    _flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise InstallationBusy(
                        "installation %s is being modified (%s, pid %s)"
                        % (installation_root, owner.get("purpose"), owner.get("pid"))) from None
                if owner.get("mode") != "qualify" or owner.get("source") != source_commit:
                    raise InstallationBusy(
                        "installation %s is leased to qualify source %s (%s, pid %s); "
                        "source %s needs its own installation"
                        % (installation_root, owner.get("source"), owner.get("purpose"),
                           owner.get("pid"), source_commit)) from None
                record = dict(owner)
            else:
                record = {"schema": LEASE_SCHEMA, "mode": "qualify", "source": source_commit,
                          "purpose": purpose, "pid": os.getpid(), "host": socket.gethostname(),
                          "since_utc": utc_now()}
                write_json_atomic(owner_path, record)
                # The guard is held, so no other acquirer observes the conversion window.
                _flock(descriptor, fcntl.LOCK_SH)
        except BaseException:
            os.close(descriptor)
            raise
        return Lease(descriptor, {**record, "installation_root": str(installation_root),
                                  "holder_pid": os.getpid()})
    finally:
        os.close(guard)


def acquire_mutation_lease(installation_root: Path, *, purpose: str = "mutation") -> Lease:
    """Hold the installation exclusively: no qualification, publication or other build runs."""
    import fcntl

    installation_root = Path(installation_root).resolve()
    guard_path, lease_path, owner_path = _lease_files(installation_root)
    guard = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        _flock(guard, fcntl.LOCK_EX)
        descriptor = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            _flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            owner = _read_lease(owner_path)
            raise InstallationBusy(
                "installation %s is in use (%s of source %s, pid %s); refusing %s"
                % (installation_root, owner.get("mode"), owner.get("source"), owner.get("pid"),
                   purpose)) from None
        record = {"schema": LEASE_SCHEMA, "mode": "mutate", "source": None, "purpose": purpose,
                  "pid": os.getpid(), "host": socket.gethostname(), "since_utc": utc_now()}
        write_json_atomic(owner_path, record)
        return Lease(descriptor, {**record, "installation_root": str(installation_root),
                                  "holder_pid": os.getpid()})
    finally:
        os.close(guard)


@contextlib.contextmanager
def qualification_lease(installation_root: Path, source_commit: str, *,
                        purpose: str = "qualification") -> Iterator[Lease]:
    lease = acquire_qualification_lease(installation_root, source_commit, purpose=purpose)
    try:
        yield lease
    finally:
        lease.release()


@contextlib.contextmanager
def mutation_lease(installation_root: Path, *, purpose: str = "mutation") -> Iterator[Lease]:
    lease = acquire_mutation_lease(installation_root, purpose=purpose)
    try:
        yield lease
    finally:
        lease.release()


def inherited_mutation(installation_root: Path) -> bool:
    """Whether an ancestor (``sdk_cohort.py mutate``) holds this installation exclusively.

    The ancestor names its lease in ``POPS_SDK_LEASE``; the lock itself must also be held
    exclusively, which a non-blocking shared probe proves without ever waiting.
    """
    try:
        record = json.loads(os.environ.get("POPS_SDK_LEASE", ""))
    except json.JSONDecodeError:
        return False
    if not isinstance(record, dict) or record.get("mode") != "mutate" \
            or record.get("installation_root") != str(Path(installation_root).resolve()):
        return False
    import fcntl

    _guard, lease_path, _owner = _lease_files(installation_root)
    probe = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        _flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(probe)
    return False


@contextlib.contextmanager
def installation_mutation(installation_root: Path, *, purpose: str) -> Iterator[Lease | None]:
    """Mutate under the exclusive lease, taking it unless an ancestor already holds it."""
    if inherited_mutation(installation_root):
        yield None
        return
    with mutation_lease(installation_root, purpose=purpose) as lease:
        yield lease


def _run_held(lease: Lease, command: Sequence[str], environment: Mapping[str, str]) -> int:
    # Only this process holds the lock.  Handing it to the command would let any daemon the
    # command leaves behind (a compiler cache helper, an MPI launcher) keep the installation leased.
    with lease:
        return subprocess.run(list(command), env=dict(environment), check=False).returncode


# --- command line ------------------------------------------------------------------------------


def _print(value: Any) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    identify = commands.add_parser("identify")
    identify.add_argument("--source", type=Path)
    identify.add_argument("--package", type=Path)
    identify.add_argument("--coherence", action="store_true")
    publish = commands.add_parser("publish")
    publish.add_argument("--source", type=Path, required=True)
    publish.add_argument("--package", type=Path, required=True)
    publish.add_argument("--build-receipt", type=Path)
    publish.add_argument("--seal", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--package", type=Path)
    verify.add_argument("--source", type=Path)
    hold = commands.add_parser("hold")
    hold.add_argument("--installation", type=Path, required=True)
    hold.add_argument("--source", type=Path, required=True)
    hold.add_argument("argv", nargs=argparse.REMAINDER)
    mutate = commands.add_parser("mutate")
    mutate.add_argument("--installation", type=Path, required=True)
    mutate.add_argument("--purpose", default="build or install")
    mutate.add_argument("argv", nargs=argparse.REMAINDER)
    held = commands.add_parser("held")
    held.add_argument("--installation", type=Path, required=True)
    attest = commands.add_parser("attest-build")
    attest.add_argument("--source", type=Path, required=True)
    attest.add_argument("--package", type=Path, required=True)
    attest.add_argument("--dimension", type=int, required=True, choices=(1, 2, 3))
    attest.add_argument("--wheel", type=Path)
    attest.add_argument("argv", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "identify":
            value: dict[str, Any] = {}
            if arguments.source is not None:
                value["source"] = source_identity(arguments.source)
            if arguments.package is not None or arguments.source is None:
                value["sdk"] = sdk_identity(arguments.package)
            if arguments.coherence:
                if arguments.source is None:
                    parser.error("--coherence requires --source")
                value["coherence"] = coherence(arguments.source, arguments.package)
            _print(value)
            return 0
        if arguments.command == "publish":
            _print(publish_cohort(arguments.source, arguments.package,
                                  build_receipt=arguments.build_receipt, seal=arguments.seal))
            return 0
        if arguments.command == "verify":
            result = verify_cohort(arguments.package, source_root=arguments.source)
            _print(result)
            return 0 if result["verified"] else 1
        if arguments.command == "held":
            if os.name != "posix":
                return 0  # no lease exists on this platform; ``mutate`` says so once
            return 0 if inherited_mutation(arguments.installation) else 1
        command = list(arguments.argv)
        if command and command[0] == "--":
            command = command[1:]
        if arguments.command == "attest-build":
            record = attest_native_build(arguments.source, arguments.package,
                                         arguments.dimension, wheel=arguments.wheel,
                                         command=command)
            _print({key: record[key] for key in ("dimension", "leaf_sha256", "attested_utc")}
                   | {"source": record["source"]["commit"],
                      "sdk_inputs_clean": record["source"]["sdk_inputs_clean"]})
            return 0
        if not command:
            parser.error("%s requires a command after --" % arguments.command)
        if os.name != "posix":
            if arguments.command == "hold":
                raise CohortError("the installation lease requires POSIX flock")
            print("sdk-cohort: no POSIX flock on this platform; %s runs without the "
                  "installation lease" % arguments.purpose, file=sys.stderr)
            return subprocess.run(command, check=False).returncode
        if arguments.command == "hold":
            source = source_identity(arguments.source)
            require_clean_source(source)
            lease = acquire_qualification_lease(arguments.installation, source["commit"],
                                                purpose="hold " + " ".join(command[:3]))
            environment = dict(os.environ, POPS_SDK_LEASE=json.dumps(lease.record))
            return _run_held(lease, command, environment)
        lease = acquire_mutation_lease(arguments.installation, purpose=arguments.purpose)
        try:
            refuse_published_installation(Path(lease.record["installation_root"]),
                                          arguments.purpose)
        except CohortError:
            lease.release()
            raise
        environment = dict(os.environ, POPS_SDK_LEASE=json.dumps(lease.record))
        return _run_held(lease, command, environment)
    except CohortError as exc:
        print("sdk-cohort: %s" % exc, file=sys.stderr)
        return 3 if isinstance(exc, InstallationBusy) else 2


if __name__ == "__main__":
    raise SystemExit(main())
