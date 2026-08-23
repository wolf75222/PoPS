#!/usr/bin/env python3
"""Create and verify a content-addressed export of the Git-visible worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath
from typing import Any

from common import load_campaign, route_requires_mpi, route_uses_gpu


MANIFEST_SCHEMA = "pops.performance.source-export.v1"
KOKKOS_EXPORT_SCHEMA = "pops.performance.kokkos-source-export.v1"
BUILD_RECEIPT_SCHEMA = "pops.performance.advection-sine.build-receipt.v3"
COMPLETE_RECEIPT_SCHEMA = "pops.performance.advection-sine.complete.v2"


class ExportError(RuntimeError):
    """The requested export is incomplete, unsafe, or no longer authentic."""


def _run_git(source: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", errors="replace").strip()
        raise ExportError(f"git {' '.join(arguments)} failed: {detail or error}") from error


def _git_toplevel(source: Path) -> Path:
    source = source.resolve(strict=True)
    top_level = Path(_run_git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top_level != source:
        raise ExportError("--source must be the repository toplevel")
    return source


def _git_head(source: Path) -> str:
    commit = _run_git(source, "rev-parse", "HEAD^{commit}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ExportError("Git did not return a lowercase 40-hex revision")
    return commit


def _tracked_git_dirty(source: Path) -> bool:
    """Return only tracked-index/worktree dirtiness; untracked exports are irrelevant."""
    return bool(_run_git(source, "status", "--porcelain", "--untracked-files=no").strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one receipt exactly once; evidence must never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ExportError(f"refusing to overwrite published receipt {path}") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _relative_file(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ExportError(f"{label} escapes its authenticated root: {path}") from error
    safe = _safe_relative(relative.as_posix())
    return safe.as_posix()


def _cmake_cache(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise ExportError(f"cannot read CMake cache {path}: {error}") from error
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if ":" not in left:
            continue
        key, _kind = left.split(":", 1)
        values[key] = value
    return values


def _cache_requires(cache: dict[str, str], key: str, expected: str) -> str:
    observed = cache.get(key)
    if observed != expected:
        raise ExportError(f"CMake cache requires {key}={expected}, found {observed!r}")
    return observed


def _command_receipt(executable: str, label: str) -> dict[str, str]:
    resolved = shutil.which(executable) if not os.path.isabs(executable) else executable
    if resolved is None or not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        raise ExportError(f"{label} executable is unavailable: {executable}")
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ExportError(f"cannot authenticate {label} version: {error}") from error
    output = (completed.stdout or completed.stderr).strip()
    if not output:
        raise ExportError(f"{label} version command returned no provenance")
    return {
        "path": str(Path(resolved).resolve()),
        "sha256": _sha256_file(Path(resolved)),
        "version": output.splitlines()[0],
    }


def _single_kokkos_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise ExportError(f"expected exactly one {name} below Kokkos root, found {len(matches)}")
    return matches[0]


def _kokkos_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config = _single_kokkos_file(root, "KokkosConfigCommon.cmake")
    header = _single_kokkos_file(root, "Kokkos_Core.hpp")
    version_file = _single_kokkos_file(root, "KokkosConfigVersion.cmake")
    text = version_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'PACKAGE_VERSION\s+"([^"]+)"', text)
    if match is None:
        raise ExportError("KokkosConfigVersion.cmake does not declare PACKAGE_VERSION")
    return {
        "version": match.group(1),
        "config": {
            "path": _relative_file(config, root, "Kokkos configuration"),
            "sha256": _sha256_file(config),
        },
        "header": {
            "path": _relative_file(header, root, "Kokkos header"),
            "sha256": _sha256_file(header),
        },
        "version_file": {
            "path": _relative_file(version_file, root, "Kokkos version file"),
            "sha256": _sha256_file(version_file),
        },
    }


def _archive_kokkos_entries(archive: Path) -> list[dict[str, Any]]:
    """Return the regular-file/symlink inventory of one authenticated Git tar."""
    if archive.is_symlink() or not archive.is_file():
        raise ExportError(f"Kokkos archive must be one regular file: {archive}")
    try:
        with tarfile.open(archive, "r:") as bundle:
            entries: list[dict[str, Any]] = []
            for member in bundle.getmembers():
                relative = _safe_relative(member.name)
                if member.isdir():
                    continue
                if member.isfile():
                    stream = bundle.extractfile(member)
                    if stream is None:
                        raise ExportError(f"cannot read Kokkos archive member {member.name}")
                    digest = hashlib.sha256(stream.read()).hexdigest()
                    entries.append(
                        {
                            "path": relative.as_posix(),
                            "type": "file",
                            "mode": stat.S_IMODE(member.mode),
                            "sha256": digest,
                        }
                    )
                elif member.issym():
                    target = member.linkname
                    entries.append(
                        {
                            "path": relative.as_posix(),
                            "type": "symlink",
                            "mode": stat.S_IMODE(member.mode),
                            "sha256": hashlib.sha256(
                                target.encode("utf-8", errors="surrogateescape")
                            ).hexdigest(),
                            "target": target,
                        }
                    )
                else:
                    raise ExportError(
                        f"unsupported Kokkos archive member type for {member.name}"
                    )
    except (OSError, tarfile.TarError) as error:
        raise ExportError(f"cannot read Kokkos Git archive {archive}: {error}") from error
    if not entries:
        raise ExportError("refusing an empty Kokkos Git archive")
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ExportError("Kokkos Git archive has duplicate file paths")
    return sorted(entries, key=lambda entry: entry["path"])


def _verify_extracted_kokkos_source(source: Path, archive: Path) -> str:
    """Prove the CMake source directory is exactly the authenticated Git archive."""
    raw_source = source
    source = source.resolve(strict=True)
    if raw_source.is_symlink() or not source.is_dir():
        raise ExportError("Kokkos CMake source must be a real extracted directory")
    expected = _archive_kokkos_entries(archive)
    observed: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = _relative_file(path, source, "Kokkos extracted source")
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            observed.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "sha256": _sha256_file(path),
                }
            )
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            observed.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": mode,
                    "sha256": hashlib.sha256(
                        target.encode("utf-8", errors="surrogateescape")
                    ).hexdigest(),
                    "target": target,
                }
            )
        else:
            raise ExportError(f"Kokkos extracted source has unsupported path: {relative}")
    if sorted(observed, key=lambda entry: entry["path"]) != expected:
        raise ExportError("Kokkos CMake source differs from its authenticated extracted Git archive")
    return _tree_digest(expected)


def create_kokkos_export(source: Path, archive: Path, receipt_path: Path) -> dict[str, Any]:
    """Create a deterministic, committed Kokkos Git archive and immutable receipt."""
    source = _git_toplevel(source)
    archive = archive.absolute()
    receipt_path = receipt_path.absolute()
    if archive.exists() or archive.is_symlink():
        raise ExportError(f"refusing to overwrite published Kokkos archive {archive}")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ExportError(f"refusing to overwrite published receipt {receipt_path}")
    if _tracked_git_dirty(source):
        raise ExportError("refusing Kokkos export from a tracked-dirty Git worktree")
    commit = _git_head(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    try:
        subprocess.run(
            ["git", "-C", str(source), "archive", "--format=tar", f"--output={temporary}", commit],
            check=True,
            capture_output=True,
        )
        try:
            os.link(temporary, archive)
        except FileExistsError as error:
            raise ExportError(f"refusing to overwrite published Kokkos archive {archive}") from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", errors="replace").strip()
        raise ExportError(f"cannot create deterministic Kokkos Git archive: {detail or error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    if _tracked_git_dirty(source) or _git_head(source) != commit:
        raise ExportError("Kokkos Git source changed while its archive was being prepared")
    receipt = {
        "schema": KOKKOS_EXPORT_SCHEMA,
        "kind": "git-export",
        "commit": commit,
        "source_toplevel": str(source),
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_format": "git-archive-tar",
    }
    _write_new_json(receipt_path, receipt)
    return receipt


def verify_kokkos_export(archive: Path, receipt_path: Path) -> dict[str, Any]:
    """Authenticate a Kokkos Git archive solely from its immutable receipt."""
    raw_archive = archive
    archive = archive.resolve(strict=True)
    if raw_archive.is_symlink() or not archive.is_file():
        raise ExportError(f"Kokkos archive must be one regular file: {archive}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read Kokkos export receipt: {error}") from error
    if (
        type(receipt) is not dict
        or receipt.get("schema") != KOKKOS_EXPORT_SCHEMA
        or receipt.get("kind") != "git-export"
        or not isinstance(receipt.get("source_toplevel"), str)
        or not isinstance(receipt.get("archive"), str)
        or not isinstance(receipt.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", receipt["commit"]) is None
        or not isinstance(receipt.get("archive_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", receipt["archive_sha256"]) is None
        or receipt.get("archive_format") != "git-archive-tar"
        or type(receipt.get("archive_bytes")) is not int
        or receipt["archive_bytes"] <= 0
    ):
        raise ExportError("Kokkos export receipt has an unsupported schema or identity")
    if Path(receipt["archive"]).resolve() != archive:
        raise ExportError("Kokkos export receipt names a different archive")
    if receipt["archive_bytes"] != archive.stat().st_size or receipt["archive_sha256"] != _sha256_file(
        archive
    ):
        raise ExportError("Kokkos archive differs from its authenticated export receipt")
    _archive_kokkos_entries(archive)
    return receipt


def _native_import_receipt(
    source: Path, build: Path, python: Path, dimension: int
) -> dict[str, Any]:
    package = build / "python" / "pops" / "_native" / f"dim{dimension}"
    candidates = sorted(
        path
        for suffix in EXTENSION_SUFFIXES
        for path in package.glob(f"_pops{suffix}")
        if path.is_file()
    )
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ExportError(
            f"expected exactly one build-tree pops._pops extension, found {len(candidates)}"
        )
    extension = candidates[0].resolve(strict=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(build / "python"), str(source)))
    environment["POPS_NATIVE_VARIANTS_ROOT"] = str(build / "python" / "pops" / "_native")
    probe = (
        "import json, pathlib, pops, sys; "
        "from pops._native_selector import select_native_dimension; "
        f"native=select_native_dimension({dimension}); "
        "print(json.dumps({'python':sys.executable,'pops':str(pathlib.Path(pops.__file__).resolve()),"
        "'extension':str(pathlib.Path(native.__file__).resolve()),"
        "'dimension':native.__native_dimension__,'has_mpi':native.__has_mpi__,"
        "'has_kokkos':native.__has_kokkos__,"
        "'build_fingerprint':native.__build_fingerprint__},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=90,
        )
        imported = json.loads(completed.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        stderr = getattr(error, "stderr", "")
        raise ExportError(
            f"selected Python cannot import the built pops._pops extension: {stderr or error}"
        ) from error
    if imported.get("extension") != str(extension):
        raise ExportError("selected Python imported a different native extension")
    if imported.get("dimension") != dimension or imported.get("has_kokkos") is not True:
        raise ExportError(
            "imported native extension does not satisfy the requested dimension/Kokkos route"
        )
    build_fingerprint = imported.get("build_fingerprint")
    if (
        type(build_fingerprint) is not str
        or len(build_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in build_fingerprint)
    ):
        raise ExportError("imported native extension has no exact build fingerprint")
    package_path = Path(str(imported.get("pops", "")))
    if (
        _relative_file(package_path, source, "imported pops Python package")
        != "python/pops/__init__.py"
    ):
        raise ExportError("selected Python did not import the authenticated source package")
    return {
        "python": {
            "path": str(python.resolve(strict=True)),
            "version": sys.version.splitlines()[0]
            if python.resolve() == Path(sys.executable).resolve()
            else None,
        },
        "extension": {
            "path": _relative_file(extension, build, "native extension"),
            "sha256": _sha256_file(extension),
            "imported_path": _relative_file(
                Path(imported["extension"]), build, "imported native extension"
            ),
            "dimension": imported["dimension"],
            "build_fingerprint": build_fingerprint,
            "has_mpi": imported["has_mpi"],
            "has_kokkos": imported["has_kokkos"],
        },
    }


def _safe_relative(raw: str) -> PurePosixPath:
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ExportError(f"unsafe Git path {raw!r}")
    return relative


def _entry(source: Path, relative: PurePosixPath) -> dict[str, Any]:
    path = source.joinpath(*relative.parts)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ExportError(f"Git-visible path disappeared during export: {relative}") from error
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISREG(metadata.st_mode):
        return {
            "path": relative.as_posix(),
            "type": "file",
            "mode": mode,
            "size": metadata.st_size,
            "sha256": _sha256_file(path),
        }
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        encoded = target.encode("utf-8", errors="surrogateescape")
        return {
            "path": relative.as_posix(),
            "type": "symlink",
            "mode": mode,
            "size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "target": target,
        }
    raise ExportError(f"unsupported Git-visible path type: {relative}")


def _git_entries(source: Path) -> list[dict[str, Any]]:
    raw_paths = _run_git(source, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = [
        name for name in raw_paths.decode("utf-8", errors="surrogateescape").split("\0") if name
    ]
    entries: list[dict[str, Any]] = []
    for name in sorted(names):
        relative = _safe_relative(name)
        path = source.joinpath(*relative.parts)
        if not os.path.lexists(path):
            # A tracked deletion is part of the dirty state but not the exported tree.
            continue
        entries.append(_entry(source, relative))
    if not entries:
        raise ExportError("refusing an empty source export")
    return entries


def _tree_digest(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8", errors="surrogateescape"
    )
    return hashlib.sha256(encoded).hexdigest()


def _tar_info(entry: dict[str, Any]) -> tarfile.TarInfo:
    info = tarfile.TarInfo(entry["path"])
    info.mode = entry["mode"]
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if entry["type"] == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = entry["target"]
        info.size = 0
    else:
        info.size = entry["size"]
    return info


def create_export(source: Path, archive: Path, manifest_path: Path) -> dict[str, Any]:
    source = source.resolve()
    if Path(_run_git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve() != source:
        raise ExportError("--source must be the repository toplevel")
    base_sha = _run_git(source, "rev-parse", "HEAD^{commit}").decode().strip()
    if len(base_sha) != 40 or any(character not in "0123456789abcdef" for character in base_sha):
        raise ExportError("Git did not return a lowercase 40-hex base revision")
    dirty = bool(_run_git(source, "status", "--porcelain", "--untracked-files=normal").strip())
    if dirty:
        raise ExportError(
            "refusing performance source export from a dirty Git worktree; "
            "commit or remove every tracked and untracked change first"
        )
    entries = _git_entries(source)

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as bundle:
            for entry in entries:
                path = source.joinpath(*PurePosixPath(entry["path"]).parts)
                info = _tar_info(entry)
                if entry["type"] == "file":
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
                else:
                    bundle.addfile(info)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)

    dirty_after = bool(
        _run_git(source, "status", "--porcelain", "--untracked-files=normal").strip()
    )
    if dirty_after or dirty_after != dirty or _git_entries(source) != entries:
        raise ExportError("worktree changed while its source archive was being prepared")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_sha": base_sha,
        "source_dirty": dirty,
        "tree_sha256": _tree_digest(entries),
        "archive_sha256": _sha256_file(archive),
        "archive_format": "tar-pax",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read export manifest {path}: {error}") from error
    if type(manifest) is not dict or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ExportError("unexpected export-manifest schema")
    entries = manifest.get("files")
    if type(entries) is not list or manifest.get("file_count") != len(entries):
        raise ExportError("manifest file inventory is incomplete")
    if manifest.get("tree_sha256") != _tree_digest(entries):
        raise ExportError("manifest tree digest is internally inconsistent")
    if manifest.get("source_dirty") is not False:
        raise ExportError(
            "performance evidence requires a clean source manifest (source_dirty=false)"
        )
    return manifest


def verify_tree(
    source: Path,
    manifest_path: Path,
    *,
    expected_base_sha: str | None = None,
    expected_tree_sha256: str | None = None,
    expected_dirty: int | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    manifest = _load_manifest(manifest_path)
    if expected_base_sha is not None and manifest.get("base_sha") != expected_base_sha:
        raise ExportError("manifest base revision differs from the submitted value")
    if expected_tree_sha256 is not None and manifest.get("tree_sha256") != expected_tree_sha256:
        raise ExportError("manifest tree digest differs from the submitted value")
    if expected_dirty is not None and manifest.get("source_dirty") is not bool(expected_dirty):
        raise ExportError("manifest dirty state differs from the submitted value")
    expected = manifest["files"]
    observed_paths = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_symlink() or not path.is_dir()
    )
    expected_paths = [entry["path"] for entry in expected]
    if observed_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(observed_paths))[:5]
        extra = sorted(set(observed_paths) - set(expected_paths))[:5]
        raise ExportError(f"exported tree inventory mismatch; missing={missing}, extra={extra}")
    observed = [_entry(source, _safe_relative(entry["path"])) for entry in expected]
    if observed != expected or _tree_digest(observed) != manifest["tree_sha256"]:
        raise ExportError("exported tree content or mode differs from its manifest")
    return manifest


def _campaign_source_entry(manifest: dict[str, Any], relative: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["files"] if entry.get("path") == relative]
    if len(matches) != 1 or matches[0].get("type") != "file":
        raise ExportError(f"campaign {relative} is absent from the authenticated source manifest")
    return matches[0]


def _normalized_campaign_sha256(campaign: dict[str, Any]) -> str:
    return _sha256_text(json.dumps(campaign, separators=(",", ":"), sort_keys=True))


def _under_root(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as error:
        raise ExportError(f"{label} must resolve below the authenticated Kokkos install root") from error


def _installed_kokkos_core(root: Path, *, require_static: bool) -> dict[str, Any]:
    def accepted(path: Path) -> bool:
        name = path.name
        if require_static:
            return name == "libkokkoscore.a"
        return bool(
            name == "libkokkoscore.a"
            or re.fullmatch(r"libkokkoscore(?:\.[0-9]+)*\.dylib", name)
            or re.fullmatch(r"libkokkoscore\.so(?:\.[0-9]+)*", name)
        )

    candidates = sorted(
        path for path in root.rglob("libkokkoscore*")
        if path.is_file() and not path.is_symlink() and accepted(path)
    )
    if len(candidates) != 1:
        expected = "static libkokkoscore.a" if require_static else "canonical Kokkos core library"
        raise ExportError(
            f"expected exactly one installed {expected} below Kokkos root, found {len(candidates)}"
        )
    core = candidates[0].resolve(strict=True)
    return {
        "kind": "static-archive" if core.suffix == ".a" else "shared-library",
        "path": _under_root(core, root, "Kokkos core library"),
        "sha256": _sha256_file(core),
    }


def _kokkos_route_configuration(cache: dict[str, str], route: str) -> dict[str, str]:
    required = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "Kokkos_ENABLE_SERIAL": "ON",
    }
    if route in {"kokkos_serial", "kokkos_openmp", "kokkos_openmp_mpi"}:
        required.update({"Kokkos_ENABLE_CUDA": "OFF"})
        required["Kokkos_ENABLE_OPENMP"] = "ON" if "openmp" in route else "OFF"
    elif route in {"kokkos_cuda", "kokkos_cuda_mpi"}:
        required.update(
            {
                "Kokkos_ENABLE_CUDA": "ON",
                "Kokkos_ENABLE_CUDA_LAMBDA": "ON",
                "Kokkos_ARCH_HOPPER90": "ON",
                "Kokkos_ENABLE_OPENMP": "OFF",
            }
        )
    else:
        raise ExportError(f"unsupported Kokkos campaign route {route}")
    return {key: _cache_requires(cache, key, value) for key, value in required.items()}


def _kokkos_build_authority(
    *,
    source_receipt: Path | None,
    kokkos_build: Path | None,
    kokkos_root: Path,
    route: str,
    main_cache: dict[str, str],
) -> dict[str, Any]:
    """Authenticate installed Kokkos and, on ROMEO, its Git-built source semantics."""
    kokkos_root = kokkos_root.resolve(strict=True)
    if (source_receipt is None) != (kokkos_build is None):
        raise ExportError("Kokkos source receipt and Kokkos build must be provided together")
    installed = _kokkos_receipt(kokkos_root)
    installed["libkokkoscore"] = _installed_kokkos_core(
        kokkos_root, require_static=source_receipt is not None
    )
    configured_dir = main_cache.get("Kokkos_DIR")
    if not configured_dir:
        raise ExportError("PoPS CMake cache lacks Kokkos_DIR")
    installed["cmake_dir"] = {
        "path": _under_root(Path(configured_dir), kokkos_root, "PoPS Kokkos_DIR")
    }
    if source_receipt is None:
        installed["source_authority"] = {"kind": "installed-distribution"}
        return installed

    kokkos_build = kokkos_build.resolve(strict=True)
    if kokkos_build.is_symlink() or not kokkos_build.is_dir():
        raise ExportError("Kokkos build must be a real CMake build directory")
    build_cache_path = kokkos_build / "CMakeCache.txt"
    build_cache = _cmake_cache(build_cache_path)
    source_directory = build_cache.get("CMAKE_HOME_DIRECTORY")
    if not source_directory:
        raise ExportError("Kokkos CMake cache lacks CMAKE_HOME_DIRECTORY")
    source_directory_path = Path(source_directory)
    source_receipt = source_receipt.resolve(strict=True)
    # The archive path is deliberately sibling-named: it prevents a receipt
    # from authorising arbitrary source bytes without also naming their tar.
    try:
        source_payload = json.loads(source_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read Kokkos export receipt: {error}") from error
    archive_value = source_payload.get("archive") if type(source_payload) is dict else None
    if type(archive_value) is not str:
        raise ExportError("Kokkos export receipt lacks its archive path identity")
    archive = Path(archive_value).resolve(strict=True)
    source_export = verify_kokkos_export(archive, source_receipt)
    source_tree_sha256 = _verify_extracted_kokkos_source(source_directory_path, archive)
    configured = _kokkos_route_configuration(build_cache, route)
    installed["source_authority"] = {
        "kind": "git-export",
        "schema": source_export["schema"],
        "commit": source_export["commit"],
        "archive_sha256": source_export["archive_sha256"],
        "archive_tree_sha256": source_tree_sha256,
    }
    installed["build"] = {
        "cache": {
            "path": _relative_file(build_cache_path, kokkos_build, "Kokkos CMake cache"),
            "sha256": _sha256_file(build_cache_path),
        },
        "source_tree_sha256": source_tree_sha256,
        "configured": configured,
    }
    return installed


def create_build_receipt(
    source: Path,
    build: Path,
    manifest_path: Path,
    campaign_path: Path,
    python: Path,
    kokkos_root: Path,
    output: Path,
    kokkos_source_receipt: Path | None = None,
    kokkos_build: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the exact native Python module before a campaign can start."""
    source = source.resolve(strict=True)
    build = build.resolve(strict=True)
    python = python.resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ExportError(f"Python interpreter is not executable: {python}")
    cache_dir = os.environ.get("POPS_CACHE_DIR")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if not cache_dir or not xdg_cache:
        raise ExportError("build receipt requires explicit POPS_CACHE_DIR and XDG_CACHE_HOME")
    manifest = verify_tree(source, manifest_path.resolve(strict=True))
    include_root = source / "include"
    configured_include = os.environ.get("POPS_INCLUDE")
    if (
        not include_root.is_dir()
        or not configured_include
        or Path(configured_include).resolve() != include_root.resolve()
    ):
        raise ExportError("POPS_INCLUDE must resolve to the authenticated source/include directory")
    include_entries = [
        entry for entry in manifest["files"] if entry.get("path", "").startswith("include/")
    ]
    if not include_entries:
        raise ExportError("authenticated source manifest contains no include inventory")
    campaign_relative = _relative_file(campaign_path, source, "campaign")
    campaign_entry = _campaign_source_entry(manifest, campaign_relative)
    campaign = load_campaign(campaign_path)
    cache_path = build / "CMakeCache.txt"
    cache = _cmake_cache(cache_path)
    configured_source = cache.get("CMAKE_HOME_DIRECTORY")
    if not configured_source or Path(configured_source).resolve() != source:
        raise ExportError(
            "CMake build tree was not configured from the authenticated exported source tree"
        )
    expected_mpi = "ON" if route_requires_mpi(campaign["route"]) else "OFF"
    configured = {
        key: _cache_requires(cache, key, expected)
        for key, expected in {
            "CMAKE_BUILD_TYPE": "Release",
            "POPS_BUILD_PYTHON": "ON",
            "POPS_USE_KOKKOS": "ON",
            "POPS_USE_MPI": expected_mpi,
            # This benchmark has no I/O workload.  MPI must never silently pull HDF5 in.
            "POPS_USE_HDF5": "OFF",
            "POPS_NATIVE_DIM": str(campaign["dimension"]),
        }.items()
    }
    compiler_path = cache.get("CMAKE_CXX_COMPILER")
    if not compiler_path:
        raise ExportError("CMake cache lacks CMAKE_CXX_COMPILER")
    native = _native_import_receipt(source, build, python, campaign["dimension"])
    if native["extension"]["has_mpi"] is not route_requires_mpi(campaign["route"]):
        raise ExportError("imported native extension MPI fact differs from the campaign route")
    imported_python = native["python"]
    probe = subprocess.run(
        [str(python), "-c", "import sys; print(sys.version.splitlines()[0])"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    imported_python["version"] = probe.stdout.strip()
    mpi: dict[str, Any] = {"enabled": route_requires_mpi(campaign["route"])}
    if mpi["enabled"]:
        mpi["launcher"] = _command_receipt("srun", "SLURM srun launcher")
        mpi["openmpi_launcher"] = _command_receipt("mpirun", "OpenMPI launcher")
        mpi_compiler = cache.get("MPI_CXX_COMPILER")
        if mpi_compiler:
            mpi["compiler"] = _command_receipt(mpi_compiler, "MPI C++ compiler")
    cuda: dict[str, Any] = {"enabled": route_uses_gpu(campaign["route"])}
    if cuda["enabled"]:
        cuda["compiler"] = _command_receipt("nvcc", "CUDA compiler")
    receipt = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "workload": "public-python",
        "build_type": "Release",
        "source": {
            "base_sha": manifest["base_sha"],
            "tree_sha256": manifest["tree_sha256"],
            "manifest_sha256": _sha256_file(manifest_path),
            "include": {
                "path": "include",
                "inventory_sha256": _tree_digest(include_entries),
                "file_count": len(include_entries),
            },
        },
        "campaign": {
            "id": campaign["id"],
            "route": campaign["route"],
            "dimension": campaign["dimension"],
            "path": campaign_relative,
            "sha256": campaign_entry["sha256"],
            "normalized_sha256": _normalized_campaign_sha256(campaign),
        },
        "cmake": {
            "cache": {
                "path": _relative_file(cache_path, build, "CMake cache"),
                "sha256": _sha256_file(cache_path),
            },
            "configured": configured,
        },
        "native_import": native,
        "runtime_cache": {
            "pops_cache_dir": str(Path(cache_dir).resolve()),
            "xdg_cache_home": str(Path(xdg_cache).resolve()),
        },
        "compiler": _command_receipt(compiler_path, "C++ compiler"),
        "kokkos": _kokkos_build_authority(
            source_receipt=kokkos_source_receipt,
            kokkos_build=kokkos_build,
            kokkos_root=kokkos_root,
            route=campaign["route"],
            main_cache=cache,
        ),
        "mpi": mpi,
        "cuda": cuda,
    }
    _write_new_json(output, receipt)
    return receipt


def _receipt_inventory(root: Path) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = _relative_file(path, root, "published evidence")
        if path.is_symlink() or not path.is_file():
            raise ExportError(f"published evidence must be a regular file: {relative}")
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not entries:
        raise ExportError(f"refusing an empty published-evidence inventory below {root}")
    return entries


def _inventory_digest(entries: list[dict[str, Any]]) -> str:
    return _sha256_text(json.dumps(entries, separators=(",", ":"), sort_keys=True))


def create_complete_receipt(
    raw: Path,
    report: Path,
    campaign: str,
    slurm_job_id: str,
    source_tree_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Seal raw inputs and collected summaries before an immutable publication."""
    if not re.fullmatch(r"[1-9][0-9]*", slurm_job_id):
        raise ExportError("SLURM job id must be positive decimal text")
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ExportError("source tree SHA-256 must be lowercase hexadecimal")
    raw = raw.resolve(strict=True)
    report = report.resolve(strict=True)
    source = _load_manifest(raw / "source.manifest.json")
    try:
        build = json.loads((raw / "build.receipt.json").read_text(encoding="utf-8"))
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read raw/build/summary evidence: {error}") from error
    if (
        type(build) is not dict
        or type(summary) is not dict
        or type(build.get("source")) is not dict
        or type(build.get("campaign")) is not dict
        or type(summary.get("source_manifest")) is not dict
        or source.get("source_dirty") is not False
        or source.get("tree_sha256") != source_tree_sha256
        or build.get("schema") != BUILD_RECEIPT_SCHEMA
        or build.get("source", {}).get("tree_sha256") != source_tree_sha256
        or build.get("campaign", {}).get("id") != campaign
        or summary.get("campaign") != campaign
        or summary.get("source_manifest", {}).get("source_dirty") is not False
        or summary.get("source_manifest") != source
        or summary.get("build_receipt") != build
        or not (report / "measurements.csv").is_file()
    ):
        raise ExportError("raw/build/summary provenance cannot be sealed into COMPLETE")
    raw_entries = _receipt_inventory(raw)
    report_entries = _receipt_inventory(report)
    receipt = {
        "schema": COMPLETE_RECEIPT_SCHEMA,
        "campaign": campaign,
        "slurm_job_id": slurm_job_id,
        "source_tree_sha256": source_tree_sha256,
        "raw": {"sha256": _inventory_digest(raw_entries), "files": raw_entries},
        "report": {"sha256": _inventory_digest(report_entries), "files": report_entries},
    }
    _write_new_json(output, receipt)
    return receipt


def verify_complete_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    try:
        receipt = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read COMPLETE receipt: {error}") from error
    if type(receipt) is not dict or receipt.get("schema") != COMPLETE_RECEIPT_SCHEMA:
        raise ExportError("unsupported COMPLETE receipt schema")
    source = _load_manifest(root / "raw" / "source.manifest.json")
    if receipt.get("source_tree_sha256") != source.get("tree_sha256"):
        raise ExportError("COMPLETE source tree differs from the clean source manifest")
    try:
        build = json.loads((root / "raw" / "build.receipt.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "report" / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read COMPLETE build/summary provenance: {error}") from error
    summary_source = summary.get("source_manifest") if type(summary) is dict else None
    if (
        type(summary_source) is not dict
        or summary_source.get("source_dirty") is not False
        or summary_source != source
        or type(build) is not dict
        or type(build.get("source")) is not dict
        or type(build.get("campaign")) is not dict
        or build.get("schema") != BUILD_RECEIPT_SCHEMA
        or build.get("source", {}).get("tree_sha256") != source.get("tree_sha256")
        or build.get("campaign", {}).get("id") != receipt.get("campaign")
        or summary.get("campaign") != receipt.get("campaign")
        or summary.get("build_receipt") != build
    ):
        raise ExportError("COMPLETE summary is not bound to the clean source/build manifest")
    for name in ("raw", "report"):
        expected = receipt.get(name)
        if type(expected) is not dict or type(expected.get("files")) is not list:
            raise ExportError(f"COMPLETE receipt lacks {name} inventory")
        observed = _receipt_inventory(root / name)
        if observed != expected["files"] or _inventory_digest(observed) != expected.get("sha256"):
            raise ExportError(f"COMPLETE receipt {name} inventory differs after publication")
    return receipt


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="write a deterministic tar and manifest")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--archive", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create_kokkos = subparsers.add_parser(
        "create-kokkos-export", help="write an immutable committed Kokkos Git archive"
    )
    create_kokkos.add_argument("--source", type=Path, required=True)
    create_kokkos.add_argument("--archive", type=Path, required=True)
    create_kokkos.add_argument("--receipt", type=Path, required=True)
    verify_kokkos = subparsers.add_parser(
        "verify-kokkos-export", help="authenticate one Kokkos Git archive"
    )
    verify_kokkos.add_argument("--archive", type=Path, required=True)
    verify_kokkos.add_argument("--receipt", type=Path, required=True)
    verify = subparsers.add_parser("verify-tree", help="authenticate an extracted tree")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expect-base-sha")
    verify.add_argument("--expect-tree-sha256")
    verify.add_argument("--expect-dirty", type=int, choices=(0, 1))
    receipt = subparsers.add_parser(
        "build-receipt", help="authenticate the imported build-tree native extension"
    )
    receipt.add_argument("--source", type=Path, required=True)
    receipt.add_argument("--build", type=Path, required=True)
    receipt.add_argument("--manifest", type=Path, required=True)
    receipt.add_argument("--campaign", type=Path, required=True)
    receipt.add_argument("--python", type=Path, required=True)
    receipt.add_argument("--kokkos-root", type=Path, required=True)
    receipt.add_argument("--kokkos-source-receipt", type=Path)
    receipt.add_argument("--kokkos-build", type=Path)
    receipt.add_argument("--output", type=Path, required=True)
    complete = subparsers.add_parser("complete-receipt", help="seal raw and summary evidence")
    complete.add_argument("--raw", type=Path, required=True)
    complete.add_argument("--report", type=Path, required=True)
    complete.add_argument("--campaign", required=True)
    complete.add_argument("--slurm-job-id", required=True)
    complete.add_argument("--source-tree-sha256", required=True)
    complete.add_argument("--output", type=Path, required=True)
    verify_complete = subparsers.add_parser(
        "verify-complete", help="verify a published COMPLETE receipt"
    )
    verify_complete.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        if args.command == "create":
            manifest = create_export(args.source, args.archive.resolve(), args.manifest.resolve())
        elif args.command == "create-kokkos-export":
            receipt = create_kokkos_export(args.source, args.archive, args.receipt)
        elif args.command == "verify-kokkos-export":
            receipt = verify_kokkos_export(args.archive, args.receipt)
        elif args.command == "verify-tree":
            manifest = verify_tree(
                args.source,
                args.manifest.resolve(),
                expected_base_sha=args.expect_base_sha,
                expected_tree_sha256=args.expect_tree_sha256,
                expected_dirty=args.expect_dirty,
            )
        elif args.command == "build-receipt":
            receipt = create_build_receipt(
                args.source,
                args.build,
                args.manifest,
                args.campaign,
                args.python,
                args.kokkos_root,
                args.output,
                args.kokkos_source_receipt,
                args.kokkos_build,
            )
        elif args.command == "complete-receipt":
            receipt = create_complete_receipt(
                args.raw,
                args.report,
                args.campaign,
                args.slurm_job_id,
                args.source_tree_sha256,
                args.output,
            )
        else:
            receipt = verify_complete_receipt(args.root)
    except ExportError as error:
        print(f"source export refused: {error}", file=sys.stderr)
        return 2
    if "manifest" in locals():
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in ("base_sha", "source_dirty", "tree_sha256", "archive_sha256")
                    if key in manifest
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps({"schema": receipt["schema"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
