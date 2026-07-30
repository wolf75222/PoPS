#!/usr/bin/env python3
"""Prove that the imported PoPS package is the exact retained release wheel."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import importlib.machinery
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote, urlparse
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROOF_SCHEMA_VERSION = 2


class InstalledWheelProofError(RuntimeError):
    """The retained wheel and the imported installation are not byte-identical."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _outside_checkout(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise InstalledWheelProofError("%s must be outside the checkout: %s" % (label, resolved))


def _direct_url_path(payload: Any) -> tuple[Path, str]:
    if not isinstance(payload, dict) or set(payload) != {"archive_info", "url"}:
        raise InstalledWheelProofError("installed distribution direct_url.json is malformed")
    archive = payload["archive_info"]
    if not isinstance(archive, dict):
        raise InstalledWheelProofError("installed distribution archive_info is malformed")
    hashes = archive.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != {"sha256"}:
        raise InstalledWheelProofError("installed distribution lacks one exact sha256 archive hash")
    digest = hashes["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise InstalledWheelProofError("installed distribution archive sha256 is malformed")
    parsed = urlparse(payload["url"])
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise InstalledWheelProofError("installed distribution did not originate from a local wheel")
    return Path(unquote(parsed.path)).resolve(), digest


def _wheel_payload_proof(
    archive: zipfile.ZipFile,
    *,
    distribution_root: Path,
) -> tuple[int, str]:
    """Authenticate every directly installed wheel member except mutable ``RECORD``."""

    rows: list[str] = []
    for name in sorted(archive.namelist()):
        if name.endswith("/") or name.endswith(".dist-info/RECORD"):
            continue
        if ".data/" in name:
            raise InstalledWheelProofError(
                "retained wheel uses an unsupported .data installation scheme"
            )
        relative = Path(name)
        installed = (distribution_root / relative).resolve()
        try:
            installed.relative_to(distribution_root)
        except ValueError as exc:
            raise InstalledWheelProofError(
                "wheel member escapes the installed distribution root: %s" % name
            ) from exc
        if not installed.is_file():
            raise InstalledWheelProofError(
                "installed distribution is missing wheel member %s" % name
            )
        wheel_digest = _sha256_bytes(archive.read(name))
        if _sha256(installed) != wheel_digest:
            raise InstalledWheelProofError(
                "installed wheel member is not byte-identical: %s" % name
            )
        rows.append("%s\0%s\n" % (name, wheel_digest))
    if not rows:
        raise InstalledWheelProofError("retained wheel has no immutable payload members")
    return len(rows), _sha256_bytes("".join(rows).encode("utf-8"))


def _installed_distribution_paths(
    distribution: importlib.metadata.Distribution,
) -> tuple[Path, Path, Path]:
    """Resolve package/native paths from distribution metadata without importing PoPS."""

    members = tuple(distribution.files or ())
    package_members = [
        member for member in members if member.as_posix() == "pops/__init__.py"
    ]
    native_members = [
        member
        for member in members
        if member.parent.as_posix() == "pops"
        and member.name.startswith("_pops.")
        and any(member.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    ]
    if len(package_members) != 1 or len(native_members) != 1:
        raise InstalledWheelProofError(
            "installed distribution lacks one unique pops package and native extension"
        )
    return (
        Path(distribution.locate_file(package_members[0])).resolve(),
        Path(distribution.locate_file(native_members[0])).resolve(),
        Path(distribution.locate_file("")).resolve(),
    )


def build_proof(
    wheel: Path,
    *,
    package_file: Path,
    native_extension: Path,
    distribution_root: Path,
    python_executable: Path,
    installed_version: str,
    direct_url: Any,
) -> dict[str, Any]:
    """Authenticate one installed distribution against one exact wheel archive."""

    retained = _outside_checkout(wheel, label="retained wheel")
    package = _outside_checkout(package_file, label="installed package")
    extension = _outside_checkout(native_extension, label="installed native extension")
    distribution = _outside_checkout(distribution_root, label="installed distribution")
    if retained.suffix != ".whl" or not retained.is_file():
        raise InstalledWheelProofError("retained wheel is not a readable .whl file")
    for label, path in (("installed package", package), ("installed native extension", extension)):
        if not path.is_file():
            raise InstalledWheelProofError("%s is not a readable file: %s" % (label, path))
    if not distribution.is_dir():
        raise InstalledWheelProofError(
            "installed distribution root is not a directory: %s" % distribution
        )
    if not isinstance(installed_version, str) or not installed_version:
        raise InstalledWheelProofError("installed distribution version is empty")

    wheel_digest = _sha256(retained)
    direct_path, direct_digest = _direct_url_path(direct_url)
    if direct_path != retained or direct_digest != wheel_digest:
        raise InstalledWheelProofError(
            "installed distribution direct URL does not authenticate the retained wheel"
        )

    try:
        with zipfile.ZipFile(retained) as archive:
            names = archive.namelist()
            native_members = [
                name
                for name in names
                if name.startswith("pops/") and Path(name).name.startswith("_pops.")
                and name.endswith((".so", ".pyd"))
            ]
            metadata_members = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(native_members) != 1:
                raise InstalledWheelProofError(
                    "retained wheel must contain exactly one pops._pops extension"
                )
            if len(metadata_members) != 1:
                raise InstalledWheelProofError(
                    "retained wheel must contain exactly one METADATA record"
                )
            native_member = native_members[0]
            native_digest = _sha256_bytes(archive.read(native_member))
            metadata = archive.read(metadata_members[0]).decode("utf-8")
            installed_member_count, installed_tree_sha256 = _wheel_payload_proof(
                archive,
                distribution_root=distribution,
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise InstalledWheelProofError("retained wheel is unreadable: %s" % exc) from exc

    metadata_version = next(
        (
            line.split(": ", 1)[1]
            for line in metadata.splitlines()
            if line.startswith("Version: ")
        ),
        None,
    )
    if metadata_version != installed_version:
        raise InstalledWheelProofError(
            "installed distribution version disagrees with retained wheel metadata"
        )
    installed_native_digest = _sha256(extension)
    if installed_native_digest != native_digest:
        raise InstalledWheelProofError(
            "installed native extension is not byte-identical to the retained wheel member"
        )

    return {
        "schema_version": PROOF_SCHEMA_VERSION,
        "python_executable": str(python_executable.resolve()),
        "distribution_root": str(distribution),
        "package_file": str(package),
        "native_extension": str(extension),
        "native_member": native_member,
        "native_sha256": native_digest,
        "installed_member_count": installed_member_count,
        "installed_tree_sha256": installed_tree_sha256,
        "proof_script_sha256": _sha256(Path(__file__).resolve()),
        "version": installed_version,
        "wheel_path": str(retained),
        "wheel_sha256": wheel_digest,
    }


def installed_wheel_proof(wheel: Path) -> dict[str, Any]:
    """Resolve the live imported distribution and authenticate it against ``wheel``."""

    distribution = importlib.metadata.distribution("pops")
    package_file, native_extension, distribution_root = _installed_distribution_paths(
        distribution
    )
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise InstalledWheelProofError(
            "installed distribution has no direct_url.json for the retained wheel"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise InstalledWheelProofError("installed direct_url.json is invalid JSON") from exc
    return build_proof(
        wheel,
        package_file=package_file,
        native_extension=native_extension,
        distribution_root=distribution_root,
        python_executable=Path(sys.executable),
        installed_version=distribution.version,
        direct_url=direct_url,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        proof = installed_wheel_proof(args.wheel)
    except (InstalledWheelProofError, OSError, ValueError) as exc:
        print("installed wheel proof failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
