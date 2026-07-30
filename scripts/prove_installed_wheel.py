#!/usr/bin/env python3
"""Prove that the imported PoPS package is the exact retained release wheel."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote, urlparse
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROOF_SCHEMA_VERSION = 1


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
        "version": installed_version,
        "wheel_path": str(retained),
        "wheel_sha256": wheel_digest,
    }


def installed_wheel_proof(wheel: Path) -> dict[str, Any]:
    """Resolve the live imported distribution and authenticate it against ``wheel``."""

    import pops
    from pops import _pops

    distribution = importlib.metadata.distribution("pops")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise InstalledWheelProofError(
            "installed distribution has no direct_url.json for the retained wheel"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise InstalledWheelProofError("installed direct_url.json is invalid JSON") from exc
    if pops.__version__ != _pops.__version__ or pops.__version__ != distribution.version:
        raise InstalledWheelProofError("installed Python/native/distribution versions disagree")
    return build_proof(
        wheel,
        package_file=Path(pops.__file__),
        native_extension=Path(_pops.__file__),
        distribution_root=Path(distribution.locate_file("")),
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
