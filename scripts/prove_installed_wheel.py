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

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    exact_dimensions,
    sha256_file,
    validate_manifest_payload,
)


ROOT = Path(__file__).resolve().parents[1]
PROOF_SCHEMA_VERSION = 4


class InstalledWheelProofError(RuntimeError):
    """The retained wheel and the imported installation are not byte-identical."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return sha256_file(path)


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
    """Resolve package/manifest paths from distribution metadata without importing PoPS."""

    members = tuple(distribution.files or ())
    package_members = [
        member for member in members if member.as_posix() == "pops/__init__.py"
    ]
    manifest_members = [
        member for member in members
        if member.as_posix() == "pops/_native/variants.json"
    ]
    if len(package_members) != 1 or len(manifest_members) != 1:
        raise InstalledWheelProofError(
            "installed distribution lacks one unique pops package and native variant manifest"
        )
    return (
        Path(distribution.locate_file(package_members[0])).resolve(),
        Path(distribution.locate_file(manifest_members[0])).resolve(),
        Path(distribution.locate_file("")).resolve(),
    )


def _is_native_member(name: str) -> bool:
    filename = Path(name).name
    return any(filename == "_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _native_variant_proof(
    archive: zipfile.ZipFile,
    *,
    distribution_root: Path,
    manifest_payload: Any,
    expected_dimensions: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Authenticate exactly the manifest-declared leaf set in wheel and installation."""
    rows = validate_manifest_payload(
        manifest_payload, expected_dimensions=expected_dimensions
    )
    expected_members = {
        "pops/_native/" + row["path"]: row for row in rows
    }
    discovered = {
        name for name in archive.namelist()
        if not name.endswith("/") and _is_native_member(name)
    }
    if discovered != set(expected_members):
        raise InstalledWheelProofError(
            "retained wheel native leaves %s do not equal manifest set %s"
            % (tuple(sorted(discovered)), tuple(sorted(expected_members)))
        )
    evidence = []
    for member, row in sorted(
        expected_members.items(), key=lambda item: item[1]["dimension"]
    ):
        info = archive.getinfo(member)
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise InstalledWheelProofError(
                "retained wheel native member is a symlink: %s" % member
            )
        wheel_digest = _sha256_bytes(archive.read(member))
        if wheel_digest != row["sha256"]:
            raise InstalledWheelProofError(
                "retained wheel Dim=%d bytes disagree with variants.json"
                % row["dimension"]
            )
        installed = (distribution_root / member).resolve()
        try:
            installed.relative_to(distribution_root)
        except ValueError as exc:
            raise InstalledWheelProofError(
                "installed native variant escapes the distribution root: %s" % member
            ) from exc
        if installed.is_symlink() or not installed.is_file():
            raise InstalledWheelProofError(
                "installed native variant is missing or a symlink: %s" % installed
            )
        if _sha256(installed) != wheel_digest:
            raise InstalledWheelProofError(
                "installed Dim=%d native extension is not byte-identical to the retained wheel"
                % row["dimension"]
            )
        evidence.append({
            "dimension": row["dimension"],
            "extension": str(installed),
            "member": member,
            "sha256": wheel_digest,
            "version": row["version"],
            "abi_key": row["abi_key"],
            "build_fingerprint": row["build_fingerprint"],
            "has_mpi": row["has_mpi"],
            "has_kokkos": row["has_kokkos"],
        })
    return tuple(evidence)


def build_proof(
    wheel: Path,
    *,
    package_file: Path,
    native_manifest: Path,
    distribution_root: Path,
    python_executable: Path,
    installed_version: str,
    direct_url: Any,
    expected_dimensions: Sequence[int],
) -> dict[str, Any]:
    """Authenticate one installed distribution against one exact wheel archive."""

    dimensions = exact_dimensions(
        expected_dimensions, where="installed wheel expected dimensions"
    )
    retained = _outside_checkout(wheel, label="retained wheel")
    package = _outside_checkout(package_file, label="installed package")
    manifest = _outside_checkout(native_manifest, label="installed native manifest")
    distribution = _outside_checkout(distribution_root, label="installed distribution")
    if retained.suffix != ".whl" or not retained.is_file():
        raise InstalledWheelProofError("retained wheel is not a readable .whl file")
    for label, path in (("installed package", package), ("installed native manifest", manifest)):
        if not path.is_file():
            raise InstalledWheelProofError("%s is not a readable file: %s" % (label, path))
    if manifest != package.parent / "_native" / "variants.json":
        raise InstalledWheelProofError(
            "installed native manifest is not under the installed pops package"
        )
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
            manifest_members = [
                name for name in names if name == "pops/_native/variants.json"
            ]
            metadata_members = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(manifest_members) != 1:
                raise InstalledWheelProofError(
                    "retained wheel must contain exactly one pops/_native/variants.json"
                )
            if len(metadata_members) != 1:
                raise InstalledWheelProofError(
                    "retained wheel must contain exactly one METADATA record"
                )
            manifest_member = manifest_members[0]
            manifest_bytes = archive.read(manifest_member)
            try:
                manifest_payload = json.loads(manifest_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InstalledWheelProofError(
                    "retained wheel native variant manifest is invalid JSON"
                ) from exc
            native_variants = _native_variant_proof(
                archive,
                distribution_root=distribution,
                manifest_payload=manifest_payload,
                expected_dimensions=dimensions,
            )
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
    if _sha256(manifest) != _sha256_bytes(manifest_bytes):
        raise InstalledWheelProofError(
            "installed variants.json is not byte-identical to the retained wheel member"
        )
    if any(variant["version"] != installed_version for variant in native_variants):
        raise InstalledWheelProofError(
            "native variant version disagrees with retained wheel metadata"
        )

    return {
        "schema_version": PROOF_SCHEMA_VERSION,
        "expected_dimensions": list(dimensions),
        "python_executable": str(python_executable.resolve()),
        "distribution_root": str(distribution),
        "package_file": str(package),
        "native_manifest": str(manifest),
        "native_manifest_member": manifest_member,
        "native_manifest_sha256": _sha256_bytes(manifest_bytes),
        "native_variants": list(native_variants),
        "installed_member_count": installed_member_count,
        "installed_tree_sha256": installed_tree_sha256,
        "proof_script_sha256": _sha256(Path(__file__).resolve()),
        "version": installed_version,
        "wheel_path": str(retained),
        "wheel_sha256": wheel_digest,
    }


def installed_wheel_proof(
    wheel: Path, *, expected_dimensions: Sequence[int]
) -> dict[str, Any]:
    """Resolve the live imported distribution and authenticate it against ``wheel``."""

    distribution = importlib.metadata.distribution("pops")
    package_file, native_manifest, distribution_root = _installed_distribution_paths(
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
        native_manifest=native_manifest,
        distribution_root=distribution_root,
        python_executable=Path(sys.executable),
        installed_version=distribution.version,
        direct_url=direct_url,
        expected_dimensions=expected_dimensions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument(
        "--expect-dim", action="append", required=True, type=int, choices=(1, 2, 3),
        help="exact variant set; repeat for an explicit multi-variant wheel",
    )
    args = parser.parse_args(argv)
    try:
        proof = installed_wheel_proof(
            args.wheel, expected_dimensions=args.expect_dim
        )
    except (InstalledWheelProofError, NativeVariantManifestError, OSError, ValueError) as exc:
        print("installed wheel proof failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
