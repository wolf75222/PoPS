"""Authenticate the installed exact-rank native leaf and its capabilities.

Planning does not require a native artifact. Execution does: the selected
dimension must have a variants.json row whose on-disk bytes match the digest.
This module does not dlopen the leaf.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.machinery
import json
import os
from typing import Any

_ROW_KEYS = {
    "dimension",
    "path",
    "sha256",
    "version",
    "abi_key",
    "has_mpi",
    "has_kokkos",
}
_KNOWN_REQUIREMENTS = {
    "mpi",
    "hdf5",
    "hdf5_collective",
    "exact_native_dimension",
    "polar_system_runtime",
}


class CapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AuthenticatedArtifact:
    """Facts from an authenticated exact-rank variants.json row."""

    dimension: int
    path: Path
    sha256: str
    version: str
    abi_key: str
    has_mpi: bool
    has_kokkos: bool
    hdf5_collective: bool
    doctor_ok: bool
    native_variant_manifest_digest: str
    native_header_signature: str
    component_catalog_digest: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _doctor_report_ok(report: Any) -> bool:
    if report is True:
        return True
    if not isinstance(report, Mapping) or not report:
        return False
    values = list(report.values())
    if all(isinstance(item, tuple) and item and item[0] is True for item in values):
        return True
    return report.get("ok") is True


def _try_doctor() -> bool:
    try:
        import pops

        report = pops.doctor(verbose=False)
    except Exception:
        return False
    return _doctor_report_ok(report)


_HEADER_SIGNATURE: str | None = None


def _header_signature() -> str:
    global _HEADER_SIGNATURE
    if _HEADER_SIGNATURE is not None:
        return _HEADER_SIGNATURE
    try:
        from pops.codegen import abi

        signature = abi.module_header_signature()
        if isinstance(signature, str) and signature:
            _HEADER_SIGNATURE = signature
            return signature
    except Exception:
        pass
    _HEADER_SIGNATURE = hashlib.sha256(
        b"pops.verification.native_header_signature-unavailable"
    ).hexdigest()
    return _HEADER_SIGNATURE


def _catalog_digest() -> str:
    try:
        from pops.release import contract

        digest = contract()["component_catalog_sha256"]
        if isinstance(digest, str) and digest:
            return digest
    except Exception:
        pass
    return hashlib.sha256(b"pops.verification.catalog-unavailable").hexdigest()


def _hdf5_collective(has_mpi: bool) -> bool:
    try:
        from pops._native_selector import selected_native_module

        module = selected_native_module(required=False)
    except Exception:
        module = None
    if module is None:
        return False
    available = getattr(module, "__has_parallel_hdf5__", None)
    return available is True and has_mpi


def resolve_variants_root(
    variants_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    if variants_root is not None:
        return Path(variants_root)
    env = os.environ if environ is None else environ
    explicit = env.get("POPS_NATIVE_VARIANTS_ROOT")
    if explicit:
        return Path(explicit)
    try:
        import pops

        for item in getattr(pops, "__path__", ()):
            candidate = Path(item) / "_native"
            if (candidate / "variants.json").is_file():
                return candidate
    except Exception:
        pass
    raise CapabilityError(
        "no authenticated exact-rank variants manifest "
        "(set POPS_NATIVE_VARIANTS_ROOT or install a native leaf)"
    )


def authenticate_installed_artifact(
    dimension: int | None = None,
    *,
    variants_root: str | Path | None = None,
    doctor_ok: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> AuthenticatedArtifact:
    """Return the authenticated exact-rank leaf for ``dimension``.

    Reads ``variants.json`` and re-hashes the leaf. Does not load the
    extension. ``doctor_ok`` may be injected; otherwise ``pops.doctor`` is
    attempted and a missing doctor is recorded as false.
    """
    if dimension not in (1, 2, 3):
        raise CapabilityError("dimension must be 1, 2, or 3")
    root = resolve_variants_root(variants_root, environ)
    manifest = root / "variants.json"
    if not manifest.is_file():
        raise CapabilityError(f"native variants manifest missing: {manifest}")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapabilityError(f"cannot read native variants manifest {manifest}") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("variants"), list)
    ):
        raise CapabilityError("unsupported native variants manifest schema")
    rows: dict[int, dict[str, Any]] = {}
    for raw in document["variants"]:
        if not isinstance(raw, dict) or set(raw) != _ROW_KEYS:
            raise CapabilityError("native variant row has an invalid schema")
        row_dim = raw["dimension"]
        if row_dim not in (1, 2, 3):
            raise CapabilityError("native variant dimension must be exactly 1, 2, or 3")
        rows[row_dim] = raw
    if dimension not in rows:
        raise CapabilityError(
            f"installed PoPS distribution has no native specialization for Dim={dimension}"
        )
    row = rows[dimension]
    suffix_ok = {f"_pops{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES}
    relative = Path(row["path"])
    if relative.parts != (f"dim{dimension}", relative.name) or relative.name not in suffix_ok:
        raise CapabilityError("native variant path must be dimN/_pops<EXT_SUFFIX>")
    leaf = root.joinpath(*relative.parts)
    if not leaf.is_file():
        raise CapabilityError(f"native variant path is absent: {leaf}")
    digest = sha256_file(leaf)
    if digest != row["sha256"]:
        raise CapabilityError(f"native variant bytes differ from variants.json: {leaf}")
    resolved_doctor = _try_doctor() if doctor_ok is None else bool(doctor_ok)
    return AuthenticatedArtifact(
        dimension=dimension,
        path=leaf,
        sha256=digest,
        version=str(row["version"]),
        abi_key=str(row["abi_key"]),
        has_mpi=bool(row["has_mpi"]),
        has_kokkos=bool(row["has_kokkos"]),
        hdf5_collective=_hdf5_collective(bool(row["has_mpi"])),
        doctor_ok=resolved_doctor,
        native_variant_manifest_digest=sha256_file(manifest),
        native_header_signature=_header_signature(),
        component_catalog_digest=_catalog_digest(),
    )


def missing_requirements(
    case: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    current_capabilities: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return installed-capability tokens the case requires but the artifact lacks."""
    current = current_capabilities or {}
    missing: list[str] = []
    for token in case.get("requires") or []:
        if token not in _KNOWN_REQUIREMENTS:
            continue
        if token == "mpi" and not artifact.has_mpi:
            missing.append(token)
        elif token in {"hdf5", "hdf5_collective"} and not artifact.hdf5_collective:
            missing.append(token)
        elif token == "polar_system_runtime" and current.get("polar_system_runtime") is not True:
            missing.append(token)
        elif token == "exact_native_dimension" and artifact.dimension not in (1, 2, 3):
            missing.append(token)
    return missing
