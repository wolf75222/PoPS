"""pops.codegen.compile_provenance : the debug provenance sidecar for a compiled Program.

``compile_problem(debug=True)`` (or ``POPS_KEEP_GENERATED``) persists the generated ``.cpp`` next
to the ``.so`` for inspection. ADC-536 makes that persisted source SELF-DESCRIBING: a leading
C++ block-comment banner carries the serialized Program IR, the program / ABI / cache hashes, the
compile flags, the toolchain and the redacted compile command, so the ``.cpp`` on disk documents
exactly WHAT was built and HOW.

STRICT invariant (R5): the banner is written ONLY into the persisted sidecar ``.cpp``, never into
the source fed to the compiler. The ``.so`` bytes and the cache key are therefore unchanged whether
``debug`` is on or off -- the banner is inert provenance, not an input to the build. ``compile_problem``
proves this by compiling the banner-free ``src`` and only decorating the sidecar copy.

The final artifact-identity sidecar is written atomically (temp file + ``os.replace``) so a crashed
or concurrent compile never leaves a half-written record that the cache-HIT guard would then read.
"""
from __future__ import annotations

from typing import Any

import json
import os


ARTIFACT_SIDECAR_SUFFIX = ".pops-artifact.json"
_ARTIFACT_SIDECAR_PROTOCOL = "pops.artifact-sidecar.v1"


def lowering_provenance_data(program: Any) -> list[dict[str, Any]]:
    """Return detached lowering lineage without mutating the authored Program."""
    if program is None:
        return []
    from pops.provenance import ProvenanceRecord
    rows = []
    for value in getattr(program, "_values", ()):
        rows.append({
            "node_id": value.id,
            "provenance": ProvenanceRecord.derive(
                (value.provenance,), transformation="lower",
                owner=program.owner_path, authoring_api="pops.codegen._compile",
            ).to_data(),
        })
    return rows


def artifact_sidecar_path(so_path: Any) -> Any:
    """Return the final artifact-identity sidecar path for ``so_path``."""
    return so_path + ARTIFACT_SIDECAR_SUFFIX


def _atomic_write(path: Any, text: Any) -> None:
    """Write @p text to @p path atomically (temp file in the same dir + ``os.replace``).

    Same-directory temp keeps the replace atomic (a cross-filesystem rename is not). A failed write
    leaves the pre-existing file untouched rather than a truncated one -- the cache HIT guard then
    reads a whole sidecar or none, never a half-written one."""
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, ".%s.tmp-%d" % (os.path.basename(path), os.getpid()))
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def write_artifact_sidecar(
    so_path: Any, *, semantic_identity: Any, spec_identity: Any,
) -> tuple[Any, Any]:
    """Authenticate a fresh binary and atomically persist its final identities."""
    from pops.identity import artifact_identity, binary_identity

    binary = binary_identity(so_path)
    artifact = artifact_identity(spec_identity, binary)
    payload = {
        "protocol": _ARTIFACT_SIDECAR_PROTOCOL,
        "semantic_identity": semantic_identity.token,
        "artifact_spec_identity": spec_identity.token,
        "binary_identity": binary.token,
        "artifact_identity": artifact.token,
    }
    _atomic_write(
        artifact_sidecar_path(so_path),
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    )
    return binary, artifact


def publish_staged_artifact(
    staging_path: Any,
    destination_path: Any,
    *,
    semantic_identity: Any,
    spec_identity: Any,
) -> tuple[Any, Any]:
    """Authenticate and publish a staged binary, committing its sidecar last.

    The caller owns the destination's identity-specific inter-process lock.  Both replacements are
    same-directory atomic operations: readers can never observe compiler writes at the final binary
    path, and the final sidecar is the commit record published only after the complete binary.
    """
    staging_path = os.path.abspath(os.fspath(staging_path))
    destination_path = os.path.abspath(os.fspath(destination_path))
    if os.path.dirname(staging_path) != os.path.dirname(destination_path):
        raise ValueError("staged artifact publication requires one filesystem directory")
    binary, artifact = write_artifact_sidecar(
        staging_path,
        semantic_identity=semantic_identity,
        spec_identity=spec_identity,
    )
    os.replace(staging_path, destination_path)
    os.replace(
        artifact_sidecar_path(staging_path),
        artifact_sidecar_path(destination_path),
    )
    return binary, artifact


def read_artifact_sidecar(so_path: Any) -> Any:
    """Read the exact current artifact sidecar schema, or ``None`` when absent."""
    path = artifact_sidecar_path(so_path)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "protocol", "semantic_identity", "artifact_spec_identity", "binary_identity",
        "artifact_identity",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise StaleArtifactError(
            "compiled artifact sidecar must contain exactly %s" % sorted(expected))
    if payload["protocol"] != _ARTIFACT_SIDECAR_PROTOCOL:
        raise StaleArtifactError("compiled artifact sidecar protocol is unsupported")
    if any(not isinstance(payload[key], str) for key in expected - {"protocol"}):
        raise StaleArtifactError("compiled artifact sidecar identities must be strings")
    return payload


def read_artifact_identities(so_path: Any) -> dict[str, Any]:
    """Return typed identities from a verified-schema sidecar."""
    from pops.identity import Identity

    payload = read_artifact_sidecar(so_path)
    if payload is None:
        raise StaleArtifactError("compiled artifact has no identity sidecar")
    result = {
        key: Identity.from_token(payload[key])
        for key in (
            "semantic_identity", "artifact_spec_identity", "binary_identity",
            "artifact_identity",
        )
    }
    expected_domains = {
        "semantic_identity": "semantic",
        "artifact_spec_identity": "artifact-spec",
        "binary_identity": "binary",
        "artifact_identity": "artifact",
    }
    for key, domain in expected_domains.items():
        if result[key].domain != domain:
            raise StaleArtifactError("%s must have domain %s" % (key, domain))
    return result


class StaleArtifactError(RuntimeError):
    """A cached ``.so`` whose sidecar is missing or disagrees with the freshly computed keys."""


def verify_cached_artifact(
    so_path: Any, *, semantic_identity: Any, spec_identity: Any,
) -> tuple[Any, Any]:
    """Re-hash a cached binary and refuse missing, foreign, or corrupt artifacts."""
    from pops.identity import artifact_identity, binary_identity

    found = read_artifact_sidecar(so_path)
    if found is None:
        raise StaleArtifactError(
            "pops.compile: cached artifact %r has no %s sidecar and is unverifiable"
            % (so_path, ARTIFACT_SIDECAR_SUFFIX))
    binary = binary_identity(so_path)
    artifact = artifact_identity(spec_identity, binary)
    expected = {
        "semantic_identity": semantic_identity.token,
        "artifact_spec_identity": spec_identity.token,
        "binary_identity": binary.token,
        "artifact_identity": artifact.token,
    }
    mismatches = {
        key: (value, found.get(key)) for key, value in expected.items()
        if found.get(key) != value
    }
    if mismatches:
        raise StaleArtifactError(
            "pops.compile: cached artifact %r failed identity verification: %r"
            % (so_path, mismatches))
    return binary, artifact


def build_debug_banner(program: Any, model: Any, *, program_hash: Any, abi_key: Any,
                       cache_key: Any, cflags: Any, lflags: Any, cxx: Any, std: Any,
                       command: Any, registry: Any) -> str:
    """Return the C++ block-comment provenance banner for the persisted debug ``.cpp`` (ADC-536).

    The banner documents WHAT the ``.so`` was built from and HOW: the serialized Program IR (the
    full documentary ``_serialize()`` blob (the identity hash uses its provenance-free projection),
    the program / ABI / cache hashes, the compile
    + link flags, the compiler and C++ standard, the redacted compile command and the route registry
    components. It is a C++ block comment (``/* ... */``), inert to the compiler.

    STRICT (R5): this string is prepended ONLY to the persisted sidecar ``.cpp``, never to the source
    fed to the compiler -- so the ``.so`` bytes and the cache key are byte-identical whether ``debug``
    is on or off. A ``*/`` in a serialized field is defanged to ``* /`` so the block comment cannot be
    closed early by the content.
    """
    ir = "(no Program IR: this handle carries no serializable time Program)"
    lowering = "[]"
    if program is not None and hasattr(program, "_serialize"):
        ir = json.dumps(program._serialize(), indent=2, sort_keys=True)
        lowering = json.dumps(lowering_provenance_data(program), indent=2, sort_keys=True)
    model_name = getattr(model, "name", None) or getattr(program, "name", None) or "problem"
    lines = [
        "pops.compile provenance banner (ADC-536) -- INERT, sidecar-only, not compiled",
        "",
        "model            : %s" % model_name,
        "program          : %s" % (getattr(program, "name", None) or "problem"),
        "program_hash     : %s" % program_hash,
        "abi_key          : %s" % abi_key,
        "cache_key        : %s" % cache_key,
        "cxx              : %s" % cxx,
        "std              : %s" % std,
        "cflags           : %s" % " ".join(cflags or []),
        "lflags           : %s" % " ".join(lflags or []),
        "compile_command  : %s" % command,
        "route_registry   : %s" % registry,
        "",
        "serialized Program IR (documentary provenance included; excluded from _ir_hash):",
        ir,
        "",
        "lowering provenance (documentary; excluded from identities):",
        lowering,
    ]
    body = "\n".join(lines).replace("*/", "* /")
    return "/*\n%s\n*/\n" % body
