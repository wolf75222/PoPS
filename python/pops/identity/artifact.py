"""Artifact-layer identities derived from semantic definitions and emitted binary bytes."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

from .digest import Identity, make_identity


ARTIFACT_SCHEMA_VERSION = 1


def _require_identity(value: Any, domain: str, where: str) -> Identity:
    if not isinstance(value, Identity) or value.domain != domain:
        raise TypeError("%s must be a pops.%s Identity" % (where, domain))
    return value


def artifact_spec_identity(
    semantic_identity: Any,
    *,
    target: Any,
    backend: Any,
    precision: Any,
    abi: Any,
    toolchain: Any,
    routes: Any,
    components: Any,
    flags: Any = (),
    libraries: Any = (),
    codegen_version: Any = "pops.codegen.v1",
) -> Identity:
    """Identity of every compile decision made before binary emission.

    Paths are deliberately absent. Libraries are content identities, never locations.
    """
    semantic = _require_identity(semantic_identity, "semantic", "artifact semantic_identity")
    for name, value in (
        ("target", target),
        ("backend", backend),
        ("precision", precision),
        ("abi", abi),
        ("toolchain", toolchain),
        ("codegen_version", codegen_version),
    ):
        if not isinstance(value, str) or not value:
            raise TypeError("artifact %s must be a non-empty string" % name)
    if not isinstance(routes, Mapping):
        raise TypeError("artifact routes must be a mapping")
    if not isinstance(components, Mapping):
        raise TypeError("artifact components must be a mapping")
    return make_identity(
        "artifact-spec",
        {
            "semantic": semantic.to_data(),
            "target": target,
            "backend": backend,
            "precision": precision,
            "abi": abi,
            "toolchain": toolchain,
            "routes": dict(routes),
            "components": dict(components),
            "flags": list(flags),
            "libraries": list(libraries),
            "codegen_version": codegen_version,
        },
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )


def binary_identity(path: Any) -> Identity:
    """Hash the exact bytes installed or loaded at ``path``; the path itself is excluded."""
    resolved = os.fspath(path)
    digest = hashlib.sha256()
    size = 0
    with open(resolved, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return make_identity(
        "binary",
        {"algorithm": "sha256", "content_digest": digest.digest(), "size": size},
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )


def artifact_identity(spec_identity: Any, emitted_binary_identity: Any) -> Identity:
    """Final compiled artifact identity: compile specification plus exact binary bytes."""
    spec = _require_identity(spec_identity, "artifact-spec", "artifact spec_identity")
    binary = _require_identity(emitted_binary_identity, "binary", "artifact binary_identity")
    return make_identity(
        "artifact",
        {"spec": spec.to_data(), "binary": binary.to_data()},
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )


def binary_bundle_identity(components: Any) -> Identity:
    """Identity of an installed artifact composed from several exact binaries."""
    if not isinstance(components, Mapping) or not components:
        raise TypeError("binary bundle components must be a non-empty mapping")
    payload = {}
    for name, identity in components.items():
        if not isinstance(name, str) or not name:
            raise TypeError("binary bundle component names must be non-empty strings")
        payload[name] = _require_identity(
            identity, "binary", "binary bundle component %r" % name).to_data()
    return make_identity(
        "binary", {"components": payload}, schema_version=ARTIFACT_SCHEMA_VERSION)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "artifact_identity",
    "artifact_spec_identity",
    "binary_bundle_identity",
    "binary_identity",
]
