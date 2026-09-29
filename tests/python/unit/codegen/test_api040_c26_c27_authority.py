"""C26/C27 countertests for production artifact authority, without native compilation.

These exercise the guards used by the public compile/bind path. They do not
claim that the reference mini-runtime's C symbols are production exports.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pops.codegen import abi
from pops.codegen.compile_provenance import (
    StaleArtifactError, artifact_sidecar_path, verify_cached_artifact,
    write_artifact_sidecar,
)
from pops.identity import make_identity


def test_compiled_header_mismatch_is_refused_before_dynamic_load(monkeypatch):
    monkeypatch.setattr(abi, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(abi, "module_header_signature", lambda: "loaded-headers")
    with pytest.raises(RuntimeError, match="headers DIFFERENT"):
        abi.check_compiled_matches_module("foreign-headers|clang++|c++20|dim=2")


def test_compiled_dimension_mismatch_is_refused_before_dynamic_load(monkeypatch):
    monkeypatch.setattr(abi, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(abi, "module_header_signature", lambda: "same-headers")
    with pytest.raises(RuntimeError, match="dimension 3 differs.*dimension 2"):
        abi.check_compiled_matches_module("same-headers|clang++|c++20|dim=3")


def test_cached_artifact_rejects_changed_binary_bytes(tmp_path):
    path = tmp_path / "component.so"
    path.write_bytes(b"original validly staged bytes")
    semantic = make_identity("semantic", {"case": "C26"})
    spec = make_identity("artifact-spec", {"abi": "native-3", "case": "C26"})
    write_artifact_sidecar(str(path), semantic_identity=semantic, spec_identity=spec)
    path.write_bytes(b"different bytes at the same cache path")
    with pytest.raises(StaleArtifactError, match="binary_identity"):
        verify_cached_artifact(str(path), semantic_identity=semantic, spec_identity=spec)


def test_cached_artifact_rejects_tampered_artifact_identity(tmp_path):
    path = tmp_path / "component.so"
    path.write_bytes(b"unchanged binary bytes")
    semantic = make_identity("semantic", {"case": "C27"})
    spec = make_identity("artifact-spec", {"abi": "native-3", "case": "C27"})
    write_artifact_sidecar(str(path), semantic_identity=semantic, spec_identity=spec)
    sidecar = artifact_sidecar_path(str(path))
    payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    payload["artifact_identity"] = make_identity("artifact", {"foreign": True}).token
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with pytest.raises(StaleArtifactError, match="artifact_identity"):
        verify_cached_artifact(str(path), semantic_identity=semantic, spec_identity=spec)
