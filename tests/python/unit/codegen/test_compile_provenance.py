#!/usr/bin/env python3
"""ADC-536 acceptance: the debug provenance banner + the cache-key sidecar (pure Python).

``compile_problem(debug=True)`` persists the generated ``.cpp`` with a leading provenance banner
(serialized IR, hashes, flags, toolchain, redacted command), and every fresh compile writes a
final artifact sidecar the cache-HIT guard re-verifies. This module pins the PURE-PYTHON parts of
that machinery -- the banner string composition and the sidecar read / write / verify logic -- with
no compiler and no ``.so`` (the real-compiler integration lives in the gated integration tests).

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.compile_provenance import (  # noqa: E402
    artifact_sidecar_path, build_debug_banner, read_artifact_sidecar,
    verify_cached_artifact, write_artifact_sidecar, StaleArtifactError)
from pops.identity import make_identity  # noqa: E402


class _FakeProgram:
    """A minimal stand-in exposing the two attributes the banner reads (name + _serialize)."""

    def __init__(self, name="prog"):
        self.name = name

    def _serialize(self):
        return {"nodes": ["a", "b"], "commit": "block0"}


class _FakeModel:
    def __init__(self, name="mymodel"):
        self.name = name


# --- the debug provenance banner ---------------------------------------------------------------

def test_banner_is_a_cpp_block_comment_with_all_fields():
    banner = build_debug_banner(
        _FakeProgram("fe"), _FakeModel("gas"), program_hash="ph123", abi_key="SIG|cxx|c++23",
        cache_key="ck456", cflags=["-O3", "-DNDEBUG"], lflags=["-lfoo"], cxx="clang++",
        std="c++23", command="clang++ ... -o problem.so", registry="routes=v1:abc;capvocab=0")
    assert banner.startswith("/*"), "the banner is a C++ block comment"
    assert banner.rstrip().endswith("*/"), "the banner closes the block comment"
    # Every provenance field is present.
    for needle in ("gas", "fe", "ph123", "SIG|cxx|c++23", "ck456", "-O3 -DNDEBUG", "-lfoo",
                   "clang++", "c++23", "clang++ ... -o problem.so", "routes=v1:abc;capvocab=0"):
        assert needle in banner, "banner must carry %r" % needle
    # The serialized IR is embedded.
    assert '"nodes"' in banner and '"commit"' in banner, "banner carries the serialized IR"


def test_banner_defangs_comment_terminator_in_content():
    # A '*/' inside a serialized field must not close the block comment early.
    class _EvilProgram(_FakeProgram):
        def _serialize(self):
            return {"note": "danger */ echo pwned"}

    banner = build_debug_banner(
        _EvilProgram(), _FakeModel(), program_hash="p", abi_key="a", cache_key="c",
        cflags=[], lflags=[], cxx="c++", std="c++23", command="cmd", registry="r")
    # The only '*/' is the final terminator; the content's '*/' is defanged to '* /'.
    assert banner.count("*/") == 1, "content '*/' is defanged so the comment closes exactly once"
    assert "* /" in banner, "the embedded terminator is defanged"


def test_banner_handles_a_handle_without_a_program():
    banner = build_debug_banner(
        None, _FakeModel("m"), program_hash="p", abi_key="a", cache_key="c", cflags=[], lflags=[],
        cxx="c++", std="c++23", command="cmd", registry="r")
    assert "no Program IR" in banner, "a handle with no serializable Program is stated honestly"


# --- the cache-key sidecar: write / read round-trip --------------------------------------------

def test_sidecar_round_trip(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    (tmp_path / "problem-abc.so").write_bytes(b"binary")
    semantic = make_identity("semantic", {"program": "p"})
    spec = make_identity("artifact-spec", {"target": "system"})
    binary, artifact = write_artifact_sidecar(
        so_path, semantic_identity=semantic, spec_identity=spec)
    assert artifact_sidecar_path(so_path).endswith(".pops-artifact.json")
    found = read_artifact_sidecar(so_path)
    assert found == {
        "protocol": "pops.artifact-sidecar.v1",
        "semantic_identity": semantic.token,
        "artifact_spec_identity": spec.token,
        "binary_identity": binary.token,
        "artifact_identity": artifact.token,
    }


def test_read_sidecar_absent_is_none(tmp_path):
    assert read_artifact_sidecar(str(tmp_path / "nope.so")) is None


# --- the cache-HIT stale/ABI guard -------------------------------------------------------------

def test_verify_accepts_a_matching_sidecar(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    (tmp_path / "problem-abc.so").write_bytes(b"binary")
    semantic = make_identity("semantic", {"program": "p"})
    spec = make_identity("artifact-spec", {"target": "system"})
    expected = write_artifact_sidecar(so_path, semantic_identity=semantic, spec_identity=spec)
    assert verify_cached_artifact(
        so_path, semantic_identity=semantic, spec_identity=spec) == expected


def test_verify_refuses_a_missing_sidecar(tmp_path):
    so_path = str(tmp_path / "legacy.so")
    open(so_path, "w").close()  # a .so with NO sidecar (a legacy artifact)
    with pytest.raises(StaleArtifactError) as exc:
        verify_cached_artifact(
            so_path,
            semantic_identity=make_identity("semantic", {}),
            spec_identity=make_identity("artifact-spec", {}),
        )
    assert "no" in str(exc.value) and "sidecar" in str(exc.value)


def test_verify_refuses_a_mismatched_spec_identity(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    (tmp_path / "problem-abc.so").write_bytes(b"binary")
    semantic = make_identity("semantic", {})
    old_spec = make_identity("artifact-spec", {"version": "old"})
    new_spec = make_identity("artifact-spec", {"version": "new"})
    write_artifact_sidecar(so_path, semantic_identity=semantic, spec_identity=old_spec)
    with pytest.raises(StaleArtifactError) as exc:
        verify_cached_artifact(
            so_path, semantic_identity=semantic, spec_identity=new_spec)
    msg = str(exc.value)
    assert new_spec.token in msg and old_spec.token in msg


def test_verify_refuses_a_mismatched_semantic_identity(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    (tmp_path / "problem-abc.so").write_bytes(b"binary")
    old_semantic = make_identity("semantic", {"program": "old"})
    new_semantic = make_identity("semantic", {"program": "new"})
    spec = make_identity("artifact-spec", {})
    write_artifact_sidecar(
        so_path, semantic_identity=old_semantic, spec_identity=spec)
    with pytest.raises(StaleArtifactError):
        verify_cached_artifact(
            so_path, semantic_identity=new_semantic, spec_identity=spec)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
