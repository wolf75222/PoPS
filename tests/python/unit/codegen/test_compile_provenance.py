#!/usr/bin/env python3
"""ADC-536 acceptance: the debug provenance banner + the cache-key sidecar (pure Python).

``compile_problem(debug=True)`` persists the generated ``.cpp`` with a leading provenance banner
(serialized IR, hashes, flags, toolchain, redacted command), and every fresh compile writes a
``<so>.cachekey`` sidecar the cache-HIT guard re-verifies. This module pins the PURE-PYTHON parts of
that machinery -- the banner string composition and the sidecar read / write / verify logic -- with
no compiler and no ``.so`` (the real-compiler integration lives in the gated integration tests).

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.compile_provenance import (  # noqa: E402
    build_debug_banner, cachekey_path, read_cachekey_sidecar, verify_cached_program_so,
    write_cachekey_sidecar, StaleArtifactError)


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
    write_cachekey_sidecar(so_path, cache_key="CK", abi_key="ABI", toolchain="clang++|c++23")
    assert cachekey_path(so_path).endswith(".cachekey")
    found = read_cachekey_sidecar(so_path)
    assert found == {"cache_key": "CK", "abi_key": "ABI", "toolchain": "clang++|c++23"}


def test_read_sidecar_absent_is_none(tmp_path):
    assert read_cachekey_sidecar(str(tmp_path / "nope.so")) is None


# --- the cache-HIT stale/ABI guard -------------------------------------------------------------

def test_verify_accepts_a_matching_sidecar(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    write_cachekey_sidecar(so_path, cache_key="CK", abi_key="ABI", toolchain="clang++|c++23")
    # A matching sidecar returns None (accept, no raise).
    assert verify_cached_program_so(so_path, cache_key="CK", abi_key="ABI") is None


def test_verify_refuses_a_missing_sidecar(tmp_path):
    so_path = str(tmp_path / "legacy.so")
    open(so_path, "w").close()  # a .so with NO sidecar (a legacy artifact)
    with pytest.raises(StaleArtifactError) as exc:
        verify_cached_program_so(so_path, cache_key="CK", abi_key="ABI")
    assert "no" in str(exc.value) and "sidecar" in str(exc.value), "names the missing sidecar"
    assert "rm " in str(exc.value), "tells the user how to clear the cache"


def test_verify_refuses_a_mismatched_cache_key(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    write_cachekey_sidecar(so_path, cache_key="OLD", abi_key="ABI", toolchain="clang++|c++23")
    with pytest.raises(StaleArtifactError) as exc:
        verify_cached_program_so(so_path, cache_key="NEW", abi_key="ABI")
    msg = str(exc.value)
    assert "expected=NEW" in msg and "found=OLD" in msg, "names expected vs found"


def test_verify_refuses_a_mismatched_abi_key(tmp_path):
    so_path = str(tmp_path / "problem-abc.so")
    write_cachekey_sidecar(so_path, cache_key="CK", abi_key="OLDABI", toolchain="clang++|c++23")
    with pytest.raises(StaleArtifactError):
        verify_cached_program_so(so_path, cache_key="CK", abi_key="NEWABI")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
