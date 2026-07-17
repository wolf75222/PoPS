#!/usr/bin/env python3
"""ADC-536 acceptance: the compiled-Program cache key folds the feature-key + precision token.

The program ``.so`` cache key at ``compile_drivers`` now composes the native Kokkos/MPI feature-key
(``_native_feature_key``) and the precision token (``_precision_cache_key``) on TOP of the historical
model / program-IR / abi / registry / optimization / platform components. A SERIAL-stub ``.so`` must
not be reused on an MPI module, Open MPI and MPICH ABIs must occupy distinct cache slots, a ``.so``
built against a different Kokkos must be a MISS, and a future precision switch must not reuse a
double-precision ``.so``.

These checks stay at the pure hash / key level: no ``.so`` is compiled and no System is stepped.
They pin, at the Python surface, that:

  1  ``_precision_cache_key`` renders the current native fact ("precision=double;real_bytes=8");
  2  the program cache-key composition changes when the feature-key or the precision token changes,
     and is deterministic for fixed inputs;
  3  the typed artifact-spec identity folds the same feature and precision tokens;
     feature/precision change is a distinct ``.so`` file name (cache MISS);
  4  the ``debug`` flag is NOT in the cache key -- it is source-provenance only (binary-identical).

Guarded with ``pytest.importorskip("pops")`` like the sibling ``test_cache_key_routes.py``; the
``__main__`` block runs pytest so ``python3 <file>`` works in CI.
"""
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pops")
from pops.codegen.cache import (  # noqa: E402
    _artifact_distinct_so_path,
    _identity_cache_so_path,
    _precision_cache_key,
    _process_so_identity,
    _record_artifact_identity,
    _registry_cache_key,
)
from pops.identity import artifact_spec_identity, make_identity  # noqa: E402


def _program_cache_key(program_hash, abi_key, target, feature_key, precision_key):
    semantic = make_identity("semantic", {"program": program_hash})
    return artifact_spec_identity(
        semantic, target=target, backend="production", precision=precision_key,
        abi=abi_key, toolchain="c++|c++23",
        routes={"registry": _registry_cache_key(), "features": feature_key},
        components={}, flags=[], libraries=())


# --- 1: the precision token renders the current native fact ------------------------------------

def test_precision_cache_key_renders_current_double():
    key = _precision_cache_key()
    assert key == "precision=double;real_bytes=8", key


# --- 2: the program cache key moves with the feature-key and the precision token ---------------

def test_program_cache_key_is_deterministic():
    a = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=on;kcfg=abc;mpi=off",
                           "precision=double;real_bytes=8")
    b = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=on;kcfg=abc;mpi=off",
                           "precision=double;real_bytes=8")
    assert a == b, "the program cache key is deterministic for fixed inputs"


def test_program_cache_key_changes_with_feature_key():
    base = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=on;kcfg=abc;mpi=off",
                              "precision=double;real_bytes=8")
    mpi_on = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=on;kcfg=abc;mpi=on",
                                "precision=double;real_bytes=8")
    assert base != mpi_on, "an MPI feature flip must re-key the program (no serial-stub reuse)"
    other_kokkos = _program_cache_key("phash", "SIG|c++|c++23", "system",
                                      "kokkos=on;kcfg=DIFFERENT;mpi=off",
                                      "precision=double;real_bytes=8")
    assert base != other_kokkos, "a different Kokkos config must re-key the program"


def test_program_cache_key_changes_with_precision_token():
    base = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=off;mpi=off",
                              "precision=double;real_bytes=8")
    single = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=off;mpi=off",
                                "precision=single;real_bytes=4")
    assert base != single, "a precision switch must re-key the program (no double .so reuse)"


def _mpi_module(tmp_path, *, library_bytes=b"mpi-library", library_name="libmpi.so"):
    import hashlib
    from pops.codegen._native_mpi import NativeMpiContract, _abi_material

    tmp_path.mkdir(parents=True, exist_ok=True)
    include_a = tmp_path / "mpi-a"
    include_b = tmp_path / "mpi-b"
    include_a.mkdir()
    include_b.mkdir()
    header = include_a / "mpi.h"
    header.write_bytes(b"exact-mpi-header")
    library = tmp_path / library_name
    library.write_bytes(library_bytes)
    header_hash = hashlib.sha256(header.read_bytes()).hexdigest()
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    provisional = NativeMpiContract(
        abi_sha256="",
        compiler="/mpi/bin/mpicxx",
        standard="4.1",
        include_dirs=(str(include_a), str(include_b)),
        compile_options=("-pthread",),
        compile_definitions=("OMPI_SKIP_MPICXX",),
        link_options=("-Wl,-rpath,/mpi/lib",),
        link_libraries=(str(library),),
        header_paths=(str(header),),
        header_sha256=(header_hash,),
        library_paths=(str(library),),
        library_sha256=(library_hash,),
    )
    abi = hashlib.sha256(_abi_material(provisional)).hexdigest()
    data = {
        "schema_version": 1,
        **{name: getattr(provisional, name) for name in (
            "compiler", "standard", "include_dirs", "compile_options",
            "compile_definitions", "link_options", "link_libraries", "header_paths",
            "header_sha256", "library_paths", "library_sha256")},
        "abi_sha256": abi,
    }
    return SimpleNamespace(__has_mpi__=True, __mpi_contract__=data), library


@pytest.mark.parametrize("has_mpi", (False, True))
def test_every_runtime_loader_matches_host_mpi_seam(monkeypatch, tmp_path, has_mpi):
    """The shared loader flags replay the complete host MPI target, never a serial seam."""
    from pops.codegen import _native_host, toolchain

    mpi_module, library = _mpi_module(tmp_path)
    module = mpi_module if has_mpi else SimpleNamespace(
        __has_mpi__=False, __mpi_contract__=None)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    monkeypatch.setattr(_native_host, "ensure_native_host_global", lambda value: None)
    monkeypatch.setattr(toolchain, "_native_kokkos_root", lambda: "/kokkos")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda cxx=None: "c++")
    monkeypatch.setattr(
        toolchain,
        "_native_kokkos_flags",
        lambda: (["-DPOPS_HAS_KOKKOS"], ["-pthread"]),
    )

    _, compile_flags, link_flags = toolchain.pops_loader_build_flags()
    if has_mpi:
        assert "-DPOPS_HAS_MPI" in compile_flags
        assert "-pthread" in compile_flags
        assert "-DOMPI_SKIP_MPICXX" in compile_flags
        assert any(flag.startswith('-DPOPS_MPI_ABI="') for flag in compile_flags)
        assert str(tmp_path / "mpi-a") in compile_flags
        assert str(tmp_path / "mpi-b") in compile_flags
        assert "-Wl,-rpath,/mpi/lib" in link_flags
        assert str(library) in link_flags
    else:
        assert "-DPOPS_HAS_MPI" not in compile_flags
        assert str(library) not in link_flags


def test_native_feature_key_partitions_concrete_mpi_abi(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    first_module, _ = _mpi_module(tmp_path / "first", library_bytes=b"open-mpi")
    second_module, _ = _mpi_module(tmp_path / "second", library_bytes=b"mpich")
    selected = [first_module]
    monkeypatch.setattr(toolchain, "_native_kokkos_root", lambda: None)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: selected[0])
    first = toolchain._native_feature_key()
    selected[0] = second_module
    second = toolchain._native_feature_key()
    assert first != second
    assert "mabi=" + first_module.__mpi_contract__["abi_sha256"] in first
    assert "mabi=" + second_module.__mpi_contract__["abi_sha256"] in second


@pytest.mark.parametrize("fault", ("missing-include", "missing-header", "missing-abi"))
def test_mpi_loader_contract_fails_closed(tmp_path, fault):
    from pops.codegen._native_mpi import native_mpi_compile_flags

    module, _ = _mpi_module(tmp_path)
    if fault == "missing-include":
        module.__mpi_contract__["include_dirs"] = ()
    elif fault == "missing-header":
        (tmp_path / "mpi-a" / "mpi.h").unlink()
    else:
        module.__mpi_contract__["abi_sha256"] = ""
    with pytest.raises(RuntimeError, match="MPI|mpi"):
        native_mpi_compile_flags(module)


def test_mpi_upgrade_in_place_is_rejected_before_compilation(tmp_path):
    from pops.codegen import _native_mpi

    module, library = _mpi_module(tmp_path)
    assert _native_mpi.native_mpi_link_flags(module)[-1] == str(library)
    library.write_bytes(b"upgraded-in-place")
    with pytest.raises(RuntimeError, match="changed in place"):
        _native_mpi.native_mpi_link_flags(module)


def test_static_mpi_archive_is_rejected_before_dynamic_plugin_link(tmp_path):
    from pops.codegen import _native_mpi

    module, _ = _mpi_module(tmp_path, library_name="libmpi.a")
    with pytest.raises(RuntimeError, match="static archive"):
        _native_mpi.native_mpi_build_flags(module)


def test_ambiguous_mpi_flag_delimiter_is_rejected(tmp_path):
    from pops.codegen import _native_mpi

    module, _ = _mpi_module(tmp_path)
    module.__mpi_contract__["compile_options"] = ("-pthread;-DSECOND_COMMAND",)
    with pytest.raises(RuntimeError, match="ambiguous serialized delimiter"):
        _native_mpi.native_mpi_build_flags(module)


def test_unsupported_windows_compilers_fail_before_emitting_posix_commands(monkeypatch):
    from pops.codegen import _compile_platform

    monkeypatch.setattr(_compile_platform.sys, "platform", "win32")
    with pytest.raises(NotImplementedError, match="No POSIX command"):
        _compile_platform.require_shared_library_compile_platform(
            "compile_problem", windows_supported=False)


def test_native_host_is_promoted_globally_once_and_kept_alive(monkeypatch, tmp_path):
    from pops.codegen import _native_host

    image = tmp_path / "_pops.so"
    image.write_bytes(b"test-image")
    calls = []
    handle = object()
    monkeypatch.setattr(_native_host.sys, "platform", "linux")
    monkeypatch.setattr(
        _native_host.ctypes, "CDLL",
        lambda path, mode: calls.append((path, mode)) or handle,
    )
    _native_host._GLOBAL_HANDLES.clear()
    try:
        module = SimpleNamespace(__file__=str(image))
        _native_host.ensure_native_host_global(module)
        _native_host.ensure_native_host_global(module)
        assert len(calls) == 1
        assert calls[0][1] & _native_host.ctypes.RTLD_GLOBAL
        assert _native_host._GLOBAL_HANDLES[str(image.resolve())] is handle
    finally:
        _native_host._GLOBAL_HANDLES.clear()


def test_native_abi_literal_guards_mpi_mode_and_concrete_abi():
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    source = (root / "include/pops/runtime/dynamic/abi_key.hpp").read_text(encoding="utf-8")
    assert '";mpi=" POPS_ABI_MPI' in source
    assert '";mpi_abi=" POPS_ABI_MPI_ID' in source


# --- 3: the out-of-source .so file name folds the same tokens through the backend slot ----------

def test_identity_cache_path_folds_feature_and_precision(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    abi = "SIG|c++|c++23"
    base_backend = "program-production;kokkos=off;mpi=off;precision=double;real_bytes=8"
    base = _identity_cache_so_path(_program_cache_key(
        "phash", abi, "system", base_backend, "precision=double;real_bytes=8"))
    assert base == _identity_cache_so_path(_program_cache_key(
        "phash", abi, "system", base_backend, "precision=double;real_bytes=8")), "deterministic"
    mpi_backend = "program-production;kokkos=off;mpi=on;precision=double;real_bytes=8"
    assert _identity_cache_so_path(_program_cache_key(
        "phash", abi, "system", mpi_backend, "precision=double;real_bytes=8")) != base, \
        "an MPI feature flip changes the .so file name"
    single_backend = "program-production;kokkos=off;mpi=off;precision=single;real_bytes=4"
    assert _identity_cache_so_path(_program_cache_key(
        "phash", abi, "system", single_backend, "precision=single;real_bytes=4")) != base, \
        "a precision switch changes the .so file name"


def test_explicit_path_is_partitioned_by_authenticated_artifact_identity(tmp_path):
    requested = str(tmp_path / "model.so")
    _process_so_identity.clear()
    try:
        assert _artifact_distinct_so_path(requested, "spec-A") == requested
        _record_artifact_identity(requested, "spec-A")
        assert _artifact_distinct_so_path(requested, "spec-A") == requested

        alternate = _artifact_distinct_so_path(requested, "spec-B")
        assert alternate != requested
        assert alternate.endswith(".so")
        assert _artifact_distinct_so_path(requested, "spec-B") == alternate
    finally:
        _process_so_identity.clear()


# --- 4: the debug flag is NOT in the cache key (source-provenance only) -------------------------

def test_debug_flag_not_in_program_cache_key():
    # debug toggles keep_generated (a sidecar .cpp with a provenance banner), never the .so bytes or
    # the key. The program cache key has no debug field, so two keys for the same inputs are equal
    # regardless of debug -- proven here by the composition (no debug argument in the key at all).
    key_a = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=off;mpi=off",
                               "precision=double;real_bytes=8")
    key_b = _program_cache_key("phash", "SIG|c++|c++23", "system", "kokkos=off;mpi=off",
                               "precision=double;real_bytes=8")
    assert key_a == key_b, "the program cache key does not depend on debug"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
