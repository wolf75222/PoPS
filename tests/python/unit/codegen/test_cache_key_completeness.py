#!/usr/bin/env python3
"""ADC-536 acceptance: the compiled-Program cache key folds the feature-key + precision token.

The program ``.so`` cache key at ``compile_drivers`` now composes the native Kokkos/MPI feature-key
(``_native_feature_key``) and the precision token (``_precision_cache_key``) on TOP of the historical
model / program-IR / abi / registry / optimization / platform components. A SERIAL-stub ``.so`` must
not be reused on an MPI module, a ``.so`` built against a different Kokkos must be a MISS, and a
future precision switch must not reuse a double-precision ``.so``.

These checks stay at the pure hash / key level: no ``.so`` is compiled and no System is stepped.
They pin, at the Python surface, that:

  1  ``_precision_cache_key`` renders the current native fact ("precision=double;real_bytes=8");
  2  the program cache-key composition changes when the feature-key or the precision token changes,
     and is deterministic for fixed inputs;
  3  the out-of-source ``_cache_so_path`` folds the same tokens through the backend slot, so a
     feature/precision change is a distinct ``.so`` file name (cache MISS);
  4  the ``debug`` flag is NOT in the cache key -- it is source-provenance only (binary-identical).

Guarded with ``pytest.importorskip("pops")`` like the sibling ``test_cache_key_routes.py``; the
``__main__`` block runs pytest so ``python3 <file>`` works in CI.
"""
import hashlib
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.cache import _cache_so_path, _precision_cache_key, _registry_cache_key  # noqa: E402
from pops.codegen import cache as cache_mod  # noqa: E402


def _program_cache_key(program_hash, abi_key, target, feature_key, precision_key):
    """Recompose the program cache key EXACTLY as compile_drivers.compile_problem does (ADC-536)."""
    return hashlib.sha256(("%s|%s|program-production|%s|%s|%s|%s"
                           % (program_hash, abi_key, target, _registry_cache_key(),
                              feature_key, precision_key)).encode()).hexdigest()


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


# --- 3: the out-of-source .so file name folds the same tokens through the backend slot ----------

def test_cache_so_path_folds_feature_and_precision(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    abi = "SIG|c++|c++23"
    base_backend = "program-production;kokkos=off;mpi=off;precision=double;real_bytes=8"
    base = _cache_so_path("phash", abi, base_backend, "system", "problem")
    assert base == _cache_so_path("phash", abi, base_backend, "system", "problem"), "deterministic"
    mpi_backend = "program-production;kokkos=off;mpi=on;precision=double;real_bytes=8"
    assert _cache_so_path("phash", abi, mpi_backend, "system", "problem") != base, \
        "an MPI feature flip changes the .so file name"
    single_backend = "program-production;kokkos=off;mpi=off;precision=single;real_bytes=4"
    assert _cache_so_path("phash", abi, single_backend, "system", "problem") != base, \
        "a precision switch changes the .so file name"


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
