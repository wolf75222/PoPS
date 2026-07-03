#!/usr/bin/env python3
"""ADC-540 acceptance: the codegen Optimization policy participates in the compile cache key.

A typed ``pops.codegen.Optimization`` selects WHICH IR / expression transforms the emitter applies
(CSE, dead-node / redundant-solve elimination, local fusion, reciprocal hoisting) and the numeric
math mode. Two artifacts built from the SAME model under DIFFERENT policies are NOT
interchangeable, so the policy signature must enter the out-of-source ``.so`` cache key: a policy
change is a cache MISS (a distinct ``.so`` name), never a silent reuse of a differently-optimised
binary. The ``.so`` name is what the manifest reports as its ``cache_key``, so folding the
signature into the path also surfaces it in the manifest.

These checks stay at the pure hash / key level: no ``.so`` is compiled and no System is stepped
(the plan mandates testing ``_cache_so_path`` / the signature directly, not via a real compile).
Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest so
``python3 <file>`` works in CI.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.cache import _cache_so_path, _optimization_cache_key  # noqa: E402
from pops.codegen.optimization import ConservativeFusion, Disabled, Optimization  # noqa: E402
from pops.codegen.math_options import FastMath, StrictMath  # noqa: E402


def _cache_args():
    """A representative (model_hash, abi_key, backend, target, name) cache-key tuple."""
    return ("0123456789abcdef0", "SIG|c++|c++23", "production", "system", "scal")


# --- 1: an Optimization policy changes the .so cache path (a policy change is a cache MISS) -----

def test_optimization_changes_cache_path(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    args = _cache_args()
    # No policy is the historical default: the path is byte-identical to the un-folded call.
    baseline = _cache_so_path(*args)
    assert _cache_so_path(*args, optimization=None) == baseline, "None keeps the historical name"
    # A policy moves the path (a differently-optimised .so is a distinct artifact).
    with_opt = _cache_so_path(*args, optimization=Optimization())
    assert with_opt != baseline, "an explicit policy re-keys the artifact"
    # The same policy is deterministic (a cache HIT on re-request).
    assert _cache_so_path(*args, optimization=Optimization()) == with_opt


def test_distinct_policies_give_distinct_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    args = _cache_args()
    # Each knob that changes the emitted code changes the cache key.
    default = _cache_so_path(*args, optimization=Optimization())
    no_cse = _cache_so_path(*args, optimization=Optimization(cse=False))
    fast = _cache_so_path(*args, optimization=Optimization(math=FastMath()))
    fused = _cache_so_path(*args, optimization=Optimization(fuse=ConservativeFusion()))
    disabled_fuse = _cache_so_path(*args, optimization=Optimization(fuse=Disabled()))
    paths = [default, no_cse, fast, fused, disabled_fuse]
    assert len(set(paths)) == len(paths), "each distinct policy must yield a distinct .so path"
    # StrictMath is the default; naming it explicitly is the SAME policy (same path).
    assert _cache_so_path(*args, optimization=Optimization(math=StrictMath())) == default


# --- 2: the optimization signature string is readable and stable --------------------------------

def test_optimization_cache_key_is_readable_and_stable():
    key = _optimization_cache_key(Optimization())
    assert key.startswith("opt="), key
    # readable: the changed knob is nameable (cse / math appear by name).
    assert "cse=True" in key and "math=StrictMath" in key, key
    # stable across calls for the same policy.
    assert key == _optimization_cache_key(Optimization())
    # None is the empty component (byte-identical historical key).
    assert _optimization_cache_key(None) == ""
    # a pre-lowered options dict of the same shape is accepted too.
    assert _optimization_cache_key(Optimization().options()) == key


def test_optimization_cache_key_tracks_every_knob():
    base = _optimization_cache_key(Optimization())
    for changed in (Optimization(cse=False), Optimization(eliminate_dead_nodes=False),
                    Optimization(eliminate_redundant_solves=False), Optimization(hoist_reciprocals=False),
                    Optimization(fuse=ConservativeFusion()), Optimization(math=FastMath())):
        assert _optimization_cache_key(changed) != base, changed.options()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
