#!/usr/bin/env python3
"""ADC-540 acceptance: the codegen Optimization policy participates in the compile cache key.

A typed ``pops.codegen.Optimization`` selects WHICH IR / expression transforms the emitter applies
(CSE, dead-node / redundant-solve elimination, local fusion, reciprocal hoisting) and the numeric
math mode. Two artifacts built from the SAME model under DIFFERENT policies are NOT
interchangeable, so the policy signature must enter the out-of-source ``.so`` cache key: a policy
change is a cache MISS (a distinct ``.so`` name), never a silent reuse of a differently-optimised
binary. The ``.so`` name is what the manifest reports as its ``cache_key``, so folding the
signature into the path also surfaces it in the manifest.

These checks stay at the typed artifact-spec identity level: no ``.so`` is compiled.
Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest so
``python3 <file>`` works in CI.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.cache import _identity_cache_so_path  # noqa: E402
from pops.identity import artifact_spec_identity, make_identity  # noqa: E402
from pops.codegen.optimization import ConservativeFusion, Disabled, Optimization  # noqa: E402
from pops.codegen.math_options import FastMath, StrictMath  # noqa: E402


def _path(policy, tmp_path):
    semantic = make_identity("semantic", {"model": "scal"})
    spec = artifact_spec_identity(
        semantic, target="system", backend="production", precision="double",
        abi="SIG|c++|c++23", toolchain="c++|c++23", routes={},
        components={"optimization": None if policy is None else policy.options()},
        flags=[], libraries=())
    return _identity_cache_so_path(spec)


# --- 1: an Optimization policy changes the .so cache path (a policy change is a cache MISS) -----

def test_optimization_changes_cache_path(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    baseline = _path(None, tmp_path)
    # A policy moves the path (a differently-optimised .so is a distinct artifact).
    with_opt = _path(Optimization(), tmp_path)
    assert with_opt != baseline, "an explicit policy re-keys the artifact"
    # The same policy is deterministic (a cache HIT on re-request).
    assert _path(Optimization(), tmp_path) == with_opt


def test_distinct_policies_give_distinct_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    # Each knob that changes the emitted code changes the cache key.
    default = _path(Optimization(), tmp_path)
    no_cse = _path(Optimization(cse=False), tmp_path)
    fast = _path(Optimization(math=FastMath()), tmp_path)
    fused = _path(Optimization(fuse=ConservativeFusion()), tmp_path)
    disabled_fuse = _path(Optimization(fuse=Disabled()), tmp_path)
    paths = [default, no_cse, fast, fused, disabled_fuse]
    assert len(set(paths)) == len(paths), "each distinct policy must yield a distinct .so path"
    # StrictMath is the default; naming it explicitly is the SAME policy (same path).
    assert _path(Optimization(math=StrictMath()), tmp_path) == default


# --- 2: the optimization signature string is readable and stable --------------------------------

def test_optimization_identity_tracks_every_knob(monkeypatch, tmp_path):
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    base = _path(Optimization(), tmp_path)
    for changed in (Optimization(cse=False), Optimization(eliminate_dead_nodes=False),
                    Optimization(eliminate_redundant_solves=False), Optimization(hoist_reciprocals=False),
                    Optimization(fuse=ConservativeFusion()), Optimization(math=FastMath())):
        assert _path(changed, tmp_path) != base, changed.options()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
