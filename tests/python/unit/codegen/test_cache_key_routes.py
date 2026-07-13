#!/usr/bin/env python3
"""ADC-599 acceptance: route registry / report vocabulary in the ABI + cache manifests.

The compiled-artifact cache keys and the inspection report now embed the typed native route
registry (:mod:`pops.runtime.routes`, mirror of ``include/pops/runtime/config/route_ids.hpp``)
and the capabilities/reports vocabulary version. An artifact built against a DIFFERENT route set
(a route added, removed or re-tokenized, a native entry renamed) or an OLDER report vocabulary
must be a cache MISS, never a silent reuse.

These checks stay at the pure hash / key / report level: no ``.so`` is compiled and no System is
stepped. They pin, at the Python surface:

  1  the artifact-spec identity folds in the registry component: same inputs -> same path; a bumped
     ``ROUTE_REGISTRY_VERSION`` (or a patched component) -> a different path (cache MISS).
  2  the registry cache-key string is the readable ``"routes=vN:<hash16>;capvocab=M"`` form, the
     registry hash and versioned signature authenticate the 64-hex semantic-catalog digest, while
     family counts remain independently pinned.
  3  a ``RuntimeParam`` VALUE never enters ``model_hash`` (seeded at bind, not compile), while a
     ``ConstParam`` value does -- so a runtime ``set_block_params`` is never a recompile.
  4  ``inspect()`` exposes the registry components via ``_route_registry_components()``, consistent
     with :mod:`pops.runtime.routes`.
  5  ``route_registry_signature()`` is the stable versioned semantic-catalog diagnostic.

Guarded with ``pytest.importorskip("pops")`` like the sibling ``test_route_ids.py``; the
``__main__`` block runs pytest so ``python3 <file>`` works in CI.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.params import ConstParam, RuntimeParam  # noqa: E402
from pops.codegen.cache import _identity_cache_so_path, _registry_cache_key  # noqa: E402
from pops.identity import artifact_spec_identity, make_identity  # noqa: E402
from pops.codegen._inspect_compiled_report import _route_registry_components  # noqa: E402
from pops.physics._facade import Model  # noqa: E402
from pops.runtime import routes  # noqa: E402

# The 12 route families in registry order with their acceptance-locked cardinalities (mirror of
# route_ids.hpp; the sibling C++ test locks the two). A new route is an additive change, but the
# per-family COUNT is the shape the compact signature advertises to a stale artifact.
_FAMILY_COUNTS = (
    # riemann grew to 6 with the explicit euler_hllc / euler_roe routes (ADC-590); the registry
    # hash changing is the expected artifact-cache re-key (ADC-599).
    ("riemann", 6), ("limiter", 4), ("recon", 2), ("time", 5),
    ("field_solver", 4), ("poisson_bc", 4), ("layout", 2), ("transport", 3), ("source", 5),
    ("elliptic", 3), ("poisson_rhs", 2), ("wall", 2),
)


def _cache_path():
    semantic = make_identity("semantic", {"model": "0123456789abcdef0"})
    spec = artifact_spec_identity(
        semantic, target="system", backend="production", precision="double",
        abi="SIG|c++|c++23", toolchain="c++|c++23",
        routes={"registry": _registry_cache_key()}, components={"name": "scal"},
        flags=[], libraries=())
    return _identity_cache_so_path(spec)


# --- 1: the registry component participates in every out-of-source .so cache path --------------

def test_identity_cache_path_folds_in_registry(monkeypatch, tmp_path):
    # Isolate the cache dir so the test never touches the user's real ~/.cache/pops/dsl.
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    assert _cache_path() == _cache_path(), "the path is deterministic"
    baseline = _cache_path()
    # Bumping the route catalog version re-keys the artifact (cache MISS, not a silent reuse).
    monkeypatch.setattr(routes, "ROUTE_REGISTRY_VERSION", routes.ROUTE_REGISTRY_VERSION + 1)
    assert _cache_path() != baseline, "a registry version bump changes the cache path"


def test_identity_cache_path_registry_component_is_wired(monkeypatch, tmp_path):
    # Patching the whole component (not just the version int) also moves the path, proving
    # _registry_cache_key -- routes + report vocabulary -- is folded into the file name.
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    baseline = _cache_path()
    monkeypatch.setattr(routes, "route_registry_hash", lambda: "deadbeefdeadbeef" * 4)
    assert _cache_path() != baseline, "the registry component drives the cache path"


# --- 2: the registry cache-key string, hash and signature shape --------------------------------

def test_registry_cache_key_string_form():
    key = _registry_cache_key()
    assert key.startswith("routes=v%d:" % routes.ROUTE_REGISTRY_VERSION), key
    assert ";capvocab=%d" % routes.CAPABILITY_VOCAB_VERSION in key, key
    # The embedded hash prefix is exactly the first 16 hex of the full digest.
    assert key == "routes=v%d:%s;capvocab=%d" % (
        routes.ROUTE_REGISTRY_VERSION, routes.route_registry_hash()[:16],
        routes.CAPABILITY_VOCAB_VERSION), key


def test_registry_hash_is_a_sha256_digest():
    digest = routes.route_registry_hash()
    assert len(digest) == 64, "sha256 hex digest is 64 chars: %d" % len(digest)
    assert all(c in "0123456789abcdef" for c in digest), digest


def test_registry_signature_authenticates_the_full_catalog_and_counts():
    # Content-authenticated signature; family counts remain locked independently below.
    signature = routes.route_registry_signature()
    assert signature.startswith("v%d:" % routes.ROUTE_REGISTRY_VERSION)
    assert len(signature.rsplit(":", 1)[1]) == 64
    assert tuple((fam, len(routes.routes_of(fam))) for fam, _ in _FAMILY_COUNTS) == _FAMILY_COUNTS


# --- 3: a runtime param VALUE never recompiles; a const param value does -----------------------

def _scalar_model(name, declaration):
    """A minimal scalar model whose x-flux reads @p param (rho advected at speed param)."""
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    param = m.value(m.param(declaration))
    m.flux(x=[param * rho], y=[rho])
    m.eigenvalues(x=[rho], y=[rho])
    return m


def test_runtime_param_value_does_not_change_model_hash():
    # A RuntimeParam reads as rparam(<name>) in the formula (its VALUE is not in the repr),
    # and the compile_model cache path hashes model_hash(m) with NO params dict, so the runtime
    # value -- seeded at bind / set_block_params, not at compile -- never reaches the hash.
    slow = _scalar_model("scal_rt", RuntimeParam("nu", default=0.25))
    fast = _scalar_model("scal_rt", RuntimeParam("nu", default=4.0))
    assert slow._model_hash() == fast._model_hash(), "a runtime param value must not recompile"


def test_const_param_value_changes_model_hash():
    # A ConstParam inlines as Const(value) in the formula repr, so its value IS part of the
    # artifact WHAT: changing it is a genuine recompile (a distinct cache key).
    slow = _scalar_model("scal_ct", ConstParam("c", 0.25))
    fast = _scalar_model("scal_ct", ConstParam("c", 4.0))
    assert slow._model_hash() != fast._model_hash(), "a const param value must recompile"


# --- 4: inspect() exposes the registry components, consistent with routes.py -------------------

def test_route_registry_components_keys_and_consistency():
    comp = _route_registry_components()
    assert set(comp) == {"version", "hash", "signature", "capability_vocab_version"}, comp
    assert comp["version"] == routes.ROUTE_REGISTRY_VERSION
    assert comp["hash"] == routes.route_registry_hash()
    assert comp["signature"] == routes.route_registry_signature()
    assert comp["capability_vocab_version"] == routes.CAPABILITY_VOCAB_VERSION


# --- 5: the signature is the stable versioned semantic-catalog diagnostic ---------------------

def test_signature_is_stable_and_all_family_counts_remain_visible():
    first = routes.route_registry_signature()
    assert first == routes.route_registry_signature(), "the signature is stable across calls"
    assert first.startswith("v%d:" % routes.ROUTE_REGISTRY_VERSION)
    assert all(len(routes.routes_of(fam)) == count for fam, count in _FAMILY_COUNTS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
