#!/usr/bin/env python3
"""ADC-536 acceptance: the compiled-artifact manifest carries per-block ghost depth.

CONTRACTS6 decision 4: the bind stream validates each block's initial-state ghosts against the
MANIFEST value, so the manifest must carry the halo depth keyed by block, not just a single scalar.
This module pins, at the pure metadata level (no compile / bind / .so):

  1  ``CompiledArtifactManifest`` has a ``ghost_depth_by_block`` dict field;
  2  it round-trips through ``to_dict`` / ``from_dict`` (additive field, schema still v1);
  3  ``build_arguments`` populates ``layout_runtime["ghost_depth_by_block"]`` keyed by each
     committed block, with the same depth as the scalar ``ghost_depth``;
  4  ``build_compiled_manifest`` threads the per-block map onto the manifest.

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.external.artifact_manifest import CompiledArtifactManifest  # noqa: E402


# --- 1 + 2: the field exists and round-trips ---------------------------------------------------

def test_manifest_carries_ghost_depth_by_block_field():
    m = CompiledArtifactManifest(model_name="gas", ghost_depth=2,
                                 ghost_depth_by_block={"ions": 2, "electrons": 2})
    assert m.ghost_depth_by_block == {"ions": 2, "electrons": 2}
    # A default is an empty dict, never None (a degraded handle exposes no bind table).
    assert CompiledArtifactManifest(model_name="x").ghost_depth_by_block == {}


def test_ghost_depth_by_block_round_trips_through_dict():
    m = CompiledArtifactManifest(model_name="gas", ghost_depth=2,
                                 ghost_depth_by_block={"ions": 2})
    data = m.to_dict()
    assert data["ghost_depth_by_block"] == {"ions": 2}, "serialized in to_dict"
    # Additive field: from_dict accepts it (schema stays v1, no unknown-field error).
    back = CompiledArtifactManifest.from_dict(data)
    assert back.ghost_depth_by_block == {"ions": 2}, "reconstructed by from_dict"


def test_ghost_depth_by_block_is_serializable_plain_values():
    # The bind stream (and the ADC-564 typed-report conversion) wrap it unchanged: plain dict of
    # name -> int, JSON-ready.
    import json
    m = CompiledArtifactManifest(model_name="gas", ghost_depth_by_block={"ions": 2})
    json.dumps(m.to_dict())  # must not raise


# --- 3: build_arguments populates the per-block map keyed by committed block --------------------

class _FakeProgram:
    def __init__(self, blocks):
        self._blocks = blocks
        self._values = []

    def commits(self):
        return {name: object() for name in self._blocks}


class _FakeModel:
    name = "gas"
    cons_names = ["rho", "mx", "my"]
    cons_roles = None
    caps = {}
    params = {}
    aux_extra_names = []


class _FakeCompiled:
    def __init__(self, blocks):
        self.program = _FakeProgram(blocks)
        self.model = _FakeModel()
        self.program_name = "gas_program"
        self.abi_key = "SIG|c++|c++23"

    def arguments(self):
        from pops.codegen.inspect_compiled import build_arguments
        return build_arguments(self)


def test_build_arguments_keys_ghost_depth_by_block():
    from pops.codegen.inspect_compiled import build_arguments
    args = build_arguments(_FakeCompiled(["ions", "electrons"]))
    lr = args.layout_runtime
    assert set(lr["ghost_depth_by_block"]) == {"ions", "electrons"}, "one entry per committed block"
    # Every block shares the scalar depth today (one physics model per Program).
    scalar = lr["ghost_depth"]
    assert all(d == scalar for d in lr["ghost_depth_by_block"].values()), \
        "per-block depth agrees with the scalar"


# --- 4: build_compiled_manifest threads the per-block map ---------------------------------------

def test_build_compiled_manifest_threads_per_block_ghost_depth():
    from pops.external.artifact_manifest import build_compiled_manifest
    manifest = build_compiled_manifest(_FakeCompiled(["ions"]))
    assert manifest.ghost_depth_by_block == {"ions": manifest.ghost_depth}, \
        "the manifest carries per-block ghost depth from arguments()"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
