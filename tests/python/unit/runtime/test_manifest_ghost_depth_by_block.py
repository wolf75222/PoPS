#!/usr/bin/env python3
"""ADC-536 acceptance: the compiled-artifact manifest carries per-block ghost depth.

CONTRACTS6 decision 4: the bind stream validates each block's initial-state ghosts against the
MANIFEST value, so the manifest must carry the halo depth keyed by block, not just a single scalar.
This module pins, at the pure metadata level (no compile / bind / .so):

  1  ``CompiledArtifactManifest`` has an immutable ``ghost_depth_by_block`` mapping;
  2  it round-trips through ``to_dict`` / ``from_dict`` (additive field, schema still v1);
  3  ``build_arguments`` populates ``layout_runtime["ghost_depth_by_block"]`` keyed by each
     committed block, with the same depth as the scalar ``ghost_depth``;
  4  ``build_compiled_manifest`` threads the per-block map onto the manifest.

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pops = pytest.importorskip("pops")
from pops.codegen._plans import InstallBlock, InstallPlan  # noqa: E402
from pops.codegen.loader import CompiledModel, CompiledProblem  # noqa: E402
from pops.external.artifact_manifest import CompiledArtifactManifest  # noqa: E402
from pops.model import Module  # noqa: E402
from pops.model.bind_schema import BindSchema  # noqa: E402
from tests.python.unit.runtime._typed_program import typed_program_states  # noqa: E402


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
    # The wire view consumed by the bind stream/reports is a detached plain dict of name -> int.
    import json
    m = CompiledArtifactManifest(model_name="gas", ghost_depth_by_block={"ions": 2})
    json.dumps(m.to_dict())  # must not raise


# --- 3: build_arguments populates the per-block map keyed by committed block --------------------

def _compiled(blocks):
    """Build a real typed Program + InstallPlan artifact without compiling a shared object."""
    module = Module("ghost-depth-model")
    state = module.state_space("U", ("rho", "mx", "my"))
    declarations = tuple((name, state) for name in blocks)
    program, _, problem, endpoints = typed_program_states(
        "gas_program", module, declarations)
    program.commit_many({endpoint.next: endpoint.n for endpoint in endpoints.values()})

    model = CompiledModel(
        so_path="/nonexistent/ghost-depth.so", backend="production",
        adder="add_native_block", cons_names=("rho", "mx", "my"), cons_roles=(),
        prim_names=("rho", "mx", "my"), n_vars=3, gamma=1.4, n_aux=0,
        params={}, caps={}, abi_key="SIG|c++|c++23", model_hash="ghost-depth-model",
        cxx="c++", std="c++23",
    )
    compiled = CompiledProblem(
        "/nonexistent/ghost-depth-problem.so", program, model,
        "SIG|c++|c++23", "c++", "c++23",
    )
    schema = BindSchema.from_problem(problem)
    snapshot = problem.freeze()
    compiled.install_plan = InstallPlan(
        snapshot_hash=snapshot.hash,
        target="system",
        layout=None,
        blocks=tuple(InstallBlock(name, model, None) for name in blocks),
        bind_schema=schema,
        field_solvers={},
        outputs=(),
        diagnostics=(),
        has_program=True,
    )
    compiled.bind_schema = schema
    compiled._problem_snapshot = snapshot
    model._seal()
    compiled._seal()
    return compiled


def test_build_arguments_keys_ghost_depth_by_block():
    from pops.codegen.inspect_compiled import build_arguments
    args = build_arguments(_compiled(("ions", "electrons")))
    lr = args.layout_runtime
    assert set(lr["ghost_depth_by_block"]) == {"ions", "electrons"}, "one entry per committed block"
    # Every block shares the scalar depth today (one physics model per Program).
    scalar = lr["ghost_depth"]
    assert all(d == scalar for d in lr["ghost_depth_by_block"].values()), \
        "per-block depth agrees with the scalar"


# --- 4: build_compiled_manifest threads the per-block map ---------------------------------------

def test_build_compiled_manifest_threads_per_block_ghost_depth():
    from pops.external.artifact_manifest import build_compiled_manifest
    manifest = build_compiled_manifest(_compiled(("ions",)))
    assert manifest.ghost_depth_by_block == {"ions": manifest.ghost_depth}, \
        "the manifest carries per-block ghost depth from arguments()"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
