"""Compiler-owned multi-layout diagnostic bounds; source/transport substitutes only."""
from types import SimpleNamespace
import json
import struct

import pytest

from pops._platform_contracts import artifact_platform_manifest
from pops.codegen import _compiled_artifact as compiled
from pops.runtime import _checkpoint_resource_budget as budget
from pops.runtime import _checkpoint_program_diagnostics as codec
from pops.output import _checkpoint_collective as collective
from pops.output._checkpoint_contract import CheckpointResourceBudget
from tests.python.unit.runtime.test_runtime_planning import _artifact


@pytest.fixture
def artifact_factory(monkeypatch):
    monkeypatch.setattr(compiled, "_common_platform_manifest", lambda **kw: artifact_platform_manifest(
        backend=kw["backend"], target=kw["target"], component=kw["blocks"][0].model))
    def make(*sources):
        base = _artifact(names=("left", "right"), heterogeneous=True)
        assert base.program is None and len(base.layout_programs) == 2
        for row, source in zip(base.layout_programs, sources, strict=True):
            row.program._generated_cpp = source
        return compiled.CompiledSimulationArtifact(base.plan, None, base.blocks, base.layout_programs)
    return make


def test_registered_layout_sources_determine_finite_union(artifact_factory):
    artifact = artifact_factory('ctx.record_scalar("left", 0); ctx.record_scalar("shared", 1);',
                                'ctx.record_scalar("right", 2); ctx.record_scalar("shared", 3);')
    names = budget._compiled_diagnostic_inventory(artifact)
    assert names == ("left", "pops.frontier.duration", "right", "shared")
    assert budget._diagnostic_inventory_capacity(names) == 40+sum(16+len(name.encode()) for name in names)


@pytest.mark.parametrize("rank,ranks", [(0, 1), (0, 2), (1, 2), (2, 3)])
def test_empty_initial_and_populated_native_image_fit_compiled_bound(artifact_factory, monkeypatch, rank, ranks):
    names = ("first", "second-λ", "pops.frontier.duration")
    artifact = artifact_factory('ctx.record_scalar("first", 0);',
                                'ctx.record_scalar('+json.dumps("second-λ")+', 0);')
    inventory = budget._compiled_diagnostic_inventory(artifact)
    capacity = budget._diagnostic_inventory_capacity(inventory)
    base = CheckpointResourceBudget("uniform", 2, 100, 100, 1000, 5000, "base")
    chosen, total = budget._diagnostic_capacity_budget(base, ("program_diagnostics_state", "program_diagnostics_offsets"), capacity, ranks)
    owner = SimpleNamespace(_checkpoint_program_diagnostic_artifact=artifact,
        _checkpoint_program_diagnostic_inventory=inventory,
        _checkpoint_program_diagnostic_base_budget=base,
        _checkpoint_program_diagnostic_member_names=("program_diagnostics_state", "program_diagnostics_offsets"),
        _checkpoint_program_diagnostic_capacity_per_rank=capacity,
        _checkpoint_program_diagnostic_ranks=ranks,
        _checkpoint_program_diagnostic_byte_capacity=total, _checkpoint_resource_budget=chosen)
    topology = collective.CheckpointTopology(rank, ranks, object() if ranks>1 else None)
    monkeypatch.setattr(collective, "checkpoint_topology", lambda _: topology)
    def vote(topology, phase, *, error=None, value=None):
        if error is not None:
            raise error
        return value
    monkeypatch.setattr(collective, "consensus", vote)
    def image(owner_rank, records):
        return b"POPSDIA1"+struct.pack("<QQQQ", 64, owner_rank, ranks, len(records))+b"".join(
            struct.pack("<Q", len(name.encode()))+name.encode()+struct.pack("<d", 1.) for name in records)
    from pops import _native_collectives
    for records in ((), names):
        images = tuple(image(peer, records) for peer in range(ranks))
        owner._s = SimpleNamespace(_checkpoint_program_diagnostics=lambda images=images: images[rank])
        monkeypatch.setattr(_native_collectives, "allgather_bytes", lambda *args, images=images: images)
        payload = {}
        codec.capture_checkpoint_program_diagnostics(owner, payload)
        assert payload["program_diagnostics_offsets"].tolist() == [len(images[0])*i for i in range(ranks+1)]
        assert payload["program_diagnostics_state"].tobytes() == b"".join(images)
        assert codec.validate_checkpoint_program_diagnostic_arrays(payload, capacity=capacity, bound=total)


@pytest.mark.parametrize("source", [None, 'ctx.record_scalar(dynamic_name, 0);'])
def test_missing_or_dynamic_slice_stays_explicit_unknown(artifact_factory, source):
    artifact = artifact_factory('ctx.record_scalar("known", 0);', source)
    assert budget._compiled_diagnostic_inventory(artifact) is None
    assert budget._diagnostic_inventory_capacity(None) is None


def test_changed_retained_slice_refused_before_reinventory(artifact_factory):
    artifact = artifact_factory('ctx.record_scalar("before", 0);', '// no records')
    artifact.layout_programs[1].program._generated_cpp = 'ctx.record_scalar("after", 0);'
    with pytest.raises(ValueError, match="retained"):
        budget._compiled_diagnostic_inventory(artifact)


def test_unknown_external_program_does_not_infer_empty_default():
    artifact = SimpleNamespace(program=None, layout_programs=(), verify=lambda: None)
    assert budget._compiled_diagnostic_inventory(artifact) is None


def test_single_program_bound_unchanged():
    artifact = SimpleNamespace(program=SimpleNamespace(_generated_cpp='ctx.record_scalar("x", 0);'), verify=lambda: None)
    assert budget._compiled_diagnostic_inventory(artifact) == ("pops.frontier.duration", "x")
