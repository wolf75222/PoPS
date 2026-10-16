"""Frozen DSO output-history authority must accompany the Uniform read route."""
import json
import re
from types import SimpleNamespace

import pops
import pytest
from examples.api_v02.independent_field_history import build_independent_history
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_history_metadata import emit_checkpoint_history_metadata
from pops.codegen.program_models import ProgramModelGraph


@pytest.fixture(scope="module")
def prepared():
    spec = build_independent_history()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time, target="system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    return resolved.time, source


def _arguments(line):
    return [json.loads(token) for token in re.findall(r'"(?:[^"\\]|\\.)*"|-?\d+', line)]


def test_uniform_read_has_one_complete_frozen_tuple_matching_registration(prepared):
    program, source = prepared
    shared = emit_checkpoint_history_metadata(program, require_owner=True)
    assert shared == emit_checkpoint_history_metadata(program, require_owner=False)
    assert source.count(shared) == 1
    registration = next(line for line in source.splitlines()
                        if 'ctx.register_history("potential"' in line)
    name, lag, components, owner, state, space, clock, interpolation, ghosts = _arguments(registration)
    expected = dict(name=name, owner=owner, state_identity=state, space_identity=space,
                    clock_identity=clock, interpolation_identity=interpolation,
                    depth=lag + 1, components=components, ghost_depth=ghosts)
    assert 'pops_program_checkpoint_history_count() { return 1; }' in source
    for column, value in expected.items():
        body = source.split("pops_program_checkpoint_history_" + column + "(int index)", 1)[1]
        literal = body.split("case 0: return ", 1)[1].split(";", 1)[0]
        assert json.loads(literal) == value
    assert 'ctx.history_observation("potential", 1, 1, 0,' in source


def test_unqualified_legacy_uniform_ring_cannot_issue_projection_authority():
    # Emitter-only legacy shape: no storage owner was declared, so no qualified
    # tuple may be invented. Runtime authority is received by native tests.
    program = SimpleNamespace(_histories={"legacy": 1}, _history_blocks={},
        _block_indices=lambda: {}, temporal_manifest=lambda: {"histories": []})
    source = emit_checkpoint_history_metadata(program, require_owner=False)
    assert 'pops_program_checkpoint_history_count() { return 0; }' in source
    with pytest.raises(ValueError, match="explicit block owner provenance"):
        emit_checkpoint_history_metadata(program, require_owner=True)
