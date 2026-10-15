"""Temporal point reconstruction and common-time hierarchy solves compose."""

import pops
import pytest

from examples.api_v02.method_temporal_field import build_temporal_field
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_hierarchy_regions import hierarchy_region_solves
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.detach import detach_compiled_program
from pops.time._program.temporal_field import is_temporal_field_solve


def test_temporal_and_synchronized_solves_preserve_their_distinct_authorities():
    spec = build_temporal_field(hierarchy_companion=True)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    program = resolved.time
    solves = [node for node in program._values if node.op == "solve_linear"]
    assert len(solves) == 2
    temporal, synchronized = solves
    assert is_temporal_field_solve(temporal)
    assert not is_temporal_field_solve(synchronized)
    assert hierarchy_region_solves(program) == (synchronized,)
    assert temporal.id != synchronized.id

    source = emit_cpp_program(
        program,
        target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
    )
    assert source.count("ctx.solve_hierarchy_field(") == 2
    assert source.count("ctx.suspend_hierarchy_barrier(") == 1
    assert source.count("ctx.advance_mapping_hierarchy(") == 1
    assert source.count("ctx.declare_temporal_field_source(") == 1
    assert source.count("ctx.temporal_field_input(") == 2

    stores = {node.name: node for node in program._values if node.op == "store_history"}
    before = stores["observed_before_barrier"].inputs[0]
    after = stores["observed_after_barrier"].inputs[0]
    assert before is after
    assert before is not stores["synchronized"].inputs[0]
    suspension = source.index("ctx.suspend_hierarchy_barrier(")
    assert "program_field_%d_snapshot->local_field()" % before.id in source[suspension:]
    assert source.count("ctx.capture_hierarchy_field_observation(") == 1
    assert source.count("ctx.reduce_hierarchy_field_observation(") == 2
    assert "ctx.reduce_hierarchy_field_component(" not in source
    assert source.index("ctx.capture_hierarchy_field_observation(") < suspension
    assert source.index("ctx.reduce_hierarchy_field_observation(") < suspension
    assert source.rindex("ctx.reduce_hierarchy_field_observation(") > suspension
    assert source.index('ctx.store_global_field_history("observed_before_barrier"') < suspension
    assert source.index('ctx.store_global_field_history("observed_after_barrier"') > suspension


def test_mixed_field_composition_survives_detachment_and_changes_equation_identity():
    exact = build_temporal_field(hierarchy_companion=True).case._time
    held = build_temporal_field(hierarchy_companion=True, constant_control=True).case._time
    detached = detach_compiled_program(exact)
    assert detached._ir_hash() == exact._ir_hash()
    assert held._ir_hash() != exact._ir_hash()
    assert len(hierarchy_region_solves(detached)) == 1


@pytest.mark.parametrize("capacity", (0, -1, True, 1.5, 1 << 63))
def test_observation_rejects_nonpositive_or_noncanonical_capacity(capacity):
    with pytest.raises(ValueError, match="byte_capacity"):
        build_temporal_field(hierarchy_companion=True, observation_byte_capacity=capacity)


def test_snapshot_budget_is_preserved_as_authored_resource_authority():
    small = build_temporal_field(hierarchy_companion=True, observation_byte_capacity=1).case._time
    large = build_temporal_field(hierarchy_companion=True).case._time
    assert small._ir_hash() != large._ir_hash()
    clone = detach_compiled_program(small)
    assert clone._ir_hash() == small._ir_hash()
    assert all(
        node.attrs["observation_byte_capacity"] == 1
        for node in clone._values
        if node.op == "field_component"
    )
