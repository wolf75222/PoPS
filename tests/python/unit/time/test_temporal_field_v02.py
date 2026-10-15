"""Authored temporal field contracts; these checks do not claim native execution."""

from pops.time._program.serialization import _json_ready
import pops
import pytest
from examples.api_v02.method_temporal_field import build_temporal_field, quadratic_evaluation
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time import TemporalEvaluation
from pops.time._program.temporal_field import validate_temporal_field


def test_local_field_point_assembles_every_level_before_real_composite_solve():
    spec = build_temporal_field()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(
        resolved.time,
        target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
    )
    assert source.count("ctx.solve_hierarchy_field(") == 1
    assert source.count("ctx.temporal_field_input(") == 2
    assert source.count("ctx.temporal_field_evaluation_time(") == 2
    assert "ctx.temporal_field_evaluation_time()" not in source
    assert source.count("ctx.declare_temporal_field_source(") == 1
    assert "ctx.suspend_hierarchy_barrier(" not in source
    assert "ctx.advance_mapping_hierarchy(" not in source
    assert "source.parameters.size() != 4" in source
    assert "ctx.for_each_program_resource_level(" in source


def test_temporal_point_map_has_distinct_native_point_and_complete_ring_inputs():
    spec = build_temporal_field()
    program = spec.case._time
    load = next(v for v in program._values if v.op == "field_problem_load")
    data = load.attrs["temporal_evaluation"]
    row = data["inputs"][0]
    assert row["transfer"]["source_slots"] == 4
    assert row["transfer"]["slots"] == 1
    assert len(row["transfer"]["parameter_names"]) == 4
    validate_temporal_field(program, load.inputs, data)
    from pops.time._program.detach import detach_compiled_program

    detached = detach_compiled_program(program)
    clone = next(v for v in detached._values if v.op == "field_problem_load")
    validate_temporal_field(detached, clone.inputs, clone.attrs["temporal_evaluation"])
    program.freeze()
    validate_temporal_field(program, load.inputs, data)


@pytest.mark.parametrize("mutation", ("name", "source", "width", "parameter"))
def test_temporal_field_rejects_changed_source_authority(mutation):
    spec = build_temporal_field()
    program = spec.case._time
    load = next(v for v in program._values if v.op == "field_problem_load")
    data = _json_ready(load.attrs["temporal_evaluation"])
    row = data["inputs"][0]
    if mutation == "name":
        row["history"] = "foreign"
    elif mutation == "source":
        row["source"] = load.inputs[0].id
    elif mutation == "width":
        row["transfer"]["source_slots"] = 3
    else:
        row["transfer"]["parameter_names"][0] = "foreign"
    with pytest.raises(ValueError):
        validate_temporal_field(program, load.inputs, data)


def test_temporal_evaluation_requires_a_declared_issued_keeper():
    with pytest.raises(TypeError):
        TemporalEvaluation(history="method.u", body=lambda *args: None)
    spec = build_temporal_field()
    program = spec.case._time
    marker = next(q for q in program._time_states.values() if q.block.name == "marker")
    with pytest.raises(ValueError, match="keep_history"):
        quadratic_evaluation(marker.prev)


def test_changing_authored_temporal_equations_changes_the_program_identity():
    exact = build_temporal_field()
    held = build_temporal_field(constant_control=True)
    assert exact.case._time._ir_hash() != held.case._time._ir_hash()


def test_issued_output_history_carries_its_authored_transfer_through_detachment():
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_history_identity import (
        history_space_identity,
        OUTPUT_HISTORY_PROJECTION_SPACE,
    )
    from pops.time._program.global_history_storage import validate_issuances

    spec = build_temporal_field(regrid_observation=True)
    program = spec.case._time
    store = next(
        node for node in program._values if node.op == "store_history" and node.name == "observed"
    )
    assert store.inputs[0].state_ref is None
    assert store.attrs["history_transfer"]["slots"] == 3
    assert history_space_identity(program, "observed") == OUTPUT_HISTORY_PROJECTION_SPACE
    validate_issuances(program)
    clone = detach_compiled_program(program)
    validate_issuances(clone)
    assert clone._ir_hash() == program._ir_hash()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(
        resolved.time,
        target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
    )
    assert 'register_history_transfer("observed"' in source
    assert "row.exact_weights" in source
    assert OUTPUT_HISTORY_PROJECTION_SPACE in source


def test_typed_output_transfer_cannot_replace_its_original_storage_authority():
    spec = build_temporal_field(regrid_observation=True)
    program = spec.case._time
    store = next(
        node for node in program._values if node.op == "store_history" and node.name == "observed"
    )
    attrs = dict(store.attrs)
    del attrs["global_field_storage"]
    object.__setattr__(store, "attrs", attrs)
    with pytest.raises(ValueError, match="originally issued|immutable authority"):
        program._serialize()
