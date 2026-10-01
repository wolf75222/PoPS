"""Real public IR19 barrier admission, without SDK/native execution."""
from pathlib import Path

import pytest
from pops.fields import CellMidpoint, CellVolumeMeasure, DirectSpatialInteraction, SpatialInteractionKernel
from pops.model import FieldSpace, PhysicalDimension
from pops.time._program.spatial_interaction import interaction_contract
from tests.review.test_sol61_amr_public_original import authored, emit

ROOT = Path(__file__).resolve().parents[2]


def witness(*, width=3, selected=0, order=None, uniform=False, budget=2**24):
    case, layout, program, _, _ = authored(width=width, order=order, seed=False, uniform=uniform)
    source = [node for node in program._values if node.op == "field_component"][selected]
    owner = next(node for node in program._values if node.op == "state")
    output = FieldSpace("closed_map", components=("integral",), sampling="cell_center",
                        frame=owner.space.frame, support=owner.space.support, clock=owner.space.clock)
    result = program.spatial_interaction(source, SpatialInteractionKernel(2, lambda x,y:1+x[0]*y[0]),
        output_space=output, measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(budget), source_scope="completed_original", owner_block=owner.block)
    program.sum_component(result, 0)
    return case, layout, program, source, result


@pytest.mark.parametrize("width,selected,order", [(1,0,None),(3,2,(2,0,1))])
def test_real_original_source_remains_global_and_output_uses_issued_storage_only(width,selected,order):
    case, layout, program, source, result = witness(width=width,selected=selected,order=order)
    assert source.space is source.block is source.state_ref is None
    assert result.block is result.state_ref is None
    assert interaction_contract(result)[0] is source
    assert program._serialize()["version"] == 19
    code, resolved = emit(case, layout)
    assert resolved.time._serialize()["version"] == 19
    assert "ctx.seal_original_field_source(" in code
    assert "ctx.prepare_closed_original_interaction(" in code
    assert "ctx.closed_original_interaction(" in code
    assert "ctx.reduce_closed_original_interaction(" in code
    assert "const std::array<int, 1> closed_components_%d{%d}" % (result.id,source.attrs["component"]) in code
    # Actual emitted setup is after the consumed Outcome, not in the local body.
    solve = code[code.index("ctx.seal_original_field_source("):]
    assert solve.index("ctx.prepare_closed_original_interaction(") < solve.index("ctx.closed_original_interaction(")


@pytest.mark.parametrize("kind,value", [("tuple_component",True),("tuple_width",3.0),("region",False)])
def test_typed_source_metadata_cannot_be_resealed(kind,value):
    *_, program, _, result = witness()
    attrs = dict(result.attrs)
    metadata = dict(attrs["closed_field_source"])
    metadata[kind] = value
    attrs["closed_field_source"] = metadata
    program._replace_value(result, attrs=attrs)
    with pytest.raises(ValueError, match="originally issued"):
        program._serialize()


def test_removed_contract_cannot_downgrade_an_issued_snapshot():
    *_, program, _, result = witness()
    attrs = dict(result.attrs)
    attrs["contract"] = "pops.spatial-interaction@1"
    attrs.pop("closed_field_source")
    program._replace_value(result, attrs=attrs)
    with pytest.raises(ValueError, match="originally issued"):
        program._serialize()


def test_same_case_valid_owner_substitution_cannot_remint_snapshot():
    *_, program, source, result = witness()
    from pops.time._program.spatial_interaction import _closed_source_contract
    owner = next(node.block for node in program._values if node.op == "state" and node.block != result.attrs["closed_field_source"]["owner_block"])
    attrs = dict(result.attrs)
    attrs["closed_field_source"] = _closed_source_contract(program, source, owner, result.space)
    program._replace_value(result, attrs=attrs)
    with pytest.raises(ValueError, match="originally issued"):
        program._serialize()


def test_unknown_source_units_are_not_borrowed_from_storage_state():
    *_, program, source, result = witness()
    output = FieldSpace("known", components=("integral",), units=(PhysicalDimension(),),
                        sampling="cell_center", frame=result.space.frame, support=result.space.support, clock=result.space.clock)
    before = program._serialize()
    with pytest.raises(ValueError, match="unknown field units"):
        program.spatial_interaction(source, SpatialInteractionKernel(2,lambda x,y:1),output_space=output,
            measure=CellVolumeMeasure(),quadrature=CellMidpoint(),realization=DirectSpatialInteraction(2**20),
            source_scope="completed_original",owner_block=result.attrs["closed_field_source"]["owner_block"])
    assert program._serialize() == before


def test_uniform_has_explicit_missing_accept_snapshot_provider():
    case, layout, *_ = witness(uniform=True)
    with pytest.raises(NotImplementedError, match="Uniform needs its own Accept snapshot"):
        emit(case, layout)


def test_freeze_and_real_resolved_detach_keep_the_original_authority():
    case, layout, program, *_ = witness()
    program.freeze()
    graph = program.to_graph()
    assert graph.to_data() and program._serialize()["version"] == 19
    _, resolved = emit(case, layout)
    assert resolved.time._serialize()["version"] == 19


def test_actual_native_source_stamp_and_complete_output_path_are_load_bearing():
    provider = (ROOT/"include/pops/runtime/amr/hierarchy_tensor_solver_provider.hpp").read_text()
    stamp = "staged->owner->original_accepted_candidate_ = staged->candidate;"
    assert provider.count(stamp) == 1
    original = provider[provider.index("SolveOutcome stage_original_field_candidate_collectively"):]
    assert original.index("original_accepted_candidate_ = nullptr;") < original.index("if (!report.solved_value_available())")
    assert "const std::shared_ptr<PreparedAmrFieldResidual<Dim>>& core" in (ROOT/"include/pops/runtime/program/amr_program_context_spatial_interaction.inc").read_text()
    accept = provider[provider.index(stamp)-150:provider.index(stamp)+180]
    assert "restore_or_terminate_" in accept and "++staged->owner->original_acceptance_generation_" in accept
    native = (ROOT/"include/pops/runtime/program/amr_program_context_spatial_interaction.inc").read_text().split("// IR19:")[1]
    assert "accepted_original_candidate_generation(&tower, lane)" in native
    assert "issued->image.emplace_back(provider->solution(level))" in native
    assert "source->image.at(level)" in native
    assert "for (std::size_t level = 0; level < levels.size(); ++level)" in native
    assert "field.distribution().replicated() && lane.rank() != 0" in native
    assert "result->source->candidate()" in native
    assert "all_ranks_agree_exact_ordered_byte_pairs" in native
    assert "facade_->prepared_amr_block_state(result->storage_owner, level)" in native
    assert "field.ncomp() != storage.ncomp()" not in native


def test_storage_only_declared_timestate_has_a_route_without_a_physical_read():
    case, _layout, program, source, result = witness()
    owner = result.attrs["closed_field_source"]["owner_block"]
    model = owner._instance_registry._blocks[owner.local_id]["model"]
    extra = case.block("storage_only", model)
    state = next(value.state for value in program._time_states.values() if value.block is owner)
    program.state(extra[state.declaration_ref])
    assert not any(node.op == "state" and node.block is extra for node in program._values)
    second = program.spatial_interaction(source, SpatialInteractionKernel(2,lambda x,y:1), output_space=result.space,
        measure=CellVolumeMeasure(), quadrature=CellMidpoint(), realization=DirectSpatialInteraction(2**24),
        source_scope="completed_original", owner_block=extra)
    assert extra in program._block_indices()
    assert not any(node.op == "state" and node.block is extra for node in program._values)
    assert second.block is None


def test_gradient_waits_for_an_authentic_complete_ghost_consumer():
    *_, program, _, result = witness()
    output = program.scalar_field("gradient_scratch", ncomp=2)
    program.gradient(output, result)
    with pytest.raises(NotImplementedError, match="ghost port"):
        program._serialize()


def test_another_genuine_read_of_the_same_tuple_cannot_replace_the_issued_source():
    *_, program, source, result = witness()
    duplicate = program._new("scalar_field", "field_component", source.inputs, dict(source.attrs),
                             "another_genuine_read", None, point=source.point)
    from pops.fields._observation_contract import validate_field_observation
    assert validate_field_observation(duplicate)[1:] == validate_field_observation(source)[1:]
    object.__setattr__(result, "inputs", (duplicate,))  # fault injection after issuance
    with pytest.raises(ValueError, match="originally issued"):
        program._serialize()


@pytest.mark.parametrize("component", [False, True, 0.0])
def test_global_component_authoring_is_exact_integer(component):
    *_, program, source, result = witness()
    with pytest.raises(ValueError, match="scalar selection"):
        program.spatial_interaction(source,SpatialInteractionKernel(2,lambda x,y:1),output_space=result.space,
            measure=CellVolumeMeasure(),quadrature=CellMidpoint(),realization=DirectSpatialInteraction(2**24),
            source_scope="completed_original",owner_block=result.attrs["closed_field_source"]["owner_block"],components=(component,))
