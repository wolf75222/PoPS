"""Unread storage-owner routes: real public contracts, no Native/JIT proof."""
from __future__ import annotations

import json
import re

import pops
import pytest

from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata
from tests.review.test_sol61_history_storage_owner_received import _storage_only


def strings(cpp, symbol):
    body = re.search(r'extern "C" const char\* '+symbol+r'\(int index\) \{(.*?)\n\}', cpp, re.S)[1]
    return {int(i): json.loads(value) for i, value in re.findall(
        r'case (\d+): return ("(?:[^"\\]|\\.)*");', body)}


def test_unread_declared_storage_owner_routes_without_creating_state_ssa():
    _case, program, _blocks, _states, _problem, value, _field, _point, owner, issued = _storage_only(5)
    program.store_history("unused-storage", value, depth=1, owner_block=owner)
    before = tuple(program._values), program._next_id, tuple(program._commits.items())
    indices = program._block_indices()
    assert owner in indices
    assert indices[owner] == 3
    assert value.block is value.state_ref is value.space is None
    assert issued not in program._time_current_values
    assert not any(node.op == "state" and node.block is owner for node in program._values)
    assert not any(state.block_ref is owner for state in program._commits)
    metadata = _emit_checkpoint_shape_metadata(program)
    identities = strings(metadata, "pops_program_checkpoint_history_state_identity")
    assert len(identities) == 1
    image = json.loads(identities[0])
    assert image["ncomp"] == 1
    assert image["contract"] == "pops.program.global-field-history-storage@1"
    assert tuple(program._values) == before[0] and program._next_id == before[1]
    assert tuple(program._commits.items()) == before[2]


def completed_case(*, adaptive, storage="unread"):
    from pops.analytic import x
    from pops.initial import InitialCondition
    from pops.lib.initial import Analytic
    from pops.projection import ConservativeCellAverage
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.layouts import Uniform
    from pops.time import FixedDt
    case, program, blocks, states, _problem, value, _field, _point, owner, issued = _storage_only(5)
    if storage != "absent":
        program.store_history("unused-storage", value, depth=1,
                              owner_block=owner if storage == "unread" else blocks[0])
    # Preserve the actual solved problem's already read physical sources. The
    # storage-only TimeState is never read, evolved or committed.
    for index, (block, state) in enumerate(zip(blocks, states, strict=True)):
        time = program.state(block[state])
        program.commit(time.next, program.value("preserve-%d" % index, 1*time.n, at=time.next.point))
    storage_state = issued.state.declaration_ref or issued.state
    all_states = tuple(zip((*blocks, owner), (*states, storage_state), strict=True))
    for block, state in all_states:
        frame = block._instance_registry._blocks[block.local_id]["model"].frame
        width = 5 if block is owner else len(tuple(state))
        case.initials.add(InitialCondition(state=block[state],
            value=Analytic(frame=frame, components=tuple(0*x(frame) for _ in range(width))),
            projection=ConservativeCellAverage()))
    program.step_strategy(FixedDt(.125))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(3, 5), periodic=PeriodicAxes(frame.axes))
    layout = Uniform(grid)
    if adaptive:
        from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
            Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
        from pops.lib.amr import StateTransfer
        from pops.math import ValueExpr
        from pops.params import RuntimeParam
        from pops.time import every
        from pops.layouts import AMR
        transfer = AMRTransfer()
        for block, state in all_states:
            transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam("physical-refinement-threshold", default=.5))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(ValueExpr(blocks[0][states[0]])[0] > case.value(threshold)), Buffer(cells=0)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.synchronous())
    return case, layout, program, owner, issued, value


@pytest.mark.parametrize("adaptive", (False, True))
def test_actual_case_materializes_storage_owner_without_evolving_it(adaptive):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case, layout, original, owner, issued, observation = completed_case(adaptive=adaptive)
    assert owner not in [value.block for value in original._values if value.op == "state"]
    resolved = pops.resolve(pops.validate(case), layout=layout)
    program = resolved.time
    indices = program._block_indices()
    storage_owner = next(block for block in indices if block.local_id == owner.local_id)
    assert indices[storage_owner] == 3
    assert tuple(block.name for block in resolved.blocks) == ("thermal", "matter", "spectator", "storage-only")
    assert len(resolved.initial_condition_plan.bindings) == 4
    assert not any(value.op == "state" and value.block is storage_owner for value in program._values)
    assert not any(state.block_ref is storage_owner for state in program._commits)
    assert observation.block is None and issued not in original._time_current_values
    cpp = emit_cpp_program(program, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target="amr_system" if adaptive else "system")
    names = re.search(r'extern "C" const char\* pops_program_block_name\(int i\) \{(.*?)\n\}', cpp, re.S)[1]
    table = {int(index): json.loads(name) for index, name in re.findall(
        r'case (\d+): return ("(?:[^"\\]|\\.)*");', names)}
    assert table == {0: "thermal", 1: "matter", 2: "spectator", 3: "storage-only"}
    assert ("ctx.store_global_field_history(" if adaptive else "ctx.store_history(") in cpp
    assert program._serialize()["version"] == 16


@pytest.mark.parametrize("mutation", ("descriptor", "owner", "width", "missing-store"))
def test_route_refuses_mutated_original_issuance_before_adding_owner(mutation):
    _case, program, blocks, _states, _problem, value, _field, _point, owner, _issued = _storage_only(5)
    node = program.store_history("immutable", value, depth=1, owner_block=owner)
    if mutation == "descriptor":
        object.__setattr__(node, "attrs", dict(node.attrs) | {"global_field_storage":
            dict(node.attrs["global_field_storage"]) | {"ncomp": True}})
    elif mutation == "owner":
        program._history_blocks["immutable"] = blocks[0]
    elif mutation == "width":
        program._histories_ncomp["immutable"] = True
    else:
        program._values[:] = [value for value in program._values if value is not node]
    before = tuple(program._values), program._next_id, tuple(program._commits.items())
    with pytest.raises(ValueError, match="global field history"):
        program._block_indices()
    assert (tuple(program._values), program._next_id, tuple(program._commits.items())) == before


@pytest.mark.parametrize("boundary", ("freeze", "rebuild", "detach", "graph"))
def test_unread_owner_survives_actual_snapshot_boundaries(boundary):
    from pops.time._program.detach import detach_compiled_program
    _case, program, _blocks, _states, _problem, value, _field, _point, owner, _issued = _storage_only(5)
    program.store_history("unused-storage", value, depth=1, owner_block=owner)
    before = program._ir_hash()
    if boundary == "freeze":
        result = program.freeze()
    elif boundary == "rebuild":
        result = program._rebuild(lambda _value: True)
    elif boundary == "detach":
        result = detach_compiled_program(program)
    else:
        program.to_graph()
        result = program
    assert result._ir_hash() == before
    mapped = next(block for block in result._block_indices() if block.local_id == owner.local_id)
    assert result._block_indices()[mapped] == 3
    assert not any(node.op == "state" and node.block is mapped for node in result._values)
    assert len(strings(_emit_checkpoint_shape_metadata(result),
                       "pops_program_checkpoint_history_state_identity")) == 1


def test_two_rings_share_one_route_and_existing_routes_keep_their_order():
    _case, program, blocks, _states, _problem, value, _field, _point, owner, _issued = _storage_only(5)
    prior = program._block_indices()
    program.store_history("z-last", value, depth=1, owner_block=owner)
    program.store_history("a-first", value, depth=1, owner_block=owner)
    program.store_history("existing", value, depth=1, owner_block=blocks[1])
    mapped = program._block_indices()
    assert tuple(mapped.items())[:len(prior)] == tuple(prior.items())
    assert mapped[owner] == 3 and len(mapped) == 4
    assert len(strings(_emit_checkpoint_shape_metadata(program),
                       "pops_program_checkpoint_history_state_identity")) == 3


def legacy_receipts():
    """Run unchanged in the parent and fixed checkout; hash full real images."""
    from hashlib import sha256
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    output = {}
    for adaptive in (False, True):
        for storage in ("absent", "existing"):
            case, layout, *_ = completed_case(adaptive=adaptive, storage=storage)
            resolved = pops.resolve(pops.validate(case), layout=layout)
            program = resolved.time
            cpp = emit_cpp_program(program, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                                   target="amr_system" if adaptive else "system")
            image = json.dumps(program._serialize(), sort_keys=True, separators=(",", ":")).encode()
            output[f"{adaptive}/{storage}"] = {
                "ir": sha256(image).hexdigest(), "cpp": sha256(cpp.encode()).hexdigest(),
                "program_hash": program._ir_hash(),
            }
    return output


if __name__ == "__main__":
    print(json.dumps(legacy_receipts(), sort_keys=True, indent=2))
