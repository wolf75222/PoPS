"""ADC-652 temporal handles are immutable declarations; Program owns every resolution table."""
from __future__ import annotations

from typed_program_support import commits_by_block, typed_state

import pytest

from pops.model import Handle, StateSpace
from pops.time import (
    HistoryHandle, Program, StageHandle, StagePoint, StateEndpointHandle,
    TimePoint, TimeState,
)
from pops.time._history.persistence import Interval
import pops.time as time_api
import pops.time.handles as temporal_handles


def _stage(state, name="predictor", offset=1):
    return state.stage(
        name,
        point=StagePoint(name, {"main": TimePoint(state.clock, offset)}),
    )


def test_private_mutable_temporal_aliases_are_removed():
    assert not hasattr(time_api, "_Version")
    assert not hasattr(time_api, "_Prev")
    assert not hasattr(temporal_handles, "_Version")
    assert not hasattr(temporal_handles, "_Prev")


def test_temporal_handle_families_are_immutable_hashable_handle_values():
    program = Program("immutable_temporal")
    state = typed_state(program, "fluid", state_name="U")
    stage = _stage(state)
    history = state.prev
    endpoint = state.next

    expected_kinds = {
        state: "time_state",
        stage: "state_stage",
        history: "state_history",
        endpoint: "state_endpoint",
    }
    for handle in (state, stage, history, endpoint):
        assert isinstance(handle, Handle)
        assert {handle: "ok"}[handle] == "ok"
        assert not hasattr(handle, "_value")
        assert handle.kind == expected_kinds[handle]
        assert handle.schema_version == 1
        assert handle.owner_path == program.owner_path
        assert handle.qualified_id.startswith("pops.handle.v1::")
        assert handle.inspect()["owner_path"] == program.owner_path.presentation().to_data()

    mutations = (
        (state, "block", "other"),
        (state, "_program", object()),
        (stage, "key", 99),
        (stage, "_key", 99),
        (stage, "_value", state.n),
        (history, "lag", 9),
        (endpoint, "state_name", "other"),
    )
    for handle, attribute, value in mutations:
        with pytest.raises(AttributeError, match="immutable"):
            setattr(handle, attribute, value)


def test_program_caches_every_temporal_declaration_and_resolution():
    program = Program("temporal_tables")
    space = StateSpace("U", ("rho", "momentum"))
    state = typed_state(program, "fluid", state_name="U", space=space)

    assert state is typed_state(program, "fluid", state_name="U", space=space)
    assert state.n is state.n
    predictor = _stage(state)
    assert _stage(state) is predictor
    assert state.prev is state.prev
    assert state.next is state.next
    assert program._time_states[(state.block, state.state, state.clock)] is state
    assert program._time_current_values[state] is state.n
    assert program._time_stage_handles[(state, "predictor")] is predictor
    assert program._time_history_handles[(state, 1)] is state.prev
    assert program._time_endpoint_handles[state] is state.next


def test_stage_resolution_is_program_owned_single_assignment():
    program = Program("stage_table")
    state = typed_state(program, "fluid", state_name="U")
    stage = _stage(state)

    with pytest.raises(ValueError, match="stage 'predictor' is undefined"):
        _ = stage.value
    result = program.value(stage, state.n)
    assert stage.value is result
    assert program._time_stage_values[stage] is result
    assert not hasattr(stage, "_value")
    assert program.value("negated", -stage).vtype == "state"
    operator = program._linear_source("relaxation")
    assert (operator @ stage).vtype == "rhs"
    with pytest.raises(ValueError, match="SSA stage already defined"):
        program.value(stage, state.n)


def test_stage_definition_refuses_non_state_values_without_partial_assignment():
    program = Program("stage_type")
    state = typed_state(program, "fluid", state_name="U")
    stage = _stage(state)
    scalar = program.norm2(state.n)
    scalar_name = scalar.name

    with pytest.raises(TypeError, match=r"T\.value stage: expected a State value"):
        program.value(stage, scalar)
    assert scalar.name == scalar_name
    assert stage not in program._time_stage_values
    with pytest.raises(ValueError, match="stage 'predictor' is undefined"):
        _ = stage.value

    scratch = program.scalar_field("scratch")
    with pytest.raises(TypeError, match=r"affine term must be a State/Rate"):
        program.value(stage, 2 * scratch)
    assert stage not in program._time_stage_values
    with pytest.raises(ValueError, match="stage 'predictor' is undefined"):
        _ = stage.value


def test_history_resolution_and_configuration_live_on_program():
    program = Program("history_table")
    state = typed_state(program, "fluid", state_name="U")
    lag1 = state.prev

    with pytest.raises(ValueError, match="keep_history first"):
        _ = lag1.value
    program.keep_history(state, depth=2)
    assert state.prev(1) is lag1
    lag2 = state.prev(2)
    assert lag1.value.op == "history" and lag1.value.attrs["lag"] == 1
    assert lag2.value.op == "history" and lag2.value.attrs["lag"] == 2
    assert program._time_history_values[lag1] is lag1.value
    with pytest.raises(ValueError, match="already configured"):
        program.keep_history(state, depth=2)


def test_history_refuses_a_cold_start_policy_the_runtime_cannot_lower():
    program = Program("history_cold_start")
    state = typed_state(program, "fluid", state_name="U")

    with pytest.raises(TypeError, match=r"cold_start must be CopyCurrent"):
        program.keep_history(state, depth=2, cold_start=object())
    assert state not in program._time_history_configs
    assert "fluid.U" not in program._history_persistence


def test_history_configuration_snapshots_and_freezes_the_supplied_policy():
    program = Program("history_policy_snapshot")
    state = typed_state(program, "fluid", state_name="U")
    supplied = Interval(3)

    program.keep_history(state, depth=3, checkpoint_policy=supplied)
    configured = program._time_history_configs[state][2]
    assert isinstance(configured, Interval)
    assert configured is not supplied and configured.k == 3

    supplied.k = 1
    assert configured.k == 3
    with pytest.raises(RuntimeError, match="frozen"):
        configured.k = 1


def test_history_configuration_is_not_published_when_ring_provenance_is_invalid():
    program = Program("history_atomic")
    declared = StateSpace("U", ("rho",))
    conflicting = StateSpace("U", ("rho", "momentum"))
    state = typed_state(program, "fluid", state_name="U", space=declared)
    program.history(
        "fluid.U", space=conflicting, block=state.block,
        state_ref=state.state)
    values_before = tuple(program._values)
    persistence_before = program._history_persistence["fluid.U"]

    with pytest.raises(ValueError, match=r"incompatible structures"):
        program.keep_history(state, depth=2)
    assert state not in program._time_history_configs
    assert state not in program._time_history_stores
    assert program._history_persistence["fluid.U"] == persistence_before
    assert len(program._values) == len(values_before)
    assert all(
        current is before
        for current, before in zip(program._values, values_before, strict=True)
    )


def test_history_configuration_refuses_incompatible_existing_ring_shapes():
    narrow_program = Program("history_narrow")
    narrow_state = typed_state(narrow_program, "fluid", state_name="U")
    narrow_program.history("fluid.U", ncomp=1)
    with pytest.raises(ValueError, match=r"narrow scalar history ring"):
        narrow_program.keep_history(narrow_state, depth=2)
    assert narrow_state not in narrow_program._time_history_configs

    deep_program = Program("history_depth")
    deep_state = typed_state(deep_program, "fluid", state_name="U")
    deep_program.history(
        "fluid.U", lag=3, space=deep_state.space,
        block=deep_state.block, state_ref=deep_state.state)
    with pytest.raises(ValueError, match=r"smaller than the already-declared lag 3"):
        deep_program.keep_history(deep_state, depth=2)
    assert deep_state not in deep_program._time_history_configs


def test_temporal_identity_is_owner_qualified_between_programs():
    first = Program("same_name")
    second = Program("same_name")
    left = typed_state(first, "fluid", state_name="U")
    right = typed_state(second, "fluid", state_name="U")

    pairs = (
        (left, right),
        (_stage(left), _stage(right)),
        (left.prev, right.prev),
        (left.next, right.next),
    )
    for a, b in pairs:
        assert a != b and a.owner_path != b.owner_path

    with pytest.raises(ValueError, match="different Program"):
        first.value(_stage(right), left.n)
    with pytest.raises(ValueError, match="different Program"):
        first.value(right.prev, left.n)
    with pytest.raises(ValueError, match="different Program"):
        first.keep_history(right, depth=1)
    with pytest.raises(ValueError, match="different Program"):
        first.commit(right.next, left.n)


def test_forged_equal_handles_cannot_bypass_program_issuance_tables():
    program = Program("forgery")
    state = typed_state(program, "fluid", state_name="U")
    issued_stage = _stage(state)
    issued_history = state.prev
    issued_endpoint = state.next

    forged_state = TimeState(
        program, state.block, state.state, clock=state.clock, space=state.space)
    forged_stage = StageHandle(
        program=program, block=state.block, state=state.state, key="predictor",
        clock=state.clock, point=issued_stage.point, space=state.space)
    forged_history = HistoryHandle(
        program=program, block=state.block, state=state.state, lag=1,
        clock=state.clock, space=state.space)
    forged_endpoint = StateEndpointHandle(
        owner=program.owner_path, block=state.block, state=state.state,
        clock=state.clock, space=state.space)

    assert forged_state == state
    assert forged_stage == issued_stage
    assert forged_history == issued_history
    assert forged_endpoint == issued_endpoint
    with pytest.raises(ValueError, match="not issued"):
        _ = forged_state.n
    with pytest.raises(ValueError, match="not issued"):
        program.value(forged_stage, state.n)
    with pytest.raises(ValueError, match="not issued"):
        _ = forged_history.value
    with pytest.raises(ValueError, match="not issued"):
        program.commit(forged_endpoint, state.n)


def test_program_rebuild_reowns_and_remaps_temporal_tables():
    program = Program("temporal_rebuild")
    state = typed_state(program, "fluid", state_name="U")
    stage = _stage(state)
    program.value(stage, state.n)
    program.keep_history(state, depth=1)
    _ = state.prev.value
    final = program.value("final", stage.value, at=state.next.point)
    program.commit(state.next, final)

    rebuilt = program.eliminate_dead_nodes()
    rebuilt_state = typed_state(rebuilt, "fluid", state_name="U")
    rebuilt_stage = _stage(rebuilt_state)
    rebuilt_history = rebuilt_state.prev
    rebuilt_endpoint = rebuilt_state.next

    assert rebuilt_state.owner_path == rebuilt.owner_path
    assert rebuilt_state.owner_path != state.owner_path
    assert rebuilt_state is not state
    assert rebuilt_stage is not stage and rebuilt_stage != stage
    assert rebuilt_history is not state.prev and rebuilt_history != state.prev
    assert rebuilt_endpoint is not state.next and rebuilt_endpoint != state.next
    assert all(handle.schema_version == 1 for handle in (
        rebuilt_state, rebuilt_stage, rebuilt_history, rebuilt_endpoint))
    assert rebuilt_state.n is rebuilt._time_current_values[rebuilt_state]
    assert rebuilt_stage.value is rebuilt._time_stage_values[rebuilt_stage]
    assert rebuilt_history.value is rebuilt._time_history_values[rebuilt_history]
    assert rebuilt._time_history_configs[rebuilt_state][0] == 1
    assert rebuilt._time_history_stores[rebuilt_state].op == "store_history"
    assert commits_by_block(rebuilt)["fluid"].point == rebuilt_endpoint.point
    with pytest.raises(ValueError, match="different Program"):
        rebuilt._require_stage(stage, "rebuild")
    with pytest.raises(ValueError, match="different Program"):
        rebuilt._require_endpoint(state.next, "rebuild")
