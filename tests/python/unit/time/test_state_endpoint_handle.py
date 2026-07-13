"""ADC-652: ``U.next`` is an immutable commit endpoint, never a readable value."""
from __future__ import annotations

from typed_program_support import commits_by_block, typed_state

import pytest

from pops.ir import ValueExpr
from pops.time import FailRun, Program, ProgramValue
from pops.time.handles import StateEndpointHandle
from pops.model import Module, StateSpace
from pops.problem import Case


def _at_endpoint(program, state, *, name="final"):
    return program.value(name, state.n, at=state.next.point)


def test_next_is_a_cached_owner_qualified_endpoint():
    program = Program("endpoint")
    state = typed_state(program, "transport", state_name="tracer")

    endpoint = state.next

    assert isinstance(endpoint, StateEndpointHandle)
    assert not isinstance(endpoint, ProgramValue)
    assert endpoint is state.next
    assert endpoint.owner_path == program.owner_path
    assert endpoint.kind == "state_endpoint"
    assert endpoint.block is state.block
    assert endpoint.block.local_id == "transport"
    assert endpoint.state is state.state
    assert endpoint.state.local_id == "tracer"
    assert "transport" in endpoint.qualified_id


def test_typed_time_state_preserves_space_on_its_readable_current_value():
    program = Program("typed_endpoint")
    space = StateSpace("U", ("rho", "momentum"))
    state = typed_state(program, "transport", state_name="U", space=space)

    assert state.space is space
    assert state.n.space is space
    assert program._serialize()["nodes"][0]["space"]["components"] == ["rho", "momentum"]


def test_endpoint_has_no_value_escape_hatch_and_no_symbolic_algebra():
    program = Program("no_read")
    state = typed_state(program, "transport", state_name="tracer")
    endpoint = state.next

    assert not hasattr(endpoint, "value")
    assert not hasattr(endpoint, "_as_value")
    with pytest.raises(TypeError):
        _ = endpoint + state.n
    with pytest.raises(TypeError):
        _ = 0.5 * endpoint
    with pytest.raises(TypeError, match="commit-only"):
        ValueExpr(endpoint)


def test_endpoint_is_immutable_hashable_declaration_identity():
    program = Program("identity")
    endpoint = typed_state(program, "transport", state_name="tracer").next

    assert {endpoint: "destination"}[endpoint] == "destination"
    with pytest.raises(AttributeError, match="immutable"):
        endpoint.block = "other"
    with pytest.raises(AttributeError, match="immutable"):
        endpoint.local_id = "other"


def test_define_rejects_next_and_points_to_the_commit_door():
    program = Program("define_rejected")
    state = typed_state(program, "transport", state_name="tracer")

    with pytest.raises(TypeError, match=r"commit-only.*T\.commit"):
        program.value(state.next, state.n)


def test_commit_accepts_endpoint_and_value_as_two_distinct_roles():
    program = Program("commit")
    state = typed_state(program, "transport", state_name="tracer")

    final = _at_endpoint(program, state)
    program.commit(state.next, final)

    assert program.commits() == {state.state: final}


def test_commit_rejects_an_endpoint_owned_by_another_program():
    destination_program = Program("destination")
    foreign_program = Program("foreign")
    destination = typed_state(destination_program, "transport", state_name="tracer")
    foreign = typed_state(foreign_program, "transport", state_name="tracer")

    with pytest.raises(ValueError, match="different Program"):
        destination_program.commit(foreign.next, destination.n)


def test_commit_rejects_a_value_from_another_block():
    program = Program("cross_block")
    destination = typed_state(program, "destination", state_name="U")
    foreign = typed_state(program, "foreign", state_name="U")

    with pytest.raises(ValueError, match=r"block 'destination'.*block 'foreign'"):
        program.commit(destination.next, foreign.n)


def test_commit_requires_a_value_and_rejects_the_old_single_argument_endpoint_form():
    program = Program("missing_value")
    state = typed_state(program, "transport", state_name="tracer")

    with pytest.raises(TypeError, match="required positional argument"):
        program.commit(state.next)


def test_commit_many_accepts_only_endpoint_to_value_mappings():
    program = Program("commit_many")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")

    left_final = _at_endpoint(program, left, name="left_final")
    right_final = _at_endpoint(program, right, name="right_final")
    program.commit_many({left.next: left_final, right.next: right_final})

    assert program.commits() == {left.state: left_final, right.state: right_final}


def test_commit_many_rejects_the_old_block_string_mapping():
    program = Program("no_string_targets")
    state = typed_state(program, "transport", state_name="U")

    with pytest.raises(TypeError, match=r"StateEndpointHandle.*block-name strings"):
        program.commit_many({"transport": state.n})
    assert program.commits() == {}


def test_commit_many_rejects_cross_block_as_one_atomic_group():
    program = Program("atomic_cross_block")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")

    with pytest.raises(ValueError, match=r"block 'right'.*block 'left'"):
        left_final = _at_endpoint(program, left, name="left_final")
        wrong_right = program.value(
            "wrong_right", left.n, at=right.next.point)
        program.commit_many({left.next: left_final, right.next: wrong_right})
    assert program.commits() == {}, "the valid first entry must not be committed partially"


def test_commit_many_rejects_foreign_endpoint_as_one_atomic_group():
    program = Program("atomic_owner")
    foreign_program = Program("foreign_owner")
    left = typed_state(program, "left", state_name="U")
    foreign = typed_state(foreign_program, "foreign", state_name="U")
    left_final = _at_endpoint(program, left, name="left_final")

    with pytest.raises(ValueError, match="different Program"):
        program.commit_many({left.next: left_final, foreign.next: foreign.n})
    assert program.commits() == {}, "endpoint-owner validation must precede every write"


def test_commit_many_accepts_distinct_qualified_states_in_the_same_block():
    program = Program("multi_state_destination")
    module = Module("transport_model")
    primary_space = module.state_space("U", ("density",))
    alternate_space = module.state_space("V", ("tracer",))
    block = Case(name="multi-state-case").block("transport", module)
    primary = program.state(block[module.state_handle(primary_space)])
    alternate = program.state(block[module.state_handle(alternate_space)])

    primary_final = program.value(
        "primary_final", primary.n, at=primary.next.point)
    alternate_final = program.value(
        "alternate_final", alternate.n, at=alternate.next.point)
    program.commit_many({primary.next: primary_final, alternate.next: alternate_final})

    assert program.commits() == {
        primary.state: primary_final,
        alternate.state: alternate_final,
    }


def _block_scalar_field(program, block, name):
    """Build a public solve_linear result whose rhs gives it an unambiguous block owner."""
    state = typed_state(program, block, state_name="U")
    operator = program.matrix_free_operator("A_" + name)
    program.set_apply(operator, lambda _program, _out, value: value)
    return program.solve_linear(
        name, operator=operator, rhs=state.n, max_iter=1).consume(action=FailRun())


def test_scalar_field_linear_combine_preserves_the_single_known_block_for_commit():
    program = Program("scalar_provenance")
    endpoint = typed_state(program, "transport", state_name="U").next
    solved = _block_scalar_field(program, "transport", "solved")
    scratch = program.scalar_field("scratch")

    combined = program.value(
        "combined", solved + scratch, at=endpoint.point)

    assert solved.block is endpoint.block
    assert scratch.block is None
    assert combined.block is endpoint.block
    program.commit(endpoint, combined)
    assert commits_by_block(program)["transport"] is combined


def test_scalar_field_linear_combine_rejects_multiple_known_blocks():
    program = Program("scalar_cross_block")
    left = _block_scalar_field(program, "left", "left_solution")
    right = _block_scalar_field(program, "right", "right_solution")

    with pytest.raises(ValueError, match=r"different blocks.*left.*right"):
        program.value("invalid", left + right)
