"""ADC-652: ``U.next`` is an immutable commit endpoint, never a readable value."""
from __future__ import annotations

import pytest

from pops.ir import ValueExpr
from pops.time import Program, ProgramValue
from pops.time.handles import StateEndpointHandle
from pops.model import StateSpace


def test_next_is_a_cached_owner_qualified_endpoint():
    program = Program("endpoint")
    state = program.state("tracer", block="transport")

    endpoint = state.next

    assert isinstance(endpoint, StateEndpointHandle)
    assert not isinstance(endpoint, ProgramValue)
    assert endpoint is state.next
    assert endpoint.owner_path == program.owner_path
    assert endpoint.kind == "state_endpoint"
    assert endpoint.block == "transport"
    assert endpoint.state_name == "tracer"
    assert "transport" in endpoint.qualified_id


def test_typed_time_state_preserves_space_on_its_readable_current_value():
    program = Program("typed_endpoint")
    space = StateSpace("U", ("rho", "momentum"))
    state = program.state("U", block="transport", space=space)

    assert state.space is space
    assert state.n.space is space
    assert program._serialize()["nodes"][0]["space"]["components"] == ["rho", "momentum"]


def test_endpoint_has_no_value_escape_hatch_and_no_symbolic_algebra():
    program = Program("no_read")
    state = program.state("tracer", block="transport")
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
    endpoint = program.state("tracer", block="transport").next

    assert {endpoint: "destination"}[endpoint] == "destination"
    with pytest.raises(AttributeError, match="immutable"):
        endpoint.block = "other"
    with pytest.raises(AttributeError, match="immutable"):
        endpoint.local_id = "other"


def test_define_rejects_next_and_points_to_the_commit_door():
    program = Program("define_rejected")
    state = program.state("tracer", block="transport")

    with pytest.raises(TypeError, match=r"commit-only.*T\.commit"):
        program.define(state.next, state.n)


def test_commit_accepts_endpoint_and_value_as_two_distinct_roles():
    program = Program("commit")
    state = program.state("tracer", block="transport")

    program.commit(state.next, state.n)

    commits = program.commits()
    assert list(commits) == ["transport"]
    assert commits["transport"] is state.n


def test_commit_rejects_an_endpoint_owned_by_another_program():
    destination_program = Program("destination")
    foreign_program = Program("foreign")
    destination = destination_program.state("tracer", block="transport")
    foreign = foreign_program.state("tracer", block="transport")

    with pytest.raises(ValueError, match="different Program"):
        destination_program.commit(foreign.next, destination.n)


def test_commit_rejects_a_value_from_another_block():
    program = Program("cross_block")
    destination = program.state("U", block="destination")
    foreign = program.state("U", block="foreign")

    with pytest.raises(ValueError, match=r"block 'destination'.*block 'foreign'"):
        program.commit(destination.next, foreign.n)


def test_commit_requires_a_value_and_rejects_the_old_single_argument_endpoint_form():
    program = Program("missing_value")
    state = program.state("tracer", block="transport")

    with pytest.raises(TypeError, match="required positional argument"):
        program.commit(state.next)


def test_commit_many_accepts_only_endpoint_to_value_mappings():
    program = Program("commit_many")
    left = program.state("U", block="left")
    right = program.state("U", block="right")

    program.commit_many({left.next: left.n, right.next: right.n})

    commits = program.commits()
    assert set(commits) == {"left", "right"}
    assert commits["left"] is left.n
    assert commits["right"] is right.n


def test_commit_many_rejects_the_old_block_string_mapping():
    program = Program("no_string_targets")
    state = program.state("U", block="transport")

    with pytest.raises(TypeError, match=r"StateEndpointHandle.*block-name strings"):
        program.commit_many({"transport": state.n})
    assert program.commits() == {}


def test_commit_many_rejects_cross_block_as_one_atomic_group():
    program = Program("atomic_cross_block")
    left = program.state("U", block="left")
    right = program.state("U", block="right")

    with pytest.raises(ValueError, match=r"block 'right'.*block 'left'"):
        program.commit_many({left.next: left.n, right.next: left.n})
    assert program.commits() == {}, "the valid first entry must not be committed partially"


def test_commit_many_rejects_foreign_endpoint_as_one_atomic_group():
    program = Program("atomic_owner")
    foreign_program = Program("foreign_owner")
    left = program.state("U", block="left")
    foreign = foreign_program.state("U", block="foreign")

    with pytest.raises(ValueError, match="different Program"):
        program.commit_many({left.next: left.n, foreign.next: foreign.n})
    assert program.commits() == {}, "endpoint-owner validation must precede every write"


def test_commit_many_rejects_two_endpoints_for_the_same_block_atomically():
    program = Program("duplicate_destination")
    primary = program.state("U", block="transport")
    alias = program.state("alternate_name", block="transport")

    with pytest.raises(ValueError, match="more than one destination endpoint"):
        program.commit_many({primary.next: primary.n, alias.next: primary.n})
    assert program.commits() == {}


def _block_scalar_field(program, block, name):
    """Build a public solve_linear result whose rhs gives it an unambiguous block owner."""
    state = program.state("U", block=block)
    operator = program.matrix_free_operator("A_" + name)
    program.set_apply(operator, lambda _program, _out, value: value)
    return program.solve_linear(name, operator=operator, rhs=state.n, max_iter=1)


def test_scalar_field_linear_combine_preserves_the_single_known_block_for_commit():
    program = Program("scalar_provenance")
    endpoint = program.state("U", block="transport").next
    solved = _block_scalar_field(program, "transport", "solved")
    scratch = program.scalar_field("scratch")

    combined = program.linear_combine("combined", solved + scratch)

    assert solved.block == "transport"
    assert scratch.block is None
    assert combined.block == "transport"
    program.commit(endpoint, combined)
    assert program.commits()["transport"] is combined


def test_scalar_field_linear_combine_rejects_multiple_known_blocks():
    program = Program("scalar_cross_block")
    left = _block_scalar_field(program, "left", "left_solution")
    right = _block_scalar_field(program, "right", "right_solution")

    with pytest.raises(ValueError, match=r"different blocks.*left.*right"):
        program.linear_combine("invalid", left + right)
