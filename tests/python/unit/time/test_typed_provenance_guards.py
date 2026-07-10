"""ADC-652: structural State/Rate, owner and authoring-region non-bypass tests."""
from __future__ import annotations

import pytest

from pops.model import Rate, StateSpace
from pops.time import Program
from pops.time.values import ProgramValue


def test_state_declaration_has_one_complete_space_contract_per_block():
    state = StateSpace("U", ("rho", "mx"))
    same = StateSpace("U", ("rho", "mx"))
    different = StateSpace("U", ("rho", "energy"))
    program = Program("state_contract")

    program.state("fluid", space=state)
    program.state("fluid", space=same)
    with pytest.raises(ValueError, match="typed and untyped"):
        program.state("fluid")
    with pytest.raises(ValueError, match="incompatible structures"):
        program.state("fluid", space=different)

    untyped = Program("untyped_contract")
    untyped.state("fluid")
    with pytest.raises(ValueError, match="typed and untyped"):
        untyped.state("fluid", space=state)


def test_rate_is_strict_tangent_of_complete_state_space():
    first = StateSpace("U", ("rho", "mx"))
    second = StateSpace("U", ("rho", "energy"))
    assert Rate(first) != Rate(second)
    assert Rate(first).base_space is first
    with pytest.raises(TypeError, match="StateSpace"):
        Rate("U")


def test_linear_combine_rejects_cross_block_structural_and_typed_untyped_bypasses():
    state = StateSpace("U", ("rho", "mx"))
    different = StateSpace("U", ("rho", "energy"))
    program = Program("combine_guards")
    left = program.state("left", space=state)
    right = program.state("right", space=state)
    with pytest.raises(ValueError, match="different blocks"):
        program.linear_combine(left + right)

    wrong_rate = program._new(
        "rhs", "rhs", (left,), {}, "wrong_rate", "left", space=Rate(different))
    with pytest.raises(ValueError, match="incompatible structures"):
        program.linear_combine(left + wrong_rate)

    untyped_rate = program._new("rhs", "rhs", (left,), {}, "untyped_rate", "left")
    with pytest.raises(ValueError, match="typed and untyped"):
        program.linear_combine(left + untyped_rate)


def test_state_preserving_ops_and_timestate_history_keep_space():
    state = StateSpace("U", ("rho", "mx"))
    program = Program("propagation")
    temporal = program.state("U", block="fluid", space=state)
    current = temporal.n
    rate = program._rhs_legacy(state=current, sources=[])
    assert rate.space == Rate(state)
    assert rate.logical_shape["n_comp"] == len(state.components)
    combined = program.linear_combine(current + program.dt * rate)
    assert combined.space == state
    assert program.fill_boundary(combined).space == state
    assert program.project(combined).space == state

    ranged = program.range(
        combined, 2, lambda builder, value: builder.linear_combine(value))
    conditional = program.if_(
        ranged, program.norm2(ranged) > 0,
        lambda builder, value: builder.linear_combine(value))
    looped = program.while_(
        conditional, lambda builder, value: builder.norm2(value) > 0,
        lambda builder, value: builder.linear_combine(value))
    assert ranged.space == conditional.space == looped.space == state

    program.keep_history(temporal, depth=2)
    previous = temporal.prev(1)
    assert previous.space == state and previous.block == "fluid"
    program.commit(temporal.next, looped)
    assert program.validate() is True


def test_callback_results_cannot_cross_program_or_region_boundaries():
    state = StateSpace("U", ("rho",))
    first = Program("first")
    second = Program("second")
    left = first.state("fluid", space=state)
    right = second.state("fluid", space=state)

    with pytest.raises(ValueError, match="different Program"):
        first.while_(
            left, lambda _builder, _value: second.norm2(right) > 0,
            lambda builder, value: builder.linear_combine(value))
    with pytest.raises(ValueError, match="different Program"):
        first.range(left, 1, lambda _builder, _value: right)
    with pytest.raises(ValueError, match="different Program"):
        first.if_(left, second.norm2(right) > 0, lambda builder, value: value)

    operator = first.matrix_free_operator("A")
    foreign_field = second.scalar_field("foreign")
    with pytest.raises(ValueError, match="different Program"):
        first.set_apply(operator, lambda _builder, _out, _in: foreign_field)
    with pytest.raises(ValueError, match="different Program"):
        first.solve_local_nonlinear(
            residual=lambda _builder, _iterate: right, initial_guess=left)
    foreign_scalar = second.hmin()
    with pytest.raises(ValueError, match="different Program"):
        first.set_dt_bound(lambda _builder, _cfl: foreign_scalar)


def test_subblock_value_and_fabricated_value_cannot_escape_to_top_level():
    state = StateSpace("U", ("rho",))
    program = Program("regions")
    temporal = program.state("U", block="fluid", space=state)
    captured = []

    def body(builder, value):
        result = builder.linear_combine(value)
        captured.append(result)
        return result

    output = program.range(temporal.n, 1, body)
    with pytest.raises(ValueError, match="region"):
        program.linear_combine(captured[0])
    with pytest.raises(ValueError, match="sub-block value"):
        program.commit(temporal.next, captured[0])

    fake = ProgramValue(
        program, 999, "state", "state", (), {}, "fabricated", "fluid", space=state)
    with pytest.raises(ValueError, match="not authored"):
        program.linear_combine(fake)

    program.commit(temporal.next, output)
    assert program.validate() is True


def test_identity_control_body_is_valid_but_keeps_region_metadata_through_rebuild():
    program = Program("identity_body")
    temporal = program.state("U", block="fluid")
    output = program.range(temporal.n, 2, lambda _builder, value: value)
    program.commit(temporal.next, output)
    assert program.validate() is True
    rebuilt = program.eliminate_dead_nodes()
    assert rebuilt.validate() is True


def test_where_dot_and_solve_linear_reject_cross_block_fields():
    state = StateSpace("U", ("rho",))
    program = Program("field_layout_guards")
    left = program.state("left", space=state)
    right = program.state("right", space=state)
    mask = program.cell_gt(right, 0)
    with pytest.raises(ValueError, match="mask"):
        program.where(mask, left, left)
    with pytest.raises(ValueError, match="same block"):
        program.dot(left, right)

    operator = program.matrix_free_operator("A")
    operator = program.set_apply(operator, lambda _builder, _out, value: value)
    with pytest.raises(ValueError, match="same block"):
        program.solve_linear(operator=operator, rhs=left, initial_guess=right, max_iter=2)


def test_freeze_guards_metadata_mutations_transactionally():
    program = Program("frozen")
    temporal = program.state("U", block="fluid")
    program.commit(temporal.next, temporal.n)
    before = program._ir_hash()
    program.freeze()

    with pytest.raises(RuntimeError, match="frozen"):
        program.history("new_history")
    with pytest.raises(RuntimeError, match="frozen"):
        program.commit(temporal.next, temporal.n)
    with pytest.raises(AttributeError, match="immutable identity anchor"):
        program.name = "renamed"
    with pytest.raises(RuntimeError, match="frozen"):
        program.dt = program.dt
    with pytest.raises(RuntimeError, match="irreversible"):
        program._frozen = False
    with pytest.raises(RuntimeError, match="frozen"):
        program.capture_source_locations()
    assert program._ir_hash() == before


def test_program_name_cannot_diverge_from_issued_handle_owners_before_freeze():
    program = Program("stable_owner")
    state = program.state("U", block="fluid")

    with pytest.raises(AttributeError, match="immutable identity anchor"):
        program.name = "renamed"
    assert state.owner_path == program.owner_path
    assert program.name == "stable_owner"


def test_replacing_a_committed_record_keeps_commit_inspection_canonical():
    program = Program("canonical_commit")
    temporal = program.state("U", block="fluid")
    current = temporal.n
    program.commit(temporal.next, current)
    renamed = program.define("renamed", current)
    assert program.commits()["fluid"] is renamed
    assert "renamed" in program.dump_operator_ir()
