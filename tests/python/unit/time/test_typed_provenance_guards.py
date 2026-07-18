"""ADC-652: structural State/Rate, owner and authoring-region non-bypass tests."""
from __future__ import annotations

import inspect
from pathlib import Path

from typed_program_support import commits_by_block, state_refs, typed_state

import pytest

from pops.model import Rate, StateSpace
from pops.provenance import ProvenanceRecord, SourceSpan
from pops.time import Program
from pops.time.values import ProgramValue


def _direct_provenance(program: Program) -> ProvenanceRecord:
    span = SourceSpan(__file__, 0)
    return ProvenanceRecord(
        primary=span, owner=program.owner_path,
        authoring_api="tests.ProgramValue", origins=(span,))


def test_state_declaration_has_one_complete_space_contract_per_block():
    state = StateSpace("U", ("rho", "mx"))
    program = Program("state_contract")

    block, declaration = state_refs(program, "fluid", space=state)
    qualified = block[declaration]
    first = program.state(qualified)
    assert program.state(qualified) is first

    foreign = Program("foreign_contract")
    foreign_block, foreign_declaration = state_refs(foreign, "fluid")
    with pytest.raises(ValueError, match="one Program cannot combine blocks from"):
        program.state(foreign_block[foreign_declaration])


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
    left = typed_state(program, "left", space=state)
    right = typed_state(program, "right", space=state)
    with pytest.raises((TypeError, ValueError), match="different blocks|not supported"):
        program.value("cross_block", left + right)

    wrong_rate = program._new(
        "rhs", "rhs", (left,), {}, "wrong_rate", left.block, space=Rate(different))
    with pytest.raises(ValueError, match="incompatible structures"):
        program.value("wrong_structure", left + wrong_rate)

    untyped_rate = program._new(
        "rhs", "rhs", (left,), {}, "untyped_rate", left.block)
    with pytest.raises(ValueError, match="complete StateSpace"):
        program.value("untyped_mix", left + untyped_rate)


def test_state_preserving_ops_and_timestate_history_keep_space():
    state = StateSpace("U", ("rho", "mx"))
    program = Program("propagation")
    temporal = typed_state(program, "fluid", state_name="U", space=state)
    current = temporal.n
    rate = program.rhs(state=current, terms=[])
    assert rate.space == Rate(state)
    assert rate.logical_shape["n_comp"] == len(state.components)
    endpoint = temporal.next
    combined = program.value(
        "combined", current + program.dt * rate, at=endpoint.point)
    assert combined.space == state
    assert program.fill_boundary(combined).space == state
    assert program.project(combined).space == state

    ranged = program.range(
        combined, 2, lambda builder, value: builder.value("range_step", 1 * value))
    conditional = program.branch(
        program.norm2(ranged) > 0,
        lambda builder: builder.value("branch_true", 1 * ranged),
        lambda _builder: ranged)
    looped = program.while_(
        conditional, lambda builder, value: builder.norm2(value) > 0,
        lambda builder, value: builder.value("while_step", 1 * value))
    assert ranged.space == conditional.space == looped.space == state

    program.keep_history(temporal, depth=2)
    previous = temporal.prev(1)
    assert previous.space == state and previous.block.local_id == "fluid"
    program.commit(endpoint, looped)
    assert program.validate() is True


def test_callback_results_cannot_cross_program_or_region_boundaries():
    state = StateSpace("U", ("rho",))
    first = Program("first")
    second = Program("second")
    left = typed_state(first, "fluid", space=state)
    right = typed_state(second, "fluid", space=state)

    with pytest.raises(ValueError, match="different Program"):
        first.while_(
            left, lambda _builder, _value: second.norm2(right) > 0,
            lambda builder, value: builder.value("while_step", 1 * value))
    with pytest.raises(ValueError, match="different Program|same block"):
        first.range(left, 1, lambda _builder, _value: right)
    with pytest.raises(ValueError, match="different Program"):
        first.branch(
            second.norm2(right) > 0,
            lambda _builder: left,
            lambda _builder: left)

    operator = first.matrix_free_operator("A")
    foreign_field = second.scalar_field("foreign")
    with pytest.raises(ValueError, match="different Program"):
        first.set_apply(operator, lambda _builder, _out, _in: foreign_field)
    with pytest.raises(ValueError, match="different Program"):
        from pops.solvers.nonlinear import LocalNewton
        from pops.time import LocalResidual
        first.solve(
            LocalResidual(lambda _builder, _iterate: right, left), solver=LocalNewton())
    foreign_scalar = second.hmin()
    with pytest.raises(ValueError, match="different Program"):
        first.set_dt_bound(lambda _builder, _cfl: foreign_scalar)


def test_subblock_value_and_fabricated_value_cannot_escape_to_top_level():
    state = StateSpace("U", ("rho",))
    program = Program("regions")
    temporal = typed_state(program, "fluid", state_name="U", space=state)
    captured = []

    def body(builder, value):
        result = builder.value("captured", 1 * value)
        captured.append(result)
        return result

    output = program.range(temporal.n, 1, body)
    with pytest.raises(ValueError, match="not present|region"):
        program.value("escaped", captured[0])
    with pytest.raises(ValueError, match="sub-block value"):
        program.commit(temporal.next, captured[0])

    fake = ProgramValue(
        program, 999, "state", "state", (), {}, "fabricated", temporal.block,
        space=state, point=temporal.n.point, provenance=_direct_provenance(program))
    with pytest.raises(ValueError, match="not authored"):
        program.value("fabricated_escape", fake)

    output_next = program.value(
        "output_next", 1 * output, at=temporal.next.point)
    program.commit(temporal.next, output_next)
    assert program.validate() is True


def test_identity_control_body_is_valid_but_keeps_region_metadata_through_rebuild():
    program = Program("identity_body")
    temporal = typed_state(program, "fluid", state_name="U")
    output = program.range(temporal.n, 2, lambda _builder, value: value)
    output_next = program.value(
        "output_next", 1 * output, at=temporal.next.point)
    program.commit(temporal.next, output_next)
    assert program.validate() is True
    rebuilt = program.eliminate_dead_nodes()
    assert rebuilt.validate() is True


def test_where_dot_and_linear_problem_reject_cross_block_fields():
    from pops.linalg import LinearProblem
    from pops.solvers import CG

    state = StateSpace("U", ("rho",))
    program = Program("field_layout_guards")
    left = typed_state(program, "left", space=state)
    right = typed_state(program, "right", space=state)
    mask = program.cell_gt(right, 0)
    with pytest.raises(ValueError, match="mask"):
        program.where(mask, left, left)
    with pytest.raises(ValueError, match="same block"):
        program.dot(left, right)

    operator = program.matrix_free_operator("A")
    operator = program.set_apply(operator, lambda _builder, _out, value: value)
    with pytest.raises(ValueError, match="same block"):
        program.solve(
            LinearProblem(operator, left, initial_guess=right, nullspace=None),
            solver=CG(max_iter=2),
    )


def test_source_capture_skips_every_nested_pops_time_builder_frame():
    program = Program("source_location")
    temporal = typed_state(program, "fluid", state_name="U")
    program.capture_source_locations()

    call_line = inspect.currentframe().f_lineno + 1
    value = program.value("authored_here", temporal.n)

    filename, line = value.source_location.rsplit(":", 1)
    assert Path(filename).resolve() == Path(__file__).resolve()
    assert int(line) == call_line


def test_freeze_guards_metadata_mutations_transactionally():
    program = Program("frozen")
    temporal = typed_state(program, "fluid", state_name="U")
    current_next = program.value(
        "current_next", 1 * temporal.n, at=temporal.next.point)
    program.commit(temporal.next, current_next)
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
    state = typed_state(program, "fluid", state_name="U")

    with pytest.raises(AttributeError, match="immutable identity anchor"):
        program.name = "renamed"
    assert state.owner_path == program.owner_path
    assert program.name == "stable_owner"


def test_replacing_a_committed_record_keeps_commit_inspection_canonical():
    program = Program("canonical_commit")
    temporal = typed_state(program, "fluid", state_name="U")
    current = program.value(
        "current_next", 1 * temporal.n, at=temporal.next.point)
    program.commit(temporal.next, current)
    renamed = program.value("renamed", current)
    assert commits_by_block(program)["fluid"] is renamed
    assert "renamed" in program.dump_operator_ir()
