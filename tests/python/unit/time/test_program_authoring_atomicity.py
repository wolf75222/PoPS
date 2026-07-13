"""Failed Program authoring calls leave no SSA, identity or region residue."""
from __future__ import annotations

from typed_program_support import typed_state

import pytest

from pops.time import Program


class _BuilderFailure(RuntimeError):
    pass


def _authoring_identity(program: Program) -> tuple[object, ...]:
    """Capture identity-sensitive state omitted from the serialized IR hash."""
    return (
        program._ir_hash(),
        program._next_id,
        program._next_region,
        id(program._values),
        tuple(id(value) for value in program._values),
        id(program._issued_values),
        tuple((key, id(value)) for key, value in program._issued_values.items()),
        id(program._recording),
        tuple((id(block), tuple(id(value) for value in block))
              for block in program._recording),
        id(program._recording_regions),
        tuple((key, id(entry[0]), entry[1])
              for key, entry in program._recording_regions.items()),
        tuple((region, tuple(sorted(imports)))
              for region, imports in program._region_imports.items()),
        tuple(program._state_spaces.items()),
        program._dt_bound,
    )


def test_dt_bound_builder_exception_restores_ids_values_regions_and_metadata():
    program = Program("atomic_dt_bound")
    leaked = []
    before = _authoring_identity(program)

    def fail_after_authoring(prog, cfl):
        leaked.extend((cfl, prog.hmin(), typed_state(prog, "temporary_block")))
        raise _BuilderFailure("dt bound failed")

    with pytest.raises(_BuilderFailure, match="dt bound failed"):
        program.set_dt_bound(fail_after_authoring)

    assert _authoring_identity(program) == before
    assert not program._recording
    assert "temporary_block" not in program._state_spaces
    assert all(program._issued_values.get(id(value)) is not value for value in leaked)
    assert program.hmin().id == before[1]


def test_dt_bound_post_validation_failure_is_atomic_too():
    program = Program("atomic_dt_bound_result")
    before = _authoring_identity(program)

    with pytest.raises(ValueError, match="Scalar"):
        program.set_dt_bound(lambda prog, _cfl: typed_state(prog, "not_a_bound"))

    assert _authoring_identity(program) == before


def test_check_invariant_validates_tolerance_before_creating_drift_nodes():
    program = Program("atomic_diagnostic")
    state = typed_state(program, "plasma")
    before_value = program.sum(state)
    after_value = program.sum(state)
    before = _authoring_identity(program)

    with pytest.raises(ValueError, match="positive"):
        program.check_invariant(
            "mass", before=before_value, after=after_value, tolerance=0)

    assert _authoring_identity(program) == before


def test_record_rolls_back_a_builder_exception_exactly():
    program = Program("atomic_record")
    state = typed_state(program, "plasma")
    leaked = []
    before = _authoring_identity(program)

    def fail(prog, current):
        leaked.append(prog.value("partial", current + current))
        raise _BuilderFailure("record failed")

    with pytest.raises(_BuilderFailure, match="record failed"):
        program._record(fail, state)

    assert _authoring_identity(program) == before
    assert all(program._issued_values.get(id(value)) is not value for value in leaked)


def test_while_body_failure_rolls_back_successful_condition_recording():
    program = Program("atomic_while")
    state = typed_state(program, "plasma")
    leaked = []
    before = _authoring_identity(program)

    def condition(prog, current):
        leaked.append(prog.norm2(current))
        return leaked[-1] > 0

    def body(prog, current):
        leaked.append(prog.value("partial", current + current))
        raise _BuilderFailure("body failed")

    with pytest.raises(_BuilderFailure, match="body failed"):
        program.while_(state, condition, body)

    assert _authoring_identity(program) == before
    assert all(program._issued_values.get(id(value)) is not value for value in leaked)


def test_control_flow_return_validation_failure_rolls_back_recorded_body():
    program = Program("atomic_range")
    state = typed_state(program, "plasma")
    before = _authoring_identity(program)

    with pytest.raises(ValueError, match="next-iteration State"):
        program.range(state, 2, lambda prog, current: prog.norm2(current))

    assert _authoring_identity(program) == before


def test_step_builder_failure_is_atomic():
    program = Program("atomic_step")
    state = typed_state(program, "plasma")
    before = _authoring_identity(program)

    def fail(prog):
        prog.value("partial", state + state)
        raise _BuilderFailure("step failed")

    with pytest.raises(_BuilderFailure, match="step failed"):
        program.step(fail)

    assert _authoring_identity(program) == before


def test_local_nonlinear_callback_failure_rolls_back_and_retry_reuses_ids():
    program = Program("atomic_local_nonlinear")
    initial = typed_state(program, "plasma")
    leaked = []
    before = _authoring_identity(program)

    def fail_after_nodes(prog, iterate, guess):
        first = prog.value("first", iterate + guess)
        second = prog.value("second", first + iterate)
        leaked.extend((iterate, guess, first, second))
        raise _BuilderFailure("local residual failed")

    with pytest.raises(_BuilderFailure, match="local residual failed"):
        program.solve_local_nonlinear(
            residual=fail_after_nodes, initial_guess=initial)

    failed_ids = tuple(value.id for value in leaked)
    assert _authoring_identity(program) == before
    assert all(program._issued_values.get(id(value)) is not value for value in leaked)

    retried = []

    def succeed(prog, iterate, guess):
        first = prog.value("first", iterate + guess)
        second = prog.value("second", first + iterate)
        retried.extend((iterate, guess, first, second))
        return second

    result = program.solve_local_nonlinear(
        residual=succeed, initial_guess=initial)

    assert tuple(value.id for value in retried) == failed_ids
    assert result.id == before[1] + len(failed_ids)


def test_set_apply_callback_failure_rolls_back_and_retry_reuses_ids():
    program = Program("atomic_apply_callback")
    operator = program.matrix_free_operator("helmholtz")
    leaked = []
    before = _authoring_identity(program)

    def fail_after_nodes(prog, out, in_):
        scratch = prog.scalar_field("scratch")
        laplacian = prog.laplacian(scratch, in_)
        leaked.extend((out, in_, scratch, laplacian))
        raise _BuilderFailure("apply failed")

    with pytest.raises(_BuilderFailure, match="apply failed"):
        program.set_apply(operator, fail_after_nodes)

    failed_ids = tuple(value.id for value in leaked)
    assert _authoring_identity(program) == before
    assert operator.attrs["apply_block"] is None
    assert all(program._issued_values.get(id(value)) is not value for value in leaked)

    retried = []

    def succeed(prog, out, in_):
        scratch = prog.scalar_field("scratch")
        laplacian = prog.laplacian(scratch, in_)
        retried.extend((out, in_, scratch, laplacian))
        return laplacian

    bound = program.set_apply(operator, succeed)

    assert tuple(value.id for value in retried) == failed_ids
    assert bound.attrs["apply_block"] is not None
    assert bound.attrs["apply_result"] is retried[-1]


def test_set_apply_post_builder_validation_failure_is_atomic():
    program = Program("atomic_apply_result")
    operator = program.matrix_free_operator("helmholtz")
    before = _authoring_identity(program)

    def return_wrong_type_after_nodes(prog, _out, in_):
        scratch = prog.scalar_field("scratch")
        prog.laplacian(scratch, in_)
        return typed_state(prog, "wrong_result")

    with pytest.raises(ValueError, match="must return the result scalar_field"):
        program.set_apply(operator, return_wrong_type_after_nodes)

    assert _authoring_identity(program) == before
    assert operator.attrs["apply_block"] is None
    assert "wrong_result" not in program._state_spaces
