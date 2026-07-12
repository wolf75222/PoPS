"""Program.freeze deeply detaches authoring tables and preserves pure reads."""
from typed_program_support import typed_state

from types import MappingProxyType

import pytest

from pops.time import Program, StagePoint, TimePoint


def _program():
    program = Program("deep-program")
    state = typed_state(program, "fluid", state_name="U")
    final = program.linear_combine("final", state.n, at=state.next.point)
    program.commit(state.next, final)
    return program


def test_program_identity_cannot_be_deleted_and_frozen_storage_is_sealed():
    program = _program()
    with pytest.raises(AttributeError, match="immutable identity anchor"):
        del program.name

    program.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        del program._values
    with pytest.raises(RuntimeError, match="frozen"):
        program._next_id = 999
    with pytest.raises(TypeError):
        program._commits["fluid"] = program._values[0]


def test_program_freeze_detaches_stale_tables_and_preserves_all_read_views():
    program = _program()
    stale_values = program._values
    stale_issued = program._issued_values
    stale_commits = program._commits
    stale_spaces = program._state_spaces
    before_hash = program._ir_hash()
    before_serialized = program._serialize()
    before_report = program.inspect().to_dict()

    program.freeze()
    assert isinstance(program._values, tuple)
    assert isinstance(program._issued_values, MappingProxyType)
    assert isinstance(program._commits, MappingProxyType)

    stale_values.clear()
    stale_issued.clear()
    stale_commits.clear()
    stale_spaces["fluid"] = "detached mutation"

    assert program._ir_hash() == before_hash
    assert program._serialize() == before_serialized
    assert program.inspect().to_dict() == before_report


def test_frozen_program_codegen_is_repeatable_and_does_not_install_caches():
    program = _program()
    program.freeze()
    before = program._ir_hash()

    first = program.emit_cpp_program()
    second = program.emit_cpp_program()

    assert first == second
    assert program._ir_hash() == before
    assert not hasattr(program, "_coupled_scratch")
    assert not hasattr(program, "_when_tokens")


def test_frozen_temporal_handles_preserve_materialized_pure_reads():
    program = Program("frozen-temporal-reads")
    state = typed_state(program, "fluid", state_name="U")
    current = state.n
    point = StagePoint("predictor", {"main": TimePoint(state.clock, 1)})
    stage = state.stage("predictor", point=point)
    defined = program.define(stage, current)
    program.keep_history(state, depth=1)
    previous = state.prev.value
    endpoint = state.next
    final = program.linear_combine("final", defined, at=endpoint.point)
    program.commit(endpoint, final)

    program.freeze()

    # Defining the stage gives the current state record a canonical named replacement with the same
    # SSA id. The frozen getter must return that canonical record without trying to republish it.
    assert state.n is defined
    assert state.stage("predictor", point=point) is stage
    assert stage.value is defined
    assert state.prev.value is previous
    assert state.next is endpoint


def test_frozen_temporal_handles_refuse_new_lazy_declarations_clearly():
    program = Program("frozen-temporal-missing")
    state = typed_state(program, "fluid", state_name="U")
    program.freeze()

    with pytest.raises(RuntimeError, match="frozen"):
        _ = state.n
    with pytest.raises(RuntimeError, match="frozen"):
        state.stage(
            "late", point=StagePoint("late", {"main": TimePoint(state.clock, 1)}))
    with pytest.raises(RuntimeError, match="frozen"):
        _ = state.prev
    with pytest.raises(RuntimeError, match="frozen"):
        _ = state.next
