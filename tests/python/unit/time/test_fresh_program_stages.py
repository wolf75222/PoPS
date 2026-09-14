"""M1.4: fresh exact stages survive capture, optimization and detachment."""
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
import json

import pytest

from typed_program_support import typed_state

from pops.time import Program, StagePoint, TimePoint


def test_shorthand_has_fresh_identity_at_equal_exact_coordinates():
    program = Program("fresh_stages")
    first = program.stage("predictor", c=Fraction(1, 3))
    second = program.stage("predictor", c=Fraction(1, 3))

    assert type(first) is StagePoint
    assert first.time == second.time == TimePoint(program.clock, Fraction(1, 3))
    assert first != second
    assert len({first: "first value", second: "second value"}) == 2
    assert (first.identity, second.identity) == (0, 1)
    assert first.to_data()["partitions"]["main"]["offset"] == {
        "kind": "rational", "numerator": "1", "denominator": "3"}
    decimal = program.stage("decimal", c=Decimal("0.125"))
    assert decimal.time.offset.to_data() == {"kind": "decimal", "value": "0.125"}
    with pytest.raises(FrozenInstanceError):
        first.name = "changed"


def test_long_stage_spelling_and_cached_temporal_handles_remain_compatible():
    program = Program("cached_stages")
    state = typed_state(program, "fluid", state_name="U")
    point = StagePoint("predictor", {"main": TimePoint(program.clock, Fraction(1, 2))})
    handle = state.stage("predictor", point=point)
    assert point.identity is None
    assert set(point.to_data()) == {"schema_version", "name", "partitions"}
    assert state.stage("predictor", point=StagePoint(
        "predictor", {"main": TimePoint(program.clock, Fraction(1, 2))})) is handle
    fresh = program.stage("predictor", c=Fraction(1, 2))
    assert fresh != point
    with pytest.raises(ValueError, match="different StagePoint"):
        state.stage("predictor", point=fresh)


def _authored_stages(first_name="first", second_name="second", *, duplicate=False):
    program = Program("stage_method")
    state = typed_state(program, "fluid", state_name="U")
    first = program.stage(first_name, c=Fraction(1, 2))
    second = program.stage(second_name, c=Fraction(1, 2))
    left = program.value("left", 2 * state.n, at=first)
    right = program.value("right", 2 * state.n, at=second)
    result = left + right
    if duplicate:
        repeated = program.value("repeated", 2 * state.n, at=first)
        result = result + repeated
    final = program.value("final", result, at=state.next.point)
    program.commit(state.next, final)
    return program


def test_stage_labels_do_not_change_the_program_semantic_identity():
    original = _authored_stages()
    relabeled = _authored_stages("diagnostic before", "diagnostic after")
    assert original._semantic_serialize() == relabeled._semantic_serialize()


def test_fresh_stages_survive_serialization_rebuild_and_compiled_graph():
    program = _authored_stages()
    program.stage("unused declaration", c=1)
    serialized = program._serialize(include_provenance=False)
    rebuilt = program._rebuild(lambda _value: True)
    assert rebuilt._serialize(include_provenance=False) == serialized
    assert rebuilt._ir_hash() == program._ir_hash()
    assert rebuilt.stage("next declaration", c=1).identity == 3

    graph = program.to_graph()
    data = json.loads(json.dumps(graph.to_data()))
    identities = {
        node["point"]["identity"] for node in data["nodes"]
        if "identity" in node.get("point", {})
    }
    assert identities == {0, 1}
    assert program._serialize(include_provenance=False) == serialized


def test_temporal_stage_handle_rebuild_preserves_fresh_identity_and_resolution():
    program = Program("fresh_temporal_handle")
    state = typed_state(program, "fluid", state_name="U")
    point = program.stage("predictor", c=Fraction(1, 2))
    handle = state.stage("predictor", point=point)
    program.value(handle, 2 * state.n)

    rebuilt = program._rebuild(lambda _value: True)
    rebound = next(iter(rebuilt._time_stage_handles.values()))
    assert rebound.point.identity == point.identity
    assert rebound.value.point == rebound.point
    assert rebound.value.prog is rebuilt
    assert rebuilt._ir_hash() == program._ir_hash()


def test_cse_distinguishes_stage_identity_but_reuses_identical_same_stage_work():
    program = _authored_stages(duplicate=True)
    optimized = program.eliminate_common_subexpressions()
    stages = [value for value in optimized._values if type(value.point) is StagePoint]
    assert len(stages) == 2
    assert {value.point.identity for value in stages} == {0, 1}
    assert len(optimized._values) == len(program._values) - 1
    optimized.validate()


def test_failed_and_frozen_stage_declarations_do_not_allocate_identities():
    program = Program("stage_mutation")
    with pytest.raises(ValueError, match="non-empty string"):
        program.stage("", c=0)
    with pytest.raises(TypeError, match="scalar"):
        program.stage("invalid coefficient", c=object())
    assert program.stage("first", c=0).identity == 0
    program.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        program.stage("late", c=1)
