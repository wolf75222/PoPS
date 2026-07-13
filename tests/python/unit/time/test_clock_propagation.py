"""ADC-662: exact clock and evaluation-point ownership in the Program graph."""
import json
from fractions import Fraction

import pytest

from typed_program_support import typed_state

from pops.time import Program, SampleAndHold
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time.schedule import AcceptedStep, Every, Hold, Schedule
from pops.time.program_value_validation import validate_input_clocks


def _stage(state, name="predictor", offset=Fraction(1, 2)):
    return state.stage(
        name,
        point=StagePoint(name, {"main": TimePoint(state.clock, offset)}),
    )


def test_time_state_current_history_endpoint_and_named_stage_have_exact_points():
    program = Program("clock_points")
    state = typed_state(program, "fluid", state_name="U")
    stage = _stage(state)

    assert state.clock is program.clock
    assert state.point == TimePoint(program.clock)
    assert state.n.point == state.point
    assert stage.key == "predictor" and stage.point.time.offset.to_data() == {
        "kind": "rational", "numerator": "1", "denominator": "2"}
    assert state.prev.point == TimePoint(program.clock, step=-1)
    assert state.next.point == TimePoint(program.clock, step=1)


def test_stage_keys_are_names_and_redeclaration_cannot_change_the_point():
    program = Program("named_stages")
    state = typed_state(program, "fluid", state_name="U")
    with pytest.raises(ValueError, match="non-empty string"):
        state.stage(1, point=StagePoint("one", {"main": TimePoint(program.clock)}))

    first = _stage(state, "predictor", Fraction(1, 3))
    assert state.stage("predictor", point=first.point) is first
    with pytest.raises(ValueError, match="different StagePoint"):
        _stage(state, "predictor", Fraction(2, 3))


def test_partitioned_stage_keeps_distinct_abscissae_on_one_clock():
    program = Program("partitioned_stage")
    state = typed_state(program, "fluid", state_name="U")
    point = StagePoint("coupled", {
        "explicit": TimePoint(program.clock, Fraction(1, 3)),
        "implicit": TimePoint(program.clock, Fraction(2, 3)),
    })
    stage = state.stage("coupled", point=point)

    assert stage.point.time_for("explicit").offset != stage.point.time_for("implicit").offset
    with pytest.raises(ValueError, match="ambiguous partition times"):
        _ = stage.point.time


def test_schedule_is_explicitly_clock_bound():
    clock = Clock("macro")
    schedule = Schedule(Every(AcceptedStep(clock), 4), off=Hold())

    assert schedule.clock is clock
    assert schedule.trigger.n == 4
    with pytest.raises(TypeError, match="exact Clock"):
        AcceptedStep("macro")


def test_cross_clock_edge_requires_a_synchronize_node():
    program = Program("cross_clock")
    fast = Clock("fast", owner=program.owner_path)
    state = typed_state(program, "fluid", state_name="U")
    value = state.n

    with pytest.raises(ValueError, match="explicit Program synchronization node"):
        validate_input_clocks(
            (value,), TimePoint(fast), "test cross-clock edge")
    with pytest.raises(TypeError, match="SynchronizationRelation"):
        program.synchronize(value, at=TimePoint(fast), relation={"kind": "hold"})

    # The sole permitted foreign-clock input is already the output of an explicit synchronize op.
    synchronized = program.synchronize(
        value, at=TimePoint(fast), relation=SampleAndHold(), name="synced")
    validate_input_clocks(
        (synchronized,), TimePoint(fast), "test synchronized edge")


def test_fixed_ratio_subcycle_has_an_explicit_schedule_and_native_sync_lowering():
    program = Program("nested_clock")
    state = typed_state(program, "fluid", state_name="U")
    fast = Clock("fast", owner=program.owner_path)
    child = program.synchronize(
        state.n, at=TimePoint(fast), relation=SampleAndHold(), name="to_fast")
    advanced = program.subcycle(
        child, clock=fast, within=program.clock, count=3,
        body_fn=lambda P, value: P.value("child_copy", 1 * value),
        name="fast_ticks",
    )
    returned = program.synchronize(
        advanced, at=state.next.point, relation=SampleAndHold(), name="to_macro")
    program.commit(state.next, returned)

    temporal = program.temporal_manifest()
    assert json.loads(json.dumps(temporal)) == temporal
    assert temporal["primary_clock"] == program.clock.qualified_id
    assert {row["id"]: row["ticks_per_macro"] for row in temporal["clocks"]} == {
        program.clock.qualified_id: 1,
        fast.qualified_id: 3,
    }
    assert temporal["subcycles"] == [{
        "node_id": advanced.id,
        "parent_clock": program.clock.qualified_id,
        "child_clock": fast.qualified_id,
        "count": 3,
    }]
    graph_loop = next(
        node for node in program.to_graph().nodes
        if getattr(node, "loop_kind", None) == "subcycle")
    assert graph_loop.parent_clock == program.clock
    assert graph_loop.clock == fast and graph_loop.count == 3
    source = program.emit_cpp_program()
    assert json.loads(json.dumps(program.temporal_manifest())) == temporal
    from pops.runtime._temporal_restart import TemporalRestartState

    accepted = TemporalRestartState()
    accepted.configure_program(temporal, time=0.0, macro_step=0)
    assert accepted.cursor_for_clock(fast)["tick"] == 0
    assert "ctx.subcycle_scope" in source
    assert "synchronize_sample_and_hold" in source
    assert "for (int i" in source and "< 3" in source


def test_clock_schedule_refuses_an_unrelated_cross_clock_without_an_execution_relation():
    program = Program("orphan_clock")
    state = typed_state(program, "fluid", state_name="U")
    orphan = Clock("orphan", owner=program.owner_path)
    program.synchronize(
        state.n, at=TimePoint(orphan), relation=SampleAndHold(), name="orphan")

    with pytest.raises(ValueError, match="has no subcycle relation"):
        program.temporal_manifest()


def test_dt_dependent_values_and_commits_require_exact_output_points():
    program = Program("explicit_points")
    state = typed_state(program, "fluid", state_name="U")
    rate = program._rhs_legacy(state=state.n, sources=[])

    with pytest.raises(ValueError, match="cannot infer an evaluation point"):
        program.value("ambiguous", state.n + program.dt * rate)

    stage_point = StagePoint(
        "predictor", {"explicit": TimePoint(program.clock, Fraction(1, 2))})
    stage = program.value(
        "stage", state.n + Fraction(1, 2) * program.dt * rate, at=stage_point)
    assert stage.point == stage_point

    with pytest.raises(ValueError, match="endpoint is at"):
        program.commit(state.next, stage)

    final = program.value(
        "final", state.n + program.dt * rate, at=state.next.point)
    program.commit(state.next, final)
    assert program.commits()[state.state].point == state.next.point
