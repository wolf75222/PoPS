"""Detached authoring Program -> immutable ProgramGraph snapshot."""
from __future__ import annotations

from fractions import Fraction

import pytest

from pops.model import Module, Rate
from pops.problem import Case
from pops.time import Program
from pops.time.graph import (
    Branch, Commit, Loop, OperatorCall, ProgramValue, Region, RegionCapture, StateRead,
    Synchronize, ValueRef,
)
from pops.time.points import Clock, TimePoint


def _program(*, with_operator=False):
    model = Module("transport")
    space = model.state_space("U", ("u",))
    state_declaration = model.state_handle(space)
    rate = None
    if with_operator:
        rate = model.operator(
            "decay",
            signature=(space,) >> Rate(space),
            kind="local_source",
            lowering={"source": "default"},
            expr={"test": "decay"},
        )
    problem = Case(name="case")
    block = problem.block("fluid", model)
    program = Program("step")._bind_operators(model)
    state = program.state(block, state_declaration)
    if rate is None:
        result = program.value("u_next", state.n, at=state.next.point)
    else:
        called = program.call(rate, state.n, name="decay_rate")
        result = program.value(
            "u_next", state.n + program.dt * called, at=state.next.point)
    program.commit(state.next, result)
    return program


def test_to_graph_detaches_values_handles_and_write_only_commit():
    program = _program()
    before = program._ir_hash()

    graph = program.to_graph()

    assert program._ir_hash() == before
    assert graph.name == "step"
    assert [type(node) for node in graph.nodes] == [StateRead, ProgramValue, Commit]
    assert graph.nodes[1].inputs == (graph.ref(graph.nodes[0]),)
    assert graph.nodes[-1].target.to_data()["endpoint"] == "next"
    assert graph.nodes[-1].point == TimePoint(graph.nodes[-1].clock, step=1)
    assert not hasattr(graph, "_values")
    assert all(not hasattr(node, "prog") for node in graph.nodes)
    assert "#authoring=" not in repr(graph.to_data())


def test_to_graph_maps_typed_operator_call_without_retaining_registry():
    program = _program(with_operator=True)

    graph = program.to_graph()
    call = next(node for node in graph.nodes if type(node) is OperatorCall)
    operator = call.operator.to_data()

    assert operator["handle"]["kind"] == "local_source"
    assert operator["lowering"]["op"] == "rhs"
    assert call.inputs == (graph.ref(graph.nodes[0]),)
    assert not hasattr(call, "operator_registry")


def test_to_graph_preserves_exact_cross_clock_synchronization():
    model = Module("clocked")
    space = model.state_space("U", ("u",))
    problem = Case(name="case")
    block = problem.block("fluid", model)
    program = Program("clock-transfer")
    state = program.state(block, model.state_handle(space))
    fast = Clock("fast", owner=program.owner_path)
    target = TimePoint(fast, Fraction(1, 3))
    from pops.time.synchronization import SampleAndHold

    program.synchronize(state.n, at=target, relation=SampleAndHold(), name="sample")

    graph = program.to_graph()
    sync = next(node for node in graph.nodes if type(node) is Synchronize)

    assert sync.source_clock.name == "macro"
    assert sync.target_clock.name == "fast"
    assert sync.point.offset.to_python() == Fraction(1, 3)
    assert {clock.name for clock in graph.clocks} == {"macro", "fast"}


def test_to_graph_is_a_deep_snapshot_of_serialized_attrs():
    program = _program()
    graph = program.to_graph()
    before = graph.to_data()

    # The source builder remains independent and mutable; a later node cannot alter the snapshot.
    state = next(value for value in program._values if value.op == "state")
    program.value("later", state)

    assert graph.to_data() == before


def test_to_graph_converts_branch_range_and_while_blocks_to_structured_regions():
    model = Module("control")
    space = model.state_space("U", ("u",))
    block = Case(name="case").block("fluid", model)
    program = Program("structured-control")
    state = program.state(block, model.state_handle(space))

    condition = program.norm2(state.n) > 0

    def copy(P, value):
        return P.value("body_copy", 1 * value)

    selected = program.branch(
        condition,
        lambda P: copy(P, state.n),
        lambda _P: state.n,
    )
    ranged = program.range(selected, 2, copy)
    program.while_(ranged, lambda P, value: P.norm2(value) > 0, copy)

    graph = program.to_graph()
    branch = next(node for node in graph.nodes if type(node) is Branch)
    loops = [node for node in graph.nodes if type(node) is Loop]

    assert type(branch.when_true) is Region and type(branch.when_false) is Region
    assert branch.when_true.nodes[0].op == "linear_combine"
    assert branch.when_true.result == ValueRef(branch.when_true.nodes[0].node_id)
    assert branch.when_false.nodes == () and branch.when_false.result == ValueRef(state.n.id)
    assert [loop.loop_kind for loop in loops] == ["range", "while"]
    assert loops[0].count == 2 and loops[0].condition is None
    assert loops[1].condition.result.node_id == loops[1].condition.nodes[-1].node_id
    assert "body_block" not in repr(graph.to_data())
    assert "cond_block" not in repr(graph.to_data())


def test_structured_region_validates_capture_identity_clock_point_and_inner_refs():
    clock = Clock("macro")
    point = TimePoint(clock)
    signature = {"value_type": "state", "space": None, "block": None}
    bad_capture = RegionCapture(
        ValueRef(1), clock, TimePoint(clock, 1), signature=signature)
    arm = Region(
        "arm", (bad_capture,), (), ValueRef(1), clocks=(clock,),
        result_signature=signature)
    identity = Region(
        "identity", (RegionCapture(ValueRef(1), clock, point, signature=signature),), (),
        ValueRef(1), clocks=(clock,), result_signature=signature)
    with pytest.raises(ValueError, match="clock and exact point"):
        Branch(
            2, ValueRef(0), arm, identity, clock, point,
            result_signature=signature)

    with pytest.raises(ValueError, match="earlier readable node or explicit capture"):
        Region(
            "bad-inner-ref",
            (),
            (ProgramValue(2, "bad", "state", "copy", (ValueRef(99),), clock, point),),
            ValueRef(2),
            clocks=(clock,),
        )
