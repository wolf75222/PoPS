"""Exact temporal points and immutable ADC-662 ProgramGraph records."""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

import pytest

from pops.time import Clock, StagePoint, TimePoint
from pops.time.graph import (
    Branch,
    Commit,
    OperatorCall,
    ProgramGraph,
    ProgramValue,
    Region,
    Solve,
    StateRead,
    Synchronize,
    Unknown,
    ValueRef,
)


def test_time_point_preserves_exact_authoring_domains():
    clock = Clock("macro")
    rational = TimePoint(clock, Fraction(1, 3), step=2)
    decimal = TimePoint(clock, Decimal("0.125"))
    binary = TimePoint(clock, 0.125)

    assert rational.offset.to_python() == Fraction(1, 3)
    assert rational.to_data()["offset"] == {
        "kind": "rational", "numerator": "1", "denominator": "3"}
    assert decimal.to_data()["offset"] == {"kind": "decimal", "value": "0.125"}
    assert binary.to_data()["offset"]["kind"] == "binary64"
    assert rational.step == 2
    assert len({rational, decimal, binary}) == 3


def test_stage_point_requires_named_partitions_and_rejects_ambiguous_time():
    clock = Clock("macro")
    explicit = TimePoint(clock, Fraction(1, 2))
    implicit = TimePoint(clock, Fraction(2, 3))
    stage = StagePoint("corrector", {"implicit": implicit, "explicit": explicit})

    assert isinstance(stage.partitions, MappingProxyType)
    assert tuple(stage.partitions) == ("explicit", "implicit")
    assert stage.time_for("explicit") is explicit
    with pytest.raises(ValueError, match="ambiguous.*time_for"):
        _ = stage.time
    with pytest.raises(KeyError, match="declared partitions"):
        stage.time_for("transport")
    with pytest.raises(TypeError, match="mapping"):
        StagePoint("positional", [explicit])

    shared = StagePoint("shared", {"explicit": explicit, "implicit": explicit})
    assert shared.time is explicit


def _point(clock):
    return TimePoint(clock, Fraction(1, 2))


def test_program_graph_has_exact_nodes_canonical_hash_and_no_builder_references():
    clock = Clock("macro")
    metadata = {"coefficients": [Fraction(1, 3), Decimal("0.25")]}
    state = StateRead(0, {"qualified_id": "case/fluid/U"}, clock, _point(clock), name="U.n")
    unknown = Unknown(1, "delta", {"space": "U"}, clock, _point(clock))
    call = OperatorCall(
        2, {"qualified_id": "model/rate"}, (ValueRef(0),), clock, _point(clock), name="rate")
    value = ProgramValue(
        3, "rhs", "state", "linear_combine", (ValueRef(0), ValueRef(2)),
        clock, _point(clock), attrs=metadata)
    solve = Solve(4, ValueRef(1), ValueRef(2), ValueRef(3), clock, _point(clock))
    commit = Commit(
        5, {"qualified_id": "case/fluid/U.next", "readable": False},
        ValueRef(4), clock, _point(clock))

    graph = ProgramGraph("step", (state, unknown, call, value, solve, commit), clocks=(clock,))
    before = graph.to_data()
    metadata["coefficients"].append(99)

    assert graph.to_data() == before
    assert graph.to_data()["kind"] == "pops.program-graph"
    assert graph.to_data()["nodes"][0]["kind"] == "state_read"
    assert graph.to_data()["nodes"][-1]["kind"] == "commit"
    assert len(graph.graph_hash) == 64
    assert graph.graph_hash == ProgramGraph(
        "step", (state, unknown, call, value, solve, commit), clocks=(clock,)).graph_hash
    assert not hasattr(graph, "_values")
    assert not hasattr(state, "prog")
    with pytest.raises(TypeError, match="write-only.*no readable"):
        graph.ref(commit)


def test_cross_clock_reads_require_one_explicit_synchronize_node():
    slow = Clock("slow")
    fast = Clock("fast")
    source = StateRead(0, {"qualified_id": "case/slow/U"}, slow, TimePoint(slow))
    illegal = ProgramValue(
        1, "illegal", "state", "copy", (ValueRef(0),), fast, TimePoint(fast))

    with pytest.raises(ValueError, match="cross-clock read slow -> fast.*Synchronize"):
        ProgramGraph("illegal", (source, illegal), clocks=(slow, fast))

    sync = Synchronize(
        1, ValueRef(0), slow, fast, {"kind": "sample_and_hold"}, TimePoint(fast))
    legal = ProgramValue(
        2, "legal", "state", "copy", (ValueRef(1),), fast, TimePoint(fast))
    graph = ProgramGraph("legal", (source, sync, legal), clocks=(slow, fast))
    assert graph.ref(sync) == ValueRef(1)
    assert graph.to_data()["nodes"][1]["relation"]["kind"] == "sample_and_hold"


def test_branch_owns_immutable_graph_arms_and_is_lazily_represented():
    clock = Clock("macro")
    point = TimePoint(clock)
    condition = ProgramValue(
        0, "condition", "bool", "compare", (), clock, point)
    selected = ProgramValue(1, "selected", "scalar", "value", (), clock, point)
    signature = {"value_type": "scalar", "space": None, "block": None}
    true_graph = Region(
        "true", (), (selected,), ValueRef(1), clocks=(clock,),
        result_signature=signature)
    false_graph = Region(
        "false", (), (selected,), ValueRef(1), clocks=(clock,),
        result_signature=signature)
    branch = Branch(
        2, ValueRef(0), true_graph, false_graph, clock, point, name="choose",
        result_signature=signature)
    graph = ProgramGraph("branch", (condition, branch), clocks=(clock,))

    data = graph.to_data()["nodes"][1]
    assert data["kind"] == "branch"
    assert data["when_true"]["name"] == "true"
    assert data["when_false"]["name"] == "false"


def test_graph_rejects_forward_commit_and_opaque_builder_references():
    clock = Clock("macro")
    point = TimePoint(clock)
    with pytest.raises(ValueError, match="earlier readable node"):
        ProgramGraph(
            "forward",
            (ProgramValue(0, "bad", "state", "copy", (ValueRef(1),), clock, point),),
            clocks=(clock,),
        )

    class MutableBuilder:
        pass

    with pytest.raises(TypeError, match="mutable/opaque MutableBuilder"):
        StateRead(0, MutableBuilder(), clock, point)
