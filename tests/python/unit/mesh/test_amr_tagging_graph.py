from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json

import pytest

from pops.mesh._amr import (
    Above,
    AllOf,
    AnyOf,
    Below,
    ConflictPolicy,
    DiscreteIndicatorContext,
    EqualityPolicy,
    GradientAbove,
    Hysteresis,
    MagnitudeAbove,
    Not,
    TagDecision,
    TagExpr,
    TagNodeRegistration,
    TagNodeRegistry,
    TaggingGraph,
    TaggingState,
)
from pops.model import Handle, OwnerKind, OwnerPath, ParamHandle


@dataclass(frozen=True, slots=True)
class _PureExtension(TagExpr):
    enabled: bool
    node_type = "pure_extension"

    def canonical_identity(self):
        return {"schema_version": 1, "node_type": self.node_type, "enabled": self.enabled}

    def operands(self):
        return ()


@dataclass(frozen=True, slots=True)
class _CallbackExtension(TagExpr):
    callback: object
    node_type = "callback_extension"

    def canonical_identity(self):
        return {"schema_version": 1, "node_type": self.node_type}

    def operands(self):
        return ()


def _indicator(name: str = "u", *, owner: OwnerPath | None = None, kind: str = "state"):
    return Handle(name, kind=kind, owner=owner or OwnerPath.model("transport"))


def _threshold(name: str = "tag", *, owner: OwnerPath | None = None):
    return ParamHandle(
        name, owner=owner or OwnerPath.case("main"), param_kind="runtime")


def _context(stencil: str = "centered_2"):
    owner = OwnerPath.case("main")
    return DiscreteIndicatorContext(
        layout=Handle("adaptive", kind="layout", owner=owner),
        discretization=Handle("fv", kind="discretization", owner=owner),
        stencil=Handle(stencil, kind="stencil", owner=owner),
    )


def _graph():
    indicator = _indicator()
    refine = Above(indicator, _threshold("refine"))
    coarsen = Below(indicator, _threshold("coarsen"))
    return TaggingGraph(
        refine=refine,
        coarsen=coarsen,
        hysteresis=Hysteresis(min_cycles=2, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.ERROR,
    )


def test_predicate_polarities_and_threshold_equality_are_explicit():
    indicator = _indicator()
    threshold = _threshold()
    above = Above(indicator, threshold)
    below = Below(indicator, threshold)
    magnitude = MagnitudeAbove(indicator, threshold)

    assert above.polarity == "high"
    assert above.comparison == "strict_greater_than"
    assert below.polarity == "low"
    assert below.comparison == "strict_less_than"
    assert magnitude.transform == "magnitude"
    assert above.equality_matches is False
    assert below.equality_matches is False
    assert above != below
    assert above == Above(indicator, threshold)
    assert above.canonical_identity()["equality_matches"] is False


def test_thresholds_and_indicators_require_canonical_typed_handles():
    indicator = _indicator()
    threshold = _threshold()
    assert Above(indicator, threshold).indicator == indicator

    with pytest.raises(TypeError, match="ParamHandle"):
        Above(indicator, 0.1)
    with pytest.raises(TypeError, match="ParamHandle"):
        Above(indicator, "threshold")
    with pytest.raises(TypeError, match="owner-qualified Handle"):
        Above("u", threshold)
    with pytest.raises(TypeError, match="owner-qualified Handle"):
        Above(lambda cell: cell, threshold)
    with pytest.raises(TypeError, match="Handle.kind"):
        Above(Handle("block", kind="block", owner=OwnerPath.case("main")), threshold)

    authoring_indicator = Handle(
        "u", kind="state", owner=OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "transport"))
    with pytest.raises(TypeError, match="post-resolution"):
        Above(authoring_indicator, threshold)
    authoring_threshold = ParamHandle(
        "tag", owner=OwnerPath.fresh(OwnerKind.CASE, "main"), param_kind="runtime")
    with pytest.raises(TypeError, match="post-resolution"):
        Above(indicator, authoring_threshold)


def test_state_and_field_indicators_are_both_supported_without_string_dispatch():
    state = Above(_indicator(kind="state"), _threshold("state_tag"))
    field = Above(_indicator("phi", kind="field"), _threshold("field_tag"))
    assert state.indicator.kind == "state"
    assert field.indicator.kind == "field"
    assert state.canonical_identity()["indicator"]["kind"] == "state"
    assert field.canonical_identity()["indicator"]["kind"] == "field"


def test_gradient_requires_complete_discrete_indicator_context():
    indicator, threshold = _indicator(), _threshold()
    context = _context()
    gradient = GradientAbove(indicator, threshold, context)
    report = gradient.inspect()
    assert report["transform"] == "discrete_gradient_magnitude"
    assert report["discrete_context"] == context.canonical_identity()

    with pytest.raises(TypeError, match="DiscreteIndicatorContext"):
        GradientAbove(indicator, threshold, None)
    with pytest.raises(TypeError, match="Handle.kind"):
        DiscreteIndicatorContext(
            layout=Handle("layout", kind="layout", owner=OwnerPath.case("main")),
            discretization=Handle("fv", kind="operator", owner=OwnerPath.case("main")),
            stencil=Handle("s", kind="stencil", owner=OwnerPath.case("main")),
        )


def test_boolean_graph_preserves_every_child_without_flattening_or_single_slot_union():
    indicator = _indicator()
    above = Above(indicator, _threshold("high"))
    below = Below(indicator, _threshold("low"))
    magnitude = MagnitudeAbove(indicator, _threshold("magnitude"))
    intersection = AllOf(below, Not(magnitude))
    union = AnyOf(above, intersection, magnitude)

    identity = union.canonical_identity()
    assert [row["node_type"] for row in identity["children"]] == [
        "above", "all_of", "magnitude_above"]
    assert identity["children"][1]["children"][1]["node_type"] == "not"
    assert len(union.children) == 3
    assert union.children[1] is intersection

    with pytest.raises(ValueError, match="at least 2"):
        AnyOf(above)
    with pytest.raises(ValueError, match="at least 2"):
        AllOf()
    with pytest.raises(TypeError, match="strings/callbacks"):
        AnyOf(above, "below")
    with pytest.raises(TypeError, match="strings/callbacks"):
        Not(lambda value: value)

    callback = _CallbackExtension(lambda value: value)
    extensible_union = AnyOf(above, callback)
    unresolved = TaggingGraph(
        refine=extensible_union, coarsen=None,
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.ERROR)
    with pytest.raises(ValueError, match="unregistered"):
        unresolved.resolve()


def test_extension_registry_is_open_but_resolve_rejects_callbacks_and_unknown_nodes():
    graph = _graph()
    resolved = graph.resolve()
    assert {row.node_type for row in resolved.registrations} == {"above", "below"}

    owner = OwnerPath.shared("test.tagging.lowering")
    pure = _PureExtension(True)
    extended_graph = TaggingGraph(
        refine=AnyOf(graph.refine, pure), coarsen=graph.coarsen,
        hysteresis=graph.hysteresis, conflict_policy=graph.conflict_policy)
    pure_registration = TagNodeRegistration(
        _PureExtension, Handle("pure", kind="tag_lowering", owner=owner))
    registry = TagNodeRegistry(
        *TagNodeRegistry.builtins().registrations, pure_registration)
    extension_resolved = extended_graph.resolve(registry)
    assert any(row.node_class is _PureExtension for row in extension_resolved.registrations)
    immutable_rows = [
        (pure_registration, "node_class", _CallbackExtension),
        (pure_registration, "lowering", Handle("other", kind="tag_lowering", owner=owner)),
        (pure_registration, "node_type", "other"),
        (registry, "registrations", ()),
        (extension_resolved, "graph", graph),
        (extension_resolved, "registrations", ()),
    ]
    for value, field, replacement in immutable_rows:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, field, replacement)

    callback_graph = TaggingGraph(
        refine=AnyOf(graph.refine, _CallbackExtension(lambda value: value)),
        coarsen=None, hysteresis=graph.hysteresis,
        conflict_policy=graph.conflict_policy)
    callback_registration = TagNodeRegistration(
        _CallbackExtension, Handle("callback", kind="tag_lowering", owner=owner))
    callback_registry = TagNodeRegistry(
        *TagNodeRegistry.builtins().registrations, callback_registration)
    with pytest.raises(TypeError, match="callbacks are forbidden"):
        callback_graph.resolve(callback_registry)


def test_hysteresis_and_conflict_semantics_are_mandatory_and_complete():
    graph = _graph()
    identity = graph.canonical_identity()
    assert identity["hysteresis"] == {
        "schema_version": 1,
        "hysteresis_type": "min_cycles",
        "min_cycles": 2,
        "equality": "hold",
    }
    assert identity["conflict_policy"] == "error"
    assert graph.qualified_id.endswith(graph.canonical_id)
    assert json.loads(json.dumps(graph.inspect())) == graph.inspect()

    with pytest.raises(TypeError, match="EqualityPolicy"):
        Hysteresis(min_cycles=1, equality="hold")
    with pytest.raises(ValueError, match=">= 0"):
        Hysteresis(min_cycles=-1, equality=EqualityPolicy.HOLD)
    with pytest.raises(TypeError, match="ConflictPolicy"):
        TaggingGraph(
            refine=graph.refine, coarsen=graph.coarsen, hysteresis=graph.hysteresis,
            conflict_policy="error")


def test_owner_qualification_prevents_homonymous_indicator_and_threshold_ambiguity():
    first_indicator = _indicator(owner=OwnerPath.model("transport_a"))
    second_indicator = _indicator(owner=OwnerPath.model("transport_b"))
    first_threshold = _threshold(owner=OwnerPath.case("case_a"))
    second_threshold = _threshold(owner=OwnerPath.case("case_b"))

    first = Above(first_indicator, first_threshold)
    second = Above(second_indicator, second_threshold)
    assert first != second
    assert first.canonical_identity() != second.canonical_identity()

    common = Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD)
    first_graph = TaggingGraph(
        refine=first, coarsen=None, hysteresis=common,
        conflict_policy=ConflictPolicy.ERROR)
    second_graph = TaggingGraph(
        refine=second, coarsen=None, hysteresis=common,
        conflict_policy=ConflictPolicy.ERROR)
    assert first_graph.canonical_id != second_graph.canonical_id


def test_every_graph_field_is_immutable_and_every_semantic_change_changes_identity():
    graph = _graph()
    gradient = GradientAbove(_indicator(), _threshold("gradient"), _context())
    union = AnyOf(graph.refine, gradient)
    cases = [
        (graph.refine, "indicator", _indicator("v")),
        (graph.refine, "threshold", _threshold("other")),
        (gradient, "context", _context("upwind_2")),
        (gradient.context, "layout", Handle(
            "uniform", kind="layout", owner=OwnerPath.case("main"))),
        (gradient.context, "discretization", Handle(
            "dg", kind="discretization", owner=OwnerPath.case("main"))),
        (gradient.context, "stencil", Handle(
            "upwind", kind="stencil", owner=OwnerPath.case("main"))),
        (union, "children", (graph.refine, graph.coarsen)),
        (Not(graph.refine), "child", graph.coarsen),
        (graph.hysteresis, "min_cycles", 7),
        (graph.hysteresis, "equality", EqualityPolicy.REFINE),
        (graph, "refine", gradient),
        (graph, "coarsen", None),
        (graph, "hysteresis", Hysteresis(0, EqualityPolicy.HOLD)),
        (graph, "conflict_policy", ConflictPolicy.HOLD),
    ]
    for value, field, replacement in cases:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, field, replacement)

    changed_threshold = TaggingGraph(
        refine=Above(_indicator(), _threshold("other")), coarsen=graph.coarsen,
        hysteresis=graph.hysteresis, conflict_policy=graph.conflict_policy)
    changed_stencil = TaggingGraph(
        refine=GradientAbove(_indicator(), _threshold("refine"), _context("upwind_2")),
        coarsen=graph.coarsen, hysteresis=graph.hysteresis,
        conflict_policy=graph.conflict_policy)
    assert len({graph.canonical_id, changed_threshold.canonical_id,
                changed_stencil.canonical_id}) == 3


def test_persistent_tagging_state_round_trips_and_min_cycle_boundary_is_inclusive():
    rules = _graph()
    graph = rules.resolve()
    scope = Handle("patch_0", kind="tagging_state", owner=OwnerPath.case("main"))
    before = TaggingState(
        scope=scope, graph=graph, cycle=4, last_decision_cycle=3,
        last_decision=TagDecision.REFINE)
    boundary = TaggingState(
        scope=scope, graph=graph, cycle=5, last_decision_cycle=3,
        last_decision=TagDecision.REFINE)
    assert before.cycles_since_decision == 1
    assert before.transition_allowed() is False
    assert boundary.cycles_since_decision == rules.hysteresis.min_cycles
    assert boundary.transition_allowed() is True

    data = boundary.canonical_identity()
    rebuilt = TaggingState.from_canonical_identity(data, graph=graph)
    assert rebuilt == boundary
    assert rebuilt.canonical_identity() == data
    assert rebuilt.inspect()["last_decision"] == "refine"
    for field, replacement in (
        ("scope", Handle("patch_1", kind="tagging_state", owner=OwnerPath.case("main"))),
        ("graph", TaggingGraph(
            refine=rules.refine, coarsen=None, hysteresis=rules.hysteresis,
            conflict_policy=rules.conflict_policy).resolve()),
        ("cycle", 6), ("last_decision_cycle", 4),
        ("last_decision", TagDecision.COARSEN),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(boundary, field, replacement)

    with pytest.raises(ValueError, match="after cycle"):
        TaggingState(
            scope=scope, graph=graph, cycle=2, last_decision_cycle=3,
            last_decision=TagDecision.HOLD)
    with pytest.raises(ValueError, match="does not authenticate"):
        TaggingState.from_canonical_identity(data, graph=TaggingGraph(
            refine=rules.refine, coarsen=None, hysteresis=rules.hysteresis,
            conflict_policy=rules.conflict_policy).resolve())
