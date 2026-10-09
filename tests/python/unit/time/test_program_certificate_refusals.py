"""The affine trace never hides unavailable SSA, clocks or unknown effects."""
from copy import deepcopy
from fractions import Fraction

import pytest
from pops.identity.scalar import scalar_data
from pops.time import (Clock, Commit, GraphProgramValue, OperatorCall, ProgramGraph,
                       StateRead, Synchronize, TimePoint, UnknownOrder, ValueRef,
                       certify_program_graph)
from test_program_certificate_affine_ssa import _nested_graph, _template, _combine


def _base():
    return _nested_graph(((),), (1,), (0,))


def _replace_value(graph, node_id, *, op=None, attrs=None, inputs=None, point=None):
    old = next(node for node in graph.nodes if node.node_id == node_id)
    replacement = GraphProgramValue(old.node_id, old.name, old.value_type, op or old.op,
        old.inputs if inputs is None else inputs, old.clock, old.point if point is None else point,
        attrs=old.attrs.to_data() if attrs is None else attrs)
    return ProgramGraph(graph.name, tuple(replacement if node is old else node for node in graph.nodes),
                        clocks=graph.clocks)


@pytest.mark.parametrize("op", ["copy", "where", "pointwise_expression", "opaque_effect"])
def test_unknown_primitive_cannot_be_mistaken_for_an_affine_alias_even_with_zero_weight(op):
    graph = _base()
    rate = next(node for node in graph.nodes if node.kind == "operator_call")
    stage = next(node for node in graph.nodes if node.node_id == rate.inputs[0].node_id)
    changed = _replace_value(graph, stage.node_id, op=op)
    assert isinstance(certify_program_graph(changed).properties.order, UnknownOrder)
    # A second pure expression multiplies this read by zero; the unknown primitive
    # must still be traced rather than erased as an algebraic value cancellation.
    payload = deepcopy(stage.attrs.to_data())
    payload["attrs"]["coeffs"] = [[(scalar_data(0), scalar_data(0))]]
    masked = GraphProgramValue(stage.node_id + 100, "zero value is not zero effect", "state",
                              "linear_combine", (ValueRef(stage.node_id),), stage.clock,
                              stage.point, attrs=payload)
    nodes = list(changed.nodes)
    position = nodes.index(next(node for node in nodes if node.node_id == stage.node_id)) + 1
    nodes.insert(position, masked)
    replacement_rate = OperatorCall(rate.node_id, rate.operator.to_data(), (ValueRef(masked.node_id),),
                                    rate.clock, rate.point, name=rate.name)
    masked_graph = ProgramGraph(graph.name, tuple(replacement_rate if node is rate else node
                                                for node in nodes), clocks=graph.clocks)
    assert isinstance(certify_program_graph(masked_graph).properties.order, UnknownOrder)


@pytest.mark.parametrize("effect", [{"read_effect": None}, {"verified_order": 4},
                                    {"schedule": {"kind": "off_cadence_cache"}}])
def test_extra_evaluation_effect_or_asserted_order_never_supplies_affine_proof(effect):
    graph = _base()
    alias = next(node for node in graph.nodes if node.kind == "program_value")
    attrs = deepcopy(alias.attrs.to_data())
    attrs["attrs"].update(effect)
    changed = _replace_value(graph, alias.node_id, attrs=attrs)
    assert isinstance(certify_program_graph(changed).properties.order, UnknownOrder)


@pytest.mark.parametrize("power", [Fraction(1, 2), -1, 1, 2])
def test_rate_alias_with_nonconstant_or_invalid_dt_power_is_not_certified(power):
    graph = _base()
    rate = next(node for node in graph.nodes if node.kind == "operator_call")
    alias = next(node for node in graph.nodes if node.kind == "program_value"
                 and any(ref.node_id == rate.node_id for ref in node.references()))
    attrs = deepcopy(alias.attrs.to_data())
    attrs["attrs"]["coeffs"] = [[(scalar_data(power), scalar_data(value))]
                                for value in (2, -1)]
    changed = _replace_value(graph, alias.node_id, attrs=attrs)
    assert isinstance(certify_program_graph(changed).properties.order, UnknownOrder)


@pytest.mark.parametrize("point_kind", ["unmatched-step", "unmatched-offset", "unmatched-extrapolation"])
def test_rate_point_must_match_actual_affine_row_not_a_backend_time_range(point_kind):
    graph = _base()
    rate = next(node for node in graph.nodes if node.kind == "operator_call")
    point = (TimePoint(rate.clock, step=1) if point_kind == "unmatched-step" else
             TimePoint(rate.clock, 1 if point_kind == "unmatched-offset" else 2))
    replacement = OperatorCall(rate.node_id, rate.operator.to_data(), rate.inputs,
                               rate.clock, point, name=rate.name)
    changed = ProgramGraph(graph.name, tuple(replacement if node is rate else node
                                            for node in graph.nodes), clocks=graph.clocks)
    certificate = certify_program_graph(changed)
    assert isinstance(certificate.properties.order, UnknownOrder)
    assert certificate.properties.order.reason == "stage point abscissae differ from reconstructed row sums"


def test_future_ssa_reference_is_rejected_by_the_real_graph_boundary():
    graph = _base()
    alias = next(node for node in graph.nodes if node.kind == "program_value")
    rate = next(node for node in graph.nodes if node.kind == "operator_call")
    with pytest.raises(ValueError, match="earlier readable node"):
        _replace_value(graph, alias.node_id, inputs=(ValueRef(rate.node_id),))


def test_valid_explicit_synchronization_does_not_become_a_single_clock_rk_alias():
    initial, operator, target = _template()
    slow, fast = initial.clock, Clock("genuinely different clock")
    relation = {"kind": "sample_and_hold", "schema_version": 1,
                "provider": {"kind": "latest_accepted_sample", "schema_version": 1}}
    nodes = [StateRead(0, initial.state.to_data(), slow, TimePoint(slow)),
             Synchronize(1, ValueRef(0), slow, fast, relation, TimePoint(fast)),
             OperatorCall(2, operator, (ValueRef(1),), fast, TimePoint(fast)),
             Synchronize(3, ValueRef(2), fast, slow, relation, TimePoint(slow))]
    endpoint = _combine(nodes, (0, 3), ({0: 1}, {1: 1}), TimePoint(slow, step=1))
    nodes.append(Commit(len(nodes), target, ValueRef(endpoint), slow, TimePoint(slow, step=1)))
    graph = ProgramGraph("real synchronization", nodes, clocks=(slow, fast))
    assert isinstance(certify_program_graph(graph).properties.order, UnknownOrder)


def test_equivalent_step_plus_offset_encoding_preserves_exact_stage_coordinate():
    graph = _nested_graph(((), (Fraction(1, 2),)), (0, 1), (0, Fraction(1, 2)))
    rates = [node for node in graph.nodes if node.kind == "operator_call"]
    last = rates[-1]
    replacement = OperatorCall(last.node_id, last.operator.to_data(), last.inputs,
        last.clock, TimePoint(last.clock, Fraction(-1, 2), step=1), name=last.name)
    encoded = ProgramGraph(graph.name, tuple(replacement if node is last else node
                                            for node in graph.nodes), clocks=graph.clocks)
    assert certify_program_graph(encoded).properties == certify_program_graph(graph).properties
    assert certify_program_graph(encoded).properties.order == 2


@pytest.mark.parametrize("kind", ["primitive", "affine-effect", "opaque-coefficient",
                                   "bad-arity", "unknown-representation"])
def test_disconnected_unknown_effect_is_not_hidden_by_endpoint_dependency_trace(kind):
    graph = _base()
    attrs = {"attrs": {"coeffs": [[(scalar_data(0), scalar_data(1))]]}}
    op = "linear_combine"
    if kind == "primitive":
        op, attrs = "opaque_effect", {"writes_state": True}
    elif kind == "affine-effect":
        attrs["attrs"]["writes_state"] = True
    elif kind == "opaque-coefficient":
        attrs["attrs"]["coeffs"] = [[(scalar_data(0), {"kind": "algebraic", "value": "opaque"})]]
    elif kind == "bad-arity":
        attrs["attrs"]["coeffs"] = []
    representation = "scalar" if kind == "unknown-representation" else "state"
    extra = GraphProgramValue(100, "not a dependency but still executed", representation, op,
                             (ValueRef(0),), graph.nodes[0].clock, graph.nodes[0].point, attrs=attrs)
    changed = ProgramGraph(graph.name, (*graph.nodes[:-1], extra, graph.nodes[-1]), clocks=graph.clocks)
    assert isinstance(certify_program_graph(changed).properties.order, UnknownOrder)


def test_repeated_canonical_dt_power_is_refused_even_if_last_entry_preserves_old_weight():
    graph = _base()
    alias = next(node for node in graph.nodes if node.kind == "program_value")
    attrs = deepcopy(alias.attrs.to_data())
    attrs["attrs"]["coeffs"][0] = [(scalar_data(0), scalar_data(17)),
                                      (scalar_data(0), scalar_data(2))]
    changed = _replace_value(graph, alias.node_id, attrs=attrs)
    assert isinstance(certify_program_graph(changed).properties.order, UnknownOrder)


def test_schedule_and_macro_cadence_are_not_unstated_rk_read_effects():
    from pops.time import AcceptedStep, Every, Hold, Schedule
    graph = _base()
    rate = next(node for node in graph.nodes if node.kind == "operator_call")
    operator = deepcopy(rate.operator.to_data())
    operator["lowering"]["attrs"]["schedule"] = Schedule(
        Every(AcceptedStep(rate.clock), 2), off=Hold()).to_data()
    replacement = OperatorCall(rate.node_id, operator, rate.inputs, rate.clock, rate.point)
    scheduled = ProgramGraph(graph.name, tuple(replacement if node is rate else node
                                              for node in graph.nodes), clocks=graph.clocks)
    assert isinstance(certify_program_graph(scheduled).properties.order, UnknownOrder)
    cadence = {**graph.cadence.to_data(), "substeps": 2}
    cadenced = ProgramGraph(graph.name, graph.nodes, clocks=graph.clocks, cadence=cadence)
    assert isinstance(certify_program_graph(cadenced).properties.order, UnknownOrder)
