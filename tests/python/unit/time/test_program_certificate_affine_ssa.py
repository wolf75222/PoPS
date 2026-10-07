"""Exact affine SSA method proofs and an independent formal nonlinear substitution."""
from fractions import Fraction

import pytest
from pops.identity.scalar import scalar_data
from pops.lib.time import ForwardEuler, SSPRK2, SSPRK3, RK4
from pops.time import (Commit, GraphProgramValue, OperatorCall, ProgramGraph,
                       StateRead, TimePoint, ValueRef, certify_program_graph)
from tests.python.unit.lib.time.test_program_factories import _authoring


def _template():
    state, _, rate, _ = _authoring()
    graph = ForwardEuler(state, rate=rate).to_graph()
    return graph.nodes[0], graph.nodes[1].operator.to_data(), graph.nodes[-1].target.to_data()


def _combine(nodes, refs, factors, point, *, name="unrelated SSA label"):
    coefficients = [[(scalar_data(power), scalar_data(value)) for power, value in factor.items()]
                    for factor in factors]
    node = GraphProgramValue(len(nodes), name, "state", "linear_combine",
                             tuple(ValueRef(index) for index in refs), point.clock, point,
                             attrs={"attrs": {"coeffs": coefficients}})
    nodes.append(node)
    return node.node_id


def _nested_graph(A, b, c, *, name="arbitrary algorithm"):
    initial, operator, target = _template()
    clock = initial.clock
    nodes = [StateRead(0, initial.state.to_data(), clock, TimePoint(clock),
                       metadata=initial.metadata.to_data())]
    base = _combine(nodes, (0, 0), ({0: 2}, {0: -1}), TimePoint(clock))
    rates = []
    for row, coordinate in zip(A, c, strict=True):
        point = TimePoint(clock, coordinate)
        stage = _combine(nodes, (base, *rates[:len(row)]),
                         ({0: 1}, *({1: coefficient} for coefficient in row)), point)
        stage = _combine(nodes, (stage, stage), ({0: 3}, {0: -2}), point)
        call = OperatorCall(len(nodes), operator, (ValueRef(stage),), clock, point,
                            name="call spelling is irrelevant")
        nodes.append(call)
        rates.append(_combine(nodes, (call.node_id, call.node_id), ({0: 2}, {0: -1}), point))
    endpoint = TimePoint(clock, step=1)
    result = _combine(nodes, (base, *rates), ({0: 1}, *({1: value} for value in b)), endpoint)
    result = _combine(nodes, (result, result), ({0: Fraction(3, 2)}, {0: Fraction(-1, 2)}), endpoint)
    nodes.append(Commit(len(nodes), target, ValueRef(result), clock, endpoint))
    return ProgramGraph(name, nodes)


def _formal_nonlinear_series(graph, degree=6):
    """Independent exact SSA algebra under the formal substitution F(y)=y**2, y0=1.

    This test computes no time steps or cell values and invokes no certificate trace.
    Its reference is the exact formal series of y'=y**2: 1/(1-h).
    """
    def multiply(a, b):
        return tuple(sum((a[j] * b[i-j] for j in range(i+1)), Fraction())
                     for i in range(degree+1))

    def literal(data):
        value = data.get("scalar", data)
        if value["kind"] == "integer":
            return Fraction(int(value["value"]))
        if value["kind"] == "rational":
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        raise AssertionError("formal reference requires exact rational coefficients")

    values = {}
    for node in graph.nodes:
        if node.kind == "state_read":
            values[node.node_id] = (Fraction(1),) + (Fraction(),) * degree
        elif node.kind == "operator_call":
            argument = values[node.references()[0].node_id]
            values[node.node_id] = multiply(argument, argument)
        elif node.kind == "program_value" and node.op == "linear_combine":
            payload = node.attrs.to_data()
            payload = payload.get("attrs", payload)
            result = [Fraction()] * (degree+1)
            for ref, encoded in zip(node.references(), payload["coeffs"], strict=True):
                weights = [Fraction()] * (degree+1)
                for power, coefficient in encoded:
                    weights[int(literal(power))] = literal(coefficient)
                term = multiply(tuple(weights), values[ref.node_id])
                result = [a+b for a, b in zip(result, term, strict=True)]
            values[node.node_id] = tuple(result)
        elif node.kind == "commit":
            return values[node.value.node_id]
        else:
            raise AssertionError("formal reference only accepts its declared algebra")
    raise AssertionError("formal graph has no endpoint")


@pytest.mark.parametrize("factory,expected_order", [(ForwardEuler, 1), (SSPRK2, 2),
                                                     (SSPRK3, 3), (RK4, 4)])
def test_actual_factory_ssa_certificate_agrees_with_independent_formal_nonlinear_series(factory, expected_order):
    state, _, rate, _ = _authoring()
    graph = factory(state, rate=rate).to_graph()
    before = graph.graph_hash
    certificate = certify_program_graph(graph)
    actual = _formal_nonlinear_series(graph)
    assert certificate.properties.order == expected_order
    assert actual[:expected_order+1] == (Fraction(1),) * (expected_order+1)
    assert actual[expected_order+1] != 1
    assert graph.graph_hash == before


@pytest.mark.parametrize("name", ["arbitrary algorithm", "RK4"])
def test_nonstandard_exact_tableau_survives_nested_ssa_without_name_based_recipe(name):
    A = ((), (Fraction(2, 5),))
    b, c = (Fraction(-1, 4), Fraction(5, 4)), (0, Fraction(2, 5))
    graph = _nested_graph(A, b, c, name=name)
    certificate = certify_program_graph(graph)
    assert certificate.properties.order == 2
    assert certificate.tableau.A == ((0, 0), (Fraction(2, 5), 0))
    assert certificate.tableau.b == b
    assert certificate.properties.abscissae == c
    actual = _formal_nonlinear_series(graph)
    assert actual[:3] == (1, 1, 1) and actual[3] != 1
