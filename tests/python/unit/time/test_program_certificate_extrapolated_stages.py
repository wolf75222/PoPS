"""Formal RK order permits causal extrapolated stages; Native capability is separate."""
from fractions import Fraction

import pytest
from pops.lib.time import RungeKutta
from pops.time import Program, StagePoint, TimePoint, certify_program_graph
from pops.time._methods.tableau import RungeKuttaTableau
from tests.python.unit.lib.time.test_program_factories import _authoring
from test_program_certificate_affine_ssa import _formal_nonlinear_series


def _public_compositions(a, b, name):
    state, _, rate, _ = _authoring()
    tableau = RungeKuttaTableau(A=((), (a,)), b=b, c=(0, a), name=name)
    factory = RungeKutta(state, rate=rate, tableau=tableau)
    program = Program(name + " inline public aliases")
    current = program.state(state)
    point0 = StagePoint("input", {"main": TimePoint(program.clock)})
    point1 = StagePoint("approximation", {"main": TimePoint(program.clock, a)})
    first = program.value("scaled first rate", (4 * rate(current.n)) / 4, at=point0)
    stage = program.value("causally computed approximation", current.n + a * program.dt * first, at=point1)
    second = program.value("scaled second rate", (4 * rate(stage)) / 4, at=point1)
    endpoint = program.value("ordinary next state", current.n + program.dt * (b[0]*first + b[1]*second),
                             at=current.next.point)
    program.commit(current.next, endpoint)
    return tableau, factory, program


def _formal_nonautonomous_series(graph, degree=4):
    """Independent exact SSA substitution F(t,y)=(1+t)y, t_n=0, y_n=1."""
    def product(left, right):
        return tuple(sum((left[j] * right[i-j] for j in range(i+1)), Fraction())
                     for i in range(degree+1))

    def literal(value):
        value = value.get("scalar", value)
        return (Fraction(int(value["value"])) if value["kind"] == "integer" else
                Fraction(int(value["numerator"]), int(value["denominator"])))

    values = {}
    for node in graph.nodes:
        if node.kind == "state_read":
            values[node.node_id] = (Fraction(1),) + (Fraction(),) * degree
        elif node.kind == "operator_call":
            point = node.point.time if isinstance(node.point, StagePoint) else node.point
            coordinate = Fraction(point.step) + Fraction(point.offset.to_python())
            factor = (Fraction(1), coordinate) + (Fraction(),) * (degree-1)
            values[node.node_id] = product(factor, values[node.references()[0].node_id])
        elif node.kind == "program_value" and node.op == "linear_combine":
            payload = node.attrs.to_data()
            data = payload.get("attrs", payload)
            result = [Fraction()] * (degree+1)
            for ref, encoded in zip(node.references(), data["coeffs"], strict=True):
                factor = [Fraction()] * (degree+1)
                for power, coefficient in encoded:
                    factor[int(literal(power))] = literal(coefficient)
                result = [a+b for a,b in zip(result, product(tuple(factor),values[ref.node_id]), strict=True)]
            values[node.node_id] = tuple(result)
        elif node.kind == "commit":
            return values[node.value.node_id]
        else:
            raise AssertionError("formal substitution only evaluates its declared algebra")
    raise AssertionError("no formal endpoint")


@pytest.mark.parametrize("a,b", [(2, (Fraction(3,4), Fraction(1,4))),
                                 (-1, (Fraction(3,2), Fraction(-1,2)))])
@pytest.mark.parametrize("name", ["unrelated extrapolated method", "RK4"])
def test_public_extrapolated_and_negative_stages_have_their_exact_formal_order(a,b,name):
    tableau, factory, inline = _public_compositions(a,b,name)
    assert sum(b) == 1 and b[1]*a == Fraction(1,2)
    assert tableau.properties.order == 2
    for program in (factory, inline):
        graph = program.to_graph()
        certificate = certify_program_graph(graph)
        assert certificate.properties.order == 2
        assert certificate.tableau.A == ((0,0),(a,0))
        assert certificate.tableau.b == b and certificate.tableau.c == (0,a)
        # Both stages read already produced SSA values. Extrapolation creates
        # an approximation from current/prior data, never a future StateRead.
        positions = {node.node_id:i for i,node in enumerate(graph.nodes)}
        assert all(positions[ref.node_id] < positions[node.node_id]
                   for node in graph.nodes for ref in node.references())
        assert _formal_nonlinear_series(graph)[:3] == (1,1,1)
        nonautonomous = _formal_nonautonomous_series(graph)
        assert nonautonomous[:3] == (1,1,1)
        # Exact exp(t+t^2/2) has h^3 coefficient 2/3; this probe does not
        # accidentally claim a higher order for the selected scalar equation.
        assert nonautonomous[3] != Fraction(2,3)
