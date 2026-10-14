"""Directional and adjoint identities, with explicit independent inputs and pairings."""
import numpy as np
import pytest

from pops._ir.expr import Var
from pops.algebra import Pairings, jvp, vjp, expm1, sin


def test_weighted_adjoint_identity_for_a_nonlinear_equation():
    x, y = Var("x", "cons"), Var("y", "cons")
    outputs = (x * y + sin(x), expm1(y) + x ** 2)
    direction, dual = (.3, -.2), (.7, .4)
    pairings = Pairings(lambda c: (5*c[0], 7*c[1]), lambda c: (c[0]/2, c[1]/3))
    tangent = jvp(outputs, wrt=(x, y), direction=direction)
    adjoint = vjp(outputs, wrt=(x, y), covector=dual, pairings=pairings)
    env = {"x": .8, "y": .25}
    lhs = sum(w * expression.eval(env) * value
              for w, expression, value in zip((5, 7), tangent, dual))
    rhs = sum(w * value * expression.eval(env)
              for w, value, expression in zip((2, 3), direction, adjoint))
    assert lhs == pytest.approx(rhs, rel=2e-14)


def test_program_differentiation_uses_ssa_identity_not_component_spelling():
    from test_program_expressions import fixture, emit
    model, _, program, q = fixture()
    later = program.value("later", (q.n[0] + 1, q.n[1] + 2))
    # The earlier producer is explicitly independent here; this is not AD through time.
    expressions = (q.n[0] ** 2 + later[0], q.n[0] * q.n[1])
    result = jvp(expressions, wrt=(q.n[0], q.n[1]), direction=(1, 2))
    value = program.value("tangent", result, at=q.next.point)
    program.commit(q.next, value)
    assert program.validate()
    assert "pops::for_each_cell" in emit(model, program)
    env = {(q.n.id, 0): 3., (q.n.id, 1): 5.}
    assert tuple(expression.eval(env) for expression in result) == (6., 11.)


def test_derivative_inputs_cannot_be_ambiguous_or_implicitly_euclidean():
    x = Var("x", "cons")
    with pytest.raises(ValueError, match="distinct"):
        jvp(x*x, wrt=(x, x), direction=(1, 2))
    with pytest.raises(TypeError, match="Pairings"):
        vjp((x*x,), wrt=(x,), covector=(1,), pairings=None)
    other_namespace = Var("x", "parameter")
    derivative = jvp(x + other_namespace, wrt=(x,), direction=(1,))
    assert derivative.eval({"x": 4}) == 1


def test_numpy_cannot_eagerly_coerce_symbolic_equations():
    x = Var("x", "cons")
    with pytest.raises(TypeError, match="NumPy"):
        np.asarray(x)
