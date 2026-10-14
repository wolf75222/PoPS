"""Construction-time derivatives of finite scalar expression products.

The operands are the original equation DAGs. The result consists of normal PoPS
expressions and is compiled by the same backend. Program field components are
independent SSA inputs here; differentiation through Program control flow, remapping,
or a converged implicit relation is a separate contract and is not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

from pops._ir.expr import Expr, Const, _wrap
from pops._ir.lowering import diff
from pops._ir.visitors import _key


def _inputs(wrt):
    inputs = tuple(wrt)
    if not inputs or any(not isinstance(value, Expr) for value in inputs):
        raise TypeError("wrt must be a nonempty tuple of explicit scalar expression inputs")
    identities = tuple(_key(value) for value in inputs)
    if len(set(identities)) != len(identities):
        raise ValueError("differentiation inputs must have distinct identities")
    return inputs


def _expressions(expressions):
    scalar = isinstance(expressions, Expr)
    result = (expressions,) if scalar else tuple(expressions)
    if not result:
        raise ValueError("an expression product cannot be empty")
    return tuple(_wrap(value) for value in result), scalar


def jvp(expressions, *, wrt=None, direction=None, differentiate_iterations=False, transport_byte_capacity=None):
    """Return J(expressions,wrt) @ direction, retaining all other inputs as parameters.

    A closed GraphFunction returns a closed primal/tangent GraphFunction. Scalar
    expressions require explicit ``wrt`` and ``direction``. Unsupported derivative
    rules fail during construction in both profiles.
    """
    from pops.graph_functions import GraphFunction
    if type(expressions) is GraphFunction:
        if wrt is not None or direction is not None:
            raise TypeError("a GraphFunction JVP accepts primal and tangent through its returned graph")
        from pops.graph_derivatives import jvp as graph_jvp
        return graph_jvp(expressions, differentiate_iterations=differentiate_iterations,
                         transport_byte_capacity=transport_byte_capacity)
    if differentiate_iterations or transport_byte_capacity is not None:
        raise TypeError("graph control/transport options require GraphFunction")
    if wrt is None or direction is None:
        raise TypeError("an expression JVP requires explicit wrt and direction")
    inputs = _inputs(wrt)
    tangent = tuple(_wrap(value) for value in direction)
    if len(inputs) != len(tangent):
        raise ValueError("direction and differentiation inputs have different widths")
    outputs, scalar = _expressions(expressions)
    results = tuple(sum((diff(output, variable) * delta
                         for variable, delta in zip(inputs, tangent, strict=True)), Const(0))
                    for output in outputs)
    return results[0] if scalar else results


@dataclass(frozen=True)
class Pairings:
    """Explicit Riesz maps for a finite vector-Jacobian action.

    ``output_to_covector`` maps the supplied output vector to its coordinate covector.
    ``input_from_covector`` maps the pulled-back covector to an input vector. The
    caller supplies their equations (including any required finite solve). Neither
    map is presumed to be an inverse, a unit conversion or an unweighted identity.
    Builders execute once and must return finite tuples of PoPS expressions.
    """
    output_to_covector: object
    input_from_covector: object

    def __post_init__(self):
        if not callable(self.output_to_covector) or not callable(self.input_from_covector):
            raise TypeError("pairings require two explicit expression builders")


def vjp(expressions, *, wrt=None, covector=None, pairings, differentiate_iterations=False, transport_byte_capacity=None):
    """Finite expression adjoint with explicitly supplied pairings.

    Despite the conventional argument name, ``covector`` is the supplied output
    dual representative; its coordinates are obtained through the declared output
    pairing. This does not guess a physical inner product from an array shape.
    """
    from pops.graph_functions import GraphFunction
    if type(expressions) is GraphFunction:
        if wrt is not None or covector is not None:
            raise TypeError("a GraphFunction VJP accepts primal and covector through its returned graph")
        from pops.graph_derivatives import vjp as graph_vjp
        return graph_vjp(expressions, pairings=pairings, differentiate_iterations=differentiate_iterations,
                         transport_byte_capacity=transport_byte_capacity)
    if differentiate_iterations or transport_byte_capacity is not None:
        raise TypeError("graph control/transport options require GraphFunction")
    if wrt is None or covector is None:
        raise TypeError("an expression VJP requires explicit wrt and covector")
    inputs = _inputs(wrt)
    outputs, _ = _expressions(expressions)
    if type(pairings) is not Pairings:
        raise TypeError("vjp requires explicit Pairings, including for Euclidean spaces")
    representative = tuple(_wrap(value) for value in covector)
    if len(representative) != len(outputs):
        raise ValueError("output dual representative has the wrong width")
    dual = tuple(_wrap(value) for value in pairings.output_to_covector(representative))
    if len(dual) != len(outputs):
        raise ValueError("output pairing returned the wrong width")
    pullback = tuple(sum((diff(output, variable) * weight
                          for output, weight in zip(outputs, dual, strict=True)), Const(0))
                     for variable in inputs)
    result = tuple(_wrap(value) for value in pairings.input_from_covector(pullback))
    if len(result) != len(inputs):
        raise ValueError("input pairing returned the wrong width")
    return result


__all__ = ["jvp", "vjp", "Pairings"]
