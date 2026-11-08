"""The additive structure of the right-hand side of a balance (spec 0.3 R7, gap C1).

R7 reads the right-hand side of a balance at its top-level additive structure. Each summand is a
typed term (a declared flux under ``div``, a declared source, a joint rate projection, a
nonconservative product) with an exact coefficient, or algebra that holds no term (``k*u**2``,
``u/(1 + u)``, ``exp(-u)``, a tuple with one value per component), which the model declares as an
anonymous source of the balance.

This module finds that structure exactly. It never rebuilds, expands, reassociates or evaluates an
operand: an operand without terms is kept as the object written, and coefficients multiply as exact
numbers. A term under any other node (a product by a quantity, a quotient by a term, a function, a
``where``), or a term used as one component of a vector, is refused with ``TermNotAdditive``: its
coverage and its delivery could not be derived exactly. The walks are iterative and have no cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pops.identity.scalar import exact_numeric_scalar, multiply_exact_scalars

from .expr import Add, Const, Div, Expr, Mul, Neg, RateTerm, Sub, _wrap, rate_term_of
from .refusals import TermNotAdditive


@dataclass(frozen=True, slots=True, eq=False)
class InlineAlgebra:
    """One operand of a balance that holds no term, as written, and the line that wrote it.

    ``value`` is an expression or a number, or a tuple or list of them with one value per
    component of the balanced state. ``site`` is the ``pops.provenance.SourceSpan`` where the
    operand joined the balance; it documents the operand and enters no identity.
    """

    value: Any
    site: Any
    __pops_ir_immutable__ = True

    @property
    def components(self) -> tuple[Any, ...]:
        return tuple(self.value) if isinstance(self.value, (tuple, list)) else (self.value,)

    def __repr__(self) -> str:
        return "inline(%r)" % (self.value,)


def exact_number(value: Any) -> Any:
    """The exact Python number that ``value`` spells, or ``None``.

    A number, or a ``Const`` without a declaration handle whose literal carries no unit or target.
    A declared constant keeps its identity, so it is not read as a bare number.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Const):
        if value.handle is not None:
            return None
        value = value.literal
    try:
        return exact_numeric_scalar(value, where="rate term coefficient")
    except (TypeError, ValueError):
        return None


def exact_reciprocal(value: Any) -> Any:
    """The exact reciprocal of a nonzero exact number, as a ``Fraction``, or ``None``."""
    number = exact_number(value)
    if number is None or number == 0:
        return None
    return Fraction(1) / Fraction(number)


def _is_rate_component(node: Any) -> bool:
    """One component of a rate output of a joint application; other outputs are data."""
    from .application import ApplicationProjection
    if not isinstance(node, ApplicationProjection):
        return False
    from pops.model.spaces import RateSpace
    output = node.application.operator.signature.output
    if isinstance(output, RateSpace):
        return node.output == "value"
    items = getattr(output, "items", None)
    return callable(items) and isinstance(dict(items()).get(node.output), RateSpace)


def is_term(node: Any) -> bool:
    """A typed term of a balance, or one component of a rate output of a joint application."""
    return isinstance(node, RateTerm) or _is_rate_component(node)


def _site() -> Any:
    from pops.provenance import source_span
    return source_span()


def term_holders(roots: Any) -> dict[int, bool]:
    """Map ``id(node)`` to whether the node is a term or holds one, for every node under ``roots``.

    Iterative post-order over the expression DAG, each node visited once.
    """
    from .visitors import _children
    holds: dict[int, bool] = {}
    expanded: set[int] = set()
    stack = [root for root in roots if isinstance(root, Expr)]
    while stack:
        node = stack[-1]
        key = id(node)
        if key in holds:
            stack.pop()
            continue
        if is_term(node):
            holds[key] = True
            stack.pop()
            continue
        children = _children(node)
        if key not in expanded:
            expanded.add(key)
            stack.extend(child for child in children if id(child) not in holds)
            continue
        stack.pop()
        holds[key] = any(holds[id(child)] for child in children)
    return holds


def first_term(node: Any) -> Any:
    """The first term under ``node``, depth first in written order."""
    from .visitors import _children
    seen: set[int] = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if is_term(item):
            return item
        stack.extend(reversed(_children(item)))
    return None


_WRITE_INSTEAD = (
    "a term of a balance is a summand of its right-hand side, scaled at most by an exact number; "
    "write the algebra of quantities inline, declare a source whose value is this expression "
    "(model.source), or, for a function of the state times a derivative, a nonconservative "
    "product (model.nonconservative_product)")


def refuse(term: Any, construct: str, message: str | None = None) -> TermNotAdditive:
    """The refusal of ``term`` under ``construct``, at the line of the caller."""
    return TermNotAdditive(message or _WRITE_INSTEAD, term=term, construct=construct,
                           site=_site())


def refuse_operand(value: Any) -> TermNotAdditive | None:
    """The refusal of a term met as the operand of another node (``_wrap``), or ``None``."""
    term = rate_term_of(value)
    if term is None:
        return None
    return refuse(term, "an operand of a node other than +, - and an exact scaling")


def _scale(coefficient: Any, number: Any) -> Any:
    return multiply_exact_scalars(coefficient, number, where="rate term coefficient")


def _describe(node: Any) -> str:
    return "a %s node %r" % (type(node).__name__, node)


def rate_summands(value: Any) -> list[tuple[str, Any, Any]]:
    """The ``(kind, payload, coefficient)`` summands of ``value``, in written order (R7).

    ``value`` is what joins the right-hand side of a balance and is not itself a term: the number
    0, which adds nothing (``sum`` starts with it), a number, an expression, or a tuple, list or
    ``VectorExpr`` with one value per component. An operand without terms is one ``"inline"``
    summand with coefficient 1; an expression that holds terms is read through ``+``, ``-``, unary
    ``-`` and a product or a quotient by an exact number.
    """
    if isinstance(value, (tuple, list)):
        if len(value) == 0:  # a VectorExpr has no truth value
            raise ValueError("an inline operand of a balance holds one value per component; "
                             "got an empty sequence")
        holds = term_holders(value)
        for item in value:
            if is_term(item) or holds.get(id(item)):
                raise refuse(item if is_term(item) else first_term(item),
                             "a component of %r" % (value,),
                             "a term of a balance is a summand of the whole state, not one "
                             "component of a vector")
        return [("inline", InlineAlgebra(value, _site()), 1)]
    if isinstance(value, bool) or value is None:
        raise TypeError("the right-hand side of a balance holds terms and algebra; got %r"
                        % (value,))
    if not isinstance(value, Expr):
        number = exact_number(value)
        if number is not None:
            return [] if number == 0 else [("inline", InlineAlgebra(value, _site()), 1)]
        try:
            value = _wrap(value)
        except TypeError:
            raise TypeError(
                "the right-hand side of a balance holds terms (div of a declared flux, a declared "
                "source, ...) and algebra; got %r" % (value,)) from None
    holds = term_holders((value,))
    summands: list[tuple[str, Any, Any]] = []
    site = None
    stack: list[tuple[Any, Any]] = [(value, 1)]
    while stack:
        node, coefficient = stack.pop()
        if not holds.get(id(node)):
            number = exact_number(node)
            if number is not None and number == 0:
                continue
            if site is None:
                site = _site()
            summands.append(("inline", InlineAlgebra(node, site), coefficient))
            continue
        if isinstance(node, RateTerm):
            summands.extend((kind, payload, _scale(scale, coefficient))
                            for kind, payload, scale in node._rate_terms())
            continue
        if isinstance(node, Add):
            stack.extend(((node.b, coefficient), (node.a, coefficient)))
            continue
        if isinstance(node, Sub):
            stack.extend(((node.b, -coefficient), (node.a, coefficient)))
            continue
        if isinstance(node, Neg):
            stack.append((node.a, -coefficient))
            continue
        if isinstance(node, Mul) and holds.get(id(node.a)) != holds.get(id(node.b)):
            inner, other = (node.a, node.b) if holds.get(id(node.a)) else (node.b, node.a)
            number = exact_number(other)
            if number is not None:
                stack.append((inner, _scale(coefficient, number)))
                continue
            raise refuse(first_term(inner), "a product by %r" % (other,))
        if isinstance(node, Div) and holds.get(id(node.a)) and not holds.get(id(node.b)):
            reciprocal = exact_reciprocal(node.b)
            if reciprocal is not None:
                stack.append((node.a, _scale(coefficient, reciprocal)))
                continue
            raise refuse(first_term(node.a), "a quotient by %r" % (node.b,))
        raise refuse(first_term(node), _describe(node))
    return summands


__all__ = ["InlineAlgebra", "exact_number", "exact_reciprocal", "first_term", "is_term",
           "rate_summands", "rate_term_of", "refuse", "refuse_operand", "term_holders"]
