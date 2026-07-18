"""pops._ir.elliptic -- the board-node elliptic field-operator algebra (Spec 5 sec.9.2).

The left-hand side of a field equation is a sum of elliptic operator terms. These nodes make
the variable-coefficient forms the Poisson family uses AUTHORABLE and inspectable:

  -laplacian(phi) + k*phi == rhs            (screened Poisson)
  -div(eps*grad(phi)) + k*phi == rhs        (anisotropic / variable-coefficient, sec.9.2)

They are INERT: IR construction only; the C++ elliptic solver executes. The base
:class:`pops._ir.expr._EllipticTerm` (which :class:`Laplacian` inherits) lives in
:mod:`pops._ir.expr`; the concrete terms + helpers live here to keep ``expr.py`` small.
Lowering the variable-coefficient operators in the elliptic codegen is the coordinated
follow-up.
"""
from __future__ import annotations

from typing import Any

from .expr import _BoardNode, _EllipticTerm
from pops.identity.scalar import exact_numeric_scalar, exact_scale_prefix, scalar_literal


def _as_elliptic(x: Any) -> Any:
    """Coerce ``x`` to an :class:`pops._ir.expr._EllipticTerm`, else a clear error."""
    if isinstance(x, _EllipticTerm):
        return x
    raise TypeError(
        "an elliptic field-equation left-hand side must be a sum of elliptic operator terms "
        "(laplacian(phi) / div(coeff*grad(phi)) / a reaction coeff*phi); got %r" % (x,))


def principal_kinds(node: Any) -> Any:
    """The set of elliptic principal-operator kinds in a field-equation LHS (empty if none)."""
    return node._principal_kinds() if isinstance(node, _EllipticTerm) else set()


def elliptic_terms(node: Any) -> tuple[Any, ...]:
    """Return the ordered primitive terms of one elliptic left-hand side.

    Consumers lower the terms through small capabilities instead of branching on every
    possible accumulated-expression shape.  A non-elliptic node has no terms.
    """
    return tuple(node._elliptic_terms()) if isinstance(node, _EllipticTerm) else ()


def constant_reaction_scalar(value: Any) -> Any:
    """Return an exact compile-time scalar coefficient, or ``NotImplemented``.

    Literal coefficients and explicit ``ConstParam``/compile-derived reads share this small
    protocol: parameter authoring lowers the latter to :class:`Const` while retaining the exact
    declaration identity on that node.  Runtime parameter reads and spatial coefficient
    descriptors deliberately do not pass this projection.
    """
    from .expr import Const

    try:
        literal = value.literal if isinstance(value, Const) else scalar_literal(value)
    except (TypeError, ValueError):
        return NotImplemented
    if literal.unit is not None or literal.target is not None:
        return NotImplemented
    try:
        scalar = literal.to_python()
    except TypeError:
        return NotImplemented
    if isinstance(scalar, bool):
        return NotImplemented
    return scalar


class Reaction(_EllipticTerm):
    """A zeroth-order reaction term ``coeff * phi`` (built by ``coeff * unknown("phi")``)."""

    def __init__(self, field: Any, coeff: Any, scale: Any = 1.0) -> None:
        self.field = field
        self.coeff = coeff
        self.scale = exact_numeric_scalar(scale, where="Reaction scale")

    def _kind(self) -> Any:
        return "reaction"

    def __neg__(self) -> Any:
        return Reaction(self.field, self.coeff, -self.scale)

    def __repr__(self) -> str:
        lead = exact_scale_prefix(self.scale)
        return "Reaction(%s%r*%r)" % (lead, self.coeff, self.field)


class CoeffGradient(_BoardNode):
    """``coeff * grad(phi)`` -- consumed by ``div(...)`` to build a :class:`DivCoeffGrad`."""

    def __init__(self, field: Any, coeff: Any, scale: Any = 1.0) -> None:
        self.field = field
        self.coeff = coeff
        self.scale = exact_numeric_scalar(scale, where="CoeffGradient scale")

    def __neg__(self) -> Any:
        return CoeffGradient(self.field, self.coeff, -self.scale)

    def __repr__(self) -> str:
        return "CoeffGradient(%r*grad(%r))" % (self.coeff, self.field)


class DivCoeffGrad(_EllipticTerm):
    """``scale * div(coeff * grad(phi))`` -- a variable / anisotropic principal operator."""

    def __init__(self, field: Any, coeff: Any, scale: Any = 1.0) -> None:
        self.field = field
        self.coeff = coeff
        self.scale = exact_numeric_scalar(scale, where="DivCoeffGrad scale")

    def _kind(self) -> Any:
        return "div_coeff_grad"

    def __neg__(self) -> Any:
        return DivCoeffGrad(self.field, self.coeff, -self.scale)

    def __repr__(self) -> str:
        lead = exact_scale_prefix(self.scale)
        return "DivCoeffGrad(%sdiv(%r*grad(%r)))" % (lead, self.coeff, self.field)


class EllipticSum(_EllipticTerm):
    """An accumulated sum of elliptic operator terms (laplacian / div-coeff-grad / reaction)."""

    def __init__(self, terms: Any) -> None:
        self.terms = tuple(terms)

    def _elliptic_terms(self) -> Any:
        return list(self.terms)

    def _principal_kinds(self) -> Any:
        kinds = set()
        for term in self.terms:
            kinds |= term._principal_kinds()
        return kinds

    def _kind(self) -> Any:
        return "sum"

    def __neg__(self) -> Any:
        return EllipticSum([-term for term in self.terms])

    def __repr__(self) -> str:
        return "EllipticSum(%r)" % (self.terms,)


__all__ = [
    "Reaction", "CoeffGradient", "DivCoeffGrad", "EllipticSum",
    "constant_reaction_scalar", "elliptic_terms", "principal_kinds",
]
