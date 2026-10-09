"""Explicit arithmetic for normalized straight-density paths (contract @1).

These expressions are authored once in Python and executed per cell in native
code. They describe arithmetic, not a closure or a physical moment ordering.
Polynomial degree is checked before fixed-capacity native multiplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Arithmetic:
    op: str
    args: tuple
    degree: int = 0
    polynomial: bool = False

    def __add__(self, other):
        return operation("add", self, other)

    def __radd__(self, other):
        return operation("add", other, self)

    def __mul__(self, other):
        return operation("multiply", self, other)

    def __rmul__(self, other):
        return operation("multiply", other, self)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -literal(other)

    def __rsub__(self, other):
        return literal(other) + -self

    def __abs__(self):
        if self.polynomial:
            raise TypeError("absolute value requires scalar arithmetic")
        return Arithmetic("abs", (self,))

    def __truediv__(self, other):
        if type(other) not in (int, float) or not isfinite(other) or other == 0:
            raise ValueError("arithmetic division requires a finite nonzero literal")
        return Arithmetic("divide_literal", (self, other), self.degree, self.polynomial)

    def __pow__(self, exponent):
        if type(exponent) is not int or exponent < 0:
            raise ValueError("polynomial power requires a nonnegative exact integer")
        return Arithmetic("power", (self, exponent), self.degree * exponent, self.polynomial)

    def to_data(self):
        return {"op": self.op, "args": tuple(
            arg.to_data() if isinstance(arg, Arithmetic) else arg for arg in self.args),
            "degree": self.degree, "polynomial": self.polynomial}


def literal(value):
    if type(value) is Arithmetic:
        return value
    if type(value) not in (int, float) or not isfinite(value):
        raise TypeError("arithmetic requires a finite real literal or an exact Arithmetic")
    return Arithmetic("literal", (value,))


def operation(op, *values):
    args = tuple(literal(value) for value in values)
    polynomial = any(arg.polynomial for arg in args)
    degree = sum(arg.degree for arg in args) if op == "multiply" else max(arg.degree for arg in args)
    if op not in ("add", "multiply") or len(args) != 2:
        raise ValueError("unknown binary arithmetic operation")
    return Arithmetic(op, args, degree, polynomial)


def derivative(value):
    value = literal(value)
    if not value.polynomial:
        raise TypeError("derivative requires a polynomial")
    return Arithmetic("derivative", (value,), max(0, value.degree - 1), True)


def fma(a, b, c):
    args = tuple(literal(value) for value in (a, b, c))
    if any(arg.polynomial for arg in args):
        raise TypeError("fma is scalar arithmetic; polynomial products are explicit")
    return Arithmetic("fma", args)


def compensated_sum(values):
    args = tuple(literal(value) for value in values)
    if not args or any(arg.polynomial for arg in args):
        raise TypeError("compensated_sum requires a nonempty scalar sequence")
    return Arithmetic("compensated_sum", args)


def square_root(value):
    value = literal(value)
    if value.polynomial:
        raise TypeError("square_root requires scalar arithmetic")
    return Arithmetic("sqrt", (value,))


class NormalizedPathInputs:
    """Inputs to either endpoint arithmetic or the normalized affine path.

    ``basis`` explicitly binds storage slots to complete Cartesian monomials.
    Only the current two-coordinate certified covariance realization is
    supported, without imposing a physical closure or total degree ceiling.
    """

    def __init__(self, basis, *, polynomial=False):
        from .basis import CartesianMonomialBasis
        if type(basis) is not CartesianMonomialBasis or type(polynomial) is not bool:
            raise TypeError("normalized inputs require an exact CartesianMonomialBasis and boolean polynomial mode")
        self.indices = tuple(basis.indices)
        self.order = basis.order
        if basis.dimension != 2 or self.order < 2:
            raise ValueError("normalized covariance realization requires a complete 2D basis of order >=2")
        self.polynomial = polynomial

    def normalized(self, index):
        slot = self.indices.index(tuple(index))
        return Arithmetic("normalized", (slot,), int(self.polynomial), self.polynomial)

    def raw(self, index):
        if self.polynomial:
            raise ValueError("raw input is an endpoint value; use the normalized polynomial path")
        return Arithmetic("raw", (self.indices.index(tuple(index)),))

    @property
    def density(self):
        return Arithmetic("density", ())

    def direction(self, axis):
        if type(axis) is not int or axis not in (0, 1):
            raise ValueError("direction axis must be an exact coordinate index")
        return Arithmetic("direction", (axis,))

    def second_moment_norm(self):
        if self.polynomial:
            raise TypeError("certified second-moment norm is an endpoint scalar operation")
        return Arithmetic("second_moment_norm", ())


def normalized_polynomial_path(basis, *, flux, integrands, factors, speed):
    """Versioned authored expressions consumed by a closure-free emitter.

    Density weighting and canonical orientation are generic native numerical
    operations. The caller supplies *all* fluxes, integrands and scalar factors.
    """
    indices = tuple(basis.indices)
    size = len(indices)
    flux, integrands, factors = tuple(flux), tuple(integrands), tuple(factors)
    if not (len(flux) == len(integrands) == len(factors) == size):
        raise ValueError("path expressions must cover every exact basis component")
    flux = tuple(literal(value) for value in flux)
    integrands = tuple(literal(value) for value in integrands)
    speed = literal(speed)
    if any(value.polynomial for value in (*flux, speed)):
        raise ValueError("flux and speed must be scalar endpoint expressions")
    if any(type(value) not in (int, float) or not isfinite(value) for value in factors):
        raise ValueError("path factors must be finite real literals")
    nodes, memo = [], {}
    def intern(value):
        args = tuple(intern(arg) if type(arg) is Arithmetic else arg for arg in value.args)
        key = value.op, args, value.degree, value.polynomial
        if key not in memo:
            memo[key] = len(nodes)
            nodes.append({"op": value.op, "args": args, "degree": value.degree,
                          "polynomial": value.polynomial})
        return memo[key]
    flux = tuple(intern(value) for value in flux)
    integrands = tuple(intern(value) for value in integrands)
    speed = intern(speed)
    polynomial_degree = max((node["degree"] for node in nodes if node["polynomial"]), default=0)
    if not 0 <= polynomial_degree < 2**31 - 1:
        raise ValueError("polynomial capacity exceeds the native signed coefficient index width")
    return {"schema_version": 1, "order": basis.order, "polynomial_degree": polynomial_degree, "indices": indices,
            "nodes": tuple(nodes), "flux": flux, "integrands": integrands,
            "factors": factors, "speed": speed,
            "integral_mode": "density_weighted_polynomial", "integral": None}


class EndpointPathInputs(NormalizedPathInputs):
    """Scalar endpoint coordinates for an explicitly authored analytic integral."""

    def left(self, index):
        return Arithmetic("left_raw", (self.indices.index(tuple(index)),))

    def right(self, index):
        return Arithmetic("right_raw", (self.indices.index(tuple(index)),))


def endpoint_polynomial_path(basis, *, flux, integral, speed):
    """Declare a straight-raw-path endpoint formula, without selecting a law.

    This shares the certified endpoint recovery and numerical orientation with
    the density-weighted realization. The supplied exact analytic integral and
    whole-path speed require a mathematical library proof. No quadrature or
    formula recognition is performed by the compiler.
    """
    integral = tuple(literal(value) for value in integral)
    if len(integral) != len(basis.indices) or any(value.polynomial for value in integral):
        raise ValueError("endpoint integral must cover every exact scalar component")
    plan = normalized_polynomial_path(basis, flux=flux,
        integrands=(0,)*len(basis.indices), factors=(0,)*len(basis.indices), speed=speed)
    nodes = list(plan["nodes"])
    memo = {(node["op"], node["args"], node["degree"], node["polynomial"]): k
            for k, node in enumerate(nodes)}
    def intern(value):
        args = tuple(intern(arg) if type(arg) is Arithmetic else arg for arg in value.args)
        key = value.op, args, value.degree, value.polynomial
        if key not in memo:
            memo[key] = len(nodes)
            nodes.append({"op": value.op, "args": args, "degree": value.degree,
                          "polynomial": value.polynomial})
        return memo[key]
    plan["integral"] = tuple(intern(value) for value in integral)
    plan["nodes"] = tuple(nodes)
    plan["integral_mode"] = "analytic_endpoint"
    return plan
