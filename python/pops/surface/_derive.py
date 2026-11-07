"""Exact derivatives of the values of a model, for the wave speeds of R9 (spec 0.3, R9, 2.6.6).

The elaborator derives the wave speed of a scalar flux F(u) as dF/du. The flux of b6 carries it as
its waves, and ``upwind()`` reads it as its speed s = F(1)·n when F = F(1) u (SD44). This module
computes the derivative exactly over the field algebra and the elementary functions that this lot
elaborates, and simplifies the sums and products by 0 and 1 that the rules produce, so that the
speed of ``a * u`` is ``a``. A value outside that algebra has no derived speed (``None``): the
scheme that needs one says so.
"""
from __future__ import annotations

from typing import Any

from ._expr import Arith, Call, is_number, walk


def reads(value: Any, symbol: Any) -> bool:
    """Whether ``value`` reads ``symbol``."""
    return any(node is symbol for node in walk(value))


def _is(value: Any, number: int) -> bool:
    return is_number(value) and value == number


def add(left: Any, right: Any) -> Any:
    if is_number(left) and is_number(right):
        return left + right
    if _is(left, 0):
        return right
    if _is(right, 0):
        return left
    return Arith("+", (left, right))


def sub(left: Any, right: Any) -> Any:
    if is_number(left) and is_number(right):
        return left - right
    if _is(right, 0):
        return left
    if _is(left, 0):
        return neg(right)
    return Arith("-", (left, right))


def neg(value: Any) -> Any:
    if is_number(value):
        return -value
    if isinstance(value, Arith) and value.op == "neg":
        return value.args[0]
    return Arith("neg", (value,))


def mul(left: Any, right: Any) -> Any:
    if is_number(left) and is_number(right):
        return left * right
    if _is(left, 0) or _is(right, 0):
        return 0
    if _is(left, 1):
        return right
    if _is(right, 1):
        return left
    if _is(left, -1):
        return neg(right)
    if _is(right, -1):
        return neg(left)
    coefficient, rest = _coefficient(right)
    if _integer(left) and coefficient is not None:
        return mul(left * coefficient, rest)
    return Arith("*", (left, right))


def div(left: Any, right: Any) -> Any:
    if _is(left, 0):
        return 0
    if _is(right, 1):
        return left
    if _integer(left) and _integer(right) and right != 0 and left % right == 0:
        return left // right
    coefficient, rest = _coefficient(left)
    if _integer(right) and right != 0 and coefficient is not None and coefficient % right == 0:
        return mul(coefficient // right, rest)
    return Arith("/", (left, right))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coefficient(value: Any) -> tuple:
    """``(c, x)`` when ``value`` is ``c * x`` with an integer ``c``, else ``(None, value)``.

    Only integer coefficients fold, so that each folded product or quotient is exact and the value
    keeps the rounding of what the user wrote: ``2 * u / 2`` is ``u``, ``u / 3`` stays.
    """
    if isinstance(value, Arith) and value.op == "*" and _integer(value.args[0]):
        return value.args[0], value.args[1]
    if isinstance(value, Arith) and value.op == "neg":
        coefficient, rest = _coefficient(value.args[0])
        if coefficient is not None:
            return -coefficient, rest
    return None, value


def power(base: Any, exponent: Any) -> Any:
    if _is(exponent, 1):
        return base
    if _is(exponent, 0):
        return 1
    return Arith("**", (base, exponent))


def substitute(value: Any, symbol: Any, number: Any) -> Any:
    """``value`` with ``symbol`` replaced by ``number``, simplified; ``None`` outside the
    algebra."""
    if value is symbol:
        return number
    if not reads(value, symbol):
        return value
    if isinstance(value, Arith):
        args = [substitute(arg, symbol, number) for arg in value.args]
        if any(arg is None for arg in args):
            return None
        if value.op == "neg":
            return neg(args[0])
        rule = {"+": add, "-": sub, "*": mul, "/": div, "**": power}[value.op]
        return rule(*args)
    return None


def derivative(value: Any, symbol: Any) -> Any:
    """d value / d symbol, exactly, or ``None`` when ``value`` leaves the algebra of this lot."""
    if value is symbol:
        return 1
    if not reads(value, symbol):
        return 0
    if isinstance(value, Arith):
        return _arith(value, symbol)
    if isinstance(value, Call) and len(value.args) == 1:
        inner = derivative(value.args[0], symbol)
        if inner is None:
            return None
        argument = value.args[0]
        outer = {
            "exp": lambda: Call("exp", (argument,)),
            "expm1": lambda: Call("exp", (argument,)),
            "log": lambda: div(1, argument),
            "sqrt": lambda: div(1, mul(2, Call("sqrt", (argument,)))),
            "sin": lambda: Call("cos", (argument,)),
            "cos": lambda: neg(Call("sin", (argument,))),
        }.get(value.word)
        return None if outer is None else mul(outer(), inner)
    return None


def _arith(value: Arith, symbol: Any) -> Any:
    if value.op == "neg":
        inner = derivative(value.args[0], symbol)
        return None if inner is None else neg(inner)
    left, right = value.args
    if value.op == "**":
        if reads(right, symbol) or not is_number(right):
            return None
        inner = derivative(left, symbol)
        return None if inner is None else mul(mul(right, power(left, right - 1)), inner)
    d_left = derivative(left, symbol)
    d_right = derivative(right, symbol)
    if d_left is None or d_right is None:
        return None
    if value.op == "+":
        return add(d_left, d_right)
    if value.op == "-":
        return sub(d_left, d_right)
    if value.op == "*":
        return add(mul(d_left, right), mul(left, d_right))
    if not reads(right, symbol):                                # value.op == "/"
        return div(d_left, right)
    return div(sub(mul(d_left, right), mul(left, d_right)), power(right, 2))


def unit_flux(value: Any, symbol: Any) -> Any:
    """F(1) when ``value`` is F(1) u, linear in the scalar ``symbol`` u; ``None`` otherwise."""
    slope = derivative(value, symbol)
    if slope is None or reads(slope, symbol):
        return None
    origin = substitute(value, symbol, 0)
    return slope if _is(origin, 0) else None
