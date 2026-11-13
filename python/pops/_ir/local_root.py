"""The pointwise recovery of the coordinates of a nonlinear accumulation (G03 lot A, note §6).

A balance ``ddt(Q(q)) == R`` stores U, the coefficients of Q(q) under its representation, and
reads its coordinates q as a definition over U: q is the root of Q(q) = U on the admissible set of
q. This module holds the two values of that recovery:

- ``LocalRoot``, the realization: its tolerance and its iteration budget. The surface keys it by
  the accumulation in ``solvers=``; below the surface, ``model.accumulation`` takes it as
  ``recovery=`` (``g03/decisions_pilote.md``, lot A);
- ``PointwiseRoot``, the expression q(U). Its one child is U, so a flux reads q like any primitive
  of U, and its derivative is the exact dq/dU = 1/Q'(q). Its native form is a bracketed Newton
  iteration on each point, which fails closed: a value of U with no admissible root, a law that is
  not finite, or a budget spent without the tolerance gives NaN, which every native recovery reads
  as a failed candidate.

The law is held over a bound variable of its own, so the scheme never sees the coordinates as a
free variable of the state and no chain rule Q'(q)·Δq ever stands in for the finite difference of
U (K6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .expr import (
    Abs, Add, Const, Div, Exp, Expr, Log, Mul, Neg, Pow, Sqrt, Sub, UnaryMath, Var, _wrap)

BOUND_KIND = "local_root"


@dataclass(frozen=True, slots=True)
class LocalRoot:
    """``LocalRoot(tolerance=, max_iterations=)``: the pointwise root of an accumulation law.

    The recovery stops at the first q of the admissible set with |Q(q) − U| <= tolerance·(1 + |U|),
    and fails after ``max_iterations`` evaluations of the law (the bracket search included)."""

    tolerance: float = 1e-13
    max_iterations: int = 200
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.tolerance) is not float or not (0.0 < self.tolerance < 1.0):
            raise ValueError("LocalRoot(tolerance=) is a binary64 number in (0, 1)")
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 10000:
            raise ValueError("LocalRoot(max_iterations=) is an integer in [1, 10000]")

    def to_data(self) -> dict[str, Any]:
        return {"kind": "local_root", "tolerance": self.tolerance.hex(),
                "max_iterations": self.max_iterations}


def substitute(expr: Any, name: str, replacement: Expr, *, kind: str = BOUND_KIND) -> Expr:
    """``expr`` with the variable ``name`` of ``kind`` replaced, over the elementary algebra only."""
    memo: dict[int, Expr] = {}

    def go(node: Any) -> Expr:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, Var):
            result = replacement if (node.kind == kind and node.name == name) else node
        elif isinstance(node, Const):
            result = node
        elif isinstance(node, (Add, Sub, Mul, Div, Pow)):
            result = type(node)(go(node.a), go(node.b))
        elif isinstance(node, (Neg, Sqrt, Exp, Abs)):
            result = type(node)(go(node.a))
        elif isinstance(node, UnaryMath):
            result = type(node)(go(node.a))
        else:
            raise TypeError("an accumulation law is written with + - * / ** sqrt exp log sin cos "
                            "and constants over its coordinates; %s is not" % type(node).__name__)
        memo[id(node)] = result
        return result

    return go(_wrap(expr))


def derivative_interval(expr: Any, name: str, lower: float, upper: float) -> tuple[float, float]:
    """An enclosure of ``expr`` for the bound variable in [lower, upper], by interval arithmetic.

    Raises ``ValueError`` when the enclosure is not defined (a division by an interval holding 0,
    a root or a logarithm of a possibly negative interval, a node outside the elementary algebra)."""

    def mul(a: tuple, b: tuple) -> tuple:
        products = [x*y for x in a for y in b if not ((x == 0 and math.isinf(y))
                                                      or (y == 0 and math.isinf(x)))]
        if len(products) < 4:
            products.append(0.0)
        return (min(products), max(products))

    def go(node: Any) -> tuple[float, float]:
        if isinstance(node, Const):
            value = float(node.value)
            return (value, value)
        if isinstance(node, Var):
            if node.kind == BOUND_KIND and node.name == name:
                return (lower, upper)
            raise ValueError("the law reads %s, which is not its coordinate" % node.name)
        if isinstance(node, Add):
            a, b = go(node.a), go(node.b)
            return (a[0] + b[0], a[1] + b[1])
        if isinstance(node, Sub):
            a, b = go(node.a), go(node.b)
            return (a[0] - b[1], a[1] - b[0])
        if isinstance(node, Neg):
            a = go(node.a)
            return (-a[1], -a[0])
        if isinstance(node, Mul):
            if node.a is node.b or (isinstance(node.a, Var) and isinstance(node.b, Var)
                                    and node.a.name == node.b.name):
                a = go(node.a)
                if a[0] >= 0:
                    return (a[0]*a[0], a[1]*a[1])
                if a[1] <= 0:
                    return (a[1]*a[1], a[0]*a[0])
                return (0.0, max(a[0]*a[0], a[1]*a[1]))
            return mul(go(node.a), go(node.b))
        if isinstance(node, Div):
            b = go(node.b)
            if b[0] <= 0 <= b[1]:
                raise ValueError("the law divides by a value that may vanish")
            return mul(go(node.a), (min(1/b[0], 1/b[1]), max(1/b[0], 1/b[1])))
        if isinstance(node, Pow) and isinstance(node.b, Const):
            exponent = node.b.value
            if isinstance(exponent, int) and exponent >= 0:
                a, result = go(node.a), (1.0, 1.0)
                if exponent % 2 == 0 and a[0] < 0 < a[1]:
                    top = max(abs(a[0]), abs(a[1]))**exponent
                    return (0.0, top)
                ends = (a[0]**exponent, a[1]**exponent)
                return (min(ends), max(ends)) if exponent else result
        if isinstance(node, Sqrt):
            a = go(node.a)
            if a[0] < 0:
                raise ValueError("the law takes the root of a value that may be negative")
            return (math.sqrt(a[0]), math.sqrt(a[1]))
        if isinstance(node, Exp):
            a = go(node.a)
            return (math.exp(a[0]) if a[0] > -745 else 0.0,
                    math.exp(a[1]) if a[1] < 709 else math.inf)
        if isinstance(node, Log):
            a = go(node.a)
            if a[0] <= 0:
                raise ValueError("the law takes the logarithm of a value that may vanish")
            return (math.log(a[0]), math.log(a[1]))
        raise ValueError("no enclosure for %s" % type(node).__name__)

    return go(_wrap(expr))


class PointwiseRoot(Expr):
    """q(U): the root of ``law`` = U over [lower, upper] (``upper`` may be +inf, held as None)."""

    def __init__(self, law: Expr, bound: Var, target: Any, *, lower: float, upper: float,
                 recovery: LocalRoot, name: str) -> None:
        from .lowering import diff
        if not isinstance(bound, Var) or bound.kind != BOUND_KIND:
            raise TypeError("PointwiseRoot binds a variable of kind %r" % BOUND_KIND)
        if type(recovery) is not LocalRoot:
            raise TypeError("PointwiseRoot takes its realization as a LocalRoot")
        if not (math.isfinite(lower) and lower < upper):
            raise ValueError("the admissible set of a recovery is [lower, upper), lower finite")
        self.law = substitute(law, bound.name, bound)
        self.slope = diff(self.law, bound)
        self.bound = bound
        self.target = _wrap(target)
        self.lower = float(lower)
        self.upper = None if math.isinf(upper) else float(upper)
        self.recovery = recovery
        self.name = name

    def __pops_ir_children__(self) -> tuple:
        return (self.target,)

    def __pops_ir_key__(self, recurse: Any) -> Any:
        from .visitors import _key
        return ("pointwise_root", self.name, _key(self.law), self.bound.name,
                self.lower.hex(), None if self.upper is None else self.upper.hex(),
                self.recovery.tolerance.hex(),
                self.recovery.max_iterations, recurse(self.target))

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Any:
        from .lowering import _s_div
        return _s_div(recurse(self.target), substitute(self.slope, self.bound.name, self))

    def deps(self) -> Any:
        return self.target.deps()

    def eval(self, env: Any) -> Any:
        import numpy as np
        values = np.asarray(self.target.eval(env), dtype=float)
        flat = [self.solve(float(value)) for value in values.reshape(-1)]
        result = np.asarray(flat, dtype=float).reshape(values.shape)
        return result if values.shape else float(result)

    def solve(self, value: float) -> float:
        """The host twin of the native iteration, step for step."""
        tolerance = self.recovery.tolerance*(1.0 + abs(value))

        def law(q: float) -> tuple[float, float]:
            env = {self.bound.name: q}
            return float(self.law.eval(env)) - value, float(self.slope.eval(env))

        if not math.isfinite(value):
            return math.nan
        a, b = self.lower, math.inf if self.upper is None else self.upper
        have_b = math.isfinite(b)
        q, step, first = self.lower, max(1.0, abs(self.lower)), True
        for _ in range(self.recovery.max_iterations):
            f, df = law(q)
            if not math.isfinite(f):
                return math.nan
            if abs(f) <= tolerance:
                return q
            if f < 0:
                a = q
            else:
                if first:
                    return math.nan
                b, have_b = q, True
            first = False
            if not have_b:
                q, step = self.lower + step, 2*step
                continue
            following = q - f/df if df > 0 else math.nan
            if not (a < following < b):
                following = a + (b - a)/2
            if following <= a or following >= b:
                return math.nan
            q = following
        return math.nan

    def emit_cpp(self, target_cpp: str) -> str:
        """The native iteration as one immediately invoked lambda; ``target_cpp`` is U."""
        from pops.codegen.cpp_writer import _cpp_expand
        q = self.bound.name
        law, slope = _cpp_expand(self.law, {}), _cpp_expand(self.slope, {})
        upper = ("Kokkos::Experimental::infinity<pops::Real>::value" if self.upper is None
                 else repr(self.upper))
        nan = "Kokkos::Experimental::quiet_NaN<pops::Real>::value"
        return "".join((
            "([&]() -> pops::Real {",
            " const pops::Real pops_root_u = %s;" % target_cpp,
            " if (!Kokkos::isfinite(pops_root_u)) return %s;" % nan,
            " const pops::Real pops_root_tolerance = %r * (pops::Real(1) + Kokkos::fabs(pops_root_u));"
            % self.recovery.tolerance,
            " const pops::Real pops_root_lower = %r;" % self.lower,
            " pops::Real pops_root_a = pops_root_lower, pops_root_b = %s;" % upper,
            " bool pops_root_bracketed = Kokkos::isfinite(pops_root_b), pops_root_first = true;",
            " pops::Real pops_root_x = pops_root_lower,"
            " pops_root_step = Kokkos::fmax(pops::Real(1), Kokkos::fabs(pops_root_lower));",
            " for (int pops_root_it = 0; pops_root_it < %d; ++pops_root_it) {"
            % self.recovery.max_iterations,
            "  const pops::Real %s = pops_root_x;" % q,
            "  const pops::Real pops_root_f = (%s) - pops_root_u;" % law,
            "  const pops::Real pops_root_df = %s;" % slope,
            "  if (!Kokkos::isfinite(pops_root_f)) return %s;" % nan,
            "  if (Kokkos::fabs(pops_root_f) <= pops_root_tolerance) return pops_root_x;",
            "  if (pops_root_f < pops::Real(0)) pops_root_a = pops_root_x;",
            "  else { if (pops_root_first) return %s;" % nan,
            " pops_root_b = pops_root_x; pops_root_bracketed = true; }",
            "  pops_root_first = false;",
            "  if (!pops_root_bracketed) {",
            " pops_root_x = pops_root_lower + pops_root_step; pops_root_step = pops::Real(2) * pops_root_step;"
            " continue; }",
            "  pops::Real pops_root_next = pops_root_df > pops::Real(0)"
            " ? pops_root_x - pops_root_f / pops_root_df : %s;" % nan,
            "  if (!(pops_root_a < pops_root_next && pops_root_next < pops_root_b))",
            " pops_root_next = pops_root_a + (pops_root_b - pops_root_a) / pops::Real(2);",
            "  if (pops_root_next <= pops_root_a || pops_root_next >= pops_root_b) return %s;" % nan,
            "  pops_root_x = pops_root_next;",
            " }",
            " return %s;" % nan,
            "}())",
        ))

    def to_cpp(self) -> str:
        return self.emit_cpp(self.target.to_cpp())

    def _str(self) -> str:
        return "root[%s](%s = %s)" % (self.name, self.law, self.target)


__all__ = ["LocalRoot", "PointwiseRoot", "derivative_interval", "substitute"]
