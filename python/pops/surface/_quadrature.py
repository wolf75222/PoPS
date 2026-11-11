"""``pops.Quadrature(nodes=, weights=, interval=)``: the one quadrature construct of the core.

Spec 0.3, D0 and 2.6.2 (review of 11/11, P3): a quadrature rule is n nodes and n weights on a
reference interval, Python numbers computed when the case is written (E4); on a cell or a product,
one rule applies on every axis, or a tuple gives one rule per axis. The core knows no rule by name:
``GaussLegendre(n)``, ``GaussHermite(n)`` and any other named rule are functions of
``pops.lib.quadrature`` that return one (A4), each with its parity test (CF4).

The nodes and weights of a rule realize every integral over what it lays out:
``sum(w * f(v) for v, w in zip(nodes, weights))`` is the integral of f over the interval. The core
does not check a degree of exactness, which belongs to the rule that the library or the user
writes; it checks only that the rule is data that a plan can store: two sequences of the same
length n >= 1 of finite numbers, the nodes inside the interval, the interval two numbers or
expressions of numbers, ``pi`` and ``infinity`` with lower < upper. A weight may be negative (a
Newton-Cotes rule of high order has some), and an end of the interval may be a node (a Lobatto
rule).
"""
from __future__ import annotations

import math as _math
from typing import Any

from ._expr import is_number
from ._geometry import fold
from ._refusal import user_site


def _numbers(values: Any, what: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError("pops.Quadrature(%s=...) takes a sequence of numbers, got %r"
                        % (what, values))
    numbers = tuple(values)
    if not numbers:
        raise ValueError("pops.Quadrature(%s=...) takes at least one number" % what)
    for value in numbers:
        if not is_number(value) or not _math.isfinite(float(value)):
            raise TypeError("pops.Quadrature(%s=...) takes finite Python numbers, computed when "
                            "the case is written (E4), got %r" % (what, value))
    return tuple(float(value) for value in numbers)


class Quadrature:
    """``pops.Quadrature(nodes=[...], weights=[...], interval=(lower, upper))``: a rule of one's
    own, n nodes and n weights on a reference interval (D0)."""

    __slots__ = ("nodes", "weights", "interval", "site")

    def __init__(self, *, nodes: Any, weights: Any, interval: Any) -> None:
        self.nodes = _numbers(nodes, "nodes")
        self.weights = _numbers(weights, "weights")
        if len(self.nodes) != len(self.weights):
            raise ValueError("pops.Quadrature takes as many weights as nodes, got %d nodes and %d "
                             "weights" % (len(self.nodes), len(self.weights)))
        if isinstance(interval, list) or not isinstance(interval, tuple) or len(interval) != 2:
            raise TypeError("pops.Quadrature(interval=...) takes a reference interval (lower, "
                            "upper), such as (-1, 1) or (-infinity, infinity), got %r"
                            % (interval,))
        lower, upper = (float(fold(end, "pops.Quadrature(interval=(lower, upper))"))
                        for end in interval)
        if not lower < upper:
            raise ValueError("pops.Quadrature(interval=(lower, upper)) needs lower < upper, got "
                             "%r" % (interval,))
        outside = [node for node in self.nodes if not lower <= node <= upper]
        if outside:
            raise ValueError("pops.Quadrature: the nodes %s lie outside the interval %r"
                             % (", ".join(repr(node) for node in outside), (lower, upper)))
        self.interval = (lower, upper)
        self.site = user_site(2)

    @property
    def n(self) -> int:
        """The number of nodes of the rule."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return "pops.Quadrature(nodes=%r, weights=%r, interval=%r)" % (
            self.nodes, self.weights, self.interval)
