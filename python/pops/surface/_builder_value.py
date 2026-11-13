"""The kind of a builder value: a load, a total, or the third kind (spec 0.3, 2.6.4; N16 4.3).

N15, FaceFlux/CellRate lots FC-1 and FC-2 (note §3.3, §3.6). A builder value is typed:

* a load, built from ``cell.load(e)``, one entry per test function (D2); under one constant mode
  a sum of loads, each scaled by a factor that reads no quantity, no coordinate and no load, whose
  rate is the mean of e under the rule of the representation (FC-2);
* a total, ``cell.integral(e)`` or ``face.integral(e)`` without ``test``, which N16 lot 2 adds;
* any other value, read at the centre of its entity under one constant mode, which the builder
  tests; under ``CellAverage()`` it is the rate itself (N16 5.1, A1).

A load added to a value of the third kind, a load under another node, and a total as a builder
value are refused (``AmbiguousElaboration``), as the spec states (2.6.4). Every branch under a
point-dependent representation is ``NotYetElaborated`` naming N16 lot 2: this module is the one
place that N16 lot 2 replaces to add the loads with several test functions, and no call site
changes.

The kinds are identifiers of this module, never refused string literals of CF2.
"""
from __future__ import annotations

from typing import Any

from ._expr import Arith, text, walk
from ._refusal import AmbiguousElaboration, NotYetElaborated, Site

_N16_LOT_2 = "N16 lot 2, with the point-dependent builder values of 4.3"
_RULE_READS = "N16 lot 2 (the reads at the points of a rule, 4.1)"
_PARAMETER_AUXILIARIES = "a later lot of N15 with the auxiliary inputs that read parameters"


class BuilderValue:
    """A builder value with its kind: ``TESTED`` (read at the centre, the rate of a cell average),
    ``LOAD`` or ``TOTAL``. ``loads`` lists the ``cell.load(e)`` of a load, in the order written."""

    TESTED = "a value read at the centre of the cell, neither a load nor a total"
    LOAD = "a load, the mean of its integrand under the rule of the representation"
    TOTAL = "total"

    __slots__ = ("kind", "value", "loads")

    def __init__(self, kind: str, value: Any, loads: tuple = ()) -> None:
        self.kind = kind
        self.value = value
        self.loads = loads

    def __repr__(self) -> str:
        return "BuilderValue(%s)" % self.kind


def _is_load(node: Any) -> bool:
    key = getattr(node, "key", None)
    return isinstance(key, tuple) and bool(key) and key[0] == "load"


def _is_total(node: Any) -> bool:
    key = getattr(node, "key", None)
    return isinstance(key, tuple) and bool(key) and key[0] == "integral"


def _reads_point(node: Any) -> bool:
    """Whether a factor reads the state or a coordinate, so that it is not a number on the cell."""
    from ._geometry import Coordinate
    from ._model import Quantity

    for inner in walk(node):
        key = getattr(inner, "key", None)
        if isinstance(inner, (Coordinate, Quantity)) or (isinstance(key, tuple) and key
                                                         and key[0] == "sample"):
            return True
    return False


def _has_load(node: Any) -> bool:
    return any(_is_load(inner) for inner in walk(node))


class _Mixed(Exception):
    def __init__(self, part: Any, hint: str) -> None:
        super().__init__(part)
        self.part = part
        self.hint = hint


def _shape(node: Any) -> bool:
    """Whether ``node`` is a sum of loads scaled by factors (True) or holds no load (False); a
    load in any other place raises ``_Mixed`` with the part and the construct to write."""
    if _is_load(node):
        return True
    if not _has_load(node):
        return False
    if isinstance(node, Arith) and node.op == "neg":
        return _shape(node.args[0])
    if isinstance(node, Arith) and node.op in ("+", "-"):
        left, right = (_shape(arg) for arg in node.args)
        if left != right:
            raise _Mixed(node, "a sum of loads, cell.load(e1) + cell.load(e2), or the value read "
                               "at the centre alone")
        return True
    if isinstance(node, Arith) and node.op in ("*", "/"):
        left, right = node.args
        loaded, factor = (left, right) if _has_load(left) else (right, left)
        if node.op == "/" and loaded is right:
            raise _Mixed(node, "cell.load(e / f), the factor inside the load")
        if _has_load(factor):
            raise _Mixed(node, "one load, cell.load(e1 * e2)")
        if _reads_point(factor):
            raise _Mixed(node, "cell.load(%s * e), the factor that varies on the cell inside "
                               "the load" % text(factor))
        return _shape(loaded)
    raise _Mixed(node, "cell.load(e) at the top of the value, or scaled by a number")


def classify(value: Any, *, word: str, definition: Any, site: Site | None) -> BuilderValue:
    """The kind of the value of the builder ``word`` under the representation ``definition`` of
    the key's quantity, or the refusal of 2.6.4 and N16 4.3."""
    from ._model import Parameter

    nodes = list(walk(value))
    if any(_is_total(node) for node in nodes):
        raise AmbiguousElaboration(
            "%s(%s) takes a total, an integral over the cell, as its value" % (word, value),
            write="cell.load(e), or cell.integral(e) / cell.measure for a value read as a "
                  "density", site=site)
    if getattr(definition, "point_dependent", False):
        raise NotYetElaborated("%s under a representation whose decode reads the point" % word,
                               _N16_LOT_2, site=site)
    try:
        loaded = _shape(value)
    except _Mixed as mixed:
        raise AmbiguousElaboration(
            "%s(%s) holds a load, cell.load(e), in %s, where it is neither a load nor a value "
            "read at the centre (D2)" % (word, text(value), text(mixed.part)),
            write=mixed.hint, site=site) from None
    if not loaded:
        return BuilderValue(BuilderValue.TESTED, value)
    loads = tuple(node for node in nodes if _is_load(node))
    if getattr(definition, "pairing", None) is None:
        raise AmbiguousElaboration(
            "%s(%s) reads a load under a representation that states no rule; the load "
            "integrates e under the rule of the representation (N16 4.4)" % (word, text(value)),
            write="CellAverage(quadrature=GaussLegendre(n)) of pops.lib.representations",
            site=site)
    for load in loads:
        for inner in walk(load.value):
            key = getattr(inner, "key", None)
            if isinstance(key, tuple) and key and key[0] == "sample":
                raise NotYetElaborated("%s, a load whose integrand reads %s"
                                       % (text(load), text(inner)), _RULE_READS, site=site)
            if _is_load(inner) and inner is not load:
                raise AmbiguousElaboration("%s, a load of a load" % text(load),
                                           write="one load, cell.load(e)", site=site)
            if isinstance(inner, Parameter):
                raise NotYetElaborated("%s, a load whose integrand reads the parameter %s"
                                       % (text(load), inner), _PARAMETER_AUXILIARIES, site=site)
    return BuilderValue(BuilderValue.LOAD, value, loads)


__all__ = ["BuilderValue", "classify"]
