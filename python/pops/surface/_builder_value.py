"""The kind of a builder value: a load, a total, or the third kind (spec 0.3, 2.6.4; N16 4.3).

N15, FaceFlux/CellRate lot FC-1 (note §3.3). A builder value is typed:

* a load, built from ``cell.load(e)``, one entry per test function (D2), served under one constant
  mode by FC-2;
* a total, ``cell.integral(e)`` or ``face.integral(e)`` without ``test``, which N16 lot 2 adds;
* any other value, read at the centre of its entity under one constant mode, which the builder
  tests; under ``CellAverage()`` it is the rate itself (N16 5.1, A1).

A load added to a value of the third kind, and a total as a builder value, are refused
(``AmbiguousElaboration``), as the spec states (2.6.4). Every branch under a point-dependent
representation is ``NotYetElaborated`` naming N16 lot 2: this module is the one place that N16
lot 2 replaces to add the loads with several test functions, and no call site changes.

The kinds are identifiers of this module, never refused string literals of CF2.
"""
from __future__ import annotations

from typing import Any

from ._expr import walk
from ._refusal import AmbiguousElaboration, NotYetElaborated, Site

_N16_LOT_2 = "N16 lot 2, with the point-dependent builder values of 4.3"


class BuilderValue:
    """A builder value with its kind: ``TESTED`` (read at the centre, the rate of a cell average),
    ``LOAD`` or ``TOTAL``."""

    TESTED = "a value read at the centre of the cell, neither a load nor a total"
    LOAD = "load"
    TOTAL = "total"

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: Any) -> None:
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        return "BuilderValue(%s)" % self.kind


def _is_load(node: Any) -> bool:
    key = getattr(node, "key", None)
    return isinstance(key, tuple) and bool(key) and key[0] == "load"


def _is_total(node: Any) -> bool:
    key = getattr(node, "key", None)
    return isinstance(key, tuple) and bool(key) and key[0] == "integral"


def classify(value: Any, *, word: str, definition: Any, site: Site | None) -> BuilderValue:
    """The kind of the value of the builder ``word`` under the representation ``definition`` of
    the key's quantity, or the refusal of 2.6.4 and N16 4.3."""
    nodes = list(walk(value))
    if any(_is_total(node) for node in nodes):
        raise AmbiguousElaboration(
            "%s(%s) takes a total, an integral over the cell, as its value" % (word, value),
            write="cell.load(e), or cell.integral(e) / cell.measure for a value read as a "
                  "density", site=site)
    if getattr(definition, "point_dependent", False):
        raise NotYetElaborated("%s under a representation whose decode reads the point" % word,
                               _N16_LOT_2, site=site)
    if any(_is_load(node) for node in nodes):
        # FC-2 serves the loads of one constant mode; until then cell.load is refused at its read.
        raise NotYetElaborated("%s of a load, cell.load(e)" % word,
                               "the N15 lot of cell.load (FaceFlux/CellRate lot FC-2)",
                               site=site)
    return BuilderValue(BuilderValue.TESTED, value)


__all__ = ["BuilderValue", "classify"]
