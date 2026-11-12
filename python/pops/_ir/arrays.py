"""The value types of IR1 that the data model needs (spec 0.3 IR1, DM1; N10 phase 1, lot 1a).

One array type: every indexed value of a Program is ``Array(D, T)``, one value of ``T`` per index
of the index set ``D`` (``pops._ir.index_sets``). A field of a block is the array over the cells
of the configuration of its support; a finite array is the array over its ``range_set``. Which
storage holds an array is a fact of the plan (DM3), never of its type, so two arrays over the same
index set with the same element type have the same type whatever their layout or ownership.

The types are frozen values with structural equality: ``Float64`` is ``Float64``, and
``Array(D, T)`` equals ``Array(D', T')`` exactly when ``D == D'`` and ``T == T'``, the index sets
comparing by instance (``index_sets``). ``Product`` and ``Variant`` keep the order of their
fields, which names their components as the bridge and the codec name them.

``IndexMap`` and ``Adjacency`` stay in ``pops._ir.index_maps``. ``GraphFunction`` is a body, not
a value type of this module.

Scope at this commit. Nothing in the lowering or the acceptance reads these types: the plan
computes them and the manifest reports them (lot 1a), and the dispatch of the emitters moves onto
them with the retyping (lot 1b). Another scalar width is a core extension under IR26.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Array", "Bool", "Float64", "Index", "IndexSetType", "Int64", "Iteration", "NotData",
    "Product", "Result", "Scalar", "Untyped", "Variant", "is_type",
]


@dataclass(frozen=True)
class Scalar:
    """A scalar type of IR1: ``Bool``, ``Int64`` or ``Float64``."""

    name: str

    def __post_init__(self):
        if self.name not in ("Bool", "Int64", "Float64"):
            raise ValueError("an IR1 scalar is Bool, Int64 or Float64, got %r" % (self.name,))

    def __repr__(self):
        return self.name


Bool = Scalar("Bool")
Int64 = Scalar("Int64")
Float64 = Scalar("Float64")


def _fields(kind: str, fields: Any, *, empty: bool = False) -> tuple[tuple[str, Any], ...]:
    rows = tuple(fields)
    if not rows and not empty:
        raise ValueError("a %s has at least one field" % kind)
    names = []
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("a %s field is a (name, type) pair" % kind)
        name, value = row
        if type(name) is not str or not name:
            raise TypeError("a %s field name is a non-empty string" % kind)
        _require_type(value, where="%s field %r" % (kind, name))
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("a %s names each field once, got %r" % (kind, names))
    return rows


@dataclass(frozen=True, init=False)
class Product:
    """The record type ``Product(name=T, ...)``, its fields in their written order.

    ``Product()`` is the empty record (spec 0.3, IR1): a Program product with no member has it.
    """

    fields: tuple[tuple[str, Any], ...]

    def __init__(self, *pairs, **named):
        if pairs and named:
            raise TypeError("a Product takes (name, type) pairs or keywords, not both")
        object.__setattr__(self, "fields",
                           _fields("Product", pairs or tuple(named.items()), empty=True))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.fields)

    def __repr__(self):
        return "Product(%s)" % ", ".join("%s=%r" % row for row in self.fields)


@dataclass(frozen=True, init=False)
class Variant:
    """The tagged union ``Variant(name=T, ...)``, its cases in their written order."""

    cases: tuple[tuple[str, Any], ...]

    def __init__(self, *pairs, **named):
        if pairs and named:
            raise TypeError("a Variant takes (name, type) pairs or keywords, not both")
        object.__setattr__(self, "cases", _fields("Variant", pairs or tuple(named.items())))

    def __repr__(self):
        return "Variant(%s)" % ", ".join("%s=%r" % row for row in self.cases)


@dataclass(frozen=True)
class Result:
    """``Result(T, E)``: a value of ``T``, or a typed failure ``E``."""

    value: Any
    failure: Any

    def __post_init__(self):
        _require_type(self.value, where="Result value")
        _require_type(self.failure, where="Result failure")


@dataclass(frozen=True)
class Iteration:
    """``Iteration(S)``: the final state of type ``S`` of ``iterate``, its step count and checks."""

    state: Any

    def __post_init__(self):
        _require_type(self.state, where="Iteration state")


@dataclass(frozen=True)
class IndexSetType:
    """``IndexSet``: the type of a value that is an index set, here the set it denotes."""

    of: Any

    def __post_init__(self):
        _require_index_set(self.of, where="IndexSet")


@dataclass(frozen=True)
class Index:
    """``Index(D)``: an index of ``D``; it indexes only arrays over ``D``."""

    domain: Any

    def __post_init__(self):
        _require_index_set(self.domain, where="Index")


@dataclass(frozen=True)
class Array:
    """``Array(D, T)``: one value of ``T`` per index of ``D``."""

    domain: Any
    element: Any

    def __post_init__(self):
        _require_index_set(self.domain, where="Array domain")
        _require_type(self.element, where="Array element")

    def __repr__(self):
        return "Array(%r, %r)" % (self.domain, self.element)


@dataclass(frozen=True)
class NotData:
    """A Program value that is not data (an operator, an effect, a solve outcome): outside DM1.

    ``spelling`` is the value type the Program builder wrote, kept for the inventory and the
    manifest.
    """

    spelling: str


@dataclass(frozen=True)
class Untyped:
    """A data value whose IR type the plan cannot derive yet, with the reason.

    Nothing refuses it (review of the N10 note, A1): the manifest and the inventory report it, and
    the issue named in ``reason`` types it.
    """

    reason: str

    def __post_init__(self):
        if type(self.reason) is not str or not self.reason:
            raise TypeError("an Untyped value states its reason")


_TYPES = (Scalar, Product, Variant, Result, Iteration, IndexSetType, Index, Array, Untyped)


def is_type(value: Any) -> bool:
    """Whether ``value`` is a value type of IR1 built by this module."""
    return isinstance(value, _TYPES)


def _require_type(value: Any, *, where: str) -> None:
    if not is_type(value):
        raise TypeError("%s must be an IR1 type, got %r" % (where, value))


def _require_index_set(value: Any, *, where: str) -> None:
    from pops._ir.index_sets import IndexSet

    if not isinstance(value, IndexSet):
        raise TypeError("%s must be an index set of pops._ir.index_sets, got %r" % (where, value))
