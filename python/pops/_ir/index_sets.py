"""Index sets of IR2 and their descriptors of IR3 (spec 0.3 IR2, IR3, DM2; N10 phase 1, lot 1a).

An index set carries meaning: the cells of a configuration, the faces normal to one of its axes,
a level of a hierarchy, a finite range computed during the run, a product, a disjoint union.
The constructors are the lowercase functions of this module; the capitalized descriptors
``Lattice``, ``ProductSet`` and ``Hierarchy`` are type-level facts that only constructors attach
(IR3), so they are true by construction and no author can attach one to an arbitrary set.

Equality is by instance, not by printed form. A configuration is named by the identity of the
support placement it lays out (spec 0.3 §4.1: κ assigns a layout to each support), so two
supports with equal grids still have two distinct configurations and two distinct sets of cells,
and co-located blocks, which share one support, share one set of cells. A hierarchy is named by
its instance, and a regrid issues a new one (DM13). A ``range_set`` is named by its instance
path, which also addresses its capacity (IR2, B7). Extents are integer constants or the names of
bound parameters.

``levels(H)`` is notation, not a constructor: the disjoint union of ``level(H, l)`` over the
stored levels, covered cells included. Each level carries ``Hierarchy``, as IR3 gives it to
``level`` and ``composite``; the union carries no descriptor.

A descriptor says nothing about storage: the plan chooses every layout and distribution (DM3).

Scope at this commit. The entity kinds are those of ``pops._ir.index_maps.Entity`` (cells and
the faces normal to an axis); edges and nodes come with G08. ``subset``, ``coarsen`` and
``refine`` of IR2 come with the issues that need them. Nothing here caps a rank (B6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops._ir.index_maps import Entity, cell, face

__all__ = [
    "BoxSet", "Composite", "Configuration", "Entities", "Hierarchy", "HierarchyInstance",
    "IndexSet", "Lattice", "Level", "ProductIndexSet", "ProductSet", "RangeSet", "Union", "box",
    "cells", "composite", "entities", "faces", "level", "levels", "product", "range_set", "union",
]


def _identity(value: Any, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be a non-empty identity string" % where)
    return value


def _extents(values: Any, where: str) -> tuple[Any, ...]:
    rows = tuple(values)
    if not rows:
        raise ValueError("%s has rank one or more" % where)
    for value in rows:
        if type(value) is int:
            if value < 1:
                raise ValueError("%s extents are positive, got %r" % (where, value))
        elif type(value) is not str or not value:
            raise TypeError("%s extents are integers or bound parameter names, got %r"
                            % (where, value))
    return rows


def _periodic(values: Any, rank: int, where: str) -> tuple[bool, ...]:
    rows = tuple(values)
    if len(rows) != rank or any(type(value) is not bool for value in rows):
        raise TypeError("%s periodic is one exact bool per axis" % where)
    return rows


# -- descriptors (IR3) ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lattice:
    """``Lattice(rank, extents, periodic, entity)``: the entities of a lattice."""

    rank: int
    extents: tuple[Any, ...]
    periodic: tuple[bool, ...]
    entity: Entity


@dataclass(frozen=True)
class ProductSet:
    """``ProductSet(X, V)``: the projection onto X and the embedding of each fibre."""

    base: Any
    fibre: Any


@dataclass(frozen=True)
class Hierarchy:
    """``Hierarchy(levels, ratio)``: the levels and the ratio between consecutive levels.

    A ratio is one integer, or one integer per axis as the b6 hierarchy writes it.
    """

    levels: int
    ratios: tuple[int, ...]


# -- what a constructor reads -------------------------------------------------------------------

@dataclass(frozen=True)
class Configuration:
    """κ: the configuration of one support placement, a lattice of cells (D0, §4.1)."""

    identity: str
    extents: tuple[Any, ...]
    periodic: tuple[bool, ...]

    def __post_init__(self):
        _identity(self.identity, "a configuration")
        object.__setattr__(self, "extents", _extents(self.extents, "a configuration"))
        object.__setattr__(self, "periodic",
                           _periodic(self.periodic, len(self.extents), "a configuration"))

    @property
    def rank(self) -> int:
        return len(self.extents)


@dataclass(frozen=True)
class HierarchyInstance:
    """H: one instance of the hierarchy of a configuration; a regrid issues a new one."""

    identity: str
    configuration: Configuration
    levels: int
    ratios: tuple[int, ...]

    def __post_init__(self):
        _identity(self.identity, "a hierarchy instance")
        if type(self.configuration) is not Configuration:
            raise TypeError("a hierarchy instance lays out one Configuration")
        if type(self.levels) is not int or self.levels < 1:
            raise ValueError("a hierarchy has one level or more")
        ratios = tuple(_ratio(ratio, self.configuration.rank) for ratio in self.ratios)
        if len(ratios) != self.levels - 1:
            raise ValueError("a hierarchy has one ratio per pair of levels")
        object.__setattr__(self, "ratios", ratios)


def _ratio(value: Any, rank: int) -> Any:
    """A ratio between two levels: one integer, or one integer per axis, refining at least once."""
    if type(value) is int:
        if value < 2:
            raise ValueError("a hierarchy ratio is 2 or more, got %r" % (value,))
        return value
    rows = tuple(value) if isinstance(value, (tuple, list)) else None
    if rows is None or len(rows) != rank or any(type(r) is not int or r < 1 for r in rows) \
            or max(rows) < 2:
        raise ValueError("a per-axis hierarchy ratio has one positive integer per axis and "
                         "refines along one axis at least, got %r" % (value,))
    return rows


# -- index sets (IR2) ---------------------------------------------------------------------------

class IndexSet:
    """An index set; its descriptors come from its constructor only."""

    __slots__ = ()

    @property
    def descriptors(self) -> tuple[Any, ...]:
        return ()

    def descriptor(self, kind: type) -> Any:
        """The descriptor of type ``kind`` that the constructor attached, or ``None``."""
        for value in self.descriptors:
            if type(value) is kind:
                return value
        return None


@dataclass(frozen=True)
class Entities(IndexSet):
    """``entities(κ, kind)``."""

    configuration: Configuration
    entity: Entity

    @property
    def descriptors(self) -> tuple[Any, ...]:
        κ = self.configuration
        return (Lattice(κ.rank, κ.extents, κ.periodic, self.entity),)


@dataclass(frozen=True)
class BoxSet(IndexSet):
    """``box(extents, periodic=...)`` in ℤ^r."""

    extents: tuple[Any, ...]
    periodic: tuple[bool, ...]

    @property
    def descriptors(self) -> tuple[Any, ...]:
        return (Lattice(len(self.extents), self.extents, self.periodic, cell),)


@dataclass(frozen=True)
class Level(IndexSet):
    """``level(H, l)``: every cell of level l of H, covered cells included."""

    hierarchy: HierarchyInstance
    level: int

    @property
    def descriptors(self) -> tuple[Any, ...]:
        return (Hierarchy(self.hierarchy.levels, self.hierarchy.ratios),)


@dataclass(frozen=True)
class Composite(IndexSet):
    """``composite(H)``: the non-overlapping cover of H."""

    hierarchy: HierarchyInstance

    @property
    def descriptors(self) -> tuple[Any, ...]:
        return (Hierarchy(self.hierarchy.levels, self.hierarchy.ratios),)


@dataclass(frozen=True)
class RangeSet(IndexSet):
    """``range_set(n)``: a cardinality computed during the run, named by its instance path."""

    path: str


@dataclass(frozen=True)
class ProductIndexSet(IndexSet):
    """``product(X, V)``."""

    base: IndexSet
    fibre: IndexSet

    @property
    def descriptors(self) -> tuple[Any, ...]:
        rows: list[Any] = [ProductSet(self.base, self.fibre)]
        x, v = self.base.descriptor(Lattice), self.fibre.descriptor(Lattice)
        if x is not None and v is not None:
            rows.append(Lattice(x.rank + v.rank, x.extents + v.extents,
                                x.periodic + v.periodic, x.entity))
        return tuple(rows)


@dataclass(frozen=True)
class Union(IndexSet):
    """A disjoint ``union(D1, ..., Dk)``; it carries no descriptor."""

    parts: tuple[IndexSet, ...]


# -- constructors -------------------------------------------------------------------------------

def entities(configuration: Configuration, kind: Entity) -> Entities:
    if type(configuration) is not Configuration:
        raise TypeError("entities() reads a Configuration")
    if type(kind) is not Entity or kind.piece is not None:
        raise TypeError("entities() takes the entity kind cell or face(a)")
    if kind.axis is not None and kind.axis >= configuration.rank:
        raise ValueError("face(%d) is not an axis of a rank-%d configuration"
                         % (kind.axis, configuration.rank))
    return Entities(configuration, kind)


def cells(configuration: Configuration) -> Entities:
    """``cells(κ) = entities(κ, cell)``."""
    return entities(configuration, cell)


def faces(configuration: Configuration, axis: int) -> Entities:
    """``faces(κ, a) = entities(κ, face(a))``."""
    return entities(configuration, face(axis))


def box(extents: Any, *, periodic: Any = None) -> BoxSet:
    rows = _extents(extents, "a box")
    flags = (False,) * len(rows) if periodic is None else _periodic(periodic, len(rows), "a box")
    return BoxSet(rows, flags)


def level(hierarchy: HierarchyInstance, index: int) -> Level:
    if type(hierarchy) is not HierarchyInstance:
        raise TypeError("level() reads a HierarchyInstance")
    if type(index) is not int or not 0 <= index < hierarchy.levels:
        raise ValueError("level %r is not a level of a %d-level hierarchy"
                         % (index, hierarchy.levels))
    return Level(hierarchy, index)


def composite(hierarchy: HierarchyInstance) -> Composite:
    if type(hierarchy) is not HierarchyInstance:
        raise TypeError("composite() reads a HierarchyInstance")
    return Composite(hierarchy)


def range_set(path: str) -> RangeSet:
    return RangeSet(_identity(path, "a range_set instance path"))


def product(base: IndexSet, fibre: IndexSet) -> ProductIndexSet:
    if not isinstance(base, IndexSet) or not isinstance(fibre, IndexSet):
        raise TypeError("product() reads two index sets")
    return ProductIndexSet(base, fibre)


def union(*parts: IndexSet) -> Union:
    if len(parts) < 2 or any(not isinstance(part, IndexSet) for part in parts):
        raise TypeError("union() reads two index sets or more")
    if len(set(parts)) != len(parts):
        raise ValueError("a disjoint union names each part once")
    return Union(tuple(parts))


def levels(hierarchy: HierarchyInstance) -> IndexSet:
    """Notation for ``union(level(H, 0), ..., level(H, L-1))``; ``level(H, 0)`` when L = 1."""
    rows = tuple(level(hierarchy, index) for index in range(hierarchy.levels))
    return rows[0] if len(rows) == 1 else union(*rows)
