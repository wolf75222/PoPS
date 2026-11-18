"""Typed product-lattice reads, independent of storage and distribution (G09 F1)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from pops._ir.index_maps import (
    Adjacency, Compose, Drop, IndexMap, Inverse, Pin, Project, Shift, Translate, Unpin, cell, dense,
)
from pops._ir.index_sets import Entities, IndexSet, Lattice, ProductIndexSet


@dataclass(frozen=True, slots=True, eq=False)
class ProductRead:
    """One point read, including the fibre of intermediate products.

    Equality is by occurrence: map equality may erase the very bounds to be checked.
    """

    array: object
    map: IndexMap
    source: IndexSet
    target: IndexSet
    fibre: IndexSet

    def __post_init__(self):
        if not isinstance(self.map, IndexMap):
            raise TypeError("a point read requires an index map")


@dataclass(frozen=True, slots=True, eq=False)
class FibreFoldRead:
    """One consumer of a fibre relation, with its authored law and optional base map."""

    array: object
    adjacency: Adjacency
    domain: ProductIndexSet
    law: object
    base_shift: IndexMap | None = None

    def __post_init__(self):
        if type(self.adjacency) is not Adjacency:
            raise TypeError("a fibre fold requires an adjacency")
        if self.base_shift is not None and not isinstance(self.base_shift, IndexMap):
            raise TypeError("the fibre fold's base map must be an index map")


@dataclass(frozen=True, slots=True)
class ProductFootprint:
    """The base displacement and, for product targets, one pin or an entire fibre."""

    offset: tuple[int, ...]
    source: Lattice
    target: Lattice
    pin: tuple[int, ...] | None = None
    fibre_extents: tuple[int, ...] = ()

    def indices(self, consumer: tuple[int, ...]):
        """The requested indices, preserving the first fibre axis as the fastest one."""
        if len(consumer) != self.source.rank:
            raise ValueError("the consumer index does not match the authored read domain")
        base = tuple(i + delta for i, delta in
                     zip(consumer[:len(self.offset)], self.offset, strict=True))
        if self.pin is not None:
            yield base + self.pin
        elif self.fibre_extents:
            axes = tuple(range(n) for n in reversed(self.fibre_extents))
            for point in product(*axes):
                yield base + point[::-1]
        else:
            yield base


class ProductReadUnknown(ValueError):
    """A product read whose domain or image is not yet proved."""


def _lattice(space):
    if not isinstance(space, IndexSet) or (row := space.descriptor(Lattice)) is None:
        raise ProductReadUnknown("the read requires a lattice descriptor")
    return row


def _base(space):
    return space.base if type(space) is ProductIndexSet else space


def _same_base(source, target):
    return (source == target or
            (type(source) is type(target) is Entities
             and source.configuration == target.configuration))


def _atoms(map_):
    """Use the structured expression before a normal form can erase a partial map."""
    pending = [map_]
    while pending:
        current = pending.pop()
        inverted = False
        while type(current) is Inverse:
            inverted = not inverted
            current = current.of
        if type(current) is Compose:
            if inverted:
                raise ProductReadUnknown("the inverse requires a domain proof through a composition")
            pending.extend((current.outer, current.inner))
        else:
            yield from reversed((Inverse(current) if inverted else current).normal_form())


def _bound_extent(size, extents):
    if type(size) is str and size not in extents:
        raise ProductReadUnknown("the fibre extent %r must be bound" % size)
    bound = extents[size] if type(size) is str else size
    if type(bound) is not int or bound <= 0:
        raise ValueError("a bound fibre extent must be a positive integer, got %r" % bound)
    return bound


def _check_pin(point, shape, extents, unresolved):
    if len(point) != len(shape):
        raise ValueError("the pin rank does not match its fibre")
    for axis, (index, size) in enumerate(zip(point, shape, strict=True)):
        if index >= 0 and type(size) is str and size not in extents:
            unresolved.add(size)
        elif index < 0 or index >= _bound_extent(size, extents):
            raise ValueError("fibre index %d is outside its extent on axis %d" % (index, axis))


def point_footprint(read: ProductRead, extents) -> ProductFootprint:
    source, target = _lattice(read.source), _lattice(read.target)
    base_rank = _lattice(_base(read.source)).rank
    if not _same_base(_base(read.source), _base(read.target)):
        raise ProductReadUnknown("the read changes its base support")
    fibre = _lattice(read.fibre)
    if fibre.entity != cell:
        raise ProductReadUnknown("the read requires a cell fibre")
    for space in (read.source, read.target):
        if type(space) is ProductIndexSet and space.fibre != read.fibre:
            raise ValueError("the read and its product must share one fibre")

    in_product = type(read.source) is ProductIndexSet
    pinned = None
    offset = (0,) * base_rank
    entity = source.entity
    unresolved = set()
    for atom in _atoms(read.map):
        if type(atom) is Translate:
            if atom.source not in (None, entity):
                raise ValueError("the translation starts on another entity")
            entity = entity if atom.target is None else atom.target
            if entity.piece is not None or (entity.axis is not None and entity.axis >= base_rank):
                raise ProductReadUnknown("the read has an entity outside the base axes")
            delta = dense(atom.offset, base_rank + fibre.rank if in_product else base_rank)
            offset = tuple(a + b for a, b in zip(offset, delta[:base_rank], strict=True))
            if any(delta[base_rank:]):
                if pinned is None:
                    raise ProductReadUnknown("a translated variable fibre needs a boundary domain")
                pinned = tuple(a + b for a, b in zip(pinned, delta[base_rank:], strict=True))
                _check_pin(pinned, fibre.extents, extents, unresolved)
        elif type(atom) in (Pin, Drop, Unpin):
            if atom.base_rank != base_rank or in_product == (type(atom) is Pin):
                raise ValueError("the rank change does not match its product domain")
            if type(atom) is Pin:
                _check_pin(atom.fibre, fibre.extents, extents, unresolved)
                pinned = atom.fibre
            elif type(atom) is Unpin:
                _check_pin(atom.fibre, fibre.extents, extents, unresolved)
                if pinned is None:
                    raise ProductReadUnknown("Unpin requires proof that the input lies on its fibre")
                if pinned != atom.fibre:
                    raise ValueError("Unpin is undefined at the current fibre point")
            in_product = type(atom) is Pin
            if not in_product:
                pinned = None
        else:
            raise ProductReadUnknown("no product domain proof for %r" % (atom,))

    if in_product != (type(read.target) is ProductIndexSet) or entity != target.entity:
        raise ValueError("the map does not arrive at its declared read support")
    if in_product and pinned is None:
        raise ProductReadUnknown("the read keeps a variable fibre coordinate")
    if unresolved:
        raise ProductReadUnknown("fibre extents must be bound: %s" % ", ".join(sorted(unresolved)))
    return ProductFootprint(offset, source, target, pin=pinned)


def fold_footprint(read: FibreFoldRead, extents) -> ProductFootprint:
    projection = read.adjacency.inverse_of
    if type(projection) is not Project:
        raise ProductReadUnknown("no fibre proof for the inverse of %r" % projection)
    base = _lattice(read.domain.base)
    fibre = _lattice(read.domain.fibre)
    if fibre.entity != cell or projection.base_rank != base.rank:
        raise ValueError("the adjacency does not project this lattice product")
    if read.base_shift is not None and type(read.base_shift) is not Shift:
        raise ProductReadUnknown("no fibre fold composition with %r" % read.base_shift)
    offset = (0,) * base.rank if read.base_shift is None else read.base_shift.offset(base.rank)
    return ProductFootprint(offset, base, _lattice(read.domain),
                            fibre_extents=tuple(_bound_extent(size, extents) for size in
                                                fibre.extents))
