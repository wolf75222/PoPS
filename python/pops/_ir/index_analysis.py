"""Three-valued analyses over typed index maps (spec 0.3 IR13, IR14, IR17, IR20, IR21).

A ``Region`` is what the analyses read of one kernel: the entities it maps over (its domain), its
reads (an array read through an index map), the signed targets through which its value leaves
when it is a scatter, and the class its builder expects. Every analysis answers Proved, Disproved
or Unknown (IR13) and reads only normal forms and coefficient lists, so its cost depends on the
number of reads and maps, never on an extent or a trip count (B4). Unknown enables nothing:
``require`` refuses it with ``PropertyNotDerived``, naming the region and the first blocking map.

The halo is exact box arithmetic, symbolic in the extents (IR14): a rank owns the cells [0, n) of
each axis, a face region computes the faces of its owned cells, [0, n] along the normal axis, and a
translation by c reads the cells [c, n - 1 + c], one more along the normal of a face. The widths
beyond the owned box do not depend on n. b6 stores one width per axis, the larger side of its own
formulas (``ghost_extent``, ``ghost_depth``); the builders report that value beside the exact one,
for IR-T1.

Classification (IR20) reaches pointwise, box stencil and face incidence when every map of the
region is a translation from its domain, and otherwise names the class its atoms give (level
transfer, axis reduction or broadcast, CSR for a data map, generic) with the first map that is not
a translation. The loop-nest generators of each class are phase 2 of N7.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from .index_maps import (
    Coarsen, Drop, Entity, IndexMap, Opaque, Pin, Refine, Unpin, Unrefine, dense, exact_weight)
from .refusals import ClassNotReached, PropertyNotDerived

CLASSES = ("pointwise", "box_stencil", "face_incidence", "axis_reduction", "line_scan",
           "level_transfer", "csr", "generic")


class Truth(enum.Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Fact:
    """The answer of one analysis: its truth, its value when proved, what blocks it otherwise."""
    property: str
    truth: Truth
    value: object = None
    blocking: object = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Read:
    """``array`` read through ``map``: a gather of IR4. ``array`` is the builder's label."""
    array: object
    map: IndexMap

    def __post_init__(self):
        if not isinstance(self.map, IndexMap):
            raise TypeError("a read goes through an index map, got %r" % (self.map,))


@dataclass(frozen=True, slots=True)
class Region:
    """What the analyses read of one kernel.

    ``targets`` holds the ``(map, weight)`` pairs of a ``scatter_reduce`` (IR17); an empty tuple
    writes element by element over the domain. ``expected`` is the class the builder records
    (IR21). ``basis`` says where the reads come from: ``"derived"`` when they are collected from a
    body, ``"declared"`` when a native catalogue states them (a substitution of IR24 whose
    reference region is not written yet).
    """
    name: str
    domain: Entity
    rank: int
    reads: tuple
    targets: tuple = ()
    expected: str | None = None
    basis: str = "derived"

    def __post_init__(self):
        if type(self.domain) is not Entity:
            raise TypeError("a region maps over the entities of a lattice, got %r" % (self.domain,))
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("a region has a rank >= 1, got %r" % (self.rank,))
        if self.domain.axis is not None and self.domain.axis >= self.rank:
            raise ValueError("%r is not an entity of a rank-%d lattice" % (self.domain, self.rank))
        if any(type(read) is not Read for read in self.reads):
            raise TypeError("a region's reads are Read records")
        for pair in self.targets:
            if type(pair) is not tuple or len(pair) != 2 or not isinstance(pair[0], IndexMap):
                raise TypeError("a scatter target is a (map, weight) pair, got %r" % (pair,))
        if self.expected is not None and self.expected not in CLASSES:
            raise ValueError("unknown lowering class %r; the classes are %s"
                             % (self.expected, ", ".join(CLASSES)))
        if self.basis not in ("derived", "declared"):
            raise ValueError("a region's reads are derived or declared, got %r" % (self.basis,))

    def halo(self, where=None):
        return halo(self, where)

    def conservation(self):
        return conservation(self)

    def classify(self):
        return classify(self)

    def check_expected_class(self):
        return check_expected_class(self)


def _translation_from(map_, domain):
    return map_.translation() is not None and map_.source in (None, domain)


def _maps(region):
    return tuple(read.map for read in region.reads) + tuple(m for m, _ in region.targets)


def halo(region, where=None):
    """Exact widths ``(low, high)`` per axis of the cells read beyond the owned box (IR14).

    ``where``, a predicate on the array labels, restricts the reads. Unknown when a read is not a
    translation from the domain, or when the domain is a boundary piece, whose widths depend on
    the extents.
    """
    if region.domain.piece is not None:
        return Fact("halo", Truth.UNKNOWN, blocking=region.domain,
                    detail="the faces of a boundary piece: the widths depend on the extents")
    low, high = [0]*region.rank, [0]*region.rank
    for read in region.reads:
        if where is not None and not where(read.array):
            continue
        if not _translation_from(read.map, region.domain):
            return Fact("halo", Truth.UNKNOWN, blocking=read.map,
                        detail="%r is not a translation from %r" % (read.map, region.domain))
        offset = dense(read.map.translation(), region.rank)
        for axis, step in enumerate(offset):
            grow = int(region.domain.kind == "face" and region.domain.axis == axis)
            low[axis] = max(low[axis], -step)
            high[axis] = max(high[axis], step + grow)
    return Fact("halo", Truth.PROVED, value=(tuple(low), tuple(high)))


def conservation(region):
    """Sum_k w_k = 0 over the targets of the region's scatter (IR17).

    Unknown when the region writes element by element (no identity to check) or when a weight is
    not an exact number.
    """
    if not region.targets:
        return Fact("conservation", Truth.UNKNOWN,
                    detail="the region writes element by element: it delivers no shared value")
    total = 0
    for _, weight in region.targets:
        exact = exact_weight(weight)
        if exact is None:
            return Fact("conservation", Truth.UNKNOWN, blocking=weight,
                        detail="the weight %r is not an exact number" % (weight,))
        total += exact
    truth = Truth.PROVED if total == 0 else Truth.DISPROVED
    return Fact("conservation", truth, value=total)


def injective(map_):
    """Whether ``map_`` sends distinct entities to distinct entities (IR9, IR14).

    Translations, ``Child`` and ``Embed`` are; ``Parent`` alone is not (fan-out r^rank). The
    fan-out of ``Project`` is the fibre cardinality of the descriptor, and a data map's inverse is
    the inspector's (IR11): both are Unknown here.
    """
    atoms = map_.normal_form()
    for atom in atoms:
        if type(atom) is Opaque:
            return Fact("injective", Truth.UNKNOWN, blocking=map_,
                        detail="a %s map: its inspector decides" % atom.provenance)
        if type(atom) is Drop:
            return Fact("injective", Truth.UNKNOWN, blocking=map_,
                        detail="Project: its fan-out is the fibre cardinality |V|")
        if type(atom) is Coarsen:
            if len(atoms) == 1:
                return Fact("injective", Truth.DISPROVED, value=("ratio^rank", atom.ratio))
            return Fact("injective", Truth.UNKNOWN, blocking=map_,
                        detail="a composition that holds Parent")
    return Fact("injective", Truth.PROVED)


def classify(region):
    """``(class, blocking map)`` from the normal forms of the region's maps (IR20)."""
    maps = _maps(region)
    blocking = next((m for m in maps if not _translation_from(m, region.domain)), None)
    if blocking is None:
        if region.domain.kind == "face":
            return "face_incidence", None
        if not region.targets and all(not m.translation() for m in maps):
            return "pointwise", None
        return "box_stencil", None
    atoms = [atom for m in maps for atom in m.normal_form()]
    if any(type(atom) is Opaque and atom.provenance == "data" for atom in atoms):
        return "csr", blocking
    if any(type(atom) is Opaque for atom in atoms):
        return "generic", blocking
    if any(type(atom) in (Coarsen, Refine, Unrefine) for atom in atoms):
        return "level_transfer", blocking
    if any(type(atom) in (Drop, Pin, Unpin) for atom in atoms):
        return "axis_reduction", blocking
    return "generic", blocking


def check_expected_class(region):
    """The class classification gives ``region``; ``ClassNotReached`` when it is not the expected one.

    Nothing falls back silently (IR21, I15).
    """
    reached, blocking = classify(region)
    if region.expected is not None and reached != region.expected:
        raise ClassNotReached(
            "region %r expects the class %s, classification gives %s; the first map that blocks "
            "it is %r" % (region.name, region.expected, reached, blocking),
            region=region.name, expected=region.expected, reached=reached, blocking=blocking)
    return reached


def require(fact, region):
    """``fact.value`` when the fact is not Unknown; otherwise ``PropertyNotDerived`` (IR13)."""
    if fact.truth is Truth.UNKNOWN:
        raise PropertyNotDerived(
            "region %r needs the property %s, which the analysis answers Unknown: %s; the first "
            "blocking map or coefficient is %r" % (region.name, fact.property, fact.detail,
                                                   fact.blocking),
            region=region.name, property=fact.property, blocking=fact.blocking)
    return fact.value


__all__ = ["CLASSES", "Fact", "Read", "Region", "Truth", "check_expected_class", "classify",
           "conservation", "halo", "injective", "require"]
