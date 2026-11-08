"""Three-valued analyses over typed index maps (spec 0.3 IR13, IR14, IR17, IR20, IR21).

A ``Region`` is what the analyses read of one kernel: the entities it maps over (its domain), its
reads (an array read through an index map), the signed targets through which its shared value
leaves when it is a scatter, its productions, and the class its builder expects. Every analysis
answers Proved, Disproved or Unknown (IR13) and reads only normal forms, types and coefficient
lists, so its cost depends on the number of reads and maps, never on an extent or a trip count
(B4). Only Proved enables a property: ``require`` refuses the others with ``PropertyNotDerived``,
or ``ConservationNotStructural`` for conservation, naming the region and the first blocking map.

The halo is exact box arithmetic, symbolic in the extents (IR14): a rank owns the cells [0, n) of
each axis, and the faces normal to a of its cells, [0, n] along a. A translation by c from the
domain reads the entities [c, m - 1 + c] of its target, m being the number of domain entities
along the axis, so it needs c + m - m' beyond the m' owned target entities: one more along the
normal of a face domain read in cells, and none when the target has the domain's kind. The widths
beyond the owned box do not depend on n. b6 stores one width per axis, the larger side of its own
formulas (``ghost_extent``, ``ghost_depth``); the builders report that value beside the exact one,
for IR-T1.

Classification (IR20) reaches pointwise, box stencil and face incidence when every map of the
region is a translation from its domain, and otherwise names the class its atoms give (level
transfer, axis reduction or broadcast, CSR for a data map, generic) with the first map that is not
a translation. The loop-nest generators of each class are phase 2 of N7.

Scope at this commit (MG7). The footprint and the import set of IR14 are derived for translations
only, and a halo through another map (``Parent``, ``Child``, ``Project``, a trace of a boundary
piece) is Unknown; their box rules, symmetry (IR16), D.C = 0 and the transfer identities of IR17,
and the scope of an implicit relation (IR18) come with later lots of N7, the inspector of data maps
(IR11) and the request protocol of computed maps (IR12) with the runtime side of N7.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from .index_maps import (
    Coarsen, Drop, Entity, IndexMap, Opaque, Pin, Refine, Unpin, Unrefine, dense, exact_weight)
from .refusals import ClassNotReached, ConservationNotStructural, PropertyNotDerived

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

    ``targets`` holds the ``(map, weight)`` pairs of the ``scatter_reduce`` of its shared value
    (IR17); an empty tuple writes element by element over the domain. ``productions`` holds the
    ``(map, weight)`` pair of each single-target ``scatter_reduce`` of a per-side value, a
    ``Production`` of §8.1 and never an exchange (IR25, D6): conservation reads the targets only.
    Every map starts from the domain. ``expected`` is the class the builder records (IR21).
    ``basis`` says where the reads come from: ``"derived"`` when they are collected from a body,
    ``"declared"`` when a native catalogue states them (a substitution of IR24 whose reference
    region is not written yet).
    """
    name: str
    domain: Entity
    rank: int
    reads: tuple
    targets: tuple = ()
    expected: str | None = None
    basis: str = "derived"
    productions: tuple = ()

    def __post_init__(self):
        if type(self.domain) is not Entity:
            raise TypeError("a region maps over the entities of a lattice, got %r" % (self.domain,))
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("a region has a rank >= 1, got %r" % (self.rank,))
        if self.domain.axis is not None and self.domain.axis >= self.rank:
            raise ValueError("%r is not an entity of a rank-%d lattice" % (self.domain, self.rank))
        if any(type(read) is not Read for read in self.reads):
            raise TypeError("a region's reads are Read records")
        for pair in self.targets + self.productions:
            if type(pair) is not tuple or len(pair) != 2 or not isinstance(pair[0], IndexMap):
                raise TypeError("a scatter target is a (map, weight) pair, got %r" % (pair,))
        for map_ in _maps(self):
            if map_.source not in (None, self.domain):
                raise TypeError("region %r maps over %r entities, but %r reads %r entities"
                                % (self.name, self.domain, map_, map_.source))
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
    return (tuple(read.map for read in region.reads)
            + tuple(m for m, _ in region.targets + region.productions))


def _normal_to(kind, axis):
    return int(kind.kind == "face" and kind.axis == axis)


def halo(region, where=None):
    """Exact widths ``(low, high)`` per axis of the entities read beyond the owned box (IR14).

    ``where``, a predicate on the array labels, restricts the reads. A read of type (None, None)
    reads entities of the domain's kind. Unknown when a read is not a translation from the domain,
    or when the domain or a read target is a boundary piece, whose widths depend on the extents.
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
        target = region.domain if read.map.target is None else read.map.target
        if target.piece is not None:
            return Fact("halo", Truth.UNKNOWN, blocking=read.map,
                        detail="%r reads the faces of a boundary piece" % (read.map,))
        offset = dense(read.map.translation(), region.rank)
        for axis, step in enumerate(offset):
            grow = _normal_to(region.domain, axis) - _normal_to(target, axis)
            low[axis] = max(low[axis], -step)
            high[axis] = max(high[axis], step + grow)
    return Fact("halo", Truth.PROVED, value=(tuple(low), tuple(high)))


def conservation(region):
    """Sum_k w_k = 0 over the targets of the scatter of the region's shared value (IR17).

    Productions carry no identity: IR17 checks the shared value only. Unknown when the region
    writes element by element (no identity to check) or when a weight is not an exact number.
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
                return Fact("injective", Truth.DISPROVED, value=(("ratio^rank", atom.ratio),))
            return Fact("injective", Truth.UNKNOWN, blocking=map_,
                        detail="a composition that holds Parent")
    return Fact("injective", Truth.PROVED)


def _identity_on(map_, domain):
    return map_.translation() == () and map_.target in (None, domain)


def classify(region):
    """``(class, blocking map)`` from the normal forms of the region's maps (IR20).

    A region whose every read and delivery is the identity of its domain is pointwise, a scatter
    to its own entities included; then a region over faces is face incidence, and a region over
    cells a box stencil.
    """
    maps = _maps(region)
    blocking = next((m for m in maps if not _translation_from(m, region.domain)), None)
    if blocking is None:
        if all(_identity_on(m, region.domain) for m in maps):
            return "pointwise", None
        if region.domain.kind == "face":
            return "face_incidence", None
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
    """``fact.value`` when the analysis proves the property; otherwise a typed refusal.

    Unknown and Disproved enable nothing (IR13): a conservation that is not proved is refused with
    ``ConservationNotStructural`` (IR17), any other property with ``PropertyNotDerived``.
    """
    if fact.truth is Truth.PROVED:
        return fact.value
    refusal = ConservationNotStructural if fact.property == "conservation" else PropertyNotDerived
    reason = (fact.detail if fact.truth is Truth.UNKNOWN
              else "the analysis disproves it (value %s)" % (fact.value,))
    blocking = ("" if fact.blocking is None
                else "; the first blocking map or coefficient is %r" % (fact.blocking,))
    raise refusal(
        "region %r needs the property %s, which the analysis answers %s: %s%s"
        % (region.name, fact.property, fact.truth.value, reason, blocking),
        region=region.name, property=fact.property, blocking=fact.blocking)


__all__ = ["CLASSES", "Fact", "Read", "Region", "Truth", "check_expected_class", "classify",
           "conservation", "halo", "injective", "require"]
