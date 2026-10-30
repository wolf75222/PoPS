"""Typed index maps (spec 0.3 IR8, IR9): the closed list of constructors and their normal forms.

An index map sends the entities of one index set to entities of another. Its provenance (IR8)
fixes how its facts are derived:

* typed: built by a constructor of the closed list, with a normal form;
* arithmetic: an ``imap`` body over coordinates, which IR10 MAY raise to a normal form;
* data: bound index arrays, inspected once per topology instance (IR11);
* computed: built from the state during the run (IR12).

The closed list of IR9 is ``Shift(o)``, ``Incidence(src, dst, side)``, ``Project``, ``Embed(v)``,
``Parent(r)``, ``Child(r, o)``, ``Trace(piece, depth)``, ``Compose(m2, m1)`` and ``Inverse(m)``.
A map keeps the form it was written in, for diagnostics, and compares by its normal form: two
maps are equal when their normal forms are equal, syntactically, after composition (IR9).

The face of index i normal to axis a lies between the cells i - e_a (its left cell) and i (its
right cell), as the face box of b6 grows the high bound of the normal axis by one
(``include/pops/numerics/spatial/nd/face_field.hpp@b6``). So ``Incidence(face(a), cell,
side="right")`` is the translation by 0 and ``side="left"`` the translation by -e_a, and every
offset of a face read counts from the right cell, as the b6 face samples do.

Normal form. A map normalizes to a tuple of atoms, outermost first:

* ``Translate(source, target, offset)``: i -> i + offset. A translation that keeps the kind of
  the entities carries no kind (``Shift`` acts on what it is composed with); ``Incidence`` and
  ``Trace`` change the kind, from faces to cells;
* ``Coarsen(r)`` and ``Refine(r, o)``, the atoms of ``Parent`` and ``Child`` on cells, and
  ``Unrefine(r, o)``, the inverse of ``Child`` on its image;
* ``Drop(base_rank)`` and ``Pin(fibre, base_rank)``, the atoms of ``Project`` and ``Embed`` on a
  product lattice whose base axes come first (IR3), and ``Unpin``, the inverse of ``Embed``;
* ``Opaque``, a map of another provenance, which no rule rewrites.

Offsets are sparse: the nonzero components by axis, so a normal form does not depend on a rank
that the map does not fix. The rewrite rules are exact identities of the box arithmetic, applied
until none applies; each merges atoms or moves a translation one way, so the number of steps is
bounded by the size of the map and never by an extent (B4):

* two adjacent translations add;
* ``Child(r2, o2)∘Child(r1, o1)`` is ``Child(r1 r2, r2 o1 + o2)`` and ``Parent(r1)∘Parent(r2)``
  is ``Parent(r1 r2)``; ``Parent(r2)∘Child(r1, o)`` is ``Parent(r2/g)∘Child(r1/g, o div g)``, g
  being the gcd of the ratios, so the identity when r1 = r2;
* a translation moves inward through ``Parent`` and ``Child``: ``Shift(q)∘Parent(r)`` is
  ``Parent(r)∘Shift(r q)``, and ``Shift(o)∘Child(r, c)`` is ``Child(r, s)∘Shift(q)`` with
  c + o = r q + s and 0 <= s < r;
* on a product lattice, the two rules of IR9: ``Project`` composed with a translation of the
  product (an ``Incidence`` among them) is the base part of that translation composed with
  ``Project``, and a translation composed with ``Embed(v)`` is ``Embed(v + w)`` composed with its
  base part, w being its fibre part; ``Project∘Embed(v)`` is the identity;
* an atom composed with its own inverse is the identity.

The rules are sound: equal normal forms are equal maps. They are complete, so that equal maps
have equal normal forms, on the translations of one lattice, on ``Parent`` and ``Child`` whose
ratios divide one another, and on ``Project`` and ``Embed`` with translations; a chain of levels
whose ratios are coprime can keep two spellings of one map apart, and is then compared as two
maps. The unit tests check both properties against the definitions of the constructors.

Inversion follows the list: a translation inverts to the opposite translation, ``Child`` and
``Embed`` invert to partial maps on their images. ``Parent`` and ``Project`` are not injective, so
the inverse of a map that holds them is an ``Adjacency`` whose fan-out comes from the descriptor
(r^rank, or |V|).

Scope at this commit. The entities are the cells and the faces of a lattice, and the faces of a
boundary piece; ``Parent`` and ``Child`` act on cells. Edges and nodes (``Incidence(edge(a),
face(b), ±)``), offsets that are bound parameters and the region builder of IR4 come with the
issues that need them. Nothing here caps a rank (B6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

_SIDES = ("left", "right")
_PIECES = ("low", "high")


# ---------------------------------------------------------------------------------------------
# Entities and sparse offsets.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Entity:
    """An entity kind of a lattice: the cells, or the faces normal to one axis.

    ``piece`` restricts faces to the boundary piece of their axis on the low or the high side,
    the source of ``Trace``.
    """
    kind: str
    axis: int | None = None
    piece: str | None = None

    def __post_init__(self):
        if self.kind == "cell":
            if self.axis is not None or self.piece is not None:
                raise ValueError("a cell entity has no axis and no boundary piece")
        elif self.kind == "face":
            _require_axis(self.axis)
            if self.piece is not None and self.piece not in _PIECES:
                raise ValueError("a boundary piece is 'low' or 'high', got %r" % (self.piece,))
        else:
            raise ValueError("an entity kind is 'cell' or 'face', got %r" % (self.kind,))

    def __repr__(self):
        if self.kind == "cell":
            return "cell"
        if self.piece is None:
            return "face(%d)" % self.axis
        return "boundary(%d, %r)" % (self.axis, self.piece)


cell = Entity("cell")


def face(axis):
    """The faces normal to ``axis``."""
    return Entity("face", axis)


def boundary(axis, side):
    """The faces of the boundary piece of ``axis`` on its ``"low"`` or ``"high"`` side."""
    return Entity("face", axis, side)


def _require_axis(axis):
    if type(axis) is not int or axis < 0:
        raise ValueError("an axis is a nonnegative integer, got %r" % (axis,))


def _sparse(offset):
    """The canonical sparse form of an integer offset vector: its nonzero components by axis."""
    if type(offset) is not tuple:
        raise TypeError("an offset is a tuple of integers, got %r" % (offset,))
    if any(type(x) is not int for x in offset):
        raise TypeError("an offset needs integer constants, got %r "
                        "(StructuredOffsetNotConstant)" % (offset,))
    return tuple((axis, x) for axis, x in enumerate(offset) if x)


def _add(a, b):
    total = dict(a)
    for axis, x in b:
        total[axis] = total.get(axis, 0) + x
    return tuple(sorted((axis, x) for axis, x in total.items() if x))


def _neg(a):
    return tuple((axis, -x) for axis, x in a)


def _scale(a, r):
    return tuple((axis, r*x) for axis, x in a)


def dense(sparse, rank):
    """The dense offset vector of rank ``rank`` of a sparse offset."""
    out = [0]*rank
    for axis, x in sparse:
        if axis >= rank:
            raise ValueError("offset %r has an axis beyond rank %d" % (dict(sparse), rank))
        out[axis] = x
    return tuple(out)


def _divmod(sparse, ratio):
    """(q, s) with offset = ratio*q + s and 0 <= s < ratio on every axis."""
    quotient = tuple((axis, x // ratio) for axis, x in sparse if x // ratio)
    remainder = tuple((axis, x % ratio) for axis, x in sparse if x % ratio)
    return quotient, remainder


def _split(sparse, base_rank):
    """(base part, fibre part re-indexed from 0) of an offset of a product lattice."""
    base = tuple((axis, x) for axis, x in sparse if axis < base_rank)
    fibre = tuple((axis - base_rank, x) for axis, x in sparse if axis >= base_rank)
    return base, fibre


# ---------------------------------------------------------------------------------------------
# Atoms of the normal form.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Translate:
    """i -> i + offset, from ``source`` entities to ``target`` entities (None: any kind)."""
    source: Entity | None
    target: Entity | None
    offset: tuple

    def apply(self, index):
        return tuple(i + o for i, o in zip(index, dense(self.offset, len(index)), strict=True))


@dataclass(frozen=True, slots=True)
class Coarsen:
    """``Parent(r)``: the cell of level l that holds a cell of level l + 1."""
    ratio: int
    source = cell
    target = cell

    def apply(self, index):
        return tuple(i // self.ratio for i in index)


@dataclass(frozen=True, slots=True)
class Refine:
    """``Child(r, o)``: the child at offset o in [0, r)^rank of a cell of level l."""
    ratio: int
    offset: tuple
    source = cell
    target = cell

    def apply(self, index):
        return tuple(self.ratio*i + o for i, o in zip(index, dense(self.offset, len(index)),
                                                       strict=True))


@dataclass(frozen=True, slots=True)
class Unrefine:
    """The inverse of ``Child(r, o)``, defined on the children at offset o."""
    ratio: int
    offset: tuple
    source = cell
    target = cell

    def apply(self, index):
        out = []
        for i, o in zip(index, dense(self.offset, len(index)), strict=True):
            if (i - o) % self.ratio:
                return None
            out.append((i - o) // self.ratio)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class Drop:
    """``Project``: the base entity of an entity of a product lattice whose base axes come first."""
    base_rank: int
    source = None
    target = None

    def apply(self, index):
        if len(index) <= self.base_rank:
            raise ValueError("Project needs an index of a product lattice beyond the base rank")
        return index[:self.base_rank]


@dataclass(frozen=True, slots=True)
class Pin:
    """``Embed(v)``: the entity of the product lattice at fibre index v over a base entity."""
    fibre: tuple
    base_rank: int
    source = None
    target = None

    def apply(self, index):
        if len(index) != self.base_rank:
            raise ValueError("Embed needs an index of its base lattice")
        return index + self.fibre


@dataclass(frozen=True, slots=True)
class Unpin:
    """The inverse of ``Embed(v)``, defined on the fibre v."""
    fibre: tuple
    base_rank: int
    source = None
    target = None

    def apply(self, index):
        if index[self.base_rank:] != self.fibre:
            return None
        return index[:self.base_rank]


@dataclass(frozen=True, slots=True)
class Opaque:
    """A map of another provenance (IR8), compared by its key and never rewritten."""
    provenance: str
    key: object
    source: Entity | None
    target: Entity | None
    inverted: bool = False

    def apply(self, index):
        raise TypeError("a %s map is evaluated by its inspector or at run time, not here"
                        % self.provenance)


_KIND_PRESERVING = (Drop, Pin, Unpin)


# ---------------------------------------------------------------------------------------------
# Normalization.
# ---------------------------------------------------------------------------------------------
def _kind_after(atom, kind):
    """The entity kind after ``atom`` reads ``kind`` (None: not known)."""
    if atom.source is not None and kind is not None and atom.source != kind:
        raise TypeError("cannot compose: %r reads %r entities, but the map before it gives %r"
                        % (atom, atom.source, kind))
    if isinstance(atom, _KIND_PRESERVING) and kind is not None and kind.axis is not None \
            and kind.axis >= atom.base_rank:
        raise TypeError("%r reads a face normal to axis %d, which is not an axis of its base"
                        % (atom, kind.axis))
    return atom.target if atom.target is not None else kind


def _check_chain(atoms):
    kind = None
    for atom in reversed(atoms):
        kind = _kind_after(atom, kind)


def _translate(source, target, offset):
    # A translation that keeps the kind of unrestricted entities acts on any kind: it carries
    # none, so Shift(q)∘Parent(r) and Parent(r)∘Shift(r q) reach the same atoms. Faces of a
    # boundary piece keep their kind, and stay on their piece only if the translation has no
    # component along their normal.
    if source is not None and source == target:
        if source.piece is None:
            return Translate(None, None, offset)
        if any(axis == source.axis for axis, _ in offset):
            raise TypeError("a shift along axis %d moves the faces of %r off their boundary "
                            "piece" % (source.axis, source))
    return Translate(source, target, offset)


def _merge_translations(outer, inner):
    if inner.source is None:            # a shift without a kind acts on what the outer one reads
        _stays_on_piece(inner, outer.source)
        return _translate(outer.source, outer.target, _add(inner.offset, outer.offset))
    if outer.source is None:
        _stays_on_piece(outer, inner.target)
        return _translate(inner.source, inner.target, _add(inner.offset, outer.offset))
    if inner.target != outer.source:
        raise TypeError("cannot compose: %r reads %r entities, but the map before it gives %r"
                        % (outer, outer.source, inner.target))
    return _translate(inner.source, outer.target, _add(inner.offset, outer.offset))


def _stays_on_piece(shift, kind):
    if kind is not None and kind.piece is not None \
            and any(axis == kind.axis for axis, _ in shift.offset):
        raise TypeError("a shift along axis %d moves the faces of %r off their boundary piece"
                        % (kind.axis, kind))


def _is_identity(atom):
    return (type(atom) is Translate and not atom.offset
            and (atom.source == atom.target or atom.source is None or atom.target is None))


def _cell_shift(atom):
    """True when ``atom`` is a translation between cells, or one without a kind of its own."""
    return (type(atom) is Translate and atom.source in (None, cell)
            and atom.target in (None, cell))


def _rewrite(outer, inner):
    """The atoms that replace the pair (outer, inner), or None when no rule applies."""
    if type(outer) is Translate and type(inner) is Translate:
        merged = _merge_translations(outer, inner)
        return () if _is_identity(merged) else (merged,)
    if type(outer) is Refine and type(inner) is Refine:
        return (Refine(inner.ratio*outer.ratio,
                       _add(_scale(inner.offset, outer.ratio), outer.offset)),)
    if type(outer) is Coarsen and type(inner) is Coarsen:
        return (Coarsen(outer.ratio*inner.ratio),)
    if type(outer) is Coarsen and type(inner) is Refine:
        # floor((g a i + c)/(g b)) = floor((a i + floor(c/g))/b), g = gcd of the two ratios.
        common = math.gcd(inner.ratio, outer.ratio)
        if common == 1:
            return None
        quotient, _ = _divmod(inner.offset, common)
        fine, coarse = inner.ratio // common, outer.ratio // common
        refine = (Refine(fine, quotient),) if fine > 1 else (
            (Translate(None, None, quotient),) if quotient else ())
        return ((Coarsen(coarse),) if coarse > 1 else ()) + refine
    if type(outer) is Unrefine and type(inner) is Refine and \
            (outer.ratio, outer.offset) == (inner.ratio, inner.offset):
        return ()
    if type(outer) is Translate and type(inner) is Refine:
        if not _cell_shift(outer) or not outer.offset:
            return None
        quotient, remainder = _divmod(_add(inner.offset, outer.offset), inner.ratio)
        shift = (Translate(None, None, quotient),) if quotient else ()
        return (Refine(inner.ratio, remainder),) + shift
    if type(outer) is Translate and type(inner) is Coarsen:
        # floor(i/r) + q = floor((i + r q)/r): a translation moves inward through Parent.
        if not outer.offset or not _cell_shift(outer):
            return None
        return (inner, Translate(None, None, _scale(outer.offset, inner.ratio)))
    if type(outer) is Drop and type(inner) is Translate:
        _kind_after(outer, inner.target)
        if inner.source is not None and inner.source.axis is not None \
                and inner.source.axis >= outer.base_rank:
            return None     # read from faces normal to a fibre axis: no base entity to move to
        base, _ = _split(inner.offset, outer.base_rank)
        moved = _translate(inner.source, inner.target, base)
        return (() if _is_identity(moved) else (moved,)) + (outer,)
    if type(outer) is Drop and type(inner) is Pin and outer.base_rank == inner.base_rank:
        return ()
    if type(outer) is Unpin and type(inner) is Pin and \
            (outer.fibre, outer.base_rank) == (inner.fibre, inner.base_rank):
        return ()
    if type(outer) is Translate and type(inner) is Pin:
        if not outer.offset:
            return None
        _kind_after(inner, outer.source)
        base, fibre = _split(outer.offset, inner.base_rank)
        if any(axis >= len(inner.fibre) for axis, _ in fibre):
            raise TypeError("a translation of the product reaches beyond the fibre of %r" % (inner,))
        moved = _translate(outer.source, outer.target, base)
        pinned = Pin(tuple(v + x for v, x in zip(inner.fibre, dense(fibre, len(inner.fibre)),
                                                 strict=True)), inner.base_rank)
        return (pinned,) + (() if _is_identity(moved) else (moved,))
    return None


def normalize(atoms):
    """The normal form of a sequence of atoms, outermost first."""
    atoms = tuple(atoms)
    _check_chain(atoms)
    previous = None
    while atoms != previous:
        previous = atoms
        atoms = tuple(atom for atom in atoms if not _is_identity(atom))
        for index in range(len(atoms) - 1):
            replacement = _rewrite(atoms[index], atoms[index + 1])
            if replacement is not None:
                atoms = atoms[:index] + tuple(replacement) + atoms[index + 2:]
                break
    _check_chain(atoms)
    return atoms


# ---------------------------------------------------------------------------------------------
# The constructors of the closed list.
# ---------------------------------------------------------------------------------------------
class IndexMap:
    """A typed map of IR9, or a map of another provenance (IR8); compared by its normal form."""

    __slots__ = ("_normal",)
    provenance = "typed"

    def _atoms(self):
        raise NotImplementedError

    def normal_form(self):
        """The tuple of atoms, outermost first, that every equal map normalizes to."""
        try:
            return self._normal
        except AttributeError:
            object.__setattr__(self, "_normal", normalize(self._atoms()))
            return self._normal

    @property
    def source(self):
        """The entity kind the map reads, or None when its normal form does not fix one."""
        for atom in reversed(self.normal_form()):
            if atom.source is not None:
                return atom.source
        return None

    @property
    def target(self):
        """The entity kind the map gives, or None when its normal form does not fix one."""
        for atom in self.normal_form():
            if atom.target is not None:
                return atom.target
        return None

    def translation(self):
        """The sparse offset of a map whose normal form is one translation, or None."""
        form = self.normal_form()
        if not form:
            return ()
        if len(form) == 1 and type(form[0]) is Translate:
            return form[0].offset
        return None

    def offset(self, rank):
        """The dense offset of rank ``rank`` of a translation map (IR14 reads it)."""
        sparse = self.translation()
        if sparse is None:
            raise TypeError("%r is not a translation: its normal form is %r"
                            % (self, self.normal_form()))
        return dense(sparse, rank)

    def __call__(self, index):
        """The index this map sends ``index`` to, or None outside the domain of a partial map."""
        if type(index) is not tuple or any(type(i) is not int for i in index):
            raise TypeError("an index is a tuple of integers")
        for atom in reversed(self.normal_form()):
            index = atom.apply(index)
            if index is None:
                return None
        return index

    def __eq__(self, other):
        if not isinstance(other, IndexMap):
            return NotImplemented
        return self.normal_form() == other.normal_form()

    def __hash__(self):
        return hash(self.normal_form())

    def __setattr__(self, name, value):
        raise AttributeError("%s is immutable" % type(self).__name__)


def _init(map_, **fields):
    for name, value in fields.items():
        object.__setattr__(map_, name, value)
    map_.normal_form()


class Shift(IndexMap):
    """``Shift(o)``: translation of a lattice by the integer vector o, on any entity kind."""

    __slots__ = ("vector",)

    def __init__(self, vector):
        _sparse(vector)
        _init(self, vector=vector)

    def _atoms(self):
        return (Translate(None, None, _sparse(self.vector)),)

    def __repr__(self):
        return "Shift(%r)" % (self.vector,)


class Incidence(IndexMap):
    """``Incidence(face(a), cell, side)``: the left or the right cell of a face normal to a."""

    __slots__ = ("src", "dst", "side")

    def __init__(self, src, dst, side):
        if type(src) is not Entity or src.kind != "face" or src.piece is not None:
            raise TypeError("Incidence reads the faces normal to one axis, face(a), got %r"
                            % (src,))
        if dst != cell:
            raise TypeError("Incidence of faces gives cells, got %r" % (dst,))
        if side not in _SIDES:
            raise ValueError("the side of a face incidence is 'left' or 'right', got %r"
                             % (side,))
        _init(self, src=src, dst=dst, side=side)

    def _atoms(self):
        offset = ((self.src.axis, -1),) if self.side == "left" else ()
        return (Translate(self.src, self.dst, offset),)

    def __repr__(self):
        return "Incidence(%r, %r, side=%r)" % (self.src, self.dst, self.side)


class Trace(IndexMap):
    """``Trace(piece, depth)``: the cell at inward depth o >= 0 behind a face of a boundary piece.

    Depth 0 is the cell adjacent to the face: the right cell of a face on the low piece, the left
    cell of a face on the high piece. The outward normal is -e_a on the low piece and +e_a on the
    high piece (``normal``).
    """

    __slots__ = ("piece", "depth")

    def __init__(self, piece, depth=0):
        if type(piece) is not Entity or piece.kind != "face" or piece.piece is None:
            raise TypeError("Trace reads the faces of a boundary piece, boundary(a, side), got %r"
                            % (piece,))
        if type(depth) is not int or depth < 0:
            raise ValueError("the inward depth of a trace is an integer >= 0, got %r" % (depth,))
        _init(self, piece=piece, depth=depth)

    @property
    def normal(self):
        """The outward normal, as a sparse offset."""
        return ((self.piece.axis, -1 if self.piece.piece == "low" else 1),)

    def _atoms(self):
        steps = self.depth if self.piece.piece == "low" else -(1 + self.depth)
        return (Translate(self.piece, cell, ((self.piece.axis, steps),) if steps else ()),)

    def __repr__(self):
        return "Trace(%r, depth=%d)" % (self.piece, self.depth)


def _ratio(ratio):
    if type(ratio) is not int or ratio < 2:
        raise ValueError("a refinement ratio is an integer >= 2, got %r" % (ratio,))
    return ratio


class Parent(IndexMap):
    """``Parent(r)``: from level l + 1 to level l, the cell that holds each fine cell."""

    __slots__ = ("ratio",)

    def __init__(self, ratio):
        _init(self, ratio=_ratio(ratio))

    def _atoms(self):
        return (Coarsen(self.ratio),)

    def __repr__(self):
        return "Parent(ratio=%d)" % self.ratio


class Child(IndexMap):
    """``Child(r, o)``: from level l to level l + 1, the child at offset o in [0, r)^rank."""

    __slots__ = ("ratio", "child_offset")

    def __init__(self, ratio, offset):
        ratio = _ratio(ratio)
        _sparse(offset)
        if any(not 0 <= x < ratio for x in offset):
            raise ValueError("a child offset lies in [0, %d) on every axis, got %r"
                             % (ratio, offset))
        _init(self, ratio=ratio, child_offset=offset)

    def _atoms(self):
        return (Refine(self.ratio, _sparse(self.child_offset)),)

    def __repr__(self):
        return "Child(ratio=%d, offset=%r)" % (self.ratio, self.child_offset)


def _base_rank(base_rank):
    if type(base_rank) is not int or base_rank < 1:
        raise ValueError("the base rank of a product lattice is an integer >= 1, got %r"
                         % (base_rank,))
    return base_rank


class Project(IndexMap):
    """``Project``: from ``ProductSet(X, V)`` to X, X being the first ``base_rank`` axes (IR3)."""

    __slots__ = ("base_rank",)

    def __init__(self, base_rank):
        _init(self, base_rank=_base_rank(base_rank))

    def _atoms(self):
        return (Drop(self.base_rank),)

    def __repr__(self):
        return "Project(base_rank=%d)" % self.base_rank


class Embed(IndexMap):
    """``Embed(v)``: from X to ``ProductSet(X, V)`` at the fibre index v."""

    __slots__ = ("fibre", "base_rank")

    def __init__(self, fibre, base_rank):
        if type(fibre) is not tuple or not fibre or any(type(x) is not int for x in fibre):
            raise TypeError("a fibre index is a nonempty tuple of integers, got %r" % (fibre,))
        _init(self, fibre=fibre, base_rank=_base_rank(base_rank))

    def _atoms(self):
        return (Pin(self.fibre, self.base_rank),)

    def __repr__(self):
        return "Embed(%r, base_rank=%d)" % (self.fibre, self.base_rank)


class Compose(IndexMap):
    """``Compose(m2, m1)``: m1 first, then m2."""

    __slots__ = ("outer", "inner")

    def __init__(self, outer, inner):
        for part in (outer, inner):
            if not isinstance(part, IndexMap):
                raise TypeError("Compose takes two index maps, got %r" % (part,))
        _init(self, outer=outer, inner=inner)

    def _atoms(self):
        return self.outer.normal_form() + self.inner.normal_form()

    def __repr__(self):
        return "Compose(%r, %r)" % (self.outer, self.inner)


def _inverse_atom(atom):
    if type(atom) is Translate:
        return Translate(atom.target, atom.source, _neg(atom.offset))
    if type(atom) is Refine:
        return Unrefine(atom.ratio, atom.offset)
    if type(atom) is Unrefine:
        return Refine(atom.ratio, atom.offset)
    if type(atom) is Pin:
        return Unpin(atom.fibre, atom.base_rank)
    if type(atom) is Unpin:
        return Pin(atom.fibre, atom.base_rank)
    if type(atom) is Opaque:
        return Opaque(atom.provenance, atom.key, atom.target, atom.source, not atom.inverted)
    return None


def _fan_out(atom):
    if type(atom) is Coarsen:
        return ("ratio^rank", atom.ratio)
    return ("fibre cardinality", atom.base_rank)


class Adjacency:
    """The inverse of a map that is not injective (IR9): each entity goes to several entities.

    ``fan_out`` names where the count comes from: ``("ratio^rank", r)`` for ``Parent(r)``, and the
    fibre cardinality |V| of the descriptor for ``Project``.
    """

    __slots__ = ("inverse_of", "fan_out")
    provenance = "typed"

    def __init__(self, inverse_of, fan_out):
        object.__setattr__(self, "inverse_of", inverse_of)
        object.__setattr__(self, "fan_out", fan_out)

    def __eq__(self, other):
        return type(other) is Adjacency and other.inverse_of == self.inverse_of

    def __hash__(self):
        return hash(("adjacency", self.inverse_of))

    def __setattr__(self, name, value):
        raise AttributeError("Adjacency is immutable")

    def __repr__(self):
        return "Inverse(%r)" % (self.inverse_of,)


class Inverse(IndexMap):
    """``Inverse(m)``: the inverse of an injective map, partial on the image of ``Child``/``Embed``.

    The inverse of a map whose normal form holds ``Parent`` or ``Project`` is an ``Adjacency``.
    """

    __slots__ = ("of",)

    def __new__(cls, of):
        if not isinstance(of, IndexMap):
            raise TypeError("Inverse takes an index map, got %r" % (of,))
        blocking = [atom for atom in of.normal_form() if _inverse_atom(atom) is None]
        if blocking:
            return Adjacency(of, _fan_out(blocking[0]))
        return super().__new__(cls)

    def __init__(self, of):
        _init(self, of=of)

    def _atoms(self):
        return tuple(_inverse_atom(atom) for atom in reversed(self.of.normal_form()))

    def __repr__(self):
        return "Inverse(%r)" % (self.of,)


class _OpaqueMap(IndexMap):
    __slots__ = ("key", "src", "dst")

    def __init__(self, key, *, source, target):
        for kind in (source, target):
            if type(kind) is not Entity:
                raise TypeError("a %s map names the entity kinds it reads and gives"
                                % self.provenance)
        hash(key)
        _init(self, key=key, src=source, dst=target)

    def _atoms(self):
        return (Opaque(self.provenance, self.key, self.src, self.dst),)

    def __repr__(self):
        return "%s(%r, source=%r, target=%r)" % (type(self).__name__, self.key, self.src, self.dst)


class ArithmeticMap(_OpaqueMap):
    """An ``imap`` body over coordinates (IR8 item 2); IR10 MAY raise it to a normal form."""
    __slots__ = ()
    provenance = "arithmetic"


class DataMap(_OpaqueMap):
    """Bound index arrays (IR8 item 3), inspected per topology instance (IR11)."""
    __slots__ = ()
    provenance = "data"


class ComputedMap(_OpaqueMap):
    """A map built from the state during the run (IR8 item 4, IR12)."""
    __slots__ = ()
    provenance = "computed"


def exact_weight(weight):
    """``weight`` as an exact rational, or None when it is not an exact number (IR17)."""
    if type(weight) in (int, Fraction):
        return Fraction(weight)
    return None


__all__ = [
    "Adjacency", "ArithmeticMap", "Child", "Coarsen", "Compose", "ComputedMap", "DataMap", "Drop",
    "Embed", "Entity", "Incidence", "IndexMap", "Inverse", "Opaque", "Parent", "Pin", "Project",
    "Refine", "Shift", "Trace", "Translate", "Unpin", "Unrefine", "boundary", "cell", "dense",
    "exact_weight", "face", "normalize",
]
