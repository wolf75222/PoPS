"""Pilot ``FaceFlux`` builder (spec 0.3 IR25, issue N7a), realized by ``StencilFlux``.

A scheme is a Python function of one face. It reads the state and the solved
fields only through the face vocabulary of spec 0.3 sec. 4.3 (``face.left(q)``,
``face.right(q)``, ``face.left_cell``, ``cell.shift(axis, k)``, ``cell.value(q)``,
``face.h(axis)``) and returns a ``FaceFlux`` of normal flux densities keyed by
component name. The author writes no stencil radius, no halo depth and no
dimension (CF8, B2, B3): every read is recorded as a typed map
``Compose(Shift(o), Incidence(face(a), cell, side))``, and the builder derives

* the halo of each input with the b6 face formulas (IR14),
* the face identities: orientation, signs and cell-face incidence (IR17),
* the lowering class that the builder expects (IR20, IR21).

The realization is the b6 native face loop, a qualified substitution (IR24):
the builder lowers to one ``StencilFlux`` whose radius is derived from the reads.

Pilot limits, each owned by a later issue. The typed maps are records kept in
``pops.lib``; N7 replaces them with the core constructors of IR9. ``FaceFlux``
carries the b6 incident ``frequency`` where spec 0.3 has a ``reduce(max)`` of
the step control (sec. 6.9, N8), and no ``covers=``, since M05 and M10 declare no
balance terms at b6 (C31). The frame stands for the configuration kappa. Nothing
here changes the core (CF0).
"""
from dataclasses import dataclass

from pops.math import Var


@dataclass(frozen=True, slots=True)
class Shift:
    """``Shift(o)``: translation of a cell lattice by the integer vector o (IR9)."""
    offset: tuple


@dataclass(frozen=True, slots=True)
class Incidence:
    """``Incidence(face(a), cell, side)``: the left or right cell of a face normal to a."""
    axis: int
    side: str

    def offset(self, dimension):
        # The face of index i along a lies between cell i - e_a (left) and
        # cell i (right); offsets count from the right cell, as at b6.
        return tuple(-int(d == self.axis and self.side == "left") for d in range(dimension))


@dataclass(frozen=True, slots=True)
class Compose:
    """``Compose(Shift(o), Incidence(...))``: a cell reached from a face (IR9)."""
    shift: Shift
    incidence: Incidence

    def offset(self, dimension):
        base = self.incidence.offset(dimension)
        return tuple(b + o for b, o in zip(base, self.shift.offset, strict=True))


@dataclass(frozen=True, slots=True)
class _Read:
    target: tuple      # ("state", component index) or ("field", field slot)
    map: Compose


class FaceFlux:
    """What a scheme returns: the shared face value of each component (D6).

    ``value`` maps each state component name to its normal flux density at the
    face; the realization applies the face and cell measures. ``frequency`` is
    the b6 incident frequency of the face (1/time).
    """
    __slots__ = ("value", "frequency")

    def __init__(self, value, *, frequency):
        if not isinstance(value, dict) or any(type(k) is not str for k in value):
            raise TypeError("FaceFlux value maps each state component name to its face density")
        object.__setattr__(self, "value", dict(value))
        object.__setattr__(self, "frequency", frequency)

    def __setattr__(self, name, value):
        raise AttributeError("FaceFlux is immutable")


def _require_axis(axis, dimension):
    if type(axis) is not int or not 0 <= axis < dimension:
        raise ValueError("axis must be an integer of the frame, 0 <= axis < %d" % dimension)


class _Cell:
    __slots__ = ("_face", "_map")

    def __init__(self, face, side, offset):
        self._face = face
        self._map = Compose(Shift(offset), Incidence(face.axis, side))

    def shift(self, axis, k=1):
        _require_axis(axis, self._face._dimension)
        if type(k) is not int:
            raise TypeError("cell.shift needs an integer constant offset (StructuredOffsetNotConstant)")
        offset = tuple(o + k*int(d == axis) for d, o in enumerate(self._map.shift.offset))
        return _Cell(self._face, self._map.incidence.side, offset)

    def value(self, quantity):
        return self._face._reader(quantity, self._map)

    def h(self, axis):
        _require_axis(axis, self._face._dimension)
        return self._face._spacing[axis]

    width = h

    # Two paths to one cell are one entity: compare the normal forms (IR9).
    def _key(self):
        return self._face.axis, self._map.offset(self._face._dimension)

    def __eq__(self, other):
        return type(other) is _Cell and other._face is self._face and other._key() == self._key()

    def __hash__(self):
        return hash(self._key())


class _Face:
    __slots__ = ("axis", "_dimension", "_spacing", "_reader", "left_cell", "right_cell")

    def __init__(self, axis, dimension, spacing, reader):
        self.axis = axis
        self._dimension = dimension
        self._spacing = spacing
        self._reader = reader
        zero = (0,)*dimension
        self.left_cell = _Cell(self, "left", zero)
        self.right_cell = _Cell(self, "right", zero)

    def left(self, quantity):
        return self.left_cell.value(quantity)

    def right(self, quantity):
        return self.right_cell.value(quantity)

    def average(self, quantity):
        return (self.left(quantity)+self.right(quantity))/2

    def normal_gradient(self, quantity):
        return (self.right(quantity)-self.left(quantity))/self.distance

    def h(self, axis):
        _require_axis(axis, self._dimension)
        return self._spacing[axis]

    @property
    def distance(self):
        return self._spacing[self.axis]


class _Targets:
    """Resolve what a scheme reads into state components and field slots."""

    def __init__(self, state):
        self.state = state
        self.names = tuple(state.components)
        self.fields = []

    def resolve(self, quantity):
        if getattr(quantity, "kind", None) == "field":
            for slot, field in enumerate(self.fields):
                if field.qualified_id == quantity.qualified_id:
                    return ("field", slot)
            self.fields.append(quantity)
            return ("field", len(self.fields)-1)
        handle = getattr(quantity, "handle", None)
        if (getattr(handle, "kind", None) != "state"
                or getattr(handle, "qualified_id", None) != self.state.qualified_id
                or type(getattr(quantity, "index", None)) is not int):
            raise TypeError("a face scheme reads components of its own state or solved fields, "
                            "got %r" % (quantity,))
        return ("state", quantity.index)


def _call_scheme(scheme, face, names):
    result = scheme(face)
    if type(result) is not FaceFlux:
        raise TypeError("a face scheme returns FaceFlux(value={...}, frequency=...)")
    if set(result.value) != set(names):
        raise ValueError("FaceFlux must give exactly one face density per state component "
                         "%s, got %s (CoverageNotExact)" % (names, tuple(result.value)))
    if result.frequency is None:
        raise ValueError("the StencilFlux realization needs the face frequency")
    return result


def _halo(offsets, dimension):
    # IR14 for reads relative to a face, offsets counted from the right cell:
    # the low side of axis a needs max(-o_a) cells and the high side
    # max(o_a + 1), never below 0 (python/pops/numerics/stencil_flux.py:96-98@b6).
    # b6 stores one width per axis, the larger side.
    low = tuple(max(0, max((-o[d] for o in offsets), default=0)) for d in range(dimension))
    high = tuple(max(0, max((o[d]+1 for o in offsets), default=0)) for d in range(dimension))
    return low, high, tuple(max(a, b) for a, b in zip(low, high))


class RealizedFaceFlux:
    """A face scheme realized over a state: its derived facts and its native substitution."""

    def __init__(self, *, state, dimension, names, fields, reads, stencil_flux):
        self.state = state
        self.dimension = dimension
        self.components = names
        self.fields = tuple(fields)
        self.reads = reads                      # axis -> tuple of _Read, in read order
        self.stencil_flux = stencil_flux

    def _offsets(self, kind, slot=None):
        return tuple(read.map.offset(self.dimension)
                     for axis_reads in self.reads.values() for read in axis_reads
                     if read.target[0] == kind and (slot is None or read.target[1] == slot))

    def _slot(self, field):
        for slot, candidate in enumerate(self.fields):
            if candidate.qualified_id == field.qualified_id:
                return slot
        raise KeyError("the scheme does not read the field %r" % (field.name,))

    def halo(self, field=None):
        """``(low, high)`` widths per axis, of the state or of one solved field (IR14)."""
        offsets = self._offsets("state") if field is None else self._offsets("field", self._slot(field))
        low, high, _ = _halo(offsets, self.dimension)
        return low, high

    @property
    def ghost_extent(self):
        return _halo(self._offsets("state"), self.dimension)[2]

    @property
    def field_ghost_extents(self):
        return tuple(_halo(self._offsets("field", slot), self.dimension)[2]
                     for slot in range(len(self.fields)))

    @property
    def ghost_depth(self):
        """The one storage depth of the state: the widest axis (IR14)."""
        return max(self.ghost_extent)

    def identities(self):
        """The face identities of IR17, for each axis of the frame.

        They follow from the incidence maps of a face (IR9) and from the
        delivery of D6: the face value J goes to ``Incidence(face(a), cell,
        left)`` with weight -1 and to ``Incidence(face(a), cell, right)`` with
        weight +1; conservation is sum_k w_k = 0. The normal points from the
        left cell to the right cell. ``reads`` lists the distinct cells, as
        ``(side, shift)`` normal forms, that the scheme reads on that axis.
        """
        out = []
        for axis in range(self.dimension):
            left, right = Incidence(axis, "left"), Incidence(axis, "right")
            lo, ro = left.offset(self.dimension), right.offset(self.dimension)
            targets = ((left, -1), (right, +1))
            out.append({
                "axis": axis,
                "normal": tuple(r - l for r, l in zip(ro, lo)),
                "left_offset": lo,
                "right_offset": ro,
                "targets": targets,
                "conservation": sum(w for _, w in targets) == 0,
                "components": self.components,
                "reads": tuple(sorted({(read.map.incidence.side, read.map.shift.offset)
                                       for read in self.reads[axis]})),
            })
        return tuple(out)

    def lowering_class(self):
        """The class the builder expects and the one its regions reach (IR20, IR21).

        Every read is a cell of the face, ``Incidence(face(a), cell, side)``,
        composed with a constant shift (IR25): the regions reach face incidence.
        """
        for axis, axis_reads in self.reads.items():
            for read in axis_reads:
                if (type(read.map) is not Compose or read.map.incidence.axis != axis
                        or any(type(x) is not int for x in read.map.shift.offset)):
                    raise ValueError("ClassNotReached: the read %r of axis %d is not a cell of "
                                     "its face" % (read.map, axis))
        shifts = sorted({read.map.shift.offset for axis_reads in self.reads.values()
                         for read in axis_reads if any(read.map.shift.offset)})
        return {"expected": "face_incidence",
                "reached": "face_incidence",
                "composed_shifts": tuple(shifts),
                "substitution": "StencilFlux (include/pops/numerics/fv/stencil_flux.hpp@b6, IR24)"}


def _read_set(reads, dimension):
    return {(read.map.offset(dimension), *read.target)
            for axis_reads in reads.values() for read in axis_reads}


def realize_face_flux(scheme, state, *, frame):
    """Realize ``scheme`` on every face of ``frame`` (``Realize(..., faces=(...))`` of D5).

    ``frame`` is the frame of the model that declares ``state``; at b6 a state
    handle names its frame only by identity, so the frame stands for the
    configuration kappa and gives the axes. The scheme runs twice per axis, as the
    b6 body does: once to record the typed maps, then once inside the native
    ``StencilFlux``, whose radius is the largest recorded offset.
    """
    from pops.numerics.stencil_flux import StencilFlux
    if getattr(state, "kind", None) != "state":
        raise TypeError("realize_face_flux needs a state handle")
    if not callable(scheme):
        raise TypeError("a face scheme is a function of one face")
    if getattr(frame, "canonical_id", None) != state.space.frame:
        raise ValueError("the state %r is not declared on this frame" % (state.name,))
    dimension = frame.coordinates.dimension
    targets = _Targets(state)
    names = targets.names
    probe_spacing = tuple(Var("face_h_%d" % d, "face_spacing") for d in range(dimension))
    reads = {}
    for axis in range(dimension):
        recorded, symbols = [], {}

        def record(quantity, path, recorded=recorded, symbols=symbols):
            read = _Read(targets.resolve(quantity), path)
            if read not in symbols:
                recorded.append(read)
                symbols[read] = Var("face_read_%d" % len(symbols), "face_read")
            return symbols[read]

        _call_scheme(scheme, _Face(axis, dimension, probe_spacing, record), names)
        reads[axis] = tuple(recorded)
    fields = tuple(targets.fields)
    radius = max((abs(x) for offset, *_ in _read_set(reads, dimension) for x in offset), default=0)

    def body(sample, spacing, axis):
        def read(quantity, path):
            kind, slot = targets.resolve(quantity)
            offset = path.offset(dimension)
            # The solved fields of a b6 model (model.field) are scalar.
            return sample(offset, slot) if kind == "state" else sample(offset, 0, field=slot)
        result = _call_scheme(scheme, _Face(axis, dimension, spacing, read), names)
        return tuple(result.value[n] for n in names), result.frequency

    # StencilFlux admits radius >= 1; a scheme reading only the right cell has radius 0.
    native = StencilFlux(dimension=dimension, components=len(names), radius=max(radius, 1),
                         field_components=(1,)*len(fields), body=body)
    lowered = ({(offset, "state", c) for offset, c, _ in native.samples}
               | {(offset, "field", f) for f, offset, _, _ in native.field_samples})
    if len(targets.fields) != len(fields) or lowered != _read_set(reads, dimension):
        raise ValueError("the scheme read other cells when it was realized; a scheme must be pure")
    return RealizedFaceFlux(state=state, dimension=dimension, names=names, fields=fields,
                            reads=reads, stencil_flux=native)


__all__ = ["Compose", "FaceFlux", "Incidence", "RealizedFaceFlux", "Shift", "realize_face_flux"]
