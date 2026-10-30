"""Pilot ``FaceFlux`` builder (spec 0.3 IR25, issue N7a), realized by ``StencilFlux``.

A scheme is a Python function of one face. It reads the state and the solved
fields only through the face vocabulary of spec 0.3 sec. 4.3 (``face.left(q)``,
``face.right(q)``, ``face.left_cell``, ``cell.shift(axis, k)``, ``cell.value(q)``,
``face.h(axis)``) and returns a ``FaceFlux`` of normal flux densities keyed by
component name. The author writes no stencil radius, no halo depth and no
dimension (CF8, B2, B3): every read is recorded as a typed map
``Compose(Shift(o), Incidence(face(a), cell, side))``, and the builder derives

* the halo of each input: the exact widths of IR14 by box arithmetic, and the
  b6 bound that its face formula stores, which IR-T1 compares,
* the face identities of IR17: orientation and cell-face incidence come from
  the incidence maps, the signs from the D6 delivery that the builder fixes,
* the lowering class, face incidence, which classification of the normal
  forms of the reads reaches (IR20, IR21).

The realization is the b6 native face loop, a qualified substitution (IR24):
the builder lowers to one ``StencilFlux`` whose radius is derived from the reads.

The typed maps are the core constructors of IR9, from ``pops.regions``, and the
facts come from the core analyses of the regions that the realization builds
(``StencilFlux.regions``). Pilot limits, each owned by a later issue: ``FaceFlux``
carries the b6 incident ``frequency`` where spec 0.3 has a ``reduce(max)`` of
the step control (sec. 6.9, N8), and no ``covers=``, since M05 and M10 declare no
balance terms at b6 (C31). The frame stands for the configuration kappa. Nothing
here changes the core (CF0).
"""
from dataclasses import dataclass

from pops.math import Var
from pops.regions import Compose, Incidence, Shift, cell, face


@dataclass(frozen=True, slots=True)
class _Read:
    target: tuple      # ("state", component index) or ("field", field slot)
    map: Compose       # Compose(Shift(o), Incidence(face(a), cell, side))

    def offset(self, dimension):
        return self.map.offset(dimension)


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
    __slots__ = ("_face", "_side", "_offset", "_map")

    def __init__(self, face_, side, offset):
        self._face = face_
        self._side = side
        self._offset = offset
        self._map = Compose(Shift(offset), Incidence(face(face_.axis), cell, side))

    def shift(self, axis, k=1):
        _require_axis(axis, self._face._dimension)
        if type(k) is not int:
            raise TypeError("cell.shift needs an integer constant offset (StructuredOffsetNotConstant)")
        offset = tuple(o + k*int(d == axis) for d, o in enumerate(self._offset))
        return _Cell(self._face, self._side, offset)

    def value(self, quantity):
        return self._face._reader(quantity, self._map)

    def h(self, axis):
        _require_axis(axis, self._face._dimension)
        return self._face._spacing[axis]

    # Two paths to one cell are one entity: typed maps compare by their normal forms (IR9).
    def __eq__(self, other):
        return type(other) is _Cell and other._face is self._face and other._map == self._map

    def __hash__(self):
        return hash(self._map)


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
        # A solved field of another model falls through to the refusal below.
        if (getattr(quantity, "kind", None) == "field"
                and getattr(quantity, "owner_path", None) == self.state.owner_path):
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
    # The +1 is exact on the face's normal axis only: the native loop adds one
    # face along that axis (nd::face_box), so a tangent axis b needs max(o_b)
    # on its high side. b6 applies the +1 to every axis, and IR-T1 compares that
    # bound. b6 stores one width per axis, the larger side.
    low = tuple(max(0, max((-o[d] for o in offsets), default=0)) for d in range(dimension))
    high = tuple(max(0, max((o[d]+1 for o in offsets), default=0)) for d in range(dimension))
    return low, high, tuple(max(a, b) for a, b in zip(low, high))


class RealizedFaceFlux:
    """A face scheme realized over a state: its derived facts and its native substitution."""

    def __init__(self, *, state, dimension, names, fields, reads, stencil_flux):
        self.state = state
        self.dimension = dimension
        self.components = names
        # In first-read order: the program binds the solved fields by position.
        self.fields = tuple(fields)
        self.reads = reads                      # axis -> tuple of _Read, in read order
        self.stencil_flux = stencil_flux

    def _offsets(self, kind, slot=None):
        return tuple(read.offset(self.dimension)
                     for axis_reads in self.reads.values() for read in axis_reads
                     if read.target[0] == kind and (slot is None or read.target[1] == slot))

    def _slot(self, field):
        for slot, candidate in enumerate(self.fields):
            if candidate.qualified_id == field.qualified_id:
                return slot
        raise KeyError("the scheme does not read the field %r" % (field.name,))

    def halo(self, field=None):
        """``(low, high)`` b6 widths per axis, of the state or of one solved field (IR14).

        These are the bounds of the b6 face formula, which IR-T1 compares. On a
        tangent axis the high side carries the +1 of the normal axis, so it can
        exceed what the reads need: M05 reads one cell on each side, and the
        bound is ``((1, 1), (2, 2))`` where ``exact_halo`` is ``((1, 1), (1, 1))``.
        """
        offsets = self._offsets("state") if field is None else self._offsets("field", self._slot(field))
        low, high, _ = _halo(offsets, self.dimension)
        return low, high

    def regions(self):
        """The face region of each axis, built by the realization (``StencilFlux.regions``)."""
        return self.stencil_flux.regions("face flux of %s" % self.state.name)

    def exact_halo(self, field=None):
        """``(low, high)`` widths per axis by the box arithmetic of IR14, over every axis.

        The faces of the owned cells along a are one more than the cells, so a
        read at offset o needs o_a + 1 cells above along a and o_b along a
        tangent axis b. The analysis is the core's, on the normal forms of the
        reads; it answers for the state, or for one solved field.
        """
        if field is None:
            def where(array):
                return array[0] == "state"
        else:
            slot = self._slot(field)

            def where(array):
                return array[:2] == ("field", slot)
        low, high = [0]*self.dimension, [0]*self.dimension
        for region in self.regions():
            fact = region.halo(where)
            for axis in range(self.dimension):
                low[axis] = max(low[axis], fact.value[0][axis])
                high[axis] = max(high[axis], fact.value[1][axis])
        return tuple(low), tuple(high)

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

        The normal and the cell offsets follow from the incidence maps of a
        face (IR9). The deliveries are the builder's own (D6, IR25), whatever
        the scheme writes: the face value J goes to ``Incidence(face(a), cell,
        side="left")`` with weight -1 and to ``Incidence(face(a), cell,
        side="right")`` with weight +1, and the core analysis proves
        ``conservation`` (sum_k w_k = 0) on the region that the realization
        builds. The executed evidence of conservation is the b6 ``mass_defect``
        invariant of each case. ``reads`` lists the distinct cells that the
        scheme reads on that axis, as offsets from the right cell (the normal
        forms of IR9).
        """
        out = []
        for region in self.regions():
            axis = region.domain.axis
            (left, wl), (right, wr) = region.targets
            lo, ro = left.offset(self.dimension), right.offset(self.dimension)
            fact = region.conservation()
            out.append({
                "axis": axis,
                "normal": tuple(r - l for r, l in zip(ro, lo)),
                "left_offset": lo,
                "right_offset": ro,
                "targets": ((left, wl), (right, wr)),
                "conservation": fact.truth.value == "proved",
                "components": self.components,
                "reads": tuple(sorted({read.offset(self.dimension) for read in self.reads[axis]})),
            })
        return tuple(out)

    def lowering_class(self):
        """The class the builder expects, face incidence, and the class reached (IR20, IR21).

        The core classifies each region from the normal forms of its maps and
        raises ``ClassNotReached`` when a region does not reach face incidence.
        The face vocabulary only reads a cell of the face,
        ``Incidence(face(a), cell, side)`` composed with an integer shift, so the
        report lists the composed shifts that the scheme reads.
        """
        reached = {region.check_expected_class() for region in self.regions()}
        shifts = sorted({read.map.outer.vector for axis_reads in self.reads.values()
                         for read in axis_reads if any(read.map.outer.vector)})
        return {"expected": "face_incidence",
                "reached": reached.pop() if len(reached) == 1 else tuple(sorted(reached)),
                "reached_by": "classification of the normal forms of the reads (IR20)",
                "composed_shifts": tuple(shifts),
                "substitution": "StencilFlux (include/pops/numerics/fv/stencil_flux.hpp@b6, IR24)"}


def _read_set(reads, dimension):
    return {(read.offset(dimension), *read.target)
            for axis_reads in reads.values() for read in axis_reads}


def realize_face_flux(scheme, state, *, frame):
    """Realize ``scheme`` on every face of ``frame`` (``Realize(..., faces=(...))`` of D5).

    ``frame`` is the frame of the model that declares ``state``; at b6 a state
    handle names its frame only by identity, so the frame stands for the
    configuration kappa and gives the axes. The scheme runs twice per axis, as the
    b6 body does: once to record the typed maps, then once inside the native
    ``StencilFlux``, whose radius is the largest recorded offset. Both passes
    must read the same cells; the rows come from the second pass.
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
        raise ValueError("the scheme read other cells when it was realized: both passes of a "
                         "face scheme must read the same cells")
    return RealizedFaceFlux(state=state, dimension=dimension, names=names, fields=fields,
                            reads=reads, stencil_flux=native)


__all__ = ["FaceFlux", "RealizedFaceFlux", "realize_face_flux"]
