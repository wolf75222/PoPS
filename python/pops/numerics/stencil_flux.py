"""Closed algebraic face stencils; no Python callback survives construction.

The body returns positive-coordinate flux density and one nonnegative incident
frequency (1/time). Its consistency and the majorant are scientific obligations
of the author. The engine checks the finite result, support, and declared reads.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, eq=False, init=False)
class StencilFlux:
    """``body(sample, spacing, axis) -> (flux_tuple, incident_frequency)``.

    ``sample(offset, component)`` reads relative to the right incident cell.
    ``sample(offset, component, field=i)`` reads the i-th typed solved observation
    or explicit ``Program.cell_field`` numerical SSA sample. A provisional sample
    retains its own evaluation point and is not an observation of the flux state.
    Offsets are integer tuples of the frame dimension, bounded by ``radius``.
    The callable executes once per axis in Python. Uniform periodic Cartesian
    meshes are the initial native realization; no AMR or wall closure is implied.
    """
    dimension: int
    components: int
    radius: int
    rows: tuple
    samples: tuple
    spacing: tuple
    field_components: tuple
    field_samples: tuple
    parameters: tuple
    parameter_symbols: tuple
    __pops_ir_immutable__ = True

    def __init__(self, *, dimension, components, radius, body, field_components=(), parameters=()):
        if type(dimension) is not int or not 1 <= dimension <= 3:
            raise ValueError("stencil flux dimension must be 1, 2, or 3")
        if type(components) is not int or components < 1:
            raise ValueError("stencil flux needs a positive component count")
        # Capacity follows native signed cell indices, not a catalogue of
        # admitted reconstruction orders. The upper face needs offset+1.
        if type(radius) is not int or not 1 <= radius < 2**31-1:
            raise ValueError("stencil radius and its upper-face halo must fit positive Int32 extents")
        if not callable(body):
            raise TypeError("stencil flux body must construct scalar expressions")
        if type(field_components) is not tuple or any(type(n) is not int or n < 1 for n in field_components):
            raise ValueError("field_components requires exact positive widths")
        if (type(parameters) is not tuple or any(type(n) is not str or not n.isidentifier() for n in parameters)
                or len(set(parameters)) != len(parameters)):
            raise ValueError("stencil parameters require distinct explicit names")
        from pops.model.stencil_expression import capture_stencil_body
        rows, samples, spacing, field_samples, scalars = capture_stencil_body(
            dimension=dimension, components=components, radius=radius,
            field_components=field_components, body=body, parameters=parameters)
        for name, value in (("dimension", dimension), ("components", components),
                            ("radius", radius), ("rows", tuple(rows)),
                            ("samples", samples), ("spacing", spacing),
                            ("field_components", field_components),
                            ("field_samples", field_samples), ("parameters", parameters),
                            ("parameter_symbols", scalars)):
            object.__setattr__(self, name, value)
        self.validate(exact=True)

    def validate(self, *, exact=False):
        from pops.model.stencil_expression import validate_closed_stencil
        if (type(self.dimension) is not int or not 1 <= self.dimension <= 3
                or type(self.components) is not int or self.components < 1
                or type(self.radius) is not int or not 1 <= self.radius < 2**31-1
                or len(self.rows) != self.dimension
                or len(self.spacing) != self.dimension
                or any(len(row) != self.components+1 for row in self.rows)):
            raise ValueError("stencil declaration rank/width/radius changed")
        if any(type(offset) is not tuple or len(offset) != self.dimension
               or any(type(x) is not int or abs(x) > self.radius for x in offset)
               or type(component) is not int or not 0 <= component < self.components
               for offset, component, symbol in self.samples):
            raise ValueError("stencil sample escaped its bounded state domain")
        if (type(self.field_components) is not tuple
                or any(type(n) is not int or n < 1 for n in self.field_components)
                or any(type(f) is not int or not 0 <= f < len(self.field_components)
                       or type(offset) is not tuple or len(offset) != self.dimension
                       or any(type(x) is not int or abs(x) > self.radius for x in offset)
                       or type(c) is not int or not 0 <= c < self.field_components[f]
                       for f, offset, c, symbol in self.field_samples)):
            raise ValueError("stencil field sample escaped its bounded domain")
        if (type(self.parameters) is not tuple or len(self.parameters) != len(self.parameter_symbols)
                or any(type(n) is not str or not n.isidentifier() for n in self.parameters)
                or len(set(self.parameters)) != len(self.parameters)):
            raise ValueError("stencil parameter declaration changed")
        validate_closed_stencil(self.rows, self.samples, self.field_samples,
                                self.spacing, exact=exact, scalars=self.parameter_symbols)
        return self

    @property
    def ghost_extent(self):
        # Right-relative positive offsets at the upper face need one extra cell.
        return tuple(max((max(-offset[d], offset[d] + 1)
                          for offset, _, _ in self.samples), default=0)
                     for d in range(self.dimension))

    @property
    def field_ghost_extents(self):
        return tuple(tuple(max((max(-offset[d], offset[d]+1)
                                for f, offset, _, _ in self.field_samples if f == index), default=0)
                           for d in range(self.dimension))
                     for index in range(len(self.field_components)))

    def regions(self, name):
        """The face region of each axis, with the reads of its row (IR14, IR17, IR20, IR21).

        A sample at offset o read on the faces normal to a goes through
        ``Compose(Shift(o), Incidence(face(a), cell, side="right"))``: offsets count from the
        right incident cell. A row reads the samples whose leaves it holds, so the reads are
        those of each axis, where ``samples`` pools them. The face value goes to the left cell
        with weight -1 and to the right cell with weight +1, the native assembly
        net(cell) = J(cell) - J(cell + e_a) (include/pops/numerics/fv/stencil_flux.hpp@b6), and
        the native face loop expects the class face incidence. The array labels are
        ``("state", component)`` and ``("field", slot, component)``.
        """
        from pops.model.face_regions import face_regions
        from pops.model.stencil_expression import row_leaf_names
        labelled = tuple((v.name, ("state", c), offset) for offset, c, v in self.samples) + tuple(
            (v.name, ("field", f, c), offset) for f, offset, c, v in self.field_samples)
        reads = []
        for row in self.rows:
            leaves = row_leaf_names(row)
            reads.append(tuple((label, offset, "right")
                               for leaf, label, offset in labelled if leaf in leaves))
        return face_regions(name, self.dimension, tuple(reads))

    def to_data(self):
        from pops.model.hash_data import canonical_hash_data
        return {"schema": 3, "kind": "stencil_flux", "dimension": self.dimension,
                "components": self.components, "radius": self.radius,
                "rows": canonical_hash_data(self.rows),
                "samples": tuple((offset, c, v.name) for offset, c, v in self.samples),
                "spacing": tuple(v.name for v in self.spacing),
                "field_components": self.field_components,
                "parameters": self.parameters,
                "parameter_symbols": tuple(v.name for v in self.parameter_symbols),
                "field_samples": tuple((f, offset, c, v.name) for f, offset, c, v in self.field_samples),
                "field_ghost_extents": self.field_ghost_extents,
                "ghost_extent": self.ghost_extent,
                "orientation": "positive_axis", "sampling": "right_incident_cell",
                "frequency": "incident_inverse_time", "boundary": "periodic"}


__all__ = ["StencilFlux"]
