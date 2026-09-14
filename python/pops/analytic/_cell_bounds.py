"""Explicit integration bounds for Cartesian conservative cell projections."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._functions import input


@dataclass(frozen=True, slots=True)
class CellBounds:
    """Typed bounds of the cell being projected, ordered by the physical frame axes.

    Expressions built from these leaves are evaluated only by an explicit cell-integral
    consumer. ``measure`` is the Cartesian volume, not an implicit quadrature weight.
    """

    frame: Any

    def __post_init__(self):
        from pops.frames import Cartesian
        from pops.domain.cartesian import CartesianDomainFrame
        from pops.domain.rectangle import RectangleFrame
        if not isinstance(self.frame, (Cartesian, CartesianDomainFrame, RectangleFrame)):
            raise TypeError("CellBounds requires a typed Cartesian frame")

    def _bound(self, axis, side):
        if axis not in self.frame.axes:
            raise ValueError("cell bound axis does not belong to the frame")
        index = self.frame.axes.index(axis)
        return input(2 * index + (side == "upper"),
                     "cell_bound:%s:%s:%s" % (self.frame.canonical_id, axis.name, side))

    def lower(self, axis):
        return self._bound(axis, "lower")

    def upper(self, axis):
        return self._bound(axis, "upper")

    @property
    def measure(self):
        """The exact native Cartesian cell volume used to normalize the integral."""
        return input(2 * len(self.frame.axes), "cell_measure:%s" % self.frame.canonical_id)


def validate_cell_integrals(expressions, frame):
    bounds = CellBounds(frame)
    expected = dict(reference for axis in frame.axes
                    for expression in (bounds.lower(axis), bounds.upper(axis))
                    for reference in expression.input_references())
    expected.update(dict(bounds.measure.input_references()))
    for expression in expressions:
        expression.validate()
        if expression.frame_id not in (None, frame.canonical_id):
            raise ValueError("cell integral expression belongs to another frame")
        if expression.time_clocks():
            raise ValueError("initial cell integrals cannot depend on a runtime clock")
        if expression.frame_id is not None:
            raise ValueError("cell integral must use cell bounds, not point coordinates")
        for slot, component in expression.input_references():
            if expected.get(slot) != component:
                raise ValueError("cell integral contains an unauthenticated bound input")


def cell_frame_from_data(data):
    from pops.frames import Cartesian
    from pops.domain.cartesian import CartesianDomainFrame
    from pops.domain.rectangle import RectangleFrame
    if not isinstance(data, Mapping):
        raise TypeError("cell integral frame must be a canonical mapping")
    frame_type = data.get("frame_type")
    providers = {"cartesian": Cartesian, "cartesian_box": CartesianDomainFrame,
                 "rectangle_cartesian_2d": RectangleFrame}
    if frame_type not in providers:
        raise ValueError("cell integral frame requires a Cartesian geometry contract")
    def decode(value):
        if isinstance(value, Mapping):
            if set(value) == {"binary64"}:
                from pops.identity.scalar import native_binary64
                return native_binary64(value, where="cell integral frame geometry")
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [decode(item) for item in value]
        return value
    return providers[frame_type].from_dict(decode(data))


def validate_cell_integral_contract(data, *, frame_id, component_count):
    """Authenticate exact user-supplied integrals and their complete native bound authority."""
    from pops.analytic import ScalarExpr
    if not isinstance(data, Mapping) or set(data) != {
            "schema_version", "frame", "measure", "exactness", "components"}:
        raise TypeError("cell integral contract has an unsupported shape")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 \
            or data["measure"] != "cartesian_volume" or data["exactness"] != "author_declared":
        raise ValueError("cell integral requires explicit Cartesian volume and declared exactness")
    frame = cell_frame_from_data(data["frame"])
    if frame.canonical_id != frame_id:
        raise ValueError("cell integral frame differs from initial expression frame")
    components = data["components"]
    if not isinstance(components, (list, tuple)) or len(components) != component_count:
        raise ValueError("cell integral component count differs from initial state")
    validate_cell_integrals(tuple(ScalarExpr.from_data(e) for e in components), frame)
    return frame
