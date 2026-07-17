"""Pure Cartesian grid descriptors derived from framed domains."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.domain import BoundaryPair, RectangleFrame
from pops.frames import CartesianAxis
from pops.identity import make_identity
from pops.identity.semantic import semantic_value

from ._layout_plan_contracts import (
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_CELL_AREA,
    NormalizedGeometry,
)


_SCHEMA_VERSION = 2


def _cells(value: Any) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError("CartesianGrid.cells must contain exactly two integers")
    result = []
    for index, count in enumerate(value):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("CartesianGrid.cells[%d] must be an integer, never bool" % index)
        if count < 1:
            raise ValueError("CartesianGrid.cells[%d] must be >= 1" % index)
        result.append(count)
    return (result[0], result[1])


@dataclass(frozen=True, slots=True)
class PeriodicAxes:
    """Typed axes whose opposite frame boundaries are identified periodically.

    This is authoring topology, not a backend boolean.  The axes must come from the grid frame;
    ``CartesianGrid`` authenticates that relation and derives the corresponding boundary pairs.
    """

    axes: tuple[CartesianAxis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes, tuple) or not self.axes:
            raise TypeError("PeriodicAxes.axes must be a non-empty tuple of CartesianAxis values")
        if any(not isinstance(axis, CartesianAxis) for axis in self.axes):
            raise TypeError("PeriodicAxes.axes must contain only CartesianAxis values")
        if len(self.axes) != len(set(self.axes)):
            raise ValueError("PeriodicAxes cannot identify one axis more than once")
        ordered = tuple(sorted(self.axes, key=lambda axis: axis.index))
        if ordered != self.axes:
            raise ValueError("PeriodicAxes must follow canonical frame-axis order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "topology_type": "periodic_axes",
            "axes": [axis.to_dict() for axis in self.axes],
        }

    canonical_identity = to_dict

    @classmethod
    def from_dict(cls, data: Any) -> PeriodicAxes:
        required = {"schema_version", "topology_type", "axes"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("PeriodicAxes data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION \
                or data["topology_type"] != "periodic_axes" \
                or not isinstance(data["axes"], list):
            raise ValueError("PeriodicAxes data uses an unsupported schema")
        result = cls(tuple(CartesianAxis.from_dict(axis) for axis in data["axes"]))
        if result.to_dict() != dict(data):
            raise ValueError("PeriodicAxes data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CartesianGridTopology:
    """Canonical periodic/physical axis partition derived from one framed grid."""

    axis_pairs: tuple[BoundaryPair, BoundaryPair]
    periodic_axes: tuple[CartesianAxis, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.axis_pairs, tuple) or len(self.axis_pairs) != 2 \
                or any(not isinstance(pair, BoundaryPair) for pair in self.axis_pairs):
            raise TypeError("CartesianGridTopology.axis_pairs must contain two BoundaryPair values")
        indices = tuple(pair.axis.index for pair in self.axis_pairs)
        if indices != (0, 1):
            raise ValueError("CartesianGridTopology axes must use canonical x,y order")
        if not isinstance(self.periodic_axes, tuple) or any(
                not isinstance(axis, CartesianAxis) for axis in self.periodic_axes):
            raise TypeError(
                "CartesianGridTopology.periodic_axes must contain CartesianAxis values")
        available = tuple(pair.axis for pair in self.axis_pairs)
        if len(self.periodic_axes) != len(set(self.periodic_axes)) \
                or any(axis not in available for axis in self.periodic_axes):
            raise ValueError(
                "CartesianGridTopology periodic axes must be a unique subset of its frame axes")
        if tuple(sorted(self.periodic_axes, key=lambda axis: axis.index)) != self.periodic_axes:
            raise ValueError("CartesianGridTopology periodic axes are not canonically ordered")

    @property
    def physical_axes(self) -> tuple[CartesianAxis, ...]:
        return tuple(
            pair.axis for pair in self.axis_pairs if pair.axis not in self.periodic_axes)

    def is_periodic(self, axis: CartesianAxis) -> bool:
        if not isinstance(axis, CartesianAxis):
            raise TypeError("CartesianGridTopology.is_periodic requires a CartesianAxis")
        if axis not in tuple(pair.axis for pair in self.axis_pairs):
            raise ValueError("axis is absent from this CartesianGridTopology")
        return axis in self.periodic_axes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "topology_type": "cartesian_axis_partition",
            "axis_pairs": [pair.to_dict() for pair in self.axis_pairs],
            "periodic_axes": [axis.to_dict() for axis in self.periodic_axes],
            "physical_axes": [axis.to_dict() for axis in self.physical_axes],
        }


@dataclass(frozen=True, slots=True, init=False)
class CartesianGrid:
    """An immutable cell grid over a :class:`~pops.domain.RectangleFrame`.

    Extent, axis order and boundary topology are derived authorities.  Users only provide the
    framed domain and one cell count per typed axis, so those facts cannot diverge.
    """

    category: ClassVar[str] = "mesh"
    __pops_ir_immutable__: ClassVar[bool] = True
    frame: RectangleFrame
    cells: tuple[int, int]
    periodic: PeriodicAxes | None

    def __init__(self, *, frame: Any, cells: Any, periodic: Any = None) -> None:
        if not isinstance(frame, RectangleFrame):
            raise TypeError(
                "CartesianGrid.frame must be a RectangleFrame returned by Rectangle.frame()")
        if periodic is not None and not isinstance(periodic, PeriodicAxes):
            raise TypeError(
                "CartesianGrid.periodic must be PeriodicAxes, never bool, strings or indices")
        if periodic is not None and any(axis not in frame.axes for axis in periodic.axes):
            raise ValueError("CartesianGrid periodic axes must belong to its exact frame")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "cells", _cells(cells))
        object.__setattr__(self, "periodic", periodic)

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def axes(self) -> tuple[CartesianAxis, CartesianAxis]:
        return self.frame.axes

    @property
    def axis_order(self) -> tuple[CartesianAxis, CartesianAxis]:
        return self.axes

    @property
    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.frame.lower, self.frame.upper)

    @property
    def cell_widths(self) -> tuple[float, float]:
        lengths = self.frame.lengths
        return (lengths[0] / self.cells[0], lengths[1] / self.cells[1])

    @property
    def topology(self) -> CartesianGridTopology:
        boundaries = self.frame.boundaries
        return CartesianGridTopology((
            boundaries.pair(self.axes[0]), boundaries.pair(self.axes[1])),
            () if self.periodic is None else self.periodic.axes,
        )

    def requirements(self) -> RequirementSet:
        return RequirementSet()

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({
            "geometry": "cartesian",
            "dim": 2,
            "bounded_axes": len(self.topology.physical_axes),
            "periodic_axes": len(self.topology.periodic_axes),
        })

    def options(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame.canonical_id,
            "cells": list(self.cells),
            "axis_order": [axis.name for axis in self.axis_order],
            "extent": {"lower": list(self.frame.lower), "upper": list(self.frame.upper)},
            "cell_widths": list(self.cell_widths),
            "topology": self.topology.to_dict(),
        }

    def normalized_geometry(self) -> NormalizedGeometry:
        """Project the exact framed grid without relying on runtime-engine internals."""
        return NormalizedGeometry(
            coordinate_system=CARTESIAN_2D_COORDINATES,
            cell_measure=CARTESIAN_CELL_AREA,
            axis_names=tuple(axis.name for axis in self.axis_order),
            lower=self.frame.lower,
            upper=self.frame.upper,
            cells=self.cells,
        )

    def validate(self, context: Any = None) -> bool:
        del context
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "grid_type": "cartesian",
            "frame": self.frame.to_dict(),
            **self.options(),
        }

    def __pops_semantic_data__(self) -> dict[str, Any]:
        """Project this immutable grid through the open semantic-value protocol.

        Layout descriptors may embed a mesh value in their own semantic data.  The dunder keeps
        that extension boundary explicit: snapshot code consumes one protocol instead of growing
        concrete ``CartesianGrid`` / third-party mesh branches.
        """
        return self.to_dict()

    canonical_identity = to_dict

    @property
    def canonical_id(self) -> str:
        payload = semantic_value(self.to_dict(), where="grid identity")
        return make_identity("grid", payload, schema_version=_SCHEMA_VERSION).token

    def inspect(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "canonical_id": self.canonical_id,
            "options": self.options(),
            "requirements": self.requirements().to_dict(),
            "capabilities": self.capabilities().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> CartesianGrid:
        required = {
            "schema_version", "grid_type", "frame", "frame_id", "cells", "axis_order",
            "extent", "cell_widths", "topology",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("CartesianGrid data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["grid_type"] != "cartesian":
            raise ValueError("CartesianGrid data uses an unsupported schema")
        topology = data["topology"]
        if not isinstance(topology, Mapping) or not isinstance(
                topology.get("periodic_axes"), list):
            raise TypeError("CartesianGrid topology has an unsupported shape")
        periodic_axes = tuple(
            CartesianAxis.from_dict(axis) for axis in topology["periodic_axes"])
        result = cls(
            frame=RectangleFrame.from_dict(data["frame"]),
            cells=data["cells"],
            periodic=None if not periodic_axes else PeriodicAxes(periodic_axes),
        )
        if result.to_dict() != dict(data):
            raise ValueError("CartesianGrid data is not canonical")
        return result


__all__ = ["CartesianGrid", "CartesianGridTopology", "PeriodicAxes"]
