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


_SCHEMA_VERSION = 1


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
class CartesianGridTopology:
    """Canonical bounded topology, ordered exactly like the frame axes."""

    axis_pairs: tuple[BoundaryPair, BoundaryPair]

    def __post_init__(self) -> None:
        if not isinstance(self.axis_pairs, tuple) or len(self.axis_pairs) != 2 \
                or any(not isinstance(pair, BoundaryPair) for pair in self.axis_pairs):
            raise TypeError("CartesianGridTopology.axis_pairs must contain two BoundaryPair values")
        indices = tuple(pair.axis.index for pair in self.axis_pairs)
        if indices != (0, 1):
            raise ValueError("CartesianGridTopology axes must use canonical x,y order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "topology_type": "bounded_cartesian",
            "axis_pairs": [pair.to_dict() for pair in self.axis_pairs],
        }


@dataclass(frozen=True, slots=True, init=False)
class CartesianGrid:
    """An immutable cell grid over a :class:`~pops.domain.RectangleFrame`.

    Extent, axis order and boundary topology are derived authorities.  Users only provide the
    framed domain and one cell count per typed axis, so those facts cannot diverge.
    """

    category: ClassVar[str] = "mesh"
    frame: RectangleFrame
    cells: tuple[int, int]

    def __init__(self, *, frame: Any, cells: Any) -> None:
        if not isinstance(frame, RectangleFrame):
            raise TypeError(
                "CartesianGrid.frame must be a RectangleFrame returned by Rectangle.frame()")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "cells", _cells(cells))

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
            boundaries.pair(self.axes[0]), boundaries.pair(self.axes[1])))

    def requirements(self) -> RequirementSet:
        return RequirementSet()

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({
            "geometry": "cartesian",
            "dim": 2,
            "bounded_axes": 2,
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
        result = cls(frame=RectangleFrame.from_dict(data["frame"]), cells=data["cells"])
        if result.to_dict() != dict(data):
            raise ValueError("CartesianGrid data is not canonical")
        return result


__all__ = ["CartesianGrid", "CartesianGridTopology"]
