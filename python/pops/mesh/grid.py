"""Pure Cartesian grid descriptors derived from framed domains."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pops._geometry_contracts import cartesian_geometry_contract
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.domain import BoundaryPair, CartesianDomainFrame, RectangleFrame
from pops.frames import CartesianAxis
from pops.identity import make_identity
from pops.identity.semantic import semantic_value

from ._layout_plan_contracts import NormalizedGeometry


_SCHEMA_VERSION = 3


def _cells(value: Any, *, dimension: int) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != dimension
    ):
        raise TypeError("CartesianGrid.cells must contain exactly %d integers" % dimension)
    result = []
    for index, count in enumerate(value):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("CartesianGrid.cells[%d] must be an integer, never bool" % index)
        if count < 1:
            raise ValueError("CartesianGrid.cells[%d] must be >= 1" % index)
        result.append(count)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RegularBlocks:
    """Deterministic Cartesian tiling authority for a uniform grid.

    ``max_cells`` is either one exact positive integer (the same maximum extent
    along every axis) or one exact-rank tuple.  It is deliberately a value,
    rather than an execution hint: its canonical boxes become part of the
    native spatial schema and therefore of a grid identity.
    """

    max_cells: int | tuple[int, ...]

    def __post_init__(self) -> None:
        value = self.max_cells
        if type(value) is int:
            if value < 1:
                raise ValueError("RegularBlocks.max_cells must be >= 1")
            return
        if type(value) is not tuple or not value:
            raise TypeError(
                "RegularBlocks.max_cells must be an exact int or non-empty tuple of ints"
            )
        if any(type(item) is not int for item in value):
            raise TypeError("RegularBlocks.max_cells tuple must contain exact integers")
        if any(item < 1 for item in value):
            raise ValueError("RegularBlocks.max_cells tuple entries must be >= 1")

    def extents(self, *, dimension: int) -> tuple[int, ...]:
        if type(dimension) is not int or dimension not in (1, 2, 3):
            raise ValueError("RegularBlocks requires dimension 1, 2, or 3")
        if type(self.max_cells) is int:
            return (self.max_cells,) * dimension
        if len(self.max_cells) != dimension:
            raise ValueError("RegularBlocks.max_cells tuple must match CartesianGrid dimension")
        return self.max_cells

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "layout_type": "regular_blocks",
            "max_cells": (self.max_cells if type(self.max_cells) is int else list(self.max_cells)),
        }

    canonical_identity = to_dict

    @classmethod
    def from_dict(cls, data: Any) -> RegularBlocks:
        required = {"schema_version", "layout_type", "max_cells"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("RegularBlocks data has an unsupported shape")
        if data["schema_version"] != 1 or data["layout_type"] != "regular_blocks":
            raise ValueError("RegularBlocks data uses an unsupported schema")
        raw = data["max_cells"]
        value = tuple(raw) if type(raw) is list else raw
        result = cls(value)
        if result.to_dict() != dict(data):
            raise ValueError("RegularBlocks data is not canonical")
        return result


def _regular_boxes(
    cells: tuple[int, ...], blocks: RegularBlocks
) -> tuple[dict[str, list[int]], ...]:
    """Tile one exact Cartesian extent in lexical lower-bound order."""
    extents = blocks.extents(dimension=len(cells))
    starts = [range(0, count, width) for count, width in zip(cells, extents, strict=True)]
    import itertools

    return tuple(
        {
            "lower": list(lower),
            "upper_exclusive": [
                min(low + width, count)
                for low, width, count in zip(lower, extents, cells, strict=True)
            ],
        }
        for lower in itertools.product(*starts)
    )


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
        if (
            data["schema_version"] != _SCHEMA_VERSION
            or data["topology_type"] != "periodic_axes"
            or not isinstance(data["axes"], list)
        ):
            raise ValueError("PeriodicAxes data uses an unsupported schema")
        result = cls(tuple(CartesianAxis.from_dict(axis) for axis in data["axes"]))
        if result.to_dict() != dict(data):
            raise ValueError("PeriodicAxes data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CartesianGridTopology:
    """Canonical periodic/physical axis partition derived from one framed grid."""

    axis_pairs: tuple[BoundaryPair, ...]
    periodic_axes: tuple[CartesianAxis, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.axis_pairs, tuple)
            or len(self.axis_pairs) not in (1, 2, 3)
            or any(not isinstance(pair, BoundaryPair) for pair in self.axis_pairs)
        ):
            raise TypeError(
                "CartesianGridTopology.axis_pairs must contain 1, 2, or 3 BoundaryPair values"
            )
        indices = tuple(pair.axis.index for pair in self.axis_pairs)
        if indices != tuple(range(len(self.axis_pairs))):
            raise ValueError("CartesianGridTopology axes must use canonical x,y,z order")
        if not isinstance(self.periodic_axes, tuple) or any(
            not isinstance(axis, CartesianAxis) for axis in self.periodic_axes
        ):
            raise TypeError("CartesianGridTopology.periodic_axes must contain CartesianAxis values")
        available = tuple(pair.axis for pair in self.axis_pairs)
        if len(self.periodic_axes) != len(set(self.periodic_axes)) or any(
            axis not in available for axis in self.periodic_axes
        ):
            raise ValueError(
                "CartesianGridTopology periodic axes must be a unique subset of its frame axes"
            )
        if tuple(sorted(self.periodic_axes, key=lambda axis: axis.index)) != self.periodic_axes:
            raise ValueError("CartesianGridTopology periodic axes are not canonically ordered")

    @property
    def physical_axes(self) -> tuple[CartesianAxis, ...]:
        return tuple(pair.axis for pair in self.axis_pairs if pair.axis not in self.periodic_axes)

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
    frame: RectangleFrame | CartesianDomainFrame
    cells: tuple[int, ...]
    periodic: PeriodicAxes | None
    blocks: RegularBlocks | None

    def __init__(self, *, frame: Any, cells: Any, periodic: Any = None, blocks: Any = None) -> None:
        if not isinstance(frame, (RectangleFrame, CartesianDomainFrame)):
            raise TypeError("CartesianGrid.frame must be returned by a bounded Cartesian domain")
        if periodic is not None and not isinstance(periodic, PeriodicAxes):
            raise TypeError(
                "CartesianGrid.periodic must be PeriodicAxes, never bool, strings or indices"
            )
        if periodic is not None and any(axis not in frame.axes for axis in periodic.axes):
            raise ValueError("CartesianGrid periodic axes must belong to its exact frame")
        if blocks is not None and type(blocks) is not RegularBlocks:
            raise TypeError("CartesianGrid.blocks must be an exact RegularBlocks value or None")
        parsed_cells = _cells(cells, dimension=len(frame.axes))
        if blocks is not None:
            blocks.extents(dimension=len(parsed_cells))
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "cells", parsed_cells)
        object.__setattr__(self, "periodic", periodic)
        object.__setattr__(self, "blocks", blocks)

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def axes(self) -> tuple[CartesianAxis, ...]:
        return self.frame.axes

    @property
    def axis_order(self) -> tuple[CartesianAxis, ...]:
        return self.axes

    @property
    def extent(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (self.frame.lower, self.frame.upper)

    @property
    def cell_widths(self) -> tuple[float, ...]:
        return tuple(
            length / count for length, count in zip(self.frame.lengths, self.cells, strict=True)
        )

    @property
    def topology(self) -> CartesianGridTopology:
        boundaries = self.frame.boundaries
        return CartesianGridTopology(
            tuple(boundaries.pair(axis) for axis in self.axes),
            () if self.periodic is None else self.periodic.axes,
        )

    def requirements(self) -> RequirementSet:
        return RequirementSet()

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            {
                "geometry": "cartesian",
                "dim": len(self.cells),
                "bounded_axes": len(self.topology.physical_axes),
                "periodic_axes": len(self.topology.periodic_axes),
            }
        )

    def options(self) -> dict[str, Any]:
        options = {
            "frame_id": self.frame.canonical_id,
            "cells": list(self.cells),
            "axis_order": [axis.name for axis in self.axis_order],
            "extent": {"lower": list(self.frame.lower), "upper": list(self.frame.upper)},
            "cell_widths": list(self.cell_widths),
            "topology": self.topology.to_dict(),
        }
        # Preserve every historical identity bit-for-bit when block authority is absent.
        if self.blocks is not None:
            options["blocks"] = self.blocks.to_dict()
        return options

    def normalized_geometry(self) -> NormalizedGeometry:
        """Project the exact framed grid without relying on runtime-engine internals."""
        dimension = len(self.cells)
        coordinate_system, cell_measure = cartesian_geometry_contract(dimension)
        return NormalizedGeometry(
            coordinate_system=coordinate_system,
            cell_measure=cell_measure,
            axis_names=tuple(axis.name for axis in self.axis_order),
            lower=self.frame.lower,
            upper=self.frame.upper,
            cells=self.cells,
            frame_id=self.frame.canonical_id,
        )

    def native_spatial_data(self) -> dict[str, Any]:
        """Exact topology and base decomposition consumed by native layout normalization."""
        periodic_indices = {axis.index for axis in self.topology.periodic_axes}
        boxes = (
            [
                {
                    "lower": [0 for _ in self.cells],
                    "upper_exclusive": list(self.cells),
                }
            ]
            if self.blocks is None
            else list(_regular_boxes(self.cells, self.blocks))
        )
        return {
            "schema_version": 1,
            "periodicity": [index in periodic_indices for index in range(len(self.cells))],
            "centering": "cell",
            "decomposition": {
                "schema_version": 1,
                "kind": "single_box" if self.blocks is None else "regular_blocks",
                "boxes": boxes,
            },
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
            "schema_version",
            "grid_type",
            "frame",
            "frame_id",
            "cells",
            "axis_order",
            "extent",
            "cell_widths",
            "topology",
        }
        allowed = required | {"blocks"}
        if not isinstance(data, Mapping) or (set(data) != required and set(data) != allowed):
            raise TypeError("CartesianGrid data has an unsupported shape")
        if data["schema_version"] != _SCHEMA_VERSION or data["grid_type"] != "cartesian":
            raise ValueError("CartesianGrid data uses an unsupported schema")
        topology = data["topology"]
        if not isinstance(topology, Mapping) or not isinstance(topology.get("periodic_axes"), list):
            raise TypeError("CartesianGrid topology has an unsupported shape")
        periodic_axes = tuple(CartesianAxis.from_dict(axis) for axis in topology["periodic_axes"])
        raw_frame = data["frame"]
        if not isinstance(raw_frame, Mapping):
            raise TypeError("CartesianGrid frame data has an unsupported shape")
        frame_type = raw_frame.get("frame_type")
        if frame_type == "rectangle_cartesian_2d":
            frame = RectangleFrame.from_dict(raw_frame)
        elif frame_type == "cartesian_box":
            frame = CartesianDomainFrame.from_dict(raw_frame)
        else:
            raise ValueError("CartesianGrid frame uses an unsupported provider")
        result = cls(
            frame=frame,
            cells=data["cells"],
            periodic=None if not periodic_axes else PeriodicAxes(periodic_axes),
            blocks=None if "blocks" not in data else RegularBlocks.from_dict(data["blocks"]),
        )
        if result.to_dict() != dict(data):
            raise ValueError("CartesianGrid data is not canonical")
        return result


__all__ = ["CartesianGrid", "CartesianGridTopology", "PeriodicAxes", "RegularBlocks"]
