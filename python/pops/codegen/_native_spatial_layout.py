"""Immutable native spatial specializations derived from ``LayoutPlan`` facts.

The native production provider is currently two-dimensional, but it must not
recover that fact from a mutable authoring descriptor while it is installing a
simulation.  This value is the one-way boundary between generic layout
normalization and native lowering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pops.identity import Identity, make_identity


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class NativeSpatialLayout:
    """Exact geometry, topology and storage facts consumed by one native layout.

    ``dimension`` is derived from ``shape`` and is deliberately read-only.  The
    generic source remains the normalized layout; this specialization only
    records the production representation that has been accepted at resolve.
    """

    layout_id: str
    coordinate_system: str
    shape: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    periodicity: tuple[bool, ...]
    centering: str = "cell"
    component_shape: tuple[int, ...] = ()
    storage_order: str = "right"
    decomposition: Mapping[str, Any] = field(default_factory=dict)
    topology: Mapping[str, Any] = field(default_factory=dict)
    layout_options: Mapping[str, Any] = field(default_factory=dict)
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise TypeError("NativeSpatialLayout.layout_id must be non-empty")
        if not isinstance(self.coordinate_system, str) or not self.coordinate_system:
            raise TypeError("NativeSpatialLayout.coordinate_system must be non-empty")
        shape = tuple(self.shape)
        lower, upper = tuple(self.lower), tuple(self.upper)
        periodicity = tuple(self.periodicity)
        if not shape or any(type(item) is not int or item < 1 for item in shape):
            raise ValueError("NativeSpatialLayout.shape must contain positive exact integers")
        if len(lower) != len(shape) or len(upper) != len(shape) or len(periodicity) != len(shape):
            raise ValueError("NativeSpatialLayout geometry facts must have one exact rank")
        if any(type(item) is not float for item in (*lower, *upper)):
            raise TypeError("NativeSpatialLayout bounds must be exact floats")
        if any(high <= low for low, high in zip(lower, upper, strict=True)):
            raise ValueError("NativeSpatialLayout upper bounds must exceed lower bounds")
        if any(type(item) is not bool for item in periodicity):
            raise TypeError("NativeSpatialLayout.periodicity must contain exact bools")
        component_shape = tuple(self.component_shape)
        if any(type(item) is not int or item < 1 for item in component_shape):
            raise ValueError("NativeSpatialLayout.component_shape must contain positive integers")
        if self.centering != "cell":
            raise NotImplementedError("native production supports only cell-centered layout storage")
        if self.storage_order not in {"right", "left", "strided"}:
            raise ValueError("NativeSpatialLayout.storage_order is unsupported")
        for name in ("decomposition", "topology", "layout_options"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError("NativeSpatialLayout.%s must be a mapping" % name)
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(self, "component_shape", component_shape)
        object.__setattr__(self, "identity", make_identity("native-spatial-layout", self.to_data()))

    @property
    def dimension(self) -> int:
        return len(self.shape)

    @property
    def component_count(self) -> int:
        result = 1
        for item in self.component_shape:
            result *= item
        return result

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(high - low for low, high in zip(self.lower, self.upper, strict=True))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "layout_id": self.layout_id,
            "coordinate_system": self.coordinate_system,
            "shape": list(self.shape),
            "lower": [item.hex() for item in self.lower],
            "upper": [item.hex() for item in self.upper],
            "periodicity": list(self.periodicity),
            "centering": self.centering,
            "component_shape": list(self.component_shape),
            "component_count": self.component_count,
            "storage_order": self.storage_order,
            "decomposition": dict(self.decomposition),
            "topology": dict(self.topology),
            "layout_options": dict(self.layout_options),
        }

    @classmethod
    def from_normalized(cls, layout: Any) -> "NativeSpatialLayout":
        """Specialize one normalized layout without consulting its authoring provider."""
        from pops.mesh import NormalizedLayout

        if type(layout) is not NormalizedLayout:
            raise TypeError("native spatial specialization requires an exact NormalizedLayout")
        geometry = layout.geometry
        options = dict(layout.options)
        topology = options.get("topology", {})
        if not isinstance(topology, Mapping):
            raise TypeError("normalized layout topology must be a mapping")
        periodic_axes = topology.get("periodic_axes", ())
        indices = {
            item.get("index") for item in periodic_axes
            if isinstance(item, Mapping) and type(item.get("index")) is int
        }
        # Polar topology is intrinsic: theta is periodic and r is physical.  The public
        # descriptor exposes no Cartesian ``topology`` map, so derive it from the normalized
        # coordinate system rather than consulting the mutable mesh object.
        if geometry.coordinate_system.startswith("pops://coordinates/polar"):
            periodicity = (False, True)
        else:
            periodicity = tuple(index in indices for index in range(geometry.dimension))
        return cls(
            layout_id=layout.handle.qualified_id,
            coordinate_system=geometry.coordinate_system,
            shape=tuple(geometry.cells), lower=tuple(float(item) for item in geometry.lower),
            upper=tuple(float(item) for item in geometry.upper), periodicity=periodicity,
            topology=dict(topology), layout_options=options,
            decomposition={"kind": "native-default", "rank": geometry.dimension},
        )


def native_spatial_layouts(layout_plan: Any, *, supported_dimensions: tuple[int, ...] = (2,)) \
        -> Mapping[str, NativeSpatialLayout]:
    """Build the one immutable native specialization per normalized layout at resolve."""
    from pops.mesh import LayoutPlan

    if type(layout_plan) is not LayoutPlan:
        raise TypeError("native spatial specialization requires an exact LayoutPlan")
    if not supported_dimensions or any(type(item) is not int for item in supported_dimensions):
        raise TypeError("supported_dimensions must be a non-empty exact integer tuple")
    rows = {}
    for normalized in layout_plan.layouts:
        row = NativeSpatialLayout.from_normalized(normalized)
        if row.dimension not in supported_dimensions:
            raise NotImplementedError(
                "native production provider supports dimensions %s, not layout %s dimension %d"
                % (supported_dimensions, row.layout_id, row.dimension))
        rows[row.layout_id] = row
    return MappingProxyType(rows)


__all__ = ["NativeSpatialLayout", "native_spatial_layouts"]
