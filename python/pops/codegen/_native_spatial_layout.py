"""Resolve-time native spatial authority derived only from immutable ``LayoutPlan`` rows."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pops._native_facts import NATIVE_SUPPORTED_DIMENSIONS

NATIVE_SUPPORTED_CENTERINGS = ("cell",)


class NativeSpatialLayoutError(ValueError):
    """Structured refusal before compilation or native storage allocation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        layout_id: str | None = None,
        evidence: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.layout_id = layout_id
        self.evidence = evidence

    def to_data(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "layout_id": self.layout_id,
            "message": str(self),
            "evidence": self.evidence,
        }


def _supported_dimensions(value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value \
            or any(type(item) is not int or item not in (1, 2, 3) for item in value) \
            or len(value) != len(set(value)):
        raise TypeError("supported_dimensions must be a unique non-empty tuple from {1,2,3}")
    return value


def native_spatial_layouts(
    layout_plan: Any,
    *,
    supported_dimensions: tuple[int, ...] = NATIVE_SUPPORTED_DIMENSIONS,
    supported_centerings: tuple[str, ...] = NATIVE_SUPPORTED_CENTERINGS,
) -> Mapping[str, Any]:
    """Return exact per-layout specializations, refusing unsupported routes fail-closed."""
    from pops.mesh import LayoutPlan, NativeSpatialLayout

    if type(layout_plan) is not LayoutPlan:
        raise TypeError("native spatial resolution requires an exact LayoutPlan")
    dimensions = _supported_dimensions(supported_dimensions)
    if not isinstance(supported_centerings, tuple) or not supported_centerings \
            or any(not isinstance(item, str) or not item for item in supported_centerings) \
            or len(supported_centerings) != len(set(supported_centerings)):
        raise TypeError("supported_centerings must be a unique non-empty tuple of names")
    rows: dict[str, NativeSpatialLayout] = {}
    selected_dimensions: set[int] = set()
    for normalized in layout_plan.layouts:
        native = normalized.native_spatial_layout
        if native is None:
            raise NativeSpatialLayoutError(
                "native_spatial_layout_unavailable",
                "layout %s has no authenticated native_spatial_data() projection"
                % normalized.handle.qualified_id,
                layout_id=normalized.handle.qualified_id,
                evidence={"supported_dimensions": list(dimensions)},
            )
        if type(native) is not NativeSpatialLayout:
            raise TypeError("LayoutPlan contains a non-exact NativeSpatialLayout")
        if native.dimension not in dimensions:
            raise NativeSpatialLayoutError(
                "native_dimension_unavailable",
                "native production supports dimensions %s, not layout %s dimension %d"
                % (dimensions, native.layout_id, native.dimension),
                layout_id=native.layout_id,
                evidence={
                    "resolved_dimension": native.dimension,
                    "supported_dimensions": list(dimensions),
                },
            )
        if native.centering not in supported_centerings:
            raise NativeSpatialLayoutError(
                "native_centering_unavailable",
                "native production does not support layout %s centering %r"
                % (native.layout_id, native.centering),
                layout_id=native.layout_id,
                evidence={
                    "centering": native.centering,
                    "supported_centerings": list(supported_centerings),
                },
            )
        rows[native.layout_id] = NativeSpatialLayout.from_data(native.to_data())
        selected_dimensions.add(native.dimension)
    if len(selected_dimensions) != 1:
        raise NativeSpatialLayoutError(
            "mixed_native_dimensions",
            "one RuntimeInstance cannot combine layouts with different dimensions",
            evidence={"resolved_dimensions": sorted(selected_dimensions)},
        )
    return MappingProxyType(rows)


def resolved_dimension(layouts: Mapping[str, Any]) -> int:
    """Return the one exact rank carried by an authenticated native-layout mapping."""
    from pops.mesh import NativeSpatialLayout

    if not isinstance(layouts, Mapping) or not layouts:
        raise TypeError("resolved_dimension requires a non-empty native-layout mapping")
    rows = tuple(layouts.values())
    if any(type(row) is not NativeSpatialLayout for row in rows):
        raise TypeError(
            "resolved_dimension requires exact NativeSpatialLayout mapping values")
    dimensions = {row.dimension for row in rows}
    if len(dimensions) != 1:
        raise ValueError("native-layout mapping does not carry one exact resolved dimension")
    return next(iter(dimensions))


def validate_program_spatial_dimension(program: Any, dimension: int) -> None:
    """Match every dimension-qualified IR node to the selected native specialization."""
    from pops.time import Program

    if type(program) is not Program:
        raise TypeError("spatial-dimension validation requires an exact Program")
    if type(dimension) is not int or dimension not in (1, 2, 3):
        raise ValueError("resolved spatial dimension must be 1, 2, or 3")
    for node in program.ir_nodes(recursive=True):
        declared = node["attrs"].get("spatial_dimension")
        if declared is None:
            continue
        if type(declared) is not int or declared not in (1, 2, 3):
            raise ValueError(
                "Program operation %r carries an invalid spatial_dimension"
                % node["op"])
        if declared != dimension:
            raise NativeSpatialLayoutError(
                "program_dimension_mismatch",
                "Program operation %r has spatial rank %d but the resolved layout has rank %d"
                % (node["op"], declared, dimension),
                evidence={
                    "operation": node["op"],
                    "program_dimension": declared,
                    "resolved_dimension": dimension,
                },
            )


__all__ = [
    "NATIVE_SUPPORTED_CENTERINGS", "NATIVE_SUPPORTED_DIMENSIONS",
    "NativeSpatialLayoutError", "native_spatial_layouts", "resolved_dimension",
    "validate_program_spatial_dimension",
]
