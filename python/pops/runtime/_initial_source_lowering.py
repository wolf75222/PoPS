"""Strict shared schema boundary for resolved initial-condition providers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


_PROJECTION_KEYS = {
    "schema_version", "projection", "formal_order", "ghost_depth",
}
_SOURCE_KEYS = {
    "bound_level_zero": {"native_route", "projection"},
    "constant_field": {"native_route", "components", "projection"},
    "gaussian_field": {
        "native_route", "frame_id", "center", "background", "amplitude",
        "inverse_width", "projection",
    },
    "analytic_expression": {
        "native_route", "frame_id", "components", "projection",
    },
    "field_mapped_analytic_expression": {
        "native_route", "frame_id", "seed_components", "components", "inputs", "projection",
    },
}


def native_binary64(value: Any, *, where: str) -> float:
    """Decode exactly one canonical finite binary64 value; no loose numeric fallback."""
    if not isinstance(value, Mapping) or set(value) != {"binary64"} \
            or not isinstance(value["binary64"], str):
        raise TypeError("%s must be one canonical binary64 value" % where)
    try:
        result = float.fromhex(value["binary64"])
    except (OverflowError, ValueError):
        raise ValueError("%s contains an invalid binary64 payload" % where) from None
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % where)
    if value["binary64"] != result.hex():
        raise ValueError(
            "%s binary64 payload is not canonical; expected %r"
            % (where, result.hex())
        )
    return result


def validate_initial_source(source: Any, *, where: str) -> str:
    """Authenticate the complete route schema before either native runtime consumes it."""
    if not isinstance(source, Mapping):
        raise TypeError("%s must be a canonical mapping" % where)
    route = source.get("native_route")
    if type(route) is not str or route not in _SOURCE_KEYS:
        raise NotImplementedError("%s route %r is not implemented" % (where, route))
    if set(source) != _SOURCE_KEYS[route]:
        raise TypeError(
            "%s route %r requires exactly keys %s"
            % (where, route, sorted(_SOURCE_KEYS[route]))
        )
    projection = source["projection"]
    if not isinstance(projection, Mapping) or set(projection) != _PROJECTION_KEYS:
        raise TypeError("%s projection has an unsupported shape" % where)
    if type(projection.get("schema_version")) is not int \
            or projection["schema_version"] != 1 \
            or projection.get("projection") != "conservative_cell_average" \
            or type(projection.get("formal_order")) is not int \
            or projection["formal_order"] != 2 \
            or projection.get("ghost_depth") != [1]:
        raise ValueError("%s requires the canonical ConservativeCellAverage contract" % where)
    if route == "constant_field":
        components = source["components"]
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) \
                or not components:
            raise TypeError("%s constant components must be a non-empty sequence" % where)
        for index, value in enumerate(components):
            native_binary64(value, where="%s.components[%d]" % (where, index))
    elif route == "gaussian_field":
        frame_id = source["frame_id"]
        center = source["center"]
        if not isinstance(frame_id, str) or not frame_id:
            raise TypeError("%s Gaussian frame_id must be non-empty" % where)
        if not isinstance(center, Mapping) or set(center) != {"x", "y"}:
            raise TypeError("%s Gaussian center must contain exactly x/y" % where)
        for name, value in (
            ("center.x", center["x"]), ("center.y", center["y"]),
            ("background", source["background"]), ("amplitude", source["amplitude"]),
            ("inverse_width", source["inverse_width"]),
        ):
            native_binary64(value, where="%s.%s" % (where, name))
    elif route == "analytic_expression":
        if not isinstance(source["frame_id"], str) or not source["frame_id"]:
            raise TypeError("%s analytic frame_id must be non-empty" % where)
        components = source["components"]
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) \
                or not components:
            raise TypeError("%s analytic components must be a non-empty sequence" % where)
    elif route == "field_mapped_analytic_expression":
        if not isinstance(source["frame_id"], str) or not source["frame_id"]:
            raise TypeError("%s field-mapped analytic frame_id must be non-empty" % where)
        for key in ("seed_components", "components"):
            components = source[key]
            if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) \
                    or not components:
                raise TypeError("%s %s must be a non-empty sequence" % (where, key))
        inputs = source["inputs"]
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence) or not inputs:
            raise TypeError("%s inputs must be a non-empty sequence" % where)
        ids = []
        for index, row in enumerate(inputs):
            if not isinstance(row, Mapping) \
                    or set(row) != {"value_id", "source", "component"}:
                raise TypeError("%s inputs[%d] has an unsupported shape" % (where, index))
            value_id = row["value_id"]
            source_name = row["source"]
            component = row["component"]
            if isinstance(value_id, bool) or not isinstance(value_id, int) or value_id < 0:
                raise TypeError("%s inputs[%d].value_id must be a non-negative integer"
                                % (where, index))
            if source_name not in ("state", "aux"):
                raise ValueError("%s inputs[%d].source must be 'state' or 'aux'"
                                 % (where, index))
            if isinstance(component, bool) or not isinstance(component, int) or component < 0:
                raise TypeError("%s inputs[%d].component must be a non-negative integer"
                                % (where, index))
            ids.append(value_id)
        if sorted(ids) != list(range(len(ids))):
            raise ValueError("%s inputs value_id entries must be contiguous from zero" % where)
    return route


__all__ = ["native_binary64", "validate_initial_source"]
