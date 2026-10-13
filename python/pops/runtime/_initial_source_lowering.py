"""Strict shared schema boundary for resolved initial-condition providers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pops.identity.scalar import native_binary64


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
        "native_route", "frame_id", "seed_components", "components", "consumer_qid", "inputs",
        "projection",
    },
    "bootstrap_program": {
        "native_route", "frame_id", "seed_sampling", "seed_components", "consumer_qid",
        "stages", "projection",
    },
}

_CARTESIAN_AXIS_NAMES = ("x", "y", "z")


def ranked_gaussian_center(source: Mapping[str, Any], *, where: str) -> tuple[float, ...]:
    """Return the exact 1D/2D/3D Cartesian center carried by one Gaussian source."""
    center = source.get("center")
    if not isinstance(center, Mapping):
        raise TypeError("%s Gaussian center must be a canonical mapping" % where)
    names = tuple(center)
    if names != _CARTESIAN_AXIS_NAMES[:len(names)] or len(names) not in (1, 2, 3):
        raise TypeError(
            "%s Gaussian center must contain the exact Cartesian axis prefix x[/y/z]" % where
        )
    return tuple(
        native_binary64(center[name], where="%s.center.%s" % (where, name))
        for name in names
    )


def validate_initial_source(source: Any, *, where: str) -> str:
    """Authenticate the complete route schema before either native runtime consumes it."""
    if not isinstance(source, Mapping):
        raise TypeError("%s must be a canonical mapping" % where)
    route = source.get("native_route")
    if type(route) is not str or route not in _SOURCE_KEYS:
        raise NotImplementedError("%s route %r is not implemented" % (where, route))
    expected_keys = _SOURCE_KEYS[route] | ({"cell_integrals"} if route == "analytic_expression"
                                             and "cell_integrals" in source else set())
    if set(source) != expected_keys:
        raise TypeError(
            "%s route %r requires exactly keys %s"
            % (where, route, sorted(expected_keys))
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
        if not isinstance(frame_id, str) or not frame_id:
            raise TypeError("%s Gaussian frame_id must be non-empty" % where)
        ranked_gaussian_center(source, where=where)
        for name, value in (
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
        from pops.analytic import ScalarExpr
        for raw in components:
            expression = ScalarExpr.from_data(raw)
            if expression.frame_id not in (None, source["frame_id"]):
                raise ValueError("initial point expression belongs to another frame")
            if expression.input_references() or expression.time_clocks():
                raise ValueError("initial point expressions cannot read discrete inputs or clocks")
        if "cell_integrals" in source:
            from pops.analytic._cell_bounds import validate_cell_integral_contract
            validate_cell_integral_contract(source["cell_integrals"], frame_id=source["frame_id"],
                                            component_count=len(components))
    elif route == "bootstrap_program":
        if not isinstance(source["frame_id"], str) or not source["frame_id"] \
                or source["seed_sampling"] != "cell_center" \
                or not isinstance(source["consumer_qid"], str) or not source["consumer_qid"]:
            raise ValueError("%s requires exact frame, consumer and cell_center seed" % where)
        from pops.analytic import ScalarExpr

        seed = source["seed_components"]
        if not isinstance(seed, Sequence) or isinstance(seed, (str, bytes)) or not seed:
            raise TypeError("%s requires a non-empty cell-centre seed" % where)
        for raw in seed:
            expression = ScalarExpr.from_data(raw)
            if expression.frame_id not in (None, source["frame_id"]) \
                    or expression.input_references() or expression.time_clocks():
                raise ValueError("%s seed has an incompatible input, frame or clock" % where)
        stages = source["stages"]
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not stages:
            raise TypeError("%s requires an ordered non-empty stage sequence" % where)
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise TypeError("%s stage must be a canonical mapping" % where)
            operation = stage.get("operation")
            if operation == "solve_fields":
                if set(stage) != {"operation"}:
                    raise TypeError("%s field solve has an unsupported shape" % where)
                continue
            if operation != "map_state" or set(stage) != {
                    "operation", "consumer_qid", "components", "inputs"}:
                raise TypeError("%s state map has an unsupported shape" % where)
            if not isinstance(stage["consumer_qid"], str) or not stage["consumer_qid"] \
                    or not isinstance(stage["components"], Sequence) \
                    or isinstance(stage["components"], (str, bytes)) \
                    or len(stage["components"]) != len(seed):
                raise ValueError("%s state map changes its consumer or complete state shape" % where)
            inputs = stage["inputs"]
            if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
                raise TypeError("%s state map inputs must be an ordered sequence" % where)
            ids = []
            for row in inputs:
                if not isinstance(row, Mapping) or type(row.get("value_id")) is not int \
                        or row["value_id"] < 0:
                    raise TypeError("%s map input requires an integer value_id" % where)
                ids.append(row["value_id"])
                if row.get("source") == "state":
                    if set(row) != {"value_id", "source", "component"} \
                            or type(row["component"]) is not int \
                            or not 0 <= row["component"] < len(seed):
                        raise ValueError("%s map state input has an unsupported shape" % where)
                    continue
                derivative = row.get("source") == "provider_derivative"
                expected = {"value_id", "source", "key"} | (
                    {"axis", "edge_order"} if derivative else set())
                if row.get("source") not in ("provider", "provider_derivative") \
                        or set(row) != expected:
                    raise TypeError("%s map provider input has an unsupported shape" % where)
                key = row["key"]
                if not isinstance(key, Mapping) or set(key) != {
                        "owner_qid", "space_kind", "space_name", "component"} \
                        or not all(isinstance(value, str) and value for value in key.values()):
                    raise TypeError("%s requires an exact owner-qualified ComponentKey" % where)
                if derivative and (type(row["axis"]) is not int or row["axis"] not in (0, 1, 2)
                                   or type(row["edge_order"]) is not int or row["edge_order"] != 2):
                    raise ValueError("%s derivative requires a ranked axis and edge_order=2" % where)
            if sorted(ids) != list(range(len(ids))):
                raise ValueError("%s map input ids must be contiguous from zero" % where)
            used = set()
            for raw in stage["components"]:
                expression = ScalarExpr.from_data(raw)
                if expression.frame_id not in (None, source["frame_id"]) or expression.time_clocks():
                    raise ValueError("%s map has an incompatible frame or clock" % where)
                used.update(value_id for value_id, _ in expression.input_references())
            if sorted(used) != list(range(len(ids))):
                raise ValueError("%s map must use exactly every declared input id" % where)
        if stages[0]["operation"] != "solve_fields" \
                or stages[-1]["operation"] != "solve_fields":
            raise ValueError("%s stages must start and finish with a field solve" % where)
    elif route == "field_mapped_analytic_expression":
        if not isinstance(source["frame_id"], str) or not source["frame_id"]:
            raise TypeError("%s field-mapped analytic frame_id must be non-empty" % where)
        if not isinstance(source["consumer_qid"], str) or not source["consumer_qid"]:
            raise TypeError("%s field-mapped analytic consumer_qid must be non-empty" % where)
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
            if not isinstance(row, Mapping):
                raise TypeError("%s inputs[%d] has an unsupported shape" % (where, index))
            if "value_id" not in row or "source" not in row:
                raise TypeError("%s inputs[%d] has an unsupported shape" % (where, index))
            value_id = row["value_id"]
            source_name = row["source"]
            if isinstance(value_id, bool) or not isinstance(value_id, int) or value_id < 0:
                raise TypeError("%s inputs[%d].value_id must be a non-negative integer"
                                % (where, index))
            if source_name == "state":
                if set(row) != {"value_id", "source", "component"}:
                    raise TypeError("%s inputs[%d] has an unsupported state shape" % (where, index))
                component = row["component"]
                if isinstance(component, bool) or not isinstance(component, int) or component < 0:
                    raise TypeError("%s inputs[%d].component must be a non-negative integer"
                                    % (where, index))
            elif source_name == "provider":
                if set(row) != {"value_id", "source", "key"}:
                    raise TypeError("%s inputs[%d] has an unsupported provider shape" % (where, index))
                key = row["key"]
                if not isinstance(key, Mapping) or set(key) != {
                    "owner_qid", "space_kind", "space_name", "component",
                } or not all(isinstance(value, str) and value for value in key.values()):
                    raise TypeError(
                        "%s inputs[%d].key must be one exact owner-qualified ComponentKey"
                        % (where, index)
                    )
            else:
                raise ValueError("%s inputs[%d].source must be 'state' or 'provider'"
                                 % (where, index))
            ids.append(value_id)
        if sorted(ids) != list(range(len(ids))):
            raise ValueError("%s inputs value_id entries must be contiguous from zero" % where)
    return route


__all__ = ["native_binary64", "ranked_gaussian_center", "validate_initial_source"]
