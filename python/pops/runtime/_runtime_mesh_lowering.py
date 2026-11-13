"""Lower exact mesh plans onto native runtime configuration seams.

This module contains only backend lowering.  Runtime selection lives in
``_runtime_executor`` and therefore never depends on historical target strings.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from pops._geometry_contracts import cartesian_geometry_contract
from pops.runtime._amr_bind_lowering import amr_config_from_layout


def _uniform_system_values(
    native_layout: Any,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[bool, ...],
    tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    str,
]:
    """Project one exact ranked uniform layout into ``SystemConfig<Dim>`` values."""
    from pops.mesh import NativeSpatialLayout

    if type(native_layout) is not NativeSpatialLayout:
        raise TypeError("native uniform lowering requires an exact NativeSpatialLayout")
    dimension = native_layout.dimension
    expected_coordinates, expected_measure = cartesian_geometry_contract(dimension)
    if (
        native_layout.coordinate_system != expected_coordinates
        or native_layout.cell_measure != expected_measure
        or native_layout.centering != "cell"
    ):
        raise NotImplementedError(
            "native uniform SystemConfig<%d> requires the exact cell-centered Cartesian "
            "coordinate and measure providers" % dimension
        )
    shape = native_layout.shape
    decomposition = native_layout.decomposition
    if decomposition.get("schema_version") != 1:
        raise TypeError("native uniform decomposition uses an unsupported schema")
    boxes = decomposition.get("boxes")
    if (
        decomposition.get("kind") not in {"single_box", "axis_bands"}
        or not isinstance(boxes, (tuple, list))
        or not boxes
    ):
        raise NotImplementedError(
            "native uniform SystemConfig requires an explicit ranked box decomposition"
        )
    ranked_boxes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for index, box in enumerate(boxes):
        if not isinstance(box, Mapping) or set(box) != {"lower", "upper_exclusive"}:
            raise TypeError("native uniform box %d has an unsupported schema" % index)
        lower = tuple(box["lower"])
        upper = tuple(box["upper_exclusive"])
        if (
            len(lower) != dimension
            or len(upper) != dimension
            or any(type(value) is not int for value in lower + upper)
        ):
            raise TypeError("native uniform boxes must contain exact ranked integer bounds")
        if any(
            low < 0 or high <= low or high > extent
            for low, high, extent in zip(lower, upper, shape, strict=True)
        ):
            raise ValueError("native uniform box lies outside the exact layout shape")
        ranked_boxes.append((lower, upper))
    for index, (lower, upper) in enumerate(ranked_boxes):
        for other_lower, other_upper in ranked_boxes[:index]:
            if all(
                max(low, other_low) < min(high, other_high)
                for low, high, other_low, other_high in zip(
                    lower, upper, other_lower, other_upper, strict=True
                )
            ):
                raise ValueError("native uniform decomposition contains overlapping boxes")
    covered = sum(
        math.prod(high - low for low, high in zip(lower, upper, strict=True))
        for lower, upper in ranked_boxes
    )
    if covered != math.prod(shape):
        raise ValueError("native uniform decomposition does not tile the exact layout shape")
    return (
        tuple(shape),
        tuple(native_layout.lower),
        tuple(native_layout.upper),
        native_layout.periodicity,
        tuple(ranked_boxes),
        native_layout.coordinate_system,
    )


def system_config_from_layout(native_layout: Any) -> Any:
    """Build the native uniform config from an authenticated layout descriptor."""
    from pops._bootstrap import SystemConfig

    shape, lower, upper, periodicity, boxes, coordinate_system = _uniform_system_values(
        native_layout
    )
    cfg = SystemConfig()
    cfg.shape = shape
    cfg.lower = lower
    cfg.upper = upper
    cfg.periodicity = periodicity
    cfg.boxes = boxes
    cfg.coordinate_system = coordinate_system
    return cfg


def install_embedded_boundary(sim: Any, normalized_layout: Any) -> None:
    """Install one signed implicit geometry in the selected exact-rank native provider.

    Geometry authoring remains open through the small ``level_set(frame)`` protocol, but that
    protocol is resolved while building the LayoutPlan.  Bind consumes only its authenticated
    canonical data and never calls a user provider.  Uniform ``System<Dim>`` and adaptive
    ``AmrSystem<Dim>`` deliberately share this one lowering path.
    """
    projection = getattr(normalized_layout, "to_data", None)
    if not callable(projection):
        raise TypeError("embedded-boundary installation requires a normalized layout")
    normalized_data = projection()
    options = normalized_data.get("options") if isinstance(normalized_data, dict) else None
    embedded = options.get("embedded_boundary") if isinstance(options, dict) else None
    if embedded is None:
        return
    if (
        not hasattr(embedded, "get")
        or embedded.get("schema_version") != 1
        or set(embedded) != {"schema_version", "level_set", "boundary", "transport"}
    ):
        raise TypeError("normalized embedded-boundary data has an unsupported shape")
    if embedded["boundary"] != {"provider": "zero_flux"}:
        raise NotImplementedError(
            "the installed embedded-boundary runtime provides only pops.boundary.ZeroFlux()"
        )
    frame_id = getattr(getattr(normalized_layout, "geometry", None), "frame_id", None)
    if not isinstance(frame_id, str) or not frame_id:
        raise TypeError("normalized embedded geometry requires a canonical frame identity")
    from pops.mesh.geometry import LevelSet

    level_set = LevelSet.from_data(embedded["level_set"])
    if level_set.frame_id not in (None, frame_id):
        raise ValueError("signed embedded LevelSet differs from the normalized layout frame")

    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    ((opcodes, literals),) = lower_analytic_components(
        (level_set.expression.to_data(),),
        frame_id=frame_id,
    )
    transport = embedded["transport"]
    if not hasattr(transport, "get") or set(transport) != {
        "mode",
        "kappa_min",
        "face_open_eps",
        "cut_theta_min",
    }:
        raise TypeError("normalized embedded transport data has an unsupported shape")
    sim._s._set_analytic_level_set(
        list(opcodes),
        list(literals),
        transport["mode"],
        transport["kappa_min"],
        transport["face_open_eps"],
        transport["cut_theta_min"],
    )


def install_static_geometry(sim: Any, normalized_layout: Any) -> None:
    """Install the static geometry arrays of a mapped layout in the Uniform native System.

    A coordinate map that reads no clock is read at resolution as static geometry arrays (N17, C1).
    Bind lowers its components to the analytic VM, with the stated rule of a chart, and the native
    System generates and certifies the arrays before the first block; no Python callback reaches a
    kernel. A box layout or a moving map installs nothing here.
    """
    from pops.mesh._static_geometry import layout_static_geometry

    reading = layout_static_geometry(normalized_layout)
    if reading is None:
        return
    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    lowered = lower_analytic_components(reading.components, frame_id=reading.frame_id)
    rule = reading.rule or {"nodes": (), "weights": ()}
    sim._s._install_static_geometry(
        normalized_layout.handle.qualified_id,
        reading.kind,
        [list(opcodes) for opcodes, _ in lowered],
        [list(literals) for _, literals in lowered],
        [float.fromhex(node) for node in rule["nodes"]],
        [float.fromhex(weight) for weight in rule["weights"]],
    )


def flow_bootstrap_tagging(
    sim: Any,
    bootstrap: Any,
    params: Any,
    *,
    clock_identity: str,
    program_routes: Any = None,
) -> None:
    """Install the tables of the refinement criterion of one adaptive layout.

    The criterion, written as regions of ``cell``, is already generated into the compiled
    Program: bind installs its field and parameter tables. ``program_routes`` is the pair (block
    routes, parameter routes) of the compiled Program, which locates each runtime parameter of a
    model that the criterion reads in its block's RuntimeParams table.
    """
    if not isinstance(clock_identity, str) or not clock_identity:
        raise ValueError("pops.bind: AMR tagging requires one exact clock identity")
    from pops.mesh._amr import ResolvedRefinementCriterion

    if type(bootstrap.tagging) is not ResolvedRefinementCriterion:
        raise TypeError("pops.bind: AMR tagging requires a ResolvedRefinementCriterion")
    _flow_refinement_criterion(
        sim, bootstrap.tagging, params, clock_identity=clock_identity,
        program_routes=program_routes)


def _flow_refinement_criterion(
    sim: Any, criterion: Any, params: Any, *, clock_identity: str, program_routes: Any = None,
) -> None:
    """Install the tables of a criterion generated into the compiled Program (DM13b)."""
    data = criterion.runtime_criterion_data(params)
    routes: tuple[tuple[int, int], ...] = ()
    if criterion.model_parameters:
        from pops.mesh._amr.refinement_criterion import criterion_parameter_routes

        if program_routes is None:
            raise ValueError("pops.bind: a refinement criterion that reads a model parameter "
                             "needs the parameter routes of its compiled Program")
        routes = criterion_parameter_routes(criterion, *program_routes)
    if (
        type(data) is not dict
        or data.get("schema_version") != 1
        or data.get("criterion_type") != "amr_refinement_criterion_runtime"
        or data.get("identity") != criterion.qualified_id
    ):
        raise ValueError("pops.bind: refinement criterion returned an unsupported runtime manifest")
    hysteresis = data.get("hysteresis")
    if type(hysteresis) is not dict or hysteresis.get("hysteresis_type") != "min_cycles":
        raise ValueError("pops.bind: unsupported AMR hysteresis manifest")
    min_cycles = hysteresis.get("min_cycles")
    if isinstance(min_cycles, bool) or not isinstance(min_cycles, int) or min_cycles < 0:
        raise ValueError("pops.bind: AMR hysteresis min_cycles must be an integer >= 0")
    fields = data.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("pops.bind: a refinement criterion reads at least one field")
    from pops.identity import make_identity
    from pops.identity.semantic import semantic_value

    payload = {
        "schema_version": 1,
        "program_type": "bound_amr_refinement_criterion",
        "criterion_identity": data["identity"],
        "fields": fields,
        "parameters": data["parameters"],
        **({"model_parameters": [
            {**row, "program_block": index, "slot": slot}
            for row, (index, slot) in zip(data["model_parameters"], routes, strict=True)]}
           if routes else {}),
        "minimum_cycles": min_cycles,
    }
    provider_identity = make_identity(
        "bound-amr-refinement-criterion",
        semantic_value(payload, where="bound AMR refinement criterion"),
    ).token
    sim._set_generated_tagging(
        data["identity"],
        [row["state"] for row in fields],
        [row["block"] for row in fields],
        [row["variables"] for row in fields],
        [row["components"] for row in fields],
        [row["reach"] for row in fields],
        data["parameters"],
        [index for index, _slot in routes],
        [slot for _index, slot in routes],
        min_cycles,
        clock_identity,
        provider_identity,
    )
    if criterion.decision is not None:
        _flow_regrid_decision(sim, criterion.decision, params)


def _flow_regrid_decision(sim: Any, decision: Any, params: Any) -> None:
    """Install the tables of the regrid decision generated into the compiled Program (DM13b)."""
    data = decision.runtime_decision_data(params)
    if (
        type(data) is not dict
        or data.get("schema_version") != 1
        or data.get("decision_type") != "amr_regrid_decision_runtime"
        or data.get("identity") != decision.qualified_id
    ):
        raise ValueError("pops.bind: regrid decision returned an unsupported runtime manifest")
    fields = data.get("fields")
    if not isinstance(fields, list):
        raise ValueError("pops.bind: a regrid decision has a list of fields")
    sim._set_generated_regrid_decision(
        data["identity"],
        [row["state"] for row in fields],
        [row["block"] for row in fields],
        [row["variables"] for row in fields],
        [row["components"] for row in fields],
        data["parameters"],
        data["indicator_count"],
    )


__all__ = [
    "amr_config_from_layout",
    "flow_bootstrap_tagging",
    "system_config_from_layout",
]
