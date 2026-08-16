"""Lower exact mesh plans onto native runtime configuration seams.

This module contains only backend lowering.  Runtime selection lives in
``_runtime_executor`` and therefore never depends on historical target strings.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI
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


def flow_bootstrap_tagging(
    sim: Any,
    bootstrap: Any,
    params: Any,
    *,
    clock_identity: str,
    field_plans: Any = None,
) -> None:
    """Compile one authenticated tagging graph to the native data-only VM."""
    if not isinstance(clock_identity, str) or not clock_identity:
        raise ValueError("pops.bind: AMR tagging requires one exact clock identity")
    data = bootstrap.tagging.runtime_tagging_data(params)
    if (
        type(data) is not dict
        or data.get("schema_version") != 1
        or data.get("graph_type") != "amr_tagging_runtime"
    ):
        raise ValueError("pops.bind: tagging provider returned an unsupported runtime manifest")

    registrations = {}
    for row in data.get("lowerings", ()):
        if type(row) is not dict or row.get("schema_version") != 1:
            raise ValueError("pops.bind: malformed tagging lowering registration")
        node_type = row.get("node_type")
        lowering = row.get("lowering", {})
        if (
            not isinstance(node_type, str)
            or not node_type
            or lowering.get("kind") != "tag_lowering"
            or lowering.get("local_id") != node_type
        ):
            raise ValueError("pops.bind: unauthenticated tagging lowering registration")
        if node_type in registrations:
            raise ValueError("pops.bind: duplicate tagging lowering registration")
        registrations[node_type] = lowering.get("qualified_id")

    resolved_field_plans = field_plans if isinstance(field_plans, Mapping) else {}
    field_plans_by_identity: dict[str, Any] = {}
    for field_name, plan in resolved_field_plans.items():
        unknown = getattr(getattr(plan, "operator", None), "unknown", None)
        identity = getattr(unknown, "qualified_id", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError(
                "pops.bind: resolved field plan %r has no qualified solved-field identity"
                % field_name
            )
        if identity in field_plans_by_identity:
            raise ValueError(
                "pops.bind: multiple resolved field plans claim solved-field identity %s" % identity
            )
        field_plans_by_identity[identity] = plan
    leaves: list[tuple[str, str, str, str, int, int, float, int]] = []
    stencils: list[dict[str, Any]] = []
    stencil_indices: dict[str, int] = {}

    def compile_node(node: Any) -> tuple[list[int], list[int]]:
        if type(node) is not dict or node.get("schema_version") != 1:
            raise ValueError("pops.bind: malformed tagging expression node")
        node_type = node.get("node_type")
        if node_type not in registrations:
            raise ValueError("pops.bind: tagging node lacks an authenticated lowering")
        leaf_op = _TAG_LEAF_OPS.get(node_type)
        if leaf_op is not None:
            indicator = node.get("indicator")
            if type(indicator) is not dict or indicator.get("kind") not in {"state", "field"}:
                raise TypeError("pops.bind: native tag leaves require a state/field Handle")
            subject_kind = indicator["kind"]
            subject_identity = indicator.get("qualified_id")
            if not isinstance(subject_identity, str) or not subject_identity:
                raise ValueError(
                    "pops.bind: native tag leaves require a qualified subject identity"
                )
            variable = node.get("variable", indicator.get("local_id"))
            threshold = node.get("threshold")
            if (
                not isinstance(variable, str)
                or not variable
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
            ):
                raise TypeError("pops.bind: malformed native tag leaf")
            block_name = ""
            field_component_index = -1
            if subject_kind == "state":
                block = indicator.get("block_ref")
                if type(block) is not dict or not isinstance(block.get("local_id"), str):
                    raise ValueError("pops.bind: native state tag leaves must be block-qualified")
                block_name = block["local_id"]
            else:
                plan = field_plans_by_identity.get(subject_identity)
                if plan is None:
                    raise ValueError(
                        "pops.bind: native field tag leaf has no authenticated field plan"
                    )
                options = getattr(plan, "native_options", None)
                output = options.get("output_route") if isinstance(options, Mapping) else None
                components = output.get("components") if isinstance(output, Mapping) else None
                if not isinstance(components, (list, tuple)) or components.count(variable) != 1:
                    raise ValueError(
                        "pops.bind: native field tag leaf is absent from its prepared output route"
                    )
                field_component_index = components.index(variable)
            stencil_index = -1
            if node_type in {"gradient_above", "gradient_below"}:
                context = node.get("discrete_context")
                lowering_data = (
                    context.get("stencil_lowering") if isinstance(context, dict) else None
                )
                from pops.numerics.indicator_stencils import DiscreteGradientStencil

                lowering = DiscreteGradientStencil.from_data(lowering_data)
                canonical = lowering.to_data()
                identity = lowering.identity
                stencil_index = stencil_indices.get(identity, -1)
                if stencil_index < 0:
                    stencil_index = len(stencils)
                    stencil_indices[identity] = stencil_index
                    stencils.append(canonical)
                elif stencils[stencil_index] != canonical:
                    raise ValueError(
                        "pops.bind: AMR stencil identity collision changed coefficients"
                    )
            leaves.append(
                (
                    subject_kind,
                    subject_identity,
                    block_name,
                    variable,
                    field_component_index,
                    leaf_op,
                    float(threshold),
                    stencil_index,
                )
            )
            return [leaf_op], [len(leaves) - 1]

        logical_op = _TAG_LOGICAL_OPS.get(node_type)
        if logical_op is None:
            raise NotImplementedError(
                "pops.bind: native tagging provider %r is registered but has no VM opcode"
                % node_type
            )
        children = node.get("children")
        if node_type == "not":
            children = (node.get("child"),)
        if not isinstance(children, (list, tuple)) or not children:
            raise ValueError("pops.bind: logical tagging node has no children")
        ops: list[int] = []
        args: list[int] = []
        for child in children:
            child_ops, child_args = compile_node(child)
            ops.extend(child_ops)
            args.extend(child_args)
        ops.append(logical_op)
        args.append(len(children))
        return ops, args

    refine_ops, refine_args = compile_node(data["refine"])
    coarsen = data.get("coarsen")
    coarsen_ops, coarsen_args = ([], []) if coarsen is None else compile_node(coarsen)
    hysteresis = data.get("hysteresis")
    if type(hysteresis) is not dict or hysteresis.get("hysteresis_type") != "min_cycles":
        raise ValueError("pops.bind: unsupported AMR hysteresis manifest")
    min_cycles = hysteresis.get("min_cycles")
    if isinstance(min_cycles, bool) or not isinstance(min_cycles, int) or min_cycles < 0:
        raise ValueError("pops.bind: AMR hysteresis min_cycles must be an integer >= 0")
    from pops.identity import make_identity
    from pops.identity.semantic import semantic_value

    program_payload = {
        "schema_version": 1,
        "program_type": "bound_amr_tagging_program",
        "resolved_graph_identity": bootstrap.tagging.qualified_id,
        "stencils": stencils,
        "leaves": [
            {
                "subject_kind": kind,
                "subject_identity": identity,
                "block": block,
                "variable": variable,
                "field_component_index": component_index,
                "opcode": opcode,
                "threshold": threshold,
                "stencil_index": stencil_index,
            }
            for (
                kind,
                identity,
                block,
                variable,
                component_index,
                opcode,
                threshold,
                stencil_index,
            ) in leaves
        ],
        "refine_opcodes": refine_ops,
        "refine_arguments": refine_args,
        "coarsen_opcodes": coarsen_ops,
        "coarsen_arguments": coarsen_args,
        "minimum_cycles": min_cycles,
        "equality_policy": str(hysteresis.get("equality")),
        "conflict_policy": str(data.get("conflict_policy")),
    }
    program_identity = make_identity(
        "bound-amr-tagging-program",
        semantic_value(program_payload, where="bound AMR tagging program"),
    ).token
    sim._set_bootstrap_tagging(
        [row[0] for row in leaves],
        [row[1] for row in leaves],
        [row[2] for row in leaves],
        [row[3] for row in leaves],
        [row[4] for row in leaves],
        [row[5] for row in leaves],
        [row[6] for row in leaves],
        [row[7] for row in leaves],
        stencils,
        refine_ops,
        refine_args,
        coarsen_ops,
        coarsen_args,
        min_cycles,
        str(hysteresis.get("equality")),
        str(data.get("conflict_policy")),
        clock_identity,
        program_identity,
    )


# The opcode table is generated from the versioned component catalog and shared with the C ABI.
# The compiler dispatches only through this data; no Python class-name switch reaches the hot loop.
_TAG_LEAF_OPS = dict(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"])
_TAG_LOGICAL_OPS = dict(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])


__all__ = [
    "amr_config_from_layout",
    "flow_bootstrap_tagging",
    "system_config_from_layout",
]
