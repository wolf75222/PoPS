"""Exact Program field-route and logical-time lowering helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pops.time.references import canonical_handle


def resolved_field_route(field_ref: Any, field_plans: Any) -> tuple[str, Any]:
    """Resolve one exact Program field identity to its authenticated install slot."""
    if not isinstance(field_plans, Mapping) or not field_plans:
        raise ValueError("field solve codegen requires resolved field install plans")
    canonical = canonical_handle(field_ref)
    from pops.problem.handles import FieldHandle
    if not isinstance(canonical, FieldHandle):
        raise TypeError(
            "Program field solves require the case-owned FieldHandle returned by Case.field(...)")
    plan = field_plans.get(canonical.local_id)
    if plan is None:
        raise ValueError(
            "Program field %r has no resolved field install plan" % canonical.local_id)
    return plan.native_options["provider_slot"], plan


def validate_program_field_routes(program: Any, field_plans: Any) -> None:
    """Authenticate every Program field solve during ``resolve``.

    Compile is a total projection of a resolved plan, so it must never discover
    a missing/ambiguous field installation or repair stale output metadata.
    """
    if program is None:
        return
    roots = list(getattr(program, "_values", ()) or ())
    dt_bound = getattr(program, "_dt_bound", None)
    if dt_bound is not None:
        roots.extend(dt_bound[0])
    for node in _walk_program_nodes(roots):
        if getattr(node, "op", None) not in ("solve_fields", "solve_fields_from_blocks"):
            continue
        attrs = getattr(node, "attrs", {})
        field_ref = attrs.get("field") if isinstance(attrs, Mapping) else None
        if field_ref is None:
            raise ValueError("Program field solve node has no exact FieldHandle identity")
        _, plan = resolved_field_route(field_ref, field_plans)
        context = getattr(node, "field_context", None)
        if context is None:
            raise ValueError("Program field solve node has no FieldContext provenance")
        if canonical_handle(context.field).canonical_identity() != canonical_handle(
                field_ref).canonical_identity():
            raise ValueError("Program field solve FieldContext disagrees with its FieldHandle")
        expected_outputs = tuple(plan.native_options["output_route"]["components"])
        if tuple(context.outputs) != expected_outputs:
            raise ValueError(
                "Program field %r outputs %r disagree with resolved components %r"
                % (plan.name, tuple(context.outputs), expected_outputs))
        space_outputs = tuple(getattr(getattr(node, "space", None), "components", ()))
        if space_outputs != expected_outputs:
            raise ValueError(
                "Program field %r value space %r disagrees with resolved components %r"
                % (plan.name, space_outputs, expected_outputs))


def _walk_program_nodes(values: Any) -> Any:
    for node in values:
        yield node
        attrs = getattr(node, "attrs", {})
        if not isinstance(attrs, Mapping):
            continue
        for key in (
            "cond_block", "body_block", "true_block", "false_block",
            "apply_block", "residual_block",
        ):
            nested = attrs.get(key)
            if isinstance(nested, (list, tuple)):
                yield from _walk_program_nodes(nested)


def field_point_cpp(program: Any, value: Any, slot: str) -> list[str]:
    """Materialize the solve node's exact logical TimePoint before native iteration."""
    from pops.time import StagePoint, TimePoint

    point = value.point
    partition_slot = 0
    if type(point) is StagePoint:
        candidates = [(name, item) for name, item in point.partitions.items()
                      if item.clock == value.clock]
        if len(candidates) != 1:
            try:
                point = point.time
            except ValueError as exc:
                raise ValueError(
                    "field solve at a partitioned StagePoint requires one coordinate on its clock"
                ) from exc
        else:
            partition, point = candidates[0]
            partition_slot = 1 + sorted(
                point_name for point_name in value.point.partitions).index(partition)
    if type(point) is not TimePoint:
        raise TypeError("field solve requires an exact TimePoint or StagePoint")
    clocks = sorted({item.clock.qualified_id for item in program._values})
    clock_slot = clocks.index(point.clock.qualified_id)
    coordinate = "(%d + %s)" % (point.step, point.offset.to_cpp())
    token = "field_point_%d" % value.id
    return [
        "pops::FieldLogicalTimePoint %s;" % token,
        "%s.time = ctx.physical_time() + (%s) * dt;" % (token, coordinate),
        "%s.dt = dt;" % token,
        "%s.clock_slot = %d;" % (token, clock_slot),
        "%s.partition_slot = %d;" % (token, partition_slot),
        "%s.stage_slot = %d;" % (token, value.id),
        "%s.step = ctx.macro_step() + %d;" % (token, point.step),
        "%s.substep = 0;" % token,
        "%s.iteration = 0;" % token,
        "ctx.set_field_logical_timepoint(%s, %s);" % (json.dumps(slot), token),
    ]
