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
    if isinstance(canonical, FieldHandle):
        plan = field_plans.get(canonical.local_id)
        if plan is None:
            raise ValueError(
                "Program field %r has no resolved field install plan" % canonical.local_id)
        return plan.native_options["provider_slot"], plan

    matches = []
    for plan in field_plans.values():
        for provider in plan.rhs_providers:
            declaration = provider.declaration_ref or provider
            if canonical_handle(declaration).canonical_identity() == canonical.canonical_identity():
                matches.append(plan)
                break
    if len(matches) != 1:
        raise ValueError(
            "field operator route must match exactly one resolved FieldHandle plan, got %d"
            % len(matches))
    return matches[0].native_options["provider_slot"], matches[0]


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
