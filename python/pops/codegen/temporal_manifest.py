"""Codegen-time temporal manifest authenticated by the persistent resource plan.

``pops.time`` owns the authoring manifest for clocks, schedules and histories.  The
compiler adds the sealed resource table to that data-only contract before it is
embedded in a candidate descriptor.  Keeping this adapter in :mod:`pops.codegen`
avoids changing the authoring manifest while making the exact slot plan part of
the artifact identity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pops.codegen.program_persistent_plan import (
    ProgramResourcePlan,
    get_program_resource_plan,
    iter_program_occurrences,
)


def _manifest_json_ready(value: Any) -> Any:
    """Close over extension values left in a temporal manifest payload.

    Schedule ``when`` payloads may retain their predicate as a ProgramValue in
    ``Schedule.to_data()``.  The authoring manifest historically passed those
    values through its internal JSON helper, but the compiler-owned wrapper
    must authenticate the final document itself.  Keep the same compact
    predicate identity used by schedule serialization and reject no qualified
    semantics here.
    """

    from pops.time.values import ProgramValue

    if isinstance(value, ProgramValue):
        return {"program_value_id": value.id}
    if isinstance(value, Mapping):
        return {key: _manifest_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_manifest_json_ready(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":")))
    return value


def build_temporal_manifest(
    program: Any,
    *,
    target: str | None = None,
    plan: ProgramResourcePlan | None = None,
) -> dict[str, Any]:
    """Return the canonical temporal manifest plus its sealed resource plan.

    The authoring clock/history rows remain unchanged except for the compiler-owned
    ``cache_slots`` schedule annotation.  ``resource_plan`` carries both the
    complete lowering rows and the authenticated digest, so a restart or host can
    reject a candidate whose slot layout differs from the authored graph.
    """

    if plan is None:
        plan = get_program_resource_plan(program, target=target)
    if type(plan) is not ProgramResourcePlan:
        raise TypeError("temporal manifest requires an exact ProgramResourcePlan")
    if target is not None and target not in {"system", "amr_system"}:
        raise ValueError("temporal manifest target must be 'system' or 'amr_system'")
    from pops.time._program.temporal_manifest import build_temporal_manifest as build_base

    base = build_base(program)
    if not isinstance(base, Mapping):
        raise TypeError("temporal authoring manifest must return a mapping")
    result = _manifest_json_ready(dict(base))

    # ``node_id`` is a control label, not a persistent-resource identity.  The
    # compiler has already lowered every occurrence through the complete key
    # ``(value_id, occurrence_path, owner, space, clock, level)``.  Resolve the
    # schedule rows against that occurrence path and expose *all* level-expanded
    # slots so a cache cursor never has to guess from a node id.
    occurrences = tuple(iter_program_occurrences(program))
    by_node: dict[int, list[tuple[Any, str]]] = {}
    for value, path in occurrences:
        value_id = getattr(value, "id", None)
        if isinstance(value_id, bool) or not isinstance(value_id, int):
            raise TypeError("temporal manifest occurrence ids must be exact integers")
        by_node.setdefault(value_id, []).append((value, path))
    schedules = []
    for index, row in enumerate(result.get("schedules", ())):
        if not isinstance(row, Mapping):
            raise TypeError("temporal schedule %d must be a mapping" % index)
        node_id = row.get("node_id")
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError(
                "temporal schedule %d node_id must be a non-negative integer" % index
            )
        matches = by_node.get(node_id, ())
        if len(matches) != 1:
            raise ValueError(
                "temporal schedule %d cannot resolve node_id %r to one complete lowering occurrence"
                % (index, node_id)
            )
        _value, occurrence_path = matches[0]
        entries = tuple(
            entry
            for entry in plan
            if entry.key.value_id == node_id
            and entry.key.occurrence_path == occurrence_path
        )
        cache_required = row.get("cache_required")
        if type(cache_required) is not bool:
            raise TypeError("temporal schedule cache_required must be bool")
        cache_slots = []
        if cache_required:
            # Resolve each optional level through the plan's public exact-key
            # lookup.  This deliberately rejects a forged plan containing two
            # owner/space/clock rows for one node and level instead of silently
            # choosing one from a node-id-only filter.
            levels = []
            for entry in entries:
                if entry.key.level not in levels:
                    levels.append(entry.key.level)
            for level in levels:
                try:
                    cache_slots.append(
                        plan.slot_for_value(
                            _value, occurrence_path=occurrence_path, level=level
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "temporal schedule node_id %r has ambiguous complete lowering slots"
                        % node_id
                    ) from error
            cache_slots.sort()
        if cache_required and not cache_slots:
            raise ValueError(
                "temporal schedule node_id %r has cache authority but no complete lowering slots"
                % node_id
            )
        schedules.append({**dict(row), "cache_slots": cache_slots})
    result["schedules"] = schedules
    result["resource_plan"] = plan.to_data()
    result["resource_plan_schema"] = plan.schema
    result["resource_plan_digest"] = plan.digest
    result["resource_plan_maximum_bytes"] = plan.maximum_bytes
    # Keep one exact codegen schema even when a caller does not select a target
    # yet.  Runtime installation supplies ``system`` or ``amr_system`` and the
    # restart validator authenticates that value before accepting the schedule.
    result["codegen_target"] = target
    return result


def render_temporal_manifest(
    program: Any,
    *,
    target: str | None = None,
    plan: ProgramResourcePlan | None = None,
) -> str:
    """Render :func:`build_temporal_manifest` deterministically for C++ embedding."""

    return json.dumps(
        build_temporal_manifest(program, target=target, plan=plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "build_temporal_manifest",
    "render_temporal_manifest",
]
