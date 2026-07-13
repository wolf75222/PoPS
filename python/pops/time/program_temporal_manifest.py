"""Canonical temporal execution contract derived from one authored Program.

The manifest is intentionally data-only.  It is persisted next to native history/cache payloads so
restart can authenticate not merely the Program hash, but the exact logical clocks, nested cadence,
cross-clock providers and history validity contract needed for the next attempt.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator

from pops.time.points import Clock
from pops.time.references import block_name, handle_data, state_name


def _walk(values: Any) -> Iterator[Any]:
    for value in values:
        yield value
        for key in (
            "cond_block", "body_block", "true_block", "false_block",
            "apply_block", "residual_block",
        ):
            nested = value.attrs.get(key)
            if nested:
                yield from _walk(nested)


def _clock_data(clock: Clock, ticks_per_macro: int) -> dict[str, Any]:
    return {
        "id": clock.qualified_id,
        "descriptor": clock.to_data(),
        "ticks_per_macro": ticks_per_macro,
    }


def _history_policy(program: Any, name: str) -> Any:
    configured = getattr(program, "_history_persistence", {}).get(name)
    return configured[1].to_manifest() if configured is not None else None


def build_temporal_manifest(program: Any) -> dict[str, Any]:
    """Build and validate the exact nested-clock execution schedule for ``program``."""
    from pops.time.program_serialization import _json_ready

    nodes = tuple(_walk(program._values))
    clocks = {program.clock}
    subcycles = []
    synchronizations = []
    schedules = []
    parents: dict[Clock, tuple[Clock, int]] = {}

    for value in nodes:
        clocks.add(value.clock)
        if value.op == "subcycle":
            child = value.clock
            parent = value.attrs["parent_clock"]
            clocks.add(parent)
            count = int(value.attrs["count"])
            prior = parents.get(child)
            relation = (parent, count)
            if prior is not None and prior != relation:
                raise ValueError(
                    "clock %r has conflicting parent/count subcycle declarations" % child.name)
            parents[child] = relation
            subcycles.append({
                "node_id": value.id,
                "parent_clock": parent.qualified_id,
                "child_clock": child.qualified_id,
                "count": count,
            })
        elif value.op == "synchronize":
            source = value.inputs[0].clock
            clocks.add(source)
            synchronizations.append({
                "node_id": value.id,
                "source_clock": source.qualified_id,
                "target_clock": value.clock.qualified_id,
                "relation": value.attrs["relation"],
                "point": value.point.to_data(),
            })
        schedule = value.attrs.get("schedule")
        if schedule is not None:
            clocks.add(schedule.clock)
            schedules.append({
                "node_id": value.id,
                "schedule": schedule.to_data(),
                "cache_required": bool(schedule.needs_cache()),
            })

    for state in getattr(program, "_time_states", {}).values():
        clocks.add(state.clock)

    ticks = {program.clock: 1}
    visiting: set[Clock] = set()

    def resolve(clock: Clock) -> int:
        known = ticks.get(clock)
        if known is not None:
            return known
        if clock in visiting:
            raise ValueError("nested clock schedule contains a cycle at %r" % clock.name)
        relation = parents.get(clock)
        if relation is None:
            raise ValueError(
                "clock %r has no subcycle relation to primary clock %r"
                % (clock.name, program.clock.name))
        visiting.add(clock)
        parent, count = relation
        result = resolve(parent) * count
        visiting.remove(clock)
        ticks[clock] = result
        return result

    for clock in tuple(clocks):
        resolve(clock)

    history_clocks: dict[str, set[Clock]] = defaultdict(set)
    for value in nodes:
        name = value.attrs.get("history")
        if isinstance(name, str):
            history_clocks[name].add(value.clock)
    for state in getattr(program, "_time_states", {}).values():
        name = "%s.%s" % (block_name(state.block), state_name(state.state))
        if state in getattr(program, "_time_history_configs", {}):
            history_clocks[name].add(state.clock)

    histories = []
    for name, depth in sorted(program._histories.items()):
        ring_clocks = history_clocks.get(name, set())
        if len(ring_clocks) != 1:
            raise ValueError(
                "history %r must belong to exactly one logical clock, got %s"
                % (name, sorted(clock.name for clock in ring_clocks)))
        clock = next(iter(ring_clocks))
        state = program._history_state_refs.get(name)
        owner = program._history_blocks.get(name)
        space = program._history_spaces.get(name)
        histories.append({
            "name": name,
            "owner": handle_data(owner) if owner is not None else None,
            "state": (handle_data(state) if state is not None
                      else {"kind": "scalar_history", "qualified_id": "scalar-history:" + name}),
            "space": space.to_data() if space is not None else {"kind": "scalar_field"},
            "clock": clock.qualified_id,
            "depth": int(depth),
            "ring_slots": int(depth) + 1,
            "ncomp": program._histories_ncomp.get(name),
            "validity": {
                "domain": "accepted_clock_ticks",
                "newest_lag": 0,
                "oldest_lag": int(depth),
            },
            "interpolation": {
                "provider": "exact",
                "schema_version": 1,
                "dense_output": False,
            },
            "checkpoint_policy": _history_policy(program, name),
        })

    return _json_ready({
        "schema_version": 1,
        "kind": "pops.temporal-program-schedule",
        "primary_clock": program.clock.qualified_id,
        "clocks": [
            _clock_data(clock, ticks[clock])
            for clock in sorted(clocks, key=lambda item: item.qualified_id)
        ],
        "subcycles": sorted(subcycles, key=lambda item: item["node_id"]),
        "synchronizations": sorted(synchronizations, key=lambda item: item["node_id"]),
        "schedules": sorted(schedules, key=lambda item: item["node_id"]),
        "histories": histories,
    })


__all__ = ["build_temporal_manifest"]
