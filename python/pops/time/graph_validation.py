"""Recursive validation for immutable ProgramGraph nodes and structured regions."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.time.points import Clock


def _nested_regions(node: Any) -> tuple[Any, ...]:
    from pops.time.graph import Branch, Loop, Region

    if type(node) is Branch:
        return tuple(
            arm for arm in (node.when_true, node.when_false) if type(arm) is Region)
    if type(node) is Loop:
        return ((node.condition,) if node.condition is not None else ()) + (node.body,)
    return ()


def _validate_region_boundary(
        region: Any, available: Mapping[int, Any], declared: set[Clock], where: str) -> None:
    if not set(region.clocks).issubset(declared):
        raise ValueError("%s region clock is not declared by its enclosing graph" % where)
    for capture in region.captures:
        source = available.get(capture.value.node_id)
        if source is None or getattr(source, "readable", True) is False:
            raise ValueError(
                "%s region capture %d must name an available readable outer node"
                % (where, capture.value.node_id))
        if source.clock != capture.clock or source.point != capture.point:
            raise ValueError(
                "%s region capture %d clock/point metadata does not match its outer value"
                % (where, capture.value.node_id))


def validate_nodes(nodes: Any, clocks: Any, available: dict[int, Any], *, where: str) -> None:
    from pops.time.graph import Branch, Loop, Synchronize, _point_clocks

    declared = set(clocks)
    for node in nodes:
        direct_ids = set()
        if type(node) is Branch:
            direct_ids = {node.state.node_id, node.condition.node_id}
        elif type(node) is Loop:
            direct_ids = {node.initial.node_id}
        if node.node_id in available:
            raise ValueError("%s node ids must be unique across captures and local nodes" % where)
        if node.clock not in declared:
            raise ValueError("%s node clock is not declared" % where)
        if _point_clocks(node.point) != frozenset((node.clock,)):
            raise ValueError("%s node point must use exactly the node clock" % where)
        for ref in node.references():
            source = available.get(ref.node_id)
            if source is None:
                raise ValueError(
                    "%s reference %d must name an earlier readable node or explicit capture"
                    % (where, ref.node_id))
            if getattr(source, "readable", True) is False:
                raise ValueError("%s cannot read a Commit node" % where)
            if type(node) is Synchronize:
                if source.clock != node.source_clock:
                    raise ValueError("Synchronize source_clock does not match its input value")
            elif ref.node_id in direct_ids or type(node) not in (Branch, Loop):
                if source.clock == node.clock:
                    continue
                raise ValueError(
                    "cross-clock read %s -> %s requires an explicit Synchronize node"
                    % (source.clock.name, node.clock.name))
        for index, region in enumerate(_nested_regions(node)):
            _validate_region_boundary(
                region, available, declared,
                "%s/%s[%d]" % (where, node.kind, index))
        available[node.node_id] = node


__all__ = ["validate_nodes"]
