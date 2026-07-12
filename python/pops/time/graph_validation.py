"""Recursive validation for immutable ProgramGraph nodes and structured regions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _residual_operator_unknowns(node: Any) -> tuple[str, ...]:
    data = node.operator.to_data()
    if data.get("kind") != "residual_operator":
        raise ValueError("ResidualEvaluation operator must be a residual_operator descriptor")
    unknown_space = data.get("unknown_space")
    if not isinstance(unknown_space, Mapping):
        raise ValueError("ResidualEvaluation operator must declare an unknown_space")
    components = unknown_space.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)) \
            or not components:
        raise ValueError("ResidualEvaluation operator unknown_space must be a non-empty tuple")
    return tuple(components)


def validate_nodes(nodes: Any, clocks: Any, available: dict[int, Any], *, where: str) -> None:
    from pops.time.graph import (
        Branch, Loop, ResidualEvaluation, ResidualSolve, Synchronize, _point_clocks,
    )

    declared = set(clocks)
    for node in nodes:
        direct_ids = set()
        if type(node) is Branch:
            direct_ids = {node.condition.node_id}
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
        if type(node) is Branch:
            condition_source = available[node.condition.node_id]
            condition_signature = getattr(condition_source, "signature", None)
            if condition_signature is not None:
                condition_signature = condition_signature.to_data()
            is_bool = (
                getattr(condition_source, "value_type", None) == "bool"
                or (condition_signature or {}).get("value_type") == "bool"
                or getattr(
                    getattr(condition_source, "result_signature", None),
                    "to_data", lambda: {})().get("value_type") == "bool")
            if not is_bool:
                raise ValueError("Branch condition must reference a scalar Bool graph value")
        if type(node) is ResidualSolve:
            residual_source = available[node.residual.node_id]
            if type(residual_source) is not ResidualEvaluation:
                raise ValueError(
                    "ResidualSolve residual must reference a ResidualEvaluation node")
            if len(node.initial) != len(residual_source.unknowns):
                raise ValueError(
                    "ResidualSolve initial product arity must match residual unknown product")
            attrs = node.attrs.to_data()
            if "unknown_count" in attrs and attrs["unknown_count"] != len(node.initial):
                raise ValueError(
                    "ResidualSolve unknown_count must match initial product arity")
        if type(node) is ResidualEvaluation:
            components = _residual_operator_unknowns(node)
            if len(components) != len(node.unknowns):
                raise ValueError(
                    "ResidualEvaluation unknown product arity must match operator unknown_space")
        for index, region in enumerate(_nested_regions(node)):
            _validate_region_boundary(
                region, available, declared,
                "%s/%s[%d]" % (where, node.kind, index))
        available[node.node_id] = node


__all__ = ["validate_nodes"]
