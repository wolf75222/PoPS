"""Recursive validation for immutable temporal graph nodes and regions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pops.time._graph.base import point_clocks
from pops.time.points import Clock


def _nested_regions(node: Any) -> tuple[Any, ...]:
    if node.kind == "branch":
        return (node.when_true, node.when_false)
    if node.kind == "loop":
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


def _solve_outcome_action(attrs: Mapping[str, Any]) -> None:
    from pops.time.solve_outcome import SOLVE_STATUSES

    attrs = _payload_attrs(attrs)
    action = attrs.get("action")
    if not isinstance(action, Mapping):
        raise ValueError(
            "solve_outcome requires explicit action=FailRun(...) or RejectAttempt(...)")
    if action.get("kind") not in ("fail_run", "reject_attempt"):
        raise ValueError("solve_outcome action must be fail_run or reject_attempt")
    statuses = action.get("statuses")
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)) or not statuses:
        raise ValueError("solve_outcome action statuses must be a non-empty sequence")
    unknown = tuple(status for status in statuses if status not in SOLVE_STATUSES)
    if unknown:
        raise ValueError("solve_outcome action has unknown status(es): %s" % ", ".join(unknown))


def _payload_attrs(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    if "attrs" in attrs and isinstance(attrs["attrs"], Mapping):
        return attrs["attrs"]
    return attrs


def _int_attr(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        scalar = value.get("scalar")
        if isinstance(scalar, Mapping) and scalar.get("kind") == "integer":
            raw = scalar.get("value")
            if isinstance(raw, str):
                return int(raw)
    return None


def _solve_arity(node: Any) -> int:
    if node.kind == "solve":
        return 1
    if node.kind == "residual_solve":
        attrs = _payload_attrs(node.attrs.to_data())
        return _int_attr(attrs.get("unknown_count")) or len(node.initial)
    if node.kind == "program_value" and node.op in (
            "solve_fields", "solve_fields_from_blocks"):
        return 1
    if node.kind == "program_value" and node.op in (
            "solve_local_linear", "solve_local_nonlinear"):
        return 1
    if node.kind == "program_value" and node.op == "solve_coupled_implicit":
        attrs = _payload_attrs(node.attrs.to_data())
        return _int_attr(attrs.get("output_count")) or 0
    raise TypeError("expected a solve graph node")


def _is_solve_token(node: Any) -> bool:
    kind = getattr(node, "kind", None)
    return (kind in ("solve", "residual_solve")
            or (kind == "program_value"
                and node.op in (
                    "solve_fields", "solve_fields_from_blocks", "solve_local_linear",
                    "solve_local_nonlinear", "solve_coupled_implicit")))


def validate_nodes(nodes: Any, clocks: Any, available: dict[int, Any], *, where: str) -> None:
    declared = set(clocks)
    local_solves: dict[int, Any] = {}
    consumed_solves: dict[int, int] = {}
    for node in nodes:
        direct_ids = set()
        if node.kind == "branch":
            direct_ids = {node.condition.node_id}
        elif node.kind == "loop":
            direct_ids = {node.initial.node_id}
        if node.node_id in available:
            raise ValueError("%s node ids must be unique across captures and local nodes" % where)
        if _is_solve_token(node):
            local_solves[node.node_id] = node
        if node.clock not in declared:
            raise ValueError("%s node clock is not declared" % where)
        if point_clocks(node.point) != frozenset((node.clock,)):
            raise ValueError("%s node point must use exactly the node clock" % where)
        for ref in node.references():
            source = available.get(ref.node_id)
            if source is None:
                raise ValueError(
                    "%s reference %d must name an earlier readable node or explicit capture"
                    % (where, ref.node_id))
            if getattr(source, "readable", True) is False:
                raise ValueError("%s cannot read a Commit node" % where)
            if _is_solve_token(source) \
                    and not (node.kind == "program_value" and node.op == "solve_outcome"):
                raise ValueError(
                    "%s cannot read an unconsumed solve token; call outcome.consume(action=...)"
                    % where)
            if node.kind == "synchronize":
                if source.clock != node.source_clock:
                    raise ValueError("Synchronize source_clock does not match its input value")
            elif ref.node_id in direct_ids or node.kind not in ("branch", "loop"):
                if source.clock == node.clock:
                    continue
                raise ValueError(
                    "cross-clock read %s -> %s requires an explicit Synchronize node"
                    % (source.clock.name, node.clock.name))
        if node.kind == "branch":
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
        if node.kind == "residual_solve":
            residual_source = available[node.residual.node_id]
            if getattr(residual_source, "kind", None) != "residual_evaluation":
                raise ValueError(
                    "ResidualSolve residual must reference a ResidualEvaluation node")
            if len(node.initial) != len(residual_source.unknowns):
                raise ValueError(
                    "ResidualSolve initial product arity must match residual unknown product")
            attrs = _payload_attrs(node.attrs.to_data())
            unknown_count = _int_attr(attrs.get("unknown_count"))
            if "unknown_count" in attrs and unknown_count != len(node.initial):
                raise ValueError(
                    "ResidualSolve unknown_count must match initial product arity")
        if node.kind == "residual_evaluation":
            components = _residual_operator_unknowns(node)
            if len(components) != len(node.unknowns):
                raise ValueError(
                    "ResidualEvaluation unknown product arity must match operator unknown_space")
        if node.kind == "program_value" and node.op == "solve_outcome":
            if len(node.inputs) != 1:
                raise ValueError("solve_outcome must consume exactly one solve token")
            source = available[node.inputs[0].node_id]
            if not _is_solve_token(source):
                raise ValueError(
                        "solve_outcome must consume an executable solve token")
            solve_id = node.inputs[0].node_id
            consumed_solves[solve_id] = consumed_solves.get(solve_id, 0) + 1
            _solve_outcome_action(node.attrs.to_data())
        if node.kind == "program_value" and node.op == "solve_outcome_component":
            if len(node.inputs) != 1:
                raise ValueError("solve_outcome_component must read exactly one solve_outcome")
            source = available[node.inputs[0].node_id]
            if not (getattr(source, "kind", None) == "program_value"
                    and source.op == "solve_outcome"):
                raise ValueError("solve_outcome_component must read a consumed solve_outcome")
            attrs = _payload_attrs(node.attrs.to_data())
            index = _int_attr(attrs.get("index"))
            solve_source = available[source.inputs[0].node_id]
            if index is None or index < 0 or index >= _solve_arity(solve_source):
                raise ValueError("solve_outcome_component index is outside solve outcome arity")
        for index, region in enumerate(_nested_regions(node)):
            _validate_region_boundary(
                region, available, declared,
                "%s/%s[%d]" % (where, node.kind, index))
        available[node.node_id] = node
    for solve_id, solve in local_solves.items():
        count = consumed_solves.get(solve_id, 0)
        if count != 1:
            raise ValueError(
                "%s solve token %r must be consumed exactly once with an explicit action; got %d"
                % (where, getattr(solve, "name", solve_id), count))


__all__ = ["validate_nodes"]
