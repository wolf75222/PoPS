"""Detach an authoring :class:`Program` into its immutable inspectable ProgramGraph snapshot."""
from __future__ import annotations

from typing import Any


def _name(value: Any) -> str:
    return value.name or "%s_%d" % (value.op, value.id)


def _semantic_attrs(program: Any, value: Any) -> dict[str, Any]:
    """Return the complete canonical serialized metadata not represented by graph fields."""
    row = program._serialize_node(value, include_provenance=False)
    return {
        key: item
        for key, item in row.items()
        if key not in {"id", "name", "vtype", "op", "inputs", "point"}
    }


def _declared_clocks(nodes: Any, primary: Any) -> tuple[Any, ...]:
    clocks = [primary]
    pending = list(nodes)
    while pending:
        node = pending.pop(0)
        candidates = [node.clock]
        if node.kind == "synchronize":
            candidates.append(node.source_clock)
        if node.kind == "branch":
            for arm in (node.when_true, node.when_false):
                candidates.extend(arm.clocks)
                pending.extend(getattr(arm, "nodes", ()))
        elif node.kind == "loop":
            regions = ((node.condition,) if node.condition is not None else ()) + (node.body,)
            for region in regions:
                candidates.extend(region.clocks)
                pending.extend(region.nodes)
        for clock in candidates:
            if clock not in clocks:
                clocks.append(clock)
    return tuple(clocks)


def program_to_graph(program: Any) -> Any:
    """Return a registry-free immutable graph without mutating the authoring Program.

    The conversion first uses the canonical compiled detachment boundary. Every readable SSA value
    keeps its authoring id, edges become ``ValueRef`` records, semantic handles/attrs are captured as
    canonical data, and commits become new write-only nodes. No Program, registry, endpoint, temporal
    handle, callback, or mutable attrs container is retained by the result.
    """
    from pops.time.program import Program

    if type(program) is not Program:
        raise TypeError("program_to_graph requires an exact pops.time.Program")
    if getattr(program, "_recording", ()):
        raise RuntimeError("cannot snapshot a Program while authoring a nested region")
    if getattr(program, "_compiled_detached", False):
        detached = program
    else:
        from pops.time.program_detach import detach_compiled_program

        detached = detach_compiled_program(program)

    from pops.time.graph import (
        Branch,
        Commit,
        Loop,
        OperatorCall,
        ProgramGraph,
        ProgramValue as GraphValue,
        ResidualEvaluation,
        ResidualSolve,
        Region,
        RegionCapture,
        Solve,
        StateRead,
        Synchronize,
        Unknown,
        ValueRef,
    )
    from pops.time.points import TimePoint
    from pops.time.references import handle_data

    all_values = list(detached._values)
    for value in detached._values:
        all_values.extend(detached._subblock_value_refs(value))
    next_id = max((value.id for value in all_values), default=-1) + 1

    def region_values(block: Any, result: Any) -> tuple[Any, ...]:
        """Outer values imported by a recorded region, including nested-control captures."""
        local_ids = {value.id for value in block}
        captures: dict[int, Any] = {}

        def add(candidate: Any) -> None:
            if candidate.id not in local_ids:
                captures.setdefault(candidate.id, candidate)

        for value in block:
            for candidate in value.inputs:
                add(candidate)
            for key in ("cond_block", "body_block", "true_block", "false_block"):
                nested = value.attrs.get(key)
                if nested is None:
                    continue
                result_key = {
                    "cond_block": "cond", "body_block": "body",
                    "true_block": "true_result", "false_block": "false_result",
                }[key]
                for candidate in region_values(nested, value.attrs[result_key]):
                    add(candidate)
        add(result)
        return tuple(captures[key] for key in sorted(captures))

    def result_signature(result: Any) -> dict[str, Any]:
        return {
            "value_type": result.vtype,
            "space": result.space.to_data() if result.space is not None else None,
            "block": result.block.inspect() if result.block is not None else None,
        }

    def convert_region(block: Any, result: Any, name: str,
                       *, signature: Any = None) -> Any:
        captures = tuple(
            RegionCapture(
                ValueRef(value.id), value.clock, value.point,
                signature=result_signature(value))
            for value in region_values(block, result)
        )
        region_nodes = convert_values(block)
        clocks = list(_declared_clocks(
            region_nodes,
            captures[0].clock if captures else detached.clock,
        ))
        for capture in captures:
            if capture.clock not in clocks:
                clocks.append(capture.clock)
        return Region(
            name, captures, region_nodes, ValueRef(result.id), clocks=clocks,
            result_signature=signature)

    def convert_value(value: Any) -> tuple[Any, ...]:
        nonlocal next_id
        prefix = ()
        inputs = tuple(ValueRef(item.id) for item in value.inputs)
        if value.op == "branch":
            signature = result_signature(value)
            when_true = convert_region(
                value.attrs["true_block"], value.attrs["true_result"],
                "%s/true" % _name(value), signature=signature)
            when_false = convert_region(
                value.attrs["false_block"], value.attrs["false_result"],
                "%s/false" % _name(value), signature=signature)
            return (Branch(
                value.id,
                inputs[0],
                when_true,
                when_false,
                value.clock,
                value.point,
                name=_name(value),
                result_signature=signature,
            ),)
        if value.op == "range":
            body = convert_region(
                value.attrs["body_block"], value.attrs["body"], "%s/body" % _name(value))
            return (Loop(
                value.id,
                "range",
                inputs[0],
                body,
                value.clock,
                value.point,
                count=value.attrs["count"],
                name=_name(value),
            ),)
        if value.op == "while":
            condition = convert_region(
                value.attrs["cond_block"], value.attrs["cond"], "%s/condition" % _name(value))
            body = convert_region(
                value.attrs["body_block"], value.attrs["body"], "%s/body" % _name(value))
            return (Loop(
                value.id,
                "while",
                inputs[0],
                body,
                value.clock,
                value.point,
                condition=condition,
                name=_name(value),
            ),)
        attrs = _semantic_attrs(detached, value)
        if value.op == "state":
            state = value.state_ref
            if state is None:
                raise ValueError("Program state node %d has no qualified state identity" % value.id)
            node = StateRead(
                value.id,
                handle_data(state),
                value.clock,
                value.point,
                name=_name(value),
                metadata=attrs,
            )
        elif value.op == "unknown":
            node = Unknown(
                value.id,
                _name(value),
                attrs.get("space", {"kind": value.vtype}),
                value.clock,
                value.point,
            )
        elif value.op == "synchronize":
            if len(inputs) != 1:
                raise ValueError("Program synchronize node must have exactly one input")
            node = Synchronize(
                value.id,
                inputs[0],
                value.inputs[0].clock,
                value.clock,
                value.attrs["relation"],
                value.point,
                name=_name(value),
            )
        elif value.op == "residual_eval":
            node = ResidualEvaluation(
                value.id,
                value.attrs["operator"],
                inputs,
                value.clock,
                value.point,
                name=_name(value),
                attrs=attrs,
            )
        elif value.op == "solve_residual":
            if len(inputs) < 2:
                raise ValueError("solve_residual graph conversion expects residual and initial")
            node = ResidualSolve(
                value.id,
                inputs[0],
                inputs[1:],
                value.clock,
                value.point,
                name=_name(value),
                attrs=attrs,
            )
        elif value.op in (
                "solve_fields", "solve_fields_from_blocks", "solve_local_linear",
                "solve_local_nonlinear", "solve_coupled_implicit"):
            # Solve tokens remain generic unreadable ProgramValue graph nodes until an explicit
            # solve_outcome consumes them.  Their operator handle is metadata, not an ordinary
            # readable OperatorCall result.
            node = GraphValue(
                value.id, _name(value), value.vtype, value.op, inputs,
                value.clock, value.point, attrs=attrs)
        elif "operator_handle" in value.attrs:
            node = OperatorCall(
                value.id,
                {
                    "handle": handle_data(value.attrs["operator_handle"]),
                    "lowering": {"op": value.op, "value_type": value.vtype, **attrs},
                },
                inputs,
                value.clock,
                value.point,
                name=_name(value),
            )
        elif value.op == "solve_linear":
            if len(inputs) not in (2, 3):
                raise ValueError("solve_linear graph conversion expects operator, rhs[, guess]")
            if len(inputs) == 3:
                unknown = inputs[2]
            else:
                unknown_node = Unknown(
                    next_id,
                    "%s_unknown" % _name(value),
                    attrs.get("space", {"kind": value.vtype}),
                    value.clock,
                    value.point,
                )
                prefix = (unknown_node,)
                unknown = ValueRef(next_id)
                next_id += 1
            if len(inputs) == 3:
                prefix = ()
            node = Solve(
                value.id,
                unknown,
                inputs[0],
                inputs[1],
                value.clock,
                value.point,
                name=_name(value),
                attrs=attrs,
            )
        else:
            node = GraphValue(
                value.id,
                _name(value),
                value.vtype,
                value.op,
                inputs,
                value.clock,
                value.point,
                attrs=attrs,
            )
        return (*prefix, node)

    def convert_values(values: Any) -> list[Any]:
        converted = []
        for value in values:
            converted.extend(convert_value(value))
        return converted

    nodes = convert_values(detached._values)

    for state_ref, value in sorted(
            detached._commits.items(), key=lambda item: item[0].qualified_id):
        point = TimePoint(value.clock, step=1)
        nodes.append(Commit(
            next_id,
            {
                "state": handle_data(state_ref),
                "block": handle_data(state_ref.block_ref),
                "endpoint": "next",
                "point": point.to_data(),
            },
            ValueRef(value.id),
            value.clock,
            point,
        ))
        next_id += 1

    graph = ProgramGraph(
        detached.name,
        nodes,
        clocks=_declared_clocks(nodes, detached.clock),
    )
    # Detachment and graph conversion are read-only; authoring identity remains stable.
    if detached._ir_hash() != program._ir_hash():
        raise RuntimeError("Program.to_graph changed authoring IR identity")
    return graph


__all__ = ["program_to_graph"]
