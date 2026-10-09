"""Exact coarse/fine RHS input authority after cell-local source transformations."""
from __future__ import annotations

from typing import Any


def _is_additive_stage_evaluation(value: Any) -> bool:
    """Recognize the exact paired coordinates also used by ARK evaluation lowering."""
    from pops.time.points import StagePoint

    point = value.point
    return (type(point) is StagePoint
            and set(point.partitions) == {"explicit", "implicit"}
            and all(coordinate.clock == value.clock for coordinate in point.partitions.values())
            and value.attrs.get("evaluation_partition") in (None, "explicit", "implicit"))


def requires_rhs_input_trace(value: Any) -> bool:
    """Select source-transformed states, leaving ordinary transport subcycling intact.

    This is a dependency property of the RHS input, not a moment-closure or
    electromagnetic model selector. The native trace then authenticates the
    exact SSA evaluation and the common clock window across adjacent levels.
    """
    program = value.prog
    transformed = {"affine_moment_update", "condensed_reconstruct", "local_transform"}
    local_source = {
        "source", "implicit_source", "solve_local_linear", "solve_local_nonlinear",
        "solve_implicit_source", "solve_coupled_implicit",
    }
    pending = [value.inputs[0]]
    seen = set()
    while pending:
        node = program._canonical_value(pending.pop())
        if id(node) in seen:
            continue
        seen.add(id(node))
        if (node.op in transformed
                or (node.op in local_source and not _is_additive_stage_evaluation(node))):
            from pops.time._program.value_validation import TOP_LEVEL_REGION

            if value.region != TOP_LEVEL_REGION or value.attrs.get("schedule") is not None:
                raise ValueError(
                    "AMR source-transformed RHS input traces require an unscheduled "
                    "top-level SSA evaluation")
            return True
        # An additive stage solve uses the existing subcycled stage-parent route.
        # Its ancestors still matter: a preceding composition transform must keep
        # the exact synchronous source-first trace, even through an IMEX stage.
        pending.extend(node.inputs)
        pending.extend(program._subblock_value_refs(node))
    return False


def emit_rhs_input_trace(value: Any, block: int, state: str, lines: list[str], target: str,
                         *, force: bool = False) -> str | None:
    if target != "amr_system" or not (requires_rhs_input_trace(value) or force):
        return None
    if force:
        from pops.time._program.value_validation import TOP_LEVEL_REGION
        if value.region != TOP_LEVEL_REGION or value.attrs.get("schedule") is not None:
            raise ValueError("path RHS traces require unscheduled top-level stage inputs")
    name = "rhs_input_trace_%d" % value.id
    source_id = value.prog._canonical_value(value.inputs[0]).id
    lines.append("auto %s = ctx.capture_rhs_input_trace(%d, %s, %d, %d);"
                 % (name, block, state, value.id, source_id))
    return "&" + name
