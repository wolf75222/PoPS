"""Generate complete synchronized operations within native continuation regions.

No operation is executed by Python: its existing C++ emitter supplies every
gather, numerical solve, and publication statement.
"""
from __future__ import annotations

from typing import Any


def is_path_rhs(value: Any) -> bool:
    return value.op == "rhs" and value.attrs.get("path_conservative", False)


def is_spatial_solve(value: Any) -> bool:
    return value.op == "solve_spatial_nonlinear"


def is_closed_interaction(value: Any) -> bool:
    return value.op == "spatial_interaction" and value.attrs.get("contract") == "pops.spatial-interaction@3"


def is_operation_barrier(value: Any) -> bool:
    return is_path_rhs(value) or is_spatial_solve(value) or is_closed_interaction(value)


def validate_operation_barrier(program: Any, value: Any) -> None:
    if is_path_rhs(value):
        if (not value.attrs.get("flux") or value.attrs.get("fluxes")
                or tuple(value.attrs.get("sources", ()))
                or value.attrs.get("schedule") is not None):
            raise ValueError("path RHS must retain one complete unscheduled flux/product balance")
    elif is_closed_interaction(value):
        from pops.time._program.spatial_interaction import interaction_contract
        interaction_contract(value)
    elif is_spatial_solve(value):
        from pops.time._program.spatial_solve import validate_spatial_request, validate_spatial_commit

        validate_spatial_request(program, value)
        validate_spatial_commit(program, value)
    else:
        raise ValueError("operation has no synchronized hierarchy continuation")


def emit_operation_barrier(program: Any, value: Any, base: Any, completed: Any,
                           var: Any, model: Any, lines: list[str], prelude: list[str],
                           block_idx: Any, field_plans: Any, *,
                           committed_ids=frozenset(), shared_interface_jvp=False) -> None:
    """Leave the generated resume callback open; the caller owns its lexical join."""
    from .program_emit_hierarchy_regions import open_hierarchy_continuation
    from .program_emit_ops import _emit_op

    validate_operation_barrier(program, value)

    def emit(environment, output, resources):
        _emit_op(program, value, base, committed_ids, environment, model, output,
                 resources, block_idx, target="amr_system", field_plans=field_plans,
                 has_shared_interface_implicit_jacvec=shared_interface_jvp)

    if is_closed_interaction(value):
        # The closed source spans the whole accepted hierarchy. Its gather/kernel
        # must execute once after all levels reach the same exact SSA consumer.
        solve = dict(var)
        solve[("original_field_hierarchy_phase",)] = "solve"
        once: list[str] = []
        emit(solve, once, [])
        var.pop(value.id, None)
        open_hierarchy_continuation(program, value, completed, var, lines, once, kind="control")
        observe = dict(var)
        observe[("original_field_hierarchy_phase",)] = "observe"
        emit(observe, lines, [])
        var[value.id] = observe[value.id]
        return

    if is_path_rhs(value):
        emit(var, lines, prelude)
        open_hierarchy_continuation(
            program, value, [*completed, value], var, lines,
            ["ctx.publish_staged_path_rhs(%d);" % value.id], kind="spatial_rhs")
        return

    # Installed resources are captured by shared owner. Per-level predecessor and
    # residual callbacks are staged anew for each invocation; the root callback
    # solves only after every exact level envelope reached this barrier.
    resources = dict(var)
    resources[("spatial_hierarchy_phase",)] = "resources"
    emit(resources, [], prelude)
    var[("spatial_hierarchy_phase",)] = "gather"
    emit(var, lines, prelude)
    solve = dict(var)
    solve[("spatial_hierarchy_phase",)] = "solve"
    once: list[str] = []
    emit(solve, once, [])
    var.pop(("spatial_hierarchy_phase",), None)
    # The solved Field alias does not exist until publication in the resume
    # callback. Capturing that unborn C++ local would be a dangling/invalid bind.
    var.pop(value.id, None)
    open_hierarchy_continuation(program, value, completed, var, lines, once,
                                kind="nonlinear_solve")
    var[("spatial_hierarchy_phase",)] = "publish"
    emit(var, lines, [])
    var.pop(("spatial_hierarchy_phase",), None)
