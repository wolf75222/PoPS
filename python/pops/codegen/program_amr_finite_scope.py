"""Invocation leases for finite values across native AMR scheduling barriers."""
from __future__ import annotations


def uses_finite_values(program):
    from .program_lowerability import all_ops
    return any(value.op.startswith(("finite_", "distributed_", "continuation_"))
               for value in all_ops(program))


def scope_body(source, enabled):
    if not source or not enabled:
        return source
    return "auto _finite_value_scope = ctx.finite_value_scope();\n" + source


def continuation_scope_capture(capture, var):
    if not var.get(("amr_finite_scope",)):
        return capture
    return capture + ", _finite_value_lease = ctx.current_finite_value_lease()"


def resume_scope(lines, var):
    if var.get(("amr_finite_scope",)):
        lines.append("auto _finite_value_scope = ctx.resume_finite_value_scope(_finite_value_lease);")
