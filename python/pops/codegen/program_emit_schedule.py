"""pops.codegen.program_emit_schedule : the unified scheduler wrap (ADC-458).

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  ``_emit_schedule_wrap`` wraps the statements a node
emitted in its schedule's due-test guard + policy branch; ``program_emit_ops._emit_op``
calls it after each op lowers itself.  ``_schedule_due_test`` / ``_split_output_decl`` are
its helpers.  Reuses the op tables in ``program_emit_kernels``.
"""

from __future__ import annotations

import json
from typing import Any

from pops.codegen.program_emit_kernels import _AUX_OUTPUT_OPS, ProgramValue
from pops.time.schedule import (
    Schedule,
    ScheduleAction,
    ScheduleComment,
    ScheduleDueKind,
    ScheduleLoweringIR,
    ScheduleTimeline,
)


def _lower_schedule_ir(v: Any, sched: Any) -> ScheduleLoweringIR:
    """Invoke the nominal schedule extension point and validate its exact return contract."""
    if not isinstance(sched, Schedule):
        raise TypeError(
            "schedule on node %r must implement the Schedule interface; got %s"
            % (v.name, type(sched).__name__)
        )
    where = "node %r (op '%s')" % (v.name, v.op)
    if not hasattr(v, "clock") or not hasattr(v, "point"):
        raise TypeError("%s lacks its exact clock/point schedule site" % where)
    sched.validate_site(clock=v.clock, point=v.point, where="schedule on %s" % where)
    lowered = sched.native_schedule_ir(where=where)
    if type(lowered) is not ScheduleLoweringIR:
        raise TypeError(
            "Schedule.native_schedule_ir() for %s must return an exact ScheduleLoweringIR, got %s"
            % (where, type(lowered).__name__)
        )
    if lowered.domain.timeline is ScheduleTimeline.STAGE:
        from pops.time.points import StagePoint
        if type(v.point) is not StagePoint:
            raise ValueError("stage schedule on %s is not attached to an exact StagePoint" % where)
        site_identity = json.dumps(
            v.point.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        if lowered.domain.stage_identity != site_identity:
            raise ValueError(
                "stage schedule on %s does not authenticate the node's exact StagePoint" % where)
    if lowered.due.kind is ScheduleDueKind.AT_END:
        raise NotImplementedError(
            "schedule AtEnd on %s is not lowerable: a compiled sim.step(dt) loop never sees an "
            "end-of-run signal, so the .so cannot know the last step; use AtStart/Every/When on "
            "AcceptedStep, or an AtEnd ConsumerGraph hook" % where
        )
    if lowered.due.kind is ScheduleDueKind.PROGRAM_PREDICATE:
        condition = lowered.due.predicate
        if not (isinstance(condition, ProgramValue) and condition.vtype == "bool"):
            raise NotImplementedError(
                "when(cond) lowers only a Program Bool predicate (e.g. P.norm2(r) < tol), not a "
                "Python callable: node %r (ADC-458). Build the condition with Program compares."
                % v.name
            )
    return lowered


def _schedule_domain_args(domain: Any) -> str:
    kinds = {
        ScheduleTimeline.ACCEPTED_STEP: "AcceptedStep",
        ScheduleTimeline.STAGE: "Stage",
        ScheduleTimeline.CLOCK_TICK: "ClockTick",
        ScheduleTimeline.AMR_LEVEL: "AmrLevel",
    }
    try:
        kind = kinds[domain.timeline]
    except KeyError:
        raise NotImplementedError(
            "native schedule timeline %s is not supported" % domain.timeline.value) from None
    return "%s, %s, %s, %d" % (
        "pops::runtime::program::ScheduleDomainKind::k" + kind,
        json.dumps(domain.clock_id),
        json.dumps(domain.stage_identity or ""),
        -1 if domain.level is None else domain.level,
    )


def _schedule_due_expression(v: Any, lowering: Any, var: Any = None) -> str:
    """Render one validated due-test IR without inspecting its concrete Trigger class."""
    due = lowering.due
    domain = _schedule_domain_args(lowering.domain)
    if due.kind is ScheduleDueKind.ALWAYS:
        return "ctx.schedule_domain_occurs(%s)" % domain
    if due.kind is ScheduleDueKind.CACHE_PERIOD:
        return "ctx.schedule_is_due(%d, %d, %s)" % (v.id, due.period, domain)
    if due.kind is ScheduleDueKind.MACRO_STEP_ZERO:
        return "ctx.schedule_at_start(%s)" % domain
    if due.kind is ScheduleDueKind.PROGRAM_PREDICATE:
        # A runtime predicate: a Program Bool value already lowered to a parenthesized C++ expr token
        # (a compare over reductions). A bare Python callable cannot lower (it is not a Program value).
        cond = due.predicate
        tokens = var if var is not None else {}
        key = ("when_predicate", cond.id)
        if key not in tokens:
            raise ValueError(
                "when(cond) on node %r references a Bool value not emitted before it; build the "
                "predicate earlier in the Program (ADC-458)" % v.name
            )
        return "(ctx.schedule_domain_occurs(%s) && (%s))" % (domain, tokens[key])
    raise NotImplementedError(
        "native schedule due primitive %s on node %r is not supported" % (due.kind.value, v.name)
    )


def _schedule_due_test(program: Any, v: Any, sched: Any, var: Any = None) -> str:
    """The C++ boolean 'is this node due this step' for a native schedule."""
    del program
    return _schedule_due_expression(v, _lower_schedule_ir(v, sched), var)


def _schedule_action_line(action: ScheduleAction, *, v: Any, out: Any, is_aux: bool) -> str:
    """Render one validated schedule action; concrete policies never inject C++ text."""
    if action is ScheduleAction.EFFECTIVE_DT:
        effective_dt = "_effdt%d" % v.id
        return "const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;" % (
            effective_dt,
            v.id,
            effective_dt,
        )
    if action is ScheduleAction.STORE:
        if is_aux:
            return "ctx.cache_store_aux(%d);" % v.id
        return "ctx.cache_store_scratch(%d, %s);" % (v.id, out)
    if action is ScheduleAction.ZERO:
        if is_aux:
            return "ctx.aux().set_val(static_cast<pops::Real>(0));"
        return "%s.set_val(static_cast<pops::Real>(0));" % out
    if action is ScheduleAction.ACCUMULATE_DT:
        return "ctx.cache_accumulate_dt(%d, dt);" % v.id
    if action is ScheduleAction.RESTORE:
        if is_aux:
            return "ctx.cache_restore_aux(%d);" % v.id
        return "ctx.cache_restore_scratch(%d, %s);" % (v.id, out)
    if action is ScheduleAction.ERROR:
        return "ctx.scheduler_error(%s);" % json.dumps(
            "node '%s' (op '%s') read off its schedule cadence (policy=error)" % (v.name, v.op)
        )
    raise NotImplementedError(
        "native schedule action %s on node %r is not supported" % (action.value, v.name)
    )


def _emit_schedule_wrap(program: Any, v: Any, var: Any, lines: Any, start: Any) -> None:
    """Wrap the C++ statements node @p v emitted (``lines[start:]``) in its schedule's due-test guard
    + policy branch (ADC-458, Spec 3 sections 17-18). Generic over the op: a field solve caches the
    System aux, any other node caches its named scratch (var[v.id]). An always()/absent schedule
    leaves the lines untouched (byte-identical to the unscheduled lowering)."""
    sched = v.attrs.get("schedule")
    if sched is None:
        return
    lowering = _lower_schedule_ir(v, sched)
    if (lowering.due.kind is ScheduleDueKind.ALWAYS
            and lowering.domain.timeline is ScheduleTimeline.ACCEPTED_STEP):
        return
    body = lines[start:]
    del lines[start:]
    due = _schedule_due_expression(v, lowering, var)
    policy = lowering.off
    is_aux = v.op in _AUX_OUTPUT_OPS
    # The scratch node's output token (the MultiFab the policy holds / zeroes). A field solve writes
    # the System aux and sets no var[v.id], so out is read only on the scratch path.
    out = None if is_aux else var.get(v.id)
    if is_aux:
        guarded_body = body
    else:
        decl, guarded_body = _split_output_decl(program, body, out, v)
        lines.append(decl)
    comment = ""
    if policy.comment is ScheduleComment.SKIP:
        comment = "  // skip: stale %s off-cadence" % ("aux" if is_aux else "value")
    lines.append("if (%s) {%s" % (due, comment))
    for action in policy.before_due:
        lines.append("  " + _schedule_action_line(action, v=v, out=out, is_aux=is_aux))
    lines += ["  " + line for line in guarded_body]
    for action in policy.after_due:
        lines.append("  " + _schedule_action_line(action, v=v, out=out, is_aux=is_aux))
    if policy.off_cadence:
        lines.append("} else {")
        for action in policy.off_cadence:
            lines.append("  " + _schedule_action_line(action, v=v, out=out, is_aux=is_aux))
    lines.append("}")


def _split_output_decl(program: Any, body: Any, out: Any, v: Any) -> tuple:
    """Split a scratch node's emitted @p body into (declaration_line, rest): the OUTPUT scratch
    ``out`` must be declared OUTSIDE the policy guard so both branches see it, while the fill stays
    inside. The op declares its output as its FIRST emitted line (``pops::MultiFab <out> = ...;``);
    hoist exactly that one line. Raises if the shape is unexpected (a node whose output is not a
    freshly-declared scratch cannot use a cache/zero policy through this path)."""
    decl_prefix = "pops::MultiFab %s = " % out
    if not body or not body[0].startswith(decl_prefix):
        raise NotImplementedError(
            "schedule policy on node %r (op '%s') needs its output scratch %r declared as its first "
            "emitted line to hoist it out of the guard; got %r (ADC-458)"
            % (v.name, v.op, out, body[0] if body else None)
        )
    return body[0], body[1:]
