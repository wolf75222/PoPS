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
    AccumulateDt, AtStart, Error, Every, Hold, Skip, When, Zero,
)


def _schedule_due_test(program: Any, v: Any, sched: Any, var: Any = None) -> str:
    """The C++ boolean 'is this node due this step' for a non-subcycle schedule kind. Reused as the
    guard of the policy branch. Raises (naming ADC-458) for a kind that needs a runtime primitive the
    compiled .so does not have (on_end: no end-of-run signal reaches a sim.step(dt) loop)."""
    trigger = sched.trigger
    if type(trigger) is Every:
        # Cadence: due cold-start, then every N macro-steps (CacheManager::is_due via macro_step()).
        return "ctx.cache_should_update(%d, %d)" % (v.id, trigger.n)
    if type(trigger) is AtStart:
        return "(ctx.macro_step() == 0)"
    if type(trigger) is When:
        # A runtime predicate: a Program Bool value already lowered to a parenthesized C++ expr token
        # (a compare over reductions). A bare Python callable cannot lower (it is not a Program value).
        cond = trigger.condition
        if not (isinstance(cond, ProgramValue) and cond.vtype == "bool"):
            raise NotImplementedError(
                "when(cond) lowers only a Program Bool predicate (e.g. P.norm2(r) < tol), not a "
                "Python callable: node %r (ADC-458). Build the condition with Program compares."
                % v.name)
        tokens = var if var is not None else {}
        key = ("when_predicate", cond.id)
        if key not in tokens:
            raise ValueError(
                "when(cond) on node %r references a Bool value not emitted before it; build the "
                "predicate earlier in the Program (ADC-458)" % v.name)
        return tokens[key]
    raise NotImplementedError(
        "schedule trigger %s on node %r is not lowerable: AtEnd needs an end-of-run signal that a "
        "compiled sim.step(dt) loop never sees (the .so cannot know the last step); use on_start()/"
        "Every/When or a later ConsumerGraph host hook." % (type(trigger).__name__, v.name))

def _emit_schedule_wrap(program: Any, v: Any, var: Any, lines: Any, start: Any) -> None:
    """Wrap the C++ statements node @p v emitted (``lines[start:]``) in its schedule's due-test guard
    + policy branch (ADC-458, Spec 3 sections 17-18). Generic over the op: a field solve caches the
    System aux, any other node caches its named scratch (var[v.id]). An always()/absent schedule
    leaves the lines untouched (byte-identical to the unscheduled lowering)."""
    sched = v.attrs.get("schedule")
    if sched is None or sched.is_always():
        return
    body = lines[start:]
    del lines[start:]
    due = _schedule_due_test(program, v, sched, var)
    policy = sched.off
    is_aux = v.op in _AUX_OUTPUT_OPS
    # The scratch node's output token (the MultiFab the policy holds / zeroes). A field solve writes
    # the System aux and sets no var[v.id], so out is read only on the scratch path.
    out = None if is_aux else var.get(v.id)
    if policy is None:
        # Run only when due; on a NOT-due step do nothing (the aux / scratch keeps its last content).
        # recompute off-cadence is simply 'run when due' -- no cache, no else branch. A scratch node
        # hoists its output declaration so the buffer stays in scope when the body does not run.
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("}")
        return
    if type(policy) is Skip:
        # Do not run the op off-cadence: the value keeps its previous content (the cacheable contract
        # -- downstream must tolerate a stale value). A scratch node hoists its output declaration so
        # the stale buffer stays in scope across the guard (no else branch: nothing happens off-
        # cadence); a field solve writes the persistent aux, so its whole body simply guards.
        if is_aux:
            lines.append("if (%s) {  // skip: stale aux off-cadence" % due)
            lines += ["  " + ln for ln in body]
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {  // skip: stale value off-cadence" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("}")
        return
    if type(policy) is Zero:
        # Off-cadence, zero the node's output. The output must EXIST in both branches: for a scratch
        # node hoist its allocation out of the guard (the first emitted line declares var[v.id]); the
        # aux always exists (System-owned).
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("} else {")
            lines.append("  ctx.aux().set_val(static_cast<pops::Real>(0));")
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("} else {")
            lines.append("  %s.set_val(static_cast<pops::Real>(0));" % out)
            lines.append("}")
        return
    if type(policy) is Hold:
        # Recompute + cache when due; restore the cached value off-cadence (no recompute). The aux
        # path uses cache_store_aux/restore_aux; a scratch node hoists its allocation and uses the
        # named-scratch cache. _validate_schedule already rejected hold on a non-cacheable operator.
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("  ctx.cache_store_aux(%d);" % v.id)
            lines.append("} else {")
            lines.append("  ctx.cache_restore_aux(%d);" % v.id)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
            lines.append("} else {")
            lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
            lines.append("}")
        return
    if type(policy) is AccumulateDt:
        # Off-cadence: accumulate THIS step's dt (the real skipped dt, never N*dt_current) and hold the
        # cached value. When due: read eff_dt = dt + sum(skipped) (resets the accumulator), recompute,
        # cache. eff_dt is bound so a dt-dependent recompute can read it (the MVP field solve / scratch
        # fill is dt-free, but eff_dt is exposed for a dt-scaled body). Cacheable (validated upstream).
        ed = "_effdt%d" % v.id
        if is_aux:
            lines.append("if (%s) {" % due)
            lines.append("  const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                         % (ed, v.id, ed))
            lines += ["  " + ln for ln in body]
            lines.append("  ctx.cache_store_aux(%d);" % v.id)
            lines.append("} else {")
            lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
            lines.append("  ctx.cache_restore_aux(%d);" % v.id)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines.append("  const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                         % (ed, v.id, ed))
            lines += ["  " + ln for ln in rest]
            lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
            lines.append("} else {")
            lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
            lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
            lines.append("}")
        return
    if type(policy) is Error:
        # Guard that a stale value is never read off-cadence: run when due, else fail loud (the node
        # asserts it is only consumed on its cadence). Emitted as a runtime throw on the not-due path.
        # A scratch node hoists its output declaration so the buffer stays in scope (the throw never
        # returns, but the C++ must still be well-scoped); a field solve guards the aux body directly.
        err = ('ctx.scheduler_error(%s);'
               % json.dumps("node '%s' (op '%s') read off its schedule cadence (policy=error)"
                            % (v.name, v.op)))
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("} else {")
            lines.append("  " + err)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("} else {")
            lines.append("  " + err)
            lines.append("}")
        return
    raise NotImplementedError(
        "schedule off-policy %s on node %r is not lowerable"
        % (type(policy).__name__, v.name))

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
            % (v.name, v.op, out, body[0] if body else None))
    return body[0], body[1:]
