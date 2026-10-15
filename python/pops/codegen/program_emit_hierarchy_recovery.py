"""Recovery and bounded iteration as ordinary generated C++ continuations."""
from __future__ import annotations

import json

from pops.time._program.structured_control import scalar_signature


def _install(value, output, frame, join, available, var, lines):
    from .program_emit_hierarchy_control import _capture
    from .program_amr_finite_scope import resume_scope
    contract = "pops.hierarchy.recovery-node@1:" + value.prog._ir_hash() + ":" + str(value.id)
    lines.extend(("ctx.push_hierarchy_failure_handler(%d, %s, %s," %
        (value.id, json.dumps(value.prog._ir_hash()), json.dumps(contract)),
        "[%s](std::exception_ptr failure) {" % _capture(available, var),
        "auto& ctx = *ctx_owner;"))
    resume_scope(lines, var)
    lines.extend(("try { std::rethrow_exception(failure); }",
        "catch (const pops::runtime::program::StepAttemptRejected& rejected) {",
        "const auto words = pops::runtime::program::hierarchy_control_failure_words(rejected);",
        "%s = words.first;" % output["code"], "%s = words.second;" % output["origin"], "}",
        "%s(%s);" % (join, frame), "});"))


def _enter(program, value, available, var, lines):
    from .program_emit_hierarchy_regions import open_hierarchy_continuation
    open_hierarchy_continuation(program, value, available, var, lines, [], kind="control")


def _finish(program, value, available, var, lines, join, frame, *, promotion=False):
    from .program_emit_hierarchy_regions import open_hierarchy_continuation
    from .program_emit_hierarchy_control import _normal_close
    once = ["ctx.close_hierarchy_failure_handlers(%d);" % value.id]
    if promotion:
        from .program_emit_versioned import _checked
        once = []
        _checked(once, "hierarchy_promotion_vote_%d" % value.id, "bool",
                 ["ctx.validate_hierarchy_scope_promotions(%d);" % value.id], "true",
                 message="accepted hierarchy child promotion failed")
        once.append("ctx.commit_hierarchy_scope_promotions(%d);" % value.id)
    open_hierarchy_continuation(program, value, available, var, lines, once, kind="control")
    lines.append("%s(%s);" % (join, frame))
    _normal_close(lines)


def open_capture(program, value, available, var, model, lines, prelude, block_idx, field_plans):
    if value.vtype == "variant":
        from .program_emit_hierarchy_typed_recovery import open_typed_capture
        return open_typed_capture(program, value, available, var, model, lines, prelude,
                                  block_idx, field_plans)
    from .program_emit_hierarchy_control import (
        _heap_slots, _capture, _open_join, _close_join, _normal_close, emit_region,
    )
    from .program_emit_structured_control import _assign_output
    if value.vtype != "result" or value.attrs.get("continuation_scope"):
        raise NotImplementedError("hierarchy capture requires complete typed payload recovery transport")
    signature = {"vtype": "result", "success": value.attrs["success_signature"]}
    frame_type, frame, output = _heap_slots(signature, value.id, "hierarchy_capture", lines)
    var[value.id] = output
    outer = dict(var)
    join = "hierarchy_capture_done_%d" % value.id
    _open_join(join, _capture(available, var), frame_type, frame, lines)

    def close(target):
        _close_join(join, target)
        _install(value, output, frame, join, available, outer, target)
        _enter(program, value, available, outer, target)
        sub = dict(outer)
        sub[("hierarchy_failure_capture",)] = True

        def terminal(current, body):
            source = current[value.attrs["body"].id]
            if value.attrs.get("body_is_result"):
                _assign_output(signature, output, source, body)
            else:
                _assign_output(signature["success"], output["value"], source, body)
            _finish(program, value, available, current, body, join, frame)

        emit_region(program, value.attrs["body_block"], sub, model, target, prelude,
            block_idx, field_plans, terminal=terminal, available=available)
        _normal_close(target)
    return close


def open_iterate(program, value, available, var, model, lines, prelude, block_idx, field_plans):
    from .program_emit_hierarchy_control import (
        _heap_slots, _capture, _open_join, _close_join, _normal_close, emit_region,
    )
    from .program_emit_structured_control import _copy, _uniform_int, _storage_width
    if value.attrs.get("error_signature") is not None:
        from .program_emit_hierarchy_typed_recovery import open_typed_iterate
        return open_typed_iterate(program, value, available, var, model, lines, prelude,
                                  block_idx, field_plans)
    state_signature = value.attrs["state_signature"]
    signature = {"vtype": "product", "members": (
        ("state", state_signature), ("code", scalar_signature("int64")),
        ("origin", scalar_signature("int64")))}
    frame_type, frame, output = _heap_slots(signature, value.id, "hierarchy_iterate", lines, carry=True)
    _, _, pending = _heap_slots(state_signature, value.id, "hierarchy_iterate_next", lines,
        carry=True, slot_offset=_storage_width(state_signature))
    _copy(state_signature, output["state"], var[value.inputs[0].id], lines)
    var[value.id] = {**output, "iterations": frame+"->iteration"}
    outer = dict(var)
    budget = var[value.inputs[1].id]
    _uniform_int(budget, lines, "hierarchy iterate budget differs across execution lane")
    join = "hierarchy_iterate_done_%d" % value.id
    _open_join(join, _capture(available, var), frame_type, frame, lines)

    def close(target):
        _close_join(join, target)
        weak = frame+"_weak"
        target.extend(("std::weak_ptr<%s> %s = %s;" % (frame_type, weak, frame),
            "std::exception_ptr %s_repeat_error;" % frame, "try {",
            "%s->repeat = [%s, retained_weak = %s]() {" % (frame, _capture(available, outer), weak),
            "auto& ctx = *ctx_owner;", "auto %s = retained_weak.lock();" % frame,
            'if (!%s) throw std::logic_error("expired hierarchy iteration continuation");' % frame,
            "if (%s < 0) {" % budget, "%s = 2;" % output["code"], "%s = 2;" % output["origin"]))
        _finish(program, value, available, outer, target, join, frame)
        target.append("}")
        condition = dict(outer)
        condition[("hierarchy_failure_capture",)] = True
        condition[value.attrs["cond_argument"].id] = output["state"]

        def condition_done(current, body):
            choice = "hierarchy_iteration_stop_%d" % value.id
            body.append("const long %s = (%s) ? 1L : 0L;" % (choice, current[value.attrs["cond"].id]))
            _uniform_int(choice, body, "hierarchy iterate stop differs across execution lane")
            body.append("if (%s != 0) {" % choice)
            _finish(program, value, available, current, body, join, frame)
            body.append("}")
            body.extend(("if (%s->iteration == %s) {" % (frame, budget),
                "%s = 1;" % output["code"], "%s = 2;" % output["origin"]))
            _finish(program, value, available, current, body, join, frame)
            body.append("}")
            step = dict(current)
            step[value.attrs["body_argument"].id] = output["state"]

            def step_done(completed, rows):
                source = completed[value.attrs["body"].id]
                if value.attrs.get("body_is_result"):
                    _uniform_int(source["code"], rows, "hierarchy iteration Result differs across execution lane")
                    rows.extend(("if (%s != 0) {" % source["code"],
                        "%s = %s;" % (output["code"], source["code"]),
                        "%s = %s;" % (output["origin"], source["origin"])))
                    _finish(program, value, available, completed, rows, join, frame)
                    rows.append("}")
                    source = source["value"]
                _copy(state_signature, pending, source, rows)
                _copy(state_signature, output["state"], pending, rows)
                rows.append("++%s->iteration;" % frame)
                _enter(program, value, available, completed, rows)
                rows.append("%s->repeat();" % frame)
                _normal_close(rows)

            emit_region(program, value.attrs["body_block"], step, model, body, prelude,
                block_idx, field_plans, terminal=step_done, available=available)

        emit_region(program, value.attrs["cond_block"], condition, model, target, prelude,
            block_idx, field_plans, terminal=condition_done, available=available)
        target.extend(("};", "} catch (...) { %s_repeat_error = std::current_exception(); }" % frame,
            'pops::collectively_rethrow_exception(%s_repeat_error, ctx.prepared_execution_lane(), "hierarchy iteration continuation allocation");' % frame))
        _install(value, output, frame, join, available, outer, target)
        _enter(program, value, available, outer, target)
        target.append("%s->repeat();" % frame)
        _normal_close(target)
    return close
