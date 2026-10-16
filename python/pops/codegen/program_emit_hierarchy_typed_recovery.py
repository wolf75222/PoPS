"""Complete typed values crossing a generated hierarchy recovery frontier.

The value is retained by its producer, never decoded into a new domain instance
or reduced to an integer error code during an asynchronous control transfer.
"""
from __future__ import annotations

import json


def _identity(signature):
    from pops._structural_data import canonical_json
    from pops.time.canonical_data import _json_ready
    return canonical_json({"schema": "pops.hierarchy.typed-error@1", "type": _json_ready(signature)})


def emit_raise(program, signature, tokens, capture, lines, block_idx):
    from .program_emit_structured_control import _inject_error
    from .program_emit_versioned import pack_local, schema_cpp
    operation, errors, output = capture
    _inject_error(errors, output, signature, tokens, lines)
    capacity = "UINT64_C(%d)" % program.terminal_failure_byte_capacity
    body = []
    value = pack_local(errors, output, body, "hierarchy_error_%d" % operation,
                       block_idx, capacity=capacity)
    lines.extend(("ctx.raise_hierarchy_typed_failure(%d, [&]() {" % operation, *body,
                  "return std::make_tuple(%s,%s,std::string(%s));" %
                  (schema_cpp(errors, block_idx), value, json.dumps(_identity(errors))),
                  "}, %s);" % capacity))


def _install(value, output, frame, join, available, var, lines, block_idx):
    from .program_emit_hierarchy_control import _capture
    from .program_amr_finite_scope import resume_scope
    from .program_emit_structured_control import _assign_output, _variant_tag
    from .program_emit_versioned import unpack_owned

    signature = value.attrs["signature"]
    errors = value.attrs["error_signature"]
    error = output["payloads"]["err"]
    runtime = error["payloads"]["runtime"]
    capacity = "UINT64_C(%d)" % value.prog.terminal_failure_byte_capacity
    contract = "pops.hierarchy.recovery-node@1:" + value.prog._ir_hash() + ":" + str(value.id)
    lines.extend(("ctx.push_hierarchy_failure_handler(%d, %s, %s," %
        (value.id, json.dumps(value.prog._ir_hash()), json.dumps(contract)),
        "[%s](std::exception_ptr failure) {" % _capture(available, var),
        "auto& ctx = *ctx_owner;"))
    resume_scope(lines, var)
    if value.attrs.get("continuation_scope"):
        lines.append("%s->child_scope.reset();" % frame)
        lines.append("ctx.continuation_control_consensus();")
    lines.extend(("try { std::rethrow_exception(failure); }",
        "catch (const pops::runtime::program::StepAttemptRejected& rejected) {",
        'if (rejected.typed_failure_schema() == "pops.hierarchy.typed-capture@1") {',
        "%s->recovered = ctx.hierarchy_failure_value(rejected,%d,%s,%s);" %
        (frame, value.id, json.dumps(_identity(errors)), capacity)))
    restored = unpack_owned(errors, frame+"->recovered->value", lines,
        "hierarchy_error_value_%d" % value.id, capacity=capacity, block_indices=block_idx,
        retain_owner=frame+"->recovered_storage")
    _assign_output(errors, error, restored, lines)
    lines.extend(("} else {",
        "const auto words = pops::runtime::program::hierarchy_control_failure_words(rejected);",
        "%s = %d;" % (error["tag"], _variant_tag(errors, "runtime")),
        "%s = words.first;" % runtime["code"], "%s = words.second;" % runtime["origin"],
        "}", "%s = %d;" % (output["tag"], _variant_tag(signature, "err")), "}",
        "%s(%s);" % (join, frame), "});"))


def open_typed_capture(program, value, available, var, model, lines, prelude, block_idx, field_plans):
    from .program_emit_hierarchy_control import (
        _heap_slots, _capture, _open_join, _close_join, _normal_close, emit_region,
    )
    from .program_emit_hierarchy_recovery import _enter, _finish
    from .program_emit_structured_control import (
        _assign_output, _variant_tag, _uniform_int, _require_success,
    )
    scoped = bool(value.attrs.get("continuation_scope"))
    signature, success = value.attrs["signature"], value.attrs["success_signature"]
    errors = dict(signature["members"])["err"]
    if scoped and success["vtype"] != "continuation_accepted":
        raise ValueError("continuation scope must return the complete accepted provisional")
    members = ["typename std::remove_cvref_t<decltype(ctx)>::hierarchy_typed_failure recovered;",
               "std::shared_ptr<const void> recovered_storage;"]
    if scoped:
        members.append("std::optional<pops::runtime::program::ContinuationAuthority<pops::kNativeDimension>::Scope> child_scope;")
    frame_type, frame, output = _heap_slots(signature, value.id, "hierarchy_typed_capture", lines,
        extra_members=tuple(members))
    var[value.id] = output
    outer = dict(var)
    join = "hierarchy_typed_capture_done_%d" % value.id
    _open_join(join, _capture(available, var), frame_type, frame, lines)

    def close(target):
        _close_join(join, target)
        _install(value, output, frame, join, available, outer, target, block_idx)
        _enter(program, value, available, outer, target)
        if scoped:
            from .program_emit_versioned import _checked
            _checked(target, "hierarchy_scope_entry_%d" % value.id, "bool",
                     ["%s->child_scope.emplace(ctx.continuation_authority().begin_scope());" % frame],
                     "true", message="continuation child scope entry failed")
            target.append("ctx.continuation_control_consensus();")
        sub = dict(outer)
        sub.pop(("typed_failure_capture",), None)
        sub[("hierarchy_failure_capture",)] = True
        sub[("hierarchy_typed_failure_capture",)] = (value.id, errors, output["payloads"]["err"])

        def terminal(current, body):
            source = current[value.attrs["body"].id]
            if value.attrs.get("body_is_variant_result"):
                from pops.time.value_signatures import value_signature
                source_signature = value_signature(value.attrs["body"])
                _uniform_int(source["tag"], body, "hierarchy typed Result differs across execution lane")
                body.append("if (%s == %d) {" % (source["tag"], _variant_tag(source_signature, "err")))
                emit_raise(program, dict(source_signature["members"])["err"], source["payloads"]["err"],
                           sub[("hierarchy_typed_failure_capture",)], body, block_idx)
                body.append("}")
                source = source["payloads"]["ok"]
            elif value.attrs.get("body_is_result"):
                _require_success(source["code"], source["origin"], current, body, "captured Result failed")
                source = source["value"]
            _assign_output(success, output["payloads"]["ok"], source, body)
            body.append("%s = %d;" % (output["tag"], _variant_tag(signature, "ok")))
            if scoped:
                from .program_emit_versioned import _checked
                _checked(body, "hierarchy_scope_plan_%d" % value.id, "bool",
                         ["ctx.prepare_hierarchy_scope_promotion(%d,*%s->child_scope,%s,%s);" %
                          (value.id, frame, output["payloads"]["ok"], frame)], "true",
                         message="accepted child promotion preparation failed")
            _finish(program, value, available, current, body, join, frame, promotion=scoped)

        emit_region(program, value.attrs["body_block"], sub, model, target, prelude,
            block_idx, field_plans, terminal=terminal, available=available)
        _normal_close(target)
    return close


def open_typed_iterate(program, value, available, var, model, lines, prelude, block_idx, field_plans):
    from .program_emit_hierarchy_control import (
        _heap_slots, _capture, _open_join, _close_join, _normal_close, emit_region,
    )
    from .program_emit_hierarchy_recovery import _enter, _finish
    from .program_emit_structured_control import (
        _copy, _uniform_int, _storage_width, _variant_tag, _require_success,
    )
    from pops.time.value_signatures import value_signature

    errors = value.attrs["error_signature"]
    signature, state_signature = value_signature(value), value.attrs["state_signature"]
    frame_type, frame, output = _heap_slots(signature, value.id, "hierarchy_typed_iterate", lines,
        carry=True, extra_members=("typename std::remove_cvref_t<decltype(ctx)>::hierarchy_typed_failure recovered;",
                       "std::shared_ptr<const void> recovered_storage;"))
    _, _, pending = _heap_slots(state_signature, value.id, "hierarchy_typed_iterate_next", lines,
        carry=True, slot_offset=_storage_width(signature))
    success, failure = output["payloads"]["ok"], output["payloads"]["err"]
    _copy(state_signature, success["state"], var[value.inputs[0].id], lines)
    var[value.id] = output
    outer, budget = dict(var), var[value.inputs[1].id]
    _uniform_int(budget, lines, "hierarchy iterate budget differs across execution lane")
    join = "hierarchy_typed_iterate_done_%d" % value.id
    _open_join(join, _capture(available, var), frame_type, frame, lines)

    def runtime_failure(target, code):
        target.extend(("%s = %d;" % (output["tag"], _variant_tag(signature, "err")),
            "%s = %d;" % (failure["tag"], _variant_tag(errors, "runtime")),
            "%s = %d;" % (failure["payloads"]["runtime"]["code"], code),
            "%s = 2;" % failure["payloads"]["runtime"]["origin"]))

    def close(target):
        _close_join(join, target)
        weak = frame+"_weak"
        target.extend(("std::weak_ptr<%s> %s = %s;" % (frame_type, weak, frame),
            "std::exception_ptr %s_repeat_error;" % frame, "try {",
            "%s->repeat = [%s, retained_weak = %s]() {" % (frame, _capture(available, outer), weak),
            "auto& ctx = *ctx_owner;", "auto %s = retained_weak.lock();" % frame,
            'if (!%s) throw std::logic_error("expired hierarchy iteration continuation");' % frame,
            "if (%s < 0) {" % budget))
        runtime_failure(target, 2)
        _finish(program, value, available, outer, target, join, frame)
        target.append("}")
        condition = dict(outer)
        condition.pop(("typed_failure_capture",), None)
        condition[("hierarchy_failure_capture",)] = True
        condition[("hierarchy_typed_failure_capture",)] = (value.id, errors, failure)
        condition[value.attrs["cond_argument"].id] = success["state"]

        def condition_done(current, body):
            choice = "hierarchy_typed_iteration_stop_%d" % value.id
            body.append("const long %s = (%s) ? 1L : 0L;" % (choice, current[value.attrs["cond"].id]))
            _uniform_int(choice, body, "hierarchy iterate stop differs across execution lane")
            body.extend(("if (%s != 0) {" % choice,
                "%s = %d;" % (output["tag"], _variant_tag(signature, "ok")),
                "%s = %s->iteration;" % (success["iterations"], frame)))
            _finish(program, value, available, current, body, join, frame)
            body.append("}")
            body.append("if (%s->iteration == %s) {" % (frame, budget))
            runtime_failure(body, 1)
            _finish(program, value, available, current, body, join, frame)
            body.append("}")
            step = dict(current)
            step[value.attrs["body_argument"].id] = success["state"]

            def step_done(completed, rows):
                source = completed[value.attrs["body"].id]
                if value.attrs.get("body_is_variant_result"):
                    source_signature = value_signature(value.attrs["body"])
                    _uniform_int(source["tag"], rows, "hierarchy iteration Result differs across execution lane")
                    rows.append("if (%s == %d) {" % (source["tag"], _variant_tag(source_signature, "err")))
                    emit_raise(program, dict(source_signature["members"])["err"], source["payloads"]["err"],
                               completed[("hierarchy_typed_failure_capture",)], rows, block_idx)
                    rows.append("}")
                    source = source["payloads"]["ok"]
                elif value.attrs.get("body_is_result"):
                    _require_success(source["code"], source["origin"], completed, rows, "iteration Result failed")
                    source = source["value"]
                _copy(state_signature, pending, source, rows)
                _copy(state_signature, success["state"], pending, rows)
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
        _install(value, output, frame, join, available, outer, target, block_idx)
        _enter(program, value, available, outer, target)
        target.append("%s->repeat();" % frame)
        _normal_close(target)
    return close
