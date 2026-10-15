"""Compile structured hierarchy regions to typed native continuation frames.

There is no graph interpreter: each callback contains the same generated PoPS
operations as a straight-line region. Only its lexical storage outlives a yield.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pops.time._program.structured_control import value_signature


def has_suspension(value: Any) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if item.op == "solve_spatial_field":
            return True
        for key in ("body_block", "true_block", "false_block", "cond_block"):
            stack.extend(item.attrs.get(key, ()))
    return False


def _heap_slots(signature, node_id, prefix, lines, *, carry=False, slot_offset=0):
    """Own scalar/finite leaves, and point to exact context-owned Field storage."""
    from .program_emit_finite_arrays import finite_cpp_type, _allocate
    from .program_emit_structured_control import _DeferredField

    frame_type, frame = "%s_%d_Frame" % (prefix, node_id), "%s_%d" % (prefix, node_id)
    declarations = ["struct %s {" % frame_type]
    result: list[Any] = [None]
    stack: list[tuple[Any, Any, Any]] = [(signature, result, 0)]
    serial, field_slot = 0, slot_offset
    while stack:
        sig, parent, key = stack.pop()
        kind = sig["vtype"]
        if kind == "product":
            output = {}
            parent[key] = output
            stack.extend((child, output, name) for name, child in reversed(sig["members"]))
            continue
        member = "s%d" % serial
        serial += 1
        token = "%s->%s" % (frame, member)
        if kind == "variant":
            declarations.append("std::int64_t %s = -1;" % member)
            output = {"tag": token, "payloads": {}}
            parent[key] = output
            stack.extend((child, output["payloads"], name) for name, child in reversed(sig["members"]))
        elif kind == "result":
            declarations.extend(("std::int64_t %s_code = 0;" % member,
                                 "std::int64_t %s_origin = 0;" % member))
            output = {"code": token+"_code", "origin": token+"_origin"}
            parent[key] = output
            stack.append((sig["success"], output, "value"))
        elif kind in ("state", "rhs", "scalar_field"):
            declarations.append("pops::MultiFab<pops::kNativeDimension>* %s = nullptr;" % member)
            parent[key] = (_DeferredField("(*%s)" % token, node_id, field_slot) if carry
                           else "(*%s)" % token)
            field_slot += 1
        else:
            declarations.append("%s %s{};" % (finite_cpp_type(sig), member))
            parent[key] = token
    if carry:
        declarations.extend(("std::int64_t iteration = 0;", "std::function<void()> repeat;"))
    declarations.append("};")
    lines.extend(declarations)
    lines.append("std::shared_ptr<%s> %s;" % (frame_type, frame))
    _allocate("std::make_shared<%s>()" % frame_type, frame, lines)
    return frame_type, frame, result[0]


def _capture(values, var):
    from .program_emit_mapping_regions import continuation_capture
    from .program_amr_finite_scope import continuation_scope_capture
    return continuation_scope_capture(continuation_capture(values, var), var)


def _normal_close(lines):
    lines.extend(("});", "return;"))


def _open_join(name, capture, frame_type, frame, lines):
    lines.extend(("std::function<void(const std::shared_ptr<%s>&)> %s;" % (frame_type, name),
                  "std::exception_ptr %s_allocation_error;" % name, "try {",
                  "%s = [%s](const std::shared_ptr<%s>& %s) {" %
                  (name, capture, frame_type, frame), "auto& ctx = *ctx_owner;"))


def _close_join(name, lines):
    lines.extend(("};", "} catch (...) { %s_allocation_error = std::current_exception(); }" % name,
                  'pops::collectively_rethrow_exception(%s_allocation_error, ctx.prepared_execution_lane(), "hierarchy region continuation allocation");' % name))


def _check_control_effects(block):
    """Reuse the checked region effects, adding only authenticated original solves."""
    from .program_emit_structured_control import _check_region_effects
    from .program_emit_amr_original_field import original_field_consumer
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    pending = list(block)
    while pending:
        value = pending.pop()
        if value.op in ("branch", "structured_fold", "capture_result", "structured_iterate"):
            for key in ("true_block", "false_block", "body_block", "cond_block"):
                pending.extend(value.attrs.get(key, ()))
        elif value.op == "solve_spatial_field":
            validate_nonlinear_field_request(value.prog, value)
        elif value.op == "field_problem_coefficients" and original_field_consumer(value) is not None:
            # The established coefficient emitter converges allocation and kernel failures.
            pass
        elif value.op in ("solve_outcome", "solve_outcome_component"):
            source = value.inputs[0]
            while source.op in ("solve_outcome", "solve_outcome_component"):
                source = source.inputs[0]
            if source.op != "solve_spatial_field":
                _check_region_effects((value,))
            else:
                # This is an alias only after the native barrier consumed its SolveReport.
                validate_nonlinear_field_request(source.prog, source)
        else:
            _check_region_effects((value,))


def emit_region(program, block, var, model, lines, prelude, block_idx, field_plans,
                *, terminal: Callable[[Any, list[str]], None], available=()):
    """Emit a region and retain its terminal action across all genuine yields."""
    from .program_emit_ops import _emit_op
    from .program_emit_hierarchy_regions import open_hierarchy_continuation
    from .program_emit_amr_original_field import emit_amr_original_field
    from .program_source_map import operation

    completed = list(available)
    closers = []
    for value in block:
        base = next((node for node in program._values if node.op == "state"
                     and node.block == value.block and
                     (value.state_ref is None or node.state_ref == value.state_ref)), None)
        if value.op in ("branch", "structured_fold", "capture_result", "structured_iterate") and has_suspension(value):
            with operation(value, lines, prelude, environment=var):
                close = open_control(program, value, completed, var, model, lines, prelude,
                                     block_idx, field_plans)
            closers.append(close)
        elif value.op == "solve_spatial_field":
            var[("original_field_hierarchy_phase",)] = "gather"
            _emit_op(program, value, base, frozenset(), var, model, lines, prelude,
                     block_idx=block_idx, field_plans=field_plans, target="amr_system")
            solve_var, once = dict(var), []
            solve_var[("original_field_hierarchy_phase",)] = "solve"
            with operation(value, once, environment=solve_var):
                emit_amr_original_field(program, value, solve_var, once, [], block_idx)
            var.pop(("original_field_hierarchy_phase",), None)
            open_hierarchy_continuation(program, value, [*completed, value], var, lines,
                                        once, kind="nonlinear_solve")
            closers.append(_normal_close)
        else:
            if has_suspension(value):
                raise NotImplementedError("hierarchy suspension in %s requires an explicit failure/control continuation" % value.op)
            _emit_op(program, value, base, frozenset(), var, model, lines, prelude,
                     block_idx=block_idx, field_plans=field_plans, target="amr_system")
        completed.append(value)
    terminal(var, lines)
    for close in reversed(closers):
        close(lines)


def open_control(program, value, available, var, model, lines, prelude, block_idx, field_plans):
    from .program_emit_structured_control import _assign_output, _copy, _uniform_int, _reject_if, _storage_width
    from .program_emit_hierarchy_regions import open_hierarchy_continuation

    _check_control_effects((value,))
    if value.op in ("capture_result", "structured_iterate"):
        from .program_emit_hierarchy_recovery import open_capture, open_iterate
        handler = open_capture if value.op == "capture_result" else open_iterate
        return handler(program, value, available, var, model, lines, prelude, block_idx, field_plans)
    signature = value_signature(value)
    if value.op == "branch":
        frame_type, frame, output = _heap_slots(signature, value.id, "hierarchy_branch", lines)
        condition = "hierarchy_choice_%d" % value.id
        lines.append("const long %s = (%s) ? 1L : 0L;" % (condition, var[value.inputs[0].id]))
        _uniform_int(condition, lines, "hierarchy branch differs across execution lane")
        var[value.id] = output
        continuation = "hierarchy_join_%d" % value.id
        outer_var = dict(var)
        _open_join(continuation, _capture(available, var), frame_type, frame, lines)

        def close(target):
            _close_join(continuation, target)
            target.append("if (%s != 0) {" % condition)
            for index, prefix in enumerate(("true", "false")):
                if index:
                    target.append("} else {")
                sub = dict(outer_var)
                if index == 0 and value.inputs[0].op == "variant_is":
                    predicate = value.inputs[0]
                    sub[("variant_guard", predicate.inputs[0].id)] = predicate.attrs["tag"]
                result = value.attrs[prefix+"_result"]

                def terminal(current, body, result=result):
                    _assign_output(signature, output, current[result.id], body)
                    body.append("%s(%s);" % (continuation, frame))

                emit_region(program, value.attrs[prefix+"_block"], sub, model, target,
                            prelude, block_idx, field_plans, terminal=terminal,
                            available=available)
            target.extend(("}", "return;"))
        return close

    if value.op != "structured_fold":
        raise NotImplementedError("hierarchy control continuation is not defined")
    initial, count_value, capacity_value = value.inputs
    signature = value_signature(initial)
    frame_type, frame, current = _heap_slots(signature, value.id, "hierarchy_fold", lines, carry=True)
    pending_type, pending_frame, pending = _heap_slots(signature, value.id, "hierarchy_next", lines,
        carry=True, slot_offset=_storage_width(signature))
    del pending_type
    _copy(signature, current, var[initial.id], lines)
    bound, budget = var[count_value.id], var[capacity_value.id]
    _uniform_int(bound, lines, "hierarchy fold count differs across execution lane")
    _uniform_int(budget, lines, "hierarchy fold capacity differs across execution lane")
    _reject_if("%s < 0 || %s < 0 || %s > %s" % (bound, budget, bound, budget),
               lines, "hierarchy fold count outside declared capacity")
    var[value.id] = current
    outer_var = dict(var)
    continuation = "hierarchy_fold_done_%d" % value.id
    _open_join(continuation, _capture(available, var), frame_type, frame, lines)

    def close(target):
        weak = frame + "_weak"
        _close_join(continuation, target)
        target.extend(("std::weak_ptr<%s> %s = %s;" % (frame_type, weak, frame),
                       "std::exception_ptr %s_repeat_error;" % frame, "try {"))
        # The stored repeat callback holds a weak reference to its own frame.
        # Every actual suspended continuation retains the live frame by value.
        target.append("%s->repeat = [%s, retained_weak = %s]() {" % (frame, _capture(available, outer_var), weak))
        target.append("auto& ctx = *ctx_owner;")
        target.append("auto %s = retained_weak.lock();" % frame)
        target.append('if (!%s) throw std::logic_error("expired hierarchy loop continuation");' % frame)
        target.append("if (%s->iteration >= %s) { %s(%s); return; }" %
                      (frame, bound, continuation, frame))
        sub = dict(outer_var)
        sub[value.attrs["body_argument"].id] = current
        sub[value.attrs["body_index"].id] = frame+"->iteration"

        def terminal(current_var, body):
            _copy(signature, pending, current_var[value.attrs["body"].id], body)
            _copy(signature, current, pending, body)
            body.append("++%s->iteration;" % frame)
            open_hierarchy_continuation(program, value, available, current_var, body,
                                        [], kind="control")
            body.append("%s->repeat();" % frame)
            _normal_close(body)

        # pending_frame is a distinct retained object, not a reference to a local.
        # Mentioning it in the generated copies causes its value capture above.
        assert pending_frame != frame
        emit_region(program, value.attrs["body_block"], sub, model, target, prelude,
                    block_idx, field_plans, terminal=terminal, available=available)
        target.extend(("};", "} catch (...) { %s_repeat_error = std::current_exception(); }" % frame,
                       'pops::collectively_rethrow_exception(%s_repeat_error, ctx.prepared_execution_lane(), "hierarchy loop continuation allocation");' % frame,
                       "%s->repeat();" % frame, "return;"))
    return close
