"""Stack-safe storage and active-arm copies for the existing Program control flow."""

from __future__ import annotations
from typing import Any


def make_slots(
    signature, sources, node_id, prefix, lines, *, slot_offset=0, pointers=False, deferred=False
) -> Any:
    from .program_emit_finite_arrays import finite_cpp_type
    from .program_emit_structured_control import _DeferredField

    result: list[Any] = [None]
    stack: list[tuple[Any, Any, Any, Any, bool, bool]] = [
        (signature, sources, result, 0, pointers, deferred)
    ]
    serial, offset = 0, slot_offset
    while stack:
        sig, source, parent, key, pointer, delay = stack.pop()
        token = "%s_%d_s%d" % (prefix, node_id, serial)
        serial += 1
        kind = sig["vtype"]
        if kind == "product":
            output = {}
            parent[key] = output
            stack.extend(
                (child, None if source is None else source[name], output, name, pointer, delay)
                for name, child in reversed(sig["members"])
            )
        elif kind == "variant":
            lines.append("std::int64_t %s = -1;" % token)
            output = {"tag": token, "payloads": {}}
            parent[key] = output
            stack.extend(
                (child, None, output["payloads"], name, True, delay or not pointer)
                for name, child in reversed(sig["members"])
            )
        elif kind == "result":
            code, origin = token + "_status", token + "_origin"
            lines += ["std::int64_t %s = 0;" % code, "std::int64_t %s = 0;" % origin]
            output = {"code": code, "origin": origin}
            parent[key] = output
            stack.append((sig["success"], None, output, "value", True, delay))
        elif kind in ("state", "rhs", "scalar_field"):
            if pointer:
                lines.append("pops::MultiFab<pops::kNativeDimension>* %s = nullptr;" % token)
                parent[key] = (
                    _DeferredField("(*%s)" % token, node_id, offset) if delay else "(*%s)" % token
                )
            else:
                lines += [
                    "pops::MultiFab<pops::kNativeDimension>* %s_storage = nullptr;" % token,
                    "std::exception_ptr %s_error;" % token,
                    "try { %s_storage = &ctx.scratch_state(%d, %d, %s); }"
                    % (token, node_id, offset, source),
                    "catch (...) { %s_error = std::current_exception(); }" % token,
                    "try { Kokkos::fence(); } catch (...) { if (!%s_error) %s_error = std::current_exception(); }"
                    % (token, token),
                    'pops::collectively_rethrow_exception(%s_error, ctx.prepared_execution_lane(), "structured control field allocation");'
                    % token,
                    "auto& %s = *%s_storage;" % (token, token),
                ]
                parent[key] = token
            offset += 1
        else:
            lines.append("%s %s{};" % (finite_cpp_type(sig), token))
            parent[key] = token
            offset += 1
    return result[0]


def copy_structure(signature, destination, source, lines, *, references):
    from .program_emit_structured_control import _DeferredField, _uniform_int

    # Each invocation owns a shallow scope. An active-arm flag is evaluated only
    # after its parent; inactive fields and payloads are never dereferenced.
    lines.append("{")
    stack = [(signature, destination, source, "true")]
    serial = 0
    while stack:
        sig, target, value, active = stack.pop()
        kind = sig["vtype"]
        if kind == "product":
            stack.extend(
                (child, target[name], value[name], active)
                for name, child in reversed(sig["members"])
            )
            continue
        lines.append("if (%s) {" % active)
        if kind == "variant":
            _uniform_int(value["tag"], lines, "Variant discriminant differs across execution lane")
            lines += [
                "%s = %s;" % (target["tag"], value["tag"]),
                'if (%s < 0 || %s >= %d) throw std::logic_error("invalid native Variant discriminant");'
                % (value["tag"], value["tag"], len(sig["members"])),
                "}",
            ]
            children = []
            for arm, (name, child) in enumerate(sig["members"]):
                flag = "structured_active_%d" % serial
                serial += 1
                lines.append("const bool %s = %s && %s == %d;" % (flag, active, value["tag"], arm))
                children.append((child, target["payloads"][name], value["payloads"][name], flag))
            stack.extend(reversed(children))
            continue
        if kind == "result":
            lines += [
                "%s = %s;" % (target["code"], value["code"]),
                "%s = %s;" % (target["origin"], value["origin"]),
                "}",
            ]
            flag = "structured_active_%d" % serial
            serial += 1
            lines.append("const bool %s = %s && %s == 0;" % (flag, active, value["code"]))
            stack.append((sig["success"], target["value"], value["value"], flag))
            continue
        if kind in ("state", "rhs", "scalar_field"):
            if references:
                lines.append("%s = &%s;" % (target[2:-1], value))
            else:
                lines += ["std::exception_ptr structured_copy_error;", "try {"]
                if isinstance(target, _DeferredField):
                    lines.append(
                        "%s = &ctx.scratch_state(%d, %d, %s);"
                        % (target[2:-1], target.node_id, target.slot, value)
                    )
                lines += [
                    "ctx.copy_with_provenance(%s, %s);" % (target, value),
                    "} catch (...) { structured_copy_error = std::current_exception(); }",
                    "try { Kokkos::fence(); } catch (...) { if (!structured_copy_error) structured_copy_error = std::current_exception(); }",
                    'pops::collectively_rethrow_exception(structured_copy_error, ctx.prepared_execution_lane(), "structured control field copy");',
                ]
        else:
            lines.append("%s = %s;" % (target, value))
        lines.append("}")
    lines.append("}")
