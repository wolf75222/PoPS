"""Stack-safe, owned storage and atomic active-arm copies for Program control flow."""

from __future__ import annotations
from typing import Any


class _OwnedFieldSlot(str):
    """A generated Field reference with an explicit owning C++ slot."""

    owner_token: str

    def __new__(cls, owner_token: str):
        value = super().__new__(cls, "(*%s)" % owner_token)
        value.owner_token = owner_token
        return value


def make_slots(
    signature, sources, node_id, prefix, lines, *, slot_offset=0, pointers=False, deferred=False
) -> Any:
    from .program_emit_finite_arrays import finite_cpp_type

    result: list[Any] = [None]
    stack: list[tuple[Any, Any, Any]] = [(signature, result, 0)]
    serial = 0
    while stack:
        sig, parent, key = stack.pop()
        token = "%s_%d_s%d" % (prefix, node_id, serial)
        serial += 1
        kind = sig["vtype"]
        if kind == "product":
            output = {}
            parent[key] = output
            stack.extend((child, output, name) for name, child in reversed(sig["members"]))
        elif kind == "variant":
            lines.append("std::int64_t %s = -1;" % token)
            output = {"tag": token, "payloads": {}}
            parent[key] = output
            stack.extend((child, output["payloads"], name)
                         for name, child in reversed(sig["members"]))
        elif kind == "result":
            code, origin = token + "_status", token + "_origin"
            lines += ["std::int64_t %s = 0;" % code, "std::int64_t %s = 0;" % origin]
            output = {"code": code, "origin": origin}
            parent[key] = output
            stack.append((sig["success"], output, "value"))
        elif kind in ("state", "rhs", "scalar_field"):
            owner = token + "_owner"
            lines.append("std::shared_ptr<pops::MultiFab<pops::kNativeDimension>> %s;" % owner)
            parent[key] = _OwnedFieldSlot(owner)
        else:
            lines.append("%s %s{};" % (finite_cpp_type(sig), token))
            parent[key] = token
    return result[0]


def copy_structure(signature, destination, source, lines, *, references):
    from .program_emit_finite_arrays import finite_cpp_type
    from .program_emit_structured_control import _uniform_int

    # All discriminants converge before any allocation. All active leaf copies
    # then prepare under a single error vote. Only noexcept moves publish slots:
    # a later failed copy cannot replace an earlier Field, scalar or tag.
    lines.append("{")
    stack = [(signature, destination, source, "true")]
    serial = 0
    leaves = []
    tags = []
    while stack:
        sig, target, value, active = stack.pop()
        kind = sig["vtype"]
        if kind == "product":
            stack.extend((child, target[name], value[name], active)
                         for name, child in reversed(sig["members"]))
            continue
        token = "structured_pending_%d" % serial
        serial += 1
        if kind in ("variant", "result"):
            discriminants = ("tag",) if kind == "variant" else ("code", "origin")
            for key in discriminants:
                name = token + "_" + key
                lines += ["std::int64_t %s = 0;" % name, "if (%s) {" % active,
                          "std::exception_ptr structured_tag_error;",
                          "try { %s = %s; }" % (name, value[key]),
                          "catch (...) { structured_tag_error = std::current_exception(); }",
                          'pops::collectively_rethrow_exception(structured_tag_error, ctx.prepared_execution_lane(), "structured control discriminant");']
                _uniform_int(name, lines, "structured discriminant differs across execution lane")
                if kind == "variant":
                    lines.append('if (%s < 0 || %s >= %d) throw std::logic_error("invalid native Variant discriminant");'
                                 % (name, name, len(sig["members"])))
                lines.append("}")
                tags.append((target[key], name, active))
            if kind == "variant":
                children = []
                for arm, (name, child) in enumerate(sig["members"]):
                    flag = token + "_active_%d" % arm
                    lines.append("const bool %s = %s && %s_tag == %d;" % (flag, active, token, arm))
                    children.append((child, target["payloads"][name], value["payloads"][name], flag))
                stack.extend(reversed(children))
            else:
                flag = token + "_active"
                lines.append("const bool %s = %s && %s_code == 0;" % (flag, active, token))
                stack.append((sig["success"], target["value"], value["value"], flag))
            continue
        is_field = kind in ("state", "rhs", "scalar_field")
        if is_field and not isinstance(target, _OwnedFieldSlot):
            raise TypeError("structured Field copy requires an explicit owning slot")
        lines.append("%s %s{};" % (finite_cpp_type(sig), token))
        leaves.append((target, value, active, token, is_field))
    lines += ["std::exception_ptr structured_copy_error;", "try {"]
    for _target, value, active, token, is_field in leaves:
        expression = "ctx.own_structural_field(%s)" % value if is_field else value
        lines.append("if (%s) %s = %s;" % (active, token, expression))
    lines += ["} catch (...) { structured_copy_error = std::current_exception(); }",
              "try { Kokkos::fence(); } catch (...) { if (!structured_copy_error) structured_copy_error = std::current_exception(); }",
              'pops::collectively_rethrow_exception(structured_copy_error, ctx.prepared_execution_lane(), "structured control owned copy");']
    for target, _value, active, token, is_field in leaves:
        lines.append("if (%s) {" % active)
        if is_field:
            lines.append("%s = std::move(%s);" % (target.owner_token, token))
        else:
            lines += ["static_assert(std::is_nothrow_move_assignable_v<std::remove_reference_t<decltype(%s)>>);" % target,
                      "%s = std::move(%s);" % (target, token)]
        lines.append("}")
    for target, token, active in tags:
        lines.append("if (%s) %s = %s;" % (active, target, token))
    lines.append("}")
