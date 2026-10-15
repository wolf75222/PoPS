"""Flat host storage for structural values; numerical maps still use Kokkos columns."""

from __future__ import annotations
from collections.abc import Iterator, Mapping
from typing import Any


def slots(
    signature: Mapping[str, Any], tokens: Any = None
) -> Iterator[tuple[int, Any, Any, tuple[tuple[int, int], ...]]]:
    """Yield preorder storage slots with the active variant ancestry.

    Products introduce no slot; variants store one discriminant. The physical
    host type has no template nesting proportional to the mathematical type.
    """
    stack: list[tuple[Any, Any, tuple[tuple[int, int], ...]]] = [(signature, tokens, ())]
    index = 0
    while stack:
        sig, token, guards = stack.pop()
        kind = sig["vtype"]
        if kind == "product":
            stack.extend(
                (child, None if token is None else token[key], guards)
                for key, child in reversed(sig["members"])
            )
        elif kind == "variant":
            yield index, sig, token, guards
            stack.extend(
                (child, None if token is None else token["payloads"][key], (*guards, (index, arm)))
                for arm, (key, child) in reversed(tuple(enumerate(sig["members"])))
            )
            index += 1
        else:
            yield index, sig, token, guards
            index += 1


def unpack_record(signature: Mapping[str, Any], token: str) -> Any:
    from .program_emit_finite_arrays import finite_cpp_type

    result: list[Any] = [None]
    stack: list[tuple[Any, Any, Any]] = [(signature, result, 0)]
    index = 0
    while stack:
        sig, parent, key = stack.pop()
        kind = sig["vtype"]
        if kind == "product":
            out = {}
            parent[key] = out
            stack.extend((child, out, name) for name, child in reversed(sig["members"]))
            continue
        cell = "(%s).values.at(%d)" % (token, index)
        index += 1
        if kind == "variant":
            out = {"tag": "std::any_cast<std::int64_t>(%s)" % cell, "payloads": {}}
            parent[key] = out
            stack.extend((child, out["payloads"], name) for name, child in reversed(sig["members"]))
        else:
            expression = "std::any_cast<const %s&>(%s)" % (finite_cpp_type(sig), cell)
            parent[key] = (
                "(*(%s))" % expression if kind in ("state", "rhs", "scalar_field") else expression
            )
    return result[0]


def pack_record(signature: Mapping[str, Any], tokens: Any, lines: list[str], prefix: str) -> str:
    from .program_emit_finite_arrays import _allocate, _packed, _store_cell
    from .program_emit_structured_control import _reject_if

    layout = list(slots(signature, tokens))
    output = prefix + "_record"
    lines.append("pops_algebra_finite::Record %s;" % output)
    _allocate("pops_algebra_finite::Record{std::vector<std::any>(%d)}" % len(layout), output, lines)
    active: dict[tuple[tuple[int, int], ...], str] = {(): "true"}
    for index, sig, token, guards in layout:
        # Each guard is computed from an already declared parent activation.
        # This avoids both nested C++ scopes and deeply nested boolean expressions.
        for depth in range(1, len(guards) + 1):
            path = guards[:depth]
            if path not in active:
                tag_slot, arm = path[-1]
                name = "%s_active_%d" % (prefix, len(active))
                lines.append(
                    "const bool %s = %s && std::any_cast<std::int64_t>(%s.values.at(%d)) == %d;"
                    % (name, active[path[:-1]], output, tag_slot, arm)
                )
                active[path] = name
        lines.append("if (%s) {" % active[guards])
        if sig["vtype"] == "variant":
            tag = token["tag"]
            _reject_if(
                "%s < 0 || %s >= %d" % (tag, tag, len(sig["members"])),
                lines,
                "invalid Variant discriminant",
            )
            _store_cell(
                "std::int64_t(%s)" % tag,
                "%s.values[%d]" % (output, index),
                lines,
                "%s_tag%d" % (prefix, index),
            )
        else:
            value = _packed(sig, token, lines, "%s_leaf%d" % (prefix, index))
            _store_cell(
                value, "%s.values[%d]" % (output, index), lines, "%s_cell%d" % (prefix, index)
            )
        lines.append("}")
    return output
