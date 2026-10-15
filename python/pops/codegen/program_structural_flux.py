"""Exact fixed-Product field projections through owned continuation capabilities.

This follows structural selection only. It never infers physical equivalence or
changes a numerical coefficient. Unknown structural paths remain unproved.
"""
from __future__ import annotations

from pops.time.values import ProgramValue


def field_projection_sources(value: ProgramValue):
    pending: list[tuple[ProgramValue, tuple[str, ...]]] = [(value, ())]
    visited: set[tuple[int, tuple[str, ...]]] = set()
    sources: dict[int, ProgramValue] = {}
    while pending:
        node, path = pending.pop()
        key = (node.id, path)
        if key in visited:
            continue
        visited.add(key)
        op = node.op
        if op == "product_get":
            pending.append((node.inputs[0], (node.attrs["field"], *path)))
        elif op == "product" and path:
            fields = tuple(node.attrs["fields"])
            if path[0] not in fields:
                return None
            pending.append((node.inputs[fields.index(path[0])], path[1:]))
        elif op == "continuation_candidate_value":
            pending.append((node.inputs[0], ("@envelope", *path)))
        elif op == "continuation_accepted_candidate":
            pending.append((node.inputs[0], ("@candidate", *path)))
        elif op == "continuation_freeze" and path and path[0] == "@envelope":
            pending.append((node.inputs[2], path[1:]))
        elif op == "continuation_accept" and path and path[0] == "@candidate":
            pending.append((node.inputs[0], path[1:]))
        elif op == "continuation_transform":
            pending.append((node.inputs[1], path))
        elif op == "variant_require_success":
            pending.append((node.inputs[0], ("@variant", "ok", *path)))
        elif op == "variant_construct" and len(path) >= 2 and path[0] == "@variant":
            if node.attrs["tag"] == path[1]:
                pending.append((node.inputs[0], path[2:]))
        elif op == "branch":
            pending.extend((node.attrs[name], path) for name in ("true_result", "false_result"))
        elif op == "capture_result" and len(path) >= 2 and path[:2] == ("@variant", "ok"):
            body = node.attrs["body"]
            pending.append((body, path if node.attrs.get("body_is_variant_result") else path[2:]))
        elif op in ("continuation_read", "continuation_working_value", "continuation_synchronized_fields"):
            # These authenticated native images are the incoming state for this
            # invocation, not another evaluation of a current-step face flux.
            continue
        elif not path and node.vtype in {"state", "rhs", "scalar_field"}:
            sources[node.id] = node
        else:
            return None
    return tuple(sources.values())
