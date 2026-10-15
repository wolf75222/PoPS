"""Exact fixed-Product field projections through owned continuation capabilities.

This follows structural selection only. It never infers physical equivalence or
changes a numerical coefficient. Unknown structural paths remain unproved.
"""
from __future__ import annotations

from pops.time.values import ProgramValue


def field_region_origins(operations):
    """Prepare the region edges once for a whole-program fixed-point analysis."""
    carries = {}
    for operation in operations:
        if operation.op in {"structured_fold", "structured_iterate"}:
            for argument in ("body_argument", "cond_argument"):
                if argument in operation.attrs:
                    carries[operation.attrs[argument].id] = operation
    return carries


def field_projection_sources(value: ProgramValue, *, region_origins=None):
    if region_origins is None:
        from .program_lowerability import all_ops
        region_origins = field_region_origins(all_ops(value.prog))

    def body_path(operation, path):
        if operation.attrs.get("body_is_variant_result"):
            return ("@variant", "ok", *path)
        if operation.attrs.get("body_is_result"):
            return ("@result", *path)
        return path

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
        elif op == "variant_payload":
            pending.append((node.inputs[0], ("@variant", node.attrs["tag"], *path)))
        elif op == "variant_unreachable":
            # The exhaustiveness guard throws; it cannot publish a Field.
            continue
        elif op == "variant_construct" and len(path) >= 2 and path[0] == "@variant":
            if node.attrs["tag"] == path[1]:
                pending.append((node.inputs[0], path[2:]))
        elif op == "branch":
            pending.extend((node.attrs[name], path) for name in ("true_result", "false_result"))
        elif op == "result_match":
            pending.extend((node.attrs[name], path) for name in ("true_result", "false_result"))
        elif op == "structured_fold" or (op == "region_argument" and node.id in region_origins):
            operation = node if op == "structured_fold" else region_origins[node.id]
            pending.append((operation.inputs[0], path))
            pending.append((operation.attrs["body"], body_path(operation, path)))
        elif op in {"result_success", "result_payload"}:
            if node.attrs.get("variant", "ok") != "ok":
                return None
            pending.append((node.inputs[0], ("@result", *path)))
        elif op in {"iteration_success", "iteration_payload"}:
            if node.attrs.get("variant", "ok") != "ok":
                return None
            pending.append((node.inputs[0], ("@iteration", *path)))
        elif op == "structured_iterate":
            prefix = (("@variant", "ok", "state") if node.vtype == "variant"
                      else ("@iteration", "state"))
            if path[:len(prefix)] != prefix:
                return None
            selected = path[len(prefix):]
            pending.append((node.inputs[0], selected))
            pending.append((node.attrs["body"], body_path(node, selected)))
        elif op == "result_ok" and path and path[0] == "@result":
            pending.append((node.inputs[0], path[1:]))
        elif op == "result_error" and path and path[0] == "@result":
            # The template defines storage/type only. This arm produces no successful Field.
            continue
        elif op == "capture_result" and len(path) >= 2 and path[:2] == ("@variant", "ok"):
            body = node.attrs["body"]
            pending.append((body, path if node.attrs.get("body_is_variant_result") else path[2:]))
        elif op == "capture_result" and path and path[0] == "@result":
            pending.append((node.attrs["body"], body_path(node, path[1:])))
        elif op in ("continuation_read", "continuation_working_value", "continuation_synchronized_fields"):
            # These authenticated native images are the incoming state for this
            # invocation, not another evaluation of a current-step face flux.
            continue
        elif not path and node.vtype in {"state", "rhs", "scalar_field"}:
            sources[node.id] = node
        else:
            return None
    return tuple(sources.values())
