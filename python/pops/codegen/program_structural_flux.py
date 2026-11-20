"""Exact fixed-Product field projections through owned continuation capabilities.

This follows structural selection only. It never infers physical equivalence or
changes a numerical coefficient. Unknown structural paths remain unproved.
"""
from __future__ import annotations

from pops.time.values import ProgramValue
from pops._ir.data_model import is_field_leaf


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
        elif not path and is_field_leaf(node):
            sources[node.id] = node
        else:
            return None
    return tuple(sources.values())


def uniform_trial_commit_sources(program, trials):
    """Give scientific readers the exact root carrier fields before native publication."""
    if not trials:
        return None
    from pops.time.value_signatures import value_signature

    roots = {value.id: value for value in program._values}

    def issued(node, op=None, arity=None):
        if (roots.get(node.id) is not node or node.region
                or (op is not None and node.op != op)
                or (arity is not None and len(node.inputs) != arity)):
            raise ValueError("Uniform continuation trial requires a direct issued root projection")
        return node

    # The continuation validator has already authenticated the only publication
    # and its schema path. This reader additionally proves the one dynamic
    # candidate, where equal typed capabilities alone do not establish identity.
    publication, = (node for node in roots.values() if node.op == "continuation_publish")
    accepted = issued(publication.inputs[0], "continuation_accept", 2)
    candidate = issued(accepted.inputs[0], "continuation_freeze", 3)
    assessment = issued(accepted.inputs[1], "continuation_attest", 3)
    if assessment.inputs[0] is not candidate:
        raise ValueError("Uniform continuation trial attestation reads a different frozen candidate")
    envelope = issued(candidate.inputs[2], "product", 4)

    def member(node, name):
        issued(node, "product")
        fields = tuple(node.attrs["fields"])
        members = tuple(field for field, _ in value_signature(node)["members"])
        if (fields != members or name not in fields or len(fields) != len(node.inputs)
                or len(set(fields)) != len(fields)):
            raise ValueError("Uniform continuation trial changed an ordered candidate product")
        return node.inputs[fields.index(name)]

    commits = {}
    for state_ref, path in publication.attrs["schema"]["fields"]:
        projected = program._commits[state_ref]
        full_path = ("continuation", *path)
        cursor = projected
        for name in reversed(full_path):
            issued(cursor, "product_get", 1)
            if cursor.attrs["field"] != name:
                raise ValueError("Uniform continuation trial changed its commit path")
            cursor = cursor.inputs[0]
        issued(cursor, "continuation_candidate_value", 1)
        carrier = issued(cursor.inputs[0], "continuation_accepted_candidate", 1)
        if carrier.inputs[0] is not accepted:
            raise ValueError("Uniform continuation trial commit reads a different accepted candidate")
        leaf = envelope
        for name in full_path:
            leaf = member(leaf, name)
        issued(leaf)
        projected_sources = field_projection_sources(projected, region_origins={})
        if (not projected_sources or len(projected_sources) != 1 or projected_sources[0] is not leaf
                or leaf.vtype != "state" or leaf.state_ref != state_ref
                or leaf.block != projected.block or leaf.space != projected.space
                or leaf.point != projected.point):
            raise ValueError("Uniform continuation trial requires one direct candidate State field")
        commits[state_ref] = leaf
    return commits
