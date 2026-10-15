"""Retain the complete structural error at the compiled Program transaction boundary."""
from __future__ import annotations

import json


def _nodes(signature):
    pending = [signature]
    seen = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        pending.extend(child for _, child in node.get("members", ()))
        if "element" in node:
            pending.append(node["element"])


def _has_field(signature):
    return any(node["vtype"] in ("state", "scalar_field") for node in _nodes(signature))


def _has_distributed(signature):
    return any(node.get("ownership") == "contiguous_distributed" for node in _nodes(signature))


def _require_supported_transport(signature):
    for node in _nodes(signature):
        if node["vtype"] in ("array", "index_set", "index"):
            if node.get("ownership") not in ("replicated", "contiguous_distributed"):
                raise NotImplementedError("typed terminal finite payload requires an exact supported ownership")
            if type(node.get("family")) is not int or type(node.get("capacity")) is not int:
                raise TypeError("typed terminal finite payload requires a bounded exact domain family")


def emit_terminal_failure(program, signature, tokens, lines, block_indices, target):
    from .program_emit_versioned import pack_local, schema_cpp
    from pops._structural_data import canonical_json
    from pops.time.canonical_data import _json_ready
    from pops.time._program.resource_budgets import validate_terminal_failure_capacities

    validate_terminal_failure_capacities(program.terminal_failure_byte_capacity,
                                        program.terminal_failure_aggregate_byte_capacity)
    per_rank = "UINT64_C(%d)" % program.terminal_failure_byte_capacity
    aggregate = "UINT64_C(%d)" % program.terminal_failure_aggregate_byte_capacity
    if target != "system" and _has_field(signature):
        raise NotImplementedError("typed terminal field payload requires a fixed-topology Uniform prototype")
    _require_supported_transport(signature)
    body = []
    value = pack_local(signature, tokens, body, "terminal_failure", block_indices,
                       capacity=per_rank)
    schema = schema_cpp(signature, block_indices)
    identity = canonical_json({"schema": "pops.typed-failure@1", "type": _json_ready(signature)})
    lines.append("pops::runtime::program::reject_typed_failure<pops::kNativeDimension>(")
    lines.append("  ctx.prepared_execution_lane(), [&]() {")
    lines.extend("    " + line for line in body)
    lines.append("    return std::make_tuple(%s, %s, std::string(%s));" % (schema, value, json.dumps(identity)))
    lines.append("  }, [&](int owner) -> const pops::MultiFab<pops::kNativeDimension>& {")
    lines.append("    return ctx.state(owner);" if _has_field(signature) else
                 '    throw std::logic_error("field-free typed failure has no field prototype");')
    limits = "pops::runtime::program::TypedFailureCapacity{%s,%s}" % (per_rank, aggregate)
    registry = ", ctx.distributed_registry_if_present()" if _has_distributed(signature) else ""
    lines.append("  }, %s%s);" % (limits, registry))
