"""Retain the complete structural error at the compiled Program transaction boundary."""
from __future__ import annotations

import json


def _has_field(signature):
    if signature["vtype"] in ("state", "scalar_field"):
        return True
    return any(_has_field(child) for _, child in signature.get("members", ())) or (
        "element" in signature and _has_field(signature["element"]))


def _has_distributed(signature):
    if signature.get("ownership") == "contiguous_distributed":
        return True
    return any(_has_distributed(child) for _, child in signature.get("members", ())) or (
        "element" in signature and _has_distributed(signature["element"]))


def _require_supported_transport(signature):
    if signature["vtype"] in ("array", "index_set", "index"):
        if signature.get("ownership") not in ("replicated", "contiguous_distributed"):
            raise NotImplementedError("typed terminal finite payload requires an exact supported ownership")
        if type(signature.get("family")) is not int or type(signature.get("capacity")) is not int:
            raise TypeError("typed terminal finite payload requires a bounded exact domain family")
    for _, child in signature.get("members", ()):
        _require_supported_transport(child)
    if "element" in signature:
        _require_supported_transport(signature["element"])


def emit_terminal_failure(signature, tokens, lines, block_indices, target):
    from .program_emit_versioned import pack_local, schema_cpp
    from pops.time.canonical_data import _json_ready

    if target != "system" and _has_field(signature):
        raise NotImplementedError("typed terminal field payload requires a fixed-topology Uniform prototype")
    _require_supported_transport(signature)
    body = []
    value = pack_local(signature, tokens, body, "terminal_failure", block_indices)
    schema = schema_cpp(signature, block_indices)
    identity = json.dumps({"schema": "pops.typed-failure@1", "type": _json_ready(signature)},
                          sort_keys=True, separators=(",", ":"))
    lines.append("pops::runtime::program::reject_typed_failure<pops::kNativeDimension>(")
    lines.append("  ctx.prepared_execution_lane(), [&]() {")
    lines.extend("    " + line for line in body)
    lines.append("    return std::make_tuple(%s, %s, std::string(%s));" % (schema, value, json.dumps(identity)))
    lines.append("  }, [&](int owner) -> const pops::MultiFab<pops::kNativeDimension>& {")
    lines.append("    return ctx.state(owner);" if _has_field(signature) else
                 '    throw std::logic_error("field-free typed failure has no field prototype");')
    if _has_distributed(signature):
        lines.append("  }, {}, ctx.distributed_registry_if_present());")
    else:
        lines.append("  });")
