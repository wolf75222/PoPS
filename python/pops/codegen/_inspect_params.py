"""Parameter inspection derived from a compiled artifact's BindSchema."""
from __future__ import annotations

from typing import Any


def build_parameter_arguments(compiled: Any, params: Any) -> dict[str, dict[str, Any]]:
    """Return qualified argument rows without consulting mutable authoring state."""
    schema = getattr(compiled, "bind_schema", None)
    if schema is None:
        # Low-level handles compiled without a Problem have no BindSchema.  Public
        # pops.compile never takes this best-effort inspection branch.
        result = {}
        for name, param in params.items():
            kind = getattr(param, "kind", "const")
            kind = getattr(kind, "value", kind)
            ptype = getattr(param, "type", None) or type(
                getattr(param, "value", 0.0)
            ).__name__
            result[name] = {
                "type": str(ptype),
                "kind": str(kind),
                "required": kind == "runtime",
            }
        return result

    result = {}
    for slot in schema.slots:
        declaration = slot.to_dict()["declaration"]
        result[slot.qid] = {
            "name": slot.handle.local_id,
            "ordinal": slot.ordinal,
            "handle": slot.handle.canonical_identity(),
            "type": declaration["dtype"],
            "dtype": declaration["dtype"],
            "kind": declaration["kind"],
            "storage": declaration["storage"],
            "phase": declaration["phase"],
            "invalidation": declaration["invalidation"],
            "unit": declaration["unit"],
            "domain": declaration["domain"],
            "default": declaration["default"],
            "provenance": declaration["provenance"],
            "required": slot.required,
        }
    return result


__all__ = ["build_parameter_arguments"]
