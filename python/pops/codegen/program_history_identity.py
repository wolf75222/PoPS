"""One frozen identity for scalar histories used exclusively for inspection/export."""
from __future__ import annotations

import json
from typing import Any


SCALAR_OUTPUT_HISTORY_SPACE = "scalar-output-field-v1"


def history_space_identity(program: Any, name: str) -> str:
    """Qualify only a width-one, two-slot ring with stores and no Program reads.

    This is a native transfer capability, so follow aliases, expression inputs and
    every recorded subblock exactly as Program liveness does. A consumer hidden in
    a matrix-free apply or branch must retain the ordinary history contract.
    """
    space = getattr(program, "_history_spaces", {}).get(name)
    if space is not None:
        return json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
    owner = getattr(program, "_history_blocks", {}).get(name)
    if (owner is None or getattr(program, "_history_state_refs", {}).get(name) is not None
            or getattr(program, "_histories_ncomp", {}).get(name) != 1
            or program._histories.get(name) != 1):
        return "scalar-field"

    seen: set[int] = set()
    stack = [*program._values, *program._commits.values()]
    stored = False
    while stack:
        value = program._canonical_value(stack.pop())
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.attrs.get("history") == name:
            if (value.op != "store_history" or len(value.inputs) != 1
                    or value.inputs[0].vtype != "scalar_field" or value.block != owner):
                return "scalar-field"
            stored = True
        stack.extend(value.inputs)
        stack.extend(program._subblock_value_refs(value))
    return SCALAR_OUTPUT_HISTORY_SPACE if stored else "scalar-field"
