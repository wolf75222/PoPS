"""One frozen identity for scalar histories used exclusively for inspection/export."""
from __future__ import annotations

import json
from typing import Any


SCALAR_OUTPUT_HISTORY_SPACE = "scalar-output-field-v1"
OUTPUT_HISTORY_PROJECTION_SPACE = "pops.program.scalar-output-history-projection@2"


def history_space_identity(program: Any, name: str) -> str:
    """Qualify a typed store-only ring; preserve the legacy width-one/two-slot profile.

    This is a native transfer capability, so follow aliases, expression inputs and
    every recorded subblock exactly as Program liveness does. A consumer hidden in
    a matrix-free apply or branch must retain the ordinary history contract.
    """
    space = getattr(program, "_history_spaces", {}).get(name)
    if space is not None:
        return json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
    owner = getattr(program, "_history_blocks", {}).get(name)
    width = getattr(program, "_histories_ncomp", {}).get(name)
    lag = program._histories.get(name)
    if (owner is None or getattr(program, "_history_state_refs", {}).get(name) is not None
            or type(width) is not int or width < 1 or type(lag) is not int or lag < 1):
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
    if not stored:
        return "scalar-field"
    # IR16 stores authenticate the physical observation independently of their
    # allocation State. Keep that exact descriptor in state_identity; this URI
    # qualifies only its spatial transfer, never a State/Q-to-observation copy.
    from pops.time._program.global_history_storage import descriptor

    storage = descriptor(program, name)
    if width == 1 and lag == 1 and storage is None:
        return SCALAR_OUTPUT_HISTORY_SPACE
    return OUTPUT_HISTORY_PROJECTION_SPACE
