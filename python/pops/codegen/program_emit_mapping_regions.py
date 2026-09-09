"""C++ continuation lowering at exact Program map ports."""
from __future__ import annotations
import json
import re


def emit_map_port(value, var, lines, block_indices):
    if value.op == "layout_map_export":
        var[value.id] = var[value.inputs[0].id]
    else:
        var[value.id] = "u%d" % value.id
        lines.append("auto& %s = ctx.scratch_state(%d, 0, %s);" %
                     (var[value.id], value.id, var[value.inputs[0].id]))


def open_map_continuation(value, values, var, lines):
    # Numerical buffers are owned by ctx_owner, not by a stack frame. C++ reference captures bind
    # those objects directly; scalar temporaries and provider handles are preserved by value.
    references = sorted({var[node.id] for node in values
                         if node.id in var and node.vtype in ("state", "rhs", "scalar_field")
                         and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", var[node.id])})
    capture = "=, " + ", ".join("&" + name for name in references) if references else "="
    lines.append("ctx.suspend_map(%s, %s, %s, [%s]() {" %
                 (json.dumps(value.attrs["invocation"]),
                  "true" if value.op == "layout_map_import" else "false", var[value.id], capture))
    lines.append("auto& ctx = *ctx_owner;")


def close_map_continuations(count, lines):
    for _ in range(count):
        lines.extend(("});", "return;"))
