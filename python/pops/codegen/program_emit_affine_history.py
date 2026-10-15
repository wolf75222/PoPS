"""Install artifact-authored rectangular history maps for parent acceptance."""
import json
from pops._frozen_data import thaw_data
from pops.codegen.program_emit_history_transfer import emit_history_transfer_callback
from pops.codegen.program_lowerability import all_ops
from pops.time._program.affine_history import validate_history_contract


def affine_history_declarations(program):
    nodes = [value for value in all_ops(program) if value.op == "accept_increment"]
    if not nodes: return []
    rows = nodes[0].attrs.get("histories", {})
    validate_history_contract(program, rows)
    lines = []
    for name, row in sorted(rows.items()):
        data = row["transfer"]
        identity = json.dumps(thaw_data(data), sort_keys=True, separators=(",", ":"))
        lines.append("ctx.register_affine_history(%s, %s, %d, %s," % (
            json.dumps(name), json.dumps(identity), program._histories[name]+1,
            "true" if row["state_endpoints"] else "false"))
        lines.extend(emit_history_transfer_callback(data))
        lines.append(");")
    return lines
