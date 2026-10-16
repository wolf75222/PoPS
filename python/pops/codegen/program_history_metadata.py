"""Frozen history tuples shared by Uniform and AMR compiled Programs."""
from __future__ import annotations

import json
from typing import Any


def emit_checkpoint_history_metadata(program: Any, *, require_owner: bool) -> str:
    block_idx = program._block_indices()
    temporal = program.temporal_manifest()
    history_manifest = {row["name"]: row for row in temporal["histories"]}
    histories = []
    for name, lag in sorted(program._histories.items()):
        owner = getattr(program, "_history_blocks", {}).get(name)
        owner_index = block_idx.get(owner) if owner is not None else None
        if owner_index is None:
            if require_owner:
                raise ValueError("AMR history %r requires explicit block owner provenance" % name)
            # Legacy Uniform owner-less rings retain their old allocation route;
            # they cannot issue an independent output-projection capability.
            continue
        state_ref = getattr(program, "_history_state_refs", {}).get(name)
        state_identity = (
            state_ref.qualified_id if state_ref is not None else "scalar-history:" + name
        )
        # Match the exact IR16 storage authority registered by the live prelude.
        # The scalar observation keeps its own space; its storage owner is not
        # permission to replace it by that owner's physical State identity.
        from pops.time._program.global_history_storage import descriptor
        storage_descriptor = descriptor(program, name)
        if storage_descriptor is not None:
            state_identity = storage_descriptor
        from pops.codegen.program_history_identity import history_space_identity, history_ghost_depth
        space_identity = history_space_identity(program, name)
        row = history_manifest[name]
        interpolation = json.dumps(row["interpolation"], sort_keys=True, separators=(",", ":"))
        component = getattr(program, "_histories_ncomp", {}).get(name)
        histories.append(
            (
                str(name),
                int(owner_index),
                str(state_identity),
                str(space_identity),
                str(row["clock"]),
                interpolation,
                int(lag) + 1,
                -1 if component is None else int(component),
                history_ghost_depth(program, name),
            )
        )

    def string_accessor(symbol: str, column: int, rows: Any) -> str:
        cases = "".join(
            "    case %d: return %s;\n" % (index, json.dumps(row[column]))
            for index, row in enumerate(rows)
        )
        return (
            f'extern "C" const char* {symbol}(int index) {{\n'
            "  switch (index) {\n" + cases + '    default: return "";\n'
            "  }\n"
            "}\n"
        )

    def integer_accessor(symbol: str, column: int) -> str:
        cases = "".join(
            "    case %d: return %d;\n" % (index, row[column])
            for index, row in enumerate(histories)
        )
        return (
            f'extern "C" int {symbol}(int index) {{\n'
            "  switch (index) {\n" + cases + "    default: return 0;\n"
            "  }\n"
            "}\n"
        )

    return (
        f'extern "C" int pops_program_checkpoint_history_count() {{ return {len(histories)}; }}\n'
        + string_accessor("pops_program_checkpoint_history_name", 0, histories)
        + integer_accessor("pops_program_checkpoint_history_owner", 1)
        + string_accessor("pops_program_checkpoint_history_state_identity", 2, histories)
        + string_accessor("pops_program_checkpoint_history_space_identity", 3, histories)
        + string_accessor("pops_program_checkpoint_history_clock_identity", 4, histories)
        + string_accessor("pops_program_checkpoint_history_interpolation_identity", 5, histories)
        + integer_accessor("pops_program_checkpoint_history_depth", 6)
        + integer_accessor("pops_program_checkpoint_history_components", 7)
        + integer_accessor("pops_program_checkpoint_history_ghost_depth", 8)
    )
