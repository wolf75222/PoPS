"""Native installation metadata for the explicit complete AMR forest."""
from __future__ import annotations

import json

from pops.time._program.continuation_forest import forest_node


def configuration_install(program):
    node = forest_node(program)
    if node is None:
        return ""
    from .program_emit_versioned import configuration_cpp
    from .program_lowerability import all_ops
    policies = {(v.attrs["storage_bytes"], v.attrs["storage_versions"])
                for v in all_ops(program) if "storage_bytes" in v.attrs}
    if len(policies) > 1:
        raise ValueError("AMR forest has conflicting distributed storage policies")
    policy = next(iter(policies), (0, 0))
    metadata = node.attrs["continuation_forest"]
    lines = []
    for scope in ("root", "level"):
        source = configuration_cpp(metadata[scope + "_schema"], program._block_indices(),
                                   distributed_storage=policy)
        lines.append("auto _forest_%s_configuration = ctx.continuation_local_phase([&] { return %s; },"
                     ' "AMR forest schema construction");' % (scope, source))
    lines.append("ctx.install_continuation_forest(std::move(_forest_root_configuration),"
                 " std::move(_forest_level_configuration));")
    return "\n".join(lines) + "\n"


def checkpoint_exports(program):
    node = forest_node(program)
    metadata = {} if node is None else node.attrs["continuation_forest"]
    lines = []
    for scope in ("root", "level"):
        schema = metadata.get(scope + "_schema", {})
        prefix = "pops_program_checkpoint_continuation_" + scope
        lines.append('extern "C" const char* %s_schema_identity() { return %s; }'
                     % (prefix, json.dumps(schema.get("identity", ""))))
        lines.append('extern "C" std::uint64_t %s_byte_capacity() { return %d; }'
                     % (prefix, schema.get("byte_capacity", 0)))
    budget = getattr(program, "history_field_metadata_budget", None)
    from pops.time._program.resource_budgets import validate_history_field_metadata_budget
    validate_history_field_metadata_budget(budget)
    for suffix, capacity in zip(("storage_bytes", "workspace_bytes"), budget or (0, 0), strict=True):
        lines.append('extern "C" std::uint64_t pops_program_history_field_metadata_%s() { return %d; }'
                     % (suffix, capacity))
    return "\n".join(lines) + "\n"
