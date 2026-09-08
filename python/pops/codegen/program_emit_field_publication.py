"""Publish already-consumed field values through the exact native provider transaction."""
from __future__ import annotations

import json
from typing import Any

from pops.fields._program_publication import validate_field_publication
from pops.identity import canonical_bytes
from pops.time._program.serialization import _json_ready


def emit_field_publication(value: Any, var: Any, lines: list[str], model: Any, *, target: str, provider_plans: Any, block_idx: Any) -> None:
    from .program_models import ProgramModelGraph
    from .program_field_publication import _key, provider_identity
    from .component_provider_packs import require_emitter_provider_carrier

    if target != "system" or type(model) is not ProgramModelGraph:
        raise ValueError("consumed field publication requires an exact Uniform Program model graph")
    bindings = validate_field_publication(value)
    if provider_plans is None:
        raise ValueError("field publication requires authenticated provider prerequisite plans")
    rows = []
    destinations = {}
    for row, source in zip(bindings, value.inputs[:len(bindings)], strict=True):
        destination = row["target"]
        emitter = model.model_for_block(destination.block_ref)
        destinations[destination.block_ref] = (destination, emitter)
        require_emitter_provider_carrier(emitter, where="consumed field publication")
        module = model.source_module_for_owner(destination.declaration_ref.owner_path)
        key = _key(module, destination, row["component"])
        plans = emitter._resolved_operations.provider_evidence.get("program_field_publications", ())
        matching = tuple(claim for claim in plans if dict(claim["key"]) == key.to_data())
        if len(matching) != 1 or matching[0]["producer"] != value.attrs["field_problem_identity"]:
            raise ValueError("field publication has no exact resolved provider claim")
        claim = matching[0]
        observed = source if source.op == "field_component" else source.inputs[0]
        if claim["observation"] != source.op or claim["source_component"] != row["source_component"] \
                or canonical_bytes(_json_ready(claim["unknown"])) != \
                canonical_bytes(_json_ready(observed.attrs["field_unknown"])):
            raise ValueError("field publication differs from its resolved output observation")
        key_cpp = "{%s}" % ", ".join(json.dumps(part) for part in key.to_data().values())
        rows.append("{%s, %s, &%s, %d}" % (
            key_cpp, json.dumps(provider_identity(claim)), var[source.id], row["source_component"]))
    from .program_emit_ops import _required_block_index
    from .program_emit_kernels import _prepare_provider_values, program_provider_consumer_qid
    from .program_field_publication import remaining_input_pack
    from pops.fields._program_publication import publication_states

    states = publication_states(value)
    for block, (destination, emitter) in destinations.items():
        require_emitter_provider_carrier(emitter, where="field publication prerequisites")
        pack = emitter._auxiliary_provider_pack
        # The shared FieldSpace is immediately available: consume solve outputs only after
        # publishing its remaining exact runtime-input prerequisites at this same state.
        published = {_key(model.source_module_for_owner(row["target"].declaration_ref.owner_path),
                          row["target"], row["component"])
                     for row in bindings if row["target"].block_ref == block}
        selected = remaining_input_pack(pack, destination, value.space, published)
        if not selected:
            continue
        binding = provider_plans.bind_pack(selected,
            program_provider_consumer_qid(emitter, value.id, block))
        lines.extend(_prepare_provider_values(binding,
            _required_block_index(block_idx, block, "field publication prerequisites"),
            var[states[block].id]))
    publication_identity = "%s/publication/%d" % (value.attrs["field_problem_identity"], value.id)
    lines.append("ctx.publish_field_components(%d, %s, {%s});" % (
        value.id, json.dumps(publication_identity), ", ".join(rows)))
    var[value.id] = var[value.inputs[0].id]
