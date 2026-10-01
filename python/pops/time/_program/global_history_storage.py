"""Explicit storage scope for a consumed global solved-field observation."""
from __future__ import annotations

from typing import Any

CONTRACT = "pops.program.global-field-history-storage@1"


def storage_contract(program: Any, source: Any, owner: Any) -> dict[str, Any]:
    from pops.fields._observation_contract import validate_field_observation
    from pops.problem.handles import BlockHandle

    if type(owner) is not BlockHandle:
        raise TypeError("global field history owner_block requires an exact BlockHandle")
    if source.prog is not program or program._issued_values.get(id(source)) is not source \
            or program._canonical_value(source) is not source:
        raise ValueError("global field history source is not the current issued observation")
    if source.block is not None or source.state_ref is not None or source.space is not None:
        raise ValueError("global field history cannot override physical State ownership")
    _width, _component, solve = validate_field_observation(source)
    from pops.codegen.program_field_plan import _nodes, _reachable
    reachable = _reachable(solve, _nodes(program))
    field = solve.attrs.get("field")
    if field is None:
        from pops.model import Handle
        native = next((value for value in reachable if value.op == "field_problem_load"), None)
        if native is None:
            raise NotImplementedError("global history storage @1 has no realized physical field authority")
        field = Handle.from_canonical_identity(native.attrs["field_handle"])
    if source.point != source.inputs[0].point or source.point != solve.point \
            or source.region != source.inputs[0].region:
        raise ValueError("global field history changes the consumed solve point or region")
    allocation = next((value.block for value in solve.inputs if value.block is not None), None)
    if allocation is None:
        allocation = next((value.block for value in reachable
                           if value.state_ref is not None and value.block is not None), None)
    if allocation is None:
        raise ValueError("global field history has no explicit solved layout witness")
    if owner.owner_path.canonical() != field.owner_path.canonical():
        raise ValueError("global field history storage belongs to a foreign Case")
    registry = owner._instance_registry
    if registry is not None and allocation._instance_registry is not None \
            and registry is not allocation._instance_registry:
        raise ValueError("global field history storage belongs to a foreign Case registry")
    if registry is not None:
        if registry.handles().get(owner.local_id) is not owner:
            raise ValueError("global field history storage block is an unissued alias")
        issued = registry.handles().get(allocation.local_id)
        if issued is None or issued != allocation:
            raise ValueError("global field history has a foreign solved layout witness")
        if registry._blocks[owner.local_id]["model"].frame != \
                registry._blocks[allocation.local_id]["model"].frame:
            raise ValueError("global field history storage changes the solved physical frame")
    declared = [state for state in program._time_states.values() if state.block is owner]
    if not declared:
        raise ValueError("global field history storage block requires an issued Program TimeState scope")
    matching = [state for state in declared if source.point is not None and state.clock == source.point.clock]
    if not matching:
        raise ValueError("global field history storage changes the observation clock")
    if any(state.state != matching[0].state for state in matching[1:]):
        raise ValueError("global field history storage has ambiguous allocation State authority")
    # No borrow of StateSpace, state identity, temperature or field ownership.
    return {"contract":CONTRACT, "owner_block":owner, "clock":source.point.clock,
        "point":source.point, "region":source.region, "layout_witness":allocation,
        "storage_state_witness":matching[0].state, "field_problem_identity":source.attrs["field_problem_identity"],
        "field_unknown":source.attrs["field_unknown"], "ncomp":1,
        "sampling":"cell", "representation":"valid_cell_copy"}


def validate_storage_node(program: Any, node: Any) -> dict[str, Any]:
    if node.op != "store_history" or len(node.inputs) != 1:
        raise ValueError("global field history storage requires exactly one observation")
    expected = storage_contract(program, node.inputs[0], node.block)
    from pops.time._program.serialization import _json_ready
    if _json_ready(node.attrs.get("global_field_storage")) != _json_ready(expected) \
            or node.point != node.inputs[0].point or node.state_ref is not None or node.space is not None:
        raise ValueError("global field history storage changed its immutable authority")
    if program._history_blocks.get(node.attrs["history"]) is not node.block \
            or program._histories_ncomp.get(node.attrs["history"]) != 1:
        raise ValueError("global field history registration changed its owner or width")
    return expected


def descriptor(program: Any, name: str) -> str | None:
    import json
    from pops.time._program.serialization import _json_ready

    stores = [node for node in program._values if node.op == "store_history"
        and node.attrs.get("history") == name and "global_field_storage" in node.attrs]
    if not stores:
        return None
    images = [_json_ready(validate_storage_node(program, node)) for node in stores]
    if any(image != images[0] for image in images[1:]):
        raise ValueError("global field history storage changes its registered physical observation")
    return json.dumps(images[0], sort_keys=True, separators=(",", ":"))
