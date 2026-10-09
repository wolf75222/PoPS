"""Explicit storage scope for a consumed global solved-field observation."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops._frozen_data import freeze_containers, thaw_data

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
    from pops.time._graph.value_traversal import walk_program_nodes, reachable_values
    reachable = reachable_values(solve, tuple(walk_program_nodes(program._values)))
    field = solve.attrs.get("field")
    if field is None:
        from pops.model import Handle
        native = next((value for value in reachable if value.op == "field_problem_load"), None)
        if native is None:
            raise NotImplementedError("global history storage @1 has no realized physical field authority")
        field = Handle.from_canonical_identity(thaw_data(native.attrs["field_handle"]))
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
    from pops.time.points import point_clock
    clock = point_clock(source.point, "global field history storage")
    matching = [state for state in declared if state.clock == clock]
    if not matching:
        raise ValueError("global field history storage changes the observation clock")
    if any(state.state != matching[0].state for state in matching[1:]):
        raise ValueError("global field history storage has ambiguous allocation State authority")
    # No borrow of StateSpace, state identity, temperature or field ownership.
    return {"contract":CONTRACT, "owner_block":owner, "clock":clock,
        "point":source.point, "region":source.region, "layout_witness":allocation,
        "storage_state_witness":matching[0].state, "field_problem_identity":source.attrs["field_problem_identity"],
        "field_unknown":source.attrs["field_unknown"], "ncomp":1,
        "sampling":"cell", "representation":"valid_cell_copy"}


_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class _IssuedStorage:
    """An original storage declaration, never reconstructed from mutable IR projections."""
    metadata: Any

    def __init__(self, metadata: Any, *, issuer: Any) -> None:
        if issuer is not _ISSUER:
            raise TypeError("global history authority requires its private issuer")
        object.__setattr__(self, "metadata", freeze_containers(metadata))


def _equal_metadata(left: Any, right: Any) -> bool:
    import json
    from pops.time.canonical_data import _json_ready

    # Container freezing may replace list/dict by tuple/MappingProxyType; _json_ready
    # restores their one wire image. JSON tokens retain bool/int/float distinctions
    # which Python container equality discards (True == 1 == 1.0).
    def image(value: Any) -> str:
        return json.dumps(_json_ready(value), sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    return image(left) == image(right)


def prepare_issuance(program: Any, name: str, metadata: Any) -> _IssuedStorage:
    """Validate a repeated store without changing an already issued ring authority."""
    issued = getattr(program, "_global_field_history_issuance", {}).get(name)
    if issued is not None:
        if type(issued) is not _IssuedStorage or not _equal_metadata(issued.metadata, metadata):
            raise ValueError("global field history changed its originally issued storage authority")
        return issued
    return _IssuedStorage(metadata, issuer=_ISSUER)


def publish_issuance(program: Any, name: str, issued: _IssuedStorage) -> None:
    current = dict(getattr(program, "_global_field_history_issuance", {}))
    if name in current and current[name] is not issued:
        raise ValueError("global field history cannot replace an issued storage authority")
    current[name] = issued
    object.__setattr__(program, "_global_field_history_issuance", MappingProxyType(current))


def validate_storage_node(program: Any, node: Any) -> dict[str, Any]:
    if node.op != "store_history" or len(node.inputs) != 1:
        raise ValueError("global field history storage requires exactly one observation")
    issued = getattr(program, "_global_field_history_issuance", {}).get(node.attrs["history"])
    if type(issued) is not _IssuedStorage:
        raise ValueError("global field history lost its originally issued storage authority")
    expected = storage_contract(program, node.inputs[0], node.block)
    if not _equal_metadata(issued.metadata, expected):
        raise ValueError("global field history changed its originally issued storage authority")
    if not _equal_metadata(node.attrs.get("global_field_storage"), issued.metadata) \
            or node.point != node.inputs[0].point or node.state_ref is not None or node.space is not None:
        raise ValueError("global field history storage changed its immutable authority")
    width = program._histories_ncomp.get(node.attrs["history"])
    if program._history_blocks.get(node.attrs["history"]) is not node.block \
            or type(width) is not int or width != 1:
        raise ValueError("global field history registration changed its owner or width")
    return expected


def validate_issuances(program: Any) -> None:
    """Authenticate source declarations before cloning, detaching or freezing any projections."""
    from .spatial_interaction import validate_closed_issuances
    validate_closed_issuances(program)
    if not getattr(program, "_global_field_history_issuance", None):
        return
    from pops.time._graph.value_traversal import walk_program_nodes
    found = set()
    for node in walk_program_nodes(program._values):
        if node.op == "store_history" and node.attrs.get("history") in program._global_field_history_issuance:
            validate_storage_node(program, node)
            found.add(node.attrs["history"])
    if found != set(program._global_field_history_issuance):
        raise ValueError("global field history lost an originally issued storage operation")


def transfer_issuances(source: Any, target: Any, remap_metadata: Any, history_names: Any) -> None:
    """Transfer only the already authenticated, original immutable declarations."""
    from .spatial_interaction import transfer_closed_issuances
    transfer_closed_issuances(source, target, remap_metadata)
    issued = getattr(source, "_global_field_history_issuance", {})
    if issued:
        object.__setattr__(target, "_global_field_history_issuance", MappingProxyType({
            name: _IssuedStorage(remap_metadata(record.metadata), issuer=_ISSUER)
            for name, record in issued.items() if name in history_names
        }))


def descriptor(program: Any, name: str) -> str | None:
    import json
    from pops.time.canonical_data import _json_ready

    validate_issuances(program)
    stores = [node for node in program._values if node.op == "store_history"
        and node.attrs.get("history") == name and "global_field_storage" in node.attrs]
    if not stores:
        return None
    images = [_json_ready(validate_storage_node(program, node)) for node in stores]
    if any(image != images[0] for image in images[1:]):
        raise ValueError("global field history storage changes its registered physical observation")
    return json.dumps(images[0], sort_keys=True, separators=(",", ":"))
