"""Reads of issued independent Field histories keep their scientific provider."""
from __future__ import annotations

from typing import Any


def _store(program: Any, name: str) -> Any:
    from .global_history_storage import validate_storage_node
    from pops.time._graph.value_traversal import walk_program_nodes
    stores = [node for node in walk_program_nodes(program._values)
              if node.op == "store_history" and node.attrs.get("history") == name
              and "global_field_storage" in node.attrs]
    if not stores:
        raise ValueError("independent history read requires an issued storage declaration")
    for node in stores:
        validate_storage_node(program, node)
    return stores[0]


def read(program: Any, name: str, lag: int, *, ncomp: Any, space: Any,
         block: Any, state_ref: Any, contract: Any) -> Any:
    from pops.time.points import TimePoint
    store = _store(program, name)
    source = store.inputs[0]
    if state_ref is not None or contract is not None:
        raise ValueError("independent history read cannot borrow State history authority")
    if block is not None and block is not store.block:
        raise ValueError("independent history read changes its allocation owner")
    if space is not None and space != source.space:
        raise ValueError("independent history read changes its physical FieldSpace")
    width = program._histories_ncomp[name]
    if ncomp is not None and (type(ncomp) is not int or ncomp != width):
        raise ValueError("independent history read changes its component count")
    if lag > program._histories[name]:
        raise ValueError("independent history read exceeds its declared maximum lag")
    return program._new("scalar_field", "history", (), {
        "history": name, "lag": lag, "ncomp": width,
        "global_field_history_read": True,
        "field_authority": source.attrs["field_authority"],
    }, name, None, space=source.space, field_context=source.field_context,
        point=TimePoint(source.clock, step=-lag), inherit_state_ref=False)


def validate_read(program: Any, node: Any) -> Any:
    from pops.time.points import TimePoint
    from .global_history_storage import _equal_metadata
    if node.op != "history" or node.attrs.get("global_field_history_read") is not True:
        raise ValueError("independent history read lacks its issued projection")
    name = node.attrs.get("history")
    store = _store(program, name)
    source = store.inputs[0]
    lag = node.attrs.get("lag")
    width = program._histories_ncomp[name]
    if type(lag) is not int or not 1 <= lag <= program._histories[name]:
        raise ValueError("independent history read has a foreign lag")
    if (node.vtype != "scalar_field" or node.inputs or node.block is not None
            or node.state_ref is not None or node.space != source.space
            or type(node.attrs.get("ncomp")) is not int or node.attrs["ncomp"] != width
            or node.point != TimePoint(source.clock, step=-lag)
            or not _equal_metadata(node.attrs.get("field_authority"), source.attrs["field_authority"])):
        raise ValueError("independent history read changed its issued scientific authority")
    return store
