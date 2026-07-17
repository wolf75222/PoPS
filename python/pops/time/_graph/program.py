"""Closed immutable container for one validated temporal SSA graph."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pops.time._graph.base import ValueRef, nonempty
from pops.time._graph.control import Branch, Loop
from pops.time._graph.nodes import NODE_TYPES
from pops.time._graph.validation import validate_nodes
from pops.time.points import Clock


GRAPH_NODE_TYPES = (*NODE_TYPES, Branch, Loop)


@dataclass(frozen=True, slots=True, init=False)
class ProgramGraph:
    """Canonical immutable graph accepted by temporal resolve/compile phases."""

    name: str
    clocks: tuple[Clock, ...]
    nodes: tuple[Any, ...]
    graph_hash: str

    def __init__(self, name: str, nodes: Any, *, clocks: Any = None) -> None:
        object.__setattr__(self, "name", nonempty(name, where="ProgramGraph name"))
        frozen_nodes = tuple(nodes)
        if any(type(node) not in GRAPH_NODE_TYPES for node in frozen_nodes):
            raise TypeError("ProgramGraph nodes must be exact graph node records")
        declared = tuple(clocks) if clocks is not None else tuple(dict.fromkeys(
            node.clock for node in frozen_nodes))
        if any(type(clock) is not Clock for clock in declared):
            raise TypeError("ProgramGraph clocks must contain exact Clock values")
        if len(set(declared)) != len(declared):
            raise ValueError("ProgramGraph clocks must be unique")
        object.__setattr__(self, "clocks", declared)
        object.__setattr__(self, "nodes", frozen_nodes)
        available: dict[int, Any] = {}
        validate_nodes(self.nodes, self.clocks, available, where="ProgramGraph")
        payload = json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "graph_hash", hashlib.sha256(payload.encode()).hexdigest())

    def ref(self, node: Any) -> ValueRef:
        if type(node) not in GRAPH_NODE_TYPES \
                or not any(candidate is node for candidate in self.nodes):
            raise ValueError("ProgramGraph.ref requires an exact node owned by this graph")
        if not node.readable:
            raise TypeError("Commit is a write-only graph endpoint and has no readable ValueRef")
        return ValueRef(node.node_id)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "pops.program-graph",
            "name": self.name,
            "clocks": [clock.to_data() for clock in self.clocks],
            "nodes": [node.to_data() for node in self.nodes],
        }


__all__ = ["ProgramGraph"]
