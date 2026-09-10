"""Structural solve equation identity shared by linear and spatial authoring."""
from __future__ import annotations
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any
from pops.time.solve_request import SolveRequestError
from pops.time.values import ProgramValue, _Affine


def _equation_value(program: Any, value: Any) -> Any:
    """Snapshot the evaluated closure, without seed data or diagnostic source locations.

    References are traversed structurally so harmless SSA renumbering at detachment
    does not change the equation. A real explicit cycle remains a refusal. Iteration
    placeholders inside an authored apply region terminate this traversal.
    """
    from pops.time.canonical_data import _json_ready

    active: set[int] = set()
    indices: dict[int, int] = {}
    nodes: list[Any] = []

    def walk(item: Any) -> Any:
        if isinstance(item, ProgramValue):
            node = program._canonical_value(item)
            if node.id in active:
                raise SolveRequestError(
                    "explicit_dependency_cycle", "equation inputs contain an explicit SSA cycle")
            if node.id in indices:
                return {"value": indices[node.id]}
            index = len(nodes)
            indices[node.id] = index
            nodes.append(None)
            active.add(node.id)
            data = {
                "name": node.name, "op": node.op, "value_type": node.vtype,
                "block": _json_ready(node.block), "quantity": _json_ready(node.state_ref),
                "space": _json_ready(node.space), "point": node.point.to_data(),
                "inputs": [walk(source) for source in node.inputs],
                "attrs": {
                    key: walk(part) for key, part in node.attrs.items()
                    if not key.endswith("_region") and key != "solve_request"
                },
            }
            active.remove(node.id)
            nodes[index] = data
            return {"value": index}
        if isinstance(item, _Affine):
            return {"affine": [[walk(node), _json_ready(coefficient.to_polynomial())]
                               for node, coefficient in item.terms]}
        if isinstance(item, Mapping):
            return {key: walk(part) for key, part in item.items()}
        if isinstance(item, (tuple, list)):
            return [walk(part) for part in item]
        if isinstance(item, (float, Decimal, Fraction)):
            from pops.identity.scalar import scalar_data

            return scalar_data(item)
        return _json_ready(item)

    root = walk(value)
    return _json_ready({"root": root, "values": nodes} if nodes else root)

