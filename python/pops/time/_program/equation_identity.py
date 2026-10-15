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

    from pops._structural_data import transform_tree

    def describe(item: Any) -> Any:
        if isinstance(item, ProgramValue):
            node = program._canonical_value(item)
            if node.id in active:
                raise SolveRequestError(
                    "explicit_dependency_cycle",
                    "equation inputs contain an explicit SSA cycle",
                )
            if node.id in indices:
                return None, {"value": indices[node.id]}
            index = len(nodes)
            indices[node.id] = index
            nodes.append(None)
            active.add(node.id)
            data = {
                "name": node.name,
                "op": node.op,
                "value_type": node.vtype,
                "block": _json_ready(node.block),
                "quantity": _json_ready(node.state_ref),
                "space": _json_ready(node.space),
                "point": node.point.to_data(),
            }
            attributes = tuple(
                (key, part)
                for key, part in node.attrs.items()
                if not key.endswith("_region") and key != "solve_request"
            )
            width = len(node.inputs)

            def finish(parts):
                data["inputs"] = parts[:width]
                data["attrs"] = dict(
                    zip((key for key, _ in attributes), parts[width:], strict=True)
                )
                active.remove(node.id)
                nodes[index] = data
                return {"value": index}

            return (*node.inputs, *(part for _, part in attributes)), finish
        if isinstance(item, _Affine):
            terms = tuple(item.terms)
            return tuple(node for node, _ in terms), lambda children: {
                "affine": [
                    [child, _json_ready(coefficient.to_polynomial())]
                    for child, (_, coefficient) in zip(children, terms, strict=True)
                ]
            }
        if isinstance(item, Mapping):
            entries = tuple(item.items())
            return tuple(part for _, part in entries), lambda children: dict(
                zip((key for key, _ in entries), children, strict=True)
            )
        if isinstance(item, (tuple, list)):
            return item, list
        if isinstance(item, (float, Decimal, Fraction)):
            from pops.identity.scalar import scalar_data

            return None, scalar_data(item)
        return None, _json_ready(item)

    try:
        root = transform_tree(value, describe)
    except ValueError as error:
        if str(error) == "structural data cannot contain a cycle":
            raise SolveRequestError(
                "explicit_dependency_cycle",
                "equation inputs contain an explicit SSA cycle",
            ) from error
        raise
    return _json_ready({"root": root, "values": nodes} if nodes else root)
