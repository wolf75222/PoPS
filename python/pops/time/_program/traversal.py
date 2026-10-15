"""Structural traversal of the authored Program, independent of lowering."""
from __future__ import annotations

from typing import Any

_STRUCTURED_BLOCK_KEYS = (
    "cond_block",
    "body_block",
    "apply_block",
    "residual_block",
    "true_block",
    "false_block",
)


def all_ops_with_ancestry(program: Any) -> Any:
    """Yield ``(node, ancestry)`` recursively with authenticated structured-region ownership."""
    seen: set[int] = set()

    def walk(value: Any, ancestry: tuple[tuple[int, str], ...]) -> Any:
        value_id = getattr(value, "id", None)
        region = getattr(value, "region", None)
        if type(value_id) is not int or value_id < 0:
            raise TypeError("Program node identity must be a non-negative exact integer")
        if value_id in seen:
            raise ValueError("Program node %d appears in multiple structured regions" % value_id)
        if type(region) is not int or region < 0:
            raise TypeError("Program node %d has an invalid structured region" % value_id)
        if (region == 0) != (not ancestry):
            raise ValueError(
                "Program node %d structured ancestry disagrees with region %d"
                % (value_id, region)
            )
        seen.add(value_id)
        yield value, ancestry
        for key in _STRUCTURED_BLOCK_KEYS:
            block = value.attrs.get(key)
            if isinstance(block, (list, tuple)):
                for nested in block:
                    yield from walk(nested, ancestry + ((value_id, key),))

    for value in program._values:
        yield from walk(value, ())


def all_ops(program: Any) -> Any:
    """Iterate every node recursively, including lazy branch and solver sub-regions."""
    for value, _ancestry in all_ops_with_ancestry(program):
        yield value
