"""Typed stencil-access capability carried by compiled time-Program nodes.

The capability describes storage access, not a named numerical scheme.  A
consumer composes a subgraph by taking the maximum required ghost depth, so a
new operation participates without being added to a central opcode table.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class StencilAccess:
    """Immutable spatial-access requirement of one Program operation.

    ``required_ghost_depth`` is the number of neighbour layers the operation
    may read.  Pointwise operations use zero.  Extension-owned operations must
    attach this capability explicitly; omission is rejected when an apply
    subgraph is composed.
    """

    required_ghost_depth: int = 0

    schema_version: ClassVar[int] = 1
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        depth = self.required_ghost_depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError(
                "StencilAccess.required_ghost_depth must be a non-negative integer"
            )

    @classmethod
    def pointwise(cls) -> StencilAccess:
        return cls(0)

    @classmethod
    def nearest_neighbour(cls) -> StencilAccess:
        return cls(1)

    @classmethod
    def compose(cls, capabilities: Iterable[Any], *, where: str) -> StencilAccess:
        """Compose exact typed capabilities without inspecting operation names."""
        depth = 0
        for capability in capabilities:
            if type(capability) is not cls:
                raise TypeError(
                    "%s contains an operation without an exact StencilAccess capability"
                    % where
                )
            depth = max(depth, capability.required_ghost_depth)
        return cls(depth)

    def to_data(self) -> dict[str, int]:
        return {
            "schema_version": self.schema_version,
            "required_ghost_depth": self.required_ghost_depth,
        }


__all__ = ["StencilAccess"]
