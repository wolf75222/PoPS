"""Shared contract for the bounded native dense-spectral provider."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops._ir.expr import Const


class DenseSpectralCapacityError(ValueError):
    """A requested dense spectral matrix exceeds the native provider capacity."""

    def __init__(
        self,
        *,
        operation: str,
        components: int,
        max_components: int,
        alternative: str,
    ) -> None:
        self.operation = operation
        self.components = components
        self.max_components = max_components
        self.alternative = alternative
        super().__init__(
            "%s requires a %dx%d dense spectral solve, but the native bounded "
            "dense provider supports at most %d components per matrix. %s"
            % (operation, components, components, max_components, alternative)
        )


@dataclass(frozen=True)
class DenseSpectralCapability:
    """Capacity descriptor mirrored by ``dense_eig.hpp``'s device stack bound."""

    max_components: int

    def require(self, components: Any, *, operation: str, alternative: str) -> None:
        if isinstance(components, bool) or not isinstance(components, int) or components < 1:
            raise TypeError("dense spectral component count must be a positive Python int")
        if components > self.max_components:
            raise DenseSpectralCapacityError(
                operation=operation,
                components=components,
                max_components=self.max_components,
                alternative=alternative,
            )


# Single Python authority for every authoring route backed by the native fixed-size
# ``real_eig_minmax`` / ``roe_*_apply`` kernels.  The matching C++ static_assert is 16 because
# each device thread owns O(N^2) stack scratch (about 2 KiB at the bound).
DENSE_SPECTRAL = DenseSpectralCapability(max_components=16)


def is_exact_block_triangular(rows: Any, blocks: Any) -> bool:
    """Return whether ``rows`` is provably block triangular under ``blocks``.

    The proof is structural, not sampled numerically.  Every state index must belong to exactly
    one block, and the directed graph induced by exact non-zero off-diagonal block entries must be
    acyclic.  A topological ordering of that graph is precisely a permutation that makes the matrix
    block triangular, so the full spectrum is the union of the diagonal-block spectra.

    ``False`` means "not certified" rather than "not triangular": an algebraically-zero expression
    that the symbolic simplifier did not reduce to :class:`Const(0)` is deliberately refused.
    """

    try:
        matrix = tuple(tuple(row) for row in rows)
        partition = tuple(tuple(int(index) for index in block) for block in blocks)
    except (TypeError, ValueError):
        return False
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix) or not partition:
        return False
    flat = tuple(index for block in partition for index in block)
    if len(flat) != size or set(flat) != set(range(size)):
        return False
    owner = {}
    for block_index, block in enumerate(partition):
        if not block:
            return False
        for index in block:
            if index in owner:
                return False
            owner[index] = block_index

    graph = {block_index: set() for block_index in range(len(partition))}

    def exact_zero(value: Any) -> bool:
        if isinstance(value, Const):
            return value.value == 0
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            source = owner[column_index]
            target = owner[row_index]
            if source != target and not exact_zero(value):
                graph[source].add(target)

    active = set()
    complete = set()

    def acyclic(block_index: int) -> bool:
        if block_index in active:
            return False
        if block_index in complete:
            return True
        active.add(block_index)
        if any(not acyclic(target) for target in graph[block_index]):
            return False
        active.remove(block_index)
        complete.add(block_index)
        return True

    return all(acyclic(block_index) for block_index in graph)


__all__ = [
    "DENSE_SPECTRAL",
    "DenseSpectralCapability",
    "DenseSpectralCapacityError",
    "is_exact_block_triangular",
]
