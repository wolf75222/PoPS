"""Public-runtime helpers for inspecting composite-active AMR state in tests."""
from __future__ import annotations

from typing import Any

import numpy as np


def _spatial_shape(simulation: Any) -> tuple[int, ...]:
    provider = getattr(simulation, "spatial_shape", None)
    if not callable(provider):
        raise TypeError("AMR snapshot helper requires spatial_shape()")
    shape = tuple(int(extent) for extent in provider())
    if not shape or any(extent < 1 for extent in shape):
        raise ValueError("spatial_shape must contain exact positive extents")
    return shape


def _ranked_box(row: Any) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not isinstance(row, tuple) or len(row) != 3:
        raise TypeError("AMR patch_boxes rows must be (level, lower, upper)")
    level, lower, upper = row
    return int(level), tuple(int(value) for value in lower), tuple(int(value) for value in upper)


def _box_slices(
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    *,
    shape: tuple[int, ...],
    scale: int = 1,
) -> tuple[slice, ...]:
    if len(lower) != len(shape) or len(upper) != len(shape):
        raise TypeError("AMR patch bounds must match the spatial rank")
    if scale < 1:
        raise ValueError("AMR patch scale must be a positive integer")
    slices = []
    for lo, hi, extent in zip(reversed(lower), reversed(upper), reversed(shape), strict=True):
        lo //= scale
        hi //= scale
        if not 0 <= lo <= hi < extent:
            raise AssertionError(
                "AMR patch [%d, %d] is outside spatial extent %d" % (lo, hi, extent)
            )
        slices.append(slice(lo, hi + 1))
    return tuple(slices)


def composite_active_mask(
    simulation: Any,
    level: int,
    *,
    refinement_ratio: int,
) -> np.ndarray:
    """Return the valid, uncovered cell mask for one globally shaped AMR level."""
    if isinstance(refinement_ratio, bool) or not isinstance(refinement_ratio, int):
        raise TypeError("refinement_ratio must be an integer")
    if refinement_ratio <= 1:
        raise ValueError("refinement_ratio must be greater than one")

    shape = tuple(extent * (refinement_ratio**level) for extent in _spatial_shape(simulation))
    boxes = tuple(_ranked_box(row) for row in simulation.patch_boxes())
    # Public arrays keep the last axis fastest, matching Index<Dim> x-first storage.
    active = (
        np.ones(tuple(reversed(shape)), dtype=np.bool_)
        if level == 0
        else np.zeros(tuple(reversed(shape)), dtype=np.bool_)
    )

    level_boxes = [box for box in boxes if box[0] == level]
    if level > 0:
        assert level_boxes, f"AMR level {level} has no patch"
        for _box_level, lower, upper in level_boxes:
            active[_box_slices(lower, upper, shape=shape)] = True

    for _child_level, lower, upper in (box for box in boxes if box[0] == level + 1):
        active[_box_slices(lower, upper, shape=shape, scale=refinement_ratio)] = False
    return active


def composite_active_block_state(
    simulation: Any,
    block: str,
    level: int,
    *,
    refinement_ratio: int,
) -> np.ndarray:
    """Return ``(component, active-cell)`` state through the public runtime surface."""
    shape = tuple(extent * (refinement_ratio**level) for extent in _spatial_shape(simulation))
    flat = np.asarray(
        simulation.block_level_state_global(block, level), dtype=np.float64
    )
    cells_per_component = int(np.prod(shape))
    assert flat.size % cells_per_component == 0
    state = flat.reshape((-1, *reversed(shape)))
    return state[
        :,
        composite_active_mask(
            simulation,
            level,
            refinement_ratio=refinement_ratio,
        ),
    ]
