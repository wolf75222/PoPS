"""Public-runtime helpers for inspecting composite-active AMR state in tests."""
from __future__ import annotations

from typing import Any

import numpy as np


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

    scale = refinement_ratio**level
    nx = simulation.nx() * scale
    ny = simulation.ny() * scale
    boxes = [tuple(int(value) for value in row) for row in simulation.patch_boxes()]
    active = (
        np.ones((ny, nx), dtype=np.bool_)
        if level == 0
        else np.zeros((ny, nx), dtype=np.bool_)
    )

    level_boxes = [box for box in boxes if box[0] == level]
    if level > 0:
        assert level_boxes, f"AMR level {level} has no patch"
        for _box_level, ilo, jlo, ihi, jhi in level_boxes:
            assert 0 <= ilo <= ihi < nx
            assert 0 <= jlo <= jhi < ny
            active[jlo : jhi + 1, ilo : ihi + 1] = True

    for _child_level, ilo, jlo, ihi, jhi in (
        box for box in boxes if box[0] == level + 1
    ):
        active[
            jlo // refinement_ratio : jhi // refinement_ratio + 1,
            ilo // refinement_ratio : ihi // refinement_ratio + 1,
        ] = False
    return active


def composite_active_block_state(
    simulation: Any,
    block: str,
    level: int,
    *,
    refinement_ratio: int,
) -> np.ndarray:
    """Return ``(component, active-cell)`` state through the public runtime surface."""
    scale = refinement_ratio**level
    nx = simulation.nx() * scale
    ny = simulation.ny() * scale
    flat = np.asarray(
        simulation.block_level_state_global(block, level), dtype=np.float64
    )
    cells_per_component = nx * ny
    assert flat.size % cells_per_component == 0
    state = flat.reshape((-1, ny, nx))
    return state[
        :,
        composite_active_mask(
            simulation,
            level,
            refinement_ratio=refinement_ratio,
        ),
    ]
