"""AM-03 gradient / second-diff tagging on a uniform 1-d mesh.

Tag a cell when |Δq| > θ or |second difference| > θ2. Buffer dilation is
1, 2, or 4 cells (periodic). Refine if above θ; coarsen if below θ/2.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

PERIOD = 1.0
N_CELLS = 128
X0 = 0.37
SIGMA = 0.08
CORE_RADIUS = 0.5 * SIGMA
THETA = 0.02
THETA2 = 0.002
BUFFER_WIDTHS = (1, 2, 4)


def uniform_centers(n_cells: int = N_CELLS) -> np.ndarray:
    """Return cell-center coordinates on periodic [0, 1]."""
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    width = PERIOD / float(count)
    return (np.arange(count, dtype=np.float64) + 0.5) * width


def minimum_image(delta, period: float = PERIOD):
    """Map a displacement onto (-period/2, period/2]."""
    width = float(period)
    return np.mod(np.asarray(delta, dtype=np.float64) + 0.5 * width, width) - 0.5 * width


def first_difference(q) -> np.ndarray:
    """Periodic undivided |Δq|: max of the two adjacent face jumps."""
    field = np.asarray(q, dtype=np.float64)
    left = np.abs(field - np.roll(field, 1))
    right = np.abs(np.roll(field, -1) - field)
    return np.maximum(left, right)


def second_difference(q) -> np.ndarray:
    """Periodic undivided |q_{i+1} - 2 q_i + q_{i-1}|."""
    field = np.asarray(q, dtype=np.float64)
    return np.abs(np.roll(field, -1) - 2.0 * field + np.roll(field, 1))


def raw_tag_mask(q, *, theta: float = THETA, theta2: float = THETA2) -> np.ndarray:
    """Tag cells where |Δq| > θ or |second difference| > θ2."""
    delta = first_difference(q)
    curvature = second_difference(q)
    return (delta > float(theta)) | (curvature > float(theta2))


def dilate_mask(mask, buffer_cells) -> np.ndarray:
    """Periodic buffer dilation by the given cell count."""
    selected = np.asarray(mask, dtype=bool)
    width = int(buffer_cells)
    if width < 0:
        raise ValueError("buffer_cells must be non-negative")
    dilated = selected.copy()
    for shift in range(1, width + 1):
        dilated |= np.roll(selected, shift)
        dilated |= np.roll(selected, -shift)
    return dilated


def hysteresis_update(
    previous,
    first_diff,
    second_diff,
    *,
    theta: float = THETA,
    theta2: float = THETA2,
) -> np.ndarray:
    """Refine if above θ / θ2; coarsen if below half those thresholds."""
    tagged = np.asarray(previous, dtype=bool).copy()
    delta = np.asarray(first_diff, dtype=np.float64)
    curvature = np.asarray(second_diff, dtype=np.float64)
    refine = (delta > float(theta)) | (curvature > float(theta2))
    coarsen = (delta < 0.5 * float(theta)) & (curvature < 0.5 * float(theta2))
    tagged[refine] = True
    tagged[coarsen] = False
    return tagged


def pulse_core_mask(centers, *, x0: float = X0, radius: float = CORE_RADIUS) -> np.ndarray:
    """Cells whose minimum-image distance to the pulse peak is ≤ CORE_RADIUS."""
    displacement = minimum_image(np.asarray(centers, dtype=np.float64) - float(x0))
    return np.abs(displacement) <= float(radius)
