"""PF-07 1-d clustering stand-in: contiguous tagged runs, min patch width 4.

Tag a scalar pulse by amplitude. Cluster each contiguous run into one patch
and grow short runs to MIN_PATCH_WIDTH. Does not import pops or read a
PoPS output.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 128
MIN_PATCH_WIDTH = 4
PERIOD = 1.0
TAG_THRESHOLD = 0.1


def uniform_centers(n_cells: int = N_CELLS) -> np.ndarray:
    """Return cell-center coordinates on periodic [0, 1]."""
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    width = PERIOD / float(count)
    return (np.arange(count, dtype=np.float64) + 0.5) * width


def raw_tag_mask(q, threshold: float = TAG_THRESHOLD) -> np.ndarray:
    """Tag cells whose sampled pulse exceeds the documented amplitude."""
    return np.asarray(q, dtype=np.float64) > float(threshold)


def tagged_runs(mask) -> list[tuple[int, int]]:
    """Return periodic (start, width) runs of tagged cells."""
    selected = np.asarray(mask, dtype=bool)
    n_cells = int(selected.size)
    if n_cells == 0 or not np.any(selected):
        return []
    if np.all(selected):
        return [(0, n_cells)]
    previous = np.roll(selected, 1)
    starts = np.flatnonzero(selected & ~previous)
    runs: list[tuple[int, int]] = []
    for start in starts:
        width = 1
        index = (int(start) + 1) % n_cells
        while selected[index] and index != int(start):
            width += 1
            index = (index + 1) % n_cells
        runs.append((int(start), int(width)))
    return runs


def _grow_run(start, width, n_cells, min_width) -> tuple[int, int]:
    count = int(n_cells)
    span = int(width)
    target = min(int(min_width), count)
    if span >= count:
        return (0, count)
    if span >= target:
        return (int(start) % count, span)
    extra = target - span
    left = extra // 2
    return ((int(start) - left) % count, target)


def coverage_mask(patches, n_cells) -> np.ndarray:
    """Return the boolean union of (start, width) patches on a periodic mesh."""
    count = int(n_cells)
    covered = np.zeros(count, dtype=bool)
    for start, width in patches:
        origin = int(start)
        span = int(width)
        for offset in range(span):
            covered[(origin + offset) % count] = True
    return covered


def cluster_runs(mask, min_width: int = MIN_PATCH_WIDTH) -> list[tuple[int, int]]:
    """Cluster tagged runs into patches of at least min_width cells."""
    width = int(min_width)
    if width < 1:
        raise ValueError("min_width must be positive")
    selected = np.asarray(mask, dtype=bool)
    n_cells = int(selected.size)
    grown = np.zeros(n_cells, dtype=bool)
    for start, span in tagged_runs(selected):
        grown_start, grown_width = _grow_run(start, span, n_cells, width)
        for offset in range(grown_width):
            grown[(grown_start + offset) % n_cells] = True
    return tagged_runs(grown)
