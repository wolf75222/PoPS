"""PF-09 rank-cell load stand-in: uniform vs hotspot over 4 ranks.

Distribute 64 cells. Uniform is 16 per rank. Hotspot places 40 cells on
rank 0 and splits the remaining 24 evenly. Does not import pops or read
a PoPS output.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 64
N_RANKS = 4
HOTSPOT_RANK0 = 40


def uniform_counts(n_cells: int = N_CELLS, n_ranks: int = N_RANKS) -> np.ndarray:
    """Return an even integer cell count per rank."""
    total = int(n_cells)
    ranks = int(n_ranks)
    if ranks <= 0:
        raise ValueError("n_ranks must be positive")
    if total % ranks != 0:
        raise ValueError("uniform load requires n_cells divisible by n_ranks")
    return np.full(ranks, total // ranks, dtype=np.int64)


def hotspot_counts(
    n_cells: int = N_CELLS,
    n_ranks: int = N_RANKS,
    hotspot: int = HOTSPOT_RANK0,
) -> np.ndarray:
    """Return rank-0 hotspot plus an even remainder on the other ranks."""
    total = int(n_cells)
    ranks = int(n_ranks)
    heavy = int(hotspot)
    if ranks < 2:
        raise ValueError("hotspot load requires at least two ranks")
    remainder = total - heavy
    others = ranks - 1
    if heavy <= 0 or remainder <= 0:
        raise ValueError("hotspot and remainder must both be positive")
    if remainder % others != 0:
        raise ValueError("hotspot remainder must split evenly")
    counts = np.empty(ranks, dtype=np.int64)
    counts[0] = heavy
    counts[1:] = remainder // others
    return counts


def load_stats(counts) -> dict:
    """Return max, mean, and population coefficient of variation."""
    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0:
        raise ValueError("counts must be non-empty")
    mean = float(values.mean())
    maximum = float(values.max())
    if mean == 0.0:
        raise ValueError("mean cell count must be positive")
    cv = float(values.std(ddof=0) / mean)
    return {"max": maximum, "mean": mean, "cv": cv, "counts": values}
