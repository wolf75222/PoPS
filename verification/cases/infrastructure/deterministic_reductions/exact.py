"""IF-06 exact-representable geometric series and chaotic leftover.

A 1-d float64 field of powers of two, 2^{-k} for k = 0 .. n-1, is exactly
representable. Sequential, pairwise, and blocked reductions of that series
must match the closed form 2 - 2^{1-n} bitwise.

A chaotic leftover mix of magnitudes may differ across reduction trees
(observation only; not a gate).

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 32
BLOCK_SIZE = 8
PERIOD = 1.0
REDUCTION_NAMES = ("sequential", "pairwise", "blocked")


def cell_volumes(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def geometric_field(n_cells: int = N_CELLS) -> np.ndarray:
    """Return the exactly representable series 2^{-k} for k = 0 .. n-1."""
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    exponents = -np.arange(count, dtype=np.int64)
    return np.ldexp(np.ones(count, dtype=np.float64), exponents)


def closed_form_sum(n_cells: int = N_CELLS) -> float:
    """Return the exact geometric sum 2 - 2^{1-n}."""
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    return float(np.ldexp(np.float64(1.0), 1) - np.ldexp(np.float64(1.0), 1 - count))


def chaotic_field() -> np.ndarray:
    """Return a mixed-magnitude leftover whose reductions may disagree."""
    pattern = np.array(
        [1.0e16, 1.0, -1.0e16, 1.0e-8, -1.0, 1.0e-8],
        dtype=np.float64,
    )
    return np.tile(pattern, 8)
