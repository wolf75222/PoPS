"""Manufactured 1-d cosine oracle on a uniform cell grid.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 32
X_LO = 0.0
X_HI = 1.0


def uniform_cell_grid(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return cell centers and widths for a uniform 1-d partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def manufactured_cosine(x):
    """Pointwise u(x) = cos(2πx)."""
    return np.cos(2.0 * np.pi * np.asarray(x, dtype=np.float64))


def exact_sample(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return (u_exact, cell_widths) sampled at uniform cell centers."""
    centers, volumes = uniform_cell_grid(n_cells, x_lo, x_hi)
    return manufactured_cosine(centers), volumes
