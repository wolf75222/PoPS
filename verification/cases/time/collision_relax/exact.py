"""TM-03 exact collision relaxation: du/dt = -ν(u - ū).

Closed form: u(t) = ū + (u0 - ū) e^{-νt}.
The barycenter moment ū is constant. Does not import pops or read PoPS output.
"""
from __future__ import annotations

import numpy as np

NU = 2.0
N_CELLS = 32
TWO_PI = 2.0 * np.pi


def uniform_cell_centers(n_cells: int = N_CELLS):
    """Return cell centers and widths on a uniform partition of [0, 1]."""
    count = int(n_cells)
    width = 1.0 / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def initial_field(x):
    """u0 = 1 + (1/2) cos(2πx). Discrete barycenter on [0, 1] is 1."""
    xx = np.asarray(x, dtype=np.float64)
    return 1.0 + 0.5 * np.cos(TWO_PI * xx)


def barycenter(u, volumes=None) -> float:
    """Volume-weighted mean; unweighted mean when volumes are omitted."""
    field = np.asarray(u, dtype=np.float64)
    if volumes is None:
        return float(np.mean(field))
    weights = np.asarray(volumes, dtype=np.float64)
    return float(np.sum(field * weights) / np.sum(weights))


def exact_relax(u0, t, *, nu, u_bar):
    """Return ū + (u0 - ū) exp(-ν t)."""
    return np.asarray(u_bar, dtype=np.float64) + (
        np.asarray(u0, dtype=np.float64) - np.asarray(u_bar, dtype=np.float64)
    ) * np.exp(-float(nu) * float(t))
