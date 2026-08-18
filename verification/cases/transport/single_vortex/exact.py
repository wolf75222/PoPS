"""TR-03 reversible SingleVortex. Incompressible time-reversible 2-d flow.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

PERIOD = 1.0
X0 = 0.5
Y0 = 0.75
RADIUS = 0.15
N_CELLS = 32


def velocity(x, y, t, *, period: float = PERIOD):
    """Return (u, v) of the time-reversible incompressible swirl."""
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    modulation = np.cos(np.pi * float(t) / float(period))
    u = np.square(np.sin(np.pi * samples_x)) * np.sin(2.0 * np.pi * samples_y) * modulation
    v = (
        -np.square(np.sin(np.pi * samples_y))
        * np.sin(2.0 * np.pi * samples_x)
        * modulation
    )
    if np.ndim(samples_x) == 0 and np.ndim(samples_y) == 0:
        return float(np.asarray(u)), float(np.asarray(v))
    return u, v


def exact_scalar(
    x,
    y,
    t=0.0,
    *,
    x0: float = X0,
    y0: float = Y0,
    radius: float = RADIUS,
    period: float = PERIOD,
):
    """Compact cos^6 disk. Closed form exists at integer multiples of T."""
    cycles = float(t) / float(period)
    if abs(cycles - round(cycles)) > 1.0e-12:
        raise ValueError("closed-form scalar exists only at integer multiples of T")
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    samples_x, samples_y = np.broadcast_arrays(samples_x, samples_y)
    dx = samples_x - float(x0)
    dy = samples_y - float(y0)
    r = np.sqrt(dx * dx + dy * dy)
    support = float(radius)
    arg = np.clip(r / support, 0.0, 1.0)
    field = np.where(
        r < support, np.power(np.cos(0.5 * np.pi * arg), 6.0), 0.0
    ).astype(np.float64, copy=False)
    if field.ndim == 0:
        return float(field)
    return field


def exact_return(x, y, *, x0: float = X0, y0: float = Y0, radius: float = RADIUS):
    """Oracle at t=T: the scalar returns exactly to the IC."""
    return exact_scalar(x, y, 0.0, x0=x0, y0=y0, radius=radius)


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the periodic unit square."""
    count = int(n_cells)
    width = 1.0 / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y, width


def cell_volumes(n_cells: int = N_CELLS) -> np.ndarray:
    """Return uniform cell volumes on the unit square."""
    count = int(n_cells)
    width = 1.0 / count
    return np.full((count, count), width * width, dtype=np.float64)


def discrete_divergence(n_cells: int = N_CELLS, t=0.0) -> np.ndarray:
    """Central-difference divergence of manufactured (u, v) at cell centres."""
    x, y, width = cell_centers(n_cells)
    u, v = velocity(x, y, t)
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * width)
    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * width)
    return du_dx + dv_dy
