"""Manufactured 1-d periodic discontinuous slot. Translation is exact.

q = 1 for |x - x0| < w/2, else 0. Canonical x0=0.5, w=0.25, a=1.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

PERIOD = 1.0
X0 = 0.5
WIDTH = 0.25
A = 1.0
Q_IN = 1.0
Q_OUT = 0.0
DEFAULT_N_CELLS = 64


def minimum_image(delta, period: float = PERIOD):
    """Map a displacement onto (-period/2, period/2]."""
    width = float(period)
    return np.mod(np.asarray(delta, dtype=np.float64) + 0.5 * width, width) - 0.5 * width


def cell_centers(n_cells: int = DEFAULT_N_CELLS, period: float = PERIOD):
    """Return uniform cell centers and volumes on the periodic interval."""
    count = int(n_cells)
    width = float(period) / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_slot(
    x,
    t,
    *,
    x0: float = X0,
    width: float = WIDTH,
    a: float = A,
    period: float = PERIOD,
) -> np.ndarray:
    """Return q(x, t) = q0(x - a t) on periodic [0, 1]."""
    departure = np.asarray(x, dtype=np.float64) - float(a) * float(t)
    radius = np.abs(minimum_image(departure - float(x0), period))
    return np.where(radius < 0.5 * float(width), Q_IN, Q_OUT).astype(np.float64)
