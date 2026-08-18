"""Manufactured 1-d periodic advection sine.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

Q0 = 1.0
EPS = 1.0e-2
A = 1.0
K = 1.0


def uniform_cell_centers(n_cells: int):
    """Return cell centers and widths on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_sine(x, t, *, q0=Q0, eps=EPS, a=A, k=K) -> np.ndarray:
    """Return q(x, t) = q0 + eps sin(2π k (x - a t)) on periodic [0, 1]."""
    points = np.asarray(x, dtype=np.float64)
    departure = np.mod(points - float(a) * float(t), 1.0)
    return float(q0) + float(eps) * np.sin(2.0 * np.pi * float(k) * departure)


def exact_sine_nd(coords, t, a, k, *, q0=Q0, eps=EPS) -> np.ndarray:
    """Optional n-d manufactured translation of the same sine."""
    axes = [np.asarray(axis, dtype=np.float64) for axis in coords]
    speeds = np.broadcast_to(np.asarray(a, dtype=np.float64), (len(axes),))
    waves = np.broadcast_to(np.asarray(k, dtype=np.float64), (len(axes),))
    phase = np.zeros(np.broadcast_shapes(*[axis.shape for axis in axes]), dtype=np.float64)
    for axis, speed, wave in zip(axes, speeds, waves, strict=True):
        phase = phase + float(wave) * np.mod(axis - float(speed) * float(t), 1.0)
    return float(q0) + float(eps) * np.sin(2.0 * np.pi * phase)
