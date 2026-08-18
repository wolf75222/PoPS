"""Manufactured advection sine for TR-01 and 1-d siblings.

Canonical 3-d data: a = (1, 1, 1), k = (1, 2, 3), T = 1 on the unit cube
via ``exact_sine_3d``. Shared 1-d sibling samples keep ``exact_sine(x, t)``.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

Q0 = 1.0
EPS = 1.0e-2
AX = 1.0
AY = 1.0
AZ = 1.0
A = (AX, AY, AZ)
KX = 1.0
KY = 2.0
KZ = 3.0
K = (KX, KY, KZ)
T_END = 1.0
RESOLUTIONS = (16, 32, 64, 128)
REQUIRED_NATIVE_DIM = 3


def uniform_cell_centers(n_cells: int):
    """1-d unit-interval centers. Not the TR-01 3-d mesh."""
    count = int(n_cells)
    width = 1.0 / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def uniform_cell_mesh(n_cells: int):
    """Return (x, y, z, volumes) on a uniform periodic unit cube.

    Centers are shaped ``(n, n, n)`` with axis order ``(z, y, x)`` to match
    the native ``state_global`` packing ``(ncomp, nz, ny, nx)``.
    """
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    width = 1.0 / float(count)
    line = (np.arange(count, dtype=np.float64) + 0.5) * width
    zz, yy, xx = np.meshgrid(line, line, line, indexing="ij")
    volumes = np.full(xx.shape, width**3, dtype=np.float64)
    return xx, yy, zz, volumes


def cell_bounds(n_cells: int):
    """Return cell ``(lo, hi)`` arrays of shape ``(n, n, n, 3)`` in (x, y, z)."""
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    width = 1.0 / float(count)
    edges = np.arange(count, dtype=np.float64) * width
    iz, iy, ix = np.meshgrid(edges, edges, edges, indexing="ij")
    lo = np.stack((ix, iy, iz), axis=-1)
    hi = lo + width
    return lo, hi


def exact_sine(x, t, *, q0=Q0, eps=EPS, a=1.0, k=1.0) -> np.ndarray:
    """1-d q(x, t) = q0 + ε sin(2π k (x − a t)) on periodic [0, 1].

    Shared by AM / IF / TM siblings. Not the canonical TR-01 3-d case.
    """
    points = np.asarray(x, dtype=np.float64)
    departure = np.mod(points - float(a) * float(t), 1.0)
    return float(q0) + float(eps) * np.sin(2.0 * np.pi * float(k) * departure)


def exact_sine_3d(
    x,
    y,
    z,
    t,
    *,
    q0=Q0,
    eps=EPS,
    a=A,
    k=K,
) -> np.ndarray:
    """Canonical q(x, y, z, t) = q0 + ε sin(2π k · (x − a t)) on [0, 1]³."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    ax, ay, az = (float(value) for value in a)
    kx, ky, kz = (float(value) for value in k)
    time = float(t)
    phase = (
        kx * (xx - ax * time)
        + ky * (yy - ay * time)
        + kz * (zz - az * time)
    )
    return float(q0) + float(eps) * np.sin(2.0 * np.pi * phase)
