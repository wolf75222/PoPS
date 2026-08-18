"""2-d exact uniform Euler flow. Oracle is the IC at every t.

Constant state rho=1, u=1, v=0, p=1, gamma=1.4 on the periodic unit
square. Spatially constant; independent of t. Does not import pops or
read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
RHO = 1.0
U = 1.0
V = 0.0
P = 1.0
PERIOD = 1.0
N_CELLS = 32
INTERFACE_X = 0.5


def background() -> dict:
    """Uniform free stream (rho, u, v, p). Canonical (u, v)=(1, 0)."""
    return {"rho": RHO, "u": U, "v": V, "p": P}


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the periodic box [0, PERIOD]^2."""
    count = int(n_cells)
    width = float(PERIOD) / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y


def cell_volumes(n_cells: int = N_CELLS) -> np.ndarray:
    """Uniform cell volumes on the periodic box [0, PERIOD]^2."""
    count = int(n_cells)
    width = float(PERIOD) / float(count)
    return np.full((count, count), width * width, dtype=np.float64)


def exact_primitives(x, y, t):
    """Primitive fields (rho, u, v, p). Constant; independent of (x, y, t)."""
    del t
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    if samples_x.shape != samples_y.shape:
        raise ValueError("x and y must have the same shape")
    shape = samples_x.shape
    return {
        "rho": np.full(shape, RHO, dtype=np.float64),
        "u": np.full(shape, U, dtype=np.float64),
        "v": np.full(shape, V, dtype=np.float64),
        "p": np.full(shape, P, dtype=np.float64),
    }


def is_spatially_constant(field) -> bool:
    """Return True when every sample equals the first finite sample."""
    values = np.asarray(field, dtype=np.float64)
    if values.size == 0:
        raise ValueError("empty field")
    if not np.all(np.isfinite(values)):
        return False
    return bool(np.all(values == values.reshape(-1)[0]))


def interface_cell_index(n_cells: int = N_CELLS) -> tuple[int, int]:
    """Return the (j, i) of the one cell just right of the mid-domain face.

    INTERFACE_X=0.5 is a cell face when n_cells is even. That face is both
    a same-level block join and a static CF interface location.
    """
    count = int(n_cells)
    if count < 2 or count % 2 != 0:
        raise ValueError(f"interface face needs even n_cells >= 2, got {n_cells!r}")
    return count // 2, count // 2
