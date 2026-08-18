"""1-d manufactured homogeneous-Neumann Poisson oracle.

Does not import pops or read a PoPS output.

    φ = cos(2πx),  φ'(0)=φ'(1)=0,  -φ'' = (2π)² cos(2πx).

The Neumann problem has a constant nullspace. Compare φ-⟨φ⟩.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
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


def phi_exact(x) -> np.ndarray:
    """Pointwise φ(x) = cos(2πx)."""
    return np.cos(TWO_PI * np.asarray(x, dtype=np.float64))


def dphi_exact(x) -> np.ndarray:
    """Pointwise φ'(x) = -2π sin(2πx). Homogeneous Neumann at x=0 and x=1."""
    return -TWO_PI * np.sin(TWO_PI * np.asarray(x, dtype=np.float64))


def rhs_exact(x) -> np.ndarray:
    """Pointwise -Δφ = (2π)² cos(2πx)."""
    return (TWO_PI**2) * phi_exact(x)


def e_exact(x) -> np.ndarray:
    """Pointwise E = -dφ/dx = 2π sin(2πx)."""
    return -dphi_exact(x)


def mean_free(values, volumes=None) -> np.ndarray:
    """Return values minus the (optionally volume-weighted) mean."""
    field = np.asarray(values, dtype=np.float64)
    if volumes is None:
        mean = float(np.mean(field))
    else:
        mean = float(np.average(field, weights=np.asarray(volumes, dtype=np.float64)))
    return field - mean
