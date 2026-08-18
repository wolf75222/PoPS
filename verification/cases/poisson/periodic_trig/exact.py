"""1-d manufactured periodic trigonometric Poisson oracle.

Does not import pops or read a PoPS output.

Reduction of φ = sin(2πx) sin(4πy) cos(2πz):
    φ = sin(2πx),  -φ'' = (2π)² φ,  E = -dφ/dx = -2π cos(2πx).
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
    """Pointwise φ(x) = sin(2πx)."""
    return np.sin(TWO_PI * np.asarray(x, dtype=np.float64))


def rhs_exact(x) -> np.ndarray:
    """Pointwise -Δφ = (2π)² sin(2πx)."""
    return (TWO_PI**2) * phi_exact(x)


def e_exact(x) -> np.ndarray:
    """Pointwise E = -dφ/dx = -2π cos(2πx)."""
    return -TWO_PI * np.cos(TWO_PI * np.asarray(x, dtype=np.float64))


def phi_exact_2d(x, y) -> np.ndarray:
    """Optional 2-d restriction φ(x,y) = sin(2πx) sin(4πy)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.sin(TWO_PI * xx) * np.sin(2.0 * TWO_PI * yy)
