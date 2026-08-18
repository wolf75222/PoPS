"""2-d manufactured Dirichlet Poisson oracle.

Does not import pops or read a PoPS output.

    φ = e^x sin(2πy) + x² y
    f = -Δφ = (4π² - 1) e^x sin(2πy) - 2y
    E = -∇φ

1-d slice (if 2-d authoring is unused): φ = e^x, f = -e^x, E = -e^x.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
N_CELLS = 32
X_LO = 0.0
X_HI = 1.0
Y_LO = 0.0
Y_HI = 1.0


def uniform_cell_grid(
    n_cells: int = N_CELLS,
    x_lo: float = X_LO,
    x_hi: float = X_HI,
    y_lo: float = Y_LO,
    y_hi: float = Y_HI,
):
    """Return 2-d cell centers and volumes on a uniform Cartesian partition."""
    count = int(n_cells)
    hx = (float(x_hi) - float(x_lo)) / count
    hy = (float(y_hi) - float(y_lo)) / count
    xs = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * hx
    ys = float(y_lo) + (np.arange(count, dtype=np.float64) + 0.5) * hy
    x, y = np.meshgrid(xs, ys, indexing="ij")
    volumes = np.full((count, count), hx * hy, dtype=np.float64)
    return x, y, volumes


def uniform_cell_grid_1d(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return cell centers and widths for a uniform 1-d partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def phi_exact(x, y) -> np.ndarray:
    """Pointwise φ(x,y) = e^x sin(2πy) + x² y."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return np.exp(xx) * np.sin(TWO_PI * yy) + xx**2 * yy


def rhs_exact(x, y) -> np.ndarray:
    """Pointwise f = -Δφ = (4π² - 1) e^x sin(2πy) - 2y."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return (TWO_PI**2 - 1.0) * np.exp(xx) * np.sin(TWO_PI * yy) - 2.0 * yy


def e_exact(x, y):
    """Pointwise E = -∇φ = (-e^x sin(2πy) - 2xy, -2π e^x cos(2πy) - x²)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ex = -np.exp(xx) * np.sin(TWO_PI * yy) - 2.0 * xx * yy
    ey = -TWO_PI * np.exp(xx) * np.cos(TWO_PI * yy) - xx**2
    return ex, ey


def phi_exact_1d(x) -> np.ndarray:
    """1-d slice φ(x) = e^x."""
    return np.exp(np.asarray(x, dtype=np.float64))


def rhs_exact_1d(x) -> np.ndarray:
    """1-d slice f = -φ'' = -e^x."""
    return -phi_exact_1d(x)


def e_exact_1d(x) -> np.ndarray:
    """1-d slice E = -dφ/dx = -e^x."""
    return -phi_exact_1d(x)
