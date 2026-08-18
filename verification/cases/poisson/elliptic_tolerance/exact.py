"""1-d PO-01 trigonometric Poisson oracle plus elliptic tolerance error model.

Does not import pops or read a PoPS output.

Reuse of φ = sin(2πx):
    φ = sin(2πx),  -φ'' = (2π)² φ,  E = -dφ/dx = -2π cos(2πx).

Algebraic vs discretization:
    discretization_error(n) ∝ h²,  algebraic_error(tol) = tol,
    combined_error = max(disc, alg).
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
N_CELLS = 32
X_LO = 0.0
X_HI = 1.0
DISCRETIZATION_ERROR_SCALE = 0.04


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


def discretization_error(n_cells: int) -> float:
    """Second-order manufactured floor C h² on a uniform partition of [0, 1]."""
    spacing = 1.0 / float(n_cells)
    return DISCRETIZATION_ERROR_SCALE * spacing**2


def algebraic_error(tol) -> float:
    """Algebraic residual contribution equals the requested relative tolerance."""
    return float(tol)


def combined_error(n_cells: int, tol) -> float:
    """Combined error is the max of discretization and algebraic contributions."""
    return max(discretization_error(n_cells), algebraic_error(tol))
