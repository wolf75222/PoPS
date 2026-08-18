"""1-d PO-01 trigonometric Poisson oracle plus CF interface placements.

Does not import pops or read a PoPS output.

Reuse of φ = sin(2πx):
    φ = sin(2πx),  -φ'' = (2π)² φ,  E = -dφ/dx = -2π cos(2πx).

Coarse-fine interface locations (unit period):
    max φ at 0.25,  zero of φ at 0,  max |φ'| at 0,  max |φ''| at 0.25.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
N_CELLS = 32
X_LO = 0.0
X_HI = 1.0
PERIOD = X_HI - X_LO
PLACEMENTS = {
    "max_phi": 0.25,
    "zero": 0.0,
    "max_abs_dphi": 0.0,
    "max_abs_d2phi": 0.25,
}


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


def dphi_exact(x) -> np.ndarray:
    """Pointwise dφ/dx = 2π cos(2πx)."""
    return TWO_PI * np.cos(TWO_PI * np.asarray(x, dtype=np.float64))


def d2phi_exact(x) -> np.ndarray:
    """Pointwise d²φ/dx² = -(2π)² sin(2πx)."""
    return -(TWO_PI**2) * phi_exact(x)


def e_exact(x) -> np.ndarray:
    """Pointwise E = -dφ/dx = -2π cos(2πx)."""
    return -dphi_exact(x)


def interface_location(name: str) -> float:
    """Return the unit-interval CF interface for a named PO-01 extremum."""
    try:
        return float(PLACEMENTS[name])
    except KeyError as exc:
        known = ", ".join(sorted(PLACEMENTS))
        raise ValueError(f"unknown CF placement {name!r}; expected one of {known}") from exc


def periodic_distance(x, x_interface, period: float = PERIOD) -> np.ndarray:
    """Periodic distance from samples to a CF interface on a unit interval."""
    width = float(period)
    shift = np.abs(np.asarray(x, dtype=np.float64) - float(x_interface)) % width
    return np.minimum(shift, width - shift)
