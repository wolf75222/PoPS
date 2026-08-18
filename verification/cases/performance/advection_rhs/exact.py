"""PF-03 1-d upwind FV RHS stand-in on the TR-01 sine.

Periodic wrap of width 1, then interior first-order upwind of q_t + a q_x.
Exact translation RHS is the analytic -a ∂x sine at cell centres.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

from verification.pops_verify.reference_errors import reference_errors

N_CELLS = 16
HALO_WIDTH = 1
PERIOD = 1.0
Q0 = 1.0
EPS = 1.0e-2
A = 1.0
K = 1.0


def cell_volumes(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def cell_centers(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return cell centres on the periodic interval."""
    count = int(n_cells)
    width = float(period) / count
    return (np.arange(count, dtype=np.float64) + 0.5) * width


def exact_sine(x, t, *, q0=Q0, eps=EPS, a=A, k=K) -> np.ndarray:
    """Return q(x, t) = q0 + eps sin(2π k (x - a t)) on periodic [0, 1]."""
    points = np.asarray(x, dtype=np.float64)
    departure = np.mod(points - float(a) * float(t), 1.0)
    return float(q0) + float(eps) * np.sin(2.0 * np.pi * float(k) * departure)


def exact_rhs(x, t, *, eps=EPS, a=A, k=K) -> np.ndarray:
    """Return the analytic translation RHS -a ∂x sine at the given points."""
    points = np.asarray(x, dtype=np.float64)
    departure = np.mod(points - float(a) * float(t), 1.0)
    omega = 2.0 * np.pi * float(k)
    return -float(a) * float(eps) * omega * np.cos(omega * departure)


def pad_interior(interior, halo_width: int = HALO_WIDTH) -> np.ndarray:
    """Return interior padded with NaN ghosts (unset until fill)."""
    valid = np.asarray(interior, dtype=np.float64)
    width = int(halo_width)
    padded = np.full(valid.size + 2 * width, np.nan, dtype=np.float64)
    padded[width : width + valid.size] = valid
    return padded


def fill_periodic_halo(padded, halo_width: int = HALO_WIDTH) -> np.ndarray:
    """Copy periodic interior into the halo cells of a 1-d padded field."""
    field = np.asarray(padded, dtype=np.float64).copy()
    width = int(halo_width)
    interior = field[width:-width]
    field[:width] = interior[-width:]
    field[-width:] = interior[:width]
    return field


def halo_linf(padded, halo_width: int = HALO_WIDTH) -> float:
    """Return L∞ of halo cells versus the periodic interior wrap."""
    field = np.asarray(padded, dtype=np.float64)
    width = int(halo_width)
    interior = field[width:-width]
    left = np.max(np.abs(field[:width] - interior[-width:]))
    right = np.max(np.abs(field[-width:] - interior[:width]))
    return float(max(left, right))


def upwind_rhs(padded, dx, *, a: float = A, halo_width: int = HALO_WIDTH) -> np.ndarray:
    """Return the interior first-order upwind FV RHS -∂x(a q)."""
    field = np.asarray(padded, dtype=np.float64)
    width = int(halo_width)
    interior = field[width:-width]
    speed = float(a)
    spacing = float(dx)
    if speed >= 0.0:
        left = field[width - 1 : -width - 1]
        return -speed * (interior - left) / spacing
    right = field[width + 1 : field.size - width + 1]
    return -speed * (right - interior) / spacing


def fd_truncation_bound(n_cells: int = N_CELLS, *, eps=EPS, a=A, k=K) -> float:
    """Return an O(dx) bound on first-order upwind versus -a ∂x sine."""
    dx = 1.0 / float(n_cells)
    omega = 2.0 * np.pi * float(k)
    return abs(float(a)) * float(eps) * omega**2 * dx


def rhs_errors(n_cells: int = N_CELLS):
    """Volume-weighted error of the interior upwind RHS versus analytic."""
    centers = cell_centers(n_cells)
    volumes = cell_volumes(n_cells)
    dx = float(volumes[0])
    interior = exact_sine(centers, 0.0)
    padded = fill_periodic_halo(pad_interior(interior), HALO_WIDTH)
    rhs = upwind_rhs(padded, dx)
    return reference_errors(rhs, exact_rhs(centers, 0.0), volumes)
