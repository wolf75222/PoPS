"""PF-01 1-d numpy MultiFab stand-in: periodic halo and exact arith.

Periodic wrap of width 2, then a = b + c and a = 2 * b.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

from verification.pops_verify.reference_errors import reference_errors

N_CELLS = 16
HALO_WIDTH = 2
PERIOD = 1.0
SAXPY_ALPHA = 2.0


def cell_volumes(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def interior_pattern(n_cells: int = N_CELLS) -> np.ndarray:
    """Return a unique interior field so wrap-around is observable."""
    return np.arange(int(n_cells), dtype=np.float64) + 1.0


def partner_pattern(n_cells: int = N_CELLS) -> np.ndarray:
    """Return a second interior field for a = b + c."""
    return 3.0 * np.arange(int(n_cells), dtype=np.float64) + 0.5


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


def add(b, c) -> np.ndarray:
    """Return a = b + c."""
    return np.asarray(b, dtype=np.float64) + np.asarray(c, dtype=np.float64)


def saxpy(b, alpha: float = SAXPY_ALPHA) -> np.ndarray:
    """Return a = alpha * b (SAXPY with a zero destination)."""
    return float(alpha) * np.asarray(b, dtype=np.float64)


def saxpy_errors(n_cells: int = N_CELLS, alpha: float = SAXPY_ALPHA):
    """Volume-weighted error of a = alpha * b versus the exact scale."""
    b = interior_pattern(n_cells)
    a = saxpy(b, alpha)
    return reference_errors(a, float(alpha) * b, cell_volumes(n_cells))
