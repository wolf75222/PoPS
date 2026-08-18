"""PF-12 synthetic HyQMOM15-width state: (n, 15) saxpy and bytes/cell.

15 float64 components versus 5-component Euler width. No pops import.
Does not read a PoPS output.
"""
from __future__ import annotations

import numpy as np

from verification.pops_verify.reference_errors import reference_errors

N_CELLS = 16
N_COMPONENTS = 15
EULER_COMPONENTS = 5
BYTES_PER_SCALAR = 8
BYTES_PER_CELL = N_COMPONENTS * BYTES_PER_SCALAR
EULER_BYTES_PER_CELL = EULER_COMPONENTS * BYTES_PER_SCALAR
PERIOD = 1.0
SAXPY_ALPHA = 2.0
# Canonical HyQMOM15 names (order 4, q outer / p inner). Width only; not a solver.
COMPONENT_NAMES = (
    "M00",
    "M10",
    "M20",
    "M30",
    "M40",
    "M01",
    "M11",
    "M21",
    "M31",
    "M02",
    "M12",
    "M22",
    "M03",
    "M13",
    "M04",
)


def bytes_per_cell(
    n_components: int = N_COMPONENTS, bytes_per_scalar: int = BYTES_PER_SCALAR
) -> int:
    """Return float64 footprint of one cell for the given component count."""
    return int(n_components) * int(bytes_per_scalar)


def euler_bytes_per_cell() -> int:
    """Return float64 footprint of one 5-component Euler cell."""
    return bytes_per_cell(EULER_COMPONENTS)


def cell_volumes(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return per-cell volumes shaped to broadcast over a (n, 15) state."""
    width = float(period) / int(n_cells)
    return np.full((int(n_cells), 1), width, dtype=np.float64)


def interior_state(
    n_cells: int = N_CELLS, n_components: int = N_COMPONENTS
) -> np.ndarray:
    """Return a unique (n, 15) field so the wide saxpy is observable."""
    cells = np.arange(int(n_cells), dtype=np.float64)[:, None]
    components = np.arange(int(n_components), dtype=np.float64)[None, :]
    return (cells + 1.0) * (components + 1.0)


def saxpy(b, alpha: float = SAXPY_ALPHA) -> np.ndarray:
    """Return a = alpha * b on a (n, 15) state (SAXPY with a zero destination)."""
    return float(alpha) * np.asarray(b, dtype=np.float64)


def saxpy_errors(n_cells: int = N_CELLS, alpha: float = SAXPY_ALPHA):
    """Volume-weighted error of a = alpha * b versus the exact scale."""
    b = interior_state(n_cells)
    a = saxpy(b, alpha)
    return reference_errors(a, float(alpha) * b, cell_volumes(n_cells))
