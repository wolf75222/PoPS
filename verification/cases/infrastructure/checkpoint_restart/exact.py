"""IF-04 checkpoint state: TR-01 sine at cell centers, plus t.

Uniform 1-d cell grid on [0, 1]. Field q is TR-01's periodic sine.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_TR01_EXACT = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "exact.py"
)
_tr01 = load_sibling_module(_TR01_EXACT)

exact_sine = _tr01.exact_sine_1d

N_CELLS = 32
X_LO = 0.0
X_HI = 1.0
T = 0.25
PERIOD = 1.0


def uniform_cell_grid(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Return cell centers and widths for a uniform 1-d partition of [x_lo, x_hi]."""
    count = int(n_cells)
    width = (float(x_hi) - float(x_lo)) / count
    centers = float(x_lo) + (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def cell_centers(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int = N_CELLS, x_lo: float = X_LO, x_hi: float = X_HI):
    """Uniform cell widths on [x_lo, x_hi]."""
    _, volumes = uniform_cell_grid(n_cells, x_lo, x_hi)
    return volumes


def manufactured_q(x, t=T):
    """Pointwise TR-01 sine q(x, t)."""
    return np.asarray(exact_sine(x, t), dtype=np.float64)


def manufactured_state(n_cells: int = N_CELLS, t: float = T):
    """Return the manufactured checkpoint dict {centers, q, t}."""
    centers, _ = uniform_cell_grid(n_cells)
    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "q": manufactured_q(centers, t),
        "t": float(t),
    }
