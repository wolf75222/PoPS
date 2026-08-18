"""PF-10 1-d numpy field for an npz checkpoint stand-in.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 32
PERIOD = 1.0
FAKE_WRITE_TIME_S = 1.0e-3
ARTIFACT_NAME = "checkpoint.npz"
FIELD_KEY = "q"


def cell_volumes(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def manufactured_field(n_cells: int = N_CELLS) -> np.ndarray:
    """Return a unique 1-d field so an npz round-trip is observable."""
    return np.arange(int(n_cells), dtype=np.float64) + 1.0
