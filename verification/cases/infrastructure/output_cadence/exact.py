"""IF-05 output cadences. Exact field is TR-01's sine.

The manufactured sine is advanced analytically to t=0.25. Dump cadences
1, 2, and 10 steps evaluate that same exact state. Dumps do not change
the field.

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

exact_sine = _tr01.exact_sine

CADENCES = (1, 2, 10)
T = 0.25
N_STEPS = 10
DT = T / float(N_STEPS)
DEFAULT_N_CELLS = 32
PERIOD = 1.0


def cell_centers(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def exact_field(x, t, **kwargs) -> np.ndarray:
    """Return the TR-01 sine. Independent of the dump cadence."""
    return np.asarray(exact_sine(x, t, **kwargs), dtype=np.float64)


def should_dump(step, cadence) -> bool:
    """True after every `cadence` completed steps, including the last step."""
    interval = int(cadence)
    if interval < 1:
        raise ValueError(f"dump cadence must be >= 1, got {cadence!r}")
    completed = int(step)
    if completed < 1:
        return False
    return completed % interval == 0


def expected_dumps(cadence, n_steps: int = N_STEPS) -> int:
    """Return the dump count N/k over a closed step interval."""
    interval = int(cadence)
    steps = int(n_steps)
    if interval < 1:
        raise ValueError(f"dump cadence must be >= 1, got {cadence!r}")
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if steps % interval != 0:
        raise ValueError(f"n_steps={steps} must be divisible by cadence={interval}")
    return steps // interval
