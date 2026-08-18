"""AM-01 static coarse-fine geometry and TR-01 sine reuse.

1-d: coarse left / fine right of x=0.5. Distance to Γ_cf is |x-0.5|.
Reuses the TR-01 manufactured sine. Does not read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

INTERFACE_X = 0.5
REFINEMENT_RATIO = 2
BAND_CELLS = 4
N_COARSE = 8

_TR01_EXACT = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "exact.py"
)
_tr01 = load_sibling_module(_TR01_EXACT)
exact_sine = _tr01.exact_sine_1d


def distance_to_interface(x) -> np.ndarray:
    """Return d(x, Γ_cf) = |x - 0.5| on the static 1-d interface."""
    return np.abs(np.asarray(x, dtype=np.float64) - INTERFACE_X)


def static_cf_centers(
    n_coarse: int = N_COARSE, *, refinement_ratio: int = REFINEMENT_RATIO
):
    """Return leaf centers and spacings: coarse on [0, 0.5], fine on [0.5, 1]."""
    n_left = int(n_coarse)
    ratio = int(refinement_ratio)
    if n_left <= 0 or ratio <= 0:
        raise ValueError("non-positive static CF inputs")
    h_coarse = INTERFACE_X / n_left
    n_fine = n_left * ratio
    h_fine = (1.0 - INTERFACE_X) / n_fine
    coarse = (np.arange(n_left, dtype=np.float64) + 0.5) * h_coarse
    fine = INTERFACE_X + (np.arange(n_fine, dtype=np.float64) + 0.5) * h_fine
    return np.concatenate([coarse, fine]), float(h_fine), float(h_coarse)
