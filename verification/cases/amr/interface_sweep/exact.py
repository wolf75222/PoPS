"""AM-08 manufactured interface-placement sweep.

Sweep Γ_cf at x0 ∈ [0, 1). Manufactured interface-band error E ∝ h².
Reuses the TR-01 sine via load_sibling_module. Does not import pops or
read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_REPO_ROOT = Path(__file__).resolve().parents[4]
_tr01 = load_sibling_module(
    _REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)

N_PLACEMENTS = 8
RESOLUTIONS = (16, 32, 64, 128)
BAND_CELLS = 4
ERROR_SCALE = 0.04
PLACEMENT_AMPLITUDE = 0.5
ORDER_THRESHOLD = 1.8


def uniform_cell_centers(n_cells: int):
    """Return TR-01 cell centers and widths on the periodic unit interval."""
    return _tr01.uniform_cell_centers(n_cells)


def oracle(x, t=0.0) -> np.ndarray:
    """Return the TR-01 manufactured sine at the sample locations."""
    return _tr01.exact_sine(x, t)


def interface_placements(count: int = N_PLACEMENTS) -> np.ndarray:
    """Return interface abscissae x0 covering the half-open interval [0, 1)."""
    n_places = int(count)
    if n_places < 2:
        raise ValueError("need at least two interface placements")
    return np.arange(n_places, dtype=np.float64) / float(n_places)


def periodic_distance(x, x0) -> np.ndarray:
    """Return periodic d(x, Γ_cf) on the unit interval."""
    delta = np.abs(np.mod(np.asarray(x, dtype=np.float64) - float(x0), 1.0))
    return np.minimum(delta, 1.0 - delta)


def placement_scale(x0) -> float:
    """Return a strictly positive placement-dependent manufactured scale."""
    return 1.0 + PLACEMENT_AMPLITUDE * np.sin(2.0 * np.pi * float(x0))


def manufactured_perturbation(h, x0) -> float:
    """Return the uniform manufactured E ∝ h² at one interface placement."""
    return float(placement_scale(x0)) * ERROR_SCALE * float(h) ** 2


def manufactured_field(x, h, x0) -> np.ndarray:
    """Return oracle + manufactured second-order perturbation."""
    return oracle(x) + manufactured_perturbation(h, x0)
