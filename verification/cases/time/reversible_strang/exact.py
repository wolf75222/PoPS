"""TM-08 exact reversible map: advance T, flip velocity, return.

Reuses the TR-01 manufactured sine. The exact translation composed with
its velocity-reversed partner is the identity, so the return error is 0.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_TR01_EXACT = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "exact.py"
)
_tr01 = load_sibling_module(_TR01_EXACT)

exact_sine = _tr01.exact_sine_1d
uniform_cell_centers = _tr01.uniform_cell_centers

N_CELLS = 64
T = 0.25
A = 1.0


def departure(x, t, *, a):
    """Periodic departure point of a constant-velocity translation."""
    points = np.asarray(x, dtype=np.float64)
    return np.mod(points - float(a) * float(t), 1.0)


def reversible_departure(x, t, *, a):
    """Advance T with +a, then T with -a. Exact identity on the circle."""
    forwarded = departure(x, t, a=a)
    return departure(forwarded, t, a=-float(a))


def after_forward(x, t, *, a=A):
    """State after advancing time t at velocity +a."""
    return exact_sine(x, t, a=a)


def after_return(x, t, *, a=A):
    """State after the reversible pair (+a then -a) of duration t."""
    returned_x = reversible_departure(x, t, a=a)
    return exact_sine(returned_x, 0.0, a=a)


def return_errors(n_cells=N_CELLS, t=T, a=A):
    """Volume-weighted error of the reversible map vs the initial sine."""
    centers, volumes = uniform_cell_centers(n_cells)
    initial = exact_sine(centers, 0.0, a=a)
    returned = after_return(centers, t, a=a)
    return reference_errors(returned, initial, volumes)
