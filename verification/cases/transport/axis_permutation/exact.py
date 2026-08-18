"""TR-06 2-d axis permutation / reflection of a TR-01 sine product.

q(x,y,t) = q0 + eps * sin(2π kx (x - ax t)) * sin(2π ky (y - ay t))

Reuses the TR-01 manufactured sine. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_TR01_EXACT = (
    Path(__file__).resolve().parents[1] / "advection_sine" / "exact.py"
)
_tr01 = load_sibling_module(_TR01_EXACT)

exact_sine = _tr01.exact_sine
uniform_cell_centers = _tr01.uniform_cell_centers
Q0 = _tr01.Q0
EPS = _tr01.EPS

KX = 1.0
KY = 2.0
AX = 1.0
AY = 0.5
N_CELLS = 32
T = 0.125
PERIOD = 1.0


def _unit_sine(coord, t, *, a, k, q0, eps):
    """Return the unit-amplitude sine factor of one TR-01 axis."""
    return (exact_sine(coord, t, q0=q0, eps=eps, a=a, k=k) - float(q0)) / float(
        eps
    )


def exact_product(
    x,
    y,
    t,
    *,
    kx=KX,
    ky=KY,
    ax=AX,
    ay=AY,
    q0=Q0,
    eps=EPS,
) -> np.ndarray:
    """Return the 2-d product of independent TR-01 sines on periodic [0, 1]^2."""
    sx = _unit_sine(x, t, a=ax, k=kx, q0=q0, eps=eps)
    sy = _unit_sine(y, t, a=ay, k=ky, q0=q0, eps=eps)
    return float(q0) + float(eps) * sx * sy


def uniform_grid_2d(n_cells: int = N_CELLS):
    """Return (x, y, volumes, axis) on a uniform periodic unit square (axis 0 is x)."""
    axis, widths = uniform_cell_centers(n_cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    volume = float(widths[0]) * float(widths[0])
    volumes = np.full(x.shape, volume, dtype=np.float64)
    return x, y, volumes, axis


def permute_xy(x, y):
    """Swap coordinates: (x, y) → (y, x)."""
    return np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)


def reflect_x(x):
    """Periodic reflection x ↦ 1 − x on the unit interval."""
    return np.mod(float(PERIOD) - np.asarray(x, dtype=np.float64), float(PERIOD))
