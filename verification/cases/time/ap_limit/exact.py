"""TM-05 exact toy IMEX relaxation: dy/dt = -(y-g)/ε + f.

Canonical data: g=0, f=0, y(0)=1. Closed form y(t)=exp(-t/ε).
The reduced limit ε→0 is y=0. Does not import pops or read PoPS output.
"""
from __future__ import annotations

import numpy as np

Y0 = 1.0
G = 0.0
F = 0.0
DT = 0.1
EPS_SWEEP = (1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4)


def reduced_limit() -> float:
    """Equilibrium of the stiff relaxation: y=g=0 when f=0."""
    return 0.0


def exact_y(t, eps, *, y0: float = Y0) -> float:
    """Return y0 * exp(-t/ε) for g=0, f=0."""
    return float(y0) * float(np.exp(-float(t) / float(eps)))
