"""Manufactured 1-d periodic Gaussian pulse. Translation is exact.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

PERIOD = 1.0
Q0 = 0.0
AMP = 1.0
X0 = 0.37
SIGMA = 0.08
A = 1.0


def minimum_image(delta, period: float = PERIOD):
    """Map a displacement onto (-period/2, period/2]."""
    width = float(period)
    return np.mod(np.asarray(delta, dtype=np.float64) + 0.5 * width, width) - 0.5 * width


def exact_gaussian(
    x,
    t,
    *,
    q0: float = Q0,
    amp: float = AMP,
    x0: float = X0,
    sigma: float = SIGMA,
    a: float = A,
):
    """Return q(x, t) on periodic [0, 1] via the minimum-image Gaussian."""
    displacement = np.asarray(x, dtype=np.float64) - (float(x0) + float(a) * float(t))
    radius = minimum_image(displacement, PERIOD)
    return float(q0) + float(amp) * np.exp(
        -np.square(radius) / (2.0 * float(sigma) ** 2)
    )
