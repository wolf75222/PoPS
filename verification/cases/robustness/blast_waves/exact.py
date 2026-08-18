"""1-d Woodward–Colella two-blast initial data (IC only).

Woodward & Colella, J. Comput. Phys. 54, 115–173 (1984). Left
(rho, u, p) = (1, 0, 1000) for x < 0.1, middle (1, 0, 0.01), right
(1, 0, 100) for x > 0.9, gamma = 1.4. Domain [0, 1], reflecting
walls. Usual final time t = 0.038 is documented only; this increment
has no closed-form evolved state (interacting blasts). Does not
import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
RHO = 1.0
U = 0.0
P_LEFT = 1000.0
P_MIDDLE = 0.01
P_RIGHT = 100.0
DOMAIN_LEFT = 0.0
DOMAIN_RIGHT = 1.0
X_LEFT = 0.1
X_RIGHT = 0.9
T_END = 0.038


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def primitives_1d(x, t=0.0) -> np.ndarray:
    """Two-blast IC W=(rho, u, p). Shape (3, n). t must be 0."""
    if float(t) != 0.0:
        raise ValueError("RB-09 increment has no time-evolved exact solution")
    samples = _as_samples(x)
    left = samples < X_LEFT
    right = samples > X_RIGHT
    density = np.full(samples.shape, RHO, dtype=np.float64)
    velocity = np.full(samples.shape, U, dtype=np.float64)
    pressure = np.full(samples.shape, P_MIDDLE, dtype=np.float64)
    pressure[left] = P_LEFT
    pressure[right] = P_RIGHT
    return np.stack((density, velocity, pressure))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_1d(x, t=0.0) -> np.ndarray:
    """Two-blast conserved IC U=(rho, rho u, E). Shape (3, n)."""
    return primitives_to_conserved_1d(primitives_1d(x, t))
