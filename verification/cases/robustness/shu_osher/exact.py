"""1-d Shu–Osher initial data (FLASH / literature numbers).

Shu & Osher, J. Comput. Phys. 83, 32–78 (1989), Example 8; FLASH
Hydro ``ShuOsher``. Left (rho, u, p) = (3.857143, 2.629369, 10.33333),
right rho = 1 + 0.2 sin(5x), u = 0, p = 1, gamma = 1.4. Domain [-5, 5],
diaphragm x0 = -4 (the usual literature cut; x = -0.8 is only the
linear image of x0 on [-1, 1] and is not used here, so sin(5x) is
unmodified). This increment is the IC only: there is no closed-form
evolved state and no uniform-fine reference. Does not import pops or
read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
# FLASH / Shu–Osher 1989 printed left state (do not replace by RH recompute).
RHO_L = 3.857143
U_L = 2.629369
P_L = 10.33333
RHO_R_MEAN = 1.0
U_R = 0.0
P_R = 1.0
AMPLITUDE = 0.2
WAVE_NUMBER = 5.0
DOMAIN_LEFT = -5.0
DOMAIN_RIGHT = 5.0
X0 = -4.0
T_END = 1.8


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def right_density(x):
    """Unperturbed right density 1 + 0.2 sin(5x)."""
    samples = _as_samples(x)
    return RHO_R_MEAN + AMPLITUDE * np.sin(WAVE_NUMBER * samples)


def primitives_1d(x, t=0.0) -> np.ndarray:
    """Shu–Osher IC W=(rho, u, p). Shape (3, n). t must be 0."""
    if float(t) != 0.0:
        raise ValueError("RB-04 increment has no time-evolved exact solution")
    samples = _as_samples(x)
    left = samples < X0
    density = np.empty(samples.shape, dtype=np.float64)
    velocity = np.empty(samples.shape, dtype=np.float64)
    pressure = np.empty(samples.shape, dtype=np.float64)
    density[left] = RHO_L
    velocity[left] = U_L
    pressure[left] = P_L
    density[~left] = right_density(samples[~left])
    velocity[~left] = U_R
    pressure[~left] = P_R
    return np.stack((density, velocity, pressure))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_1d(x, t=0.0) -> np.ndarray:
    """Shu–Osher conserved IC U=(rho, rho u, E). Shape (3, n)."""
    return primitives_to_conserved_1d(primitives_1d(x, t))
