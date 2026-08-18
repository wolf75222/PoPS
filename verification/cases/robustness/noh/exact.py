"""Planar 1-d Noh exact self-similar oracle.

Cold inflow toward x=0: (rho, u, p) = (1, -sign(x), 0), gamma=5/3.
For t>0 a pair of infinite-Mach shocks recede at |S|=1/3. Post-shock
states are (rho, u, p)=(4, 0, 4/3) for |x|<t/3; unshocked gas is the
inflow. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 5.0 / 3.0
RHO0 = 1.0
U_INFLOW = 1.0
P0 = 0.0
X0 = 0.0
T_END = 0.6
# Strong-shock density ratio (gamma+1)/(gamma-1) = 4.
POST_SHOCK_DENSITY = RHO0 * (GAMMA + 1.0) / (GAMMA - 1.0)
POST_SHOCK_VELOCITY = 0.0
# p* = (gamma+1)/2 * rho0 * u^2 = 4/3.
POST_SHOCK_PRESSURE = 0.5 * (GAMMA + 1.0) * RHO0 * U_INFLOW * U_INFLOW
# |S| = |u| (gamma-1)/2 = 1/3.
SHOCK_SPEED = U_INFLOW * (GAMMA - 1.0) / 2.0


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def shock_position(t) -> float:
    """Return the self-similar front |x_s| = t/3."""
    time = float(t)
    if time < 0.0:
        raise ValueError("time must be non-negative")
    return SHOCK_SPEED * time


def shock_positions(t) -> tuple[float, float]:
    """Return the left and right shock locations (-t/3, +t/3)."""
    radius = shock_position(t)
    return (-radius, radius)


def primitives_1d(x, t) -> np.ndarray:
    """Exact planar Noh primitives W=(rho, u, p). Shape (3, n)."""
    samples = _as_samples(x)
    time = float(t)
    density = np.full(samples.shape, RHO0, dtype=np.float64)
    velocity = -U_INFLOW * np.sign(samples)
    pressure = np.full(samples.shape, P0, dtype=np.float64)
    if time > 0.0:
        post = np.abs(samples - X0) < shock_position(time)
        density[post] = POST_SHOCK_DENSITY
        velocity[post] = POST_SHOCK_VELOCITY
        pressure[post] = POST_SHOCK_PRESSURE
    return np.stack((density, velocity, pressure))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_1d(x, t) -> np.ndarray:
    """Exact conserved U=(rho, rho u, E). Shape (3, n)."""
    return primitives_to_conserved_1d(primitives_1d(x, t))
