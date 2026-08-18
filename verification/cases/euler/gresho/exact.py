"""2-d Gresho vortex. Stationary compressible Euler equilibrium.

Piecewise azimuthal speed (Liska & Wendroff / Gresho & Chan):

    u_θ(r) = 5r            r < 0.2
           = 2 − 5r        0.2 ≤ r < 0.4
           = 0             r ≥ 0.4

Density ρ = 1, γ = 1.4. Pressure is the integral of centrifugal balance
∇p = ρ u_θ² / r, continuous at the kinks:

    p(r) = 5 + 12.5 r²                         r < 0.2
         = 9 + 12.5 r² − 20 r + 4 ln(5r)       0.2 ≤ r < 0.4
         = 3 + 4 ln(2)                         r ≥ 0.4

The field is independent of t. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
RHO = 1.0
R1 = 0.2
R2 = 0.4
X0 = 0.5
Y0 = 0.5
PERIOD = 1.0
P_CENTER = 5.0
P_OUTER = 3.0 + 4.0 * math.log(2.0)


def _as_samples(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def u_theta(r):
    """Piecewise azimuthal speed of the Gresho vortex."""
    radius = _as_samples(r)
    speed = np.zeros_like(radius)
    inner = radius < R1
    middle = (radius >= R1) & (radius < R2)
    speed[inner] = 5.0 * radius[inner]
    speed[middle] = 2.0 - 5.0 * radius[middle]
    return speed


def pressure_inner(r):
    """p = 5 + 12.5 r² on the inner disk r < 0.2 (also valid at r = 0.2)."""
    radius = _as_samples(r)
    return P_CENTER + 12.5 * radius * radius


def pressure_middle(r):
    """p = 9 + 12.5 r² − 20 r + 4 ln(5r) on the ring 0.2 ≤ r < 0.4."""
    radius = _as_samples(r)
    return 9.0 + 12.5 * radius * radius - 20.0 * radius + 4.0 * np.log(5.0 * radius)


def pressure_outer(r):
    """Constant p = 3 + 4 ln(2) for r ≥ 0.4."""
    return np.full_like(_as_samples(r), P_OUTER)


def pressure(r):
    """Three-piece Gresho pressure. Continuous at r = 0.2 and r = 0.4."""
    radius = _as_samples(r)
    values = np.empty_like(radius)
    inner = radius < R1
    middle = (radius >= R1) & (radius < R2)
    outer = radius >= R2
    values[inner] = pressure_inner(radius[inner])
    values[middle] = pressure_middle(radius[middle])
    values[outer] = pressure_outer(radius[outer])
    return values


def dp_dr(r):
    """Analytic dp/dr of the three-piece pressure. Equals ρ u_θ² / r."""
    radius = _as_samples(r)
    deriv = np.zeros_like(radius)
    inner = radius < R1
    middle = (radius >= R1) & (radius < R2)
    deriv[inner] = 25.0 * radius[inner]
    deriv[middle] = 4.0 / radius[middle] - 20.0 + 25.0 * radius[middle]
    return deriv


def centrifugal_residual(r):
    """dp/dr − ρ u_θ² / r. Zero (to rounding) inside each piece."""
    radius = _as_samples(r)
    speed = u_theta(radius)
    centripetal = np.zeros_like(radius)
    nonzero = radius > 0.0
    centripetal[nonzero] = RHO * np.square(speed[nonzero]) / radius[nonzero]
    return dp_dr(radius) - centripetal


def exact_gresho(x, y, t=0.0):
    """Stationary primitives (rho, u, v, p). Independent of t."""
    del t
    xx = _as_samples(x)
    yy = _as_samples(y)
    dx = xx - X0
    dy = yy - Y0
    radius = np.hypot(dx, dy)
    speed = u_theta(radius)
    velocity_x = np.zeros_like(radius)
    velocity_y = np.zeros_like(radius)
    nonzero = radius > 0.0
    velocity_x[nonzero] = -speed[nonzero] * dy[nonzero] / radius[nonzero]
    velocity_y[nonzero] = speed[nonzero] * dx[nonzero] / radius[nonzero]
    return {
        "rho": np.full_like(radius, RHO),
        "u": velocity_x,
        "v": velocity_y,
        "p": pressure(radius),
    }


def radial_velocity(x, y, t=0.0):
    """Radial velocity u_r of the exact field. Identically 0."""
    state = exact_gresho(x, y, t)
    xx = _as_samples(x)
    yy = _as_samples(y)
    dx = xx - X0
    dy = yy - Y0
    radius = np.hypot(dx, dy)
    radial = np.zeros_like(radius)
    nonzero = radius > 0.0
    radial[nonzero] = (
        state["u"][nonzero] * dx[nonzero] / radius[nonzero]
        + state["v"][nonzero] * dy[nonzero] / radius[nonzero]
    )
    return radial
