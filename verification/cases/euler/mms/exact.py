"""1-d / 2-d gamma-law Euler manufactured solution.

1-d reduction (periodic unit interval), gamma=1.4:

    rho = 2 + 0.1 sin(2 pi (x - t))
    u   = 0.3 + 0.1 cos(2 pi (x - t))
    p   = 1 + 0.05 sin(2 pi (x - t))

2-d plan fields (plus the v, p completion from the campaign plan):

    rho = 2 + 0.1 sin(2 pi (x + y - t))
    u   = 0.3 + 0.1 cos(2 pi (x - t))
    v   = -0.2 + 0.1 sin(2 pi (y + t))
    p   = 1 + 0.1 cos(2 pi (x - y + t))

Manufactured sources are the residual S = dU/dt + dF/dx of the 1-d
conservative Euler system. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
TWO_PI = 2.0 * np.pi


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def primitives_1d(x, t) -> np.ndarray:
    """1-d primitives W=(rho, u, p). Shape (3, n)."""
    samples = _as_samples(x)
    phase = TWO_PI * (samples - float(t))
    density = 2.0 + 0.1 * np.sin(phase)
    velocity = 0.3 + 0.1 * np.cos(phase)
    pressure = 1.0 + 0.05 * np.sin(phase)
    return np.stack((density, velocity, pressure))


def primitives_2d(x, y, t) -> np.ndarray:
    """2-d primitives W=(rho, u, v, p). Shape (4, n)."""
    xs = _as_samples(x)
    ys = _as_samples(y)
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same shape")
    time = float(t)
    density = 2.0 + 0.1 * np.sin(TWO_PI * (xs + ys - time))
    velocity_x = 0.3 + 0.1 * np.cos(TWO_PI * (xs - time))
    velocity_y = -0.2 + 0.1 * np.sin(TWO_PI * (ys + time))
    pressure = 1.0 + 0.1 * np.cos(TWO_PI * (xs - ys + time))
    return np.stack((density, velocity_x, velocity_y, pressure))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_1d(x, t) -> np.ndarray:
    """1-d conserved U=(rho, rho u, E). Shape (3, n)."""
    return primitives_to_conserved_1d(primitives_1d(x, t))


def flux_1d(x, t) -> np.ndarray:
    """1-d Euler flux F=(rho u, rho u^2 + p, u (E + p)). Shape (3, n)."""
    density, velocity, pressure = primitives_1d(x, t)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack(
        (
            density * velocity,
            density * velocity * velocity + pressure,
            velocity * (energy + pressure),
        )
    )


def _primitive_derivatives_1d(x, t):
    """Analytic (d/dt, d/dx) of the 1-d primitive fields."""
    samples = _as_samples(x)
    phase = TWO_PI * (samples - float(t))
    cosine = np.cos(phase)
    sine = np.sin(phase)
    density_t = -0.2 * np.pi * cosine
    density_x = 0.2 * np.pi * cosine
    velocity_t = 0.2 * np.pi * sine
    velocity_x = -0.2 * np.pi * sine
    pressure_t = -0.1 * np.pi * cosine
    pressure_x = 0.1 * np.pi * cosine
    return {
        "rho_t": density_t,
        "rho_x": density_x,
        "u_t": velocity_t,
        "u_x": velocity_x,
        "p_t": pressure_t,
        "p_x": pressure_x,
    }


def sources_1d(x, t) -> np.ndarray:
    """Manufactured residual S = dU/dt + dF/dx. Shape (3, n)."""
    density, velocity, pressure = primitives_1d(x, t)
    deriv = _primitive_derivatives_1d(x, t)
    momentum_t = deriv["rho_t"] * velocity + density * deriv["u_t"]
    momentum_x = deriv["rho_x"] * velocity + density * deriv["u_x"]
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    energy_t = (
        deriv["p_t"] / (GAMMA - 1.0)
        + 0.5 * deriv["rho_t"] * velocity * velocity
        + density * velocity * deriv["u_t"]
    )
    energy_x = (
        deriv["p_x"] / (GAMMA - 1.0)
        + 0.5 * deriv["rho_x"] * velocity * velocity
        + density * velocity * deriv["u_x"]
    )
    flux_momentum_x = (
        deriv["rho_x"] * velocity * velocity
        + 2.0 * density * velocity * deriv["u_x"]
        + deriv["p_x"]
    )
    flux_energy_x = deriv["u_x"] * (energy + pressure) + velocity * (
        energy_x + deriv["p_x"]
    )
    return np.stack(
        (
            deriv["rho_t"] + momentum_x,
            momentum_t + flux_momentum_x,
            energy_t + flux_energy_x,
        )
    )
