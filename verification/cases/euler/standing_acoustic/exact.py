"""1-d linear standing acoustic wave in a reflecting cavity.

Cavity [0, 1], walls at x=0 and x=1 (u=0):

    rho = rho_bar + eps cos(k pi x) cos(omega t)
    u   = (eps c / rho_bar) sin(k pi x) sin(omega t)
    p   = p_bar + eps c^2 cos(k pi x) cos(omega t)

k=1, omega = k pi c, c = sqrt(gamma p_bar / rho_bar), gamma=1.4,
rho_bar=1, p_bar=1/gamma so c=1, eps=1e-3. Wave period is 2/c.

Linear acoustic energy density:

    e = p'^2 / (2 rho_bar c^2) + (rho_bar u^2) / 2

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
K = 1
EPS = 1.0e-3
RHO_BAR = 1.0
P_BAR = 1.0 / GAMMA
N_CELLS = 32


def background() -> dict:
    """Uniform rest state. p=1/gamma so c=1 at gamma=1.4."""
    return {"rho": RHO_BAR, "u": 0.0, "p": P_BAR}


def acoustic_speed(W=None) -> float:
    """c = sqrt(gamma p / rho) for a gamma-law ideal gas."""
    state = background() if W is None else W
    rho = float(state["rho"])
    pressure = float(state["p"])
    if rho <= 0.0 or pressure <= 0.0:
        raise ValueError("non-positive thermodynamic state")
    return float(math.sqrt(GAMMA * pressure / rho))


def angular_frequency(*, k: int = K) -> float:
    """Standing-wave frequency omega = k pi c."""
    return float(k) * math.pi * acoustic_speed(background())


def period(*, k: int = K) -> float:
    """Full wave period 2 pi / omega = 2 / (k c). Equals 2/c at k=1."""
    return 2.0 * math.pi / angular_frequency(k=k)


def _xt(x, t):
    return np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
    )


def primitives_1d(x, t, *, k: int = K, eps: float = EPS) -> np.ndarray:
    """Primitive standing wave W=(rho, u, p). Shape (3, *broadcast)."""
    xx, tt = _xt(x, t)
    state = background()
    speed = acoustic_speed(state)
    omega = float(k) * math.pi * speed
    spatial = float(k) * math.pi * xx
    cosine = np.cos(spatial)
    sine = np.sin(spatial)
    if float(k).is_integer():
        # sin(k π n) = 0 exactly at integer nodes (the reflecting walls).
        sine = np.where(np.mod(xx, 1.0) == 0.0, 0.0, sine)
    density = state["rho"] + float(eps) * cosine * np.cos(omega * tt)
    velocity = (float(eps) * speed / state["rho"]) * sine * np.sin(omega * tt)
    pressure = state["p"] + float(eps) * speed * speed * cosine * np.cos(omega * tt)
    return np.stack((density, velocity, pressure))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def acoustic_energy_density(x, t, *, k: int = K, eps: float = EPS) -> np.ndarray:
    """Linear acoustic energy density on the same grid as ``x``."""
    density, velocity, pressure = primitives_1d(x, t, k=k, eps=eps)
    state = background()
    speed = acoustic_speed(state)
    pressure_pert = pressure - state["p"]
    potential = (pressure_pert * pressure_pert) / (
        2.0 * state["rho"] * speed * speed
    )
    kinetic = 0.5 * state["rho"] * velocity * velocity
    return potential + kinetic


def total_acoustic_energy(x, volumes, t, *, k: int = K, eps: float = EPS) -> float:
    """Volume-weighted integral of the linear acoustic energy density."""
    density = acoustic_energy_density(x, t, k=k, eps=eps)
    cell_volumes = np.asarray(volumes, dtype=np.float64)
    return float(np.sum(density * cell_volumes))
