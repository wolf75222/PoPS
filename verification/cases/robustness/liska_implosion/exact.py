"""Liska–Wendroff implosion IC and x=y leftover residual.

Domain [0, 0.3]². Gas at rest. Diamond cut x+y = 0.15:

    (ρ, p) = (1, 1)        if x + y > 0.15
           = (0.125, 0.14) otherwise

u = v = 0, γ = 1.4. There is no closed-form late-time solution. The
oracle is the IC plus reflection across x=y. Does not import pops or
read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
RHO_OUT = 1.0
P_OUT = 1.0
RHO_IN = 0.125
P_IN = 0.14
U0 = 0.0
V0 = 0.0
DIAGONAL = 0.15
DOMAIN_LOWER = (0.0, 0.0)
DOMAIN_UPPER = (0.3, 0.3)
# Morphological comparison time in Liska & Wendroff (2003). Not an oracle.
T_END = 2.5

_VELOCITY_PAIRS = (("u", "v"), ("rho_u", "rho_v"))


def _as_samples(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def primitives(x, y, t=0.0):
    """Implosion primitives (rho, u, v, p). Defined at t=0 only."""
    if float(t) != 0.0:
        raise ValueError("no analytic late-time implosion solution")
    xx = _as_samples(x)
    yy = _as_samples(y)
    outside = (xx + yy) > DIAGONAL
    density = np.where(outside, RHO_OUT, RHO_IN)
    pressure = np.where(outside, P_OUT, P_IN)
    velocity_x = np.full(xx.shape, U0, dtype=np.float64)
    velocity_y = np.full(yy.shape, V0, dtype=np.float64)
    return {
        "rho": density,
        "u": velocity_x,
        "v": velocity_y,
        "p": pressure,
    }


def primitives_to_conserved(primitives_state) -> dict:
    """Convert primitive (rho, u, v, p) to conserved (rho, rho u, rho v, E)."""
    density = _as_samples(primitives_state["rho"])
    velocity_x = _as_samples(primitives_state["u"])
    velocity_y = _as_samples(primitives_state["v"])
    pressure = _as_samples(primitives_state["p"])
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return {
        "rho": density,
        "rho_u": density * velocity_x,
        "rho_v": density * velocity_y,
        "E": energy,
    }


def conserved(x, y, t=0.0) -> dict:
    """Conserved IC U=(rho, rho u, rho v, E). Defined at t=0 only."""
    return primitives_to_conserved(primitives(x, y, t))


def _swap_velocity_pair(state: dict) -> dict:
    reflected = {key: np.array(value, copy=True) for key, value in state.items()}
    for first, second in _VELOCITY_PAIRS:
        if first in reflected and second in reflected:
            reflected[first], reflected[second] = reflected[second], reflected[first]
    return reflected


def reflect(q) -> dict:
    """Reflect a primitive or conserved field across x=y.

    Velocity / momentum components swap. Square 2-d arrays are also
    transposed so that the sample at (x, y) maps to (y, x).
    """
    reflected = _swap_velocity_pair(q)
    arrays = [np.asarray(value) for value in reflected.values()]
    if arrays and all(item.ndim == 2 and item.shape[0] == item.shape[1] for item in arrays):
        return {key: np.asarray(value).T for key, value in reflected.items()}
    return reflected


def leftover_residual(q) -> dict:
    """Return q − reflect(q). Zero iff q is symmetric under (x, y)↔(y, x)."""
    reflected = reflect(q)
    return {
        key: _as_samples(q[key]) - _as_samples(reflected[key]) for key in q
    }
