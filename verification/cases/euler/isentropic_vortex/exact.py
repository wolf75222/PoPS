"""2-d isentropic vortex. Exact at t is translation by (u_inf, v_inf).

Classic Yee/Sandham/Djomehri perturbation on a periodic box. 1-d is not
applicable. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
BETA = 5.0
PERIOD = 10.0
X0 = 5.0
Y0 = 5.0
RHO_INF = 1.0
U_INF = 1.0
V_INF = 0.0
P_INF = 1.0


def background() -> dict:
    """Uniform free stream (rho_inf, u_inf, v_inf, p_inf). Canonical (u,v)=(1,0)."""
    return {"rho": RHO_INF, "u": U_INF, "v": V_INF, "p": P_INF}


def minimum_image(delta, period: float = PERIOD):
    """Map a displacement onto (-period/2, period/2]."""
    width = float(period)
    return np.mod(np.asarray(delta, dtype=np.float64) + 0.5 * width, width) - 0.5 * width


def entropy_function(rho, p, *, gamma: float = GAMMA):
    """Isentropic invariant p / rho^gamma."""
    density = np.asarray(rho, dtype=np.float64)
    pressure = np.asarray(p, dtype=np.float64)
    return pressure / np.power(density, float(gamma))


def exact_vortex(x, y, t, *, u_inf: float = U_INF, v_inf: float = V_INF):
    """Primitive fields (rho, u, v, p) of the translated isentropic vortex."""
    samples_x = np.asarray(x, dtype=np.float64)
    samples_y = np.asarray(y, dtype=np.float64)
    center_x = X0 + float(u_inf) * float(t)
    center_y = Y0 + float(v_inf) * float(t)
    dx = minimum_image(samples_x - center_x)
    dy = minimum_image(samples_y - center_y)
    radius_sq = dx * dx + dy * dy
    gaussian = np.exp(0.5 * (1.0 - radius_sq))
    du = -BETA * dy / (2.0 * math.pi) * gaussian
    dv = BETA * dx / (2.0 * math.pi) * gaussian
    temperature_inf = P_INF / RHO_INF
    delta_t = (
        -(GAMMA - 1.0)
        * BETA
        * BETA
        / (8.0 * GAMMA * math.pi * math.pi)
        * np.exp(1.0 - radius_sq)
    )
    temperature = temperature_inf + delta_t
    ratio = temperature / temperature_inf
    density = RHO_INF * np.power(ratio, 1.0 / (GAMMA - 1.0))
    pressure = P_INF * np.power(ratio, GAMMA / (GAMMA - 1.0))
    return {
        "rho": density,
        "u": float(u_inf) + du,
        "v": float(v_inf) + dv,
        "p": pressure,
    }
