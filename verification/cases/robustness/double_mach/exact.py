"""Woodward–Colella double Mach reflection. Geometry helpers only.

Mach 10 shock into (rho, u, p) = (1.4, 0, 1), gamma = 1.4, wedge 30°,
t_end = 0.2. Documents Rankine–Hugoniot pre/post states and the t=0
shock-front line. No late-time analytic solution. Does not import pops
or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
MACH = 10.0
RHO_PRE = 1.4
U_PRE = 0.0
V_PRE = 0.0
P_PRE = 1.0
# Documented shock / wedge angle. In the WC box the wall is the x-axis
# and the shock-wall incidence is the complement 60°.
WEDGE_ANGLE_DEG = 30.0
SHOCK_ANGLE_DEG = 30.0
SHOCK_WALL_ANGLE_DEG = 90.0 - SHOCK_ANGLE_DEG
X_FOOT = 1.0 / 6.0
T_END = 0.2
DOMAIN_LOWER = (0.0, 0.0)
DOMAIN_UPPER = (4.0, 1.0)
N_CELLS = 32


def _as_samples(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def rankine_hugoniot(
    mach=MACH,
    gamma=GAMMA,
    rho0=RHO_PRE,
    u0=U_PRE,
    p0=P_PRE,
) -> dict:
    """Normal Rankine–Hugoniot jump for a shock of Mach M into (rho0, u0, p0)."""
    mach_n = float(mach)
    gamma_n = float(gamma)
    density0 = float(rho0)
    velocity0 = float(u0)
    pressure0 = float(p0)
    sound = float(np.sqrt(gamma_n * pressure0 / density0))
    density = density0 * ((gamma_n + 1.0) * mach_n * mach_n) / (
        (gamma_n - 1.0) * mach_n * mach_n + 2.0
    )
    pressure = pressure0 * (2.0 * gamma_n * mach_n * mach_n - (gamma_n - 1.0)) / (
        gamma_n + 1.0
    )
    shock_speed = velocity0 + mach_n * sound
    velocity = shock_speed - (shock_speed - velocity0) * density0 / density
    return {
        "rho": float(density),
        "p": float(pressure),
        "u": float(velocity),
        "sound": sound,
        "shock_speed": float(shock_speed),
    }


def shock_normal() -> tuple[float, float]:
    """Unit normal pointing into the pre-shock gas (toward +x)."""
    wall = np.deg2rad(SHOCK_WALL_ANGLE_DEG)
    return (float(np.sin(wall)), float(-np.cos(wall)))


def pre_shock_state() -> dict:
    """Ambient pre-shock primitives (rho, u, v, p)."""
    return {
        "rho": float(RHO_PRE),
        "u": float(U_PRE),
        "v": float(V_PRE),
        "p": float(P_PRE),
    }


def post_shock_state() -> dict:
    """Post-shock primitives. Velocity is along ``shock_normal()``."""
    jump = rankine_hugoniot()
    normal_x, normal_y = shock_normal()
    return {
        "rho": float(jump["rho"]),
        "u": float(jump["u"] * normal_x),
        "v": float(jump["u"] * normal_y),
        "p": float(jump["p"]),
    }


def shock_front_x(y):
    """t=0 shock-front abscissa: x(y) = x_foot + y / tan(beta)."""
    heights = _as_samples(y)
    wall = np.deg2rad(SHOCK_WALL_ANGLE_DEG)
    return X_FOOT + heights / np.tan(wall)


def is_post_shock(x, y):
    """True on the post-shock side of the t=0 shock-front line."""
    return _as_samples(x) < shock_front_x(y)


def primitives(x, y, t=0.0) -> dict:
    """t=0 DMR primitives (rho, u, v, p). No late-time closed form."""
    if float(t) != 0.0:
        raise ValueError("RB-08 increment has no late-time analytic solution")
    xx = _as_samples(x)
    yy = np.broadcast_to(_as_samples(y), xx.shape)
    post = is_post_shock(xx, yy)
    pre_state = pre_shock_state()
    post_state = post_shock_state()
    density = np.where(post, post_state["rho"], pre_state["rho"])
    velocity_x = np.where(post, post_state["u"], pre_state["u"])
    velocity_y = np.where(post, post_state["v"], pre_state["v"])
    pressure = np.where(post, post_state["p"], pre_state["p"])
    return {
        "rho": np.asarray(density, dtype=np.float64),
        "u": np.asarray(velocity_x, dtype=np.float64),
        "v": np.asarray(velocity_y, dtype=np.float64),
        "p": np.asarray(pressure, dtype=np.float64),
    }


def primitives_to_conserved(fields) -> dict:
    """Convert primitive (rho, u, v, p) to conserved (rho, rho u, rho v, E)."""
    rho = np.asarray(fields["rho"], dtype=np.float64)
    velocity_x = np.asarray(fields["u"], dtype=np.float64)
    velocity_y = np.asarray(fields["v"], dtype=np.float64)
    pressure = np.asarray(fields["p"], dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return {
        "rho": rho,
        "rho_u": rho * velocity_x,
        "rho_v": rho * velocity_y,
        "E": energy,
    }


def conserved(x, y, t=0.0) -> dict:
    """t=0 DMR conserved U=(rho, rho u, rho v, E)."""
    return primitives_to_conserved(primitives(x, y, t))
