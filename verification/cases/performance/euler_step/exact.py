"""PF-04 1-d numpy Rusanov / local Lax-Friedrichs Euler step.

One conservative face-flux update of a uniform free stream must remain
that stream. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
RHO = 1.0
U = 1.0
P = 1.0
PERIOD = 1.0
N_CELLS = 32
CFL = 0.4
FREE_STREAM_ATOL = 1.0e-15


def background() -> dict:
    """Uniform free stream (rho, u, p)."""
    return {"rho": RHO, "u": U, "p": P}


def acoustic_speed(state) -> float:
    """c = sqrt(gamma p / rho) for a gamma-law ideal gas."""
    density = float(state["rho"])
    pressure = float(state["p"])
    if density <= 0.0 or pressure <= 0.0:
        raise ValueError("non-positive thermodynamic state")
    return float(math.sqrt(GAMMA * pressure / density))


def cell_width(n_cells: int = N_CELLS, period: float = PERIOD) -> float:
    """Uniform cell width on the periodic interval."""
    return float(period) / float(n_cells)


def cell_centers(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Uniform cell centers on the periodic interval [0, period]."""
    width = cell_width(n_cells, period)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int = N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Uniform cell volumes on the periodic interval."""
    return np.full(int(n_cells), cell_width(n_cells, period), dtype=np.float64)


def exact_primitives(x, t) -> dict:
    """Primitive fields (rho, u, p). Constant; independent of (x, t)."""
    del t
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    shape = samples.shape
    return {
        "rho": np.full(shape, RHO, dtype=np.float64),
        "u": np.full(shape, U, dtype=np.float64),
        "p": np.full(shape, P, dtype=np.float64),
    }


def primitives_to_conserved(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E). Shape (3, n)."""
    density = np.asarray(primitives["rho"], dtype=np.float64)
    velocity = np.asarray(primitives["u"], dtype=np.float64)
    pressure = np.asarray(primitives["p"], dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_to_primitives(conserved) -> dict:
    """Convert conserved (rho, rho u, E) to primitive (rho, u, p)."""
    density, momentum, energy = np.asarray(conserved, dtype=np.float64)
    velocity = momentum / density
    pressure = (GAMMA - 1.0) * (energy - 0.5 * density * velocity * velocity)
    return {"rho": density, "u": velocity, "p": pressure}


def physical_flux(conserved) -> np.ndarray:
    """1-d Euler flux F = (rho u, rho u^2 + p, u (E + p))."""
    primitives = conserved_to_primitives(conserved)
    density = primitives["rho"]
    velocity = primitives["u"]
    pressure = primitives["p"]
    energy = np.asarray(conserved, dtype=np.float64)[2]
    return np.stack(
        (
            density * velocity,
            density * velocity * velocity + pressure,
            velocity * (energy + pressure),
        )
    )


def max_wave_speed(conserved) -> float:
    """Spectral radius |u| + c of one conserved column or a (3,) state."""
    state = np.asarray(conserved, dtype=np.float64)
    if state.ndim == 1:
        primitives = conserved_to_primitives(state.reshape(3, 1))
        velocity = float(primitives["u"][0])
        sound = acoustic_speed(
            {"rho": float(primitives["rho"][0]), "p": float(primitives["p"][0])}
        )
        return abs(velocity) + sound
    primitives = conserved_to_primitives(state)
    sound = np.sqrt(GAMMA * primitives["p"] / primitives["rho"])
    return float(np.max(np.abs(primitives["u"]) + sound))


def rusanov_flux(left, right) -> np.ndarray:
    """Local Lax-Friedrichs / Rusanov face flux. left, right shape (3,)."""
    u_left = np.asarray(left, dtype=np.float64).reshape(3)
    u_right = np.asarray(right, dtype=np.float64).reshape(3)
    flux_left = physical_flux(u_left).reshape(3)
    flux_right = physical_flux(u_right).reshape(3)
    alpha = max(max_wave_speed(u_left), max_wave_speed(u_right))
    return 0.5 * (flux_left + flux_right) - 0.5 * alpha * (u_right - u_left)


def cfl_dt(dx: float, conserved=None, cfl: float = CFL) -> float:
    """Forward-Euler step from CFL and the max |u|+c on the field."""
    if conserved is None:
        speed = acoustic_speed(background()) + abs(float(background()["u"]))
    else:
        speed = max_wave_speed(conserved)
    if speed <= 0.0:
        raise ValueError("non-positive wave speed")
    return float(cfl) * float(dx) / speed


def rusanov_step(conserved, dt, dx) -> np.ndarray:
    """One periodic first-order Rusanov Euler step. Python face loop."""
    field = np.asarray(conserved, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] != 3:
        raise ValueError("conserved state must have shape (3, n)")
    n_cells = field.shape[1]
    if n_cells < 2:
        raise ValueError("need at least two cells")
    face_flux = np.empty((3, n_cells), dtype=np.float64)
    for index in range(n_cells):
        left = field[:, index]
        right = field[:, (index + 1) % n_cells]
        face_flux[:, index] = rusanov_flux(left, right)
    updated = np.empty_like(field)
    ratio = float(dt) / float(dx)
    for index in range(n_cells):
        left_face = face_flux[:, (index - 1) % n_cells]
        right_face = face_flux[:, index]
        updated[:, index] = field[:, index] - ratio * (right_face - left_face)
    return updated


def free_stream_residuals(density, velocity, pressure) -> dict:
    """L∞ of each primitive versus the uniform free stream."""
    stream = background()
    return {
        "rho": float(np.max(np.abs(np.asarray(density, dtype=np.float64) - stream["rho"]))),
        "u": float(np.max(np.abs(np.asarray(velocity, dtype=np.float64) - stream["u"]))),
        "p": float(np.max(np.abs(np.asarray(pressure, dtype=np.float64) - stream["p"]))),
    }
